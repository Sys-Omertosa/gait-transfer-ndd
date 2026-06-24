"""
src/train.py

Within-condition baseline training utilities for gait classifier evaluation.

Functions implemented here:
  build_pipeline          -- constructs the correct ImbPipeline per classifier
  run_nested_loso         -- outer LOSO evaluation with inner GridSearchCV tuning
  get_modal_params        -- extracts modal best params from per-fold records
  get_classifier_configs  -- returns all 7 (clf, param_grid) configurations
  run_within_condition    -- orchestrates the full experiment for one condition
  run_cross_condition     -- zero-shot transfer from source to target condition

SMOTE convention: Mittal et al. (Frontiers Robotics AI, 2025) and Chu et al.
(MDPI Entropy, 2020) both apply SMOTE on training folds only on PhysioNet NDD
gait data. This module follows the same convention via ImbPipeline. In all
three within-condition pools, the control class is the minority (8 Control
Group A subjects versus 13-19 disease subjects), so SMOTE augments healthy
control strides rather than disease strides. SMOTE-generated synthetic control
strides are interpolations within the stride space defined by Control Group A;
the disjoint Control Group B keeps cross-condition evaluation on independent
healthy subjects not seen during training-time augmentation.

Feature scaling: RobustScaler is applied for SVM and KNN only -- distance-
and margin-based classifiers whose kernels and distance metrics are distorted
by unequal feature scales. Tree-based methods and QDA are left in the original
feature space so that absolute timing magnitude remains available to the model
and to the downstream SHAP interpretation.

F1 evaluation: each outer LOSO fold holds out one subject, who belongs to
exactly one class. Per-fold F1 macro is mathematically undefined over a
single-class test set. The correct approach is to accumulate predictions across
all folds and compute a single aggregate F1 macro over the full vector.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import ALL_FEATURE_COLS
from v4_provenance import atomic_write_bytes, atomic_write_json, sha256_bytes, sha256_file

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(
    'ignore',
    message='.*covariance matrix.*not full rank.*',
    category=UserWarning,
)

DEFAULT_FEATURE_MATRIX_FILE = 'v4/gait_features_v4.csv'
DEFAULT_FEATURE_SET_VERSION = 'v4'
DEFAULT_NORMALIZATION = 'none'
IMBALANCE_STRATEGIES = ('synthetic', 'balanced', 'raw')
DEFAULT_SUBJECT_AGGREGATION_RULE = 'mean_probability'
DEFAULT_SUBJECT_PROBABILITY_THRESHOLD = 0.5
DEFAULT_TIE_BREAK_RULE = 'subject_probability_loss_then_lexicographic'
PROBABILITY_AGGREGATION_RULES = frozenset(
    {'mean_probability', 'median_probability'})
DECISION_AGGREGATION_RULES = frozenset({'mean_decision_score'})


def candidate_strategies_for_classifier(
    clf_name: str,
    candidate_imbalance_strategies: tuple[str, ...] | dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Resolve the allowed imbalance strategies for one classifier."""
    if isinstance(candidate_imbalance_strategies, dict):
        strategies = candidate_imbalance_strategies.get(clf_name)
        if strategies is None:
            raise KeyError(
                f'No candidate strategy policy provided for classifier {clf_name!r}.'
            )
        return tuple(strategies)
    return tuple(candidate_imbalance_strategies)

# ── Shared helpers ───────────────────────────────────────────────────────────


def _params_to_key(d: dict) -> tuple:
    """Convert a params dict to a hashable tuple for counting and comparison."""
    return tuple(sorted(d.items()))


def _json_safe(value: Any) -> Any:
    """Convert numpy-heavy nested values to JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    return sha256_file(path)


def _fast_f1_binary(yt: np.ndarray, yp: np.ndarray) -> float:
    """Fast F1 macro for binary {0,1} classification. Avoids sklearn overhead in bootstrap loops."""
    tp1 = int(np.sum((yt == 1) & (yp == 1)))
    fp1 = int(np.sum((yt == 0) & (yp == 1)))
    fn1 = int(np.sum((yt == 1) & (yp == 0)))
    tp0 = int(np.sum((yt == 0) & (yp == 0)))
    fp0 = int(np.sum((yt == 1) & (yp == 0)))
    fn0 = int(np.sum((yt == 0) & (yp == 1)))
    f1_1 = (2 * tp1) / (2 * tp1 + fp1 +
                        fn1) if (2 * tp1 + fp1 + fn1) > 0 else 0.0
    f1_0 = (2 * tp0) / (2 * tp0 + fp0 +
                        fn0) if (2 * tp0 + fp0 + fn0) > 0 else 0.0
    return 0.5 * (f1_0 + f1_1)


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights equivalent to class_weight='balanced' for binary labels."""
    counts = np.bincount(y.astype(int), minlength=2)
    if np.any(counts == 0):
        return np.ones_like(y, dtype=float)

    total = float(len(y))
    weight_0 = total / (2.0 * counts[0])
    weight_1 = total / (2.0 * counts[1])
    return np.where(y == 0, weight_0, weight_1).astype(float)


def _normalize_subject_aggregation_rule(subject_aggregation_rule: str) -> str:
    """Normalize supported subject-level aggregation-rule identifiers."""
    aliases = {
        'mean_probability_0.5': 'mean_probability',
        'median_probability_0.5': 'median_probability',
        'majority_vote_hard_predictions': 'majority_vote',
        'mean_decision_score_threshold_0': 'mean_decision_score',
    }
    normalized = aliases.get(subject_aggregation_rule,
                             subject_aggregation_rule)
    supported = PROBABILITY_AGGREGATION_RULES | DECISION_AGGREGATION_RULES | {
        'majority_vote'}
    if normalized not in supported:
        raise ValueError(
            f"Unknown subject aggregation rule '{subject_aggregation_rule}'. "
            f'Expected one of {sorted(supported)}.'
        )
    return normalized


def _subject_level_log_loss(
    y_true: np.ndarray,
    subject_scores: np.ndarray,
) -> float | None:
    """
    Compute subject-level log loss when the aggregation rule yields probabilities.

    Returns None for degenerate or non-probabilistic cases.
    """
    if len(subject_scores) == 0:
        return None
    if np.any(subject_scores < 0.0) or np.any(subject_scores > 1.0):
        return None
    clipped = np.clip(subject_scores, 1e-6, 1.0 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return None
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def _subject_level_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray | list[str],
    decision_scores: np.ndarray | None = None,
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
) -> dict[str, Any]:
    """
    Collapse stride-level outputs to one decision per subject.

    The aggregation rule is review-gated in the v4 hardening pass, so the
    authoritative pipeline stores the rule explicitly with every subject-level
    metric bundle instead of assuming mean probability silently.
    """
    subj_true_arr, subj_pred_arr, subj_score_arr, unique_subjects = _subject_level_arrays(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        subject_ids=subject_ids,
        decision_scores=decision_scores,
        subject_aggregation_rule=subject_aggregation_rule,
    )
    log_loss_value = _subject_level_log_loss(subj_true_arr, subj_score_arr)

    return {
        'aggregation_rule': _normalize_subject_aggregation_rule(subject_aggregation_rule),
        'n_subjects': int(len(unique_subjects)),
        'subject_ids': unique_subjects,
        'y_true': subj_true_arr.tolist(),
        'y_pred': subj_pred_arr.tolist(),
        'subject_scores': np.round(subj_score_arr, 6).tolist(),
        'y_prob_mean': (
            subj_score_arr.round(6).tolist()
            if _normalize_subject_aggregation_rule(subject_aggregation_rule) in PROBABILITY_AGGREGATION_RULES
            else None
        ),
        'subject_log_loss': (
            round(float(log_loss_value), 6) if log_loss_value is not None else None
        ),
        'f1_macro': round(float(f1_score(subj_true_arr, subj_pred_arr, average='macro')), 6),
        'accuracy': round(float(accuracy_score(subj_true_arr, subj_pred_arr)), 6),
        'recall_disease': round(float(recall_score(
            subj_true_arr, subj_pred_arr, pos_label=1, zero_division=0
        )), 6),
        'recall_control': round(float(recall_score(
            subj_true_arr, subj_pred_arr, pos_label=0, zero_division=0
        )), 6),
    }


def _subject_level_arrays(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray | list[str],
    decision_scores: np.ndarray | None = None,
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Collapse stride-level outputs to one subject-level decision per subject.

    Returns subject-level y_true, y_pred, and subject-score arrays aligned to
    the unique-subject order preserved from the input sequence.
    """
    aggregation_rule = _normalize_subject_aggregation_rule(
        subject_aggregation_rule)
    subject_arr = np.asarray(subject_ids)
    unique_subjects = list(dict.fromkeys(subject_arr.tolist()))

    subj_true: list[int] = []
    subj_pred: list[int] = []
    subj_scores: list[float] = []

    for subject_id in unique_subjects:
        idx = np.where(subject_arr == subject_id)[0]
        true_vals = y_true[idx]
        assert len(np.unique(true_vals)) == 1, (
            f'Subject {subject_id} has mixed true labels, which should be impossible'
        )
        subj_true.append(int(true_vals[0]))
        if aggregation_rule == 'mean_probability':
            subject_score = float(np.mean(y_prob[idx]))
            subject_pred = int(
                subject_score >= DEFAULT_SUBJECT_PROBABILITY_THRESHOLD)
        elif aggregation_rule == 'median_probability':
            subject_score = float(np.median(y_prob[idx]))
            subject_pred = int(
                subject_score >= DEFAULT_SUBJECT_PROBABILITY_THRESHOLD)
        elif aggregation_rule == 'majority_vote':
            vote_share = float(np.mean(y_pred[idx].astype(np.float64)))
            subject_score = vote_share
            subject_pred = int(
                vote_share >= DEFAULT_SUBJECT_PROBABILITY_THRESHOLD)
        elif aggregation_rule == 'mean_decision_score':
            if decision_scores is None:
                raise ValueError(
                    "Aggregation rule 'mean_decision_score' requires decision_scores."
                )
            subject_score = float(np.mean(np.asarray(decision_scores)[idx]))
            subject_pred = int(subject_score >= 0.0)
        else:
            raise ValueError(
                f"Unhandled aggregation rule '{aggregation_rule}'")
        subj_scores.append(subject_score)
        subj_pred.append(subject_pred)

    return (
        np.asarray(subj_true, dtype=int),
        np.asarray(subj_pred, dtype=int),
        np.asarray(subj_scores, dtype=float),
        unique_subjects,
    )


def _configure_classifier_for_resampling(
    classifier_name: str,
    clf: Any,
    imbalance_strategy: str,
) -> Any:
    """
    Clone a classifier and apply any resampling-specific fixed parameters.

    The authoritative v3 rerun compares synthetic minority-control augmentation
    against non-synthetic balancing and a raw sanity arm. To keep those arms
    interpretable, the base classifier configs are unweighted; any class
    weighting is applied here only for the 'balanced' arm.
    """
    configured = clone(clf)
    name = classifier_name.lower()

    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy '{imbalance_strategy}'. "
            f'Expected one of {IMBALANCE_STRATEGIES}.'
        )

    if imbalance_strategy == 'balanced' and name in {'rf', 'svm', 'dt', 'lgbm'}:
        configured.set_params(class_weight='balanced')

    return configured


def _get_fit_kwargs(
    classifier_name: str,
    y_fit: np.ndarray,
    imbalance_strategy: str,
) -> dict[str, Any]:
    """
    Build fit kwargs for classifiers that need per-fit balancing metadata.

    XGBoost does not expose sklearn-style class_weight in this environment, so
    the non-synthetic balanced arm uses balanced per-sample weights as the
    closest equivalent. GridSearchCV will subset these weights correctly inside
    the inner LOSO folds because the array length matches the outer training fold.
    """
    if imbalance_strategy == 'balanced' and classifier_name.lower() == 'xgb':
        return {'clf__sample_weight': _balanced_sample_weight(y_fit)}
    return {}


def _subject_resampled_stride_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray | list[str],
    rng: np.random.Generator,
    n_resamples: int = 10_000,
) -> tuple[float, float, int, float]:
    """
    Subject-resampled stride-level bootstrap CI for macro-F1.

    Subjects are resampled with replacement, then all strides belonging to the
    chosen subjects are concatenated in the sampled order. Degenerate resamples
    containing only one class are rejected because macro-F1 would be undefined.

    This is a sensitivity interval for the stride-level metric under clustered
    resampling, not the primary subject-level confidence interval.
    """
    slow_f1 = f1_score(y_true, y_pred, average='macro')
    fast_f1 = _fast_f1_binary(y_true, y_pred)
    assert abs(fast_f1 - slow_f1) < 1e-10, (
        f'Fast F1 mismatch: {fast_f1} vs {slow_f1}'
    )

    subject_arr = np.asarray(subject_ids)
    unique_subjects = list(dict.fromkeys(subject_arr.tolist()))
    n_subj = len(unique_subjects)
    subj_idx: dict[str, np.ndarray] = {
        s: np.where(subject_arr == s)[0]
        for s in unique_subjects
    }

    subj_boot_f1 = np.empty(n_resamples, dtype=float)
    collected = 0
    n_rejected = 0

    while collected < n_resamples:
        chosen = rng.choice(unique_subjects, size=n_subj, replace=True)
        boot_idx = np.concatenate([subj_idx[s] for s in chosen])
        yt_boot = y_true[boot_idx]

        if len(np.unique(yt_boot)) < 2:
            n_rejected += 1
            continue

        yp_boot = y_pred[boot_idx]
        subj_boot_f1[collected] = _fast_f1_binary(yt_boot, yp_boot)
        collected += 1

    rejection_rate = n_rejected / (n_resamples + n_rejected)
    ci_lower = float(np.percentile(subj_boot_f1, 2.5))
    ci_upper = float(np.percentile(subj_boot_f1, 97.5))
    return ci_lower, ci_upper, n_rejected, rejection_rate


def _subject_primary_bootstrap_ci(
    y_true_subject: np.ndarray,
    y_pred_subject: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
) -> tuple[float, float, int, float]:
    """
    True subject-level bootstrap CI for macro-F1.

    Each resample draws subjects with replacement from the aggregated
    subject-level arrays directly, giving one equal-weight decision per subject.
    Degenerate one-class resamples are rejected.
    """
    n_subjects = len(y_true_subject)
    boot_f1 = np.empty(n_resamples, dtype=float)
    collected = 0
    n_rejected = 0

    while collected < n_resamples:
        idx = rng.integers(0, n_subjects, size=n_subjects)
        yt_boot = y_true_subject[idx]
        if len(np.unique(yt_boot)) < 2:
            n_rejected += 1
            continue
        yp_boot = y_pred_subject[idx]
        boot_f1[collected] = _fast_f1_binary(yt_boot, yp_boot)
        collected += 1

    rejection_rate = n_rejected / (n_resamples + n_rejected)
    return (
        float(np.percentile(boot_f1, 2.5)),
        float(np.percentile(boot_f1, 97.5)),
        n_rejected,
        rejection_rate,
    )


def _fitted_pipeline_feature_count(pipeline: Any) -> int | None:
    """Return the fitted feature count for a pipeline or classifier when available."""
    n_features = getattr(pipeline, 'n_features_in_', None)
    if n_features is not None:
        return int(n_features)

    clf = getattr(pipeline, 'named_steps', {}).get('clf')
    if clf is not None:
        clf_n_features = getattr(clf, 'n_features_in_', None)
        if clf_n_features is not None:
            return int(clf_n_features)

    return None


# ── Classifiers that require feature scaling ──────────────────────────────────
# SVM and KNN are sensitive to feature scale: the RBF kernel and Euclidean/
# Manhattan distances are dominated by large-valued features without scaling.
# The v2 gait feature set still spans very different ranges, so RobustScaler is
# required for these two only.
_SCALE_REQUIRED = {'svm', 'knn'}


def build_pipeline(
    classifier_name: str,
    clf: Any,
    use_smote: bool = True,
    imbalance_strategy: str | None = None,
) -> ImbPipeline:
    """
    Construct an ImbPipeline for a given classifier.

    For SVM and KNN (distance- and margin-based classifiers sensitive to
    feature scale), the pipeline includes RobustScaler first. SMOTE is
    included only when use_smote=True. In these within-condition source pools,
    that SMOTE step augments the minority control class by interpolating within
    the stride space defined by the 8 Control Group A subjects. Removing SMOTE
    for the ablation leaves the scaler in place for SVM and KNN, and routes
    imbalance correction to class weighting or sample weighting when the model
    family supports it.

    Args:
        classifier_name: One of 'rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm'.
                         Case-insensitive.
        clf: An instantiated sklearn-compatible classifier object.
        use_smote: Legacy compatibility flag. When provided, True maps to the
                   'synthetic' arm and False maps to the 'balanced' arm.
        imbalance_strategy: One of 'synthetic', 'balanced', or 'raw'.

    Returns:
        ImbPipeline with steps appropriate for the given classifier.
    """
    name = classifier_name.lower()
    steps: list[tuple[str, Any]] = []

    if imbalance_strategy is None:
        imbalance_strategy = 'synthetic' if use_smote else 'balanced'
    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy '{imbalance_strategy}'. "
            f'Expected one of {IMBALANCE_STRATEGIES}.'
        )

    needs_scaler = (name in _SCALE_REQUIRED) or (
        imbalance_strategy == 'synthetic')
    if needs_scaler:
        steps.append(('scaler', RobustScaler()))
    if imbalance_strategy == 'synthetic':
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('clf', clf))

    return ImbPipeline(steps)


def _params_json(params: dict[str, Any]) -> str:
    """Deterministic JSON string for a parameter dictionary."""
    return json.dumps(params, sort_keys=True, separators=(',', ':'))


def _candidate_lexicographic_key(candidate_summary: dict[str, Any]) -> tuple[str, str]:
    """Stable deterministic ordering for tied grouped-selection candidates."""
    return (
        str(candidate_summary['imbalance_strategy']),
        _params_json(candidate_summary['params']),
    )


def _candidate_sort_key(
    candidate_summary: dict[str, Any],
    *,
    tie_break_rule: str,
) -> tuple[Any, ...]:
    """Return the ranking tuple for a grouped-selection candidate."""
    subject_f1 = float(candidate_summary['inner_subject_f1'])
    stride_f1 = float(candidate_summary['inner_stride_f1'])
    lexicographic = _candidate_lexicographic_key(candidate_summary)

    if tie_break_rule == 'stride_macro_f1_then_lexicographic':
        return (-subject_f1, -stride_f1, lexicographic)

    if tie_break_rule == 'subject_probability_loss_then_lexicographic':
        subject_loss = candidate_summary.get('inner_subject_log_loss')
        subject_loss_key = float(
            'inf') if subject_loss is None else float(subject_loss)
        return (-subject_f1, subject_loss_key, lexicographic)

    if tie_break_rule == 'lexicographic_only':
        return (-subject_f1, lexicographic)

    raise ValueError(
        f"Unknown tie_break_rule '{tie_break_rule}'. "
        "Expected one of {'stride_macro_f1_then_lexicographic', "
        "'subject_probability_loss_then_lexicographic', 'lexicographic_only'}."
    )


def _save_selection_trace_sidecar(
    *,
    sidecar_path: Path,
    candidates: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Persist a compressed candidate-trace sidecar and return its SHA-256 hash."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata_json': np.asarray([json.dumps(metadata, sort_keys=True)], dtype='<U100000'),
        'candidate_index': np.asarray([int(c['candidate_index']) for c in candidates], dtype=np.int32),
        'imbalance_strategy': np.asarray([str(c['imbalance_strategy']) for c in candidates], dtype='<U32'),
        'inner_subject_f1': np.asarray([float(c['inner_subject_f1']) for c in candidates], dtype=np.float64),
        'inner_stride_f1': np.asarray([float(c['inner_stride_f1']) for c in candidates], dtype=np.float64),
        'inner_subject_log_loss': np.asarray([
            np.nan if c.get('inner_subject_log_loss') is None else float(
                c['inner_subject_log_loss'])
            for c in candidates
        ], dtype=np.float64),
        'params_json': np.asarray([_params_json(dict(c['params'])) for c in candidates], dtype='<U10000'),
        'subject_ids_json': np.asarray([
            json.dumps(list(c.get('subject_ids', [])), separators=(',', ':'))
            for c in candidates
        ], dtype='<U100000'),
        'subject_y_true_json': np.asarray([
            json.dumps(list(c.get('subject_y_true', [])),
                       separators=(',', ':'))
            for c in candidates
        ], dtype='<U100000'),
        'subject_y_pred_json': np.asarray([
            json.dumps(list(c.get('subject_y_pred', [])),
                       separators=(',', ':'))
            for c in candidates
        ], dtype='<U100000'),
        'subject_score_json': np.asarray([
            json.dumps(list(c.get('subject_scores', [])),
                       separators=(',', ':'))
            for c in candidates
        ], dtype='<U100000'),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    atomic_payload = buffer.getvalue()
    atomic_write_bytes(sidecar_path, atomic_payload)
    return sha256_bytes(atomic_payload)


def _select_best_grouped_candidate(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    clf_template: Any,
    param_grid: dict[str, list],
    classifier_name: str,
    candidate_imbalance_strategies: tuple[str, ...],
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Evaluate all (strategy, params) candidates via pooled grouped inner LOSO.

    Returns the best candidate summary and the full sorted ranking table.
    """
    parameter_grid = list(ParameterGrid(param_grid))

    candidate_summaries: list[dict[str, Any]] = []
    candidate_counter = 0

    for imbalance_strategy in candidate_imbalance_strategies:
        for params in parameter_grid:
            inner_loso = LeaveOneGroupOut()
            row_true_all: list[np.ndarray] = []
            row_pred_all: list[np.ndarray] = []
            row_prob_all: list[np.ndarray] = []
            row_subjects_all: list[np.ndarray] = []

            for inner_train_idx, inner_test_idx in inner_loso.split(X_train, y_train, groups_train):
                clf_variant = _configure_classifier_for_resampling(
                    classifier_name,
                    clf_template,
                    imbalance_strategy,
                )
                pipeline = build_pipeline(
                    classifier_name,
                    clf_variant,
                    imbalance_strategy=imbalance_strategy,
                )
                pipeline.set_params(**params)

                X_inner_train = X_train[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_test = X_train[inner_test_idx]
                y_inner_test = y_train[inner_test_idx]
                groups_inner_test = groups_train[inner_test_idx]

                fit_kwargs = _get_fit_kwargs(
                    classifier_name,
                    y_inner_train,
                    imbalance_strategy,
                )
                pipeline.fit(X_inner_train, y_inner_train, **fit_kwargs)

                y_inner_pred = pipeline.predict(X_inner_test)
                y_inner_prob = pipeline.predict_proba(X_inner_test)[:, 1]

                row_true_all.append(y_inner_test)
                row_pred_all.append(y_inner_pred)
                row_prob_all.append(y_inner_prob)
                row_subjects_all.append(groups_inner_test)

            row_true = np.concatenate(row_true_all)
            row_pred = np.concatenate(row_pred_all)
            row_prob = np.concatenate(row_prob_all)
            row_subjects = np.concatenate(row_subjects_all)
            subj_true, subj_pred, subj_scores, subject_order = _subject_level_arrays(
                y_true=row_true,
                y_pred=row_pred,
                y_prob=row_prob,
                subject_ids=row_subjects,
                subject_aggregation_rule=subject_aggregation_rule,
            )
            subject_log_loss = _subject_level_log_loss(subj_true, subj_scores)

            candidate_summaries.append({
                'candidate_index': candidate_counter,
                'imbalance_strategy': imbalance_strategy,
                'params': dict(params),
                'inner_subject_f1': float(f1_score(
                    subj_true, subj_pred, average='macro'
                )),
                'inner_stride_f1': float(f1_score(
                    row_true, row_pred, average='macro'
                )),
                'inner_subject_log_loss': (
                    float(subject_log_loss)
                    if subject_log_loss is not None else None
                ),
                'aggregation_rule': subject_aggregation_rule,
                'tie_break_rule': tie_break_rule,
                'subject_ids': subject_order,
                'subject_y_true': subj_true.tolist(),
                'subject_y_pred': subj_pred.tolist(),
                'subject_scores': subj_scores.tolist(),
            })
            candidate_counter += 1

    ranked = sorted(
        candidate_summaries,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            tie_break_rule=tie_break_rule,
        ),
    )
    return ranked[0], ranked


def run_nested_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_template: Any,
    param_grid: dict[str, list],
    classifier_name: str,
    candidate_imbalance_strategies: tuple[str, ...] | dict[str, tuple[str, ...]] = (
        'synthetic',
        'balanced',
        'raw',
    ),
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
) -> dict[str, Any]:
    """
    Outer LOSO-CV loop with manual grouped inner selection on the training fold.

    The v4 protocol replaces per-fold GridSearchCV scoring on single-subject
    inner folds with pooled evaluation across all held-out inner subjects. For
    each outer fold and each candidate (imbalance strategy × hyperparameter
    configuration), we:
      1. fit on inner-training subjects only;
      2. predict the held-out inner subject;
      3. pool predictions across all inner held-out subjects;
      4. score pooled subject-level macro-F1 first, pooled subject-level log
         loss second, and use a deterministic lexicographic fallback last.

    The winning candidate for each outer fold is then refit on the full outer
    training pool and evaluated on the held-out outer subject.

    Args:
        X: Feature matrix, shape (n_strides, n_features), float64.
        y: Binary label vector, shape (n_strides,), int.
        groups: Subject-ID array, shape (n_strides,), str or object.
        clf_template: Base sklearn-compatible classifier object.
        param_grid: Dict mapping pipeline step param names to value lists.
        classifier_name: Short classifier name used to derive fit kwargs.
        candidate_imbalance_strategies: Imbalance strategies admitted to the
            grouped inner selection loop.

    Returns:
        Dict with aggregate stride- and subject-level predictions plus detailed
        per-fold grouped-selection metadata.
    """
    outer_loso = LeaveOneGroupOut()

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    subject_ids_all: list[np.ndarray] = []
    fold_params: list[dict] = []
    fold_best_scores: list[float] = []
    fold_best_stride_scores: list[float] = []
    fold_best_strategies: list[str] = []
    outer_fold_details: list[dict[str, Any]] = []
    fitted_outer_pipelines: list[Any] = []

    for train_idx, test_idx in outer_loso.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        best_candidate, candidate_rankings = _select_best_grouped_candidate(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            clf_template=clf_template,
            param_grid=param_grid,
            classifier_name=classifier_name,
            candidate_imbalance_strategies=candidate_imbalance_strategies,
            subject_aggregation_rule=subject_aggregation_rule,
            tie_break_rule=tie_break_rule,
        )
        fold_params.append(best_candidate['params'])
        fold_best_scores.append(float(best_candidate['inner_subject_f1']))
        fold_best_stride_scores.append(
            float(best_candidate['inner_stride_f1']))
        fold_best_strategies.append(str(best_candidate['imbalance_strategy']))

        clf_variant = _configure_classifier_for_resampling(
            classifier_name,
            clf_template,
            best_candidate['imbalance_strategy'],
        )
        pipeline = build_pipeline(
            classifier_name,
            clf_variant,
            imbalance_strategy=best_candidate['imbalance_strategy'],
        )
        pipeline.set_params(**best_candidate['params'])

        fit_kwargs = _get_fit_kwargs(
            classifier_name,
            y_train,
            best_candidate['imbalance_strategy'],
        )
        pipeline.fit(X_train, y_train, **fit_kwargs)
        fitted_outer_pipelines.append(pipeline)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        subject_ids_all.append(groups_test)
        outer_fold_details.append({
            'held_out_subject_id': str(groups_test[0]),
            'held_out_true_label': int(y_test[0]),
            'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
            'selected_params': dict(best_candidate['params']),
            'selected_inner_subject_f1': float(best_candidate['inner_subject_f1']),
            'selected_inner_stride_f1': float(best_candidate['inner_stride_f1']),
            'selected_inner_subject_log_loss': (
                float(best_candidate['inner_subject_log_loss'])
                if best_candidate.get('inner_subject_log_loss') is not None else None
            ),
            'candidate_rankings': [
                {
                    'rank': rank_idx + 1,
                    'candidate_index': int(candidate['candidate_index']),
                    'imbalance_strategy': candidate['imbalance_strategy'],
                    'params': dict(candidate['params']),
                    'inner_subject_f1': float(candidate['inner_subject_f1']),
                    'inner_stride_f1': float(candidate['inner_stride_f1']),
                    'inner_subject_log_loss': (
                        float(candidate['inner_subject_log_loss'])
                        if candidate.get('inner_subject_log_loss') is not None else None
                    ),
                }
                for rank_idx, candidate in enumerate(candidate_rankings)
            ],
        })

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    y_prob_concat = np.concatenate(y_prob_all)
    subject_ids_concat = np.concatenate(subject_ids_all)
    subj_true_concat, subj_pred_concat, subj_prob_concat, subj_ids_concat = _subject_level_arrays(
        y_true=y_true_concat,
        y_pred=y_pred_concat,
        y_prob=y_prob_concat,
        subject_ids=subject_ids_concat,
        subject_aggregation_rule=subject_aggregation_rule,
    )

    return {
        'f1_macro':         f1_score(y_true_concat, y_pred_concat, average='macro'),
        'y_true_all':       y_true_concat,
        'y_pred_all':       y_pred_concat,
        'y_prob_all':       y_prob_concat,
        'subject_ids_all':  subject_ids_concat,
        'fold_params':      fold_params,
        'fold_best_scores': fold_best_scores,
        'fold_best_stride_scores': fold_best_stride_scores,
        'fold_best_strategies': fold_best_strategies,
        'outer_folds': outer_fold_details,
        'fitted_outer_pipelines': fitted_outer_pipelines,
        'subject_aggregation_rule': subject_aggregation_rule,
        'tie_break_rule': tie_break_rule,
        'subject_level': {
            'y_true_all': subj_true_concat,
            'y_pred_all': subj_pred_concat,
            'y_prob_all': subj_prob_concat,
            'subject_ids_all': subj_ids_concat,
            'f1_macro': float(f1_score(subj_true_concat, subj_pred_concat, average='macro')),
        },
    }


def run_grouped_outer_fold(
    *,
    condition: str,
    clf_name: str,
    clf_template: Any,
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_fold_index: int,
    candidate_imbalance_strategies: tuple[str, ...],
    models_dir: Path | None = None,
    selection_trace_dir: Path | None = None,
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
) -> dict[str, Any]:
    """
    Evaluate one outer LOSO fold under the grouped v4 selector.

    This is the fold-granular companion to ``run_nested_loso`` used by the
    Step 8 recovery runner to checkpoint expensive classifiers at the outer-fold
    level without changing grouped candidate selection semantics.
    """
    outer_loso = LeaveOneGroupOut()
    splits = list(outer_loso.split(X, y, groups))
    if outer_fold_index < 0 or outer_fold_index >= len(splits):
        raise IndexError(
            f'outer_fold_index {outer_fold_index} is out of range for '
            f'{len(splits)} grouped folds.'
        )

    train_idx, test_idx = splits[outer_fold_index]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    best_candidate, candidate_rankings = _select_best_grouped_candidate(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        clf_template=clf_template,
        param_grid=param_grid,
        classifier_name=clf_name,
        candidate_imbalance_strategies=candidate_imbalance_strategies,
        subject_aggregation_rule=subject_aggregation_rule,
        tie_break_rule=tie_break_rule,
    )

    clf_variant = _configure_classifier_for_resampling(
        clf_name,
        clf_template,
        best_candidate['imbalance_strategy'],
    )
    pipeline = build_pipeline(
        clf_name,
        clf_variant,
        imbalance_strategy=best_candidate['imbalance_strategy'],
    )
    pipeline.set_params(**best_candidate['params'])
    fit_kwargs = _get_fit_kwargs(
        clf_name,
        y_train,
        best_candidate['imbalance_strategy'],
    )
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    held_out_subject_id = str(groups_test[0])

    if selection_trace_dir is not None:
        selection_trace_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = selection_trace_dir / (
            f'{condition}_{clf_name}_fold_{outer_fold_index:02d}_candidate_trace.npz'
        )
        sidecar_metadata = {
            'kind': 'outer_fold_grouped_selection_trace',
            'condition': condition,
            'classifier': clf_name,
            'outer_fold_index': outer_fold_index,
            'held_out_subject_id': held_out_subject_id,
            'aggregation_rule': subject_aggregation_rule,
            'tie_break_rule': tie_break_rule,
            'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        }
        candidate_trace_sha256 = _save_selection_trace_sidecar(
            sidecar_path=sidecar_path,
            candidates=[
                {
                    'rank': rank_idx + 1,
                    'candidate_index': int(candidate['candidate_index']),
                    'imbalance_strategy': candidate['imbalance_strategy'],
                    'params': dict(candidate['params']),
                    'inner_subject_f1': float(candidate['inner_subject_f1']),
                    'inner_stride_f1': float(candidate['inner_stride_f1']),
                    'inner_subject_log_loss': (
                        float(candidate['inner_subject_log_loss'])
                        if candidate.get('inner_subject_log_loss') is not None else None
                    ),
                    'subject_ids': list(candidate.get('subject_ids', [])),
                    'subject_y_true': list(candidate.get('subject_y_true', [])),
                    'subject_y_pred': list(candidate.get('subject_y_pred', [])),
                    'subject_scores': list(candidate.get('subject_scores', [])),
                }
                for rank_idx, candidate in enumerate(candidate_rankings)
            ],
            metadata=sidecar_metadata,
        )
        candidate_trace_relpath = str(sidecar_path)
    else:
        candidate_trace_sha256 = None
        candidate_trace_relpath = None

    if models_dir is not None:
        fold_model_path = (
            models_dir
            / 'within_folds'
            / f'{condition}_{clf_name}_fold_{outer_fold_index:02d}_{held_out_subject_id}.joblib'
        )
        fold_model_sha256 = _save_pipeline_artifact(pipeline, fold_model_path)
        fold_model_relpath = str(fold_model_path.relative_to(models_dir))
    else:
        fold_model_sha256 = None
        fold_model_relpath = None

    return {
        'outer_fold_index': int(outer_fold_index),
        'held_out_subject_id': held_out_subject_id,
        'held_out_true_label': int(y_test[0]),
        'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
        'selected_params': dict(best_candidate['params']),
        'selected_inner_subject_f1': float(best_candidate['inner_subject_f1']),
        'selected_inner_stride_f1': float(best_candidate['inner_stride_f1']),
        'selected_inner_subject_log_loss': (
            float(best_candidate['inner_subject_log_loss'])
            if best_candidate.get('inner_subject_log_loss') is not None else None
        ),
        'candidate_rankings': [
            {
                'rank': rank_idx + 1,
                'candidate_index': int(candidate['candidate_index']),
                'imbalance_strategy': candidate['imbalance_strategy'],
                'params': dict(candidate['params']),
                'inner_subject_f1': float(candidate['inner_subject_f1']),
                'inner_stride_f1': float(candidate['inner_stride_f1']),
                'inner_subject_log_loss': (
                    float(candidate['inner_subject_log_loss'])
                    if candidate.get('inner_subject_log_loss') is not None else None
                ),
            }
            for rank_idx, candidate in enumerate(candidate_rankings)
        ],
        'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        'subject_aggregation_rule': subject_aggregation_rule,
        'tie_break_rule': tie_break_rule,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist(),
        'subject_ids': groups_test.tolist(),
        'candidate_trace_relpath': candidate_trace_relpath,
        'candidate_trace_sha256': candidate_trace_sha256,
        'fold_model_relpath': fold_model_relpath,
        'fold_model_sha256': fold_model_sha256,
    }


def get_modal_params(
    fold_params: list[dict],
    fold_scores: list[float] | None = None,
) -> dict:
    """
    Return the modal (most frequently selected) parameter combination.

    Each element of fold_params is the best_params_ dict from one outer LOSO
    fold. The mode is the combination that appears most often across folds.

    Tiebreaker: when two or more combinations appear equally often, the one
    with the higher mean inner-CV score (from fold_scores) across the folds
    where it was selected is returned. If fold_scores is not provided, the
    first-encountered tied combination is returned.
    """
    keys = [_params_to_key(p) for p in fold_params]
    counter: collections.Counter = collections.Counter(keys)
    top_count = counter.most_common(1)[0][1]

    tied = [k for k, c in counter.items() if c == top_count]

    if len(tied) == 1 or fold_scores is None:
        return dict(tied[0])

    best_key = max(
        tied,
        key=lambda k: float(np.mean([
            s for key, s in zip(keys, fold_scores) if key == k
        ])),
    )
    return dict(best_key)


def get_modal_strategy(
    fold_strategies: list[str],
    fold_scores: list[float] | None = None,
) -> str:
    """
    Return the modal imbalance strategy, tie-broken by mean inner subject-F1.
    """
    counter: collections.Counter[str] = collections.Counter(fold_strategies)
    top_count = counter.most_common(1)[0][1]
    tied = [strategy for strategy, count in counter.items() if count ==
            top_count]
    if len(tied) == 1 or fold_scores is None:
        return tied[0]
    best_strategy = max(
        tied,
        key=lambda strategy: float(np.mean([
            score
            for fold_strategy, score in zip(fold_strategies, fold_scores)
            if fold_strategy == strategy
        ])),
    )
    return best_strategy


# ── Classifier configurations ─────────────────────────────────────────────────

def get_classifier_configs() -> dict[str, dict[str, Any]]:
    """
    Return instantiated classifiers and their tuning grids for all 7 classifiers.

    Each value is a dict with keys:
      - 'clf': instantiated classifier
      - 'param_grid': hyperparameter grid using clf__ prefixes for ImbPipeline
    compatibility.

    Parallelism is handled at the GridSearchCV level.

    SVM is instantiated with probability=True, which is required for SHAP
    KernelExplainer to compute probability-based explanations.
    """
    configs: dict[str, dict[str, Any]] = {
        'rf': {
            'clf': RandomForestClassifier(
                random_state=42,
                n_jobs=1,
            ),
            'param_grid': {
                'clf__n_estimators':     [100, 300],
                'clf__max_depth':        [None, 10, 20],
                'clf__max_features':     ['sqrt', None],
                'clf__min_samples_leaf': [1, 2, 5],
            },
        },
        'knn': {
            'clf': KNeighborsClassifier(),
            'param_grid': {
                'clf__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'clf__weights':     ['distance', 'uniform'],
                'clf__metric':      ['euclidean', 'manhattan'],
            },
        },
        'svm': {
            'clf': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
            ),
            'param_grid': {
                'clf__C':     [0.1, 1, 10, 100],
                'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            },
        },
        'dt': {
            'clf': DecisionTreeClassifier(
                random_state=42,
            ),
            'param_grid': {
                'clf__max_depth':        [None, 5, 10, 20],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__criterion':        ['gini', 'entropy'],
            },
        },
        'qda': {
            'clf': QuadraticDiscriminantAnalysis(),
            'param_grid': {
                'clf__reg_param': [0.001, 0.01, 0.1, 0.5, 0.9],
            },
        },
        'xgb': {
            'clf': XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                n_jobs=1,
                tree_method='hist',
            ),
            'param_grid': {
                'clf__n_estimators':     [100, 200],
                'clf__max_depth':        [3, 5],
                'clf__learning_rate':    [0.01, 0.1, 0.3],
                'clf__subsample':        [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0],
            },
        },
        'lgbm': {
            'clf': LGBMClassifier(
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
            'param_grid': {
                'clf__n_estimators':      [100, 200],
                'clf__max_depth':         [-1, 5],
                'clf__learning_rate':     [0.01, 0.1, 0.3],
                'clf__num_leaves':        [31],
                'clf__subsample':         [0.8, 1.0],
                'clf__feature_fraction':  [0.8, 1.0],
                'clf__min_child_samples': [20],
                'clf__subsample_freq':    [1],
            },
        },
    }
    return configs


def _save_pipeline_artifact(pipeline: Any, path: Path) -> str:
    """Persist a fitted pipeline and return its SHA-256 hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return _sha256_file(path)


def _fit_grouped_full_source_model(
    *,
    clf_name: str,
    clf: Any,
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_imbalance_strategies: tuple[str, ...],
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
) -> dict[str, Any]:
    """
    Select and fit the final full-source model using grouped inner LOSO only.

    This uses the same candidate space and grouped objective as the outer-fold
    selector, but all available source subjects participate in the inner LOSO.
    """
    best_candidate, selection_trace = _select_best_grouped_candidate(
        X_train=X,
        y_train=y,
        groups_train=groups,
        clf_template=clf,
        param_grid=param_grid,
        classifier_name=clf_name,
        candidate_imbalance_strategies=candidate_imbalance_strategies,
        subject_aggregation_rule=subject_aggregation_rule,
        tie_break_rule=tie_break_rule,
    )
    best_strategy = str(best_candidate['imbalance_strategy'])
    best_params = dict(best_candidate['params'])

    clf_variant = _configure_classifier_for_resampling(
        clf_name,
        clf,
        best_strategy,
    )
    pipeline = build_pipeline(
        clf_name,
        clf_variant,
        imbalance_strategy=best_strategy,
    )
    pipeline.set_params(**best_params)
    fit_kwargs = _get_fit_kwargs(clf_name, y, best_strategy)
    pipeline.fit(X, y, **fit_kwargs)

    return {
        'pipeline': pipeline,
        'selected_imbalance_strategy': best_strategy,
        'selected_params': best_params,
        'selection_subject_f1': float(best_candidate['inner_subject_f1']),
        'selection_stride_f1': float(best_candidate['inner_stride_f1']),
        'selection_subject_log_loss': best_candidate.get('inner_subject_log_loss'),
        'selection_trace': selection_trace,
        'subject_aggregation_rule': subject_aggregation_rule,
        'tie_break_rule': tie_break_rule,
    }


def fit_grouped_full_source_with_artifacts(
    *,
    condition: str,
    clf_name: str,
    clf: Any,
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_imbalance_strategies: tuple[str, ...],
    models_dir: Path | None = None,
    selection_trace_dir: Path | None = None,
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
) -> dict[str, Any]:
    """Fit the grouped full-source model and persist its canonical artifacts."""
    full_source = _fit_grouped_full_source_model(
        clf_name=clf_name,
        clf=clf,
        param_grid=param_grid,
        X=X,
        y=y,
        groups=groups,
        candidate_imbalance_strategies=candidate_imbalance_strategies,
        subject_aggregation_rule=subject_aggregation_rule,
        tie_break_rule=tie_break_rule,
    )

    if selection_trace_dir is not None:
        selection_trace_dir.mkdir(parents=True, exist_ok=True)
        full_trace_path = selection_trace_dir / (
            f'{condition}_{clf_name}_full_source_candidate_trace.npz'
        )
        full_trace_metadata = {
            'kind': 'full_source_grouped_selection_trace',
            'condition': condition,
            'classifier': clf_name,
            'aggregation_rule': subject_aggregation_rule,
            'tie_break_rule': tie_break_rule,
            'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        }
        selection_trace_sha256 = _save_selection_trace_sidecar(
            sidecar_path=full_trace_path,
            candidates=full_source['selection_trace'],
            metadata=full_trace_metadata,
        )
        selection_trace_path = str(full_trace_path)
    else:
        selection_trace_sha256 = None
        selection_trace_path = None

    if models_dir is not None:
        model_path = models_dir / f'{condition}_{clf_name}.joblib'
        model_sha256 = _save_pipeline_artifact(
            full_source['pipeline'],
            model_path,
        )
        model_relpath = str(model_path.relative_to(models_dir))
    else:
        model_sha256 = None
        model_relpath = None

    return {
        'selected_imbalance_strategy': full_source['selected_imbalance_strategy'],
        'selected_params': _json_safe(full_source['selected_params']),
        'selection_subject_f1': float(full_source['selection_subject_f1']),
        'selection_stride_f1': float(full_source['selection_stride_f1']),
        'selection_subject_log_loss': (
            float(full_source['selection_subject_log_loss'])
            if full_source.get('selection_subject_log_loss') is not None else None
        ),
        'selection_trace': _json_safe(full_source['selection_trace']),
        'selection_trace_path': selection_trace_path,
        'selection_trace_sha256': selection_trace_sha256,
        'model_path': model_relpath,
        'model_sha256': model_sha256,
        'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        'subject_aggregation_rule': subject_aggregation_rule,
        'tie_break_rule': tie_break_rule,
    }


def _strategy_to_legacy_label(imbalance_strategy: str) -> str:
    """Map the explicit v3 imbalance strategy to the legacy v2 JSON label."""
    if imbalance_strategy == 'synthetic':
        return 'smote'
    if imbalance_strategy == 'balanced':
        return 'no_smote'
    if imbalance_strategy == 'raw':
        return 'raw_unbalanced'
    raise ValueError(f"Unknown imbalance strategy '{imbalance_strategy}'")


def _evaluate_within_condition_classifier(
    *,
    condition: str,
    clf_name: str,
    clf: Any,
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_imbalance_strategies: tuple[str, ...],
    models_dir: Path | None,
    selection_trace_dir: Path | None,
    subject_aggregation_rule: str,
    tie_break_rule: str,
) -> dict[str, Any]:
    """
    Run one within-condition classifier evaluation under the v4 grouped protocol.

    The candidate space is the Cartesian product of candidate imbalance
    strategies and the classifier's hyperparameter grid. Inner selection is
    subject-pooled and the final full-source model is retuned on all source
    subjects with the same grouped selection logic.
    """
    t_start = time.time()
    start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f'  {clf_name:<6} [grouped {"+".join(candidate_imbalance_strategies)}] '
        f'started: {start_ts}',
        flush=True,
    )

    loso_out = run_nested_loso(
        X=X,
        y=y,
        groups=groups,
        clf_template=clf,
        param_grid=param_grid,
        classifier_name=clf_name,
        candidate_imbalance_strategies=candidate_imbalance_strategies,
        subject_aggregation_rule=subject_aggregation_rule,
        tie_break_rule=tie_break_rule,
    )

    elapsed = time.time() - t_start
    end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    outer_modal_params = get_modal_params(
        loso_out['fold_params'],
        loso_out['fold_best_scores'],
    )
    outer_modal_strategy = get_modal_strategy(
        loso_out['fold_best_strategies'],
        loso_out['fold_best_scores'],
    )

    modal_key = _params_to_key(outer_modal_params)
    outer_modal_frequency = sum(
        1 for p in loso_out['fold_params'] if _params_to_key(p) == modal_key
    )
    outer_modal_strategy_frequency = sum(
        1 for strategy in loso_out['fold_best_strategies']
        if strategy == outer_modal_strategy
    )

    f1_stored = round(float(loso_out['f1_macro']), 6)
    y_true = loso_out['y_true_all']
    y_pred = loso_out['y_pred_all']
    y_prob = loso_out['y_prob_all']

    # Verify the stored lists reproduce the stored F1 exactly.
    assert abs(f1_score(y_true, y_pred, average='macro') - f1_stored) < 1e-6, (
        f'{clf_name}: F1 mismatch between stored value and prediction lists'
    )

    ci_rng = np.random.default_rng(42)
    ci_lower, ci_upper, n_rejected, rejection_rate = _subject_resampled_stride_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        subject_ids=loso_out['subject_ids_all'],
        rng=ci_rng,
        n_resamples=10_000,
    )
    subject_metrics = _subject_level_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        subject_ids=loso_out['subject_ids_all'],
        subject_aggregation_rule=subject_aggregation_rule,
    )
    subj_true = np.asarray(subject_metrics['y_true'], dtype=int)
    subj_pred = np.asarray(subject_metrics['y_pred'], dtype=int)
    subject_ci_lower, subject_ci_upper, subject_n_rejected, subject_rejection_rate = (
        _subject_primary_bootstrap_ci(
            y_true_subject=subj_true,
            y_pred_subject=subj_pred,
            rng=np.random.default_rng(4242),
            n_resamples=10_000,
        )
    )

    if n_rejected > 0:
        print(
            f'    [{clf_name}] subject bootstrap rejected '
            f'{n_rejected} degenerate resamples ({rejection_rate:.2%})',
            flush=True,
        )

    full_source = _fit_grouped_full_source_model(
        clf_name=clf_name,
        clf=clf,
        param_grid=param_grid,
        X=X,
        y=y,
        groups=groups,
        candidate_imbalance_strategies=candidate_imbalance_strategies,
        subject_aggregation_rule=subject_aggregation_rule,
        tie_break_rule=tie_break_rule,
    )

    if selection_trace_dir is not None:
        selection_trace_dir.mkdir(parents=True, exist_ok=True)
        for fold_idx, fold_detail in enumerate(loso_out['outer_folds']):
            sidecar_path = selection_trace_dir / (
                f'{condition}_{clf_name}_fold_{fold_idx:02d}_candidate_trace.npz'
            )
            sidecar_metadata = {
                'kind': 'outer_fold_grouped_selection_trace',
                'condition': condition,
                'classifier': clf_name,
                'outer_fold_index': fold_idx,
                'held_out_subject_id': fold_detail['held_out_subject_id'],
                'aggregation_rule': subject_aggregation_rule,
                'tie_break_rule': tie_break_rule,
                'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
            }
            sidecar_sha = _save_selection_trace_sidecar(
                sidecar_path=sidecar_path,
                candidates=fold_detail['candidate_rankings'],
                metadata=sidecar_metadata,
            )
            fold_detail['candidate_trace_relpath'] = str(sidecar_path)
            fold_detail['candidate_trace_sha256'] = sidecar_sha

        full_trace_path = selection_trace_dir / (
            f'{condition}_{clf_name}_full_source_candidate_trace.npz'
        )
        full_trace_metadata = {
            'kind': 'full_source_grouped_selection_trace',
            'condition': condition,
            'classifier': clf_name,
            'aggregation_rule': subject_aggregation_rule,
            'tie_break_rule': tie_break_rule,
            'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        }
        full_trace_sha = _save_selection_trace_sidecar(
            sidecar_path=full_trace_path,
            candidates=full_source['selection_trace'],
            metadata=full_trace_metadata,
        )
    else:
        full_trace_path = None
        full_trace_sha = None

    full_source_model_relpath: str | None = None
    full_source_model_sha256: str | None = None
    if models_dir is not None:
        full_source_model_path = models_dir / f'{condition}_{clf_name}.joblib'
        full_source_model_sha256 = _save_pipeline_artifact(
            full_source['pipeline'],
            full_source_model_path,
        )
        full_source_model_relpath = str(
            full_source_model_path.relative_to(models_dir))

        for fold_idx, (fold_detail, fold_pipeline) in enumerate(zip(
            loso_out['outer_folds'],
            loso_out['fitted_outer_pipelines'],
        )):
            subject_id = fold_detail['held_out_subject_id']
            fold_model_path = (
                models_dir
                / 'within_folds'
                / f'{condition}_{clf_name}_fold_{fold_idx:02d}_{subject_id}.joblib'
            )
            fold_sha = _save_pipeline_artifact(fold_pipeline, fold_model_path)
            fold_detail['fold_model_relpath'] = str(
                fold_model_path.relative_to(models_dir))
            fold_detail['fold_model_sha256'] = fold_sha

    print(
        f'  {clf_name:<6} F1={f1_stored:.4f} '
        f'subj_F1={subject_metrics["f1_macro"]:.4f} '
        f'subj_CI=[{subject_ci_lower:.4f},{subject_ci_upper:.4f}] '
        f'outer_modal={outer_modal_strategy}/{outer_modal_params} '
        f'full_source={full_source["selected_imbalance_strategy"]}/{full_source["selected_params"]} '
        f'start={start_ts} end={end_ts} ({elapsed:.0f}s)',
        flush=True,
    )

    return {
        'f1_macro': f1_stored,
        'f1_macro_ci_lower': round(ci_lower, 6),
        'f1_macro_ci_upper': round(ci_upper, 6),
        'subject_resampled_stride_f1_ci_lower': round(ci_lower, 6),
        'subject_resampled_stride_f1_ci_upper': round(ci_upper, 6),
        'subject_primary_f1_ci_lower': round(subject_ci_lower, 6),
        'subject_primary_f1_ci_upper': round(subject_ci_upper, 6),
        'subject_primary_bootstrap_rejections': subject_n_rejected,
        'subject_primary_bootstrap_rejection_rate': round(subject_rejection_rate, 6),
        'subject_resampled_stride_bootstrap_rejections': n_rejected,
        'subject_resampled_stride_bootstrap_rejection_rate': round(rejection_rate, 6),
        'subject_primary_f1_macro': round(float(subject_metrics['f1_macro']), 6),
        'subject_aggregation_rule': subject_aggregation_rule,
        'tie_break_rule': tie_break_rule,
        'modal_params': outer_modal_params,
        'modal_frequency': outer_modal_frequency,
        'modal_strategy': outer_modal_strategy,
        'modal_strategy_frequency': outer_modal_strategy_frequency,
        'subject_metrics': subject_metrics,
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': np.round(y_prob, 6).tolist(),
        'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        'outer_fold_selection_trace': _json_safe(loso_out['outer_folds']),
        'full_source_selected_params': _json_safe(full_source['selected_params']),
        'full_source_selected_imbalance_strategy': full_source['selected_imbalance_strategy'],
        'full_source_selection_subject_f1': round(float(full_source['selection_subject_f1']), 6),
        'full_source_selection_stride_f1': round(float(full_source['selection_stride_f1']), 6),
        'full_source_selection_subject_log_loss': (
            round(float(full_source['selection_subject_log_loss']), 6)
            if full_source.get('selection_subject_log_loss') is not None else None
        ),
        'full_source_selection_trace': _json_safe(full_source['selection_trace']),
        'full_source_selection_trace_path': str(full_trace_path) if full_trace_path is not None else None,
        'full_source_selection_trace_sha256': full_trace_sha,
        'full_source_model_path': full_source_model_relpath,
        'full_source_model_sha256': full_source_model_sha256,
    }


def assemble_grouped_within_classifier_result(
    *,
    fold_outputs: list[dict[str, Any]],
    full_source_artifacts: dict[str, Any],
    candidate_imbalance_strategies: tuple[str, ...],
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
) -> dict[str, Any]:
    """
    Assemble a within-condition classifier payload from persisted fold shards.

    This mirrors the payload produced by ``_evaluate_within_condition_classifier``
    so callers can resume long-running classifiers without changing downstream
    schema contracts.
    """
    if not fold_outputs:
        raise ValueError('At least one outer-fold shard is required.')

    ordered_folds = sorted(
        (_json_safe(dict(fold)) for fold in fold_outputs),
        key=lambda fold: int(fold['outer_fold_index']),
    )
    expected_indices = list(range(len(ordered_folds)))
    actual_indices = [int(fold['outer_fold_index']) for fold in ordered_folds]
    if actual_indices != expected_indices:
        raise ValueError(
            'Outer-fold shard indices are incomplete or out of order: '
            f'expected={expected_indices}, got={actual_indices}'
        )

    y_true = np.concatenate([
        np.asarray(fold['y_true'], dtype=int) for fold in ordered_folds
    ])
    y_pred = np.concatenate([
        np.asarray(fold['y_pred'], dtype=int) for fold in ordered_folds
    ])
    y_prob = np.concatenate([
        np.asarray(fold['y_prob'], dtype=float) for fold in ordered_folds
    ])
    subject_ids = np.concatenate([
        np.asarray(fold['subject_ids'], dtype=object) for fold in ordered_folds
    ])

    f1_stored = round(float(f1_score(y_true, y_pred, average='macro')), 6)
    assert abs(f1_score(y_true, y_pred, average='macro') - f1_stored) < 1e-6

    fold_params = [dict(fold['selected_params']) for fold in ordered_folds]
    fold_best_scores = [
        float(fold['selected_inner_subject_f1']) for fold in ordered_folds
    ]
    fold_best_stride_scores = [
        float(fold['selected_inner_stride_f1']) for fold in ordered_folds
    ]
    fold_best_strategies = [
        str(fold['selected_imbalance_strategy']) for fold in ordered_folds
    ]

    outer_modal_params = get_modal_params(
        fold_params,
        fold_best_scores,
    )
    outer_modal_strategy = get_modal_strategy(
        fold_best_strategies,
        fold_best_scores,
    )
    modal_key = _params_to_key(outer_modal_params)
    outer_modal_frequency = sum(
        1 for params in fold_params if _params_to_key(params) == modal_key
    )
    outer_modal_strategy_frequency = sum(
        1 for strategy in fold_best_strategies if strategy == outer_modal_strategy
    )

    ci_rng = np.random.default_rng(42)
    ci_lower, ci_upper, n_rejected, rejection_rate = _subject_resampled_stride_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        subject_ids=subject_ids,
        rng=ci_rng,
        n_resamples=10_000,
    )
    subject_metrics = _subject_level_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        subject_ids=subject_ids,
        subject_aggregation_rule=subject_aggregation_rule,
    )
    subj_true = np.asarray(subject_metrics['y_true'], dtype=int)
    subj_pred = np.asarray(subject_metrics['y_pred'], dtype=int)
    subject_ci_lower, subject_ci_upper, subject_n_rejected, subject_rejection_rate = (
        _subject_primary_bootstrap_ci(
            y_true_subject=subj_true,
            y_pred_subject=subj_pred,
            rng=np.random.default_rng(4242),
            n_resamples=10_000,
        )
    )

    return {
        'f1_macro': f1_stored,
        'f1_macro_ci_lower': round(ci_lower, 6),
        'f1_macro_ci_upper': round(ci_upper, 6),
        'subject_resampled_stride_f1_ci_lower': round(ci_lower, 6),
        'subject_resampled_stride_f1_ci_upper': round(ci_upper, 6),
        'subject_primary_f1_ci_lower': round(subject_ci_lower, 6),
        'subject_primary_f1_ci_upper': round(subject_ci_upper, 6),
        'subject_primary_bootstrap_rejections': subject_n_rejected,
        'subject_primary_bootstrap_rejection_rate': round(subject_rejection_rate, 6),
        'subject_resampled_stride_bootstrap_rejections': n_rejected,
        'subject_resampled_stride_bootstrap_rejection_rate': round(rejection_rate, 6),
        'subject_primary_f1_macro': round(float(subject_metrics['f1_macro']), 6),
        'subject_aggregation_rule': subject_aggregation_rule,
        'tie_break_rule': tie_break_rule,
        'modal_params': outer_modal_params,
        'modal_frequency': outer_modal_frequency,
        'modal_strategy': outer_modal_strategy,
        'modal_strategy_frequency': outer_modal_strategy_frequency,
        'subject_metrics': subject_metrics,
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': np.round(y_prob, 6).tolist(),
        'candidate_imbalance_strategies': list(candidate_imbalance_strategies),
        'outer_fold_selection_trace': [
            {
                'held_out_subject_id': str(fold['held_out_subject_id']),
                'held_out_true_label': int(fold['held_out_true_label']),
                'selected_imbalance_strategy': str(fold['selected_imbalance_strategy']),
                'selected_params': dict(fold['selected_params']),
                'selected_inner_subject_f1': float(fold['selected_inner_subject_f1']),
                'selected_inner_stride_f1': float(fold['selected_inner_stride_f1']),
                'selected_inner_subject_log_loss': (
                    float(fold['selected_inner_subject_log_loss'])
                    if fold.get('selected_inner_subject_log_loss') is not None else None
                ),
                'candidate_rankings': _json_safe(fold['candidate_rankings']),
                'candidate_trace_relpath': fold.get('candidate_trace_relpath'),
                'candidate_trace_sha256': fold.get('candidate_trace_sha256'),
                'fold_model_relpath': fold.get('fold_model_relpath'),
                'fold_model_sha256': fold.get('fold_model_sha256'),
            }
            for fold in ordered_folds
        ],
        'full_source_selected_params': _json_safe(full_source_artifacts['selected_params']),
        'full_source_selected_imbalance_strategy': full_source_artifacts['selected_imbalance_strategy'],
        'full_source_selection_subject_f1': round(
            float(full_source_artifacts['selection_subject_f1']), 6),
        'full_source_selection_stride_f1': round(
            float(full_source_artifacts['selection_stride_f1']), 6),
        'full_source_selection_subject_log_loss': (
            round(float(full_source_artifacts['selection_subject_log_loss']), 6)
            if full_source_artifacts.get('selection_subject_log_loss') is not None else None
        ),
        'full_source_selection_trace': _json_safe(full_source_artifacts['selection_trace']),
        'full_source_selection_trace_path': full_source_artifacts.get('selection_trace_path'),
        'full_source_selection_trace_sha256': full_source_artifacts.get('selection_trace_sha256'),
        'full_source_model_path': full_source_artifacts.get('model_path'),
        'full_source_model_sha256': full_source_artifacts.get('model_sha256'),
    }


def _build_within_condition_output(
    *,
    condition: str,
    pool_subjects: int,
    pool_strides: int,
    selected_feature_cols: list[str],
    feature_matrix_file: str,
    feature_set_version: str,
    normalization: str,
    models_dir: str | None,
    clf_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON payload shared by final and partial within-condition writes."""
    return {
        'condition': condition,
        'pool_subjects': pool_subjects,
        'pool_strides': pool_strides,
        'feature_cols': selected_feature_cols,
        'n_features': len(selected_feature_cols),
        'feature_matrix_file': feature_matrix_file,
        'feature_set_version': feature_set_version,
        'normalization': normalization,
        'models_dir': models_dir,
        'classifiers': clf_results,
    }


# ── Full within-condition orchestration ───────────────────────────────────────

def run_within_condition(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str] | None = None,
    results_dir: str | Path = 'experiments/results',
    feature_cols: list[str] | None = None,
    feature_matrix_file: str = DEFAULT_FEATURE_MATRIX_FILE,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    normalization: str = DEFAULT_NORMALIZATION,
    *,
    control_subjects: list[str] | None = None,
    results_filename: str | None = None,
    models_dir: str | Path | None = None,
    classifier_names: list[str] | None = None,
    classifier_configs: dict[str, dict[str, Any]] | None = None,
    candidate_imbalance_strategies: tuple[str, ...] = (
        'synthetic', 'balanced', 'raw'),
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
    tie_break_rule: str = DEFAULT_TIE_BREAK_RULE,
    feature_matrix_hash: str | None = None,
    partition_hash: str | None = None,
    protocol_manifest_hash: str | None = None,
    preprocessing_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """
    Orchestrate the full within-condition experiment for one disease condition.

    Constructs the binary training pool (disease strides + Control Group A
    strides), runs nested LOSO-CV with GridSearchCV tuning for all 7
    classifiers, and saves results to a JSON file.

    The publication-grade v4 protocol evaluates the full candidate space
    (imbalance strategy × hyperparameter configuration) inside each outer
    training fold using grouped inner LOSO selection, then retunes one final
    full-source model on the entire source pool.
    """
    if control_subjects is not None:
        if control_a is not None and control_subjects != control_a:
            raise ValueError('control_a and control_subjects disagree')
        control_a = control_subjects

    if control_a is None:
        raise ValueError('control_a or control_subjects must be provided')

    selected_feature_cols = list(
        feature_cols) if feature_cols is not None else ALL_FEATURE_COLS
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir_path = Path(models_dir) if models_dir is not None else None
    if models_dir_path is not None:
        models_dir_path.mkdir(parents=True, exist_ok=True)
    selection_trace_dir = results_dir / 'selection_traces'
    selection_trace_dir.mkdir(parents=True, exist_ok=True)

    pool = df.filter(
        (pl.col('condition') == condition) |
        pl.col('subject_id').is_in(control_a)
    )
    pool_subjects = pool.n_unique('subject_id')
    pool_strides = pool.shape[0]

    X = pool.select(selected_feature_cols).to_numpy().astype(np.float64)
    y = pool['label'].to_numpy().astype(int)
    groups = pool['subject_id'].to_numpy()

    clf_results: dict[str, dict[str, Any]] = {}
    configs = classifier_configs or get_classifier_configs()
    if classifier_names is not None:
        classifier_name_set = set(classifier_names)
        configs = {
            clf_name: config
            for clf_name, config in configs.items()
            if clf_name in classifier_name_set
        }

    filename = results_filename or f'{condition}_results.json'
    partial_filename = (
        results_filename.replace('.json', '_partial.json')
        if results_filename else f'{condition}_results_v4_partial.json'
    )
    partial_path = results_dir / partial_filename

    for clf_name, config in configs.items():
        clf_candidate_strategies = candidate_strategies_for_classifier(
            clf_name,
            candidate_imbalance_strategies,
        )
        for arm in clf_candidate_strategies:
            if arm not in IMBALANCE_STRATEGIES:
                raise ValueError(
                    f"Unknown imbalance arm '{arm}'. Expected one of {IMBALANCE_STRATEGIES}."
                )
        clf = config['clf']
        param_grid = config['param_grid']
        result = _evaluate_within_condition_classifier(
            condition=condition,
            clf_name=clf_name,
            clf=clf,
            param_grid=param_grid,
            X=X,
            y=y,
            groups=groups,
            candidate_imbalance_strategies=clf_candidate_strategies,
            models_dir=models_dir_path,
            selection_trace_dir=selection_trace_dir,
            subject_aggregation_rule=subject_aggregation_rule,
            tie_break_rule=tie_break_rule,
        )

        clf_results[clf_name] = {
            'f1_macro': result['f1_macro'],
            'f1_macro_ci_lower': result['f1_macro_ci_lower'],
            'f1_macro_ci_upper': result['f1_macro_ci_upper'],
            'subject_resampled_stride_f1_ci_lower': result['subject_resampled_stride_f1_ci_lower'],
            'subject_resampled_stride_f1_ci_upper': result['subject_resampled_stride_f1_ci_upper'],
            'subject_primary_f1_ci_lower': result['subject_primary_f1_ci_lower'],
            'subject_primary_f1_ci_upper': result['subject_primary_f1_ci_upper'],
            'subject_primary_f1_macro': result['subject_primary_f1_macro'],
            'subject_aggregation_rule': result['subject_aggregation_rule'],
            'tie_break_rule': result['tie_break_rule'],
            'modal_params': result['modal_params'],
            'modal_frequency': result['modal_frequency'],
            'modal_strategy': result['modal_strategy'],
            'modal_strategy_frequency': result['modal_strategy_frequency'],
            'subject_metrics': result['subject_metrics'],
            'y_true': result['y_true'],
            'y_pred': result['y_pred'],
            'y_prob': result['y_prob'],
            'selected_resampling': _strategy_to_legacy_label(result['full_source_selected_imbalance_strategy']),
            'selected_imbalance_strategy': result['full_source_selected_imbalance_strategy'],
            'candidate_imbalance_strategies': result['candidate_imbalance_strategies'],
            'outer_fold_selection_trace': result['outer_fold_selection_trace'],
            'full_source_selected_params': result['full_source_selected_params'],
            'full_source_selected_imbalance_strategy': result['full_source_selected_imbalance_strategy'],
            'full_source_selection_subject_f1': result['full_source_selection_subject_f1'],
            'full_source_selection_stride_f1': result['full_source_selection_stride_f1'],
            'full_source_selection_subject_log_loss': result['full_source_selection_subject_log_loss'],
            'full_source_selection_trace': result['full_source_selection_trace'],
            'full_source_selection_trace_path': result['full_source_selection_trace_path'],
            'full_source_selection_trace_sha256': result['full_source_selection_trace_sha256'],
            'full_source_model_path': result['full_source_model_path'],
            'full_source_model_sha256': result['full_source_model_sha256'],
        }

        partial_output = _build_within_condition_output(
            condition=condition,
            pool_subjects=pool_subjects,
            pool_strides=pool_strides,
            selected_feature_cols=selected_feature_cols,
            feature_matrix_file=feature_matrix_file,
            feature_set_version=feature_set_version,
            normalization=normalization,
            models_dir=str(
                models_dir_path) if models_dir_path is not None else None,
            clf_results=clf_results,
        )
        partial_output['candidate_strategy_policy'] = (
            {
                clf_key: list(candidate_strategies_for_classifier(
                    clf_key,
                    candidate_imbalance_strategies,
                ))
                for clf_key in configs
            }
            if isinstance(candidate_imbalance_strategies, dict)
            else list(candidate_imbalance_strategies)
        )
        partial_output['subject_aggregation_rule'] = subject_aggregation_rule
        partial_output['subject_probability_threshold'] = DEFAULT_SUBJECT_PROBABILITY_THRESHOLD
        partial_output['tie_break_rule'] = tie_break_rule
        partial_output['feature_matrix_hash'] = feature_matrix_hash
        partial_output['partition_hash'] = partition_hash
        partial_output['protocol_manifest_hash'] = protocol_manifest_hash
        partial_output['preprocessing_manifest_hash'] = preprocessing_manifest_hash
        atomic_write_json(partial_path, partial_output)

    output = _build_within_condition_output(
        condition=condition,
        pool_subjects=pool_subjects,
        pool_strides=pool_strides,
        selected_feature_cols=selected_feature_cols,
        feature_matrix_file=feature_matrix_file,
        feature_set_version=feature_set_version,
        normalization=normalization,
        models_dir=str(
            models_dir_path) if models_dir_path is not None else None,
        clf_results=clf_results,
    )
    output['candidate_strategy_policy'] = (
        {
            clf_key: list(candidate_strategies_for_classifier(
                clf_key,
                candidate_imbalance_strategies,
            ))
            for clf_key in configs
        }
        if isinstance(candidate_imbalance_strategies, dict)
        else list(candidate_imbalance_strategies)
    )
    output['subject_aggregation_rule'] = subject_aggregation_rule
    output['subject_probability_threshold'] = DEFAULT_SUBJECT_PROBABILITY_THRESHOLD
    output['tie_break_rule'] = tie_break_rule
    output['feature_matrix_hash'] = feature_matrix_hash
    output['partition_hash'] = partition_hash
    output['protocol_manifest_hash'] = protocol_manifest_hash
    output['preprocessing_manifest_hash'] = preprocessing_manifest_hash

    out_path = results_dir / filename
    atomic_write_json(out_path, output)
    if partial_path.exists():
        partial_path.unlink()

    return output


# ── Zero-shot cross-condition transfer ───────────────────────────────────────

def _subject_level_permutation_test(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray | list[str],
    rng: np.random.Generator,
    n_permutations: int = 10_000,
    decision_scores: np.ndarray | None = None,
    subject_aggregation_rule: str = DEFAULT_SUBJECT_AGGREGATION_RULE,
) -> dict[str, Any]:
    """
    Subject-aware permutation test on aggregated subject predictions.
    """
    subj_true, subj_pred, _, subj_order = _subject_level_arrays(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        subject_ids=subject_ids,
        decision_scores=decision_scores,
        subject_aggregation_rule=subject_aggregation_rule,
    )
    observed = float(f1_score(subj_true, subj_pred, average='macro'))
    perm_scores = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_true = rng.permutation(subj_true)
        perm_scores[i] = float(f1_score(perm_true, subj_pred, average='macro'))
    exceedances = int(np.sum(perm_scores >= observed))
    p_value = (exceedances + 1.0) / (n_permutations + 1.0)
    return {
        'p_value': round(float(p_value), 6),
        'observed_subject_f1': round(observed, 6),
        'n_permutations': int(n_permutations),
        'monte_carlo_resolution': round(float(1.0 / (n_permutations + 1.0)), 6),
        'minimum_attainable_p_value': round(float(1.0 / (n_permutations + 1.0)), 6),
        'subject_ids': subj_order,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust a family of p-values."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (original_idx, p_value) in enumerate(indexed):
        factor = m - rank
        adjusted_p = min(1.0, factor * float(p_value))
        running_max = max(running_max, adjusted_p)
        adjusted[original_idx] = min(1.0, running_max)
    return adjusted


def annotate_cross_condition_results(
    cross_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Attach the restrained multiplicity policy metadata to combined Step 3 results.

    Direction-level primary-family adjustment is applied only when each direction
    provides an explicit direction-level primary p-value. Classifier-level
    permutation p-values remain supplementary regardless.
    """
    ordered_direction_keys = [
        'pd_to_hd',
        'hd_to_pd',
        'pd_to_als',
        'als_to_pd',
        'hd_to_als',
        'als_to_hd',
    ]
    available_p_values: list[float] = []
    available_keys: list[str] = []
    for direction_key in ordered_direction_keys:
        direction_result = cross_results.get(direction_key)
        if direction_result is None:
            continue
        p_value = direction_result.get(
            'direction_level_primary_subject_p_value')
        if p_value is not None:
            available_keys.append(direction_key)
            available_p_values.append(float(p_value))

    metadata = {
        'primary_family': {
            'description': 'Six direction-level subject-level matched-degradation claims',
            'correction': 'Holm',
            'classifier_level_claims': 'supplementary_only',
            'cross_direction_dependence_note': (
                'Control B subjects recur across directions, so dependence is expected.'
            ),
        },
    }
    if len(available_keys) == len(ordered_direction_keys):
        adjusted = _holm_adjust(available_p_values)
        for direction_key, adjusted_p in zip(available_keys, adjusted):
            cross_results[direction_key]['direction_level_primary_subject_p_value_holm'] = (
                round(float(adjusted_p), 6)
            )
        metadata['primary_family']['adjustment_status'] = 'complete'
    else:
        metadata['primary_family']['adjustment_status'] = 'pending_direction_level_primary_p_values'
        metadata['primary_family']['available_direction_count'] = len(
            available_keys)

    cross_results['__reporting_contract__'] = metadata
    return cross_results


def run_cross_condition(
    source_condition: str,
    target_condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    control_b: list[str],
    source_results: dict,
    results_dir: str | Path,
    models_dir: str | Path,
    feature_cols: list[str] | None = None,
    feature_matrix_file: str = DEFAULT_FEATURE_MATRIX_FILE,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    normalization: str = DEFAULT_NORMALIZATION,
    allow_refit: bool = False,
    protocol_manifest_hash: str | None = None,
    preprocessing_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """
    Zero-shot transfer of source-condition classifiers to a target condition.

    The v4 protocol consumes the full-source grouped-retuned source models from
    Step 2 and evaluates them on the disjoint target pool without any target
    tuning. Both stride-level and subject-level metrics are reported, with
    subject-aware permutation inference replacing stride-label shuffling.
    """
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    selected_feature_cols = list(
        feature_cols) if feature_cols is not None else ALL_FEATURE_COLS

    source_pool = df.filter(
        (pl.col('condition') == source_condition) |
        pl.col('subject_id').is_in(control_a)
    )
    source_pool_subjects = source_pool.n_unique('subject_id')
    source_pool_strides = source_pool.shape[0]

    assert source_pool_subjects == source_results['pool_subjects'], (
        f'Source pool subject count mismatch: feature matrix has '
        f'{source_pool_subjects}, JSON says {source_results["pool_subjects"]}. '
        f'Re-run within-condition training or verify the feature matrix.'
    )
    assert source_pool_strides == source_results['pool_strides'], (
        f'Source pool stride count mismatch: feature matrix has '
        f'{source_pool_strides}, JSON says {source_results["pool_strides"]}. '
        f'Re-run within-condition training or verify the feature matrix.'
    )

    source_feature_cols = source_results.get('feature_cols')
    if source_feature_cols is not None:
        assert source_feature_cols == selected_feature_cols, (
            f'Source feature columns do not match the current run for {source_condition}. '
            f'Expected {len(source_feature_cols)} features from the JSON and '
            f'{len(selected_feature_cols)} in this run.'
        )

    source_n_features = source_results.get('n_features')
    if source_n_features is not None:
        assert int(source_n_features) == len(selected_feature_cols), (
            f'Source feature count mismatch: JSON says {source_n_features}, '
            f'current run uses {len(selected_feature_cols)}.'
        )

    X_source = source_pool.select(
        selected_feature_cols).to_numpy().astype(np.float64)
    y_source = source_pool['label'].to_numpy().astype(int)

    target_pool = df.filter(
        (pl.col('condition') == target_condition) |
        pl.col('subject_id').is_in(control_b)
    )
    target_pool_subjects = target_pool.n_unique('subject_id')
    target_pool_strides = target_pool.shape[0]

    X_target = target_pool.select(
        selected_feature_cols).to_numpy().astype(np.float64)
    y_target = target_pool['label'].to_numpy().astype(int)
    target_subject_ids = target_pool['subject_id'].to_numpy()

    assert len(np.unique(y_target)) == 2, (
        f'Target pool ({target_condition} + Control B) contains only one class. '
        f'Check control_partition.json and the feature matrix.'
    )

    clf_results: dict[str, dict[str, Any]] = {}
    direction_subject_deltas: dict[str, float] = {}
    direction_stride_deltas: dict[str, float] = {}

    source_best_subject_classifier = max(
        source_results['classifiers'].items(),
        key=lambda item: float(item[1].get(
            'subject_primary_f1_macro', item[1]['f1_macro'])),
    )[0]
    source_best_stride_classifier = max(
        source_results['classifiers'].items(),
        key=lambda item: float(item[1]['f1_macro']),
    )[0]

    for clf_name, config in get_classifier_configs().items():
        clf_instance = config['clf']
        clf_source = source_results['classifiers'][clf_name]
        subject_aggregation_rule = clf_source.get(
            'subject_aggregation_rule',
            source_results.get('subject_aggregation_rule',
                               DEFAULT_SUBJECT_AGGREGATION_RULE),
        )
        tie_break_rule = clf_source.get(
            'tie_break_rule',
            source_results.get('tie_break_rule', DEFAULT_TIE_BREAK_RULE),
        )
        full_source_params = clf_source.get(
            'full_source_selected_params',
            clf_source['modal_params'],
        )
        selected_imbalance_strategy = clf_source.get(
            'full_source_selected_imbalance_strategy',
            clf_source.get(
                'selected_imbalance_strategy',
                'synthetic' if clf_source.get(
                    'selected_resampling', 'smote') == 'smote' else 'balanced',
            ),
        )
        selected_resampling = _strategy_to_legacy_label(
            selected_imbalance_strategy)

        clf_variant = _configure_classifier_for_resampling(
            clf_name,
            clf_instance,
            imbalance_strategy=selected_imbalance_strategy,
        )
        pipeline = build_pipeline(
            clf_name,
            clf_variant,
            imbalance_strategy=selected_imbalance_strategy,
        )
        pipeline.set_params(**full_source_params)

        stored_model_path = clf_source.get('full_source_model_path')
        if stored_model_path:
            candidate_model_path = Path(stored_model_path)
            model_path = (
                candidate_model_path
                if candidate_model_path.is_absolute()
                else models_dir / candidate_model_path
            )
        else:
            model_path = models_dir / f'{source_condition}_{clf_name}.joblib'
        should_refit = True

        if model_path.exists():
            loaded_pipeline = joblib.load(model_path)
            loaded_has_smote = 'smote' in loaded_pipeline.named_steps
            expected_has_smote = selected_imbalance_strategy == 'synthetic'
            params_match = all(
                loaded_pipeline.get_params().get(k) == v
                for k, v in full_source_params.items()
            )
            feature_count_match = (
                _fitted_pipeline_feature_count(
                    loaded_pipeline) == len(selected_feature_cols)
            )
            hash_match = True
            expected_hash = clf_source.get('full_source_model_sha256')
            if expected_hash is not None:
                hash_match = (_sha256_file(model_path) == expected_hash)
            source_protocol_hash = source_results.get('protocol_manifest_hash')
            source_preproc_hash = source_results.get(
                'preprocessing_manifest_hash')
            protocol_ok = (
                protocol_manifest_hash is None or source_protocol_hash is None
                or protocol_manifest_hash == source_protocol_hash
            )
            preprocessing_ok = (
                preprocessing_manifest_hash is None or source_preproc_hash is None
                or preprocessing_manifest_hash == source_preproc_hash
            )
            if (
                loaded_has_smote == expected_has_smote
                and params_match
                and feature_count_match
                and hash_match
                and protocol_ok
                and preprocessing_ok
            ):
                pipeline = loaded_pipeline
                should_refit = False

        if should_refit:
            if not allow_refit:
                raise FileNotFoundError(
                    'Authoritative cross-condition evaluation requires the exact full-source '
                    f'model for {source_condition}/{clf_name}. Missing or mismatched: {model_path}. '
                    'Rebuild Step 2 artifacts or explicitly set allow_refit=True only in a '
                    'non-authoritative development namespace.'
                )
            fit_kwargs = _get_fit_kwargs(
                clf_name,
                y_source,
                selected_imbalance_strategy,
            )
            pipeline.fit(X_source, y_source, **fit_kwargs)
            _save_pipeline_artifact(pipeline, model_path)

        y_pred = pipeline.predict(X_target)
        y_prob = pipeline.predict_proba(X_target)[:, 1]
        decision_scores = None
        if hasattr(pipeline, 'decision_function'):
            decision_scores = np.asarray(
                pipeline.decision_function(X_target), dtype=np.float64)
        y_true = y_target

        f1_val = round(float(f1_score(y_true, y_pred, average='macro')), 6)
        precision_val = round(float(precision_score(
            y_true, y_pred, average='macro', zero_division=0
        )), 6)
        recall_val = round(float(recall_score(
            y_true, y_pred, average='macro', zero_division=0
        )), 6)
        accuracy_val = round(float(accuracy_score(y_true, y_pred)), 6)

        rng = np.random.default_rng(
            42 + 100 * ['rf', 'knn', 'svm', 'dt',
                        'qda', 'xgb', 'lgbm'].index(clf_name)
        )
        n = len(y_true)
        boot_f1 = np.empty(1000)
        for i in range(1000):
            idx = rng.integers(0, n, size=n)
            boot_f1[i] = _fast_f1_binary(y_true[idx], y_pred[idx])
        ci_lower = round(float(np.percentile(boot_f1, 2.5)), 6)
        ci_upper = round(float(np.percentile(boot_f1, 97.5)), 6)

        subj_stride_ci_lower_raw, subj_stride_ci_upper_raw, n_rejected, rejection_rate = _subject_resampled_stride_bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            subject_ids=target_subject_ids,
            rng=rng,
            n_resamples=10_000,
        )
        subj_stride_ci_lower = round(subj_stride_ci_lower_raw, 6)
        subj_stride_ci_upper = round(subj_stride_ci_upper_raw, 6)

        if n_rejected > 0:
            print(
                f'    [subject bootstrap] {n_rejected} degenerate resamples rejected '
                f'({rejection_rate:.2%} rejection rate)',
                flush=True,
            )

        subject_metrics = _subject_level_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            subject_ids=target_subject_ids,
            decision_scores=decision_scores,
            subject_aggregation_rule=subject_aggregation_rule,
        )
        subj_true, subj_pred, subj_scores, subj_ids = _subject_level_arrays(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            subject_ids=target_subject_ids,
            decision_scores=decision_scores,
            subject_aggregation_rule=subject_aggregation_rule,
        )
        subject_ci_lower_raw, subject_ci_upper_raw, subject_n_rejected, subject_rejection_rate = (
            _subject_primary_bootstrap_ci(
                y_true_subject=subj_true,
                y_pred_subject=subj_pred,
                rng=np.random.default_rng(
                    6_000 + 100 * ['rf', 'knn', 'svm', 'dt',
                                   'qda', 'xgb', 'lgbm'].index(clf_name)
                ),
                n_resamples=10_000,
            )
        )
        permutation = _subject_level_permutation_test(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            subject_ids=target_subject_ids,
            decision_scores=decision_scores,
            subject_aggregation_rule=subject_aggregation_rule,
            rng=np.random.default_rng(
                5_000 + 100 * ['rf', 'knn', 'svm', 'dt',
                               'qda', 'xgb', 'lgbm'].index(clf_name)
            ),
            n_permutations=10_000,
        )
        disease_mask = y_true == 1
        control_mask = y_true == 0
        target_disease_recall_stride = round(
            float(np.mean(y_pred[disease_mask] == 1)), 6)
        control_specificity_stride = round(
            float(np.mean(y_pred[control_mask] == 0)), 6)
        control_fpr_stride = round(
            float(np.mean(y_pred[control_mask] == 1)), 6)
        disease_fnr_stride = round(
            float(np.mean(y_pred[disease_mask] == 0)), 6)
        subject_disease_mask = subj_true == 1
        subject_control_mask = subj_true == 0
        target_disease_recall_subject = round(
            float(np.mean(subj_pred[subject_disease_mask] == 1)), 6)
        control_specificity_subject = round(
            float(np.mean(subj_pred[subject_control_mask] == 0)), 6)
        control_fpr_subject = round(
            float(np.mean(subj_pred[subject_control_mask] == 1)), 6)
        disease_fnr_subject = round(
            float(np.mean(subj_pred[subject_disease_mask] == 0)), 6)
        within_stride_f1 = float(clf_source['f1_macro'])
        within_subject_f1 = float(clf_source.get(
            'subject_primary_f1_macro',
            clf_source.get('subject_metrics', {}).get(
                'f1_macro', clf_source['f1_macro']),
        ))
        delta_stride = round(within_stride_f1 - f1_val, 6)
        delta_subject = round(within_subject_f1 -
                              float(subject_metrics['f1_macro']), 6)
        direction_subject_deltas[clf_name] = delta_subject
        direction_stride_deltas[clf_name] = delta_stride
        subject_records = [
            {
                'subject_id': str(sid),
                'cohort': target_condition if int(yt) == 1 else 'control_B',
                'y_true': int(yt),
                'y_pred': int(yp),
                'subject_score': round(float(score), 6),
            }
            for sid, yt, yp, score in zip(subj_ids, subj_true, subj_pred, subj_scores)
        ]

        clf_results[clf_name] = {
            'f1_macro':               f1_val,
            'precision_macro':        precision_val,
            'recall_macro':           recall_val,
            'accuracy':               accuracy_val,
            'f1_macro_ci_lower':      ci_lower,
            'f1_macro_ci_upper':      ci_upper,
            'subject_resampled_stride_f1_ci_lower': subj_stride_ci_lower,
            'subject_resampled_stride_f1_ci_upper': subj_stride_ci_upper,
            'subject_primary_f1_ci_lower': round(subject_ci_lower_raw, 6),
            'subject_primary_f1_ci_upper': round(subject_ci_upper_raw, 6),
            'subject_primary_f1_macro': round(float(subject_metrics['f1_macro']), 6),
            'subject_aggregation_rule': subject_aggregation_rule,
            'tie_break_rule': tie_break_rule,
            'permutation_p_value':    permutation['p_value'],
            'permutation_test':       {
                'unit': 'subject',
                'n_permutations': permutation['n_permutations'],
                'monte_carlo_resolution': permutation['monte_carlo_resolution'],
                'minimum_attainable_p_value': permutation['minimum_attainable_p_value'],
                'observed_subject_f1': permutation['observed_subject_f1'],
            },
            'source_modal_params':    clf_source['modal_params'],
            'source_full_params':     full_source_params,
            'selected_resampling':    selected_resampling,
            'selected_imbalance_strategy': selected_imbalance_strategy,
            'subject_metrics':        subject_metrics,
            'subject_records':        subject_records,
            'subject_primary_bootstrap_rejections': subject_n_rejected,
            'subject_primary_bootstrap_rejection_rate': round(subject_rejection_rate, 6),
            'subject_resampled_stride_bootstrap_rejections': n_rejected,
            'subject_resampled_stride_bootstrap_rejection_rate': round(rejection_rate, 6),
            'within_subject_f1_macro': round(within_subject_f1, 6),
            'within_stride_f1_macro': round(within_stride_f1, 6),
            'delta_f1_subject':       delta_subject,
            'delta_f1_stride':        delta_stride,
            'target_disease_recall_stride':  target_disease_recall_stride,
            'target_disease_recall_subject': target_disease_recall_subject,
            'control_b_specificity_stride':  control_specificity_stride,
            'control_b_specificity_subject': control_specificity_subject,
            'control_b_false_positive_rate_stride': control_fpr_stride,
            'control_b_false_positive_rate_subject': control_fpr_subject,
            'target_disease_false_negative_rate_stride': disease_fnr_stride,
            'target_disease_false_negative_rate_subject': disease_fnr_subject,
            'y_true':                 y_true.tolist(),
            'y_pred':                 y_pred.tolist(),
            'y_prob':                 np.round(y_prob, 6).tolist(),
        }

        print(
            f'  {source_condition}->{target_condition}  {clf_name:<6}  '
            f'F1={f1_val:.4f}  '
            f'subj_F1={subject_metrics["f1_macro"]:.4f}  '
            f'stride_CI=[{ci_lower:.4f},{ci_upper:.4f}]  '
            f'subj_CI=[{subject_ci_lower_raw:.4f},{subject_ci_upper_raw:.4f}]  '
            f'p={permutation["p_value"]:.4f}',
            flush=True,
        )

    subject_delta_values = np.asarray(
        list(direction_subject_deltas.values()), dtype=np.float64)
    stride_delta_values = np.asarray(
        list(direction_stride_deltas.values()), dtype=np.float64)
    source_best_subject_delta = float(
        direction_subject_deltas[source_best_subject_classifier])
    source_best_stride_delta = float(
        direction_stride_deltas[source_best_stride_classifier])
    mean_cross_subject_f1 = float(np.mean([
        clf_results[clf_name]['subject_primary_f1_macro']
        for clf_name in clf_results
    ]))
    mean_cross_stride_f1 = float(np.mean([
        clf_results[clf_name]['f1_macro']
        for clf_name in clf_results
    ]))
    within_best_subject = float(source_results['classifiers'][source_best_subject_classifier].get(
        'subject_primary_f1_macro',
        source_results['classifiers'][source_best_subject_classifier]['f1_macro'],
    ))
    within_best_stride = float(
        source_results['classifiers'][source_best_stride_classifier]['f1_macro'])

    return {
        'source_condition':     source_condition,
        'target_condition':     target_condition,
        'source_pool_subjects': source_pool_subjects,
        'source_pool_strides':  source_pool_strides,
        'target_pool_subjects': target_pool_subjects,
        'target_pool_strides':  target_pool_strides,
        'target_subject_ids':   target_subject_ids.tolist(),
        'feature_cols':         selected_feature_cols,
        'n_features':           len(selected_feature_cols),
        'feature_matrix_file':  feature_matrix_file,
        'feature_set_version':  feature_set_version,
        'normalization':        normalization,
        'subject_aggregation_rule': source_results.get(
            'subject_aggregation_rule',
            DEFAULT_SUBJECT_AGGREGATION_RULE,
        ),
        'tie_break_rule': source_results.get('tie_break_rule', DEFAULT_TIE_BREAK_RULE),
        'protocol_manifest_hash': protocol_manifest_hash or source_results.get('protocol_manifest_hash'),
        'preprocessing_manifest_hash': (
            preprocessing_manifest_hash or source_results.get(
                'preprocessing_manifest_hash')
        ),
        'primary_endpoint': 'subject_primary_f1_macro',
        'secondary_endpoint': 'stride_f1_macro',
        'sensitivity_endpoint': 'subject_resampled_stride_f1_macro',
        'mean_matched_degradation_subject': round(float(np.mean(subject_delta_values)), 6),
        'median_matched_degradation_subject': round(float(np.median(subject_delta_values)), 6),
        'mean_matched_degradation_stride': round(float(np.mean(stride_delta_values)), 6),
        'median_matched_degradation_stride': round(float(np.median(stride_delta_values)), 6),
        'source_best_subject_classifier': source_best_subject_classifier,
        'source_best_stride_classifier': source_best_stride_classifier,
        'source_best_same_classifier_degradation_subject': round(source_best_subject_delta, 6),
        'source_best_same_classifier_degradation_stride': round(source_best_stride_delta, 6),
        'legacy_within_best_minus_mean_cross_subject': round(within_best_subject - mean_cross_subject_f1, 6),
        'legacy_within_best_minus_mean_cross_stride': round(within_best_stride - mean_cross_stride_f1, 6),
        'cohort_breakdown_subject_level': {
            target_condition: int(target_pool.filter(pl.col('label') == 1).n_unique('subject_id')),
            'control_B': int(target_pool.filter(pl.col('label') == 0).n_unique('subject_id')),
        },
        'classifiers':          clf_results,
    }
