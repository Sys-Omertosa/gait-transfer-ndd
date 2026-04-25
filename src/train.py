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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import ALL_FEATURE_COLS

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(
    'ignore',
    message='.*covariance matrix.*not full rank.*',
    category=UserWarning,
)

DEFAULT_FEATURE_MATRIX_FILE = 'v2/gait_features_v2.csv'
DEFAULT_FEATURE_SET_VERSION = 'v2'
DEFAULT_NORMALIZATION = 'none'

# ── Shared helpers ───────────────────────────────────────────────────────────


def _params_to_key(d: dict) -> tuple:
    """Convert a params dict to a hashable tuple for counting and comparison."""
    return tuple(sorted(d.items()))


def _fast_f1_binary(yt: np.ndarray, yp: np.ndarray) -> float:
    """Fast F1 macro for binary {0,1} classification. Avoids sklearn overhead in bootstrap loops."""
    tp1 = int(np.sum((yt == 1) & (yp == 1)))
    fp1 = int(np.sum((yt == 0) & (yp == 1)))
    fn1 = int(np.sum((yt == 1) & (yp == 0)))
    tp0 = int(np.sum((yt == 0) & (yp == 0)))
    fp0 = int(np.sum((yt == 1) & (yp == 0)))
    fn0 = int(np.sum((yt == 0) & (yp == 1)))
    f1_1 = (2 * tp1) / (2 * tp1 + fp1 + fn1) if (2 * tp1 + fp1 + fn1) > 0 else 0.0
    f1_0 = (2 * tp0) / (2 * tp0 + fp0 + fn0) if (2 * tp0 + fp0 + fn0) > 0 else 0.0
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


def _configure_classifier_for_resampling(
    classifier_name: str,
    clf: Any,
    use_smote: bool,
) -> Any:
    """
    Clone a classifier and apply any resampling-specific fixed parameters.

    Most classifiers already carry their appropriate class balancing in the
    base config. LightGBM receives class_weight='balanced' only for the
    no-SMOTE ablation so the minority control class is upweighted without
    generating synthetic control strides. XGBoost uses balanced sample weights
    at fit time because the installed sklearn wrapper does not expose
    class_weight.
    """
    configured = clone(clf)
    name = classifier_name.lower()

    if not use_smote and name == 'lgbm':
        configured.set_params(class_weight='balanced')

    return configured


def _get_fit_kwargs(
    classifier_name: str,
    y_fit: np.ndarray,
    use_smote: bool,
) -> dict[str, Any]:
    """
    Build fit kwargs for classifiers that need per-fit balancing metadata.

    XGBoost does not expose sklearn-style class_weight in this environment, so
    the no-SMOTE ablation uses balanced per-sample weights as the closest
    equivalent. In the current within-condition pools this upweights the
    minority control class rather than the disease class. GridSearchCV will
    subset these weights correctly inside the inner LOSO folds because the
    array length matches the outer training fold.
    """
    if not use_smote and classifier_name.lower() == 'xgb':
        return {'clf__sample_weight': _balanced_sample_weight(y_fit)}
    return {}


def _subject_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray | list[str],
    rng: np.random.Generator,
    n_resamples: int = 10_000,
) -> tuple[float, float, int, float]:
    """
    Subject-level bootstrap CI for macro-F1 using binary fast-path scoring.

    Subjects are resampled with replacement, then all strides belonging to the
    chosen subjects are concatenated in the sampled order. Degenerate resamples
    containing only one class are rejected because macro-F1 would be undefined.
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
        use_smote: Whether to include the SMOTE step inside the pipeline for
                   minority-control augmentation on the training folds.

    Returns:
        ImbPipeline with steps appropriate for the given classifier.
    """
    name = classifier_name.lower()
    steps: list[tuple[str, Any]] = []

    if name in _SCALE_REQUIRED:
        steps.append(('scaler', RobustScaler()))
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('clf', clf))

    return ImbPipeline(steps)


def run_nested_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: ImbPipeline,
    param_grid: dict[str, list],
    classifier_name: str,
    use_smote: bool = True,
) -> dict[str, Any]:
    """
    Outer LOSO-CV loop with inner GridSearchCV on the training fold only.

    For each outer fold (one held-out subject):
      1. Split pool into (n-1)-subject training set and 1-subject test set.
      2. Fit a fresh GridSearchCV with LeaveOneGroupOut on the training set,
         scoring by f1_macro. The inner splitter sees only training-fold
         subjects -- the held-out test subject is never visible during tuning.
      3. Predict on the held-out subject with the best estimator.
      4. Accumulate predictions, subject IDs, and best params.

    Args:
        X: Feature matrix, shape (n_strides, n_features), float64.
        y: Binary label vector, shape (n_strides,), int.
        groups: Subject-ID array, shape (n_strides,), str or object.
        pipeline: ImbPipeline produced by build_pipeline().
        param_grid: Dict mapping pipeline step param names to value lists.
        classifier_name: Short classifier name used to derive fit kwargs.
        use_smote: Whether the pipeline includes SMOTE.

    Returns:
        Dict with aggregate F1, concatenated labels/predictions, concatenated
        test subject IDs, and per-fold modal-selection metadata.
    """
    outer_loso = LeaveOneGroupOut()

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    subject_ids_all: list[np.ndarray] = []
    fold_params: list[dict] = []
    fold_best_scores: list[float] = []

    for train_idx, test_idx in outer_loso.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        # Inner GridSearchCV uses a fresh LOSO splitter restricted to the
        # training fold. This ensures the held-out test subject's strides
        # never influence hyperparameter selection.
        inner_loso = LeaveOneGroupOut()
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_loso,
            scoring='f1_macro',
            n_jobs=-1,
            refit=True,
        )
        fit_kwargs = _get_fit_kwargs(classifier_name, y_train, use_smote)
        grid.fit(X_train, y_train, groups=groups_train, **fit_kwargs)

        y_pred = grid.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        subject_ids_all.append(groups_test)
        fold_params.append(grid.best_params_)
        fold_best_scores.append(float(grid.best_score_))

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    subject_ids_concat = np.concatenate(subject_ids_all)

    return {
        'f1_macro':         f1_score(y_true_concat, y_pred_concat, average='macro'),
        'y_true_all':       y_true_concat,
        'y_pred_all':       y_pred_concat,
        'subject_ids_all':  subject_ids_concat,
        'fold_params':      fold_params,
        'fold_best_scores': fold_best_scores,
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
                class_weight='balanced',
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
                class_weight='balanced',
                random_state=42,
            ),
            'param_grid': {
                'clf__C':     [0.1, 1, 10, 100],
                'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            },
        },
        'dt': {
            'clf': DecisionTreeClassifier(
                class_weight='balanced',
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


def _evaluate_within_condition_classifier(
    clf_name: str,
    clf: Any,
    param_grid: dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_smote: bool,
) -> dict[str, Any]:
    """
    Run one within-condition classifier evaluation variant and package its outputs.

    The returned dict is JSON-ready and contains only the fields needed to
    select the authoritative winner between the SMOTE and no-SMOTE variants.
    When use_smote=True, the oversampled class is healthy control rather than
    disease because Control Group A is the minority side of every source pool.
    """
    clf_variant = _configure_classifier_for_resampling(clf_name, clf, use_smote)
    pipeline = build_pipeline(clf_name, clf_variant, use_smote=use_smote)
    variant_label = 'smote' if use_smote else 'no_smote'

    t_start = time.time()
    start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'  {clf_name:<6} [{variant_label}] started: {start_ts}', flush=True)

    loso_out = run_nested_loso(
        X,
        y,
        groups,
        pipeline,
        param_grid,
        classifier_name=clf_name,
        use_smote=use_smote,
    )

    elapsed = time.time() - t_start
    end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    modal = get_modal_params(
        loso_out['fold_params'],
        loso_out['fold_best_scores'],
    )

    modal_key = _params_to_key(modal)
    modal_frequency = sum(
        1 for p in loso_out['fold_params'] if _params_to_key(p) == modal_key
    )

    f1_stored = round(float(loso_out['f1_macro']), 6)
    y_true = loso_out['y_true_all']
    y_pred = loso_out['y_pred_all']

    # Verify the stored lists reproduce the stored F1 exactly.
    assert abs(f1_score(y_true, y_pred, average='macro') - f1_stored) < 1e-6, (
        f'{clf_name}: F1 mismatch between stored value and prediction lists'
    )

    ci_rng = np.random.default_rng(42)
    ci_lower, ci_upper, n_rejected, rejection_rate = _subject_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        subject_ids=loso_out['subject_ids_all'],
        rng=ci_rng,
        n_resamples=10_000,
    )

    if n_rejected > 0:
        print(
            f'    [{clf_name} {variant_label}] subject bootstrap rejected '
            f'{n_rejected} degenerate resamples ({rejection_rate:.2%})',
            flush=True,
        )

    print(
        f'  {clf_name:<6} [{variant_label}] F1={f1_stored:.4f} '
        f'CI=[{ci_lower:.4f},{ci_upper:.4f}] modal={modal} '
        f'start={start_ts} end={end_ts} ({elapsed:.0f}s)',
        flush=True,
    )

    return {
        'f1_macro': f1_stored,
        'f1_macro_ci_lower': round(ci_lower, 6),
        'f1_macro_ci_upper': round(ci_upper, 6),
        'modal_params': modal,
        'modal_frequency': modal_frequency,
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
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
    classifier_names: list[str] | None = None,
    classifier_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Orchestrate the full within-condition experiment for one disease condition.

    Constructs the binary training pool (disease strides + Control Group A
    strides), runs nested LOSO-CV with GridSearchCV tuning for all 7
    classifiers, and saves results to a JSON file.

    PD, HD, and ALS additionally run a no-SMOTE ablation after the main SMOTE
    pass. Because the healthy controls are the minority class in all three
    source pools, SMOTE augments synthetic control strides rather than disease
    strides. The authoritative stored outputs are whichever variant achieves
    the higher macro-F1, while the JSON also records both SMOTE and no-SMOTE
    scores so the ablation remains explicit and backward-compatible.

    Current interpretation of the v2 results: HD universally prefers no-SMOTE
    because the 8 real Control Group A subjects already provide sufficient
    boundary definition against HD's distinctly choreiform gait, so synthetic
    control interpolation adds boundary noise; ALS shows modest SMOTE benefit
    because its 1.17:1 source pool is closest to balanced; PD was initially
    omitted from the ablation under an incorrect disease-minority assumption
    and is now evaluated with the same source-side sensitivity logic.
    """
    if control_subjects is not None:
        if control_a is not None and control_subjects != control_a:
            raise ValueError('control_a and control_subjects disagree')
        control_a = control_subjects

    if control_a is None:
        raise ValueError('control_a or control_subjects must be provided')

    selected_feature_cols = list(feature_cols) if feature_cols is not None else ALL_FEATURE_COLS
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    run_no_smote_ablation = condition in {'pd', 'hd', 'als'}
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
        if results_filename else f'{condition}_results_v2_partial.json'
    )
    partial_path = results_dir / partial_filename

    for clf_name, config in configs.items():
        clf = config['clf']
        param_grid = config['param_grid']
        smote_result = _evaluate_within_condition_classifier(
            clf_name=clf_name,
            clf=clf,
            param_grid=param_grid,
            X=X,
            y=y,
            groups=groups,
            use_smote=True,
        )

        no_smote_result: dict[str, Any] | None = None
        if run_no_smote_ablation:
            no_smote_result = _evaluate_within_condition_classifier(
                clf_name=clf_name,
                clf=clf,
                param_grid=param_grid,
                X=X,
                y=y,
                groups=groups,
                use_smote=False,
            )

        selected_resampling = 'smote'
        winner = smote_result
        if (
            no_smote_result is not None
            and no_smote_result['f1_macro'] > smote_result['f1_macro']
        ):
            selected_resampling = 'no_smote'
            winner = no_smote_result

        if selected_resampling == 'no_smote':
            assert no_smote_result is not None, (
                f'{clf_name}: selected_resampling=no_smote but no ablation result exists'
            )

        clf_results[clf_name] = {
            'f1_macro': winner['f1_macro'],
            'f1_macro_ci_lower': winner['f1_macro_ci_lower'],
            'f1_macro_ci_upper': winner['f1_macro_ci_upper'],
            'modal_params': winner['modal_params'],
            'modal_frequency': winner['modal_frequency'],
            'y_true': winner['y_true'],
            'y_pred': winner['y_pred'],
            'f1_macro_smote': smote_result['f1_macro'],
            'f1_macro_no_smote': (
                no_smote_result['f1_macro'] if no_smote_result is not None else None
            ),
            'selected_resampling': selected_resampling,
        }

        partial_output = _build_within_condition_output(
            condition=condition,
            pool_subjects=pool_subjects,
            pool_strides=pool_strides,
            selected_feature_cols=selected_feature_cols,
            feature_matrix_file=feature_matrix_file,
            feature_set_version=feature_set_version,
            normalization=normalization,
            clf_results=clf_results,
        )
        with open(partial_path, 'w') as f:
            json.dump(partial_output, f, indent=2)

    output = _build_within_condition_output(
        condition=condition,
        pool_subjects=pool_subjects,
        pool_strides=pool_strides,
        selected_feature_cols=selected_feature_cols,
        feature_matrix_file=feature_matrix_file,
        feature_set_version=feature_set_version,
        normalization=normalization,
        clf_results=clf_results,
    )

    out_path = results_dir / filename
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    return output


# ── Zero-shot cross-condition transfer ───────────────────────────────────────

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
) -> dict[str, Any]:
    """
    Zero-shot transfer of source-condition classifiers to a target condition.

    The source model is trained once on the full source pool (source condition
    strides + Control Group A strides) using the modal hyperparameters
    identified during within-condition LOSO-CV. No hyperparameter tuning is
    performed on target-condition data -- none is available under the zero-shot
    protocol.
    """
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    selected_feature_cols = list(feature_cols) if feature_cols is not None else ALL_FEATURE_COLS

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

    X_source = source_pool.select(selected_feature_cols).to_numpy().astype(np.float64)
    y_source = source_pool['label'].to_numpy().astype(int)

    target_pool = df.filter(
        (pl.col('condition') == target_condition) |
        pl.col('subject_id').is_in(control_b)
    )
    target_pool_subjects = target_pool.n_unique('subject_id')
    target_pool_strides = target_pool.shape[0]

    X_target = target_pool.select(selected_feature_cols).to_numpy().astype(np.float64)
    y_target = target_pool['label'].to_numpy().astype(int)
    target_subject_ids = target_pool['subject_id'].to_list()

    assert len(np.unique(y_target)) == 2, (
        f'Target pool ({target_condition} + Control B) contains only one class. '
        f'Check control_partition.json and the feature matrix.'
    )

    rng = np.random.default_rng(42)
    clf_results: dict[str, dict[str, Any]] = {}

    for clf_name, config in get_classifier_configs().items():
        clf_instance = config['clf']
        clf_source = source_results['classifiers'][clf_name]
        modal_params = clf_source['modal_params']
        selected_resampling = clf_source.get('selected_resampling', 'smote')
        use_smote = selected_resampling != 'no_smote'

        clf_variant = _configure_classifier_for_resampling(
            clf_name,
            clf_instance,
            use_smote=use_smote,
        )
        pipeline = build_pipeline(clf_name, clf_variant, use_smote=use_smote)
        pipeline.set_params(**modal_params)

        model_path = models_dir / f'{source_condition}_{clf_name}.joblib'
        should_refit = True

        if model_path.exists():
            loaded_pipeline = joblib.load(model_path)
            loaded_has_smote = 'smote' in loaded_pipeline.named_steps
            params_match = all(
                loaded_pipeline.get_params().get(k) == v
                for k, v in modal_params.items()
            )
            feature_count_match = (
                _fitted_pipeline_feature_count(loaded_pipeline) == len(selected_feature_cols)
            )
            if loaded_has_smote == use_smote and params_match and feature_count_match:
                pipeline = loaded_pipeline
                should_refit = False

        if should_refit:
            fit_kwargs = _get_fit_kwargs(clf_name, y_source, use_smote)
            pipeline.fit(X_source, y_source, **fit_kwargs)
            joblib.dump(pipeline, model_path)

        y_pred = pipeline.predict(X_target)
        y_true = y_target

        f1_val = round(float(f1_score(y_true, y_pred, average='macro')), 6)
        precision_val = round(float(precision_score(
            y_true, y_pred, average='macro', zero_division=0
        )), 6)
        recall_val = round(float(recall_score(
            y_true, y_pred, average='macro', zero_division=0
        )), 6)
        accuracy_val = round(float(accuracy_score(y_true, y_pred)), 6)

        n = len(y_true)
        boot_f1 = np.empty(1000)
        for i in range(1000):
            idx = rng.integers(0, n, size=n)
            boot_f1[i] = _fast_f1_binary(y_true[idx], y_pred[idx])
        ci_lower = round(float(np.percentile(boot_f1, 2.5)), 6)
        ci_upper = round(float(np.percentile(boot_f1, 97.5)), 6)

        perm_f1 = np.empty(1000)
        for i in range(1000):
            y_shuffled = rng.permutation(y_true)
            perm_f1[i] = f1_score(y_shuffled, y_pred, average='macro')
        p_value = round(float(np.mean(perm_f1 >= f1_val)), 4)

        subj_ci_lower_raw, subj_ci_upper_raw, n_rejected, rejection_rate = _subject_bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            subject_ids=target_subject_ids,
            rng=rng,
            n_resamples=10_000,
        )
        subj_ci_lower = round(subj_ci_lower_raw, 6)
        subj_ci_upper = round(subj_ci_upper_raw, 6)

        if n_rejected > 0:
            print(
                f'    [subject bootstrap] {n_rejected} degenerate resamples rejected '
                f'({rejection_rate:.2%} rejection rate)',
                flush=True,
            )

        clf_results[clf_name] = {
            'f1_macro':               f1_val,
            'precision_macro':        precision_val,
            'recall_macro':           recall_val,
            'accuracy':               accuracy_val,
            'f1_macro_ci_lower':      ci_lower,
            'f1_macro_ci_upper':      ci_upper,
            'f1_macro_subj_ci_lower': subj_ci_lower,
            'f1_macro_subj_ci_upper': subj_ci_upper,
            'permutation_p_value':    p_value,
            'source_modal_params':    modal_params,
            'selected_resampling':    selected_resampling,
            'y_true':                 y_true.tolist(),
            'y_pred':                 y_pred.tolist(),
        }

        print(
            f'  {source_condition}->{target_condition}  {clf_name:<6}  '
            f'F1={f1_val:.4f}  '
            f'stride_CI=[{ci_lower:.4f},{ci_upper:.4f}]  '
            f'subj_CI=[{subj_ci_lower:.4f},{subj_ci_upper:.4f}]  '
            f'p={p_value:.4f}',
            flush=True,
        )

    return {
        'source_condition':     source_condition,
        'target_condition':     target_condition,
        'source_pool_subjects': source_pool_subjects,
        'source_pool_strides':  source_pool_strides,
        'target_pool_subjects': target_pool_subjects,
        'target_pool_strides':  target_pool_strides,
        'target_subject_ids':   target_subject_ids,
        'feature_cols':         selected_feature_cols,
        'n_features':           len(selected_feature_cols),
        'feature_matrix_file':  feature_matrix_file,
        'feature_set_version':  feature_set_version,
        'normalization':        normalization,
        'classifiers':          clf_results,
    }
