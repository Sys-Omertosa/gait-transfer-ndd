"""
Step 5: noise robustness and sensitivity analysis.

Shared pipeline helpers mirror train.py intentionally, and this module now
imports the train-side builder/configuration utilities directly so Step 5 uses
the exact same model-construction semantics as the main benchmark line.

- Gaussian noise on evaluation inputs only.
- Per-feature sensitivity via column shuffle on evaluation inputs.
- Per-subject accuracy from stored LOSO / transfer predictions.
- Structured corruption benchmark (engineered masking / gain-bias drift /
  Gaussian feature jitter / row dropout / label corruption).
- Split-conformal prediction diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import ALL_FEATURE_COLS
from train import (
    IMBALANCE_STRATEGIES,
    _configure_classifier_for_resampling as train_configure_classifier_for_resampling,
    _fitted_pipeline_feature_count as train_fitted_pipeline_feature_count,
    _get_fit_kwargs as train_get_fit_kwargs,
    build_pipeline as train_build_pipeline,
    get_classifier_configs,
)
from v4_provenance import atomic_write_json, sha256_file

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CLF_ORDER: tuple[str, ...] = ("rf", "knn", "svm", "dt", "qda", "xgb", "lgbm")

# PARAMETER REVIEW
# - SIGMA_LEVELS spans 0.00 to 0.50 of the source-pool feature standard
#   deviation, which covers the clinically relevant mild-to-severe sensor-noise
#   range for inertial / wearable gait features without collapsing the sweep
#   into redundant near-zero settings.
# - N_NOISE_REPEATS=30 is intentionally conservative. Lower repeat counts may
#   converge in some settings, but 30 keeps mean and spread estimates stable
#   enough for paper-facing comparisons, so it is retained unchanged.
# - CORRUPTION_LEVELS cover the main engineered feature-space perturbations
#   already in scope here: masking, gain/bias drift, additive jitter, row
#   dropout, and annotation noise. Other corruptions could be studied later,
#   but none of the current types are redundant.
# - CONFORMAL_ALPHAS=(0.05, 0.10, 0.20) correspond to 95%, 90%, and 80%
#   nominal coverage, which are standard operating points for uncertainty
#   reporting in clinically cautious binary screening settings.
# - AGG_N_STRIDES=(1, 3, 5, 10, 25, 50) spans single-stride use, short bedside
#   windows, and longer hallway / lab aggregates, so it is broad enough for
#   clinically meaningful stride aggregation analysis.
SIGMA_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
N_NOISE_REPEATS: int = 30

_SCALE_REQUIRED = frozenset({"svm", "knn"})
_LEFT_COL_HINTS = ("left_stride_s", "left_swing_s",
                   "left_swing_pct", "left_stance_s")
_RIGHT_COL_HINTS = ("right_stride_s", "right_swing_s",
                    "right_swing_pct", "right_stance_s")

_COND_IDX = {"pd": 0, "hd": 1, "als": 2}

DIRECTIONS_ORDER: tuple[tuple[str, str], ...] = (
    ("pd", "hd"),
    ("hd", "pd"),
    ("pd", "als"),
    ("als", "pd"),
    ("hd", "als"),
    ("als", "hd"),
)

CORRUPTION_LEVELS: dict[str, dict[str, tuple[float, ...] | float]] = {
    "engineered_feature_masking": {"light": (0.05,), "medium": (0.15,), "heavy": (0.30,)},
    "engineered_feature_gain_bias_drift": {
        "light": (0.02, 0.05),
        "medium": (0.05, 0.10),
        "heavy": (0.10, 0.20),
    },
    "gaussian_feature_space_jitter": {"light": (0.10,), "medium": (0.25,), "heavy": (0.50,)},
    "evaluation_row_dropout": {"light": (0.10,), "medium": (0.30,), "heavy": (0.50,)},
    "benchmark_label_corruption": {"light": (0.05,), "medium": (0.10,), "heavy": (0.20,)},
}
CONFORMAL_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)
AGG_N_STRIDES: tuple[int, ...] = (1, 3, 5, 10, 25, 50)


def _feature_family(feature_name: str) -> str:
    if feature_name in {"cv_stride", "cv_swing"}:
        return "variability"
    if feature_name == "dfa_alpha_stride":
        return "fractal"
    if "asymmetry" in feature_name:
        return "asymmetry"
    if feature_name.endswith("_pct"):
        return "phase_percentage"
    return "raw_timing"


def _aggregate_feature_family_drops(feature_drops: dict[str, float]) -> dict[str, float]:
    family_totals: dict[str, float] = {}
    for feature_name, value in feature_drops.items():
        family_totals.setdefault(_feature_family(feature_name), 0.0)
        family_totals[_feature_family(feature_name)] += float(value)
    return {family: round(total, 6) for family, total in sorted(family_totals.items())}


def _feature_family_indices(feature_cols: list[str]) -> dict[str, np.ndarray]:
    """Map each feature family to the active matrix columns it spans."""
    family_map: dict[str, list[int]] = {}
    for idx, feature_name in enumerate(feature_cols):
        family_map.setdefault(_feature_family(feature_name), []).append(idx)
    return {
        family: np.asarray(indices, dtype=int)
        for family, indices in sorted(family_map.items())
    }


def direction_key_to_pair(key: str) -> tuple[str, str]:
    if "_to_" not in key:
        raise ValueError(f"Invalid direction key: {key}")
    a, b = key.split("_to_", 1)
    return a, b


def _direction_index(source: str, target: str) -> int:
    return DIRECTIONS_ORDER.index((source, target))


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2)
    if np.any(counts == 0):
        return np.ones_like(y, dtype=float)
    total = float(len(y))
    w0 = total / (2.0 * counts[0])
    w1 = total / (2.0 * counts[1])
    return np.where(y == 0, w0, w1).astype(float)


def _normalize_imbalance_strategy(
    use_smote: bool | None = None,
    imbalance_strategy: str | None = None,
) -> str:
    """Bridge legacy boolean SMOTE flags to the explicit v4 strategy labels."""
    if imbalance_strategy is None:
        if use_smote is None:
            imbalance_strategy = "synthetic"
        else:
            imbalance_strategy = "synthetic" if use_smote else "balanced"
    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy '{imbalance_strategy}'. "
            f"Expected one of {IMBALANCE_STRATEGIES}."
        )
    return imbalance_strategy


def _configure_classifier_for_resampling(
    classifier_name: str,
    clf: Any,
    use_smote: bool | None = None,
    imbalance_strategy: str | None = None,
) -> Any:
    strategy = _normalize_imbalance_strategy(use_smote, imbalance_strategy)
    return train_configure_classifier_for_resampling(
        classifier_name,
        clf,
        strategy,
    )


def _get_fit_kwargs(
    classifier_name: str,
    y_fit: np.ndarray,
    use_smote: bool | None = None,
    imbalance_strategy: str | None = None,
) -> dict[str, Any]:
    strategy = _normalize_imbalance_strategy(use_smote, imbalance_strategy)
    return train_get_fit_kwargs(classifier_name, y_fit, strategy)


def build_pipeline(
    classifier_name: str,
    clf: Any,
    use_smote: bool = True,
    imbalance_strategy: str | None = None,
) -> ImbPipeline:
    strategy = _normalize_imbalance_strategy(use_smote, imbalance_strategy)
    return train_build_pipeline(
        classifier_name,
        clf,
        imbalance_strategy=strategy,
    )


def _fresh_classifier(clf_name: str) -> Any:
    """New classifier instance matching train.get_classifier_configs() defaults."""
    configs = get_classifier_configs()
    if clf_name not in configs:
        raise ValueError(f"Unknown classifier: {clf_name}")
    return clone(configs[clf_name]["clf"])


def inject_gaussian_noise(
    X: np.ndarray,
    sigma_frac: float,
    feature_std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma_frac <= 0:
        return X
    std = np.maximum(feature_std.astype(np.float64), 1e-12)
    scale = sigma_frac * std
    noise = rng.standard_normal(size=X.shape).astype(np.float64) * scale
    return X.astype(np.float64, copy=False) + noise


def _within_pool(df: pl.DataFrame, condition: str, control_a: list[str]) -> pl.DataFrame:
    return df.filter(
        (pl.col("condition") == condition) | pl.col(
            "subject_id").is_in(control_a)
    )


def _concat_loso_test_indices(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outer = LeaveOneGroupOut()
    parts: list[np.ndarray] = []
    for _, test_idx in outer.split(X, y, groups):
        parts.append(test_idx)
    concat_idx = np.concatenate(parts)
    return concat_idx, groups[concat_idx]


FittedFold = tuple[Any, np.ndarray]


def _fitted_pipeline_feature_count(pipeline: Any) -> int | None:
    return train_fitted_pipeline_feature_count(pipeline)


def _assert_pipeline_feature_count(pipeline: Any, expected: int) -> None:
    got = _fitted_pipeline_feature_count(pipeline)
    if got is None:
        return
    if got != expected:
        raise ValueError(
            f"Pipeline feature count mismatch: expected={expected}, got={got}")


def _sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    return sha256_file(path)


def loso_fit_all_folds_fixed(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str,
    modal_params: dict[str, Any],
    use_smote: bool | None = None,
    imbalance_strategy: str | None = None,
) -> list[FittedFold]:
    strategy = _normalize_imbalance_strategy(use_smote, imbalance_strategy)
    clf_template = _fresh_classifier(clf_name)
    outer_loso = LeaveOneGroupOut()
    fitted: list[FittedFold] = []
    for train_idx, test_idx in outer_loso.split(X, y, groups):
        clf_variant = _configure_classifier_for_resampling(
            clf_name,
            clf_template,
            imbalance_strategy=strategy,
        )
        base = build_pipeline(
            clf_name,
            clf_variant,
            imbalance_strategy=strategy,
        )
        base.set_params(**modal_params)
        pipe = clone(base)
        X_train, y_train = X[train_idx], y[train_idx]
        fit_kwargs = _get_fit_kwargs(
            clf_name,
            y_train,
            imbalance_strategy=strategy,
        )
        pipe.fit(X_train, y_train, **fit_kwargs)
        fitted.append((pipe, test_idx))
    return fitted


def loso_predict_from_fitted(
    fitted_folds: list[FittedFold],
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    sigma_frac: float,
    feature_std: np.ndarray,
    permute_col: int | None = None,
    permute_cols: np.ndarray | None = None,
) -> float:
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    for pipe, test_idx in fitted_folds:
        y_test = y[test_idx]
        Xt = np.array(X[test_idx], dtype=np.float64, copy=True)
        if permute_col is not None:
            col = Xt[:, permute_col].copy()
            rng.shuffle(col)
            Xt[:, permute_col] = col
        if permute_cols is not None and len(permute_cols) > 0:
            row_permutation = rng.permutation(len(Xt))
            Xt[:, permute_cols] = Xt[row_permutation][:, permute_cols]
        if sigma_frac > 0:
            Xt = inject_gaussian_noise(Xt, sigma_frac, feature_std, rng)
        y_pred = pipe.predict(Xt)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
    y_t = np.concatenate(y_true_all)
    y_p = np.concatenate(y_pred_all)
    return float(f1_score(y_t, y_p, average="macro"))


def fit_within_condition_folds(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    feature_cols: list[str] | None = None,
    *,
    allow_approximate_refit: bool = False,
    feature_matrix_hash: str | None = None,
    partition_hash: str | None = None,
    protocol_manifest_hash: str | None = None,
    preprocessing_manifest_hash: str | None = None,
) -> dict[str, list[FittedFold]]:
    """
    Fit LOSO folds for all classifiers for one within-condition pool once.

    The Step 5 within-condition analyses are deterministic for a fixed source
    pool, modal parameter set, and imbalance strategy, so the same fitted LOSO
    folds can be reused across noise, permutation, corruption, and conformal
    diagnostics without changing any scientific result.
    """
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()

    fitted: dict[str, list[FittedFold]] = {}
    for clf_name in CLF_ORDER:
        clf_res = within_results["classifiers"][clf_name]
        strategy = clf_res.get(
            "selected_imbalance_strategy",
            "synthetic" if clf_res.get("selected_resampling", "smote") == "smote"
            else "balanced",
        )
        outer_trace = clf_res.get("outer_fold_selection_trace", [])
        loaded_folds: list[FittedFold] = []
        can_replay_exact = bool(outer_trace)
        replay_failure_reason = 'missing outer_fold_selection_trace'

        if can_replay_exact:
            top_level_requirements = {
                'feature_matrix_hash': feature_matrix_hash,
                'partition_hash': partition_hash,
                'protocol_manifest_hash': protocol_manifest_hash,
                'preprocessing_manifest_hash': preprocessing_manifest_hash,
            }
            for field_name, expected_value in top_level_requirements.items():
                if expected_value is None:
                    continue
                stored_value = within_results.get(field_name)
                if stored_value is None:
                    can_replay_exact = False
                    replay_failure_reason = (
                        f'missing required within-results field {field_name!r}'
                    )
                    break
                if stored_value != expected_value:
                    can_replay_exact = False
                    replay_failure_reason = (
                        f'within-results field {field_name!r} does not match the '
                        'authoritative Step 5 prerequisites'
                    )
                    break

        if can_replay_exact:
            outer_loso = LeaveOneGroupOut()
            split_iter = list(outer_loso.split(X, y, groups))
            if len(split_iter) != len(outer_trace):
                can_replay_exact = False
                replay_failure_reason = 'outer trace length does not match current LOSO splits'
            else:
                for fold_detail, (_, test_idx) in zip(outer_trace, split_iter):
                    model_relpath = fold_detail.get("fold_model_relpath")
                    if not model_relpath:
                        can_replay_exact = False
                        replay_failure_reason = 'missing fold_model_relpath in outer trace'
                        break
                    model_path = Path(model_relpath)
                    if not model_path.is_absolute():
                        models_root = within_results.get("models_dir")
                        model_path = Path(
                            models_root) / model_relpath if models_root else model_path
                    if not model_path.exists():
                        can_replay_exact = False
                        replay_failure_reason = f'missing fold model artifact at {model_path}'
                        break
                    expected_subject = str(
                        fold_detail.get("held_out_subject_id"))
                    actual_subject = str(groups[test_idx][0])
                    if expected_subject != actual_subject:
                        can_replay_exact = False
                        replay_failure_reason = (
                            f'held-out subject mismatch for {condition}/{clf_name}: '
                            f'expected {expected_subject}, got {actual_subject}'
                        )
                        break
                    expected_hash = fold_detail.get("fold_model_sha256")
                    if expected_hash is None:
                        can_replay_exact = False
                        replay_failure_reason = 'missing fold_model_sha256 in outer trace'
                        break
                    if _sha256_file(model_path) != expected_hash:
                        can_replay_exact = False
                        replay_failure_reason = f'fold model hash mismatch at {model_path}'
                        break
                    selected_strategy = fold_detail.get(
                        "selected_imbalance_strategy")
                    if selected_strategy is None:
                        can_replay_exact = False
                        replay_failure_reason = (
                            'missing selected_imbalance_strategy in outer trace'
                        )
                        break
                    selected_params = fold_detail.get("selected_params")
                    if not isinstance(selected_params, dict):
                        can_replay_exact = False
                        replay_failure_reason = 'missing selected_params in outer trace'
                        break
                    loaded_pipeline = joblib.load(model_path)
                    _assert_pipeline_feature_count(
                        loaded_pipeline, len(sel_cols))
                    loaded_has_smote = 'smote' in loaded_pipeline.named_steps
                    expected_has_smote = selected_strategy == 'synthetic'
                    if loaded_has_smote != expected_has_smote:
                        can_replay_exact = False
                        replay_failure_reason = (
                            f'fold model SMOTE state does not match recorded '
                            f'selected_imbalance_strategy for {condition}/{clf_name}/{expected_subject}'
                        )
                        break
                    params_match = all(
                        loaded_pipeline.get_params().get(param_name) == param_value
                        for param_name, param_value in selected_params.items()
                    )
                    if not params_match:
                        can_replay_exact = False
                        replay_failure_reason = (
                            f'fold model params do not match recorded selected_params '
                            f'for {condition}/{clf_name}/{expected_subject}'
                        )
                        break
                    loaded_folds.append((loaded_pipeline, test_idx))

        if can_replay_exact and loaded_folds:
            fitted[clf_name] = loaded_folds
            continue

        if not allow_approximate_refit:
            raise FileNotFoundError(
                'Authoritative Step 5 replay requires exact saved outer-fold artifacts. '
                f'Unable to validate replay for {condition}/{clf_name}. '
                f'Reason: {replay_failure_reason}. '
                'Use allow_approximate_refit=True only in a non-authoritative diagnostic namespace.'
            )

        modal = clf_res.get(
            "full_source_selected_params",
            clf_res["modal_params"],
        )
        fitted[clf_name] = loso_fit_all_folds_fixed(
            X,
            y,
            groups,
            clf_name,
            modal,
            imbalance_strategy=strategy,
        )
    return fitted


def evaluate_noise_sweep_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    feature_cols: list[str] | None = None,
    prefit_folds: dict[str, list[FittedFold]] | None = None,
) -> dict[str, dict[str, list[float]]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    feature_std = X.std(axis=0, dtype=np.float64)
    cond_idx = _COND_IDX[condition]

    out: dict[str, dict[str, list[float]]] = {}
    for ci, clf_name in enumerate(CLF_ORDER):
        clf_res = within_results["classifiers"][clf_name]
        modal = clf_res["modal_params"]
        strategy = clf_res.get(
            "selected_imbalance_strategy",
            "synthetic" if clf_res.get("selected_resampling", "smote") == "smote"
            else "balanced"
        )
        if prefit_folds is not None:
            fitted_folds = prefit_folds[clf_name]
        else:
            fitted_folds = loso_fit_all_folds_fixed(
                X,
                y,
                groups,
                clf_name,
                modal,
                imbalance_strategy=strategy,
            )
        clf_out: dict[str, list[float]] = {}
        for si, sigma in enumerate(SIGMA_LEVELS):
            n_rep = 1 if sigma == 0.0 else N_NOISE_REPEATS
            reps: list[float] = []
            for r in range(n_rep):
                rng = np.random.default_rng(
                    42 + 1_000_000 * cond_idx + 10_000 * ci + 100 * si + r
                )
                f1v = loso_predict_from_fitted(
                    fitted_folds, X, y, rng, sigma, feature_std, permute_col=None
                )
                reps.append(round(float(f1v), 6))
            clf_out[str(sigma)] = reps
        out[clf_name] = clf_out
    return out


def evaluate_noise_sweep_cross(
    source_condition: str,
    target_condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    control_b: list[str],
    models_dir: str | Path,
    feature_cols: list[str] | None = None,
    loaded_models: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    models_dir = Path(models_dir)
    source_pool = df.filter(
        (pl.col("condition") == source_condition)
        | pl.col("subject_id").is_in(control_a)
    )
    X_source = source_pool.select(sel_cols).to_numpy().astype(np.float64)
    feature_std_source = X_source.std(axis=0, dtype=np.float64)

    target_pool = df.filter(
        (pl.col("condition") == target_condition)
        | pl.col("subject_id").is_in(control_b)
    )
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)
    feature_std_target = X_target.std(axis=0, dtype=np.float64)
    feature_std_pooled = np.vstack(
        [X_source, X_target]).std(axis=0, dtype=np.float64)

    dir_idx = _direction_index(source_condition, target_condition)
    scale_references = {
        "source_relative": feature_std_source,
        "target_relative": feature_std_target,
        "pooled_relative": feature_std_pooled,
    }
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for scale_name, feature_std in scale_references.items():
        scale_out: dict[str, dict[str, list[float]]] = {}
        for ci, clf_name in enumerate(CLF_ORDER):
            if loaded_models is not None:
                pipeline = loaded_models.get(clf_name)
                if pipeline is None:
                    raise FileNotFoundError(
                        f"Missing loaded model for {source_condition}/{clf_name} "
                        f"while evaluating {source_condition}->{target_condition} noise robustness."
                    )
            else:
                model_path = models_dir / \
                    f"{source_condition}_{clf_name}.joblib"
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Missing authoritative source model at {model_path} for "
                        f"{source_condition}->{target_condition} noise robustness."
                    )
                pipeline = joblib.load(model_path)
            _assert_pipeline_feature_count(pipeline, len(sel_cols))
            clf_out: dict[str, list[float]] = {}
            for si, sigma in enumerate(SIGMA_LEVELS):
                n_rep = 1 if sigma == 0.0 else N_NOISE_REPEATS
                reps: list[float] = []
                for r in range(n_rep):
                    rng = np.random.default_rng(
                        500_000 + 50_000 * dir_idx + 1_000 * ci + 100 * si + r
                    )
                    Xt = np.array(X_target, dtype=np.float64, copy=True)
                    if sigma > 0:
                        Xt = inject_gaussian_noise(Xt, sigma, feature_std, rng)
                    y_pred = pipeline.predict(Xt)
                    reps.append(
                        round(float(f1_score(y_target, y_pred, average="macro")), 6)
                    )
                clf_out[str(sigma)] = reps
            scale_out[clf_name] = clf_out
        out[scale_name] = scale_out
    return out


def permutation_importance_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    baseline_f1: dict[str, float],
    feature_cols: list[str] | None = None,
    prefit_folds: dict[str, list[FittedFold]] | None = None,
) -> dict[str, dict[str, float]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    feature_std = X.std(axis=0, dtype=np.float64)
    cond_idx = _COND_IDX[condition]
    family_indices = _feature_family_indices(sel_cols)

    out: dict[str, dict[str, float]] = {}
    for ci, clf_name in enumerate(CLF_ORDER):
        clf_res = within_results["classifiers"][clf_name]
        modal = clf_res["modal_params"]
        strategy = clf_res.get(
            "selected_imbalance_strategy",
            "synthetic" if clf_res.get("selected_resampling", "smote") == "smote"
            else "balanced"
        )
        base = baseline_f1[clf_name]
        if prefit_folds is not None:
            fitted_folds = prefit_folds[clf_name]
        else:
            fitted_folds = loso_fit_all_folds_fixed(
                X,
                y,
                groups,
                clf_name,
                modal,
                imbalance_strategy=strategy,
            )
        feat_drops: dict[str, float] = {}
        for j, fname in enumerate(sel_cols):
            rng = np.random.default_rng(
                700_000 + 10_000 * cond_idx + 1_000 * ci + j)
            f1p = loso_predict_from_fitted(
                fitted_folds, X, y, rng, 0.0, feature_std, permute_col=j
            )
            feat_drops[fname] = round(float(base - f1p), 6)
        joint_family_drops: dict[str, float] = {}
        for family_idx, (family_name, col_indices) in enumerate(family_indices.items()):
            rng = np.random.default_rng(
                705_000 + 10_000 * cond_idx + 1_000 * ci + family_idx
            )
            f1p = loso_predict_from_fitted(
                fitted_folds,
                X,
                y,
                rng,
                0.0,
                feature_std,
                permute_cols=col_indices,
            )
            joint_family_drops[family_name] = round(float(base - f1p), 6)
        marginal_family_drops = _aggregate_feature_family_drops(feat_drops)
        out[clf_name] = {
            "per_feature": feat_drops,
            "per_family": marginal_family_drops,
            "sum_of_marginal_feature_drops": marginal_family_drops,
            "joint_family_permutation_drop": joint_family_drops,
        }
    return out


def permutation_importance_cross(
    source_condition: str,
    target_condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    control_b: list[str],
    models_dir: str | Path,
    baseline_f1: dict[str, float],
    feature_cols: list[str] | None = None,
    loaded_models: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    models_dir = Path(models_dir)
    target_pool = df.filter(
        (pl.col("condition") == target_condition)
        | pl.col("subject_id").is_in(control_b)
    )
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)
    family_indices = _feature_family_indices(sel_cols)

    dir_idx = _direction_index(source_condition, target_condition)
    out: dict[str, dict[str, float]] = {}
    for ci, clf_name in enumerate(CLF_ORDER):
        if loaded_models is not None:
            pipeline = loaded_models.get(clf_name)
            if pipeline is None:
                raise FileNotFoundError(
                    f"Missing loaded model for {source_condition}/{clf_name} while "
                    f"evaluating {source_condition}->{target_condition} feature sensitivity."
                )
        else:
            model_path = models_dir / f"{source_condition}_{clf_name}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing authoritative source model at {model_path} for "
                    f"{source_condition}->{target_condition} feature sensitivity."
                )
            pipeline = joblib.load(model_path)
        _assert_pipeline_feature_count(pipeline, len(sel_cols))
        base = baseline_f1[clf_name]
        feat_drops: dict[str, float] = {}
        for j, fname in enumerate(sel_cols):
            rng = np.random.default_rng(
                900_000 + 10_000 * dir_idx + 1_000 * ci + j)
            Xt = np.array(X_target, dtype=np.float64, copy=True)
            col = Xt[:, j].copy()
            Xt[:, j] = rng.permutation(col)
            y_pred = pipeline.predict(Xt)
            f1p = float(f1_score(y_target, y_pred, average="macro"))
            feat_drops[fname] = round(float(base - f1p), 6)
        joint_family_drops: dict[str, float] = {}
        for family_idx, (family_name, col_indices) in enumerate(family_indices.items()):
            rng = np.random.default_rng(
                905_000 + 10_000 * dir_idx + 1_000 * ci + family_idx)
            Xt = np.array(X_target, dtype=np.float64, copy=True)
            row_permutation = rng.permutation(len(Xt))
            Xt[:, col_indices] = Xt[row_permutation][:, col_indices]
            y_pred = pipeline.predict(Xt)
            f1p = float(f1_score(y_target, y_pred, average="macro"))
            joint_family_drops[family_name] = round(float(base - f1p), 6)
        marginal_family_drops = _aggregate_feature_family_drops(feat_drops)
        out[clf_name] = {
            "per_feature": feat_drops,
            "per_family": marginal_family_drops,
            "sum_of_marginal_feature_drops": marginal_family_drops,
            "joint_family_permutation_drop": joint_family_drops,
        }
    return out


def per_subject_sensitivity_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    clf_name: str,
    feature_cols: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    _, subj_per_row = _concat_loso_test_indices(X, y, groups)

    y_true = np.array(within_results["classifiers"]
                      [clf_name]["y_true"], dtype=int)
    y_pred = np.array(within_results["classifiers"]
                      [clf_name]["y_pred"], dtype=int)
    if len(y_true) != len(subj_per_row):
        raise ValueError(
            f"Length mismatch within {condition}/{clf_name}: "
            f"y_true={len(y_true)} vs loso_test_rows={len(subj_per_row)}"
        )

    out: dict[str, dict[str, float]] = {}
    for sid in np.unique(subj_per_row):
        mask = subj_per_row == sid
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        out[str(sid)] = {"accuracy": round(acc, 6),
                         "n_strides": int(mask.sum())}
    return out


def per_subject_sensitivity_cross(
    direction_result: dict[str, Any],
    clf_name: str,
) -> dict[str, dict[str, float]]:
    subj_ids = direction_result["target_subject_ids"]
    y_true = np.array(
        direction_result["classifiers"][clf_name]["y_true"], dtype=int)
    y_pred = np.array(
        direction_result["classifiers"][clf_name]["y_pred"], dtype=int)
    if len(y_true) != len(subj_ids):
        raise ValueError(
            f"Length mismatch cross/{clf_name}: y_true={len(y_true)} vs subjects={len(subj_ids)}"
        )

    subj_arr = np.array(subj_ids)
    out: dict[str, dict[str, float]] = {}
    for sid in np.unique(subj_arr):
        mask = subj_arr == sid
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        out[str(sid)] = {"accuracy": round(acc, 6),
                         "n_strides": int(mask.sum())}
    return out


def build_subject_sensitivity_json(
    conditions: tuple[str, ...],
    df: pl.DataFrame,
    control_a: list[str],
    within_by_cond: dict[str, dict[str, Any]],
    cross_results: dict[str, Any] | None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    subject_out: dict[str, Any] = {"within": {}, "cross": {}}

    for cond in conditions:
        wr = within_by_cond[cond]
        subject_out["within"][cond] = {}
        for clf_name in CLF_ORDER:
            subject_out["within"][cond][clf_name] = per_subject_sensitivity_within(
                cond, df, control_a, wr, clf_name, feature_cols=feature_cols
            )

    if cross_results:
        for source_condition, target_condition in DIRECTIONS_ORDER:
            direction_key = f"{source_condition}_to_{target_condition}"
            if direction_key not in cross_results:
                raise KeyError(
                    f"Missing required cross-condition direction: {direction_key}"
                )

            dr = cross_results[direction_key]
            if not isinstance(dr, dict):
                raise TypeError(
                    f"Cross-condition direction payload must be a dict: {direction_key}"
                )

            subject_out["cross"][direction_key] = {}
            for clf_name in CLF_ORDER:
                if clf_name not in dr.get("classifiers", {}):
                    continue
                subject_out["cross"][direction_key][clf_name] = (
                    per_subject_sensitivity_cross(dr, clf_name)
                )

    return subject_out


def _safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro"))


def _left_right_feature_indices(feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    left = [i for i, c in enumerate(feature_cols) if c in _LEFT_COL_HINTS]
    right = [i for i, c in enumerate(feature_cols) if c in _RIGHT_COL_HINTS]
    return np.array(left, dtype=int), np.array(right, dtype=int)


def _apply_structured_corruption(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    feature_cols: list[str],
    corruption_type: str,
    severity_params: tuple[float, ...] | float,
    feature_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    Xt = np.array(X, dtype=np.float64, copy=True)
    yt = np.array(y, dtype=int, copy=True)
    if isinstance(severity_params, float):
        params: tuple[float, ...] = (severity_params,)
    else:
        params = severity_params

    legacy_alias = {
        "missing_strides": "evaluation_row_dropout",
        "label_noise": "benchmark_label_corruption",
        "sensor_dropout": "engineered_feature_masking",
        "gain_bias_drift": "engineered_feature_gain_bias_drift",
        "jitter": "gaussian_feature_space_jitter",
    }
    corruption_type = legacy_alias.get(corruption_type, corruption_type)

    if corruption_type == "engineered_feature_masking":
        p = params[0]
        left_idx, right_idx = _left_right_feature_indices(feature_cols)
        if len(left_idx) == 0 and len(right_idx) == 0:
            return Xt, yt
        mask_rows = rng.random(len(Xt)) < p
        if np.any(mask_rows):
            side = "left" if rng.random() < 0.5 else "right"
            idx = left_idx if side == "left" else right_idx
            if len(idx) > 0:
                Xt[np.ix_(mask_rows, idx)] = 0.0
        return Xt, yt

    if corruption_type == "engineered_feature_gain_bias_drift":
        sigma_g, sigma_b = params[0], params[1]
        unique_subj = np.unique(subject_ids)
        safe_std = np.maximum(feature_std.astype(np.float64), 1e-12)
        for sid in unique_subj:
            mask = subject_ids == sid
            gain = rng.normal(loc=1.0, scale=sigma_g,
                              size=Xt.shape[1]).astype(np.float64)
            bias = rng.normal(loc=0.0, scale=sigma_b * safe_std,
                              size=Xt.shape[1]).astype(np.float64)
            Xt[mask] = Xt[mask] * gain + bias
        return Xt, yt

    if corruption_type == "gaussian_feature_space_jitter":
        sigma = params[0]
        return inject_gaussian_noise(Xt, sigma, feature_std, rng), yt

    if corruption_type == "evaluation_row_dropout":
        frac = params[0]
        keep = rng.random(len(Xt)) >= frac
        if np.sum(keep) < 2:
            keep[rng.choice(len(Xt), size=min(
                2, len(Xt)), replace=False)] = True
        return Xt[keep], yt[keep]

    if corruption_type == "benchmark_label_corruption":
        p = params[0]
        flip = rng.random(len(yt)) < p
        yt[flip] = 1 - yt[flip]
        return Xt, yt

    raise ValueError(f"Unknown corruption type: {corruption_type}")


def evaluate_corruption_sweep_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    feature_cols: list[str] | None = None,
    prefit_folds: dict[str, list[FittedFold]] | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    feature_std = X.std(axis=0, dtype=np.float64)
    cond_idx = _COND_IDX[condition]

    if prefit_folds is not None:
        fitted_by_clf = prefit_folds
    else:
        fitted_by_clf: dict[str, list[FittedFold]] = {}
        for clf_name in CLF_ORDER:
            clf_res = within_results["classifiers"][clf_name]
            modal = clf_res["modal_params"]
            strategy = clf_res.get(
                "selected_imbalance_strategy",
                "synthetic" if clf_res.get("selected_resampling", "smote") == "smote"
                else "balanced"
            )
            fitted_by_clf[clf_name] = loso_fit_all_folds_fixed(
                X,
                y,
                groups,
                clf_name,
                modal,
                imbalance_strategy=strategy,
            )

    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for corruption_type, levels in CORRUPTION_LEVELS.items():
        out[corruption_type] = {}
        for severity_name, params in levels.items():
            out[corruption_type][severity_name] = {}
            for ci, clf_name in enumerate(CLF_ORDER):
                reps: list[float] = []
                fitted_folds = fitted_by_clf[clf_name]
                for r in range(N_NOISE_REPEATS):
                    rng = np.random.default_rng(
                        1_100_000 + 100_000 * cond_idx + 5_000 * ci +
                        100 * list(levels).index(severity_name) + r
                    )
                    y_true_all: list[np.ndarray] = []
                    y_pred_all: list[np.ndarray] = []
                    for pipe, test_idx in fitted_folds:
                        Xt = np.array(X[test_idx], dtype=np.float64, copy=True)
                        yt = np.array(y[test_idx], dtype=int, copy=True)
                        subj = groups[test_idx]
                        Xt_cor, yt_cor = _apply_structured_corruption(
                            Xt, yt, subj, sel_cols, corruption_type, params, feature_std, rng
                        )
                        if len(Xt_cor) == 0:
                            continue
                        y_pred = pipe.predict(Xt_cor)
                        y_true_all.append(yt_cor)
                        y_pred_all.append(np.array(y_pred, dtype=int))
                    if not y_true_all:
                        reps.append(0.0)
                    else:
                        y_t = np.concatenate(y_true_all)
                        y_p = np.concatenate(y_pred_all)
                        reps.append(round(_safe_macro_f1(y_t, y_p), 6))
                out[corruption_type][severity_name][clf_name] = reps
    return out


def evaluate_corruption_sweep_cross(
    source_condition: str,
    target_condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    control_b: list[str],
    models_dir: str | Path,
    feature_cols: list[str] | None = None,
    loaded_models: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    models_dir = Path(models_dir)
    source_pool = df.filter(
        (pl.col("condition") == source_condition) | pl.col(
            "subject_id").is_in(control_a)
    )
    X_source = source_pool.select(sel_cols).to_numpy().astype(np.float64)
    feature_std_source = X_source.std(axis=0, dtype=np.float64)

    target_pool = df.filter(
        (pl.col("condition") == target_condition) | pl.col(
            "subject_id").is_in(control_b)
    )
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)
    subj_target = target_pool["subject_id"].to_numpy()
    feature_std_target = X_target.std(axis=0, dtype=np.float64)
    feature_std_pooled = np.vstack(
        [X_source, X_target]).std(axis=0, dtype=np.float64)
    dir_idx = _direction_index(source_condition, target_condition)
    scale_references = {
        "source_relative": feature_std_source,
        "target_relative": feature_std_target,
        "pooled_relative": feature_std_pooled,
    }
    scale_dependent_corruptions = {
        "engineered_feature_gain_bias_drift",
        "gaussian_feature_space_jitter",
    }

    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for corruption_type, levels in CORRUPTION_LEVELS.items():
        if corruption_type in scale_dependent_corruptions:
            out[corruption_type] = {}
            scale_iter = scale_references.items()
        else:
            out[corruption_type] = {}
            scale_iter = (("not_applicable", feature_std_source),)

        for scale_idx, (scale_name, feature_std) in enumerate(scale_iter):
            target_payload = out[corruption_type]
            if corruption_type in scale_dependent_corruptions:
                target_payload[scale_name] = {}
                severity_container = target_payload[scale_name]
            else:
                severity_container = target_payload

            for severity_name, params in levels.items():
                severity_container[severity_name] = {}
                sev_idx = list(levels).index(severity_name)
                for ci, clf_name in enumerate(CLF_ORDER):
                    if loaded_models is not None:
                        pipeline = loaded_models.get(clf_name)
                        if pipeline is None:
                            raise FileNotFoundError(
                                f"Missing loaded model for {source_condition}/{clf_name} while "
                                f"evaluating {source_condition}->{target_condition} corruption robustness."
                            )
                    else:
                        model_path = models_dir / \
                            f"{source_condition}_{clf_name}.joblib"
                        if not model_path.exists():
                            raise FileNotFoundError(
                                f"Missing authoritative source model at {model_path} for "
                                f"{source_condition}->{target_condition} corruption robustness."
                            )
                        pipeline = joblib.load(model_path)
                    _assert_pipeline_feature_count(pipeline, len(sel_cols))
                    reps: list[float] = []
                    for r in range(N_NOISE_REPEATS):
                        rng = np.random.default_rng(
                            1_500_000
                            + 50_000 * dir_idx
                            + 5_000 * ci
                            + 500 * scale_idx
                            + 100 * sev_idx
                            + r
                        )
                        Xt_cor, yt_cor = _apply_structured_corruption(
                            X_target,
                            y_target,
                            subj_target,
                            sel_cols,
                            corruption_type,
                            params,
                            feature_std,
                            rng,
                        )
                        if len(Xt_cor) == 0:
                            reps.append(0.0)
                            continue
                        y_pred = pipeline.predict(Xt_cor)
                        reps.append(round(_safe_macro_f1(
                            yt_cor, np.array(y_pred, dtype=int)), 6))
                    severity_container[severity_name][clf_name] = reps
    return out


def _compute_nonconformity(
    probas: np.ndarray,
    y_true: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "lac":
        return 1.0 - probas[np.arange(len(y_true)), y_true]
    if method == "aps":
        vals = np.empty(len(y_true), dtype=np.float64)
        for i in range(len(y_true)):
            p = probas[i]
            order = np.argsort(-p)
            rank = int(np.where(order == y_true[i])[0][0])
            vals[i] = float(np.sum(p[order[: rank + 1]]))
        return vals
    raise ValueError(f"Unknown conformal method: {method}")


def _prediction_sets_from_q(
    probas: np.ndarray,
    qhat: float,
    method: str,
) -> dict[str, np.ndarray]:
    n = len(probas)
    raw_sets = np.zeros((n, 2), dtype=bool)
    if method == "lac":
        scores = 1.0 - probas
        raw_sets = scores <= qhat
    elif method == "aps":
        for i in range(n):
            p = probas[i]
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            for cls in (0, 1):
                rank = int(np.where(order == cls)[0][0])
                raw_sets[i, cls] = bool(cum[rank] <= qhat)
    else:
        raise ValueError(f"Unknown conformal method: {method}")
    final_sets = raw_sets.copy()
    empty = ~final_sets.any(axis=1)
    if np.any(empty):
        argmax = np.argmax(probas[empty], axis=1)
        final_sets[np.where(empty)[0], argmax] = True
    return {
        "raw_sets": raw_sets,
        "final_sets": final_sets,
        "raw_sizes": raw_sets.sum(axis=1).astype(int),
        "final_sizes": final_sets.sum(axis=1).astype(int),
        "raw_empty_mask": (~raw_sets.any(axis=1)),
    }


def _qhat_from_scores(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def _hard_label_consensus_curve(
    probs: np.ndarray,
    subject_ids: np.ndarray,
    n_list: tuple[int, ...] = AGG_N_STRIDES,
    n_boot: int = 200,
) -> dict[str, float]:
    out: dict[str, float] = {}
    pred = np.argmax(probs, axis=1)
    unique_subj = np.unique(subject_ids)
    for n in n_list:
        ok = 0
        total = 0
        for sid in unique_subj:
            idx = np.where(subject_ids == sid)[0]
            if len(idx) == 0:
                continue
            for r in range(n_boot):
                take = idx[np.random.default_rng(
                    42 + n + len(idx) + r * 7919
                ).integers(0, len(idx), size=min(n, len(idx)))]
                labels = pred[take]
                total += 1
                ok += int(np.all(labels == labels[0]))
        out[str(n)] = round(float(ok / total), 6) if total > 0 else 0.0
    return out


def _stride_aggregation_singleton_rate(
    probs: np.ndarray,
    subject_ids: np.ndarray,
    n_list: tuple[int, ...] = AGG_N_STRIDES,
    n_boot: int = 200,
) -> dict[str, float]:
    return _hard_label_consensus_curve(
        probs,
        subject_ids,
        n_list=n_list,
        n_boot=n_boot,
    )


def _set_size_summary(sizes: np.ndarray) -> dict[str, float]:
    """Summarize empty/singleton/doubleton rates for binary prediction sets."""
    return {
        "empty_rate": round(float(np.mean(sizes == 0)), 6),
        "singleton_rate": round(float(np.mean(sizes == 1)), 6),
        "doubleton_rate": round(float(np.mean(sizes == 2)), 6),
        "mean_set_size": round(float(np.mean(sizes)), 6),
    }


def _class_conditional_coverage(
    cover: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Binary class-conditional coverage summary."""
    out: dict[str, float] = {}
    for cls, label in ((0, "control"), (1, "disease")):
        mask = y_true == cls
        out[label] = round(float(np.mean(cover[mask])),
                           6) if np.any(mask) else 0.0
    return out


def _subject_level_coverage_summary(
    cover: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, float]:
    """Aggregate stride-level coverage to one mean coverage score per subject."""
    subject_arr = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_arr)
    per_subject = []
    for subject_id in unique_subjects:
        mask = subject_arr == subject_id
        per_subject.append(float(np.mean(cover[mask])))
    arr = np.asarray(per_subject, dtype=np.float64)
    return {
        "n_subjects": int(len(arr)),
        "mean_subject_coverage": round(float(np.mean(arr)), 6) if len(arr) else 0.0,
        "min_subject_coverage": round(float(np.min(arr)), 6) if len(arr) else 0.0,
        "max_subject_coverage": round(float(np.max(arr)), 6) if len(arr) else 0.0,
    }


def evaluate_conformal_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    feature_cols: list[str] | None = None,
    prefit_folds: dict[str, list[FittedFold]] | None = None,
) -> dict[str, dict[str, Any]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()

    out: dict[str, dict[str, Any]] = {}
    for clf_name in CLF_ORDER:
        clf_res = within_results["classifiers"][clf_name]
        modal = clf_res["modal_params"]
        strategy = clf_res.get(
            "selected_imbalance_strategy",
            "synthetic" if clf_res.get("selected_resampling", "smote") == "smote"
            else "balanced"
        )
        if prefit_folds is not None:
            fitted_folds = prefit_folds[clf_name]
        else:
            fitted_folds = loso_fit_all_folds_fixed(
                X,
                y,
                groups,
                clf_name,
                modal,
                imbalance_strategy=strategy,
            )
        fold_probs: list[np.ndarray] = []
        fold_true: list[np.ndarray] = []
        fold_groups: list[np.ndarray] = []
        for pipe, test_idx in fitted_folds:
            Xt = np.array(X[test_idx], dtype=np.float64, copy=False)
            prob = pipe.predict_proba(Xt).astype(np.float64)
            fold_probs.append(prob)
            fold_true.append(y[test_idx])
            fold_groups.append(groups[test_idx])

        probs_arr = np.concatenate(fold_probs, axis=0)
        grp_arr = np.concatenate(fold_groups)
        clf_out: dict[str, Any] = {
            "lac": {},
            "aps": {},
            "hard_label_consensus_curve": _hard_label_consensus_curve(
                probs_arr,
                grp_arr,
                n_list=AGG_N_STRIDES,
            ),
        }
        for method in ("lac", "aps"):
            alpha_out: dict[str, Any] = {}
            for alpha in CONFORMAL_ALPHAS:
                cover_all: list[np.ndarray] = []
                raw_size_all: list[np.ndarray] = []
                size_all: list[np.ndarray] = []
                probs_all: list[np.ndarray] = []
                groups_all: list[np.ndarray] = []
                for i in range(len(fitted_folds)):
                    test_prob = fold_probs[i]
                    test_true = fold_true[i]
                    test_groups = fold_groups[i]
                    cal_prob = np.concatenate(
                        [fold_probs[j] for j in range(len(fitted_folds)) if j != i], axis=0)
                    cal_true = np.concatenate(
                        [fold_true[j] for j in range(len(fitted_folds)) if j != i], axis=0)
                    cal_scores = _compute_nonconformity(
                        cal_prob, cal_true, method)
                    qhat = _qhat_from_scores(cal_scores, alpha)
                    set_payload = _prediction_sets_from_q(
                        test_prob, qhat, method)
                    cover = set_payload["final_sets"][np.arange(
                        len(test_true)), test_true]
                    cover_all.append(cover.astype(int))
                    raw_size_all.append(set_payload["raw_sizes"])
                    size_all.append(set_payload["final_sizes"])
                    probs_all.append(test_prob)
                    groups_all.append(test_groups)
                cov_arr = np.concatenate(cover_all)
                raw_size_arr = np.concatenate(raw_size_all)
                size_arr = np.concatenate(size_all)
                alpha_out[str(alpha)] = {
                    "policy": "exploratory_only",
                    "coverage_marginal": round(float(np.mean(cov_arr)), 6),
                    "coverage_class_conditional": _class_conditional_coverage(
                        cov_arr.astype(np.float64),
                        np.concatenate(fold_true),
                    ),
                    "coverage_subject_level": _subject_level_coverage_summary(
                        cov_arr.astype(np.float64),
                        grp_arr,
                    ),
                    "raw_set_stats": _set_size_summary(raw_size_arr),
                    "post_fallback_set_stats": _set_size_summary(size_arr),
                }
            clf_out[method] = alpha_out
        out[clf_name] = clf_out
    return out


def evaluate_conformal_cross(
    source_condition: str,
    target_condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    control_b: list[str],
    models_dir: str | Path,
    feature_cols: list[str] | None = None,
    loaded_models: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    sel_cols = feature_cols if feature_cols is not None else list(
        ALL_FEATURE_COLS)
    models_dir = Path(models_dir)

    source_pool = df.filter(
        (pl.col("condition") == source_condition) | pl.col(
            "subject_id").is_in(control_a)
    )
    target_pool = df.filter(
        (pl.col("condition") == target_condition) | pl.col(
            "subject_id").is_in(control_b)
    )
    X_source = source_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_source = source_pool["label"].to_numpy().astype(int)
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)
    groups_target = target_pool["subject_id"].to_numpy()

    # Cross-condition conformal calibration deliberately uses source-pool
    # nonconformity scores only. Target data are never used to set qhat, and
    # the control_A / control_B split keeps the healthy-subject IDs disjoint
    # across calibration and evaluation.
    source_subjects = set(source_pool["subject_id"].to_list())
    target_subjects = set(target_pool["subject_id"].to_list())
    overlap = source_subjects & target_subjects
    if overlap:
        raise ValueError(
            'Cross conformal calibration/evaluation subject overlap detected: '
            f'{sorted(overlap)}'
        )

    out: dict[str, dict[str, Any]] = {}
    for clf_name in CLF_ORDER:
        if loaded_models is not None:
            pipeline = loaded_models.get(clf_name)
            if pipeline is None:
                raise FileNotFoundError(
                    f"Missing loaded model for {source_condition}/{clf_name} while "
                    f"evaluating {source_condition}->{target_condition} conformal diagnostics."
                )
        else:
            model_path = models_dir / f"{source_condition}_{clf_name}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing authoritative source model at {model_path} for "
                    f"{source_condition}->{target_condition} conformal diagnostics."
                )
            pipeline = joblib.load(model_path)
        _assert_pipeline_feature_count(pipeline, len(sel_cols))
        # Calibration scores come only from source predictions against y_source.
        src_prob = pipeline.predict_proba(X_source).astype(np.float64)
        # Coverage is then measured only on target predictions against y_target.
        tgt_prob = pipeline.predict_proba(X_target).astype(np.float64)

        clf_out: dict[str, Any] = {
            "lac": {},
            "aps": {},
            "hard_label_consensus_curve": _hard_label_consensus_curve(
                tgt_prob,
                groups_target,
                n_list=AGG_N_STRIDES,
            ),
        }
        for method in ("lac", "aps"):
            cal_scores = _compute_nonconformity(src_prob, y_source, method)
            alpha_out: dict[str, Any] = {}
            for alpha in CONFORMAL_ALPHAS:
                qhat = _qhat_from_scores(cal_scores, alpha)
                set_payload = _prediction_sets_from_q(tgt_prob, qhat, method)
                cover = set_payload["final_sets"][np.arange(
                    len(y_target)), y_target]
                alpha_out[str(alpha)] = {
                    "policy": "exploratory_only_under_shift",
                    "coverage_marginal": round(float(np.mean(cover)), 6),
                    "coverage_class_conditional": _class_conditional_coverage(
                        cover.astype(np.float64),
                        y_target,
                    ),
                    "coverage_subject_level": _subject_level_coverage_summary(
                        cover.astype(np.float64),
                        groups_target,
                    ),
                    "raw_set_stats": _set_size_summary(set_payload["raw_sizes"]),
                    "post_fallback_set_stats": _set_size_summary(set_payload["final_sizes"]),
                }
            clf_out[method] = alpha_out
        out[clf_name] = clf_out
    return out


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
