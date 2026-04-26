"""
Step 5: noise robustness and sensitivity analysis.

Self-contained pipeline helpers (mirrors train.build_pipeline / classifier defaults)
so this module does not import train.py (strict Step 5 file scope).

- Gaussian noise on evaluation inputs only.
- Per-feature sensitivity via column shuffle on evaluation inputs.
- Per-subject accuracy from stored LOSO / transfer predictions.
- Structured corruption benchmark (sensor dropout / drift / jitter / missing / label noise).
- Split-conformal prediction diagnostics.
"""

from __future__ import annotations

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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CLF_ORDER: tuple[str, ...] = ("rf", "knn", "svm", "dt", "qda", "xgb", "lgbm")

SIGMA_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
N_NOISE_REPEATS: int = 30

_SCALE_REQUIRED = frozenset({"svm", "knn"})
_LEFT_COL_HINTS = ("left_stride_s", "left_swing_s", "left_swing_pct", "left_stance_s", "left_stance_pct")
_RIGHT_COL_HINTS = ("right_stride_s", "right_swing_s", "right_swing_pct", "right_stance_s", "right_stance_pct")

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
    "sensor_dropout": {"light": (0.05,), "medium": (0.15,), "heavy": (0.30,)},
    "gain_bias_drift": {
        "light": (0.02, 0.05),
        "medium": (0.05, 0.10),
        "heavy": (0.10, 0.20),
    },
    "jitter": {"light": (0.10,), "medium": (0.25,), "heavy": (0.50,)},
    "missing_strides": {"light": (0.10,), "medium": (0.30,), "heavy": (0.50,)},
    "label_noise": {"light": (0.05,), "medium": (0.10,), "heavy": (0.20,)},
}
CONFORMAL_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)
AGG_N_STRIDES: tuple[int, ...] = (1, 3, 5, 10, 25, 50)


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


def _configure_classifier_for_resampling(
    classifier_name: str,
    clf: Any,
    use_smote: bool,
) -> Any:
    configured = clone(clf)
    name = classifier_name.lower()
    if not use_smote and name == "lgbm":
        configured.set_params(class_weight="balanced")
    return configured


def _get_fit_kwargs(
    classifier_name: str,
    y_fit: np.ndarray,
    use_smote: bool,
) -> dict[str, Any]:
    if not use_smote and classifier_name.lower() == "xgb":
        return {"clf__sample_weight": _balanced_sample_weight(y_fit)}
    return {}


def build_pipeline(classifier_name: str, clf: Any, use_smote: bool = True) -> ImbPipeline:
    """Same step order as src/train.py build_pipeline."""
    name = classifier_name.lower()
    steps: list[tuple[str, Any]] = []
    if name in _SCALE_REQUIRED:
        steps.append(("scaler", RobustScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def _fresh_classifier(clf_name: str) -> Any:
    """New classifier instance matching train.get_classifier_configs() defaults."""
    n = clf_name.lower()
    if n == "rf":
        return RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=1
        )
    if n == "knn":
        return KNeighborsClassifier()
    if n == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    if n == "dt":
        return DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        )
    if n == "qda":
        return QuadraticDiscriminantAnalysis()
    if n == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=1, tree_method="hist"
        )
    if n == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=42, n_jobs=1, verbose=-1
        )
    raise ValueError(f"Unknown classifier: {clf_name}")


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
        (pl.col("condition") == condition) | pl.col("subject_id").is_in(control_a)
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
    n_features = getattr(pipeline, "n_features_in_", None)
    if n_features is not None:
        return int(n_features)
    clf = getattr(pipeline, "named_steps", {}).get("clf")
    if clf is not None:
        clf_n_features = getattr(clf, "n_features_in_", None)
        if clf_n_features is not None:
            return int(clf_n_features)
    return None


def _assert_pipeline_feature_count(pipeline: Any, expected: int) -> None:
    got = _fitted_pipeline_feature_count(pipeline)
    if got is None:
        return
    if got != expected:
        raise ValueError(f"Pipeline feature count mismatch: expected={expected}, got={got}")


def loso_fit_all_folds_fixed(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str,
    modal_params: dict[str, Any],
    use_smote: bool,
) -> list[FittedFold]:
    clf_template = _fresh_classifier(clf_name)
    outer_loso = LeaveOneGroupOut()
    fitted: list[FittedFold] = []
    for train_idx, test_idx in outer_loso.split(X, y, groups):
        clf_variant = _configure_classifier_for_resampling(clf_name, clf_template, use_smote)
        base = build_pipeline(clf_name, clf_variant, use_smote=use_smote)
        base.set_params(**modal_params)
        pipe = clone(base)
        X_train, y_train = X[train_idx], y[train_idx]
        fit_kwargs = _get_fit_kwargs(clf_name, y_train, use_smote)
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
        if sigma_frac > 0:
            Xt = inject_gaussian_noise(Xt, sigma_frac, feature_std, rng)
        y_pred = pipe.predict(Xt)
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
    y_t = np.concatenate(y_true_all)
    y_p = np.concatenate(y_pred_all)
    return float(f1_score(y_t, y_p, average="macro"))


def evaluate_noise_sweep_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    feature_cols: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
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
        use_smote = clf_res.get("selected_resampling", "smote") != "no_smote"
        fitted_folds = loso_fit_all_folds_fixed(
            X, y, groups, clf_name, modal, use_smote=use_smote
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
) -> dict[str, dict[str, list[float]]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    models_dir = Path(models_dir)
    source_pool = df.filter(
        (pl.col("condition") == source_condition)
        | pl.col("subject_id").is_in(control_a)
    )
    X_source = source_pool.select(sel_cols).to_numpy().astype(np.float64)
    feature_std = X_source.std(axis=0, dtype=np.float64)

    target_pool = df.filter(
        (pl.col("condition") == target_condition)
        | pl.col("subject_id").is_in(control_b)
    )
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)

    dir_idx = _direction_index(source_condition, target_condition)
    out: dict[str, dict[str, list[float]]] = {}
    for ci, clf_name in enumerate(CLF_ORDER):
        model_path = models_dir / f"{source_condition}_{clf_name}.joblib"
        if not model_path.exists():
            continue
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
        out[clf_name] = clf_out
    return out


def permutation_importance_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    baseline_f1: dict[str, float],
    feature_cols: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    feature_std = X.std(axis=0, dtype=np.float64)
    cond_idx = _COND_IDX[condition]

    out: dict[str, dict[str, float]] = {}
    for ci, clf_name in enumerate(CLF_ORDER):
        clf_res = within_results["classifiers"][clf_name]
        modal = clf_res["modal_params"]
        use_smote = clf_res.get("selected_resampling", "smote") != "no_smote"
        base = baseline_f1[clf_name]
        fitted_folds = loso_fit_all_folds_fixed(
            X, y, groups, clf_name, modal, use_smote=use_smote
        )
        feat_drops: dict[str, float] = {}
        for j, fname in enumerate(sel_cols):
            rng = np.random.default_rng(700_000 + 10_000 * cond_idx + 1_000 * ci + j)
            f1p = loso_predict_from_fitted(
                fitted_folds, X, y, rng, 0.0, feature_std, permute_col=j
            )
            feat_drops[fname] = round(float(base - f1p), 6)
        out[clf_name] = feat_drops
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
) -> dict[str, dict[str, float]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    models_dir = Path(models_dir)
    target_pool = df.filter(
        (pl.col("condition") == target_condition)
        | pl.col("subject_id").is_in(control_b)
    )
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)

    dir_idx = _direction_index(source_condition, target_condition)
    out: dict[str, dict[str, float]] = {}
    for ci, clf_name in enumerate(CLF_ORDER):
        model_path = models_dir / f"{source_condition}_{clf_name}.joblib"
        if not model_path.exists():
            continue
        pipeline = joblib.load(model_path)
        _assert_pipeline_feature_count(pipeline, len(sel_cols))
        base = baseline_f1[clf_name]
        feat_drops: dict[str, float] = {}
        for j, fname in enumerate(sel_cols):
            rng = np.random.default_rng(900_000 + 10_000 * dir_idx + 1_000 * ci + j)
            Xt = np.array(X_target, dtype=np.float64, copy=True)
            col = Xt[:, j].copy()
            Xt[:, j] = rng.permutation(col)
            y_pred = pipeline.predict(Xt)
            f1p = float(f1_score(y_target, y_pred, average="macro"))
            feat_drops[fname] = round(float(base - f1p), 6)
        out[clf_name] = feat_drops
    return out


def per_subject_sensitivity_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    clf_name: str,
    feature_cols: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    _, subj_per_row = _concat_loso_test_indices(X, y, groups)

    y_true = np.array(within_results["classifiers"][clf_name]["y_true"], dtype=int)
    y_pred = np.array(within_results["classifiers"][clf_name]["y_pred"], dtype=int)
    if len(y_true) != len(subj_per_row):
        raise ValueError(
            f"Length mismatch within {condition}/{clf_name}: "
            f"y_true={len(y_true)} vs loso_test_rows={len(subj_per_row)}"
        )

    out: dict[str, dict[str, float]] = {}
    for sid in np.unique(subj_per_row):
        mask = subj_per_row == sid
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        out[str(sid)] = {"accuracy": round(acc, 6), "n_strides": int(mask.sum())}
    return out


def per_subject_sensitivity_cross(
    direction_result: dict[str, Any],
    clf_name: str,
) -> dict[str, dict[str, float]]:
    subj_ids = direction_result["target_subject_ids"]
    y_true = np.array(direction_result["classifiers"][clf_name]["y_true"], dtype=int)
    y_pred = np.array(direction_result["classifiers"][clf_name]["y_pred"], dtype=int)
    if len(y_true) != len(subj_ids):
        raise ValueError(
            f"Length mismatch cross/{clf_name}: y_true={len(y_true)} vs subjects={len(subj_ids)}"
        )

    subj_arr = np.array(subj_ids)
    out: dict[str, dict[str, float]] = {}
    for sid in np.unique(subj_arr):
        mask = subj_arr == sid
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        out[str(sid)] = {"accuracy": round(acc, 6), "n_strides": int(mask.sum())}
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
        for direction_key, dr in cross_results.items():
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

    if corruption_type == "sensor_dropout":
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

    if corruption_type == "gain_bias_drift":
        sigma_g, sigma_b = params[0], params[1]
        unique_subj = np.unique(subject_ids)
        safe_std = np.maximum(feature_std.astype(np.float64), 1e-12)
        for sid in unique_subj:
            mask = subject_ids == sid
            gain = rng.normal(loc=1.0, scale=sigma_g, size=Xt.shape[1]).astype(np.float64)
            bias = rng.normal(loc=0.0, scale=sigma_b * safe_std, size=Xt.shape[1]).astype(np.float64)
            Xt[mask] = Xt[mask] * gain + bias
        return Xt, yt

    if corruption_type == "jitter":
        sigma = params[0]
        return inject_gaussian_noise(Xt, sigma, feature_std, rng), yt

    if corruption_type == "missing_strides":
        frac = params[0]
        keep = rng.random(len(Xt)) >= frac
        if np.sum(keep) < 2:
            keep[rng.choice(len(Xt), size=min(2, len(Xt)), replace=False)] = True
        return Xt[keep], yt[keep]

    if corruption_type == "label_noise":
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
) -> dict[str, dict[str, dict[str, list[float]]]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()
    feature_std = X.std(axis=0, dtype=np.float64)
    cond_idx = _COND_IDX[condition]

    fitted_by_clf: dict[str, list[FittedFold]] = {}
    for clf_name in CLF_ORDER:
        clf_res = within_results["classifiers"][clf_name]
        modal = clf_res["modal_params"]
        use_smote = clf_res.get("selected_resampling", "smote") != "no_smote"
        fitted_by_clf[clf_name] = loso_fit_all_folds_fixed(
            X, y, groups, clf_name, modal, use_smote=use_smote
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
                        1_100_000 + 100_000 * cond_idx + 5_000 * ci + 100 * list(levels).index(severity_name) + r
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
) -> dict[str, dict[str, dict[str, list[float]]]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    models_dir = Path(models_dir)
    source_pool = df.filter(
        (pl.col("condition") == source_condition) | pl.col("subject_id").is_in(control_a)
    )
    X_source = source_pool.select(sel_cols).to_numpy().astype(np.float64)
    feature_std = X_source.std(axis=0, dtype=np.float64)

    target_pool = df.filter(
        (pl.col("condition") == target_condition) | pl.col("subject_id").is_in(control_b)
    )
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)
    subj_target = target_pool["subject_id"].to_numpy()
    dir_idx = _direction_index(source_condition, target_condition)

    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for corruption_type, levels in CORRUPTION_LEVELS.items():
        out[corruption_type] = {}
        for severity_name, params in levels.items():
            out[corruption_type][severity_name] = {}
            sev_idx = list(levels).index(severity_name)
            for ci, clf_name in enumerate(CLF_ORDER):
                model_path = models_dir / f"{source_condition}_{clf_name}.joblib"
                if not model_path.exists():
                    continue
                pipeline = joblib.load(model_path)
                _assert_pipeline_feature_count(pipeline, len(sel_cols))
                reps: list[float] = []
                for r in range(N_NOISE_REPEATS):
                    rng = np.random.default_rng(1_500_000 + 50_000 * dir_idx + 5_000 * ci + 100 * sev_idx + r)
                    Xt_cor, yt_cor = _apply_structured_corruption(
                        X_target, y_target, subj_target, sel_cols, corruption_type, params, feature_std, rng
                    )
                    if len(Xt_cor) == 0:
                        reps.append(0.0)
                        continue
                    y_pred = pipeline.predict(Xt_cor)
                    reps.append(round(_safe_macro_f1(yt_cor, np.array(y_pred, dtype=int)), 6))
                out[corruption_type][severity_name][clf_name] = reps
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
) -> tuple[np.ndarray, np.ndarray]:
    n = len(probas)
    sets = np.zeros((n, 2), dtype=bool)
    if method == "lac":
        scores = 1.0 - probas
        sets = scores <= qhat
    elif method == "aps":
        for i in range(n):
            p = probas[i]
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            for cls in (0, 1):
                rank = int(np.where(order == cls)[0][0])
                sets[i, cls] = bool(cum[rank] <= qhat)
    else:
        raise ValueError(f"Unknown conformal method: {method}")
    # Ensure non-empty sets.
    empty = ~sets.any(axis=1)
    if np.any(empty):
        argmax = np.argmax(probas[empty], axis=1)
        sets[np.where(empty)[0], argmax] = True
    sizes = sets.sum(axis=1).astype(int)
    return sets, sizes


def _qhat_from_scores(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def _stride_aggregation_singleton_rate(
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
            for _ in range(n_boot):
                take = idx[np.random.default_rng(42 + n + len(idx)).integers(0, len(idx), size=min(n, len(idx)))]
                labels = pred[take]
                total += 1
                ok += int(np.all(labels == labels[0]))
        out[str(n)] = round(float(ok / total), 6) if total > 0 else 0.0
    return out


def evaluate_conformal_within(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    within_results: dict[str, Any],
    feature_cols: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    pool = _within_pool(df, condition, control_a)
    X = pool.select(sel_cols).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()

    out: dict[str, dict[str, Any]] = {}
    for clf_name in CLF_ORDER:
        clf_res = within_results["classifiers"][clf_name]
        modal = clf_res["modal_params"]
        use_smote = clf_res.get("selected_resampling", "smote") != "no_smote"
        fitted_folds = loso_fit_all_folds_fixed(X, y, groups, clf_name, modal, use_smote=use_smote)
        fold_probs: list[np.ndarray] = []
        fold_true: list[np.ndarray] = []
        fold_groups: list[np.ndarray] = []
        for pipe, test_idx in fitted_folds:
            Xt = np.array(X[test_idx], dtype=np.float64, copy=False)
            prob = pipe.predict_proba(Xt).astype(np.float64)
            fold_probs.append(prob)
            fold_true.append(y[test_idx])
            fold_groups.append(groups[test_idx])

        clf_out: dict[str, Any] = {"lac": {}, "aps": {}}
        for method in ("lac", "aps"):
            alpha_out: dict[str, Any] = {}
            for alpha in CONFORMAL_ALPHAS:
                cover_all: list[np.ndarray] = []
                size_all: list[np.ndarray] = []
                probs_all: list[np.ndarray] = []
                groups_all: list[np.ndarray] = []
                for i in range(len(fitted_folds)):
                    test_prob = fold_probs[i]
                    test_true = fold_true[i]
                    test_groups = fold_groups[i]
                    cal_prob = np.concatenate([fold_probs[j] for j in range(len(fitted_folds)) if j != i], axis=0)
                    cal_true = np.concatenate([fold_true[j] for j in range(len(fitted_folds)) if j != i], axis=0)
                    cal_scores = _compute_nonconformity(cal_prob, cal_true, method)
                    qhat = _qhat_from_scores(cal_scores, alpha)
                    sets, sizes = _prediction_sets_from_q(test_prob, qhat, method)
                    cover = sets[np.arange(len(test_true)), test_true]
                    cover_all.append(cover.astype(int))
                    size_all.append(sizes)
                    probs_all.append(test_prob)
                    groups_all.append(test_groups)
                cov_arr = np.concatenate(cover_all)
                size_arr = np.concatenate(size_all)
                probs_arr = np.concatenate(probs_all, axis=0)
                grp_arr = np.concatenate(groups_all)
                alpha_out[str(alpha)] = {
                    "coverage_marginal": round(float(np.mean(cov_arr)), 6),
                    "mean_set_size": round(float(np.mean(size_arr)), 6),
                    "stride_aggregation_curve": _stride_aggregation_singleton_rate(
                        probs_arr, grp_arr, n_list=AGG_N_STRIDES
                    ),
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
) -> dict[str, dict[str, Any]]:
    sel_cols = feature_cols if feature_cols is not None else list(ALL_FEATURE_COLS)
    models_dir = Path(models_dir)

    source_pool = df.filter(
        (pl.col("condition") == source_condition) | pl.col("subject_id").is_in(control_a)
    )
    target_pool = df.filter(
        (pl.col("condition") == target_condition) | pl.col("subject_id").is_in(control_b)
    )
    X_source = source_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_source = source_pool["label"].to_numpy().astype(int)
    X_target = target_pool.select(sel_cols).to_numpy().astype(np.float64)
    y_target = target_pool["label"].to_numpy().astype(int)
    groups_target = target_pool["subject_id"].to_numpy()

    out: dict[str, dict[str, Any]] = {}
    for clf_name in CLF_ORDER:
        model_path = models_dir / f"{source_condition}_{clf_name}.joblib"
        if not model_path.exists():
            continue
        pipeline = joblib.load(model_path)
        _assert_pipeline_feature_count(pipeline, len(sel_cols))
        src_prob = pipeline.predict_proba(X_source).astype(np.float64)
        tgt_prob = pipeline.predict_proba(X_target).astype(np.float64)

        clf_out: dict[str, Any] = {"lac": {}, "aps": {}}
        for method in ("lac", "aps"):
            cal_scores = _compute_nonconformity(src_prob, y_source, method)
            alpha_out: dict[str, Any] = {}
            for alpha in CONFORMAL_ALPHAS:
                qhat = _qhat_from_scores(cal_scores, alpha)
                sets, sizes = _prediction_sets_from_q(tgt_prob, qhat, method)
                cover = sets[np.arange(len(y_target)), y_target]
                alpha_out[str(alpha)] = {
                    "coverage_marginal": round(float(np.mean(cover)), 6),
                    "mean_set_size": round(float(np.mean(sizes)), 6),
                    "stride_aggregation_curve": _stride_aggregation_singleton_rate(
                        tgt_prob, groups_target, n_list=AGG_N_STRIDES
                    ),
                }
            clf_out[method] = alpha_out
        out[clf_name] = clf_out
    return out


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
