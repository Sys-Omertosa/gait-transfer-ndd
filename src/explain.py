"""
src/explain.py

SHAP-based transfer-failure diagnosis for gait classifier evaluation.

Implements the δj metric: the absolute shift in mean feature importance between
the within-condition and cross-condition settings. Large δj for feature j means
the model relied on that feature differently when confronted with out-of-distribution
gait data, identifying it as a transfer-failure feature.

Reference:
  Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions."
    NeurIPS 30. Shapley value theory and TreeExplainer / KernelExplainer.
  Xiang et al. (2025) — "XAI in Gait Analysis." Frontiers Bioengineering.
    Confirms SHAP is used in 11 gait studies, all within-condition only.
    Defines the cross-condition diagnostic application as a methodological novelty.

Explainer assignment per classifier (all output in probability scale):
  RF, DT   — shap.TreeExplainer, feature_perturbation='tree_path_dependent'.
               RF/DT tree_path_dependent raw output is already probability scale.
               Completeness: base[1] + sv[:,:,1].sum(axis=1) == predict_proba[:,1].
  XGB, LGB — shap.TreeExplainer, feature_perturbation='interventional',
               model_output='probability'. Default (tree_path_dependent) for
               XGB/LGB produces log-odds output, which is incomparable to RF/DT.
               Interventional mode with probability output gives exact completeness.
  SVM, QDA, KNN — shap.KernelExplainer(pipeline.predict_proba, background).
               Full ImbPipeline predict_proba is passed so SHAP values are in the
               original 14-dimensional feature space regardless of internal scaling.

SMOTE convention: SMOTE is part of the ImbPipeline and is skipped at predict time
(ImbPipeline does not run samplers during transform/predict). Passing the full
pipeline predict_proba to KernelExplainer is therefore equivalent to passing the
underlying classifier predict_proba on pre-scaled data, with the correct output.

Background data: a class-balanced shap.kmeans summary (k=100) computed once per
source condition and reused across all explainer types for that source. Balancing
ensures the k-means centers represent both disease and control strides equally,
so the base value (E[f(background)]) is close to 0.5 for all classifiers, making
waterfall plots and base-value comparisons interpretable across classifier families.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import shap

from features import ALL_FEATURE_COLS

warnings.filterwarnings('ignore', category=UserWarning)


# ── Explainer configuration ───────────────────────────────────────────────────

# Classifiers whose TreeExplainer default (tree_path_dependent) already outputs
# probability-scale SHAP values. Confirmed empirically: for these estimator
# types, shap.TreeExplainer(clf).expected_value is a length-2 array of class
# probabilities, and base[1] + sv[:,:,1].sum(axis=1) == predict_proba[:,1].
_TREE_PATH_DEPENDENT = {'rf', 'dt'}

# Classifiers requiring interventional + model_output='probability' because their
# default tree_path_dependent output is in log-odds (unbounded) space, not [0,1].
_TREE_INTERVENTIONAL = {'xgb', 'lgbm'}

# Classifiers requiring KernelExplainer (no tree structure).
_KERNEL_EXPLAINERS = {'svm', 'qda', 'knn'}

# KernelExplainer settings applied uniformly to all three kernel classifiers.
# 1,000 stratified samples and nsamples=1024 coalitions gives stable mean(|phi|)
# estimates and is consistent across classifiers for the methods section.
_KERNEL_N_EXPLAINED = 1000
_KERNEL_NSAMPLES = 1024

# δj normalisation: cap on normalised δj to bound the "emerged features" case
# where mean(|phi_within|) is near zero. Features hitting this cap are flagged.
_DELTA_J_NORM_CAP = 10.0
_EMERGED_THRESHOLD = 1e-3   # mean(|phi_within|) below this → "emerged"
_EPS = 1e-10                # numerical floor for division


def get_shap_config(clf_name: str) -> dict[str, Any]:
    """
    Return explainer configuration for a given classifier.

    Determines which SHAP algorithm to use and the sample budget for
    KernelExplainer classifiers. TreeExplainer classifiers always operate
    on the full pool (no sample budget).

    Args:
        clf_name: One of 'rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm'.

    Returns:
        Dict with keys:
          'explainer'    — 'tree_tpd' | 'tree_int' | 'kernel'
          'n_explained'  — int for kernel (explained sample count),
                           None for tree (always uses full pool)
          'nsamples'     — int for kernel (KernelExplainer coalition count),
                           None for tree
    """
    name = clf_name.lower()
    if name in _TREE_PATH_DEPENDENT:
        return {'explainer': 'tree_tpd', 'n_explained': None, 'nsamples': None}
    if name in _TREE_INTERVENTIONAL:
        return {'explainer': 'tree_int', 'n_explained': None, 'nsamples': None}
    if name in _KERNEL_EXPLAINERS:
        return {
            'explainer':   'kernel',
            'n_explained': _KERNEL_N_EXPLAINED,
            'nsamples':    _KERNEL_NSAMPLES,
        }
    raise ValueError(
        f"Unknown classifier '{clf_name}'. "
        f"Expected one of: {sorted(_TREE_PATH_DEPENDENT | _TREE_INTERVENTIONAL | _KERNEL_EXPLAINERS)}"
    )


# ── Background data ───────────────────────────────────────────────────────────

def get_background_data(
    X_source: np.ndarray,
    y_source: np.ndarray,
    k: int = 100,
    rng: np.random.Generator | None = None,
) -> Any:
    """
    Compute a class-balanced k-means summary of the source pool for background data.

    The background is class-balanced before clustering: equal numbers of disease
    and control strides are sampled as input to k-means so that the resulting
    cluster centres represent both classes equally. Without balancing, the
    disease-heavy source pool (~64% disease) causes k-means centres to concentrate
    in the disease region, producing base values of ~0.88 for interventional
    TreeExplainer (XGB/LGB) instead of ~0.50. This makes waterfall plots misleading
    and base values non-comparable across classifiers.

    RF/DT tree_path_dependent is not affected by the background content (it
    computes base values from the tree structure internally). Class balancing
    primarily corrects interventional TreeExplainer and KernelExplainer.

    The balanced background is used by:
      - Interventional TreeExplainer (XGB, LGB): raw .data array passed as data=
      - KernelExplainer (SVM, QDA, KNN): DenseData object passed as background

    Args:
        X_source: Feature matrix of the source pool, shape (n_source, 14).
        y_source: Binary label vector, shape (n_source,). Used to balance classes.
        k:        Number of k-means cluster centres. Default 100.
        rng:      NumPy Generator for reproducible balanced sampling.
                  If None, uses np.random.default_rng(0) as a fixed fallback.

    Returns:
        shap.kmeans DenseData object. For interventional TreeExplainer, extract
        .data (handled inside compute_shap_values). For KernelExplainer, pass
        the object directly.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    disease_idx = np.where(y_source == 1)[0]
    control_idx = np.where(y_source == 0)[0]

    # Draw equal numbers from each class; cap at 5× k//2 per class so the
    # input to k-means is at most k*10 points but never undersamples a small class.
    n_per_class = min(len(disease_idx), len(control_idx), k // 2 * 5)
    sampled_disease = rng.choice(disease_idx, size=n_per_class, replace=False)
    sampled_control = rng.choice(control_idx, size=n_per_class, replace=False)

    X_balanced = np.vstack(
        [X_source[sampled_disease], X_source[sampled_control]])

    # shap.sample (random subsample) is used instead of shap.kmeans here because
    # k-means centroids land in the geometric centre of the feature space, which
    # for XGB/LGB sits inside the high-confidence disease region even after class
    # balancing — producing base values of ~0.70 instead of ~0.50. shap.sample
    # preserves the actual balanced data distribution, giving base values close
    # to 0.50 for all classifiers. The total background size is k points drawn
    # from the 2*n_per_class balanced pool.
    return shap.sample(X_balanced, k, random_state=42)


# ── Stratified subsampling ────────────────────────────────────────────────────

def subsample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw a stratified subsample of n rows from X, preserving class proportions.

    Used for KernelExplainer classifiers (SVM, QDA, KNN) to limit the
    explained sample count while maintaining the disease/control ratio.
    TreeExplainer classifiers always receive the full pool — this function
    is not called for them.

    If n >= len(X), the full array is returned with original indices.

    The subsample is drawn without replacement from each class independently,
    rounding the per-class count to the nearest integer. The final count may
    differ from n by at most 1 due to rounding.

    Args:
        X:   Feature matrix, shape (n_total, 14).
        y:   Binary label vector, shape (n_total,).
        n:   Target subsample size.
        rng: NumPy Generator for reproducible sampling.

    Returns:
        (X_sub, y_sub, indices) where indices are positions in the original X/y.
    """
    if n >= len(X):
        return X, y, np.arange(len(X))

    classes = np.unique(y)
    selected: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        # Proportion of this class in the original array.
        proportion = len(cls_idx) / len(y)
        n_cls = max(1, round(n * proportion))
        # Cap at available count (cannot draw more than exists).
        n_cls = min(n_cls, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_cls, replace=False)
        selected.append(chosen)

    indices = np.concatenate(selected)
    # shuffle to avoid class-contiguous ordering
    indices = rng.permutation(indices)
    return X[indices], y[indices], indices


# ── Parallel KernelExplainer worker ──────────────────────────────────────────

def _kernel_shap_worker(args: tuple) -> np.ndarray:
    """
    Worker function for parallel KernelExplainer computation.

    Each worker process receives a contiguous batch of rows from X and computes
    SHAP values independently on a separate CPU core. Workers are separate
    processes (not threads) so they bypass Python's GIL and run truly in parallel.

    This function must be at module level (not a lambda or nested function) so
    that multiprocessing can pickle it when dispatching to worker processes.

    In SHAP 0.51.0, shap.sample() returns a plain numpy array (not a DenseData
    object). The background is passed directly as a numpy array and used
    as-is to construct the KernelExplainer in each worker.

    Args:
        args: Tuple of (pipeline_path, background_array, X_batch, nsamples)
              - pipeline_path:     str path to the .joblib pipeline file.
                                   Each worker loads its own copy — joblib files
                                   are read-only and safe to open concurrently.
              - background_array:  np.ndarray of shape (k, 14). Passed directly
                                   to KernelExplainer as background data.
              - X_batch:           np.ndarray of shape (batch_size, 14).
              - nsamples:          int, number of SHAP coalitions per sample.

    Returns:
        np.ndarray of shape (batch_size, 14), class-1 SHAP values for the batch.
    """
    pipeline_path, bg_array, X_batch, nsamples = args

    import warnings as _warnings
    _warnings.filterwarnings('ignore')

    import joblib as _joblib
    import numpy as _np
    import shap as _shap

    _pipeline = _joblib.load(pipeline_path)
    _explainer = _shap.KernelExplainer(_pipeline.predict_proba, bg_array)
    sv = _explainer.shap_values(X_batch, nsamples=nsamples, silent=True)

    # KernelExplainer returns shape (batch_size, 14, 2) for binary classification.
    arr = _np.asarray(sv, dtype=_np.float64)
    if arr.ndim == 3:
        return arr[:, :, 1]
    # Some SHAP versions return a list [class0, class1] or a 2D array directly.
    if isinstance(sv, list):
        return _np.asarray(sv[1], dtype=_np.float64)
    return arr


# ── Core SHAP computation ─────────────────────────────────────────────────────

def _extract_class1_shap(sv: np.ndarray | list) -> np.ndarray:
    """
    Extract class-1 SHAP values from TreeExplainer or KernelExplainer output.

    TreeExplainer (tree_path_dependent for RF/DT) returns a 3D array (n, 14, 2)
    or a list of two arrays [(n, 14), (n, 14)]. Class 1 is at index [1].

    TreeExplainer (interventional for XGB/LGB) returns a 2D array (n, 14).
    This is already the single-class probability-scale output for class 1.

    KernelExplainer returns a 3D array (n, 14, 2) for binary classification.

    Args:
        sv: Raw shap_values output from the explainer.

    Returns:
        2D float64 array of shape (n_samples, 14) for class 1.
    """
    if isinstance(sv, list):
        # TreeExplainer list format: [class_0_array, class_1_array]
        return sv[1].astype(np.float64)
    arr = np.asarray(sv, dtype=np.float64)
    if arr.ndim == 3:
        # Shape (n, 14, 2): last axis is class index
        return arr[:, :, 1]
    if arr.ndim == 2:
        # Shape (n, 14): already class-1 (XGB/LGB interventional or kernel)
        return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def _extract_base_value(expected_value: Any) -> float:
    """
    Extract the class-1 base value from TreeExplainer or KernelExplainer.

    RF/DT tree_path_dependent: returns a length-2 array [base_0, base_1].
    XGB/LGB interventional: returns a scalar (class-1 probability).
    KernelExplainer: returns a length-2 array [base_0, base_1].
    """
    if isinstance(expected_value, (list, np.ndarray)):
        ev = np.asarray(expected_value, dtype=np.float64).ravel()
        if len(ev) == 1:
            return float(ev[0])
        return float(ev[1])  # class 1
    return float(expected_value)


def compute_shap_values(
    clf_name: str,
    pipeline: Any,
    X: np.ndarray,
    y: np.ndarray,
    background: Any,
    sample_indices: np.ndarray,
    pipeline_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute SHAP values for one (classifier, pool) combination.

    Selects the appropriate SHAP algorithm based on classifier type (see module
    docstring), computes class-1 probability-scale SHAP values, verifies the
    Shapley completeness axiom, and returns a result dict suitable for both
    δj computation and .npz serialisation.

    For tree classifiers (RF, DT, XGB, LGB), X is the full pool — no
    subsampling is applied inside this function for these classifiers.
    For kernel classifiers (SVM, QDA, KNN), X and y should already be the
    stratified subsample produced by subsample_stratified(); this function
    does not subsample internally.

    Completeness verification:
      - TreeExplainer (tree_path_dependent): asserts max error < 1e-4.
        RF/DT return exact probability-scale values; error is machine precision.
      - TreeExplainer (interventional): asserts max error < 1e-4.
        Interventional mode with probability output is exact up to floating point.
      - KernelExplainer: reports median error (approximate by design with finite
        nsamples). Caller should verify median < 0.02.

    Parallelisation (kernel classifiers only):
      KernelExplainer iterates over each sample in a Python loop and cannot be
      parallelised internally. This function splits X into per-core batches and
      dispatches them to a multiprocessing.Pool, achieving near-linear speedup
      on multi-core machines. Each worker loads its own pipeline copy from
      pipeline_path (required for kernel classifiers).

    Args:
        clf_name:       One of 'rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm'.
        pipeline:       Fitted ImbPipeline loaded from joblib. Used directly for
                        tree classifiers; workers load their own copy for kernel.
        X:              Feature matrix to explain, shape (n, 14). For tree
                        classifiers this is the full pool; for kernel classifiers
                        this is the stratified subsample.
        y:              True labels, shape (n,). Stored in .npz for downstream
                        waterfall and misclassification analysis.
        background:     numpy array from get_background_data(), shape (k, 14).
                        Used as data= for interventional TreeExplainer and as
                        background for KernelExplainer.
        sample_indices: Original row indices of X within the full pool. For
                        tree classifiers this is np.arange(n); for kernel
                        classifiers it is the indices returned by
                        subsample_stratified().
        pipeline_path:  Path to the .joblib file for this pipeline. Required for
                        kernel classifiers (each worker process loads its own copy).
                        Unused for tree classifiers and may be None.

    Returns:
        Dict with keys:
          'shap_values'        — np.ndarray, shape (n, 14), float64
          'base_value'         — float, expected model output for class 1
          'X_explained'        — np.ndarray, shape (n, 14), the feature values
                                 that were explained (same as X)
          'y_true'             — np.ndarray, shape (n,), true labels
          'sample_indices'     — np.ndarray, shape (n,), original indices
          'explainer_type'     — str, one of 'tree_tpd', 'tree_int', 'kernel'
          'completeness_error' — float, max absolute error (tree) or median
                                 absolute error (kernel)
    """
    config = get_shap_config(clf_name)
    explainer_type = config['explainer']

    clf = pipeline.named_steps['clf']

    if explainer_type == 'tree_tpd':
        # RF and DT: tree_path_dependent produces exact probability-scale output.
        # No background data argument needed — the tree structure encodes the
        # marginal distributions internally.
        explainer = shap.TreeExplainer(clf)
        sv_raw = explainer.shap_values(X)
        base_value = _extract_base_value(explainer.expected_value)
        shap_vals = _extract_class1_shap(sv_raw)

        predicted = pipeline.predict_proba(X)[:, 1]
        reconstructed = base_value + shap_vals.sum(axis=1)
        err = float(np.max(np.abs(reconstructed - predicted)))
        assert err < 1e-4, (
            f"{clf_name} TreeExplainer completeness error {err:.2e} exceeds 1e-4. "
            f"Verify that RF/DT tree_path_dependent returns probability-scale output."
        )
        completeness_error = err

    elif explainer_type == 'tree_int':
        # XGB and LightGBM: must use interventional + model_output='probability'
        # because tree_path_dependent returns log-odds (unbounded), which would
        # make δj incomparable to RF/DT probability-scale values.
        # LightGBM requires feature_names to suppress the "fitted with feature
        # names but received numpy array" warning that would otherwise appear
        # thousands of times per run.
        # In SHAP 0.51.0, the data= argument for interventional TreeExplainer
        # requires a raw numpy array, not a shap.kmeans DenseData object.
        bg_array = np.asarray(background.data) if hasattr(
            background, 'data') else np.asarray(background)
        explainer = shap.TreeExplainer(
            clf,
            data=bg_array,
            model_output='probability',
            feature_perturbation='interventional',
            feature_names=ALL_FEATURE_COLS,
        )
        sv_raw = explainer.shap_values(X)
        base_value = _extract_base_value(explainer.expected_value)
        shap_vals = _extract_class1_shap(sv_raw)

        predicted = pipeline.predict_proba(X)[:, 1]
        reconstructed = base_value + shap_vals.sum(axis=1)
        err = float(np.max(np.abs(reconstructed - predicted)))
        assert err < 1e-4, (
            f"{clf_name} interventional TreeExplainer completeness error {err:.2e} "
            f"exceeds 1e-4."
        )
        completeness_error = err

    elif explainer_type == 'kernel':
        # SVM, QDA, KNN: KernelExplainer with the full pipeline's predict_proba.
        # Passing pipeline.predict_proba (not clf.predict_proba) ensures SHAP
        # values are in the original 14-dimensional feature space. The StandardScaler
        # inside the pipeline for SVM/KNN is absorbed into the pipeline call — no
        # manual pre-scaling is needed and no post-hoc space correction is required.
        #
        # KernelExplainer iterates over each sample in a single-threaded Python
        # loop, so adding CPU cores via n_jobs has no effect on the core loop.
        # We parallelise by splitting X into per-core batches and dispatching
        # to a multiprocessing pool. Each worker loads its own pipeline copy
        # (the loaded pipeline object is not safely shareable across processes).
        # On Modal with cpu=16 this uses 15 workers; locally it uses cpu_count-1.
        if pipeline_path is None:
            raise ValueError(
                f"pipeline_path is required for kernel classifier '{clf_name}' "
                f"(parallel workers need it to load their own pipeline copy)."
            )
        bg_array = np.asarray(background.data) if hasattr(
            background, 'data') else np.asarray(background)

        # Use fork context: safe on Linux (Modal and WSL), avoids re-importing
        # the full module stack in each worker. Spawn would also work but is slower.
        n_workers = max(1, (os.cpu_count() or 4) - 1)
        n_workers = min(n_workers, len(X))   # no more workers than samples
        batches = np.array_split(X, n_workers)
        worker_args = [
            (str(pipeline_path), bg_array, batch, config['nsamples'])
            for batch in batches
        ]

        ctx = mp.get_context('fork')
        with ctx.Pool(processes=n_workers) as pool:
            batch_results = pool.map(_kernel_shap_worker, worker_args)

        shap_vals = np.vstack(batch_results)

        # Base value: compute once from a single-sample explainer on the main
        # process — the expected_value is a property of the background, not X.
        base_value = float(pipeline.predict_proba(bg_array)[:, 1].mean())

        # KernelExplainer is approximate: report median completeness error.
        predicted = pipeline.predict_proba(X)[:, 1]
        reconstructed = base_value + shap_vals.sum(axis=1)
        completeness_error = float(
            np.median(np.abs(reconstructed - predicted)))

    else:
        raise ValueError(f"Unknown explainer_type '{explainer_type}'")

    return {
        'shap_values':        shap_vals,
        'base_value':         base_value,
        'X_explained':        X,
        'y_true':             y,
        'sample_indices':     sample_indices,
        'explainer_type':     explainer_type,
        'completeness_error': completeness_error,
    }


# ── δj computation ────────────────────────────────────────────────────────────

def compute_delta_j(
    shap_within: np.ndarray,
    shap_cross: np.ndarray,
) -> dict[str, Any]:
    """
    Compute the δj transfer-failure metric for one (classifier, direction) pair.

    δj = |mean(|φj_within|) − mean(|φj_cross|)|

    where mean is over all explained strides and |·| is absolute value.
    Using mean absolute SHAP values (rather than signed means) ensures that
    positive and negative contributions do not cancel, correctly measuring
    how much the model relies on each feature in each setting.

    The normalised δj scales the shift by the within-condition importance:

      δj_norm = δj / mean(|φj_within|)

    This converts the raw shift to a fractional change relative to baseline
    importance, enabling cross-classifier comparison across different SHAP
    magnitude scales (RF vs. QDA vs. SVM all have different absolute ranges).
    δj_norm is capped at 10.0 to bound the "emerged feature" case where
    mean(|φj_within|) ≈ 0. Features hitting the cap are flagged as "emerged."

    Reference: Lundberg & Lee (2017) — SHAP framework.
               Xiang et al. (2025) — baseline for "within-condition importance."

    Args:
        shap_within: SHAP values for within-condition pool, shape (n_within, 14).
                     From compute_shap_values() on the source pool.
        shap_cross:  SHAP values for cross-condition pool, shape (n_cross, 14).
                     From compute_shap_values() on the target pool.

    Returns:
        Dict with keys:
          'mean_abs_within'   — np.ndarray shape (14,): mean(|φj|) on source pool
          'mean_abs_cross'    — np.ndarray shape (14,): mean(|φj|) on target pool
          'delta_j'           — np.ndarray shape (14,): raw δj, non-negative
          'delta_j_normalized'— np.ndarray shape (14,): δj_norm, capped at 10.0
          'emerged_features'  — list[int]: feature indices where
                                mean_abs_within < 1e-3 AND delta_j > 0
    """
    mean_abs_within = np.mean(np.abs(shap_within), axis=0)
    mean_abs_cross = np.mean(np.abs(shap_cross),  axis=0)
    delta_j = np.abs(mean_abs_within - mean_abs_cross)

    # Normalise by within-condition importance; floor at eps to avoid division by zero.
    delta_j_norm = np.minimum(
        delta_j / np.maximum(mean_abs_within, _EPS),
        _DELTA_J_NORM_CAP,
    )

    # Emerged features: within-condition importance was negligible but cross-condition
    # importance is nonzero — the model began relying on a feature it previously ignored.
    emerged = [
        int(j)
        for j in range(len(mean_abs_within))
        if mean_abs_within[j] < _EMERGED_THRESHOLD and delta_j[j] > 0
    ]

    return {
        'mean_abs_within':    mean_abs_within,
        'mean_abs_cross':     mean_abs_cross,
        'delta_j':            delta_j,
        'delta_j_normalized': delta_j_norm,
        'emerged_features':   emerged,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_shap_npz(
    path: str | Path,
    shap_result: dict[str, Any],
) -> None:
    """
    Save the output of compute_shap_values() to a compressed .npz file.

    Arrays are stored as float32 to reduce file size (~2× compression vs float64).
    The base_value is stored as a float64 scalar wrapped in a 0-d ndarray so
    np.load retrieves it uniformly with the other keys via loaded['base_value'].item().

    File naming convention (enforced by caller, not this function):
      experiments/shap/{source_cond}_{clf_name}_{pool_type}.npz
      where pool_type is 'within' or 'cross_{target_cond}'.

    Args:
        path:        Destination file path. Parent directory must exist.
        shap_result: Dict returned by compute_shap_values().
    """
    np.savez_compressed(
        str(path),
        shap_values=shap_result['shap_values'].astype(np.float32),
        base_value=np.float64(shap_result['base_value']),
        X_explained=shap_result['X_explained'].astype(np.float32),
        y_true=shap_result['y_true'].astype(np.int32),
        sample_indices=shap_result['sample_indices'].astype(np.int64),
    )


# ── Direction-level orchestration ─────────────────────────────────────────────

def run_shap_for_direction(
    source_condition: str,
    target_condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    control_b: list[str],
    models_dir: str | Path,
    shap_dir: str | Path,
    reuse_within: bool = True,
) -> dict[str, Any]:
    """
    Compute SHAP values and δj for all 7 classifiers for one transfer direction.

    This function handles one (source_condition → target_condition) direction.
    It is the unit of work for parallelisation in the Modal runner: each
    direction can be processed in an independent container.

    Within-condition SHAP (source_pool) and cross-condition SHAP (target_pool)
    are computed for each classifier using the fitted source model. The within-
    condition .npz file is written once per (source_condition, clf_name) pair
    and reused across both target directions sharing the same source — the file
    path encodes only the source condition, not the target. If the within-
    condition file already exists when this function runs, the stored arrays are
    loaded rather than recomputed, saving significant time on XGB/LGB.

    Background data (class-balanced shap.kmeans, k=100) is computed once from the
    source pool and shared across all 7 classifiers, ensuring a consistent marginal
    reference distribution for both interventional TreeExplainer and KernelExplainer.

    RNG seeding: a direction-specific seed is derived deterministically from the
    source and target condition names so that each (source, target) pair produces
    identical subsamples regardless of execution order. This means pd→hd and pd→als
    can run in parallel on separate Modal containers and produce the same results
    as if run sequentially. The caller does not need to manage an rng parameter.

    Computational profile per direction:
      - RF, DT (tree_path_dependent, full pool ~5–7 K strides): seconds each
      - XGB, LGB (interventional, full pool): tens of seconds each
      - SVM, QDA, KNN (KernelExplainer, 1 K stratified strides, 1 024 nsamples):
        tens of minutes each. On Modal with per-direction containers these run
        concurrently across directions but serially within a single container.

    Args:
        source_condition: One of 'pd', 'hd', 'als'. Source classifier condition.
        target_condition: One of 'pd', 'hd', 'als'. Must differ from source.
        df:               Full feature DataFrame from gait_features.csv.
        control_a:        Control Group A subject IDs (source pool only).
        control_b:        Control Group B subject IDs (target pool only).
        models_dir:       Directory containing fitted pipelines as
                          {source_condition}_{clf_name}.joblib.
        shap_dir:         Directory to write .npz files. Created if absent.
        reuse_within:     If True (default), load the within-condition .npz from
                          disk if it already exists — safe for sequential local runs
                          where each source condition is processed once at a time.
                          Set to False in the Modal parallel runner to avoid the race
                          condition where two containers sharing the same source
                          condition could simultaneously write and read the same file.

    Returns:
        Dict keyed by classifier name, each value containing:
          'explainer_type'            — str
          'n_samples_within'          — int
          'n_samples_cross'           — int
          'base_value_within'         — float
          'base_value_cross'          — float
          'completeness_error_within' — float
          'completeness_error_cross'  — float
          'mean_abs_within'           — list[float], length 14
          'mean_abs_cross'            — list[float], length 14
          'delta_j'                   — list[float], length 14
          'delta_j_normalized'        — list[float], length 14
          'emerged_features'          — list[int]
    """
    models_dir = Path(models_dir)
    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic per-direction RNG: seed derived from condition names so each
    # (source, target) pair produces identical subsamples regardless of which
    # other directions were run before it or in parallel.
    _CONDITION_SEEDS = {'pd': 0, 'hd': 1, 'als': 2}
    direction_seed = 42 + \
        _CONDITION_SEEDS[source_condition] * 10 + \
        _CONDITION_SEEDS[target_condition]
    local_rng = np.random.default_rng(direction_seed)

    # ── Construct pools ───────────────────────────────────────────────────────
    source_pool = df.filter(
        (pl.col('condition') == source_condition) |
        pl.col('subject_id').is_in(control_a)
    )
    target_pool = df.filter(
        (pl.col('condition') == target_condition) |
        pl.col('subject_id').is_in(control_b)
    )

    X_source = source_pool.select(
        ALL_FEATURE_COLS).to_numpy().astype(np.float64)
    y_source = source_pool['label'].to_numpy().astype(int)
    X_target = target_pool.select(
        ALL_FEATURE_COLS).to_numpy().astype(np.float64)
    y_target = target_pool['label'].to_numpy().astype(int)

    # Class-balanced background computed once from source pool, shared across all 7
    # classifiers. local_rng seeds the balanced sampling deterministically per direction.
    background = get_background_data(X_source, y_source, k=100, rng=local_rng)

    clf_names = ['rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm']
    direction_results: dict[str, Any] = {}

    for clf_name in clf_names:
        model_path = models_dir / f'{source_condition}_{clf_name}.joblib'
        pipeline = joblib.load(model_path)

        config = get_shap_config(clf_name)
        is_kernel = config['explainer'] == 'kernel'

        # ── Within-condition SHAP ─────────────────────────────────────────────
        # File path encodes source condition only (not direction), so it is
        # shared across both target directions for this source model.
        within_npz_path = shap_dir / \
            f'{source_condition}_{clf_name}_within.npz'

        if reuse_within and within_npz_path.exists():
            # Load pre-computed within-condition SHAP to avoid redundant computation
            # when this source condition is used for its second target direction.
            # Disabled (reuse_within=False) in the Modal parallel runner to eliminate
            # the race condition where two containers sharing the same source condition
            # could simultaneously write and read the same within-condition file.
            loaded_w = np.load(within_npz_path)
            within_shap = loaded_w['shap_values'].astype(np.float64)
            base_within = float(loaded_w['base_value'])
            X_within_loaded = loaded_w['X_explained'].astype(np.float64)
            predicted_w = pipeline.predict_proba(X_within_loaded)[:, 1]
            reconstructed_w = base_within + within_shap.sum(axis=1)
            errors_w = np.abs(reconstructed_w - predicted_w)
            completeness_within = (
                float(np.median(errors_w)) if is_kernel
                else float(np.max(errors_w))
            )
            n_within = len(within_shap)
        else:
            if is_kernel:
                X_w, y_w, idx_w = subsample_stratified(
                    X_source, y_source, config['n_explained'], local_rng
                )
            else:
                X_w = X_source
                y_w = y_source
                idx_w = np.arange(len(y_source))

            result_within = compute_shap_values(
                clf_name, pipeline, X_w, y_w, background, idx_w,
                pipeline_path=model_path,
            )
            save_shap_npz(within_npz_path, result_within)
            within_shap = result_within['shap_values']
            base_within = result_within['base_value']
            completeness_within = result_within['completeness_error']
            n_within = len(within_shap)

        # ── Cross-condition SHAP ──────────────────────────────────────────────
        cross_npz_path = (
            shap_dir /
            f'{source_condition}_{clf_name}_cross_{target_condition}.npz'
        )

        if is_kernel:
            X_c, y_c, idx_c = subsample_stratified(
                X_target, y_target, config['n_explained'], local_rng
            )
        else:
            X_c = X_target
            y_c = y_target
            idx_c = np.arange(len(y_target))

        result_cross = compute_shap_values(
            clf_name, pipeline, X_c, y_c, background, idx_c,
            pipeline_path=model_path,
        )
        save_shap_npz(cross_npz_path, result_cross)
        cross_shap = result_cross['shap_values']
        base_cross = result_cross['base_value']
        completeness_cross = result_cross['completeness_error']
        n_cross = len(cross_shap)

        # ── δj ───────────────────────────────────────────────────────────────
        dj = compute_delta_j(within_shap, cross_shap)

        direction_results[clf_name] = {
            'explainer_type':            config['explainer'],
            'n_samples_within':          n_within,
            'n_samples_cross':           n_cross,
            'base_value_within':         base_within,
            'base_value_cross':          base_cross,
            'completeness_error_within': completeness_within,
            'completeness_error_cross':  completeness_cross,
            'mean_abs_within':           dj['mean_abs_within'].tolist(),
            'mean_abs_cross':            dj['mean_abs_cross'].tolist(),
            'delta_j':                   dj['delta_j'].tolist(),
            'delta_j_normalized':        dj['delta_j_normalized'].tolist(),
            'emerged_features':          dj['emerged_features'],
        }

        top3 = np.argsort(dj['delta_j'])[::-1][:3]
        top3_str = ', '.join(
            f'{ALL_FEATURE_COLS[j]}={dj["delta_j"][j]:.4f}'
            for j in top3
        )
        print(
            f'  {source_condition}->{target_condition}  {clf_name:<6}  '
            f'type={config["explainer"]}  '
            f'n_within={n_within}  n_cross={n_cross}  '
            f'completeness_within={completeness_within:.2e}  '
            f'completeness_cross={completeness_cross:.2e}  '
            f'top_delta_j: [{top3_str}]',
            flush=True,
        )

    return direction_results
