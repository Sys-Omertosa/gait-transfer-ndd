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
               original feature space regardless of internal scaling.

SMOTE convention: SMOTE is part of the ImbPipeline and is skipped at predict time
(ImbPipeline does not run samplers during transform/predict). Passing the full
pipeline predict_proba to KernelExplainer is therefore equivalent to passing the
underlying classifier predict_proba on pre-scaled data, with the correct output.
In the within-condition source pools, SMOTE augments the minority control class
defined by the 8 Control Group A subjects rather than the disease class. Those
synthetic control strides are created during source-model training only; SHAP
evaluation always runs on real source or target strides, and the disjoint
Control Group B keeps healthy transfer evaluation independent of any
training-time augmentation.

Background data: a class-balanced source-specific random sample (k=100)
computed once per source condition and reused across both target directions.
Balancing keeps disease and control represented equally in the background and
makes the base value more comparable across classifier families.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import joblib
import numpy as np
import polars as pl
import shap
from joblib import Parallel, delayed as jl_delayed

from features import ALL_FEATURE_COLS, get_feature_cols

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
_STABILITY_N_RESAMPLES = 200


def infer_feature_family(feature_name: str) -> str:
    """Map an individual feature name to a coarser interpretation family."""
    if feature_name in {'cv_stride', 'cv_swing'}:
        return 'variability'
    if feature_name == 'dfa_alpha_stride':
        return 'fractal'
    if 'asymmetry' in feature_name:
        return 'asymmetry'
    if feature_name.endswith('_pct'):
        return 'phase_percentage'
    return 'raw_timing'


def get_feature_families(feature_cols: list[str]) -> dict[str, list[int]]:
    """Group feature indices into interpretation families."""
    families: dict[str, list[int]] = {}
    for idx, feature_name in enumerate(feature_cols):
        families.setdefault(infer_feature_family(feature_name), []).append(idx)
    return families


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

def build_source_background(
    X_source: np.ndarray,
    y_source: np.ndarray,
    k: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Build a class-balanced source-specific background sample for SHAP.

    The v4 design makes the background source-specific rather than
    direction-specific. This eliminates the within-cache reuse bug where the
    same within-source SHAP file could be paired with different backgrounds
    depending on which target direction ran first.

    Args:
        X_source: Feature matrix of the source pool, shape (n_source, 14).
        y_source: Binary label vector, shape (n_source,). Used to balance classes.
        k:        Maximum background size. Default 100.
        random_seed: Deterministic seed used for balanced sampling.

    Returns:
        Dict containing:
          'background': np.ndarray of shape (k, n_features)
          'source_indices': list[int] into the original source pool
          'class_counts': disease/control counts used before final sampling
          'random_seed': the seed used to build this background
    """
    rng = np.random.default_rng(random_seed)

    disease_idx = np.where(y_source == 1)[0]
    control_idx = np.where(y_source == 0)[0]

    n_per_class = min(len(disease_idx), len(control_idx), max(k // 2, 1))
    if n_per_class < 1:
        raise ValueError(
            'SHAP background requires at least one disease row and one control row '
            f'(got disease={len(disease_idx)}, control={len(control_idx)}).'
        )
    sampled_disease = rng.choice(disease_idx, size=n_per_class, replace=False)
    sampled_control = rng.choice(control_idx, size=n_per_class, replace=False)
    selected = rng.permutation(
        np.concatenate([sampled_disease, sampled_control])
    ).astype(int, copy=False)
    return {
        'background': X_source[selected].astype(np.float64, copy=False),
        'source_indices': selected.astype(int).tolist(),
        'class_counts': {
            'disease': int(n_per_class),
            'control': int(n_per_class),
        },
        'eligible_class_rows': {
            'disease': int(len(disease_idx)),
            'control': int(len(control_idx)),
        },
        'requested_background_size': int(k),
        'random_seed': int(random_seed),
    }


def get_background_data(
    X_source: np.ndarray,
    y_source: np.ndarray,
    k: int = 100,
    random_seed: int = 42,
) -> np.ndarray:
    """Backward-compatible wrapper returning only the background array."""
    return build_source_background(
        X_source,
        y_source,
        k=k,
        random_seed=random_seed,
    )['background']


def _hash_bytes(payload: bytes) -> str:
    """SHA-256 helper for small in-memory payloads."""
    return hashlib.sha256(payload).hexdigest()


def _stable_sampling_seed(*parts: str) -> int:
    """Stable 32-bit seed derived from logical sampling identifiers."""
    payload = '::'.join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2 ** 32)


def _hash_array(arr: np.ndarray) -> str:
    """Stable SHA-256 hash of an ndarray's raw bytes."""
    contiguous = np.ascontiguousarray(arr)
    return _hash_bytes(contiguous.tobytes())


def _hash_file(path: str | Path) -> str:
    """SHA-256 hash of a file on disk."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f'.{path.name}.tmp')
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    tmp_path = path.with_name(f'.{path.stem}.tmp{path.suffix}')
    np.savez_compressed(str(tmp_path), **arrays)
    tmp_path.replace(path)


def _partition_hash(control_a: list[str], control_b: list[str]) -> str:
    """Stable hash of the control partition used to define the source/target pools."""
    payload = json.dumps(
        {
            'control_A': list(control_a),
            'control_B': list(control_b),
        },
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    return _hash_bytes(payload)


def _load_or_build_background(
    *,
    shap_dir: Path,
    source_condition: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    source_subject_ids: np.ndarray | list[str],
    feature_cols: list[str],
    control_a: list[str],
    control_b: list[str],
    background_size: int = 100,
    random_seed: int = 42,
    protocol_manifest_hash: str | None = None,
    preprocessing_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """
    Persist and reload one source-specific SHAP background per source condition.

    The background is shared across both target directions for a source and is
    therefore safe to pair with the shared within-condition SHAP cache.
    """
    background_path = shap_dir / f'{source_condition}_background.npz'
    metadata_path = shap_dir / f'{source_condition}_background_meta.json'
    expected_partition_hash = _partition_hash(control_a, control_b)
    expected_source_subject_ids = list(
        dict.fromkeys(np.asarray(source_subject_ids).astype(str).tolist())
    )
    expected_source_pool_hash = _hash_bytes(
        np.ascontiguousarray(X_source).tobytes()
        + np.ascontiguousarray(y_source.astype(np.int32)).tobytes()
    )
    expected_n_per_class = min(
        int(np.sum(y_source == 1)),
        int(np.sum(y_source == 0)),
        max(int(background_size) // 2, 1),
    )
    expected_class_counts = {
        'disease': int(expected_n_per_class),
        'control': int(expected_n_per_class),
    }
    expected_background_rows = int(expected_n_per_class * 2)
    expected_shap_version = getattr(shap, '__version__', 'unknown')

    if background_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            with np.load(background_path, allow_pickle=False) as loaded:
                if 'background' not in loaded:
                    raise KeyError('background')
                background = loaded['background'].astype(np.float64)
        except (BadZipFile, OSError, ValueError, KeyError, json.JSONDecodeError):
            background = None
            metadata = None
        if background is not None and metadata is not None and (
            metadata.get('source_condition') == source_condition
            and metadata.get('feature_cols') == feature_cols
            and metadata.get('partition_hash') == expected_partition_hash
            and metadata.get('background_sha256') == _hash_array(background)
            and metadata.get('background_rows_sha256') == _hash_array(background)
            and metadata.get('background_size') == expected_background_rows
            and metadata.get('class_counts') == expected_class_counts
            and metadata.get('source_pool_rows') == int(len(X_source))
            and metadata.get('source_pool_sha256') == expected_source_pool_hash
            and metadata.get('source_subject_ids') == expected_source_subject_ids
            and metadata.get('source_subject_count') == int(len(expected_source_subject_ids))
            and metadata.get('random_seed') == int(random_seed)
            and metadata.get('shap_version') == expected_shap_version
            and metadata.get('protocol_manifest_hash') == protocol_manifest_hash
            and metadata.get('preprocessing_manifest_hash') == preprocessing_manifest_hash
        ):
            return {
                'background': background,
                'source_indices': metadata.get('source_indices', []),
                'class_counts': metadata.get('class_counts', {}),
                'random_seed': int(metadata.get('random_seed', random_seed)),
                'metadata_path': str(metadata_path),
                'background_sha256': metadata['background_sha256'],
                'metadata': metadata,
            }

    background_bundle = build_source_background(
        X_source,
        y_source,
        k=background_size,
        random_seed=random_seed,
    )
    background = np.asarray(background_bundle['background'], dtype=np.float64)
    background_sha256 = _hash_array(background)

    _atomic_savez_compressed(background_path, background=background)
    metadata = {
        'source_condition': source_condition,
        'feature_cols': feature_cols,
        'source_pool_rows': int(len(X_source)),
        'source_subject_ids': expected_source_subject_ids,
        'source_subject_count': int(len(np.unique(np.asarray(source_subject_ids).astype(str)))),
        'source_pool_sha256': expected_source_pool_hash,
        'source_indices': background_bundle['source_indices'],
        'class_counts': background_bundle['class_counts'],
        'eligible_class_rows': background_bundle['eligible_class_rows'],
        'requested_background_size': int(background_size),
        'random_seed': int(background_bundle['random_seed']),
        'partition_hash': expected_partition_hash,
        'background_sha256': background_sha256,
        'background_rows_sha256': background_sha256,
        'background_size': int(background.shape[0]),
        'shap_version': expected_shap_version,
        'protocol_manifest_hash': protocol_manifest_hash,
        'preprocessing_manifest_hash': preprocessing_manifest_hash,
    }
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
    background_bundle['metadata_path'] = str(metadata_path)
    background_bundle['background_sha256'] = background_sha256
    background_bundle['metadata'] = metadata
    return background_bundle


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


def _transform_tree_inputs(
    pipeline: Any,
    X: np.ndarray,
    background: Any,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """
    Align tree-explainer inputs with the fitted classifier's feature space.

    In the v4 protocol, synthetic arms scale before SMOTE for every classifier,
    so tree models may now see scaled inputs. TreeExplainer must therefore
    operate on the same transformed representation the fitted classifier uses.
    """
    clf = pipeline.named_steps['clf']
    scaler = pipeline.named_steps.get('scaler')
    bg_array = np.asarray(background.data) if hasattr(background, 'data') else np.asarray(background)
    X_tree = np.asarray(X, dtype=np.float64)
    bg_tree = np.asarray(bg_array, dtype=np.float64)
    if scaler is not None:
        X_tree = scaler.transform(X_tree).astype(np.float64, copy=False)
        bg_tree = scaler.transform(bg_tree).astype(np.float64, copy=False)
    return clf, X_tree, bg_tree


def compute_shap_values(
    clf_name: str,
    pipeline: Any,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    background: Any,
    sample_indices: np.ndarray,
    feature_cols: list[str],
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
      dispatches them to joblib.Parallel with the loky backend, which uses
      separate worker processes compatible with Modal's gVisor sandbox and WSL2.
      Each worker loads its own pipeline copy from pipeline_path (required for
      kernel classifiers).

    Args:
        clf_name:       One of 'rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm'.
        pipeline:       Fitted ImbPipeline loaded from joblib. Used directly for
                        tree classifiers; workers load their own copy for kernel.
        X:              Feature matrix to explain, shape (n, 14). For tree
                        classifiers this is the full pool; for kernel classifiers
                        this is the stratified subsample.
        y:              True labels, shape (n,). Stored in .npz for downstream
                        waterfall and misclassification analysis.
        subject_ids:    Subject IDs aligned to X and y.
        background:     numpy array from get_background_data(), shape (k, n_features).
                        Used as data= for interventional TreeExplainer and as
                        background for KernelExplainer.
        sample_indices: Original row indices of X within the full pool. For
                        tree classifiers this is np.arange(n); for kernel
                        classifiers it is the indices returned by
                        subsample_stratified().
        feature_cols:   Feature names aligned to the columns of X.
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
          'subject_ids'        — np.ndarray, shape (n,), subject IDs
          'sample_indices'     — np.ndarray, shape (n,), original indices
          'explainer_type'     — str, one of 'tree_tpd', 'tree_int', 'kernel'
          'completeness_error' — float, max absolute error (tree) or median
                                 absolute error (kernel)
    """
    config = get_shap_config(clf_name)
    explainer_type = config['explainer']

    if explainer_type == 'tree_tpd':
        # RF and DT: tree_path_dependent produces exact probability-scale output.
        # No background data argument needed — the tree structure encodes the
        # marginal distributions internally.
        clf, X_tree, _ = _transform_tree_inputs(pipeline, X, background)
        explainer = shap.TreeExplainer(clf)
        sv_raw = explainer.shap_values(X_tree)
        base_value = _extract_base_value(explainer.expected_value)
        shap_vals = _extract_class1_shap(sv_raw)

        predicted = clf.predict_proba(X_tree)[:, 1]
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
        clf, X_tree, bg_array = _transform_tree_inputs(pipeline, X, background)
        explainer = shap.TreeExplainer(
            clf,
            data=bg_array,
            model_output='probability',
            feature_perturbation='interventional',
            feature_names=feature_cols,
        )
        sv_raw = explainer.shap_values(X_tree)
        base_value = _extract_base_value(explainer.expected_value)
        shap_vals = _extract_class1_shap(sv_raw)

        predicted = clf.predict_proba(X_tree)[:, 1]
        reconstructed = base_value + shap_vals.sum(axis=1)
        err = float(np.max(np.abs(reconstructed - predicted)))
        if err >= 0.05:
            raise AssertionError(
                f"{clf_name} interventional TreeExplainer completeness error "
                f"{err:.2e} exceeds 0.05 — this indicates a genuine "
                f"implementation problem (wrong output scale or background)."
            )
        if err >= 1e-4:
            print(
                f"  [{clf_name}] interventional TreeExplainer completeness warning: "
                f"{err:.2e} (expected for lgbm with finite background; "
                f"does not affect δj)",
                flush=True,
            )
        completeness_error = err

    elif explainer_type == 'kernel':
        # SVM, QDA, KNN: KernelExplainer with the full pipeline's predict_proba.
        # Passing pipeline.predict_proba (not clf.predict_proba) ensures SHAP
        # values stay aligned to the original feature columns. The RobustScaler
        # inside the pipeline for SVM/KNN is absorbed into the pipeline call — no
        # manual pre-scaling is needed and no post-hoc space correction is required.
        #
        # KernelExplainer iterates over each sample in a single-threaded Python
        # loop, so adding CPU cores via n_jobs has no effect on the core loop.
        # We parallelise by splitting X into per-core batches and dispatching
        # them via joblib's loky backend. Each worker loads its own pipeline copy
        # (the loaded pipeline object is not safely shareable across processes).
        # On Modal with cpu=16 this uses 15 workers; locally it uses cpu_count-1.
        if pipeline_path is None:
            raise ValueError(
                f"pipeline_path is required for kernel classifier '{clf_name}' "
                f"(parallel workers need it to load their own pipeline copy)."
            )
        bg_array = np.asarray(background.data) if hasattr(
            background, 'data') else np.asarray(background)

        n_workers = max(1, (os.cpu_count() or 4) - 1)
        n_workers = min(n_workers, len(X))   # no more workers than samples
        batches = np.array_split(X, n_workers)
        worker_args = [
            (str(pipeline_path), bg_array, batch, config['nsamples'])
            for batch in batches
        ]

        results_raw = Parallel(
            n_jobs=n_workers,
            backend='loky',
            verbose=0,
        )(jl_delayed(_kernel_shap_worker)(args) for args in worker_args)

        batch_results = [np.asarray(r, dtype=np.float64) for r in results_raw]
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
        'subject_ids':        subject_ids,
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


def compute_family_delta_j(
    *,
    feature_cols: list[str],
    mean_abs_within: np.ndarray,
    mean_abs_cross: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-feature SHAP reliance to coarser interpretation families."""
    family_map = get_feature_families(feature_cols)
    family_summary: dict[str, dict[str, Any]] = {}
    for family_name, idxs in family_map.items():
        within_sum = float(np.sum(mean_abs_within[idxs]))
        cross_sum = float(np.sum(mean_abs_cross[idxs]))
        total_movement = float(np.sum(np.abs(mean_abs_within[idxs] - mean_abs_cross[idxs])))
        family_summary[family_name] = {
            'features': [feature_cols[idx] for idx in idxs],
            'mean_abs_within_sum': round(within_sum, 6),
            'mean_abs_cross_sum': round(cross_sum, 6),
            'net_shift': round(abs(within_sum - cross_sum), 6),
            'total_movement': round(total_movement, 6),
            'delta_j_sum': round(abs(within_sum - cross_sum), 6),
        }
    return family_summary


def compute_delta_j_stability(
    *,
    shap_within: np.ndarray,
    shap_cross: np.ndarray,
    subject_ids_within: np.ndarray,
    subject_ids_cross: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = _STABILITY_N_RESAMPLES,
) -> dict[str, Any]:
    """
    Subject-bootstrap stability summary for δj without recomputing SHAP values.

    Subjects are resampled with replacement within the source and target pools
    separately, preserving the within-subject clustering of strides.
    """
    n_features = shap_within.shape[1]
    top1_counts = np.zeros(n_features, dtype=int)
    top3_counts = np.zeros(n_features, dtype=int)
    delta_boot = np.empty((n_resamples, n_features), dtype=np.float64)

    within_subject_ids = np.asarray(subject_ids_within)
    cross_subject_ids = np.asarray(subject_ids_cross)
    unique_within = list(dict.fromkeys(within_subject_ids.tolist()))
    unique_cross = list(dict.fromkeys(cross_subject_ids.tolist()))
    within_rows = {s: np.where(within_subject_ids == s)[0] for s in unique_within}
    cross_rows = {s: np.where(cross_subject_ids == s)[0] for s in unique_cross}

    for resample_idx in range(n_resamples):
        chosen_within = rng.choice(unique_within, size=len(unique_within), replace=True)
        chosen_cross = rng.choice(unique_cross, size=len(unique_cross), replace=True)
        idx_within = np.concatenate([within_rows[s] for s in chosen_within])
        idx_cross = np.concatenate([cross_rows[s] for s in chosen_cross])

        mean_abs_within = np.mean(np.abs(shap_within[idx_within]), axis=0)
        mean_abs_cross = np.mean(np.abs(shap_cross[idx_cross]), axis=0)
        delta = np.abs(mean_abs_within - mean_abs_cross)
        delta_boot[resample_idx] = delta

        rank_desc = np.argsort(delta)[::-1]
        top1_counts[rank_desc[0]] += 1
        top3_counts[rank_desc[:3]] += 1

    return {
        'n_resamples': n_resamples,
        'delta_j_ci_lower': np.percentile(delta_boot, 2.5, axis=0).round(6).tolist(),
        'delta_j_ci_upper': np.percentile(delta_boot, 97.5, axis=0).round(6).tolist(),
        'top1_frequency': (top1_counts / n_resamples).round(6).tolist(),
        'top3_frequency': (top3_counts / n_resamples).round(6).tolist(),
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
      <shap_dir>/{source_cond}_{clf_name}_{pool_type}.npz
      where pool_type is 'within' or 'cross_{target_cond}'.

    Args:
        path:        Destination file path. Parent directory must exist.
        shap_result: Dict returned by compute_shap_values().
    """
    _atomic_savez_compressed(
        Path(path),
        shap_values=shap_result['shap_values'].astype(np.float32),
        base_value=np.float64(shap_result['base_value']),
        X_explained=shap_result['X_explained'].astype(np.float32),
        y_true=shap_result['y_true'].astype(np.int32),
        subject_ids=shap_result['subject_ids'].astype(str),
        sample_indices=shap_result['sample_indices'].astype(np.int64),
    )


def _save_shap_artifact_with_metadata(
    *,
    npz_path: Path,
    metadata_path: Path,
    shap_result: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[str, str]:
    """Persist one SHAP cache plus a JSON sidecar and return both SHA-256 hashes."""
    save_shap_npz(npz_path, shap_result)
    npz_sha = _hash_file(npz_path)
    payload = dict(metadata)
    payload['npz_sha256'] = npz_sha
    _atomic_write_text(metadata_path, json.dumps(payload, indent=2))
    meta_sha = _hash_file(metadata_path)
    return npz_sha, meta_sha


def _load_validated_shap_artifact(
    *,
    npz_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Load a cached SHAP artifact only when its sidecar metadata matches expectations."""
    if not npz_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None
    for key, expected_value in expected_metadata.items():
        if expected_value is None:
            continue
        if metadata.get(key) != expected_value:
            return None
    if metadata.get('npz_sha256') != _hash_file(npz_path):
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            required_keys = {
                'shap_values',
                'base_value',
                'X_explained',
                'y_true',
                'subject_ids',
                'sample_indices',
            }
            if not required_keys.issubset(set(loaded.files)):
                return None
            return {
                'metadata': metadata,
                'shap_values': loaded['shap_values'].astype(np.float64),
                'base_value': float(loaded['base_value']),
                'X_explained': loaded['X_explained'].astype(np.float64),
                'y_true': loaded['y_true'].astype(np.int64),
                'subject_ids': loaded['subject_ids'].astype(str),
                'sample_indices': loaded['sample_indices'].astype(np.int64),
            }
    except (BadZipFile, OSError, ValueError, KeyError):
        return None


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
    feature_cols: list[str] | None = None,
    feature_set_version: str = 'v4',
    stability_n_resamples: int = _STABILITY_N_RESAMPLES,
    protocol_manifest_hash: str | None = None,
    preprocessing_manifest_hash: str | None = None,
    downstream_execution_manifest_hash: str | None = None,
    downstream_execution_id: str | None = None,
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

    Background data is computed once per source condition, persisted under shap_dir,
    and reused across both target directions. Kernel-explainer subsampling remains
    direction-specific so the cross-pool explained rows are still deterministic per
    direction while the within-condition background stays source-intrinsic.

    Computational profile per direction:
      - RF, DT (tree_path_dependent, full pool ~5–7 K strides): seconds each
      - XGB, LGB (interventional, full pool): tens of seconds each
      - SVM, QDA, KNN (KernelExplainer, 1 K stratified strides, 1 024 nsamples):
        tens of minutes each. On Modal with per-direction containers these run
        concurrently across directions but serially within a single container.

    Args:
        source_condition: One of 'pd', 'hd', 'als'. Source classifier condition.
        target_condition: One of 'pd', 'hd', 'als'. Must differ from source.
        df:               Full feature DataFrame for the active experiment version.
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

    # Deterministic sampling seeds are derived independently for each logical
    # draw so cache existence and direction order cannot perturb KernelExplainer
    # explained-row sampling.
    source_seed = _stable_sampling_seed('background', source_condition)

    # ── Construct pools ───────────────────────────────────────────────────────
    source_pool = df.filter(
        (pl.col('condition') == source_condition) |
        pl.col('subject_id').is_in(control_a)
    )
    target_pool = df.filter(
        (pl.col('condition') == target_condition) |
        pl.col('subject_id').is_in(control_b)
    )

    selected_feature_cols = (
        list(feature_cols) if feature_cols is not None
        else get_feature_cols(feature_set_version)
    )
    X_source = source_pool.select(
        selected_feature_cols).to_numpy().astype(np.float64)
    y_source = source_pool['label'].to_numpy().astype(int)
    source_subject_ids = source_pool['subject_id'].to_numpy()
    X_target = target_pool.select(
        selected_feature_cols).to_numpy().astype(np.float64)
    y_target = target_pool['label'].to_numpy().astype(int)
    target_subject_ids = target_pool['subject_id'].to_numpy()
    source_pool_sha256 = _hash_bytes(
        np.ascontiguousarray(X_source).tobytes()
        + np.ascontiguousarray(y_source.astype(np.int32)).tobytes()
    )
    target_pool_sha256 = _hash_bytes(
        np.ascontiguousarray(X_target).tobytes()
        + np.ascontiguousarray(y_target.astype(np.int32)).tobytes()
    )

    # Source-specific class-balanced background persisted once per source.
    background_bundle = _load_or_build_background(
        shap_dir=shap_dir,
        source_condition=source_condition,
        X_source=X_source,
        y_source=y_source,
        source_subject_ids=source_subject_ids,
        feature_cols=selected_feature_cols,
        control_a=control_a,
        control_b=control_b,
        background_size=100,
        random_seed=source_seed,
        protocol_manifest_hash=protocol_manifest_hash,
        preprocessing_manifest_hash=preprocessing_manifest_hash,
    )
    background = background_bundle['background']

    clf_names = ['rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm']
    clf_seed_offset = {name: idx for idx, name in enumerate(clf_names)}
    direction_results: dict[str, Any] = {}

    for clf_name in clf_names:
        model_path = models_dir / f'{source_condition}_{clf_name}.joblib'
        pipeline = joblib.load(model_path)
        model_sha256 = _hash_file(model_path)

        config = get_shap_config(clf_name)
        is_kernel = config['explainer'] == 'kernel'
        within_sampling_seed = _stable_sampling_seed(
            'within_explained_rows',
            source_condition,
            clf_name,
        )
        cross_sampling_seed = _stable_sampling_seed(
            'cross_explained_rows',
            source_condition,
            target_condition,
            clf_name,
        )
        if is_kernel:
            _, _, expected_idx_w = subsample_stratified(
                X_source,
                y_source,
                config['n_explained'],
                np.random.default_rng(within_sampling_seed),
            )
        else:
            expected_idx_w = np.arange(len(y_source), dtype=int)

        # ── Within-condition SHAP ─────────────────────────────────────────────
        # File path encodes source condition only (not direction), so it is
        # shared across both target directions for this source model.
        within_npz_path = shap_dir / f'{source_condition}_{clf_name}_within.npz'
        within_meta_path = shap_dir / f'{source_condition}_{clf_name}_within.meta.json'

        expected_within_metadata = {
            'source_condition': source_condition,
            'target_condition': None,
            'pool_role': 'within',
            'classifier': clf_name,
            'feature_cols': selected_feature_cols,
            'model_sha256': model_sha256,
            'background_sha256': background_bundle.get('background_sha256'),
            'source_pool_sha256': source_pool_sha256,
            'source_pool_rows': int(len(X_source)),
            'source_subject_ids': list(dict.fromkeys(np.asarray(source_subject_ids).astype(str).tolist())),
            'partition_hash': background_bundle['metadata']['partition_hash'],
            'background_size': background_bundle['metadata'].get('background_size'),
            'class_counts': background_bundle['metadata'].get('class_counts'),
            'explainer_type': config['explainer'],
            'shap_version': getattr(shap, '__version__', 'unknown'),
            'protocol_manifest_hash': protocol_manifest_hash,
            'preprocessing_manifest_hash': preprocessing_manifest_hash,
            'downstream_execution_manifest_hash': downstream_execution_manifest_hash,
            'downstream_execution_id': downstream_execution_id,
            'sampling_seed': int(within_sampling_seed),
            'sample_indices_sha256': _hash_array(expected_idx_w.astype(np.int64)),
        }

        loaded_within = None
        if reuse_within:
            loaded_within = _load_validated_shap_artifact(
                npz_path=within_npz_path,
                metadata_path=within_meta_path,
                expected_metadata=expected_within_metadata,
            )

        if loaded_within is not None:
            # Load pre-computed within-condition SHAP to avoid redundant computation
            # when this source condition is used for its second target direction.
            # Disabled (reuse_within=False) in the Modal parallel runner to eliminate
            # the race condition where two containers sharing the same source condition
            # could simultaneously write and read the same within-condition file.
            within_shap = loaded_within['shap_values']
            base_within = loaded_within['base_value']
            X_within_loaded = loaded_within['X_explained']
            within_subject_ids = loaded_within['subject_ids']
            predicted_w = pipeline.predict_proba(X_within_loaded)[:, 1]
            reconstructed_w = base_within + within_shap.sum(axis=1)
            errors_w = np.abs(reconstructed_w - predicted_w)
            completeness_within = (
                float(np.median(errors_w)) if is_kernel
                else float(np.max(errors_w))
            )
            n_within = len(within_shap)
            within_npz_sha = loaded_within['metadata']['npz_sha256']
            within_meta_sha = _hash_file(within_meta_path)
        else:
            if is_kernel:
                within_rng = np.random.default_rng(within_sampling_seed)
                X_w, y_w, idx_w = subsample_stratified(
                    X_source, y_source, config['n_explained'], within_rng
                )
                subject_ids_w = source_subject_ids[idx_w]
            else:
                X_w = X_source
                y_w = y_source
                idx_w = np.arange(len(y_source))
                subject_ids_w = source_subject_ids

            result_within = compute_shap_values(
                clf_name, pipeline, X_w, y_w, subject_ids_w, background, idx_w,
                selected_feature_cols,
                pipeline_path=model_path,
            )
            within_metadata = {
                **expected_within_metadata,
                'source_subject_ids': list(dict.fromkeys(np.asarray(source_subject_ids).astype(str).tolist())),
                'sample_indices': result_within['sample_indices'].astype(int).tolist(),
                'sample_indices_sha256': _hash_array(result_within['sample_indices']),
                'subject_ids': result_within['subject_ids'].astype(str).tolist(),
                'explained_row_count': int(len(result_within['shap_values'])),
                'background_metadata_path': background_bundle.get('metadata_path'),
                'background_metadata_sha256': _hash_file(background_bundle['metadata_path']),
                'sampling_seed': int(within_sampling_seed),
                'kernel_nsamples': config.get('nsamples'),
                'preprocessing_manifest_hash': preprocessing_manifest_hash,
                'downstream_execution_manifest_hash': downstream_execution_manifest_hash,
                'downstream_execution_id': downstream_execution_id,
                'source_pool_rows': int(len(X_source)),
                'background_size': background_bundle['metadata'].get('background_size'),
                'class_counts': background_bundle['metadata'].get('class_counts'),
            }
            within_npz_sha, within_meta_sha = _save_shap_artifact_with_metadata(
                npz_path=within_npz_path,
                metadata_path=within_meta_path,
                shap_result=result_within,
                metadata=within_metadata,
            )
            within_shap = result_within['shap_values']
            base_within = result_within['base_value']
            completeness_within = result_within['completeness_error']
            within_subject_ids = result_within['subject_ids']
            n_within = len(within_shap)

        # ── Cross-condition SHAP ──────────────────────────────────────────────
        cross_npz_path = (
            shap_dir /
            f'{source_condition}_{clf_name}_cross_{target_condition}.npz'
        )
        cross_meta_path = (
            shap_dir /
            f'{source_condition}_{clf_name}_cross_{target_condition}.meta.json'
        )

        if is_kernel:
            cross_rng = np.random.default_rng(cross_sampling_seed)
            X_c, y_c, idx_c = subsample_stratified(
                X_target, y_target, config['n_explained'], cross_rng
            )
            subject_ids_c = target_subject_ids[idx_c]
        else:
            X_c = X_target
            y_c = y_target
            idx_c = np.arange(len(y_target))
            subject_ids_c = target_subject_ids

        result_cross = compute_shap_values(
            clf_name, pipeline, X_c, y_c, subject_ids_c, background, idx_c,
            selected_feature_cols,
            pipeline_path=model_path,
        )
        cross_metadata = {
            'source_condition': source_condition,
            'target_condition': target_condition,
            'pool_role': 'cross',
            'classifier': clf_name,
            'feature_cols': selected_feature_cols,
            'model_sha256': model_sha256,
            'background_sha256': background_bundle.get('background_sha256'),
            'source_pool_sha256': source_pool_sha256,
            'source_pool_rows': int(len(X_source)),
            'target_pool_sha256': target_pool_sha256,
            'source_subject_ids': list(dict.fromkeys(np.asarray(source_subject_ids).astype(str).tolist())),
            'partition_hash': background_bundle['metadata']['partition_hash'],
            'background_size': background_bundle['metadata'].get('background_size'),
            'class_counts': background_bundle['metadata'].get('class_counts'),
            'explainer_type': config['explainer'],
            'protocol_manifest_hash': protocol_manifest_hash,
            'preprocessing_manifest_hash': preprocessing_manifest_hash,
            'downstream_execution_manifest_hash': downstream_execution_manifest_hash,
            'downstream_execution_id': downstream_execution_id,
            'shap_version': getattr(shap, '__version__', 'unknown'),
            'sample_indices': result_cross['sample_indices'].astype(int).tolist(),
            'sample_indices_sha256': _hash_array(result_cross['sample_indices']),
            'subject_ids': result_cross['subject_ids'].astype(str).tolist(),
            'explained_row_count': int(len(result_cross['shap_values'])),
            'background_metadata_path': background_bundle.get('metadata_path'),
            'background_metadata_sha256': _hash_file(background_bundle['metadata_path']),
            'sampling_seed': int(cross_sampling_seed),
            'kernel_nsamples': config.get('nsamples'),
        }
        cross_npz_sha, cross_meta_sha = _save_shap_artifact_with_metadata(
            npz_path=cross_npz_path,
            metadata_path=cross_meta_path,
            shap_result=result_cross,
            metadata=cross_metadata,
        )
        cross_shap = result_cross['shap_values']
        base_cross = result_cross['base_value']
        completeness_cross = result_cross['completeness_error']
        cross_subject_ids = result_cross['subject_ids']
        n_cross = len(cross_shap)

        # ── δj ───────────────────────────────────────────────────────────────
        dj = compute_delta_j(within_shap, cross_shap)
        family_delta_j = compute_family_delta_j(
            feature_cols=selected_feature_cols,
            mean_abs_within=dj['mean_abs_within'],
            mean_abs_cross=dj['mean_abs_cross'],
        )
        stability = compute_delta_j_stability(
            shap_within=within_shap,
            shap_cross=cross_shap,
            subject_ids_within=within_subject_ids,
            subject_ids_cross=cross_subject_ids,
            rng=np.random.default_rng(
                _stable_sampling_seed(
                    'delta_j_stability',
                    source_condition,
                    target_condition,
                    clf_name,
                )
            ),
            n_resamples=stability_n_resamples,
        )

        direction_results[clf_name] = {
            'explainer_type':            config['explainer'],
            'n_samples_within':          n_within,
            'n_samples_cross':           n_cross,
            'base_value_within':         base_within,
            'base_value_cross':          base_cross,
            'completeness_error_within': completeness_within,
            'completeness_error_cross':  completeness_cross,
            'model_path':                str(model_path),
            'model_sha256':              model_sha256,
            'background_metadata_path':  background_bundle.get('metadata_path'),
            'background_sha256':         background_bundle.get('background_sha256'),
            'within_npz_path':           str(within_npz_path),
            'within_metadata_path':      str(within_meta_path),
            'within_npz_sha256':         within_npz_sha,
            'within_metadata_sha256':    within_meta_sha,
            'cross_npz_path':            str(cross_npz_path),
            'cross_metadata_path':       str(cross_meta_path),
            'cross_npz_sha256':          cross_npz_sha,
            'cross_metadata_sha256':     cross_meta_sha,
            'mean_abs_within':           dj['mean_abs_within'].tolist(),
            'mean_abs_cross':            dj['mean_abs_cross'].tolist(),
            'delta_j':                   dj['delta_j'].tolist(),
            'delta_j_normalized':        dj['delta_j_normalized'].tolist(),
            'emerged_features':          dj['emerged_features'],
            'family_delta_j':            family_delta_j,
            'stability':                 stability,
        }

        top3 = np.argsort(dj['delta_j'])[::-1][:3]
        top3_str = ', '.join(
            f'{selected_feature_cols[j]}={dj["delta_j"][j]:.4f}'
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

    top1_count = np.zeros(len(selected_feature_cols), dtype=int)
    top3_count = np.zeros(len(selected_feature_cols), dtype=int)
    for clf_name in clf_names:
        delta = np.asarray(direction_results[clf_name]['delta_j'], dtype=np.float64)
        rank_desc = np.argsort(delta)[::-1]
        top1_count[rank_desc[0]] += 1
        top3_count[rank_desc[:3]] += 1

    direction_results['__consensus__'] = {
        'feature_cols': selected_feature_cols,
        'top1_count': top1_count.tolist(),
        'top3_count': top3_count.tolist(),
        'background_metadata_path': background_bundle.get('metadata_path'),
        'background_sha256': background_bundle.get('background_sha256'),
    }
    return direction_results
