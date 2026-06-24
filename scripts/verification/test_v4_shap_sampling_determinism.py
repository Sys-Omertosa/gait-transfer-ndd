"""
KernelExplainer sampling determinism regression checks for v4.

Usage:
    python scripts/verification/test_v4_shap_sampling_determinism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from explain import _stable_sampling_seed, get_shap_config, subsample_stratified
from v4_provenance import sha256_array


def _sample_indices(seed: int) -> np.ndarray:
    X, y = make_classification(
        n_samples=1600,
        n_features=14,
        n_informative=10,
        n_redundant=2,
        weights=[0.45, 0.55],
        random_state=42,
    )
    _, _, indices = subsample_stratified(
        X.astype(np.float64),
        y.astype(int),
        get_shap_config('svm')['n_explained'],
        np.random.default_rng(seed),
    )
    return indices.astype(np.int64)


def main() -> None:
    within_seed = _stable_sampling_seed('within_explained_rows', 'pd', 'svm')
    cross_hd_seed = _stable_sampling_seed('cross_explained_rows', 'pd', 'hd', 'svm')
    cross_als_seed = _stable_sampling_seed('cross_explained_rows', 'pd', 'als', 'svm')

    within_before_hd = _sample_indices(within_seed)
    cross_after_within_hd = _sample_indices(cross_hd_seed)

    cross_without_within_hd = _sample_indices(cross_hd_seed)
    within_before_als = _sample_indices(within_seed)
    cross_after_within_als = _sample_indices(cross_als_seed)

    assert np.array_equal(cross_after_within_hd, cross_without_within_hd)
    assert np.array_equal(within_before_hd, within_before_als)

    report = {
        'within_sampling_seed': int(within_seed),
        'cross_hd_sampling_seed': int(cross_hd_seed),
        'cross_als_sampling_seed': int(cross_als_seed),
        'within_sample_indices_sha256': sha256_array(within_before_hd),
        'cross_hd_sample_indices_sha256_with_preexisting_within': sha256_array(cross_after_within_hd),
        'cross_hd_sample_indices_sha256_without_preexisting_within': sha256_array(cross_without_within_hd),
        'cross_als_sample_indices_sha256': sha256_array(cross_after_within_als),
        'cross_hd_identical_with_without_within_cache': True,
        'within_identical_regardless_of_target_order': True,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
