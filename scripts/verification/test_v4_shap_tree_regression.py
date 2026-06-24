"""
Tree-SHAP transformed-input regression checks for v4.

Usage:
    python scripts/verification/test_v4_shap_tree_regression.py
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

from explain import compute_shap_values, get_background_data
from train import _configure_classifier_for_resampling, build_pipeline, get_classifier_configs


def _run_case(clf_name: str, strategy: str) -> dict[str, float | str]:
    X, y = make_classification(
        n_samples=160,
        n_features=14,
        n_informative=10,
        n_redundant=2,
        weights=[0.4, 0.6],
        random_state=42,
    )
    subject_ids = np.asarray([f's{i // 4:03d}' for i in range(len(X))], dtype=object)
    background = get_background_data(X, y, k=40, random_seed=42)
    config = get_classifier_configs()[clf_name]
    clf_variant = _configure_classifier_for_resampling(clf_name, config['clf'], strategy)
    pipeline = build_pipeline(clf_name, clf_variant, imbalance_strategy=strategy)
    first_params = {
        key: values[0]
        for key, values in config['param_grid'].items()
    }
    pipeline.set_params(**first_params)
    fit_kwargs = {}
    if strategy == 'balanced' and clf_name == 'xgb':
        from train import _balanced_sample_weight  # noqa: SLF001
        fit_kwargs = {'clf__sample_weight': _balanced_sample_weight(y)}
    pipeline.fit(X, y, **fit_kwargs)
    result = compute_shap_values(
        clf_name=clf_name,
        pipeline=pipeline,
        X=X[:32],
        y=y[:32],
        subject_ids=subject_ids[:32],
        background=background,
        sample_indices=np.arange(32),
        feature_cols=[f'f{i}' for i in range(X.shape[1])],
    )
    return {
        'classifier': clf_name,
        'strategy': strategy,
        'explainer_type': result['explainer_type'],
        'completeness_error': round(float(result['completeness_error']), 8),
    }


def main() -> None:
    report = []
    for clf_name in ('rf', 'dt', 'xgb', 'lgbm'):
        for strategy in ('synthetic', 'raw'):
            report.append(_run_case(clf_name, strategy))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
