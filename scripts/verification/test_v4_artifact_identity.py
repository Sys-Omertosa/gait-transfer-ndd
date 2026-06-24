"""
Fail-closed artifact-identity check for Step 5 replay.

Usage:
    python scripts/verification/test_v4_artifact_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import robustness  # type: ignore
from features import ALL_FEATURE_COLS  # type: ignore


def _toy_df() -> pl.DataFrame:
    rows = []
    for subject_id, condition, label, base in (
        ('pd1', 'pd', 1, 1.0),
        ('control1', 'control', 0, 0.5),
    ):
        for stride_idx in range(4):
            row = {
                feature_name: base + 0.01 * (stride_idx + feature_idx)
                for feature_idx, feature_name in enumerate(ALL_FEATURE_COLS)
            }
            row.update({
                'subject_id': subject_id,
                'condition': condition,
                'label': label,
            })
            rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    df = _toy_df()
    missing_trace = [{
        'held_out_subject_id': 'pd1',
        'fold_model_relpath': 'does/not/exist.joblib',
        'fold_model_sha256': 'missing',
    }]
    within_results = {
        'models_dir': str(REPO_ROOT / 'experiments' / 'models' / 'v4'),
        'classifiers': {
            clf_name: {
                'selected_imbalance_strategy': 'raw',
                'modal_params': {},
                'outer_fold_selection_trace': missing_trace,
            }
            for clf_name in robustness.CLF_ORDER
        },
    }

    try:
        robustness.fit_within_condition_folds(
            'pd',
            df,
            ['control1'],
            within_results,
            feature_cols=list(ALL_FEATURE_COLS),
        )
    except FileNotFoundError:
        print('Artifact-identity fail-closed behavior passed.')
        return

    raise SystemExit('Expected fit_within_condition_folds() to fail closed on missing artifacts.')


if __name__ == '__main__':
    main()
