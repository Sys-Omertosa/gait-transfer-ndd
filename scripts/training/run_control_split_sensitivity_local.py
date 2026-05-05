"""
Control-split sensitivity runner for the publication-track v3 pipeline.

This is part of the planned sensitivity layer rather than the main Step 1 to 4
rerun. It evaluates a small shortlist of near-optimal balanced control
partitions and reports whether the direction ordering and sign pattern of the
transfer benchmark stay stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from features import get_feature_cols  # noqa: E402
from preprocessing import enumerate_balanced_control_partitions  # noqa: E402
from train import run_cross_condition, run_within_condition  # noqa: E402

FEATURES_PATH = REPO_ROOT / 'data' / 'processed' / 'v3' / 'gait_features_v3.csv'
RESULTS_ROOT = REPO_ROOT / 'experiments' / 'results' / 'v3' / 'control_split_sensitivity'
MODELS_ROOT = REPO_ROOT / 'experiments' / 'models' / 'v3' / 'control_split_sensitivity'
DIRECTIONS = [
    ('pd', 'hd'),
    ('hd', 'pd'),
    ('pd', 'als'),
    ('als', 'pd'),
    ('hd', 'als'),
    ('als', 'hd'),
]
MAX_PARTITIONS = 3


def main() -> None:
    df = pl.read_csv(FEATURES_PATH)
    feature_cols = get_feature_cols('v3')
    candidates = enumerate_balanced_control_partitions(top_k=MAX_PARTITIONS)
    summary: dict[str, dict] = {}

    for idx, candidate in enumerate(candidates, start=1):
        partition_key = f'partition_{idx}'
        partition = {
            'control_A': candidate['control_A'],
            'control_B': candidate['control_B'],
        }
        print(f'\n=== {partition_key}: score={candidate["score"]:.6f} ===', flush=True)

        partition_results_dir = RESULTS_ROOT / partition_key
        partition_models_dir = MODELS_ROOT / partition_key
        within_results: dict[str, dict] = {}

        for condition in ('pd', 'hd', 'als'):
            within_results[condition] = run_within_condition(
                condition=condition,
                df=df,
                control_a=partition['control_A'],
                results_dir=partition_results_dir,
                feature_cols=feature_cols,
                feature_matrix_file='v3/gait_features_v3.csv',
                feature_set_version='v3',
                normalization='none',
                results_filename=f'{condition}_results_v3_{partition_key}.json',
                imbalance_arms=('synthetic', 'balanced', 'raw'),
                selection_arms=('synthetic', 'balanced'),
            )

        cross_results: dict[str, dict] = {}
        for source_cond, target_cond in DIRECTIONS:
            direction_key = f'{source_cond}_to_{target_cond}'
            cross_results[direction_key] = run_cross_condition(
                source_condition=source_cond,
                target_condition=target_cond,
                df=df,
                control_a=partition['control_A'],
                control_b=partition['control_B'],
                source_results=within_results[source_cond],
                results_dir=partition_results_dir,
                models_dir=partition_models_dir,
                feature_cols=feature_cols,
                feature_matrix_file='v3/gait_features_v3.csv',
                feature_set_version='v3',
                normalization='none',
            )

        summary[partition_key] = {
            'partition': partition,
            'score': candidate['score'],
            'age_delta_years': candidate['age_delta_years'],
            'gait_speed_delta_m_per_s': candidate['gait_speed_delta_m_per_s'],
            'cross_results': {
                direction_key: {
                    clf_name: clf_out['f1_macro']
                    for clf_name, clf_out in direction_out['classifiers'].items()
                }
                for direction_key, direction_out in cross_results.items()
            },
        }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_ROOT / 'control_split_sensitivity_summary_v3.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSensitivity summary written to {out_path}', flush=True)


if __name__ == '__main__':
    main()
