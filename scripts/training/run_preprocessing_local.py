"""
Local Step 1 runner for the publication-track v3 preprocessing pipeline.

Writes:
  data/processed/v3/gait_features_v3.csv
  data/processed/v3/gait_features_v3_timing_sensitivity.csv
  data/processed/v3/control_partition_v3.json
  data/processed/v3/preprocessing_manifest_v3.json
  data/processed/v3/preprocessing_manifest_v3_timing_sensitivity.json
  data/processed/v3/control_partition_candidates_v3.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from features import build_feature_matrix, get_feature_cols  # noqa: E402
from preprocessing import enumerate_balanced_control_partitions  # noqa: E402

PROCESSED_DIR = REPO_ROOT / 'data' / 'processed' / 'v3'


def main() -> None:
    feature_cols = get_feature_cols('v3')
    df, partition = build_feature_matrix(
        processed_dir=PROCESSED_DIR,
        output_filename='gait_features_v3.csv',
        feature_cols=feature_cols,
        feature_set_version='v3',
        filter_strategy='v3',
        control_partition_version='v3',
        partition_output_filename='control_partition_v3.json',
        manifest_filename='preprocessing_manifest_v3.json',
    )
    v3_timing_sensitivity_cols = [
        col for col in get_feature_cols('v3')
        if col != 'left_stance_s'
    ]
    timing_df, _ = build_feature_matrix(
        processed_dir=PROCESSED_DIR,
        output_filename='gait_features_v3_timing_sensitivity.csv',
        feature_cols=v3_timing_sensitivity_cols,
        feature_set_version='v3',
        filter_strategy='v3',
        control_partition_version='v3',
        partition_output_filename='control_partition_v3.json',
        manifest_filename='preprocessing_manifest_v3_timing_sensitivity.json',
    )

    candidates = enumerate_balanced_control_partitions(top_k=5)
    candidates_path = PROCESSED_DIR / 'control_partition_candidates_v3.json'
    with open(candidates_path, 'w') as f:
        json.dump(candidates, f, indent=2)

    print('v3 preprocessing complete', flush=True)
    print(f'  rows      : {df.height}', flush=True)
    print(f'  subjects  : {df.n_unique("subject_id")}', flush=True)
    print(f'  features  : {len(feature_cols)}', flush=True)
    print(f'  timing rows: {timing_df.height}', flush=True)
    print(f'  timing features: {len(v3_timing_sensitivity_cols)}', flush=True)
    print(f'  control A : {partition["control_A"]}', flush=True)
    print(f'  control B : {partition["control_B"]}', flush=True)
    print(f'  candidates: {candidates_path}', flush=True)


if __name__ == '__main__':
    main()
