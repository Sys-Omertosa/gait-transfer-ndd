"""
Modal Step 1 runner for the publication-track v3 preprocessing pipeline.

Writes to the Modal volume:
  /results/processed_v3/gait_features_v3.csv
  /results/processed_v3/control_partition_v3.json
  /results/processed_v3/preprocessing_manifest_v3.json
  /results/processed_v3/control_partition_candidates_v3.json

Usage:
    modal run scripts/training/run_preprocessing_modal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
import numpy as np

RAW_DATA_LOCAL = 'data/raw/gait-in-neurodegenerative-disease-database-1.0.0'
RAW_DATA_REMOTE = '/root/data/raw/gait-in-neurodegenerative-disease-database-1.0.0'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
    .add_local_dir(RAW_DATA_LOCAL, remote_path=RAW_DATA_REMOTE)
)

app = modal.App('gait-transfer-preprocessing', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)


@app.function(
    cpu=8,
    memory=8192,
    timeout=3600,
    volumes={'/results': volume},
    retries=1,
)
def run_preprocessing() -> str:
    import os

    from features import build_feature_matrix, get_feature_cols
    from preprocessing import enumerate_balanced_control_partitions

    processed_dir = Path('/results/processed_v3')
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols('v3')
    df, partition = build_feature_matrix(
        processed_dir=processed_dir,
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
        processed_dir=processed_dir,
        output_filename='gait_features_v3_timing_sensitivity.csv',
        feature_cols=v3_timing_sensitivity_cols,
        feature_set_version='v3',
        filter_strategy='v3',
        control_partition_version='v3',
        partition_output_filename='control_partition_v3.json',
        manifest_filename='preprocessing_manifest_v3_timing_sensitivity.json',
    )

    n_min_strides = 100
    hausdorff_scales = np.unique(np.round(
        np.logspace(np.log10(4), np.log10(n_min_strides // 4), num=20)
    ).astype(int))
    dfa_df, _ = build_feature_matrix(
        processed_dir=processed_dir,
        output_filename='gait_features_v3_dfa_sensitivity.csv',
        feature_cols=get_feature_cols('v3'),
        feature_set_version='v3',
        filter_strategy='v3',
        control_partition_version='v3',
        partition_output_filename='control_partition_v3.json',
        manifest_filename='preprocessing_manifest_v3_dfa_sensitivity.json',
        dfa_scale_mode='pragmatic',
        dfa_custom_scales=hausdorff_scales,
    )

    candidates = enumerate_balanced_control_partitions(top_k=5)
    candidates_path = processed_dir / 'control_partition_candidates_v3.json'
    with open(candidates_path, 'w') as f:
        json.dump(candidates, f, indent=2)

    output = {
        'processed_dir': str(processed_dir),
        'features_path': str(processed_dir / 'gait_features_v3.csv'),
        'partition_path': str(processed_dir / 'control_partition_v3.json'),
        'manifest_path': str(processed_dir / 'preprocessing_manifest_v3.json'),
        'timing_sensitivity_features_path': str(
            processed_dir / 'gait_features_v3_timing_sensitivity.csv'
        ),
        'timing_sensitivity_manifest_path': str(
            processed_dir / 'preprocessing_manifest_v3_timing_sensitivity.json'
        ),
        'dfa_sensitivity_features_path': str(
            processed_dir / 'gait_features_v3_dfa_sensitivity.csv'
        ),
        'dfa_sensitivity_manifest_path': str(
            processed_dir / 'preprocessing_manifest_v3_dfa_sensitivity.json'
        ),
        'candidates_path': str(candidates_path),
        'rows': int(df.height),
        'subjects': int(df.n_unique('subject_id')),
        'n_features': len(feature_cols),
        'timing_sensitivity_rows': int(timing_df.height),
        'timing_sensitivity_n_features': len(v3_timing_sensitivity_cols),
        'dfa_sensitivity_rows': int(dfa_df.height),
        'dfa_sensitivity_n_features': len(get_feature_cols('v3')),
        'control_A': partition['control_A'],
        'control_B': partition['control_B'],
        'raw_data_dir': RAW_DATA_REMOTE,
        'container_cwd': os.getcwd(),
    }
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main() -> None:
    print('Launching Step 1 preprocessing on Modal...', flush=True)
    print('Single container: 8 CPU, 8192 MB RAM.', flush=True)
    print()

    summary = json.loads(run_preprocessing.remote())
    print(json.dumps(summary, indent=2), flush=True)
    print('\nDownload artifacts if needed:', flush=True)
    print('  modal volume get gait-results processed_v3/gait_features_v3.csv', flush=True)
    print(
        '  modal volume get gait-results processed_v3/control_partition_v3.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results processed_v3/preprocessing_manifest_v3.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v3/gait_features_v3_timing_sensitivity.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v3/preprocessing_manifest_v3_timing_sensitivity.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v3/gait_features_v3_dfa_sensitivity.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v3/preprocessing_manifest_v3_dfa_sensitivity.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results processed_v3/control_partition_candidates_v3.json',
        flush=True,
    )
