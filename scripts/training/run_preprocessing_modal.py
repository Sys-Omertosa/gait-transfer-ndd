"""
Modal Step 1 runner for the publication-track v4 preprocessing pipeline.

Writes to the Modal volume:
  /results/processed_v4/gait_features_v4.csv
  /results/processed_v4/gait_features_v4_per_stride_only.csv
  /results/processed_v4/gait_features_v4_subject_level.csv
  /results/processed_v4/control_partition_v4.json
  /results/processed_v4/preprocessing_manifest_v4.json
  /results/processed_v4/control_partition_candidates_v4.json

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
SUPPORTED_DFA_POLICY = 'concatenated'
SUPPORTED_PROTOCOL_SCHEMA_VERSIONS = {'v4-protocol-manifest-v2'}
SUPPORTED_METHODOLOGY_VERSIONS = {'v4-hardening'}
SUPPORTED_AGGREGATION_RULE = 'mean_probability'
SUPPORTED_TIE_BREAK_RULE = 'subject_probability_loss_then_lexicographic'
SUPPORTED_SUBJECT_PROBABILITY_THRESHOLD = 0.5
PREPROCESSING_MANIFEST_SCHEMA_VERSION = 'v4-preprocessing-manifest-v2'
PRAGMATIC_DFA_DESCRIPTOR_NOTE = (
    'pragmatic_dfa_derived_descriptor_from_discontinuous_stride_series'
)

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
    .add_local_dir(RAW_DATA_LOCAL, remote_path=RAW_DATA_REMOTE)
)

app = modal.App('gait-transfer-preprocessing-v4', image=image)
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

    from features import (
        build_feature_matrix,
        build_per_stride_only_matrix,
        build_subject_level_matrix,
        get_feature_cols,
    )
    from preprocessing import enumerate_balanced_control_partitions
    from v4_provenance import atomic_write_json, sha256_file

    processed_dir = Path('/results/processed_v4')
    processed_dir.mkdir(parents=True, exist_ok=True)
    protocol_manifest_path = processed_dir / 'v4_protocol_manifest.json'

    if not protocol_manifest_path.exists():
        raise FileNotFoundError(
            'Missing frozen protocol manifest at '
            '/results/processed_v4/v4_protocol_manifest.json. '
            'Freeze the v4 protocol manifest before running Step 1.'
        )

    with open(protocol_manifest_path) as f:
        protocol_manifest = json.load(f)

    if protocol_manifest.get('approved') is not True:
        raise ValueError(
            'The authoritative Step 1 runner requires an approved frozen protocol '
            'manifest.'
        )
    if protocol_manifest.get('schema_version') not in SUPPORTED_PROTOCOL_SCHEMA_VERSIONS:
        raise ValueError(
            'Unsupported protocol manifest schema version: '
            f'{protocol_manifest.get("schema_version")!r}.'
        )
    if protocol_manifest.get('methodology_version') not in SUPPORTED_METHODOLOGY_VERSIONS:
        raise ValueError(
            'Unsupported methodology_version in frozen protocol manifest: '
            f'{protocol_manifest.get("methodology_version")!r}.'
        )

    robust_mad_multiplier = float(protocol_manifest['robust_mad_multiplier'])
    dfa_policy = str(protocol_manifest['dfa_policy'])
    aggregation_rule = str(protocol_manifest['aggregation_rule'])
    subject_probability_threshold = float(
        protocol_manifest['subject_probability_threshold']
    )
    tie_break_rule = str(protocol_manifest['tie_break_rule'])
    if dfa_policy != SUPPORTED_DFA_POLICY:
        raise ValueError(
            'The authoritative Step 1 runner only supports '
            f"dfa_policy='{SUPPORTED_DFA_POLICY}'. "
            f"Received '{dfa_policy}'."
        )
    if robust_mad_multiplier != 3.0:
        raise ValueError(
            'The publication-track authoritative Step 1 runner expects '
            'robust_mad_multiplier=3.0.'
        )
    if aggregation_rule != SUPPORTED_AGGREGATION_RULE:
        raise ValueError(
            'The publication-track authoritative Step 1 runner expects '
            f'aggregation_rule={SUPPORTED_AGGREGATION_RULE!r}.'
        )
    if subject_probability_threshold != SUPPORTED_SUBJECT_PROBABILITY_THRESHOLD:
        raise ValueError(
            'The publication-track authoritative Step 1 runner expects '
            f'subject_probability_threshold='
            f'{SUPPORTED_SUBJECT_PROBABILITY_THRESHOLD}.'
        )
    if tie_break_rule != SUPPORTED_TIE_BREAK_RULE:
        raise ValueError(
            'The publication-track authoritative Step 1 runner expects '
            f'tie_break_rule={SUPPORTED_TIE_BREAK_RULE!r}.'
        )

    feature_cols = get_feature_cols('v4')
    canonical_partition_filename = 'control_partition_v4.json'
    df, partition = build_feature_matrix(
        processed_dir=processed_dir,
        output_filename='gait_features_v4.csv',
        feature_cols=feature_cols,
        feature_set_version='v4',
        filter_strategy='v3',
        control_partition_version='v4',
        partition_output_filename=canonical_partition_filename,
        manifest_filename='preprocessing_manifest_v4.json',
        robust_mad_multiplier=robust_mad_multiplier,
    )

    per_stride_only = build_per_stride_only_matrix(df, feature_set_version='v4')
    per_stride_only_path = processed_dir / 'gait_features_v4_per_stride_only.csv'
    per_stride_only.write_csv(per_stride_only_path)

    subject_level = build_subject_level_matrix(df, feature_set_version='v4')
    subject_level_path = processed_dir / 'gait_features_v4_subject_level.csv'
    subject_level.write_csv(subject_level_path)

    v3_timing_sensitivity_cols = [
        col for col in get_feature_cols('v4')
        if col != 'left_stance_s'
    ]
    timing_df, _ = build_feature_matrix(
        processed_dir=processed_dir,
        output_filename='gait_features_v4_timing_sensitivity.csv',
        feature_cols=v3_timing_sensitivity_cols,
        feature_set_version='v4',
        filter_strategy='v3',
        control_partition_version='v4',
        partition_output_filename='control_partition_v4_timing_sensitivity.json',
        manifest_filename='preprocessing_manifest_v4_timing_sensitivity.json',
        robust_mad_multiplier=robust_mad_multiplier,
    )

    n_min_strides = 100
    hausdorff_scales = np.unique(np.round(
        np.logspace(np.log10(4), np.log10(n_min_strides // 4), num=20)
    ).astype(int))
    dfa_df, _ = build_feature_matrix(
        processed_dir=processed_dir,
        output_filename='gait_features_v4_dfa_sensitivity.csv',
        feature_cols=get_feature_cols('v4'),
        feature_set_version='v4',
        filter_strategy='v3',
        control_partition_version='v4',
        partition_output_filename='control_partition_v4_dfa_sensitivity.json',
        manifest_filename='preprocessing_manifest_v4_dfa_sensitivity.json',
        dfa_scale_mode='pragmatic',
        dfa_custom_scales=hausdorff_scales,
        robust_mad_multiplier=robust_mad_multiplier,
    )

    no_dfa_feature_cols = [
        col for col in get_feature_cols('v4')
        if col != 'dfa_alpha_stride'
    ]
    no_dfa_df, _ = build_feature_matrix(
        processed_dir=processed_dir,
        output_filename='gait_features_v4_no_dfa_sensitivity.csv',
        feature_cols=no_dfa_feature_cols,
        feature_set_version='v4',
        filter_strategy='v3',
        control_partition_version='v4',
        partition_output_filename='control_partition_v4_no_dfa_sensitivity.json',
        manifest_filename='preprocessing_manifest_v4_no_dfa_sensitivity.json',
        robust_mad_multiplier=robust_mad_multiplier,
    )

    candidates = enumerate_balanced_control_partitions(top_k=5)
    candidates_path = processed_dir / 'control_partition_candidates_v4.json'
    atomic_write_json(candidates_path, candidates)

    canonical_partition_path = processed_dir / canonical_partition_filename
    sensitivity_partition_paths = {
        'timing': processed_dir / 'control_partition_v4_timing_sensitivity.json',
        'dfa': processed_dir / 'control_partition_v4_dfa_sensitivity.json',
        'no_dfa': processed_dir / 'control_partition_v4_no_dfa_sensitivity.json',
    }
    canonical_partition_hash = sha256_file(canonical_partition_path)
    for label, partition_path in sensitivity_partition_paths.items():
        with open(partition_path) as f:
            branch_partition = json.load(f)
        if branch_partition != partition:
            raise ValueError(
                f'{label} sensitivity partition does not match the canonical v4 '
                'control partition.'
            )
        if sha256_file(partition_path) != canonical_partition_hash:
            raise ValueError(
                f'{label} sensitivity partition hash does not match the canonical '
                'v4 control partition hash.'
            )

    preprocessing_manifest_path = processed_dir / 'preprocessing_manifest_v4.json'
    with open(preprocessing_manifest_path) as f:
        preprocessing_manifest = json.load(f)
    feature_matrix_path = processed_dir / 'gait_features_v4.csv'
    timing_matrix_path = processed_dir / 'gait_features_v4_timing_sensitivity.csv'
    dfa_matrix_path = processed_dir / 'gait_features_v4_dfa_sensitivity.csv'
    no_dfa_matrix_path = processed_dir / 'gait_features_v4_no_dfa_sensitivity.csv'
    preprocessing_manifest['schema_version'] = PREPROCESSING_MANIFEST_SCHEMA_VERSION
    preprocessing_manifest['approved_protocol_manifest_required'] = True
    preprocessing_manifest['methodology_version'] = protocol_manifest['methodology_version']
    preprocessing_manifest['protocol_manifest_path'] = str(protocol_manifest_path)
    preprocessing_manifest['protocol_manifest_sha256'] = sha256_file(protocol_manifest_path)
    preprocessing_manifest['feature_matrix_sha256'] = sha256_file(feature_matrix_path)
    preprocessing_manifest['partition_sha256'] = canonical_partition_hash
    preprocessing_manifest['robust_mad_multiplier'] = robust_mad_multiplier
    preprocessing_manifest['dfa_policy'] = dfa_policy
    preprocessing_manifest['aggregation_rule'] = aggregation_rule
    preprocessing_manifest['subject_probability_threshold'] = subject_probability_threshold
    preprocessing_manifest['tie_break_rule'] = tie_break_rule
    preprocessing_manifest['dfa_descriptor_interpretation'] = PRAGMATIC_DFA_DESCRIPTOR_NOTE
    preprocessing_manifest['per_stride_only_matrix_path'] = str(per_stride_only_path)
    preprocessing_manifest['per_stride_only_matrix_sha256'] = sha256_file(per_stride_only_path)
    preprocessing_manifest['subject_level_matrix_path'] = str(subject_level_path)
    preprocessing_manifest['subject_level_matrix_sha256'] = sha256_file(subject_level_path)
    preprocessing_manifest['timing_sensitivity_matrix_path'] = str(timing_matrix_path)
    preprocessing_manifest['timing_sensitivity_matrix_sha256'] = sha256_file(timing_matrix_path)
    preprocessing_manifest['timing_sensitivity_partition_sha256'] = sha256_file(
        sensitivity_partition_paths['timing']
    )
    preprocessing_manifest['dfa_sensitivity_matrix_path'] = str(dfa_matrix_path)
    preprocessing_manifest['dfa_sensitivity_matrix_sha256'] = sha256_file(dfa_matrix_path)
    preprocessing_manifest['dfa_sensitivity_partition_sha256'] = sha256_file(
        sensitivity_partition_paths['dfa']
    )
    preprocessing_manifest['no_dfa_sensitivity_matrix_path'] = str(no_dfa_matrix_path)
    preprocessing_manifest['no_dfa_sensitivity_matrix_sha256'] = sha256_file(
        no_dfa_matrix_path
    )
    preprocessing_manifest['no_dfa_sensitivity_partition_sha256'] = sha256_file(
        sensitivity_partition_paths['no_dfa']
    )
    atomic_write_json(preprocessing_manifest_path, preprocessing_manifest)

    volume.commit()

    output = {
        'processed_dir': str(processed_dir),
        'features_path': str(processed_dir / 'gait_features_v4.csv'),
        'per_stride_only_features_path': str(per_stride_only_path),
        'subject_level_features_path': str(subject_level_path),
        'partition_path': str(processed_dir / 'control_partition_v4.json'),
        'manifest_path': str(processed_dir / 'preprocessing_manifest_v4.json'),
        'protocol_manifest_path': str(protocol_manifest_path),
        'timing_sensitivity_features_path': str(
            processed_dir / 'gait_features_v4_timing_sensitivity.csv'
        ),
        'timing_sensitivity_manifest_path': str(
            processed_dir / 'preprocessing_manifest_v4_timing_sensitivity.json'
        ),
        'dfa_sensitivity_features_path': str(
            processed_dir / 'gait_features_v4_dfa_sensitivity.csv'
        ),
        'dfa_sensitivity_manifest_path': str(
            processed_dir / 'preprocessing_manifest_v4_dfa_sensitivity.json'
        ),
        'no_dfa_sensitivity_features_path': str(no_dfa_matrix_path),
        'no_dfa_sensitivity_manifest_path': str(
            processed_dir / 'preprocessing_manifest_v4_no_dfa_sensitivity.json'
        ),
        'candidates_path': str(candidates_path),
        'rows': int(df.height),
        'subjects': int(df.n_unique('subject_id')),
        'n_features': len(feature_cols),
        'robust_mad_multiplier': robust_mad_multiplier,
        'dfa_policy': dfa_policy,
        'aggregation_rule': aggregation_rule,
        'subject_probability_threshold': subject_probability_threshold,
        'tie_break_rule': tie_break_rule,
        'protocol_manifest_sha256': preprocessing_manifest['protocol_manifest_sha256'],
        'preprocessing_manifest_sha256': sha256_file(preprocessing_manifest_path),
        'timing_sensitivity_rows': int(timing_df.height),
        'timing_sensitivity_n_features': len(v3_timing_sensitivity_cols),
        'dfa_sensitivity_rows': int(dfa_df.height),
        'dfa_sensitivity_n_features': len(get_feature_cols('v4')),
        'no_dfa_sensitivity_rows': int(no_dfa_df.height),
        'no_dfa_sensitivity_n_features': len(no_dfa_feature_cols),
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
    print('  modal volume get gait-results processed_v4/gait_features_v4.csv', flush=True)
    print(
        '  modal volume get gait-results processed_v4/control_partition_v4.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results processed_v4/preprocessing_manifest_v4.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results processed_v4/gait_features_v4_per_stride_only.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results processed_v4/gait_features_v4_subject_level.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v4/gait_features_v4_timing_sensitivity.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v4/preprocessing_manifest_v4_timing_sensitivity.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v4/gait_features_v4_dfa_sensitivity.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v4/gait_features_v4_no_dfa_sensitivity.csv',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v4/preprocessing_manifest_v4_dfa_sensitivity.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results '
        'processed_v4/preprocessing_manifest_v4_no_dfa_sensitivity.json',
        flush=True,
    )
    print(
        '  modal volume get gait-results processed_v4/control_partition_candidates_v4.json',
        flush=True,
    )
