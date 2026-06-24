"""
Modal control-split sensitivity runner for the publication-track v4 layer.

Required path in this pass:
  - full-protocol retuning sensitivity
  - existing partition x classifier within-condition workers
  - existing partition x direction cross-condition workers

Detached-safe orchestration:
  --action retune-submit-within
  --action retune-status
  --action retune-status-optimized
  --action retune-dry-run
  --action retune-diagnose-missing
  --action retune-submit-within-missing
  --action retune-submit-within-shards
  --action retune-assemble-within-missing
  --action retune-assemble-within
  --action retune-submit-cross
  --action retune-assemble
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np


CONDITIONS = ('pd', 'hd', 'als')
DIRECTIONS = (
    ('pd', 'hd'),
    ('hd', 'pd'),
    ('pd', 'als'),
    ('als', 'pd'),
    ('hd', 'als'),
    ('als', 'hd'),
)
CLF_ORDER = ('rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm')
MAX_PARTITIONS = 3
STRATEGY_POLICY = {
    'rf': ('synthetic', 'balanced', 'raw'),
    'svm': ('synthetic', 'balanced', 'raw'),
    'dt': ('synthetic', 'balanced', 'raw'),
    'xgb': ('synthetic', 'balanced', 'raw'),
    'lgbm': ('synthetic', 'balanced', 'raw'),
    'knn': ('synthetic', 'raw'),
    'qda': ('synthetic', 'raw'),
}
WITHIN_PARTIAL_SCHEMA_VERSION = 'v4-step8-within-partial-v1'
WITHIN_FOLD_SHARD_SCHEMA_VERSION = 'v4-step8-within-fold-shard-v1'
WITHIN_FULL_SOURCE_SHARD_SCHEMA_VERSION = 'v4-step8-within-full-source-shard-v1'
CROSS_PARTIAL_SCHEMA_VERSION = 'v4-step8-cross-partial-v1'
WITHIN_ASSEMBLED_SCHEMA_VERSION = 'v4-step8-within-assembled-v1'
CROSS_COMBINED_SCHEMA_VERSION = 'v4-step8-cross-combined-v1'
SUMMARY_SCHEMA_VERSION = 'v4-step8-summary-v1'
SOURCE_WITHIN_RETRY_ATTEMPTS = 5
SOURCE_WITHIN_RETRY_SLEEP_SECONDS = 2.0
WITHIN_FOLD_SHARD_CPU = 4
WITHIN_FOLD_SHARD_MEMORY_MB = 6144
WITHIN_FOLD_SHARD_TIMEOUT_SECONDS = 21600
WITHIN_FOLD_SHARD_MAX_CONTAINERS = 10
WITHIN_FULL_SOURCE_CPU = 8
WITHIN_FULL_SOURCE_MEMORY_MB = 8192
WITHIN_FULL_SOURCE_TIMEOUT_SECONDS = 43200
WITHIN_FULL_SOURCE_MAX_CONTAINERS = 3

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-sensitivity', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=False)


def _partition_key(partition_index: int) -> str:
    return f'partition_{partition_index}'


def _candidate_source_api_path(candidate_family: str) -> str:
    if candidate_family == 'near_optimal':
        return 'processed_v4/control_partition_candidates_v4.json'
    if candidate_family == 'diverse':
        return 'processed_v4/control_partition_diverse_candidates_v4.json'
    raise ValueError(f'Unknown candidate_family: {candidate_family}')


def _family_summary_api_path(candidate_family: str) -> str:
    if candidate_family == 'near_optimal':
        return 'results_v4/control_split_sensitivity/sensitivity_summary_v4.json'
    if candidate_family == 'diverse':
        return 'results_v4/control_split_sensitivity_diverse/sensitivity_summary_diverse_v4.json'
    raise ValueError(f'Unknown candidate_family: {candidate_family}')


def _run_api_root(*, execution_id: str, candidate_family: str) -> str:
    return (
        f'results_v4/control_split_sensitivity_runs/'
        f'{execution_id}/{candidate_family}'
    )


def _run_models_root(*, execution_id: str, candidate_family: str) -> str:
    return (
        f'/results/models_v4/control_split_sensitivity_runs/'
        f'{execution_id}/{candidate_family}'
    )


def _run_results_root(*, execution_id: str, candidate_family: str) -> str:
    return (
        f'/results/results_v4/control_split_sensitivity_runs/'
        f'{execution_id}/{candidate_family}'
    )


def _within_results_filename(condition: str, partition_index: int) -> str:
    return f'{condition}_results_v4_{_partition_key(partition_index)}.json'


def _clf_partial_results_filename(
    condition: str,
    clf_name: str,
    partition_index: int,
) -> str:
    return (
        f'{condition}_{clf_name}_partial_v4_'
        f'{_partition_key(partition_index)}.json'
    )


def _cross_partial_filename(
    partition_index: int,
    source_condition: str,
    target_condition: str,
) -> str:
    return (
        f'{source_condition}_to_{target_condition}_'
        f'cross_results_v4_{_partition_key(partition_index)}.json'
    )


def _cross_combined_filename(partition_index: int) -> str:
    return f'cross_condition_results_v4_{_partition_key(partition_index)}.json'


def _run_models_api_root(*, execution_id: str, candidate_family: str) -> str:
    return (
        f'models_v4/control_split_sensitivity_runs/'
        f'{execution_id}/{candidate_family}'
    )


def _volume_path(volume_root: Path, api_path: str) -> Path:
    return volume_root / api_path


def _partition_results_dir(
    *,
    volume_root: Path,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
) -> Path:
    return _volume_path(
        volume_root,
        f'{_run_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}',
    )


def _partition_models_dir(
    *,
    volume_root: Path,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
) -> Path:
    return _volume_path(
        volume_root,
        f'{_run_models_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}',
    )


def _within_shards_api_dir(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
    condition: str,
    clf_name: str,
) -> str:
    return (
        f'{_run_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}/within_shards/{condition}/{clf_name}'
    )


def _within_fold_shard_api_path(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
    condition: str,
    clf_name: str,
    outer_fold_index: int,
    held_out_subject_id: str,
) -> str:
    shard_dir = _within_shards_api_dir(
        execution_id=execution_id,
        candidate_family=candidate_family,
        partition_index=partition_index,
        condition=condition,
        clf_name=clf_name,
    )
    return (
        f'{shard_dir}/outer_fold_{outer_fold_index:02d}_{held_out_subject_id}.json'
    )


def _within_full_source_shard_api_path(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
    condition: str,
    clf_name: str,
) -> str:
    shard_dir = _within_shards_api_dir(
        execution_id=execution_id,
        candidate_family=candidate_family,
        partition_index=partition_index,
        condition=condition,
        clf_name=clf_name,
    )
    return f'{shard_dir}/full_source.json'


def _read_volume_bytes(path: str) -> bytes:
    return b''.join(volume.read_file(path))


def _read_volume_json(path: str) -> dict[str, Any]:
    return json.loads(_read_volume_bytes(path).decode())


def _read_root_json(volume_root: Path, api_path: str) -> dict[str, Any]:
    return json.loads((volume_root / api_path).read_text())


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _sha256_dict(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def _partition_hash(control_a: list[str], control_b: list[str]) -> str:
    return _sha256_dict(
        {
            'control_A': list(control_a),
            'control_B': list(control_b),
        }
    )


def _candidate_metadata(
    *,
    candidate: dict[str, Any],
    candidate_family: str,
    partition_index: int,
    main_partition: dict[str, Any],
) -> dict[str, Any]:
    control_a = list(candidate['control_A'])
    control_b = list(candidate['control_B'])
    main_a = list(main_partition['control_A'])
    main_b = list(main_partition['control_B'])
    overlap_with_main_control_a = len(set(control_a) & set(main_a))
    is_main_partition = control_a == main_a and control_b == main_b
    is_role_reversal_of_main = control_a == main_b and control_b == main_a
    return {
        'partition_index': partition_index,
        'partition_key': _partition_key(partition_index),
        'candidate_family': candidate_family,
        'control_A': control_a,
        'control_B': control_b,
        'partition_hash': _partition_hash(control_a, control_b),
        'is_main_partition': bool(is_main_partition),
        'is_role_reversal_of_main': bool(is_role_reversal_of_main),
        'overlap_with_main_control_a': int(overlap_with_main_control_a),
        'independent_sensitivity_candidate': bool(
            not is_main_partition and not is_role_reversal_of_main
        ),
        'score': candidate.get('score'),
        'age_delta_years': candidate.get('age_delta_years'),
        'gait_speed_delta_m_per_s': candidate.get('gait_speed_delta_m_per_s'),
        'selection_bin': candidate.get('selection_bin'),
        'overlap_with_main': candidate.get('overlap_with_main'),
    }


def _within_partial_api_path(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
    condition: str,
    clf_name: str,
) -> str:
    return (
        f'{_run_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}/{_clf_partial_results_filename(condition, clf_name, partition_index)}'
    )


def _within_assembled_api_path(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
    condition: str,
) -> str:
    return (
        f'{_run_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}/{_within_results_filename(condition, partition_index)}'
    )


def _cross_partial_api_path(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
    source_condition: str,
    target_condition: str,
) -> str:
    return (
        f'{_run_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}/{_cross_partial_filename(partition_index, source_condition, target_condition)}'
    )


def _cross_combined_api_path(
    *,
    execution_id: str,
    candidate_family: str,
    partition_index: int,
) -> str:
    return (
        f'{_run_api_root(execution_id=execution_id, candidate_family=candidate_family)}/'
        f'{_partition_key(partition_index)}/{_cross_combined_filename(partition_index)}'
    )


def _best_within_f1(within_result: dict[str, Any], *, subject_level: bool) -> float:
    return max(
        float(
            clf_out['subject_primary_f1_macro']
            if subject_level else clf_out['f1_macro']
        )
        for clf_out in within_result['classifiers'].values()
    )


def _direction_summary(
    *,
    within_by_source: dict[str, dict[str, Any]],
    cross_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[float], list[str]]:
    directions_summary: dict[str, dict[str, Any]] = {}
    mean_delta_vector: list[float] = []
    sign_vector: list[str] = []

    for source_condition, target_condition in DIRECTIONS:
        direction_key = f'{source_condition}_to_{target_condition}'
        direction_out = cross_results[direction_key]
        per_classifier_subject_f1 = {
            clf_name: float(
                direction_out['classifiers'][clf_name]['subject_primary_f1_macro'])
            for clf_name in CLF_ORDER
        }
        per_classifier_stride_f1 = {
            clf_name: float(direction_out['classifiers'][clf_name]['f1_macro'])
            for clf_name in CLF_ORDER
        }
        per_classifier_subject_delta = {
            clf_name: round(
                float(
                    within_by_source[source_condition]['classifiers'][clf_name]['subject_primary_f1_macro']
                    - per_classifier_subject_f1[clf_name]
                ),
                6,
            )
            for clf_name in CLF_ORDER
        }
        per_classifier_stride_delta = {
            clf_name: round(
                float(
                    within_by_source[source_condition]['classifiers'][clf_name]['f1_macro']
                    - per_classifier_stride_f1[clf_name]
                ),
                6,
            )
            for clf_name in CLF_ORDER
        }
        mean_delta_f1 = float(
            np.mean(list(per_classifier_subject_delta.values())))
        median_delta_f1 = float(
            np.median(list(per_classifier_subject_delta.values())))
        direction_sign = '+' if mean_delta_f1 > 0 else '-'
        within_best_subject = _best_within_f1(
            within_by_source[source_condition], subject_level=True)
        within_best_stride = _best_within_f1(
            within_by_source[source_condition], subject_level=False)
        mean_cross_subject = float(
            np.mean(list(per_classifier_subject_f1.values())))
        mean_cross_stride = float(
            np.mean(list(per_classifier_stride_f1.values())))

        directions_summary[direction_key] = {
            'primary_mean_matched_degradation_subject': round(mean_delta_f1, 6),
            'median_matched_degradation_subject': round(median_delta_f1, 6),
            'direction_sign': direction_sign,
            'per_classifier_subject_f1': {
                clf_name: round(f1_val, 6)
                for clf_name, f1_val in per_classifier_subject_f1.items()
            },
            'per_classifier_stride_f1': {
                clf_name: round(f1_val, 6)
                for clf_name, f1_val in per_classifier_stride_f1.items()
            },
            'per_classifier_matched_degradation_subject': per_classifier_subject_delta,
            'per_classifier_matched_degradation_stride': per_classifier_stride_delta,
            'legacy_within_best_minus_mean_cross_subject': round(
                within_best_subject - mean_cross_subject,
                6,
            ),
            'legacy_within_best_minus_mean_cross_stride': round(
                within_best_stride - mean_cross_stride,
                6,
            ),
        }
        mean_delta_vector.append(mean_delta_f1)
        sign_vector.append(direction_sign)

    return directions_summary, mean_delta_vector, sign_vector


def _validate_partial_within_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    condition: str,
    clf_name: str,
    partition_metadata: dict[str, Any],
) -> None:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != WITHIN_PARTIAL_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 within-partial schema version.')
    if payload.get('condition') != condition:
        raise ValueError('Step 8 within-partial condition mismatch.')
    if payload.get('candidate_family') != partition_metadata['candidate_family']:
        raise ValueError('Step 8 within-partial candidate_family mismatch.')
    if payload.get('partition_index') != partition_metadata['partition_index']:
        raise ValueError('Step 8 within-partial partition_index mismatch.')
    if payload.get('partition_hash') != partition_metadata['partition_hash']:
        raise ValueError('Step 8 within-partial partition hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 within-partial feature hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 within-partial protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 8 within-partial preprocessing hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError(
            'Step 8 within-partial downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError(
            'Step 8 within-partial downstream manifest hash mismatch.')
    classifiers = payload.get('classifiers', {})
    if set(classifiers) != {clf_name}:
        raise ValueError('Step 8 within-partial classifier payload mismatch.')


def _validate_within_fold_shard_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    condition: str,
    clf_name: str,
    partition_metadata: dict[str, Any],
    outer_fold_index: int | None = None,
    held_out_subject_id: str | None = None,
) -> None:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != WITHIN_FOLD_SHARD_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 within-fold shard schema version.')
    if payload.get('condition') != condition:
        raise ValueError('Step 8 within-fold shard condition mismatch.')
    if payload.get('classifier') != clf_name:
        raise ValueError('Step 8 within-fold shard classifier mismatch.')
    if payload.get('candidate_family') != partition_metadata['candidate_family']:
        raise ValueError('Step 8 within-fold shard candidate_family mismatch.')
    if payload.get('partition_index') != partition_metadata['partition_index']:
        raise ValueError('Step 8 within-fold shard partition_index mismatch.')
    if payload.get('partition_hash') != partition_metadata['partition_hash']:
        raise ValueError('Step 8 within-fold shard partition hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 within-fold shard feature hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 within-fold shard protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 8 within-fold shard preprocessing hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Step 8 within-fold shard downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Step 8 within-fold shard downstream manifest hash mismatch.')
    if payload.get('subject_aggregation_rule') != context['step2_payloads'][condition]['subject_aggregation_rule']:
        raise ValueError('Step 8 within-fold shard aggregation rule mismatch.')
    if payload.get('tie_break_rule') != context['step2_payloads'][condition]['tie_break_rule']:
        raise ValueError('Step 8 within-fold shard tie-break rule mismatch.')
    if outer_fold_index is not None and int(payload.get('outer_fold_index')) != int(outer_fold_index):
        raise ValueError('Step 8 within-fold shard outer_fold_index mismatch.')
    if held_out_subject_id is not None and payload.get('held_out_subject_id') != held_out_subject_id:
        raise ValueError('Step 8 within-fold shard held_out_subject_id mismatch.')
    expected_policy = list(STRATEGY_POLICY[clf_name])
    if list(payload.get('candidate_imbalance_strategies', [])) != expected_policy:
        raise ValueError('Step 8 within-fold shard strategy policy mismatch.')
    y_true = payload.get('y_true', [])
    y_pred = payload.get('y_pred', [])
    y_prob = payload.get('y_prob', [])
    subject_ids = payload.get('subject_ids', [])
    lengths = {len(y_true), len(y_pred), len(y_prob), len(subject_ids)}
    if len(lengths) != 1:
        raise ValueError('Step 8 within-fold shard arrays disagree in length.')
    if not subject_ids:
        raise ValueError('Step 8 within-fold shard has no held-out predictions.')
    if any(subject_id != payload.get('held_out_subject_id') for subject_id in subject_ids):
        raise ValueError('Step 8 within-fold shard subject_ids do not match the held-out subject.')
    fold_model_relpath = payload.get('fold_model_relpath')
    fold_model_sha256 = payload.get('fold_model_sha256')
    if not isinstance(fold_model_relpath, str) or not fold_model_relpath:
        raise ValueError('Step 8 within-fold shard is missing fold_model_relpath.')
    if not isinstance(fold_model_sha256, str) or not fold_model_sha256:
        raise ValueError('Step 8 within-fold shard is missing fold_model_sha256.')
    models_dir = payload.get('models_dir')
    if not isinstance(models_dir, str) or not models_dir:
        raise ValueError('Step 8 within-fold shard is missing models_dir.')
    fold_model_path = Path(models_dir) / fold_model_relpath
    if not fold_model_path.exists():
        raise FileNotFoundError(f'Missing Step 8 fold model artifact: {fold_model_path}')
    from v4_downstream import sha256_file
    if sha256_file(fold_model_path) != fold_model_sha256:
        raise ValueError('Step 8 within-fold shard fold-model hash mismatch.')
    candidate_trace_relpath = payload.get('candidate_trace_relpath')
    candidate_trace_sha256 = payload.get('candidate_trace_sha256')
    if not isinstance(candidate_trace_relpath, str) or not candidate_trace_relpath:
        raise ValueError('Step 8 within-fold shard is missing candidate_trace_relpath.')
    if not isinstance(candidate_trace_sha256, str) or not candidate_trace_sha256:
        raise ValueError('Step 8 within-fold shard is missing candidate_trace_sha256.')
    candidate_trace_path = Path(candidate_trace_relpath)
    if not candidate_trace_path.exists():
        raise FileNotFoundError(
            f'Missing Step 8 within-fold candidate trace artifact: {candidate_trace_path}'
        )
    if sha256_file(candidate_trace_path) != candidate_trace_sha256:
        raise ValueError('Step 8 within-fold shard candidate-trace hash mismatch.')


def _validate_within_full_source_shard_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    condition: str,
    clf_name: str,
    partition_metadata: dict[str, Any],
) -> None:
    from v4_downstream import sha256_file, validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != WITHIN_FULL_SOURCE_SHARD_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 full-source shard schema version.')
    if payload.get('condition') != condition:
        raise ValueError('Step 8 full-source shard condition mismatch.')
    if payload.get('classifier') != clf_name:
        raise ValueError('Step 8 full-source shard classifier mismatch.')
    if payload.get('candidate_family') != partition_metadata['candidate_family']:
        raise ValueError('Step 8 full-source shard candidate_family mismatch.')
    if payload.get('partition_index') != partition_metadata['partition_index']:
        raise ValueError('Step 8 full-source shard partition_index mismatch.')
    if payload.get('partition_hash') != partition_metadata['partition_hash']:
        raise ValueError('Step 8 full-source shard partition hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 full-source shard feature hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 full-source shard protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 8 full-source shard preprocessing hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Step 8 full-source shard downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Step 8 full-source shard downstream manifest hash mismatch.')
    if payload.get('subject_aggregation_rule') != context['step2_payloads'][condition]['subject_aggregation_rule']:
        raise ValueError('Step 8 full-source shard aggregation rule mismatch.')
    if payload.get('tie_break_rule') != context['step2_payloads'][condition]['tie_break_rule']:
        raise ValueError('Step 8 full-source shard tie-break rule mismatch.')
    expected_policy = list(STRATEGY_POLICY[clf_name])
    if list(payload.get('candidate_imbalance_strategies', [])) != expected_policy:
        raise ValueError('Step 8 full-source shard strategy policy mismatch.')
    model_relpath = payload.get('model_path')
    model_sha256 = payload.get('model_sha256')
    if not isinstance(model_relpath, str) or not model_relpath:
        raise ValueError('Step 8 full-source shard is missing model_path.')
    if not isinstance(model_sha256, str) or not model_sha256:
        raise ValueError('Step 8 full-source shard is missing model_sha256.')
    models_dir = payload.get('models_dir')
    if not isinstance(models_dir, str) or not models_dir:
        raise ValueError('Step 8 full-source shard is missing models_dir.')
    model_path = Path(models_dir) / model_relpath
    if not model_path.exists():
        raise FileNotFoundError(f'Missing Step 8 full-source model artifact: {model_path}')
    if sha256_file(model_path) != model_sha256:
        raise ValueError('Step 8 full-source shard model hash mismatch.')
    selection_trace_path = payload.get('selection_trace_path')
    selection_trace_sha256 = payload.get('selection_trace_sha256')
    if not isinstance(selection_trace_path, str) or not selection_trace_path:
        raise ValueError('Step 8 full-source shard is missing selection_trace_path.')
    if not isinstance(selection_trace_sha256, str) or not selection_trace_sha256:
        raise ValueError('Step 8 full-source shard is missing selection_trace_sha256.')
    trace_path = Path(selection_trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f'Missing Step 8 full-source selection trace artifact: {trace_path}')
    if sha256_file(trace_path) != selection_trace_sha256:
        raise ValueError('Step 8 full-source shard trace hash mismatch.')


def _validate_within_assembled_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    condition: str,
    partition_metadata: dict[str, Any],
) -> None:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != WITHIN_ASSEMBLED_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 within-assembled schema version.')
    if payload.get('condition') != condition:
        raise ValueError('Step 8 within-assembled condition mismatch.')
    if payload.get('candidate_family') != partition_metadata['candidate_family']:
        raise ValueError('Step 8 within-assembled candidate_family mismatch.')
    if payload.get('partition_index') != partition_metadata['partition_index']:
        raise ValueError('Step 8 within-assembled partition_index mismatch.')
    if payload.get('partition_hash') != partition_metadata['partition_hash']:
        raise ValueError('Step 8 within-assembled partition hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 within-assembled feature hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 within-assembled protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError(
            'Step 8 within-assembled preprocessing hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError(
            'Step 8 within-assembled downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError(
            'Step 8 within-assembled downstream manifest hash mismatch.')
    if set(payload.get('classifiers', {})) != set(CLF_ORDER):
        raise ValueError('Step 8 within-assembled classifier set mismatch.')


def _validate_cross_partial_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    direction_key: str,
    partition_metadata: dict[str, Any],
) -> None:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != CROSS_PARTIAL_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 cross-partial schema version.')
    if payload.get('candidate_family') != partition_metadata['candidate_family']:
        raise ValueError('Step 8 cross-partial candidate_family mismatch.')
    if payload.get('partition_index') != partition_metadata['partition_index']:
        raise ValueError('Step 8 cross-partial partition_index mismatch.')
    if payload.get('partition_hash') != partition_metadata['partition_hash']:
        raise ValueError('Step 8 cross-partial partition hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 cross-partial feature hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 cross-partial protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 8 cross-partial preprocessing hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError(
            'Step 8 cross-partial downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError(
            'Step 8 cross-partial downstream manifest hash mismatch.')
    actual_direction_key = (
        f"{payload.get('source_condition')}_to_"
        f"{payload.get('target_condition')}"
    )
    if actual_direction_key != direction_key:
        raise ValueError(
            'Step 8 cross-partial direction mismatch: '
            f'expected={direction_key}, got={actual_direction_key}'
        )


def _validate_cross_combined_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    partition_metadata: dict[str, Any],
) -> None:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != CROSS_COMBINED_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 cross-combined schema version.')
    if payload.get('candidate_family') != partition_metadata['candidate_family']:
        raise ValueError('Step 8 cross-combined candidate_family mismatch.')
    if payload.get('partition_index') != partition_metadata['partition_index']:
        raise ValueError('Step 8 cross-combined partition_index mismatch.')
    if payload.get('partition_hash') != partition_metadata['partition_hash']:
        raise ValueError('Step 8 cross-combined partition hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 cross-combined feature hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 cross-combined protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 8 cross-combined preprocessing hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError(
            'Step 8 cross-combined downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError(
            'Step 8 cross-combined downstream manifest hash mismatch.')
    if set(payload.get('cross_results', {})) != {
        f'{source_condition}_to_{target_condition}'
        for source_condition, target_condition in DIRECTIONS
    }:
        raise ValueError('Step 8 cross-combined direction set mismatch.')


def _validate_summary_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    candidate_family: str,
) -> None:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != SUMMARY_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 8 summary schema version.')
    if payload.get('candidate_family') != candidate_family:
        raise ValueError('Step 8 summary candidate_family mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Step 8 summary downstream execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Step 8 summary downstream manifest hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 8 summary protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 8 summary preprocessing hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 8 summary feature hash mismatch.')
    if payload.get('partition_hash') != context['partition_sha256']:
        raise ValueError('Step 8 summary partition hash mismatch.')
    if not isinstance(payload.get('partitions'), list):
        raise ValueError('Step 8 summary partitions payload is missing.')


def _load_candidate_set(candidate_family: str) -> list[dict[str, Any]]:
    return _read_volume_json(_candidate_source_api_path(candidate_family))[:MAX_PARTITIONS]


def _load_candidate_set_from_root(
    volume_root: Path,
    candidate_family: str,
) -> list[dict[str, Any]]:
    return _read_root_json(volume_root, _candidate_source_api_path(candidate_family))[:MAX_PARTITIONS]


def _load_validated_source_within_with_retry(
    *,
    source_within_path: Path,
    context: dict[str, Any],
    source_condition: str,
    partition_metadata: dict[str, Any],
) -> dict[str, Any]:
    import time

    last_error: Exception | None = None
    for attempt in range(1, SOURCE_WITHIN_RETRY_ATTEMPTS + 1):
        volume.reload()
        if not source_within_path.exists():
            last_error = FileNotFoundError(
                f'Missing assembled within-condition results for {source_condition} '
                f'in {_partition_key(partition_metadata["partition_index"])}: {source_within_path}'
            )
        else:
            try:
                source_results = _load_json(source_within_path)
                _validate_within_assembled_payload(
                    payload=source_results,
                    context=context,
                    condition=source_condition,
                    partition_metadata=partition_metadata,
                )
                return source_results
            except (json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
                last_error = exc
        if attempt < SOURCE_WITHIN_RETRY_ATTEMPTS:
            time.sleep(SOURCE_WITHIN_RETRY_SLEEP_SECONDS)
    if last_error is None:
        raise RuntimeError(
            'Failed to validate assembled within-condition payload.')
    raise RuntimeError(
        f'Unable to validate assembled within-condition results for {source_condition} '
        f'after {SOURCE_WITHIN_RETRY_ATTEMPTS} reload attempts.'
    ) from last_error


def _build_step8_context() -> dict[str, Any]:
    from v4_downstream import build_authoritative_step12_context

    return build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=(
            'src/v4_downstream.py',
            'scripts/training/run_control_split_sensitivity_modal.py',
        ),
        repo_root=None,
    )


def _load_partition_condition_pool(
    *,
    features_path: Path,
    condition: str,
    control_a: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, int]:
    import polars as pl

    from features import get_feature_cols

    df = pl.read_csv(str(features_path))
    feature_cols = get_feature_cols('v4')
    pool = df.filter(
        (pl.col('condition') == condition) |
        pl.col('subject_id').is_in(control_a)
    )
    return (
        pool.select(feature_cols).to_numpy().astype(np.float64),
        pool['label'].to_numpy().astype(int),
        pool['subject_id'].to_numpy(),
        feature_cols,
        int(pool.n_unique('subject_id')),
        int(pool.shape[0]),
    )


def _expected_outer_fold_subject_ids(groups: np.ndarray) -> list[str]:
    from sklearn.model_selection import LeaveOneGroupOut

    splitter = LeaveOneGroupOut()
    dummy = np.zeros(len(groups), dtype=np.int8)
    return [
        str(groups[test_idx][0])
        for _, test_idx in splitter.split(dummy, dummy, groups)
    ]


def _inspect_within_classifier_state(
    *,
    volume_root: Path,
    context: dict[str, Any],
    candidate_family: str,
    partition_index: int,
    partition_metadata: dict[str, Any],
    condition: str,
    clf_name: str,
    expected_subject_ids: list[str],
) -> dict[str, Any]:
    partial_path = _partition_results_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    ) / _clf_partial_results_filename(condition, clf_name, partition_index)
    if not partial_path.exists():
        partial_state = 'missing'
    else:
        try:
            payload = _load_json(partial_path)
            _validate_partial_within_payload(
                payload=payload,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
            )
            partial_state = 'completed'
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            partial_state = 'invalid'

    per_fold_status: dict[str, str] = {}
    missing_folds: list[dict[str, Any]] = []
    invalid_folds: list[dict[str, Any]] = []
    completed_folds = 0
    for outer_fold_index, held_out_subject_id in enumerate(expected_subject_ids):
        fold_key = f'{outer_fold_index}:{held_out_subject_id}'
        fold_path = _volume_path(
            volume_root,
            _within_fold_shard_api_path(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
                partition_index=partition_index,
                condition=condition,
                clf_name=clf_name,
                outer_fold_index=outer_fold_index,
                held_out_subject_id=held_out_subject_id,
            ),
        )
        if not fold_path.exists():
            per_fold_status[fold_key] = 'missing'
            missing_folds.append(
                {
                    'outer_fold_index': outer_fold_index,
                    'held_out_subject_id': held_out_subject_id,
                    'path': str(fold_path),
                }
            )
            continue
        try:
            payload = _load_json(fold_path)
            _validate_within_fold_shard_payload(
                payload=payload,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
                outer_fold_index=outer_fold_index,
                held_out_subject_id=held_out_subject_id,
            )
            per_fold_status[fold_key] = 'completed'
            completed_folds += 1
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            per_fold_status[fold_key] = 'invalid'
            invalid_folds.append(
                {
                    'outer_fold_index': outer_fold_index,
                    'held_out_subject_id': held_out_subject_id,
                    'path': str(fold_path),
                }
            )

    full_source_path = _volume_path(
        volume_root,
        _within_full_source_shard_api_path(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
            partition_index=partition_index,
            condition=condition,
            clf_name=clf_name,
        ),
    )
    if not full_source_path.exists():
        full_source_state = 'missing'
    else:
        try:
            payload = _load_json(full_source_path)
            _validate_within_full_source_shard_payload(
                payload=payload,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
            )
            full_source_state = 'completed'
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            full_source_state = 'invalid'

    return {
        'partial_state': partial_state,
        'partial_path': str(partial_path),
        'expected_outer_folds': len(expected_subject_ids),
        'completed_outer_folds': completed_folds,
        'missing_outer_folds': len(missing_folds),
        'invalid_outer_folds': len(invalid_folds),
        'per_outer_fold': per_fold_status,
        'missing_fold_specs': missing_folds,
        'invalid_fold_specs': invalid_folds,
        'full_source_state': full_source_state,
        'full_source_path': str(full_source_path),
        'assemble_ready': (
            partial_state != 'completed'
            and completed_folds == len(expected_subject_ids)
            and not invalid_folds
            and full_source_state == 'completed'
        ),
    }


def _assemble_classifier_partial_from_shards(
    *,
    volume_root: Path,
    context: dict[str, Any],
    candidate_family: str,
    partition_index: int,
    partition_metadata: dict[str, Any],
    condition: str,
    clf_name: str,
    control_a: list[str],
) -> dict[str, Any]:
    from train import (
        DEFAULT_SUBJECT_PROBABILITY_THRESHOLD,
        _build_within_condition_output,
        _strategy_to_legacy_label,
        assemble_grouped_within_classifier_result,
    )
    from v4_downstream import write_payload_json

    partition_results_dir = _partition_results_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    )
    partition_models_dir = _partition_models_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    )
    partition_results_dir.mkdir(parents=True, exist_ok=True)
    partition_models_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partition_results_dir / _clf_partial_results_filename(
        condition,
        clf_name,
        partition_index,
    )
    if partial_path.exists():
        try:
            existing = _load_json(partial_path)
            _validate_partial_within_payload(
                payload=existing,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
            )
            return {'status': 'skipped', 'output_path': str(partial_path)}
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    features_path = volume_root / 'processed_v4' / 'gait_features_v4.csv'
    X, y, groups, feature_cols, pool_subjects, pool_strides = _load_partition_condition_pool(
        features_path=features_path,
        condition=condition,
        control_a=control_a,
    )
    expected_subject_ids = _expected_outer_fold_subject_ids(groups)
    fold_payloads: list[dict[str, Any]] = []
    for outer_fold_index, held_out_subject_id in enumerate(expected_subject_ids):
        fold_path = _volume_path(
            volume_root,
            _within_fold_shard_api_path(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
                partition_index=partition_index,
                condition=condition,
                clf_name=clf_name,
                outer_fold_index=outer_fold_index,
                held_out_subject_id=held_out_subject_id,
            ),
        )
        if not fold_path.exists():
            raise FileNotFoundError(
                f'Missing Step 8 within-fold shard: {fold_path}'
            )
        fold_payload = _load_json(fold_path)
        _validate_within_fold_shard_payload(
            payload=fold_payload,
            context=context,
            condition=condition,
            clf_name=clf_name,
            partition_metadata=partition_metadata,
            outer_fold_index=outer_fold_index,
            held_out_subject_id=held_out_subject_id,
        )
        fold_payloads.append(fold_payload)

    full_source_path = _volume_path(
        volume_root,
        _within_full_source_shard_api_path(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
            partition_index=partition_index,
            condition=condition,
            clf_name=clf_name,
        ),
    )
    if not full_source_path.exists():
        raise FileNotFoundError(
            f'Missing Step 8 full-source shard: {full_source_path}'
        )
    full_source_payload = _load_json(full_source_path)
    _validate_within_full_source_shard_payload(
        payload=full_source_payload,
        context=context,
        condition=condition,
        clf_name=clf_name,
        partition_metadata=partition_metadata,
    )

    classifier_result = assemble_grouped_within_classifier_result(
        fold_outputs=fold_payloads,
        full_source_artifacts=full_source_payload,
        candidate_imbalance_strategies=tuple(STRATEGY_POLICY[clf_name]),
        subject_aggregation_rule=full_source_payload['subject_aggregation_rule'],
        tie_break_rule=full_source_payload['tie_break_rule'],
    )
    clf_results = {
        clf_name: {
            'f1_macro': classifier_result['f1_macro'],
            'f1_macro_ci_lower': classifier_result['f1_macro_ci_lower'],
            'f1_macro_ci_upper': classifier_result['f1_macro_ci_upper'],
            'subject_resampled_stride_f1_ci_lower': classifier_result['subject_resampled_stride_f1_ci_lower'],
            'subject_resampled_stride_f1_ci_upper': classifier_result['subject_resampled_stride_f1_ci_upper'],
            'subject_primary_f1_ci_lower': classifier_result['subject_primary_f1_ci_lower'],
            'subject_primary_f1_ci_upper': classifier_result['subject_primary_f1_ci_upper'],
            'subject_primary_f1_macro': classifier_result['subject_primary_f1_macro'],
            'subject_aggregation_rule': classifier_result['subject_aggregation_rule'],
            'tie_break_rule': classifier_result['tie_break_rule'],
            'modal_params': classifier_result['modal_params'],
            'modal_frequency': classifier_result['modal_frequency'],
            'modal_strategy': classifier_result['modal_strategy'],
            'modal_strategy_frequency': classifier_result['modal_strategy_frequency'],
            'subject_metrics': classifier_result['subject_metrics'],
            'y_true': classifier_result['y_true'],
            'y_pred': classifier_result['y_pred'],
            'y_prob': classifier_result['y_prob'],
            'selected_resampling': _strategy_to_legacy_label(
                classifier_result['full_source_selected_imbalance_strategy']),
            'selected_imbalance_strategy': classifier_result['full_source_selected_imbalance_strategy'],
            'candidate_imbalance_strategies': classifier_result['candidate_imbalance_strategies'],
            'outer_fold_selection_trace': classifier_result['outer_fold_selection_trace'],
            'full_source_selected_params': classifier_result['full_source_selected_params'],
            'full_source_selected_imbalance_strategy': classifier_result['full_source_selected_imbalance_strategy'],
            'full_source_selection_subject_f1': classifier_result['full_source_selection_subject_f1'],
            'full_source_selection_stride_f1': classifier_result['full_source_selection_stride_f1'],
            'full_source_selection_subject_log_loss': classifier_result['full_source_selection_subject_log_loss'],
            'full_source_selection_trace': classifier_result['full_source_selection_trace'],
            'full_source_selection_trace_path': classifier_result['full_source_selection_trace_path'],
            'full_source_selection_trace_sha256': classifier_result['full_source_selection_trace_sha256'],
            'full_source_model_path': classifier_result['full_source_model_path'],
            'full_source_model_sha256': classifier_result['full_source_model_sha256'],
        }
    }
    partial_output = _build_within_condition_output(
        condition=condition,
        pool_subjects=pool_subjects,
        pool_strides=pool_strides,
        selected_feature_cols=feature_cols,
        feature_matrix_file='v4/gait_features_v4.csv',
        feature_set_version='v4',
        normalization='none',
        models_dir=str(partition_models_dir),
        clf_results=clf_results,
    )
    partial_output['candidate_strategy_policy'] = {clf_name: list(STRATEGY_POLICY[clf_name])}
    partial_output['subject_aggregation_rule'] = classifier_result['subject_aggregation_rule']
    partial_output['subject_probability_threshold'] = DEFAULT_SUBJECT_PROBABILITY_THRESHOLD
    partial_output['tie_break_rule'] = classifier_result['tie_break_rule']
    partial_output['feature_matrix_hash'] = context['feature_matrix_sha256']
    partial_output['partition_hash'] = partition_metadata['partition_hash']
    partial_output['protocol_manifest_hash'] = context['protocol_manifest_sha256']
    partial_output['preprocessing_manifest_hash'] = context['preprocessing_manifest_sha256']
    partial_output['schema_version'] = WITHIN_PARTIAL_SCHEMA_VERSION
    partial_output['candidate_family'] = candidate_family
    partial_output['partition_index'] = partition_index
    partial_output['partition_key'] = _partition_key(partition_index)
    partial_output['control_A'] = list(control_a)
    partial_output['control_B'] = list(partition_metadata['control_B'])
    partial_output['downstream_execution_id'] = context['downstream_execution_id']
    partial_output['downstream_execution_manifest_hash'] = context['downstream_manifest_sha256']
    partial_output['partition_metadata'] = partition_metadata
    write_payload_json(partial_path, partial_output)
    _validate_partial_within_payload(
        payload=_load_json(partial_path),
        context=context,
        condition=condition,
        clf_name=clf_name,
        partition_metadata=partition_metadata,
    )
    return {'status': 'completed', 'output_path': str(partial_path)}


@app.function(
    cpu=WITHIN_FOLD_SHARD_CPU,
    memory=WITHIN_FOLD_SHARD_MEMORY_MB,
    timeout=WITHIN_FOLD_SHARD_TIMEOUT_SECONDS,
    max_containers=WITHIN_FOLD_SHARD_MAX_CONTAINERS,
    volumes={'/results': volume},
    retries=3,
)
def run_partition_condition_outer_fold_shard(
    partition_index: int,
    condition: str,
    clf_name: str,
    control_a: list[str],
    control_b: list[str],
    candidate_family: str,
    partition_metadata: dict[str, Any],
    outer_fold_index: int,
    held_out_subject_id: str,
) -> str:
    from train import (
        candidate_strategies_for_classifier,
        get_classifier_configs,
        run_grouped_outer_fold,
    )
    from v4_downstream import write_payload_json

    start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f'[step8 fold] partition={partition_index} condition={condition} clf={clf_name} '
        f'fold={outer_fold_index:02d} held_out={held_out_subject_id} start={start_ts}',
        flush=True,
    )
    volume.reload()
    context = _build_step8_context()
    volume_root = Path('/results')
    partition_results_dir = _partition_results_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    )
    partition_models_dir = _partition_models_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    )
    partition_results_dir.mkdir(parents=True, exist_ok=True)
    partition_models_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partition_results_dir / _clf_partial_results_filename(
        condition,
        clf_name,
        partition_index,
    )
    if partial_path.exists():
        try:
            existing = _load_json(partial_path)
            _validate_partial_within_payload(
                payload=existing,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
            )
            return json.dumps(
                {'status': 'skipped', 'reason': 'existing_valid_classifier_partial', 'output_path': str(partial_path)},
                indent=2,
            )
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    shard_path = _volume_path(
        volume_root,
        _within_fold_shard_api_path(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
            partition_index=partition_index,
            condition=condition,
            clf_name=clf_name,
            outer_fold_index=outer_fold_index,
            held_out_subject_id=held_out_subject_id,
        ),
    )
    if shard_path.exists():
        try:
            existing = _load_json(shard_path)
            _validate_within_fold_shard_payload(
                payload=existing,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
                outer_fold_index=outer_fold_index,
                held_out_subject_id=held_out_subject_id,
            )
            return json.dumps(
                {'status': 'skipped', 'reason': 'existing_valid_outer_fold_shard', 'output_path': str(shard_path)},
                indent=2,
            )
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    features_path = volume_root / 'processed_v4' / 'gait_features_v4.csv'
    X, y, groups, _, _, _ = _load_partition_condition_pool(
        features_path=features_path,
        condition=condition,
        control_a=control_a,
    )
    expected_subject_ids = _expected_outer_fold_subject_ids(groups)
    if expected_subject_ids[outer_fold_index] != held_out_subject_id:
        raise ValueError(
            'Deterministic outer-fold subject mismatch: '
            f'expected={expected_subject_ids[outer_fold_index]}, '
            f'got={held_out_subject_id}'
        )

    config = get_classifier_configs()[clf_name]
    candidate_strategies = candidate_strategies_for_classifier(
        clf_name,
        STRATEGY_POLICY,
    )
    fold_output = run_grouped_outer_fold(
        condition=condition,
        clf_name=clf_name,
        clf_template=config['clf'],
        param_grid=config['param_grid'],
        X=X,
        y=y,
        groups=groups,
        outer_fold_index=outer_fold_index,
        candidate_imbalance_strategies=candidate_strategies,
        models_dir=partition_models_dir,
        selection_trace_dir=partition_results_dir / 'selection_traces',
        subject_aggregation_rule=context['step2_payloads'][condition]['subject_aggregation_rule'],
        tie_break_rule=context['step2_payloads'][condition]['tie_break_rule'],
    )
    payload = {
        'schema_version': WITHIN_FOLD_SHARD_SCHEMA_VERSION,
        'condition': condition,
        'classifier': clf_name,
        'candidate_family': candidate_family,
        'partition_index': partition_index,
        'partition_key': _partition_key(partition_index),
        'partition_hash': partition_metadata['partition_hash'],
        'control_A': list(control_a),
        'control_B': list(control_b),
        'models_dir': str(partition_models_dir),
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'downstream_execution_id': context['downstream_execution_id'],
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'partition_metadata': partition_metadata,
        **fold_output,
    }
    write_payload_json(shard_path, payload)
    volume.commit()
    end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f'[step8 fold] partition={partition_index} condition={condition} clf={clf_name} '
        f'fold={outer_fold_index:02d} held_out={held_out_subject_id} '
        f'selected={fold_output["selected_imbalance_strategy"]}/{fold_output["selected_params"]} '
        f'end={end_ts} shard={shard_path}',
        flush=True,
    )
    return json.dumps({'status': 'completed', 'output_path': str(shard_path)}, indent=2)


@app.function(
    cpu=WITHIN_FULL_SOURCE_CPU,
    memory=WITHIN_FULL_SOURCE_MEMORY_MB,
    timeout=WITHIN_FULL_SOURCE_TIMEOUT_SECONDS,
    max_containers=WITHIN_FULL_SOURCE_MAX_CONTAINERS,
    volumes={'/results': volume},
    retries=2,
)
def run_partition_condition_full_source_shard(
    partition_index: int,
    condition: str,
    clf_name: str,
    control_a: list[str],
    control_b: list[str],
    candidate_family: str,
    partition_metadata: dict[str, Any],
) -> str:
    from train import (
        candidate_strategies_for_classifier,
        fit_grouped_full_source_with_artifacts,
        get_classifier_configs,
    )
    from v4_downstream import write_payload_json

    start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f'[step8 full-source] partition={partition_index} condition={condition} '
        f'clf={clf_name} start={start_ts}',
        flush=True,
    )
    volume.reload()
    context = _build_step8_context()
    volume_root = Path('/results')
    partition_results_dir = _partition_results_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    )
    partition_models_dir = _partition_models_dir(
        volume_root=volume_root,
        execution_id=context['downstream_execution_id'],
        candidate_family=candidate_family,
        partition_index=partition_index,
    )
    partition_results_dir.mkdir(parents=True, exist_ok=True)
    partition_models_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partition_results_dir / _clf_partial_results_filename(
        condition,
        clf_name,
        partition_index,
    )
    if partial_path.exists():
        try:
            existing = _load_json(partial_path)
            _validate_partial_within_payload(
                payload=existing,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
            )
            return json.dumps(
                {'status': 'skipped', 'reason': 'existing_valid_classifier_partial', 'output_path': str(partial_path)},
                indent=2,
            )
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    shard_path = _volume_path(
        volume_root,
        _within_full_source_shard_api_path(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
            partition_index=partition_index,
            condition=condition,
            clf_name=clf_name,
        ),
    )
    if shard_path.exists():
        try:
            existing = _load_json(shard_path)
            _validate_within_full_source_shard_payload(
                payload=existing,
                context=context,
                condition=condition,
                clf_name=clf_name,
                partition_metadata=partition_metadata,
            )
            return json.dumps(
                {'status': 'skipped', 'reason': 'existing_valid_full_source_shard', 'output_path': str(shard_path)},
                indent=2,
            )
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    features_path = volume_root / 'processed_v4' / 'gait_features_v4.csv'
    X, y, groups, _, _, _ = _load_partition_condition_pool(
        features_path=features_path,
        condition=condition,
        control_a=control_a,
    )
    config = get_classifier_configs()[clf_name]
    candidate_strategies = candidate_strategies_for_classifier(
        clf_name,
        STRATEGY_POLICY,
    )
    full_source_output = fit_grouped_full_source_with_artifacts(
        condition=condition,
        clf_name=clf_name,
        clf=config['clf'],
        param_grid=config['param_grid'],
        X=X,
        y=y,
        groups=groups,
        candidate_imbalance_strategies=candidate_strategies,
        models_dir=partition_models_dir,
        selection_trace_dir=partition_results_dir / 'selection_traces',
        subject_aggregation_rule=context['step2_payloads'][condition]['subject_aggregation_rule'],
        tie_break_rule=context['step2_payloads'][condition]['tie_break_rule'],
    )
    payload = {
        'schema_version': WITHIN_FULL_SOURCE_SHARD_SCHEMA_VERSION,
        'condition': condition,
        'classifier': clf_name,
        'candidate_family': candidate_family,
        'partition_index': partition_index,
        'partition_key': _partition_key(partition_index),
        'partition_hash': partition_metadata['partition_hash'],
        'control_A': list(control_a),
        'control_B': list(control_b),
        'models_dir': str(partition_models_dir),
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'downstream_execution_id': context['downstream_execution_id'],
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'partition_metadata': partition_metadata,
        **full_source_output,
    }
    write_payload_json(shard_path, payload)
    volume.commit()
    end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f'[step8 full-source] partition={partition_index} condition={condition} '
        f'clf={clf_name} selected={full_source_output["selected_imbalance_strategy"]}/'
        f'{full_source_output["selected_params"]} end={end_ts} shard={shard_path}',
        flush=True,
    )
    return json.dumps({'status': 'completed', 'output_path': str(shard_path)}, indent=2)


@app.function(
    cpu=4,
    memory=6144,
    timeout=43200,
    volumes={'/results': volume},
    retries=2,
    max_containers=3,
)
def run_partition_direction(
    partition_index: int,
    source_condition: str,
    target_condition: str,
    control_a: list[str],
    control_b: list[str],
    candidate_family: str,
    partition_metadata: dict[str, Any],
) -> str:
    import polars as pl

    from features import get_feature_cols
    from train import run_cross_condition
    from v4_downstream import (
        DIRECTIONS,
        build_authoritative_step12_context,
        load_json,
        validate_step3_results_payload,
        write_payload_json,
    )

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=(
            'src/v4_downstream.py',
            'scripts/training/run_control_split_sensitivity_modal.py',
        ),
        repo_root=None,
    )
    step3_payload = load_json(
        context['authoritative_results_dir'] / 'cross_condition_results_v4.json')
    validate_step3_results_payload(
        payload=step3_payload,
        output_namespace='results_v4',
        expected_directions=DIRECTIONS,
        context=context,
    )
    direction_key = f'{source_condition}_to_{target_condition}'
    results_root = Path(
        _run_results_root(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
        )
    )
    models_root = Path(
        _run_models_root(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
        )
    )
    partition_results_dir = results_root / _partition_key(partition_index)
    partition_models_dir = models_root / _partition_key(partition_index)
    partition_results_dir.mkdir(parents=True, exist_ok=True)
    partition_models_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_results_dir / _cross_partial_filename(
        partition_index,
        source_condition,
        target_condition,
    )
    if output_path.exists():
        try:
            existing = _load_json(output_path)
            _validate_cross_partial_payload(
                payload=existing,
                context=context,
                direction_key=direction_key,
                partition_metadata=partition_metadata,
            )
            return json.dumps({'status': 'skipped', 'output_path': str(output_path)}, indent=2)
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    features_path = Path('/results/processed_v4/gait_features_v4.csv')
    df = pl.read_csv(str(features_path))
    feature_cols = get_feature_cols('v4')
    source_within_path = partition_results_dir / _within_results_filename(
        source_condition,
        partition_index,
    )
    source_results = _load_validated_source_within_with_retry(
        source_within_path=source_within_path,
        context=context,
        source_condition=source_condition,
        partition_metadata=partition_metadata,
    )

    output = run_cross_condition(
        source_condition=source_condition,
        target_condition=target_condition,
        df=df,
        control_a=control_a,
        control_b=control_b,
        source_results=source_results,
        results_dir=partition_results_dir,
        models_dir=partition_models_dir,
        feature_cols=feature_cols,
        feature_matrix_file='v4/gait_features_v4.csv',
        feature_set_version='v4',
        normalization='none',
        allow_refit=False,
        protocol_manifest_hash=context['protocol_manifest_sha256'],
        preprocessing_manifest_hash=context['preprocessing_manifest_sha256'],
    )
    output['source_condition'] = source_condition
    output['target_condition'] = target_condition
    output['schema_version'] = CROSS_PARTIAL_SCHEMA_VERSION
    output['candidate_family'] = candidate_family
    output['partition_index'] = partition_index
    output['partition_key'] = _partition_key(partition_index)
    output['partition_hash'] = partition_metadata['partition_hash']
    output['feature_matrix_hash'] = context['feature_matrix_sha256']
    output['control_A'] = list(control_a)
    output['control_B'] = list(control_b)
    output['downstream_execution_id'] = context['downstream_execution_id']
    output['downstream_execution_manifest_hash'] = context['downstream_manifest_sha256']
    output['partition_metadata'] = partition_metadata
    write_payload_json(output_path, output)
    volume.commit()
    return json.dumps({'status': 'completed', 'output_path': str(output_path)}, indent=2)

def _iter_within_classifier_states(
    *,
    volume_root: Path,
    context: dict[str, Any],
    candidate_family: str,
):
    import polars as pl

    main_partition = _read_root_json(volume_root, 'processed_v4/control_partition_v4.json')
    candidates = _load_candidate_set_from_root(volume_root, candidate_family)
    if not candidates:
        raise RuntimeError(
            f'No candidates available for candidate_family={candidate_family!r}.')

    subject_order_df = pl.read_csv(
        str(volume_root / 'processed_v4' / 'gait_features_v4.csv'),
        columns=['condition', 'subject_id'],
    )
    for idx, candidate in enumerate(candidates, start=1):
        partition_metadata = _candidate_metadata(
            candidate=candidate,
            candidate_family=candidate_family,
            partition_index=idx,
            main_partition=main_partition,
        )
        for condition in CONDITIONS:
            pool = subject_order_df.filter(
                (pl.col('condition') == condition) |
                pl.col('subject_id').is_in(candidate['control_A'])
            )
            expected_subject_ids = _expected_outer_fold_subject_ids(
                pool['subject_id'].to_numpy()
            )
            for clf_name in CLF_ORDER:
                state = _inspect_within_classifier_state(
                    volume_root=volume_root,
                    context=context,
                    candidate_family=candidate_family,
                    partition_index=idx,
                    partition_metadata=partition_metadata,
                    condition=condition,
                    clf_name=clf_name,
                    expected_subject_ids=expected_subject_ids,
                )
                yield {
                    'partition_index': idx,
                    'partition_key': _partition_key(idx),
                    'candidate': candidate,
                    'partition_metadata': partition_metadata,
                    'condition': condition,
                    'classifier': clf_name,
                    'expected_subject_ids': expected_subject_ids,
                    'state': state,
                }


def _assemble_missing_within_partials(
    *,
    volume_root: Path,
    context: dict[str, Any],
    candidate_family: str,
) -> list[str]:
    assembled_paths: list[str] = []
    incomplete: list[str] = []
    for entry in _iter_within_classifier_states(
        volume_root=volume_root,
        context=context,
        candidate_family=candidate_family,
    ):
        state = entry['state']
        if state['partial_state'] == 'completed':
            continue
        key = f'{entry["partition_key"]}:{entry["condition"]}:{entry["classifier"]}'
        if not state['assemble_ready']:
            incomplete.append(key)
            continue
        result = _assemble_classifier_partial_from_shards(
            volume_root=volume_root,
            context=context,
            candidate_family=candidate_family,
            partition_index=entry['partition_index'],
            partition_metadata=entry['partition_metadata'],
            condition=entry['condition'],
            clf_name=entry['classifier'],
            control_a=entry['candidate']['control_A'],
        )
        if result['status'] != 'skipped':
            assembled_paths.append(result['output_path'])
    if incomplete:
        raise RuntimeError(
            'Missing or invalid Step 8 within shards prevent classifier-partial assembly: '
            + ', '.join(incomplete)
        )
    return assembled_paths


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def retune_dry_run_remote(candidate_family: str = 'near_optimal') -> str:
    volume.reload()
    context = _build_step8_context()
    volume_root = Path('/results')
    completed = 0
    missing = 0
    invalid = 0
    pending_fold_shards: list[str] = []
    pending_full_source_shards: list[str] = []
    missing_classifier_partials: list[str] = []
    per_classifier: dict[str, Any] = {}

    for entry in _iter_within_classifier_states(
        volume_root=volume_root,
        context=context,
        candidate_family=candidate_family,
    ):
        key = f'{entry["partition_key"]}:{entry["condition"]}:{entry["classifier"]}'
        state = entry['state']
        per_classifier[key] = state
        if state['partial_state'] == 'completed':
            completed += 1
            continue
        if state['partial_state'] == 'invalid':
            invalid += 1
        else:
            missing += 1
        missing_classifier_partials.append(key)
        for spec in state['missing_fold_specs'] + state['invalid_fold_specs']:
            pending_fold_shards.append(
                f'{key}:fold_{spec["outer_fold_index"]:02d}:{spec["held_out_subject_id"]}'
            )
        if state['full_source_state'] != 'completed':
            pending_full_source_shards.append(key)

    return json.dumps(
        {
            'status': 'dry_run',
            'candidate_family': candidate_family,
            'downstream_execution_id': context['downstream_execution_id'],
            'completed_classifier_partials_will_be_skipped': True,
            'within_classifier_partials': {
                'expected': MAX_PARTITIONS * len(CONDITIONS) * len(CLF_ORDER),
                'completed': completed,
                'missing': missing,
                'invalid': invalid,
                'missing_list': missing_classifier_partials,
            },
            'pending_outer_fold_shards': pending_fold_shards,
            'pending_full_source_shards': pending_full_source_shards,
            'resource_ceiling': {
                'within_outer_fold_worker': {
                    'cpu': WITHIN_FOLD_SHARD_CPU,
                    'memory_mb': WITHIN_FOLD_SHARD_MEMORY_MB,
                    'timeout_seconds': WITHIN_FOLD_SHARD_TIMEOUT_SECONDS,
                    'max_containers': WITHIN_FOLD_SHARD_MAX_CONTAINERS,
                    'max_reserved_cores': WITHIN_FOLD_SHARD_CPU * WITHIN_FOLD_SHARD_MAX_CONTAINERS,
                    'max_reserved_memory_mb': WITHIN_FOLD_SHARD_MEMORY_MB * WITHIN_FOLD_SHARD_MAX_CONTAINERS,
                },
                'within_full_source_worker': {
                    'cpu': WITHIN_FULL_SOURCE_CPU,
                    'memory_mb': WITHIN_FULL_SOURCE_MEMORY_MB,
                    'timeout_seconds': WITHIN_FULL_SOURCE_TIMEOUT_SECONDS,
                    'max_containers': WITHIN_FULL_SOURCE_MAX_CONTAINERS,
                    'max_reserved_cores': WITHIN_FULL_SOURCE_CPU * WITHIN_FULL_SOURCE_MAX_CONTAINERS,
                    'max_reserved_memory_mb': WITHIN_FULL_SOURCE_MEMORY_MB * WITHIN_FULL_SOURCE_MAX_CONTAINERS,
                },
                'combined_max_reserved_cores': (
                    WITHIN_FOLD_SHARD_CPU * WITHIN_FOLD_SHARD_MAX_CONTAINERS
                    + WITHIN_FULL_SOURCE_CPU * WITHIN_FULL_SOURCE_MAX_CONTAINERS
                ),
                'combined_max_reserved_memory_mb': (
                    WITHIN_FOLD_SHARD_MEMORY_MB * WITHIN_FOLD_SHARD_MAX_CONTAINERS
                    + WITHIN_FULL_SOURCE_MEMORY_MB * WITHIN_FULL_SOURCE_MAX_CONTAINERS
                ),
            },
            'message': (
                'Only missing or invalid outer-fold/full-source shards would be submitted. '
                'Existing valid classifier partials will be skipped.'
            ),
        },
        indent=2,
    )


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def submit_retune_within_remote(candidate_family: str = 'near_optimal') -> str:
    volume.reload()
    context = _build_step8_context()
    volume_root = Path('/results')
    submitted_fold_shards: list[str] = []
    submitted_full_source_shards: list[str] = []
    skipped_completed_partials: list[str] = []

    for entry in _iter_within_classifier_states(
        volume_root=volume_root,
        context=context,
        candidate_family=candidate_family,
    ):
        key = f'{entry["partition_key"]}:{entry["condition"]}:{entry["classifier"]}'
        state = entry['state']
        if state['partial_state'] == 'completed':
            skipped_completed_partials.append(key)
            continue
        for spec in state['missing_fold_specs'] + state['invalid_fold_specs']:
            run_partition_condition_outer_fold_shard.spawn(
                partition_index=entry['partition_index'],
                condition=entry['condition'],
                clf_name=entry['classifier'],
                control_a=entry['candidate']['control_A'],
                control_b=entry['candidate']['control_B'],
                candidate_family=candidate_family,
                partition_metadata=entry['partition_metadata'],
                outer_fold_index=spec['outer_fold_index'],
                held_out_subject_id=spec['held_out_subject_id'],
            )
            submitted_fold_shards.append(
                f'{key}:fold_{spec["outer_fold_index"]:02d}:{spec["held_out_subject_id"]}'
            )
        if state['full_source_state'] != 'completed':
            run_partition_condition_full_source_shard.spawn(
                partition_index=entry['partition_index'],
                condition=entry['condition'],
                clf_name=entry['classifier'],
                control_a=entry['candidate']['control_A'],
                control_b=entry['candidate']['control_B'],
                candidate_family=candidate_family,
                partition_metadata=entry['partition_metadata'],
            )
            submitted_full_source_shards.append(key)

    return json.dumps(
        {
            'status': 'submitted',
            'downstream_execution_id': context['downstream_execution_id'],
            'candidate_family': candidate_family,
            'submitted_outer_fold_shards': submitted_fold_shards,
            'submitted_full_source_shards': submitted_full_source_shards,
            'skipped_completed_classifier_partials': skipped_completed_partials,
        },
        indent=2,
    )


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def assemble_within_remote(candidate_family: str = 'near_optimal') -> str:
    from v4_downstream import write_payload_json

    volume.reload()
    context = _build_step8_context()
    volume_root = Path('/results')
    assembled_partial_paths = _assemble_missing_within_partials(
        volume_root=volume_root,
        context=context,
        candidate_family=candidate_family,
    )
    main_partition = _read_volume_json('processed_v4/control_partition_v4.json')
    candidates = _load_candidate_set(candidate_family)
    assembled_paths: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        partition_metadata = _candidate_metadata(
            candidate=candidate,
            candidate_family=candidate_family,
            partition_index=idx,
            main_partition=main_partition,
        )
        partition_root = _partition_results_dir(
            volume_root=volume_root,
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
            partition_index=idx,
        )
        for condition in CONDITIONS:
            first_partial_path = partition_root / _clf_partial_results_filename(
                condition,
                CLF_ORDER[0],
                idx,
            )
            if not first_partial_path.exists():
                raise FileNotFoundError(
                    f'Missing Step 8 first classifier partial: {first_partial_path}')
            first_partial = _load_json(first_partial_path)
            _validate_partial_within_payload(
                payload=first_partial,
                context=context,
                condition=condition,
                clf_name=CLF_ORDER[0],
                partition_metadata=partition_metadata,
            )
            assembled = {
                'schema_version': WITHIN_ASSEMBLED_SCHEMA_VERSION,
                'condition': first_partial['condition'],
                'candidate_family': candidate_family,
                'partition_index': idx,
                'partition_key': _partition_key(idx),
                'partition_hash': partition_metadata['partition_hash'],
                'control_A': partition_metadata['control_A'],
                'control_B': partition_metadata['control_B'],
                'downstream_execution_id': context['downstream_execution_id'],
                'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
                'partition_metadata': partition_metadata,
                'pool_subjects': first_partial['pool_subjects'],
                'pool_strides': first_partial['pool_strides'],
                'feature_cols': first_partial['feature_cols'],
                'n_features': first_partial['n_features'],
                'feature_matrix_file': first_partial['feature_matrix_file'],
                'feature_set_version': first_partial['feature_set_version'],
                'normalization': first_partial['normalization'],
                'feature_matrix_hash': first_partial['feature_matrix_hash'],
                'protocol_manifest_hash': first_partial['protocol_manifest_hash'],
                'preprocessing_manifest_hash': first_partial['preprocessing_manifest_hash'],
                'classifiers': {},
            }
            for clf_name in CLF_ORDER:
                clf_partial_path = partition_root / _clf_partial_results_filename(
                    condition,
                    clf_name,
                    idx,
                )
                if not clf_partial_path.exists():
                    raise FileNotFoundError(
                        f'Missing Step 8 classifier partial: {clf_partial_path}')
                clf_partial = _load_json(clf_partial_path)
                _validate_partial_within_payload(
                    payload=clf_partial,
                    context=context,
                    condition=condition,
                    clf_name=clf_name,
                    partition_metadata=partition_metadata,
                )
                assembled['classifiers'][clf_name] = clf_partial['classifiers'][clf_name]
            assembled_path = partition_root / _within_results_filename(condition, idx)
            write_payload_json(assembled_path, assembled)
            assembled_paths.append(str(assembled_path))

    volume.commit()
    return json.dumps(
        {
            'status': 'completed',
            'downstream_execution_id': context['downstream_execution_id'],
            'candidate_family': candidate_family,
            'assembled_classifier_partial_paths': assembled_partial_paths,
            'assembled_within_paths': assembled_paths,
        },
        indent=2,
    )


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def assemble_within_missing_remote(candidate_family: str = 'near_optimal') -> str:
    volume.reload()
    context = _build_step8_context()
    assembled_partial_paths = _assemble_missing_within_partials(
        volume_root=Path('/results'),
        context=context,
        candidate_family=candidate_family,
    )
    volume.commit()
    return json.dumps(
        {
            'status': 'completed',
            'downstream_execution_id': context['downstream_execution_id'],
            'candidate_family': candidate_family,
            'assembled_classifier_partial_paths': assembled_partial_paths,
        },
        indent=2,
    )


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def submit_retune_cross_remote(candidate_family: str = 'near_optimal') -> str:
    from v4_downstream import build_authoritative_step12_context

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=(
            'scripts/training/run_control_split_sensitivity_modal.py',),
        repo_root=None,
    )
    main_partition = _read_volume_json(
        'processed_v4/control_partition_v4.json')
    candidates = _load_candidate_set(candidate_family)
    if not candidates:
        raise RuntimeError(
            f'No candidates available for candidate_family={candidate_family!r}.')

    for idx, candidate in enumerate(candidates, start=1):
        partition_metadata = _candidate_metadata(
            candidate=candidate,
            candidate_family=candidate_family,
            partition_index=idx,
            main_partition=main_partition,
        )
        for source_condition in CONDITIONS:
            assembled_path = Path('/results') / _within_assembled_api_path(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
                partition_index=idx,
                condition=source_condition,
            )
            if not assembled_path.exists():
                raise FileNotFoundError(
                    'retune-submit-cross requires assembled within-condition JSONs first: '
                    f'{assembled_path}'
                )
            assembled_payload = _load_json(assembled_path)
            _validate_within_assembled_payload(
                payload=assembled_payload,
                context=context,
                condition=source_condition,
                partition_metadata=partition_metadata,
            )

    submitted_cross: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        partition_metadata = _candidate_metadata(
            candidate=candidate,
            candidate_family=candidate_family,
            partition_index=idx,
            main_partition=main_partition,
        )
        for source_condition, target_condition in DIRECTIONS:
            partial_path = Path('/results') / _cross_partial_api_path(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
                partition_index=idx,
                source_condition=source_condition,
                target_condition=target_condition,
            )
            needs_submit = True
            if partial_path.exists():
                try:
                    existing = _load_json(partial_path)
                    _validate_cross_partial_payload(
                        payload=existing,
                        context=context,
                        direction_key=f'{source_condition}_to_{target_condition}',
                        partition_metadata=partition_metadata,
                    )
                    needs_submit = False
                except (json.JSONDecodeError, OSError, ValueError, KeyError):
                    needs_submit = True
            if needs_submit:
                run_partition_direction.spawn(
                    partition_index=idx,
                    source_condition=source_condition,
                    target_condition=target_condition,
                    control_a=candidate['control_A'],
                    control_b=candidate['control_B'],
                    candidate_family=candidate_family,
                    partition_metadata=partition_metadata,
                )
                submitted_cross.append(
                    f'{idx}:{source_condition}_to_{target_condition}')

    return json.dumps(
        {
            'status': 'submitted',
            'downstream_execution_id': context['downstream_execution_id'],
            'candidate_family': candidate_family,
            'submitted_cross_jobs': submitted_cross,
        },
        indent=2,
    )


@app.function(
    cpu=2,
    memory=4096,
    timeout=3600,
    volumes={'/results': volume},
    retries=1,
)
def collect_retune_status_remote(candidate_family: str = 'near_optimal') -> str:
    volume.reload()
    context = _build_step8_context()
    volume_root = Path('/results')
    main_partition = _read_volume_json('processed_v4/control_partition_v4.json')
    candidates = _load_candidate_set(candidate_family)
    status: dict[str, Any] = {
        'candidate_family': candidate_family,
        'downstream_execution_id': context['downstream_execution_id'],
        'within_classifier_partials': {
            'expected': MAX_PARTITIONS * len(CONDITIONS) * len(CLF_ORDER),
            'completed': 0,
            'missing': 0,
            'invalid': 0,
            'missing_list': [],
        },
        'within_missing_details': {},
        'within_partials': {},
        'within_assembled': {},
        'cross_partials': {},
        'cross_combined': {},
        'summary': 'missing',
    }

    states_by_partition: dict[str, list[dict[str, Any]]] = {
        _partition_key(idx): []
        for idx in range(1, len(candidates) + 1)
    }
    for entry in _iter_within_classifier_states(
        volume_root=volume_root,
        context=context,
        candidate_family=candidate_family,
    ):
        partition_key = entry['partition_key']
        key = f'{entry["condition"]}:{entry["classifier"]}'
        states_by_partition[partition_key].append(entry)
        status['within_partials'].setdefault(partition_key, {})[key] = entry['state']['partial_state']
        if entry['state']['partial_state'] == 'completed':
            status['within_classifier_partials']['completed'] += 1
        elif entry['state']['partial_state'] == 'invalid':
            status['within_classifier_partials']['invalid'] += 1
            status['within_classifier_partials']['missing_list'].append(
                f'{partition_key}:{entry["condition"]}:{entry["classifier"]}'
            )
            status['within_missing_details'][
                f'{partition_key}:{entry["condition"]}:{entry["classifier"]}'
            ] = entry['state']
        else:
            status['within_classifier_partials']['missing'] += 1
            status['within_classifier_partials']['missing_list'].append(
                f'{partition_key}:{entry["condition"]}:{entry["classifier"]}'
            )
            status['within_missing_details'][
                f'{partition_key}:{entry["condition"]}:{entry["classifier"]}'
            ] = entry['state']

    for idx, candidate in enumerate(candidates, start=1):
        partition_metadata = _candidate_metadata(
            candidate=candidate,
            candidate_family=candidate_family,
            partition_index=idx,
            main_partition=main_partition,
        )
        within_assembled_status: dict[str, str] = {}
        cross_partial_status: dict[str, str] = {}
        combined_path = Path('/results') / _cross_combined_api_path(
            execution_id=context['downstream_execution_id'],
            candidate_family=candidate_family,
            partition_index=idx,
        )
        for condition in CONDITIONS:
            assembled_path = Path('/results') / _within_assembled_api_path(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
                partition_index=idx,
                condition=condition,
            )
            if not assembled_path.exists():
                within_assembled_status[condition] = 'missing'
            else:
                try:
                    payload = _load_json(assembled_path)
                    _validate_within_assembled_payload(
                        payload=payload,
                        context=context,
                        condition=condition,
                        partition_metadata=partition_metadata,
                    )
                    within_assembled_status[condition] = 'completed'
                except (json.JSONDecodeError, OSError, ValueError, KeyError):
                    within_assembled_status[condition] = 'invalid'

        for source_condition, target_condition in DIRECTIONS:
            direction_key = f'{source_condition}_to_{target_condition}'
            path = Path('/results') / _cross_partial_api_path(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
                partition_index=idx,
                source_condition=source_condition,
                target_condition=target_condition,
            )
            if not path.exists():
                cross_partial_status[direction_key] = 'missing'
            else:
                try:
                    payload = _load_json(path)
                    _validate_cross_partial_payload(
                        payload=payload,
                        context=context,
                        direction_key=direction_key,
                        partition_metadata=partition_metadata,
                    )
                    cross_partial_status[direction_key] = 'completed'
                except (json.JSONDecodeError, OSError, ValueError, KeyError):
                    cross_partial_status[direction_key] = 'invalid'

        status['within_assembled'][_partition_key(
            idx)] = within_assembled_status
        status['cross_partials'][_partition_key(idx)] = cross_partial_status
        if not combined_path.exists():
            status['cross_combined'][_partition_key(idx)] = 'missing'
        else:
            try:
                payload = _load_json(combined_path)
                _validate_cross_combined_payload(
                    payload=payload,
                    context=context,
                    partition_metadata=partition_metadata,
                )
                status['cross_combined'][_partition_key(idx)] = 'completed'
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                status['cross_combined'][_partition_key(idx)] = 'invalid'

    summary_path = Path('/results') / \
        _family_summary_api_path(candidate_family)
    if not summary_path.exists():
        status['summary'] = 'missing'
    else:
        try:
            payload = _load_json(summary_path)
            _validate_summary_payload(
                payload=payload,
                context=context,
                candidate_family=candidate_family,
            )
            status['summary'] = 'completed'
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            status['summary'] = 'invalid'
    return json.dumps(status, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def assemble_retune_remote(candidate_family: str = 'near_optimal') -> str:
    from scipy.stats import spearmanr

    from v4_downstream import (
        DIRECTIONS,
        build_authoritative_step12_context,
        load_json,
        validate_step3_results_payload,
        write_payload_json,
    )

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=(
            'src/v4_downstream.py',
            'scripts/training/run_control_split_sensitivity_modal.py',
        ),
        repo_root=None,
    )
    step3_payload = load_json(
        context['authoritative_results_dir'] / 'cross_condition_results_v4.json')
    validate_step3_results_payload(
        payload=step3_payload,
        output_namespace='results_v4',
        expected_directions=DIRECTIONS,
        context=context,
    )
    main_partition = _read_volume_json(
        'processed_v4/control_partition_v4.json')
    candidates = _load_candidate_set(candidate_family)
    if not candidates:
        raise RuntimeError(
            f'No candidates available for candidate_family={candidate_family!r}.')

    main_within_by_condition = context['step2_payloads']
    main_cross = step3_payload
    _, main_delta_vector, main_sign_vector = _direction_summary(
        within_by_source=main_within_by_condition,
        cross_results=main_cross,
    )

    partitions_summary: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        partition_metadata = _candidate_metadata(
            candidate=candidate,
            candidate_family=candidate_family,
            partition_index=idx,
            main_partition=main_partition,
        )
        partition_root = Path(
            _run_results_root(
                execution_id=context['downstream_execution_id'],
                candidate_family=candidate_family,
            )
        ) / _partition_key(idx)
        within_by_condition: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            assembled_path = partition_root / \
                _within_results_filename(condition, idx)
            if not assembled_path.exists():
                raise FileNotFoundError(
                    'retune-assemble requires assembled within-condition JSONs first: '
                    f'{assembled_path}'
                )
            assembled = _load_json(assembled_path)
            _validate_within_assembled_payload(
                payload=assembled,
                context=context,
                condition=condition,
                partition_metadata=partition_metadata,
            )
            within_by_condition[condition] = assembled

        cross_results_by_partition: dict[str, dict[str, Any]] = {}
        for source_condition, target_condition in DIRECTIONS:
            direction_key = f'{source_condition}_to_{target_condition}'
            cross_partial_path = partition_root / _cross_partial_filename(
                idx,
                source_condition,
                target_condition,
            )
            cross_partial = _load_json(cross_partial_path)
            _validate_cross_partial_payload(
                payload=cross_partial,
                context=context,
                direction_key=direction_key,
                partition_metadata=partition_metadata,
            )
            cross_results_by_partition[direction_key] = cross_partial
        combined_cross_path = partition_root / _cross_combined_filename(idx)
        combined_cross_payload = {
            'schema_version': CROSS_COMBINED_SCHEMA_VERSION,
            'candidate_family': candidate_family,
            'partition_index': idx,
            'partition_key': _partition_key(idx),
            'partition_hash': partition_metadata['partition_hash'],
            'control_A': partition_metadata['control_A'],
            'control_B': partition_metadata['control_B'],
            'downstream_execution_id': context['downstream_execution_id'],
            'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
            'feature_matrix_hash': context['feature_matrix_sha256'],
            'protocol_manifest_hash': context['protocol_manifest_sha256'],
            'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
            'partition_metadata': partition_metadata,
            'cross_results': cross_results_by_partition,
        }
        write_payload_json(combined_cross_path, combined_cross_payload)

        direction_summary, partition_delta_vector, partition_sign_vector = _direction_summary(
            within_by_source=within_by_condition,
            cross_results=cross_results_by_partition,
        )
        rho = spearmanr(partition_delta_vector, main_delta_vector).statistic
        if rho is not None:
            rho = float(rho)
            if math.isnan(rho):
                rho = None
        same_sign_count = sum(
            int(partition_sign == main_sign)
            for partition_sign, main_sign in zip(partition_sign_vector, main_sign_vector)
        )

        partitions_summary.append({
            **partition_metadata,
            'directions': direction_summary,
            'stability_check': {
                'n_directions_same_sign': same_sign_count,
                'ordering_rank_correlation': None if rho is None else round(rho, 6),
            },
        })

    summary = {
        'schema_version': SUMMARY_SCHEMA_VERSION,
        'downstream_execution_id': context['downstream_execution_id'],
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'partition_hash': context['partition_sha256'],
        'candidate_family': candidate_family,
        'max_partitions_requested': MAX_PARTITIONS,
        'n_partitions_evaluated': len(candidates),
        'main_authoritative': {
            'within_best_subject_f1_by_source': {
                condition: round(_best_within_f1(
                    within_result, subject_level=True), 6)
                for condition, within_result in main_within_by_condition.items()
            },
            'primary_direction_mean_matched_degradation_subject': {
                f'{source_condition}_to_{target_condition}': round(main_delta_vector[idx], 6)
                for idx, (source_condition, target_condition) in enumerate(DIRECTIONS)
            },
            'direction_signs_subject': {
                f'{source_condition}_to_{target_condition}': main_sign_vector[idx]
                for idx, (source_condition, target_condition) in enumerate(DIRECTIONS)
            },
        },
        'partitions': partitions_summary,
    }

    summary_path = Path('/results') / \
        _family_summary_api_path(candidate_family)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_payload_json(summary_path, summary)
    volume.commit()
    return json.dumps(
        {
            'status': 'completed',
            'candidate_family': candidate_family,
            'downstream_execution_id': context['downstream_execution_id'],
            'summary_path': str(summary_path),
        },
        indent=2,
    )


@app.local_entrypoint()
def main(
    action: str = 'retune-submit-within',
    candidate_family: str = 'near_optimal',
) -> None:
    if action in (
        'retune-submit-within',
        'retune-submit-within-missing',
        'retune-submit-within-shards',
    ):
        print(submit_retune_within_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    if action in ('retune-status', 'retune-status-optimized'):
        print(collect_retune_status_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    if action in ('retune-dry-run', 'retune-diagnose-missing'):
        print(retune_dry_run_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    if action == 'retune-assemble-within-missing':
        print(assemble_within_missing_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    if action == 'retune-assemble-within':
        print(assemble_within_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    if action == 'retune-submit-cross':
        print(submit_retune_cross_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    if action == 'retune-assemble':
        print(assemble_retune_remote.remote(
            candidate_family=candidate_family), flush=True)
        return
    raise SystemExit(
        f'Unknown action {action!r}. Expected one of: '
        'retune-submit-within, retune-submit-within-missing, '
        'retune-submit-within-shards, retune-status, retune-status-optimized, '
        'retune-dry-run, retune-diagnose-missing, retune-assemble-within-missing, '
        'retune-assemble-within, retune-submit-cross, retune-assemble.'
    )
