"""
Modal runner for publication-track v4 SHAP transfer diagnosis.

Worker decomposition is preserved:
  - 3 source-group Modal workers in parallel
  - each worker processes its 2 target directions serially
  - within-source SHAP cache is reused safely across those 2 targets

Detached-safe orchestration is provided via:
  --action submit
  --action status
  --action assemble
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


SOURCE_TARGETS: dict[str, tuple[str, str]] = {
    'pd': ('hd', 'als'),
    'hd': ('pd', 'als'),
    'als': ('pd', 'hd'),
}
GROUP_SCHEMA_VERSION = 'v4-step4-shap-source-group-v1'
FINAL_SCHEMA_VERSION = 'v4-step4-shap-final-v1'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-shap-v4', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)


def _group_api_path(*, execution_id: str, source_condition: str) -> str:
    return (
        f'results_v4/downstream_runs/{execution_id}/step4/'
        f'source_groups/{source_condition}.json'
    )


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _expected_direction_keys(source_condition: str) -> set[str]:
    return {
        f'{source_condition}_to_{target_condition}'
        for target_condition in SOURCE_TARGETS[source_condition]
    }


def _validate_group_payload(
    *,
    payload: dict[str, Any],
    source_condition: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != GROUP_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 4 source-group schema version.')
    if payload.get('source_condition') != source_condition:
        raise ValueError('Step 4 source-group source_condition mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 4 source-group protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 4 source-group preprocessing hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 4 source-group feature hash mismatch.')
    if payload.get('partition_hash') != context['partition_sha256']:
        raise ValueError('Step 4 source-group partition hash mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Step 4 source-group downstream manifest hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Step 4 source-group downstream execution id mismatch.')
    expected_targets = list(SOURCE_TARGETS[source_condition])
    if payload.get('target_conditions') != expected_targets:
        raise ValueError('Step 4 source-group target list mismatch.')
    completed = payload.get('completed_directions', {})
    if not isinstance(completed, dict):
        raise ValueError('Invalid Step 4 completed_directions payload.')
    completed_keys = set(completed)
    expected_keys = _expected_direction_keys(source_condition)
    if not completed_keys.issubset(expected_keys):
        raise ValueError('Step 4 source-group direction key mismatch.')
    return completed


def _validate_final_step4_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    from v4_downstream import validate_downstream_final_payload

    data = validate_downstream_final_payload(
        payload=payload,
        context=context,
        schema_version=FINAL_SCHEMA_VERSION,
    )
    expected_direction_keys = {
        f'{source_condition}_to_{target_condition}'
        for source_condition, targets in SOURCE_TARGETS.items()
        for target_condition in targets
    }
    if set(data) != expected_direction_keys:
        raise ValueError('Unexpected Step 4 final direction keys.')
    return data


@app.function(
    cpu=16,
    memory=20480,
    timeout=86400,
    volumes={'/results': volume},
    retries=2,
)
def run_source_group(source_condition: str) -> str:
    import polars as pl

    from explain import run_shap_for_direction
    from features import get_feature_cols
    from v4_downstream import (
        DIRECTIONS,
        api_path_to_mount_path,
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
            'scripts/training/run_shap_modal.py',
            'src/explain.py',
        ),
        repo_root=None,
    )
    step3_path = context['authoritative_results_dir'] / 'cross_condition_results_v4.json'
    step3_payload = load_json(step3_path)
    validate_step3_results_payload(
        payload=step3_payload,
        output_namespace='results_v4',
        expected_directions=DIRECTIONS,
        context=context,
    )

    processed_dir = Path('/results/processed_v4')
    results_dir = Path('/results/results_v4')
    models_dir = Path('/results/models_v4')
    shap_dir = Path('/results/shap_v4')
    features_path = processed_dir / 'gait_features_v4.csv'
    partition_path = processed_dir / 'control_partition_v4.json'

    df = pl.read_csv(str(features_path))
    partition = _load_json(partition_path)
    control_a: list[str] = partition['control_A']
    control_b: list[str] = partition['control_B']
    feature_cols = get_feature_cols('v4')
    shap_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    group_path = api_path_to_mount_path(
        _group_api_path(
            execution_id=context['downstream_execution_id'],
            source_condition=source_condition,
        )
    )
    completed_directions: dict[str, Any] = {}
    if group_path.exists():
        try:
            existing_payload = _load_json(group_path)
            completed_directions = _validate_group_payload(
                payload=existing_payload,
                source_condition=source_condition,
                context=context,
            )
        except (json.JSONDecodeError, OSError, ValueError):
            completed_directions = {}

    for target_condition in SOURCE_TARGETS[source_condition]:
        direction_key = f'{source_condition}_to_{target_condition}'
        if direction_key in completed_directions:
            print(f'Skipping completed SHAP direction {direction_key}', flush=True)
            continue

        direction_result = run_shap_for_direction(
            source_condition=source_condition,
            target_condition=target_condition,
            df=df,
            control_a=control_a,
            control_b=control_b,
            models_dir=models_dir,
            shap_dir=shap_dir,
            feature_cols=feature_cols,
            feature_set_version='v4',
            reuse_within=True,
            stability_n_resamples=200,
            protocol_manifest_hash=context['protocol_manifest_sha256'],
            preprocessing_manifest_hash=context['preprocessing_manifest_sha256'],
            downstream_execution_manifest_hash=context['downstream_manifest_sha256'],
            downstream_execution_id=context['downstream_execution_id'],
        )
        completed_directions[direction_key] = direction_result
        payload = {
            'schema_version': GROUP_SCHEMA_VERSION,
            'source_condition': source_condition,
            'target_conditions': list(SOURCE_TARGETS[source_condition]),
            'downstream_execution_id': context['downstream_execution_id'],
            'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
            'protocol_manifest_hash': context['protocol_manifest_sha256'],
            'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
            'feature_matrix_hash': context['feature_matrix_sha256'],
            'partition_hash': context['partition_sha256'],
            'completed_directions': completed_directions,
        }
        write_payload_json(group_path, payload)
        volume.commit()

    final_payload = {
        'schema_version': GROUP_SCHEMA_VERSION,
        'source_condition': source_condition,
        'target_conditions': list(SOURCE_TARGETS[source_condition]),
        'downstream_execution_id': context['downstream_execution_id'],
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'partition_hash': context['partition_sha256'],
        'completed_directions': completed_directions,
    }
    return json.dumps(final_payload, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=3600,
    volumes={'/results': volume},
    retries=1,
)
def submit_shap_groups_remote() -> str:
    from v4_downstream import build_authoritative_step12_context

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=('scripts/training/run_shap_modal.py',),
        repo_root=None,
    )
    submitted: list[str] = []
    for source_condition in SOURCE_TARGETS:
        group_path = Path('/results') / _group_api_path(
            execution_id=context['downstream_execution_id'],
            source_condition=source_condition,
        )
        should_submit = True
        if group_path.exists():
            try:
                existing_payload = _load_json(group_path)
                completed = _validate_group_payload(
                    payload=existing_payload,
                    source_condition=source_condition,
                    context=context,
                )
                should_submit = len(completed) != len(SOURCE_TARGETS[source_condition])
            except (json.JSONDecodeError, OSError, ValueError):
                should_submit = True
        if should_submit:
            run_source_group.spawn(source_condition=source_condition)
            submitted.append(source_condition)
    return json.dumps(
        {
            'status': 'submitted',
            'submitted_source_groups': submitted,
            'downstream_execution_id': context['downstream_execution_id'],
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
def collect_status_remote() -> str:
    from v4_downstream import build_authoritative_step12_context

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=('scripts/training/run_shap_modal.py',),
        repo_root=None,
    )
    statuses: dict[str, str] = {}
    for source_condition in SOURCE_TARGETS:
        group_path = Path('/results') / _group_api_path(
            execution_id=context['downstream_execution_id'],
            source_condition=source_condition,
        )
        if not group_path.exists():
            statuses[source_condition] = 'missing'
            continue
        try:
            payload = _load_json(group_path)
            completed = _validate_group_payload(
                payload=payload,
                source_condition=source_condition,
                context=context,
            )
            statuses[source_condition] = (
                'completed'
                if len(completed) == len(SOURCE_TARGETS[source_condition])
                else 'partial'
            )
        except (json.JSONDecodeError, OSError, ValueError):
            statuses[source_condition] = 'invalid'
    final_output = Path('/results/results_v4/shap_results_v4.json')
    final_output_status = 'missing'
    if final_output.exists():
        try:
            _validate_final_step4_payload(
                payload=_load_json(final_output),
                context=context,
            )
            final_output_status = 'completed'
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            final_output_status = 'invalid'
    return json.dumps(
        {
            'step': 'step4',
            'downstream_execution_id': context['downstream_execution_id'],
            'source_groups': statuses,
            'final_output': final_output_status,
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
def assemble_shap_results_remote() -> str:
    from v4_downstream import (
        build_authoritative_step12_context,
        build_downstream_final_payload,
        write_payload_json,
    )

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=(
            'src/v4_downstream.py',
            'scripts/training/run_shap_modal.py',
            'src/explain.py',
        ),
        repo_root=None,
    )
    combined: dict[str, Any] = {}
    for source_condition in SOURCE_TARGETS:
        group_path = Path('/results') / _group_api_path(
            execution_id=context['downstream_execution_id'],
            source_condition=source_condition,
        )
        if not group_path.exists():
            raise FileNotFoundError(f'Missing Step 4 source-group payload: {group_path}')
        payload = _load_json(group_path)
        completed = _validate_group_payload(
            payload=payload,
            source_condition=source_condition,
            context=context,
        )
        if set(completed) != _expected_direction_keys(source_condition):
            raise RuntimeError(
                f'Source-group payload incomplete for {source_condition}: '
                f'{sorted(completed)}'
            )
        combined.update(completed)

    final_path = Path('/results/results_v4/shap_results_v4.json')
    final_payload = build_downstream_final_payload(
        schema_version=FINAL_SCHEMA_VERSION,
        context=context,
        data=combined,
        extra_fields={
            'step': 'step4',
        },
    )
    write_payload_json(final_path, final_payload)
    volume.commit()
    return json.dumps(
        {
            'status': 'completed',
            'downstream_execution_id': context['downstream_execution_id'],
            'results_path': str(final_path),
            'direction_count': len(combined),
        },
        indent=2,
    )


@app.local_entrypoint()
def main(action: str = 'submit') -> None:
    if action == 'submit':
        print(submit_shap_groups_remote.remote(), flush=True)
        return
    if action == 'status':
        print(collect_status_remote.remote(), flush=True)
        return
    if action == 'assemble':
        print(assemble_shap_results_remote.remote(), flush=True)
        return
    raise SystemExit(
        f'Unknown action {action!r}. Expected one of: submit, status, assemble.'
    )
