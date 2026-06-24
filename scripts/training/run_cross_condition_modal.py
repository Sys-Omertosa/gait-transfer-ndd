"""
Modal runner for the publication-track v4 zero-shot transfer benchmark.

This runner preserves the authoritative one-container sequential Step 3 flow.
It always reads canonical prerequisites from:
  /results/processed_v4/
  /results/results_v4/{pd,hd,als}_results_v4.json
  /results/models_v4/

Outputs are written only under the requested output namespace.

Examples:
  modal run scripts/training/run_cross_condition_modal.py \
    --output-namespace results_v4_smoke \
    --directions "pd:hd"

Sequential partial checkpoints are stored under:
  results_v4/downstream_runs/<execution_id>/step3/
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


DEFAULT_OUTPUT_NAMESPACE = 'results_v4'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-cross-condition-v4', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)


def _parse_directions(value: str) -> tuple[tuple[str, str], ...]:
    if not value.strip():
        return (
            ('pd', 'hd'),
            ('hd', 'pd'),
            ('pd', 'als'),
            ('als', 'pd'),
            ('hd', 'als'),
            ('als', 'hd'),
        )
    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_item in value.split(','):
        item = raw_item.strip()
        if not item:
            continue
        try:
            source_condition, target_condition = item.split(':', 1)
        except ValueError as exc:
            raise ValueError(
                f'Invalid direction specifier {item!r}; expected "source:target".'
            ) from exc
        pair = (source_condition.strip(), target_condition.strip())
        if pair[0] == pair[1]:
            raise ValueError(f'Invalid self-direction {item!r}.')
        if pair in seen:
            continue
        seen.add(pair)
        parsed.append(pair)
    if not parsed:
        raise ValueError('At least one direction must be requested.')
    return tuple(parsed)


def _local_results_dir(output_namespace: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    suffix = output_namespace.removeprefix('results_')
    return repo_root / 'experiments' / 'results' / suffix


@app.function(
    cpu=16,
    memory=24576,
    timeout=86400,
    volumes={'/results': volume},
    retries=2,
)
def run_all_directions(
    output_namespace: str = DEFAULT_OUTPUT_NAMESPACE,
    directions: str = '',
) -> str:
    import time
    from pathlib import Path as _Path

    import polars as pl

    from features import get_feature_cols
    from train import annotate_cross_condition_results, run_cross_condition
    from v4_downstream import (
        DIRECTIONS,
        api_path_to_mount_path,
        build_authoritative_step12_context,
        build_step3_partial_payload,
        downstream_run_api_root,
        finalize_cross_reporting_contract,
        load_json,
        validate_payload_digest,
        write_payload_json,
    )
    volume.reload()

    requested_directions = _parse_directions(directions)
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=(
            'src/v4_downstream.py',
            'scripts/training/run_cross_condition_modal.py',
        ),
        repo_root=None,
    )

    processed_dir = _Path('/results/processed_v4')
    authoritative_results_dir = _Path('/results/results_v4')
    authoritative_models_dir = _Path('/results/models_v4')
    output_results_dir = _Path('/results') / output_namespace
    output_results_dir.mkdir(parents=True, exist_ok=True)

    features_path = processed_dir / 'gait_features_v4.csv'
    partition_path = processed_dir / 'control_partition_v4.json'
    final_output_path = output_results_dir / 'cross_condition_results_v4.json'

    run_root_api = downstream_run_api_root(
        output_namespace=output_namespace,
        execution_id=context['downstream_execution_id'],
        step_name='step3',
    )
    partial_api_path = f'{run_root_api}/cross_condition_results_v4_partial.json'
    partial_mount_path = api_path_to_mount_path(partial_api_path)

    df = pl.read_csv(str(features_path))
    partition = load_json(partition_path)
    control_a = partition['control_A']
    control_b = partition['control_B']
    feature_cols = get_feature_cols('v4')
    source_results = context['step2_payloads']

    accumulated: dict[str, dict] = {}
    if partial_mount_path.exists():
        partial_payload = load_json(partial_mount_path)
        validate_payload_digest(partial_payload)
        if (
            partial_payload.get('output_namespace') == output_namespace
            and partial_payload.get('downstream_execution_id') == context['downstream_execution_id']
            and partial_payload.get('downstream_execution_manifest_hash') == context['downstream_manifest_sha256']
            and partial_payload.get('protocol_manifest_hash') == context['protocol_manifest_sha256']
            and partial_payload.get('preprocessing_manifest_hash') == context['preprocessing_manifest_sha256']
            and partial_payload.get('feature_matrix_hash') == context['feature_matrix_sha256']
            and partial_payload.get('partition_hash') == context['partition_sha256']
            and partial_payload.get('requested_directions') == [
                f'{source_condition}:{target_condition}'
                for source_condition, target_condition in requested_directions
            ]
        ):
            accumulated = dict(partial_payload.get('completed_directions', {}))

    t_total_start = time.time()
    for source_condition, target_condition in requested_directions:
        direction_key = f'{source_condition}_to_{target_condition}'
        if direction_key in accumulated:
            print(f'Skipping completed direction {direction_key}', flush=True)
            continue

        print(f"{'=' * 60}", flush=True)
        print(
            f'Direction: {source_condition.upper()} -> {target_condition.upper()}',
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)
        t_dir_start = time.time()

        result = run_cross_condition(
            source_condition=source_condition,
            target_condition=target_condition,
            df=df,
            control_a=control_a,
            control_b=control_b,
            source_results=source_results[source_condition],
            results_dir=output_results_dir,
            models_dir=authoritative_models_dir,
            feature_cols=feature_cols,
            feature_matrix_file='v4/gait_features_v4.csv',
            feature_set_version='v4',
            normalization='none',
            allow_refit=False,
            protocol_manifest_hash=context['protocol_manifest_sha256'],
            preprocessing_manifest_hash=context['preprocessing_manifest_sha256'],
        )
        result['downstream_execution_manifest_hash'] = context['downstream_manifest_sha256']
        result['downstream_execution_id'] = context['downstream_execution_id']
        result['feature_matrix_hash'] = context['feature_matrix_sha256']
        result['partition_hash'] = context['partition_sha256']
        result['output_namespace'] = output_namespace
        result['source_results_sha256'] = context['step2_result_hashes'][source_condition]

        accumulated[direction_key] = result
        partial_payload = build_step3_partial_payload(
            output_namespace=output_namespace,
            execution_id=context['downstream_execution_id'],
            requested_directions=requested_directions,
            completed_directions=accumulated,
            context=context,
        )
        write_payload_json(partial_mount_path, partial_payload)
        volume.commit()

        elapsed = time.time() - t_dir_start
        print(f'\nDirection {direction_key} complete in {elapsed:.0f}s', flush=True)
        print(flush=True)

    combined_results = annotate_cross_condition_results(accumulated)
    combined_results = finalize_cross_reporting_contract(combined_results)
    combined_results['output_namespace'] = output_namespace
    combined_results['downstream_execution_manifest_hash'] = context['downstream_manifest_sha256']
    combined_results['downstream_execution_id'] = context['downstream_execution_id']
    combined_results['feature_matrix_hash'] = context['feature_matrix_sha256']
    combined_results['partition_hash'] = context['partition_sha256']
    combined_results['protocol_manifest_hash'] = context['protocol_manifest_sha256']
    combined_results['preprocessing_manifest_hash'] = context['preprocessing_manifest_sha256']
    combined_results['requested_directions'] = [
        f'{source_condition}:{target_condition}'
        for source_condition, target_condition in requested_directions
    ]
    combined_results = write_payload_json(final_output_path, combined_results)
    volume.commit()

    total_elapsed = time.time() - t_total_start
    print(f'All requested directions complete in {total_elapsed:.0f}s', flush=True)
    print(f'Results written to Modal volume: {final_output_path}', flush=True)

    return json.dumps(combined_results, indent=2)


@app.local_entrypoint()
def main(
    output_namespace: str = DEFAULT_OUTPUT_NAMESPACE,
    directions: str = '',
) -> None:
    from v4_downstream import (
        build_authoritative_step12_context,
        validate_step3_results_payload,
    )
    from v4_provenance import atomic_write_json

    local_results_dir = _local_results_dir(output_namespace)
    local_results_dir.mkdir(parents=True, exist_ok=True)

    print('Submitting cross-condition job to Modal...', flush=True)
    print('Single container: 16 CPU, 24576 MB RAM.', flush=True)
    print(
        'Inputs are read from gait-results:/processed_v4, '
        'gait-results:/results_v4, and gait-results:/models_v4.',
        flush=True,
    )
    print(f'Output namespace: {output_namespace}', flush=True)
    if directions.strip():
        print(f'Requested directions: {directions}', flush=True)
    print(flush=True)

    result_json = run_all_directions.remote(
        output_namespace=output_namespace,
        directions=directions,
    )
    result = json.loads(result_json)
    context = build_authoritative_step12_context(
        volume_root=Path(__file__).resolve().parents[2],
        require_downstream_manifest=True,
        required_downstream_files=(
            'src/v4_downstream.py',
            'scripts/training/run_cross_condition_modal.py',
        ),
        repo_root=Path(__file__).resolve().parents[2],
    )
    validate_step3_results_payload(
        payload=result,
        output_namespace=output_namespace,
        expected_directions=_parse_directions(directions),
        context=context,
    )
    out_path = local_results_dir / 'cross_condition_results_v4.json'
    atomic_write_json(out_path, result)

    print('\nJob complete. Download results:', flush=True)
    print(
        f'  modal volume get gait-results {output_namespace}/cross_condition_results_v4.json',
        flush=True,
    )
    print(f'Local copy written to {out_path}', flush=True)
