"""
Downstream-only v4 provenance and artifact-validation helpers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from v4_provenance import (
    atomic_write_json,
    canonical_payload_sha256,
    collect_package_versions,
    sha256_file,
    utc_now_iso,
)

DOWNSTREAM_MANIFEST_SCHEMA_VERSION = 'v4-downstream-execution-manifest-v1'
DOWNSTREAM_METHODOLOGY_VERSION = 'v4-downstream-hardening'
DOWNSTREAM_ID_EXCLUDED_KEYS = ('downstream_execution_id', 'payload_sha256')
DOWNSTREAM_ID_SHORT_HASH_LEN = 12

CONDITIONS: tuple[str, ...] = ('pd', 'hd', 'als')
CLF_ORDER: tuple[str, ...] = ('rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm')
DIRECTIONS: tuple[tuple[str, str], ...] = (
    ('pd', 'hd'),
    ('hd', 'pd'),
    ('pd', 'als'),
    ('als', 'pd'),
    ('hd', 'als'),
    ('als', 'hd'),
)
STEP2_MODEL_ROOT_MARKERS: tuple[tuple[str, ...], ...] = (
    ('experiments', 'models', 'v4'),
    ('results', 'models_v4'),
    ('models_v4',),
)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def git_head(repo_root: str | Path) -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_root,
        text=True,
    ).strip()


def git_status_porcelain(repo_root: str | Path) -> str:
    return subprocess.check_output(
        ['git', 'status', '--porcelain=v1'],
        cwd=repo_root,
        text=True,
    )


def payload_with_sha(payload: dict[str, Any]) -> dict[str, Any]:
    materialized = dict(payload)
    materialized['payload_sha256'] = canonical_payload_sha256(materialized)
    return materialized


def write_payload_json(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    materialized = payload_with_sha(payload)
    atomic_write_json(path, materialized)
    return materialized


def validate_payload_digest(payload: dict[str, Any]) -> None:
    stored_digest = payload.get('payload_sha256')
    if not isinstance(stored_digest, str) or not stored_digest:
        raise ValueError('Missing payload_sha256.')
    expected_digest = canonical_payload_sha256(payload)
    if expected_digest != stored_digest:
        raise ValueError('payload_sha256 mismatch.')


def stable_downstream_manifest_digest(payload: dict[str, Any]) -> str:
    return canonical_payload_sha256(
        payload,
        exclude_keys=DOWNSTREAM_ID_EXCLUDED_KEYS,
    )


def derive_downstream_execution_id(
    *,
    created_at_utc: str,
    manifest_payload: dict[str, Any],
) -> str:
    short_ts = (
        created_at_utc
        .replace('-', '')
        .replace(':', '')
        .replace('+00:00', 'Z')
        .replace('T', 'T')
    )
    digest = stable_downstream_manifest_digest(manifest_payload)
    return f'v4d-{short_ts}-{digest[:DOWNSTREAM_ID_SHORT_HASH_LEN]}'


def downstream_run_api_root(
    *,
    output_namespace: str,
    execution_id: str,
    step_name: str,
) -> str:
    return f'{output_namespace}/downstream_runs/{execution_id}/{step_name}'


def api_path_to_mount_path(api_path: str) -> Path:
    return Path('/results') / api_path


def _find_path_marker(parts: tuple[str, ...], marker: tuple[str, ...]) -> int | None:
    if len(parts) < len(marker):
        return None
    last_start = len(parts) - len(marker)
    for start in range(last_start + 1):
        if parts[start:start + len(marker)] == marker:
            return start
    return None


def _normalize_relative_logical_path(path: Path) -> str:
    parts = path.parts
    if not parts:
        raise ValueError('Logical model path cannot be empty.')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError(
            f'Logical model path escapes the expected model root: {path}')
    return Path(*parts).as_posix()


def normalize_step2_model_logical_path(
    model_reference: str | Path,
    *,
    models_dir: str | Path | None = None,
) -> str:
    path = Path(model_reference)
    if not path.is_absolute():
        return _normalize_relative_logical_path(path)

    if models_dir is not None:
        try:
            return _normalize_relative_logical_path(
                path.relative_to(Path(models_dir))
            )
        except ValueError:
            pass

    parts = path.parts
    for marker in STEP2_MODEL_ROOT_MARKERS:
        start = _find_path_marker(parts, marker)
        if start is None:
            continue
        return _normalize_relative_logical_path(
            Path(*parts[start + len(marker):])
        )

    raise ValueError(
        f'Unable to normalize Step 2 model path outside the expected roots: {path}'
    )


def normalize_step2_model_hashes(
    step2_model_hashes: dict[str, dict[str, Any]],
    *,
    models_dir: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for logical_key, entry in step2_model_hashes.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f'Invalid Step 2 model inventory entry for {logical_key}.')
        model_reference = entry.get('logical_path', entry.get('path'))
        sha256 = entry.get('sha256')
        if not isinstance(model_reference, str) or not model_reference:
            raise ValueError(
                f'Missing Step 2 model logical path for {logical_key}.')
        if not isinstance(sha256, str) or not sha256:
            raise ValueError(
                f'Missing Step 2 model SHA-256 for {logical_key}.')
        normalized[logical_key] = {
            'logical_path': normalize_step2_model_logical_path(
                model_reference,
                models_dir=models_dir,
            ),
            'sha256': sha256,
        }
    return normalized


def ensure_protocol_manifest_supported(protocol_manifest: dict[str, Any]) -> None:
    if not protocol_manifest.get('approved', False):
        raise ValueError('Protocol manifest is not approved.')
    if protocol_manifest.get('schema_version') != 'v4-protocol-manifest-v2':
        raise ValueError('Unsupported protocol manifest schema version.')
    if protocol_manifest.get('methodology_version') != 'v4-hardening':
        raise ValueError('Unsupported protocol methodology version.')


def ensure_preprocessing_manifest_supported(
    preprocessing_manifest: dict[str, Any],
    *,
    protocol_manifest_sha256: str,
    feature_matrix_sha256: str,
    partition_sha256: str,
) -> None:
    if preprocessing_manifest.get('schema_version') != 'v4-preprocessing-manifest-v2':
        raise ValueError('Unsupported preprocessing manifest schema version.')
    if preprocessing_manifest.get('methodology_version') != 'v4-hardening':
        raise ValueError('Unsupported preprocessing methodology version.')
    if preprocessing_manifest.get('protocol_manifest_sha256') != protocol_manifest_sha256:
        raise ValueError('Preprocessing manifest protocol hash mismatch.')
    if preprocessing_manifest.get('feature_matrix_sha256') != feature_matrix_sha256:
        raise ValueError('Preprocessing manifest feature hash mismatch.')
    if preprocessing_manifest.get('partition_sha256') != partition_sha256:
        raise ValueError('Preprocessing manifest partition hash mismatch.')


def required_step2_model_inventory() -> list[str]:
    return [f'{condition}:{clf_name}' for condition in CONDITIONS for clf_name in CLF_ORDER]


def _step2_result_path(results_dir: Path, condition: str) -> Path:
    return results_dir / f'{condition}_results_v4.json'


def _read_step2_payloads(
    *,
    results_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payloads: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        path = _step2_result_path(results_dir, condition)
        if not path.exists():
            raise FileNotFoundError(
                f'Missing authoritative Step 2 result: {path}')
        payload = load_json(path)
        payloads[condition] = payload
        hashes[condition] = sha256_file(path)
    return payloads, hashes


def validate_step2_payloads(
    *,
    payloads: dict[str, dict[str, Any]],
    protocol_manifest_sha256: str,
    preprocessing_manifest_sha256: str,
    feature_matrix_sha256: str,
    partition_sha256: str,
) -> None:
    for condition, payload in payloads.items():
        if payload.get('condition') != condition:
            raise ValueError(
                f'Step 2 payload condition mismatch for {condition}.')
        if payload.get('protocol_manifest_hash') != protocol_manifest_sha256:
            raise ValueError(f'Step 2 protocol hash mismatch for {condition}.')
        if payload.get('preprocessing_manifest_hash') != preprocessing_manifest_sha256:
            raise ValueError(
                f'Step 2 preprocessing hash mismatch for {condition}.')
        if payload.get('feature_matrix_hash') != feature_matrix_sha256:
            raise ValueError(f'Step 2 feature hash mismatch for {condition}.')
        if payload.get('partition_hash') != partition_sha256:
            raise ValueError(
                f'Step 2 partition hash mismatch for {condition}.')


def validate_step2_full_source_models(
    *,
    payloads: dict[str, dict[str, Any]],
    models_dir: Path,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for condition, payload in payloads.items():
        for clf_name in CLF_ORDER:
            clf_payload = payload['classifiers'][clf_name]
            model_relpath = clf_payload.get('full_source_model_path')
            model_sha256 = clf_payload.get('full_source_model_sha256')
            if not isinstance(model_relpath, str) or not model_relpath:
                raise ValueError(
                    f'Missing full_source_model_path for {condition}/{clf_name}.')
            if not isinstance(model_sha256, str) or not model_sha256:
                raise ValueError(
                    f'Missing full_source_model_sha256 for {condition}/{clf_name}.')
            logical_path = normalize_step2_model_logical_path(
                model_relpath,
                models_dir=models_dir,
            )
            model_path = models_dir / logical_path
            if not model_path.exists():
                raise FileNotFoundError(
                    f'Missing authoritative Step 2 model: {model_path}')
            actual_sha256 = sha256_file(model_path)
            if actual_sha256 != model_sha256:
                raise ValueError(
                    f'Authoritative Step 2 model hash mismatch at {model_path}.')
            inventory[f'{condition}:{clf_name}'] = {
                'condition': condition,
                'classifier': clf_name,
                'logical_path': logical_path,
                'resolved_path': str(model_path),
                'sha256': actual_sha256,
            }
    expected = set(required_step2_model_inventory())
    found = set(inventory)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            f'Step 2 model inventory mismatch. missing={missing} extra={extra}'
        )
    return inventory


def build_authoritative_step12_context(
    *,
    volume_root: str | Path,
    require_downstream_manifest: bool,
    required_downstream_files: tuple[str, ...] = (),
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(volume_root)
    if (root / 'processed_v4').exists():
        processed_dir = root / 'processed_v4'
        authoritative_results_dir = root / 'results_v4'
        authoritative_models_dir = root / 'models_v4'
    else:
        processed_dir = root / 'data' / 'processed' / 'v4'
        authoritative_results_dir = root / 'experiments' / 'results' / 'v4'
        authoritative_models_dir = root / 'experiments' / 'models' / 'v4'

    features_path = processed_dir / 'gait_features_v4.csv'
    partition_path = processed_dir / 'control_partition_v4.json'
    protocol_manifest_path = processed_dir / 'v4_protocol_manifest.json'
    preprocessing_manifest_path = processed_dir / 'preprocessing_manifest_v4.json'
    artifact_index_path = authoritative_results_dir / 'v4_artifact_index.json'
    downstream_manifest_path = processed_dir / \
        'v4_downstream_execution_manifest.json'

    required_paths = [
        features_path,
        partition_path,
        protocol_manifest_path,
        preprocessing_manifest_path,
        authoritative_results_dir,
        authoritative_models_dir,
    ]
    if require_downstream_manifest:
        required_paths.append(downstream_manifest_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f'Missing authoritative prerequisites: {missing}')

    protocol_manifest = load_json(protocol_manifest_path)
    ensure_protocol_manifest_supported(protocol_manifest)
    protocol_manifest_sha256 = sha256_file(protocol_manifest_path)

    feature_matrix_sha256 = sha256_file(features_path)
    partition_sha256 = sha256_file(partition_path)

    preprocessing_manifest = load_json(preprocessing_manifest_path)
    ensure_preprocessing_manifest_supported(
        preprocessing_manifest,
        protocol_manifest_sha256=protocol_manifest_sha256,
        feature_matrix_sha256=feature_matrix_sha256,
        partition_sha256=partition_sha256,
    )
    preprocessing_manifest_sha256 = sha256_file(preprocessing_manifest_path)

    step2_payloads, step2_result_hashes = _read_step2_payloads(
        results_dir=authoritative_results_dir,
    )
    validate_step2_payloads(
        payloads=step2_payloads,
        protocol_manifest_sha256=protocol_manifest_sha256,
        preprocessing_manifest_sha256=preprocessing_manifest_sha256,
        feature_matrix_sha256=feature_matrix_sha256,
        partition_sha256=partition_sha256,
    )
    step2_model_inventory = validate_step2_full_source_models(
        payloads=step2_payloads,
        models_dir=authoritative_models_dir,
    )

    context: dict[str, Any] = {
        'volume_root': str(root),
        'processed_dir': processed_dir,
        'authoritative_results_dir': authoritative_results_dir,
        'authoritative_models_dir': authoritative_models_dir,
        'features_path': features_path,
        'partition_path': partition_path,
        'protocol_manifest_path': protocol_manifest_path,
        'preprocessing_manifest_path': preprocessing_manifest_path,
        'protocol_manifest': protocol_manifest,
        'preprocessing_manifest': preprocessing_manifest,
        'protocol_manifest_sha256': protocol_manifest_sha256,
        'preprocessing_manifest_sha256': preprocessing_manifest_sha256,
        'feature_matrix_sha256': feature_matrix_sha256,
        'partition_sha256': partition_sha256,
        'step2_payloads': step2_payloads,
        'step2_result_hashes': step2_result_hashes,
        'step2_model_inventory': step2_model_inventory,
        'artifact_index_path': artifact_index_path,
        'artifact_index_sha256': (
            sha256_file(artifact_index_path)
            if artifact_index_path.exists() else None
        ),
    }

    if require_downstream_manifest:
        downstream_manifest = load_json(downstream_manifest_path)
        validate_downstream_manifest(
            downstream_manifest=downstream_manifest,
            repo_root=repo_root,
            required_files=required_downstream_files,
            protocol_manifest_sha256=protocol_manifest_sha256,
            preprocessing_manifest_sha256=preprocessing_manifest_sha256,
            feature_matrix_sha256=feature_matrix_sha256,
            partition_sha256=partition_sha256,
            step2_result_hashes=step2_result_hashes,
            step2_model_inventory=step2_model_inventory,
            artifact_index_sha256=context['artifact_index_sha256'],
        )
        context['downstream_manifest_path'] = downstream_manifest_path
        context['downstream_manifest'] = downstream_manifest
        context['downstream_manifest_sha256'] = sha256_file(
            downstream_manifest_path)
        context['downstream_execution_id'] = downstream_manifest['downstream_execution_id']

    return context


def validate_downstream_manifest(
    *,
    downstream_manifest: dict[str, Any],
    repo_root: str | Path | None,
    required_files: tuple[str, ...],
    protocol_manifest_sha256: str,
    preprocessing_manifest_sha256: str,
    feature_matrix_sha256: str,
    partition_sha256: str,
    step2_result_hashes: dict[str, str],
    step2_model_inventory: dict[str, dict[str, Any]],
    artifact_index_sha256: str | None,
) -> None:
    if downstream_manifest.get('schema_version') != DOWNSTREAM_MANIFEST_SCHEMA_VERSION:
        raise ValueError('Unsupported downstream manifest schema version.')
    if downstream_manifest.get('methodology_version') != DOWNSTREAM_METHODOLOGY_VERSION:
        raise ValueError('Unsupported downstream methodology version.')
    if downstream_manifest.get('protocol_manifest_sha256') != protocol_manifest_sha256:
        raise ValueError('Downstream manifest protocol hash mismatch.')
    if downstream_manifest.get('preprocessing_manifest_sha256') != preprocessing_manifest_sha256:
        raise ValueError('Downstream manifest preprocessing hash mismatch.')
    if downstream_manifest.get('feature_matrix_sha256') != feature_matrix_sha256:
        raise ValueError('Downstream manifest feature hash mismatch.')
    if downstream_manifest.get('partition_sha256') != partition_sha256:
        raise ValueError('Downstream manifest partition hash mismatch.')
    if downstream_manifest.get('step2_result_hashes') != step2_result_hashes:
        raise ValueError('Downstream manifest Step 2 result hash mismatch.')
    stored_model_hashes = normalize_step2_model_hashes(
        downstream_manifest.get('step2_model_hashes', {}),
        models_dir=None,
    )
    expected_model_hashes = {
        logical_key: {
            'logical_path': entry['logical_path'],
            'sha256': entry['sha256'],
        }
        for logical_key, entry in step2_model_inventory.items()
    }
    if stored_model_hashes != expected_model_hashes:
        raise ValueError('Downstream manifest Step 2 model hash mismatch.')
    if artifact_index_sha256 is not None:
        if downstream_manifest.get('artifact_index_sha256') != artifact_index_sha256:
            raise ValueError(
                'Downstream manifest artifact-index hash mismatch.')
    validate_payload_digest(downstream_manifest)
    derived_id = derive_downstream_execution_id(
        created_at_utc=str(downstream_manifest['created_at_utc']),
        manifest_payload=downstream_manifest,
    )
    if downstream_manifest.get('downstream_execution_id') != derived_id:
        raise ValueError('Downstream execution id derivation mismatch.')

    code_hashes = downstream_manifest.get('downstream_code_hashes', {})
    missing_required = [
        path for path in required_files if path not in code_hashes]
    if missing_required:
        raise ValueError(
            f'Downstream manifest missing required code hashes: {missing_required}')
    if repo_root is not None:
        root = Path(repo_root)
        for relpath, expected_sha256 in code_hashes.items():
            path = root / relpath
            if not path.exists():
                raise FileNotFoundError(
                    f'Downstream manifest code path missing: {path}')
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f'Downstream manifest code hash mismatch for {relpath}')


def validate_step3_results_payload(
    *,
    payload: dict[str, Any],
    output_namespace: str,
    expected_directions: tuple[tuple[str, str], ...],
    context: dict[str, Any],
) -> None:
    validate_payload_digest(payload)
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 3 protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 3 preprocessing hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 3 feature hash mismatch.')
    if payload.get('partition_hash') != context['partition_sha256']:
        raise ValueError('Step 3 partition hash mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Step 3 downstream manifest hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Step 3 downstream execution id mismatch.')
    if payload.get('output_namespace') != output_namespace:
        raise ValueError('Step 3 output namespace mismatch.')
    for source_condition, target_condition in expected_directions:
        direction_key = f'{source_condition}_to_{target_condition}'
        if direction_key not in payload:
            raise ValueError(f'Missing Step 3 direction {direction_key}.')


def build_downstream_final_payload(
    *,
    schema_version: str,
    context: dict[str, Any],
    data: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'schema_version': schema_version,
        'downstream_execution_id': context['downstream_execution_id'],
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'partition_hash': context['partition_sha256'],
        'data': data,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def validate_downstream_final_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    schema_version: str,
) -> dict[str, Any]:
    validate_payload_digest(payload)
    if payload.get('schema_version') != schema_version:
        raise ValueError('Unexpected downstream final payload schema version.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Downstream final payload execution id mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Downstream final payload manifest hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Downstream final payload protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError(
            'Downstream final payload preprocessing hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Downstream final payload feature hash mismatch.')
    if payload.get('partition_hash') != context['partition_sha256']:
        raise ValueError('Downstream final payload partition hash mismatch.')
    data = payload.get('data')
    if not isinstance(data, dict):
        raise ValueError(
            'Downstream final payload data must be a JSON object.')
    return data


def finalize_cross_reporting_contract(cross_results: dict[str, Any]) -> dict[str, Any]:
    reporting = cross_results.setdefault('__reporting_contract__', {})
    primary_family = reporting.setdefault('primary_family', {})
    primary_family['adjustment_status'] = 'not_applicable_no_direction_level_test'
    primary_family['direction_level_primary_subject_p_value_status'] = (
        'not_emitted_effect_size_first_analysis'
    )
    primary_family['classifier_level_claims'] = 'supplementary_only'
    return cross_results


def build_step3_partial_payload(
    *,
    output_namespace: str,
    execution_id: str,
    requested_directions: tuple[tuple[str, str], ...],
    completed_directions: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        'schema_version': 'v4-step3-partial-v1',
        'output_namespace': output_namespace,
        'downstream_execution_id': execution_id,
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'partition_hash': context['partition_sha256'],
        'requested_directions': [
            f'{source_condition}:{target_condition}'
            for source_condition, target_condition in requested_directions
        ],
        'completed_directions': completed_directions,
    }


def build_status_payload(
    *,
    step_name: str,
    execution_id: str,
    expected: dict[str, str],
    completed: dict[str, str],
    output_namespace: str | None = None,
    candidate_family: str | None = None,
) -> dict[str, Any]:
    missing = {
        key: path
        for key, path in expected.items()
        if key not in completed
    }
    payload: dict[str, Any] = {
        'step': step_name,
        'downstream_execution_id': execution_id,
        'expected_count': len(expected),
        'completed_count': len(completed),
        'missing_count': len(missing),
        'completed': completed,
        'missing': missing,
    }
    if output_namespace is not None:
        payload['output_namespace'] = output_namespace
    if candidate_family is not None:
        payload['candidate_family'] = candidate_family
    return payload


def collect_downstream_manifest_payload(
    *,
    repo_root: str | Path,
    snapshot_dir: str | Path,
    code_paths: tuple[str, ...],
) -> dict[str, Any]:
    root = Path(repo_root)
    snapshot = Path(snapshot_dir)
    processed_dir = root / 'data' / 'processed' / 'v4'
    results_dir = root / 'experiments' / 'results' / 'v4'
    models_dir = root / 'experiments' / 'models' / 'v4'

    protocol_manifest_path = processed_dir / 'v4_protocol_manifest.json'
    preprocessing_manifest_path = processed_dir / 'preprocessing_manifest_v4.json'
    features_path = processed_dir / 'gait_features_v4.csv'
    partition_path = processed_dir / 'control_partition_v4.json'
    artifact_index_path = results_dir / 'v4_artifact_index.json'

    step2_result_hashes = {
        condition: sha256_file(results_dir / f'{condition}_results_v4.json')
        for condition in CONDITIONS
    }

    step2_results = {
        condition: load_json(results_dir / f'{condition}_results_v4.json')
        for condition in CONDITIONS
    }
    step2_model_hashes: dict[str, dict[str, str]] = {}
    for condition, payload in step2_results.items():
        for clf_name in CLF_ORDER:
            clf_payload = payload['classifiers'][clf_name]
            model_relpath = clf_payload['full_source_model_path']
            logical_path = normalize_step2_model_logical_path(
                model_relpath,
                models_dir=models_dir,
            )
            model_path = models_dir / logical_path
            step2_model_hashes[f'{condition}:{clf_name}'] = {
                'logical_path': logical_path,
                'sha256': sha256_file(model_path),
            }

    downstream_code_hashes = {
        relpath: sha256_file(root / relpath)
        for relpath in code_paths
    }

    git_status_text = git_status_porcelain(root)
    created_at_utc = utc_now_iso()
    payload = {
        'schema_version': DOWNSTREAM_MANIFEST_SCHEMA_VERSION,
        'methodology_version': DOWNSTREAM_METHODOLOGY_VERSION,
        'created_at_utc': created_at_utc,
        'git_commit': git_head(root),
        'dirty_tree': bool(git_status_text.strip()),
        'git_status_porcelain': git_status_text,
        'working_tree_patch_sha256': sha256_file(snapshot / 'working_tree.patch'),
        'untracked_archive_sha256': sha256_file(snapshot / 'untracked_files.tar.gz'),
        'snapshot_manifest_sha256': sha256_file(snapshot / 'snapshot_manifest.json'),
        'protocol_manifest_sha256': sha256_file(protocol_manifest_path),
        'source_file_hashes_sha256': sha256_file(snapshot / 'source_file_hashes.json'),
        'preprocessing_manifest_sha256': sha256_file(preprocessing_manifest_path),
        'feature_matrix_sha256': sha256_file(features_path),
        'partition_sha256': sha256_file(partition_path),
        'step2_result_hashes': step2_result_hashes,
        'step2_model_hashes': step2_model_hashes,
        'artifact_index_sha256': (
            sha256_file(artifact_index_path)
            if artifact_index_path.exists() else None
        ),
        'downstream_code_hashes': downstream_code_hashes,
        'package_versions': collect_package_versions(),
        'downstream_execution_id_derivation': {
            'digest': 'sha256(canonical_json_without_excluded_keys)',
            'excluded_keys': list(DOWNSTREAM_ID_EXCLUDED_KEYS),
            'short_hash_len': DOWNSTREAM_ID_SHORT_HASH_LEN,
        },
    }
    payload['downstream_execution_id'] = derive_downstream_execution_id(
        created_at_utc=created_at_utc,
        manifest_payload=payload,
    )
    return payload_with_sha(payload)
