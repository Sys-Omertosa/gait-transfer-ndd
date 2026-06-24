"""
Modal runner for the publication-track v4 within-condition benchmark.

Non-SVM classifier families execute as one condition x classifier shard per
container. SVM executes as one condition x outer LOSO fold shard per container,
then assembles to a classifier shard before final per-condition assembly.

Usage:
    modal run scripts/training/run_within_condition_modal.py --action status
    modal run scripts/training/run_within_condition_modal.py --action submit
    modal run scripts/training/run_within_condition_modal.py --action assemble
    modal run scripts/training/run_within_condition_modal.py --action smoke
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import modal

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-training-v4', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)

CONDITIONS = ('pd', 'hd', 'als')
CLASSIFIER_ORDER = ('rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm')
NON_SVM_CLASSIFIERS = tuple(
    clf_name for clf_name in CLASSIFIER_ORDER if clf_name != 'svm'
)

DEFAULT_RESULTS_NAMESPACE = 'results_v4'
DEFAULT_PROCESSED_NAMESPACE = 'processed_v4'
DEFAULT_MAX_IN_FLIGHT = 64
DEFAULT_FEATURE_MATRIX_FILE = 'v4/gait_features_v4.csv'
DEFAULT_FEATURE_SET_VERSION = 'v4'
DEFAULT_NORMALIZATION = 'none'

PROTOCOL_MANIFEST_SCHEMA_VERSIONS = {'v4-protocol-manifest-v2'}
PREPROCESSING_MANIFEST_SCHEMA_VERSIONS = {'v4-preprocessing-manifest-v2'}
SUPPORTED_METHODOLOGY_VERSIONS = {'v4-hardening'}
SUPPORTED_AGGREGATION_RULES = {'mean_probability'}
SUPPORTED_TIE_BREAK_RULES = {'subject_probability_loss_then_lexicographic'}
SUPPORTED_SUBJECT_PROBABILITY_THRESHOLDS = {0.5}
SUPPORTED_DFA_POLICIES = {'concatenated'}
APPROVED_CANDIDATE_STRATEGY_POLICY: dict[str, tuple[str, ...]] = {
    'rf': ('synthetic', 'balanced', 'raw'),
    'svm': ('synthetic', 'balanced', 'raw'),
    'dt': ('synthetic', 'balanced', 'raw'),
    'xgb': ('synthetic', 'balanced', 'raw'),
    'lgbm': ('synthetic', 'balanced', 'raw'),
    'knn': ('synthetic', 'raw'),
    'qda': ('synthetic', 'raw'),
}

CLASSIFIER_SHARD_SCHEMA_VERSION = 'v4-within-condition-classifier-shard-v2'
FINAL_RESULTS_SCHEMA_VERSION = 'v4-within-condition-results-v2'
SVM_OUTER_FOLD_SCHEMA_VERSION = 'v4-svm-outer-fold-shard-v2'


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _volume_api_relpath(path: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to('/results'))
        except ValueError:
            return str(path)
    return str(path)


def _read_volume_bytes(path: str) -> bytes:
    return b''.join(volume.read_file(path))


def _read_volume_json(path: str) -> dict[str, Any]:
    return json.loads(_read_volume_bytes(path).decode())


def _write_volume_json(path: str, payload: dict[str, Any]) -> None:
    with volume.batch_upload(force=True) as batch:
        batch.put_file(
            io.BytesIO(json.dumps(payload, indent=2, sort_keys=True).encode()),
            path,
        )


def _results_namespace_to_models_namespace(namespace: str) -> str:
    if namespace == DEFAULT_RESULTS_NAMESPACE:
        return 'models_v4'
    if namespace.startswith('results_'):
        return 'models_' + namespace[len('results_'):]
    return f'{namespace}_models'


def _namespace_paths(
    namespace: str,
    *,
    volume_root: Path = Path('/results'),
) -> dict[str, Path]:
    processed_dir = volume_root / DEFAULT_PROCESSED_NAMESPACE
    results_dir = volume_root / namespace
    models_dir = volume_root / _results_namespace_to_models_namespace(namespace)
    classifier_shard_dir = results_dir / 'classifier_shards'
    selection_trace_dir = results_dir / 'selection_traces'
    return {
        'processed_dir': processed_dir,
        'results_dir': results_dir,
        'models_dir': models_dir,
        'classifier_shard_dir': classifier_shard_dir,
        'svm_outer_fold_dir': classifier_shard_dir / 'svm_outer_folds',
        'selection_trace_dir': selection_trace_dir,
    }


def _classifier_shard_relpath(condition: str, clf_name: str) -> str:
    return f'classifier_shards/{condition}_{clf_name}_results_v4_shard.json'


def _classifier_shard_path(paths: dict[str, Path], condition: str, clf_name: str) -> Path:
    return paths['results_dir'] / _classifier_shard_relpath(condition, clf_name)


def _svm_outer_fold_relpath(
    condition: str,
    outer_fold_index: int,
    outer_subject_id: str,
) -> str:
    return (
        'classifier_shards/svm_outer_folds/'
        f'{condition}_svm_fold_{outer_fold_index:02d}_{outer_subject_id}.json'
    )


def _svm_outer_fold_path(
    paths: dict[str, Path],
    condition: str,
    outer_fold_index: int,
    outer_subject_id: str,
) -> Path:
    return paths['results_dir'] / _svm_outer_fold_relpath(
        condition,
        outer_fold_index,
        outer_subject_id,
    )


def _full_results_path(paths: dict[str, Path], condition: str) -> Path:
    return paths['results_dir'] / f'{condition}_results_v4.json'


def _resolve_stored_path(path_text: str, *, default_parent: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else default_parent / path


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _parse_targets(targets: str | None) -> list[tuple[str, str | None]]:
    if not targets:
        return []

    parsed: list[tuple[str, str | None]] = []
    for raw_token in targets.split(','):
        token = raw_token.strip().lower()
        if not token:
            continue
        parts = token.split(':')
        if len(parts) not in (1, 2):
            raise ValueError(
                f"Invalid target '{raw_token}'. Expected condition or condition:classifier."
            )
        condition = parts[0]
        if condition not in CONDITIONS:
            raise ValueError(
                f"Unknown condition '{condition}'. Expected one of {CONDITIONS}."
            )
        clf_name = parts[1] if len(parts) == 2 else None
        if clf_name is not None and clf_name not in CLASSIFIER_ORDER:
            raise ValueError(
                f"Unknown classifier '{clf_name}'. Expected one of {CLASSIFIER_ORDER}."
            )
        parsed.append((condition, clf_name))
    return parsed


def _parse_svm_fold_indices(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None
    indices: list[int] = []
    for token in text.split(','):
        stripped = token.strip()
        if not stripped:
            continue
        fold_idx = int(stripped)
        if fold_idx < 0:
            raise ValueError('svm fold indices must be non-negative.')
        indices.append(fold_idx)
    deduped = _dedupe_preserve_order([str(idx) for idx in indices])
    return [int(idx) for idx in deduped]


def _expand_submit_pairs(targets: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
    if not targets:
        return [(condition, clf_name) for condition in CONDITIONS for clf_name in CLASSIFIER_ORDER]

    expanded: list[tuple[str, str]] = []
    for condition, clf_name in targets:
        if clf_name is None:
            expanded.extend((condition, classifier) for classifier in CLASSIFIER_ORDER)
        else:
            expanded.append((condition, clf_name))
    encoded = _dedupe_preserve_order(
        [f'{condition}:{clf_name}' for condition, clf_name in expanded]
    )
    return [tuple(item.split(':', 1)) for item in encoded]  # type: ignore[return-value]


def _expand_assemble_requests(
    targets: list[tuple[str, str | None]],
) -> tuple[list[str], list[str]]:
    if not targets:
        all_conditions = list(CONDITIONS)
        return all_conditions, all_conditions

    svm_conditions: list[str] = []
    final_conditions: list[str] = []
    for condition, clf_name in targets:
        if clf_name is None:
            svm_conditions.append(condition)
            final_conditions.append(condition)
        elif clf_name == 'svm':
            svm_conditions.append(condition)
    return _dedupe_preserve_order(svm_conditions), _dedupe_preserve_order(final_conditions)


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError('batch size must be positive')
    return [values[idx:idx + size] for idx in range(0, len(values), size)]


def _normalize_strategy_policy(
    protocol_manifest: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    raw_policy = protocol_manifest.get('candidate_strategy_policy')
    if not isinstance(raw_policy, dict):
        raise ValueError('Protocol manifest is missing candidate_strategy_policy.')

    normalized: dict[str, tuple[str, ...]] = {}
    for clf_name in CLASSIFIER_ORDER:
        strategies = raw_policy.get(clf_name)
        if not isinstance(strategies, list) or not strategies:
            raise ValueError(
                f'Protocol manifest is missing candidate strategies for {clf_name}.'
            )
        normalized[clf_name] = tuple(str(strategy) for strategy in strategies)

    if normalized != APPROVED_CANDIDATE_STRATEGY_POLICY:
        raise ValueError(
            'Protocol manifest candidate_strategy_policy does not match the '
            'approved classifier-specific v4 policy.'
        )
    return normalized


def _validate_protocol_manifest(protocol_manifest: dict[str, Any]) -> None:
    if protocol_manifest.get('approved') is not True:
        raise ValueError('Protocol manifest must be approved.')
    if protocol_manifest.get('schema_version') not in PROTOCOL_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            'Unsupported protocol manifest schema_version: '
            f'{protocol_manifest.get("schema_version")!r}.'
        )
    if protocol_manifest.get('methodology_version') not in SUPPORTED_METHODOLOGY_VERSIONS:
        raise ValueError(
            'Unsupported protocol manifest methodology_version: '
            f'{protocol_manifest.get("methodology_version")!r}.'
        )
    if float(protocol_manifest.get('robust_mad_multiplier')) != 3.0:
        raise ValueError('Publication-track v4 requires robust_mad_multiplier=3.0.')
    if protocol_manifest.get('dfa_policy') not in SUPPORTED_DFA_POLICIES:
        raise ValueError('Publication-track v4 requires dfa_policy=concatenated.')
    if protocol_manifest.get('aggregation_rule') not in SUPPORTED_AGGREGATION_RULES:
        raise ValueError('Unsupported aggregation_rule in protocol manifest.')
    if float(protocol_manifest.get('subject_probability_threshold')) not in (
        SUPPORTED_SUBJECT_PROBABILITY_THRESHOLDS
    ):
        raise ValueError('Unsupported subject_probability_threshold in protocol manifest.')
    if protocol_manifest.get('tie_break_rule') not in SUPPORTED_TIE_BREAK_RULES:
        raise ValueError('Unsupported tie_break_rule in protocol manifest.')
    _normalize_strategy_policy(protocol_manifest)


def _validate_preprocessing_manifest(
    preprocessing_manifest: dict[str, Any],
    *,
    protocol_manifest: dict[str, Any],
    protocol_manifest_hash: str,
    feature_matrix_hash: str,
    partition_hash: str,
) -> None:
    if (
        preprocessing_manifest.get('schema_version')
        not in PREPROCESSING_MANIFEST_SCHEMA_VERSIONS
    ):
        raise ValueError(
            'Unsupported preprocessing manifest schema_version: '
            f'{preprocessing_manifest.get("schema_version")!r}.'
        )
    if (
        preprocessing_manifest.get('methodology_version')
        not in SUPPORTED_METHODOLOGY_VERSIONS
    ):
        raise ValueError(
            'Unsupported preprocessing manifest methodology_version: '
            f'{preprocessing_manifest.get("methodology_version")!r}.'
        )
    if preprocessing_manifest.get('protocol_manifest_sha256') != protocol_manifest_hash:
        raise ValueError('Preprocessing manifest protocol_manifest_sha256 mismatch.')
    if preprocessing_manifest.get('feature_matrix_sha256') != feature_matrix_hash:
        raise ValueError('Preprocessing manifest feature_matrix_sha256 mismatch.')
    if preprocessing_manifest.get('partition_sha256') != partition_hash:
        raise ValueError('Preprocessing manifest partition_sha256 mismatch.')
    if preprocessing_manifest.get('approved_protocol_manifest_required') is not True:
        raise ValueError(
            'Preprocessing manifest must require an approved protocol manifest.'
        )
    for key in (
        'robust_mad_multiplier',
        'dfa_policy',
        'aggregation_rule',
        'subject_probability_threshold',
        'tie_break_rule',
    ):
        if preprocessing_manifest.get(key) != protocol_manifest.get(key):
            raise ValueError(
                f'Preprocessing manifest {key} does not match protocol manifest.'
            )
    required_artifact_keys = (
        ('per_stride_only_matrix_path', 'per_stride_only_matrix_sha256'),
        ('subject_level_matrix_path', 'subject_level_matrix_sha256'),
        ('timing_sensitivity_matrix_path', 'timing_sensitivity_matrix_sha256'),
        ('dfa_sensitivity_matrix_path', 'dfa_sensitivity_matrix_sha256'),
        ('no_dfa_sensitivity_matrix_path', 'no_dfa_sensitivity_matrix_sha256'),
    )
    for path_key, hash_key in required_artifact_keys:
        if path_key not in preprocessing_manifest or hash_key not in preprocessing_manifest:
            raise ValueError(
                f'Preprocessing manifest is missing {path_key} / {hash_key}.'
            )
        artifact_path = Path(str(preprocessing_manifest[path_key]))
        if not artifact_path.exists():
            raise FileNotFoundError(
                f'Preprocessing manifest artifact is missing: {artifact_path}'
            )
        from v4_provenance import sha256_file
        if sha256_file(artifact_path) != preprocessing_manifest[hash_key]:
            raise ValueError(
                f'Preprocessing manifest hash mismatch for {artifact_path}.'
            )


def _load_validated_authoritative_context(
    condition: str,
    *,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    volume_root: Path = Path('/results'),
) -> dict[str, Any]:
    import polars as pl
    from sklearn.model_selection import LeaveOneGroupOut

    from features import get_feature_cols
    from v4_provenance import sha256_file

    if condition not in CONDITIONS:
        raise ValueError(
            f'Unknown condition {condition!r}. Expected one of {CONDITIONS}.'
        )

    paths = _namespace_paths(namespace, volume_root=volume_root)
    features_path = paths['processed_dir'] / 'gait_features_v4.csv'
    partition_path = paths['processed_dir'] / 'control_partition_v4.json'
    protocol_manifest_path = paths['processed_dir'] / 'v4_protocol_manifest.json'
    preprocessing_manifest_path = paths['processed_dir'] / 'preprocessing_manifest_v4.json'
    missing = [
        str(path)
        for path in (
            features_path,
            partition_path,
            protocol_manifest_path,
            preprocessing_manifest_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            'Missing authoritative Step 1 artifacts on the Modal volume:\n'
            + '\n'.join(missing)
        )

    protocol_manifest = _load_json(protocol_manifest_path)
    _validate_protocol_manifest(protocol_manifest)
    policy = _normalize_strategy_policy(protocol_manifest)

    protocol_manifest_hash = sha256_file(protocol_manifest_path)
    feature_matrix_hash = sha256_file(features_path)
    partition_hash = sha256_file(partition_path)

    preprocessing_manifest = _load_json(preprocessing_manifest_path)
    _validate_preprocessing_manifest(
        preprocessing_manifest,
        protocol_manifest=protocol_manifest,
        protocol_manifest_hash=protocol_manifest_hash,
        feature_matrix_hash=feature_matrix_hash,
        partition_hash=partition_hash,
    )

    df = pl.read_csv(str(features_path))
    partition = _load_json(partition_path)
    feature_cols = get_feature_cols('v4')
    pool = df.filter(
        (pl.col('condition') == condition)
        | pl.col('subject_id').is_in(partition['control_A'])
    )
    X = pool.select(feature_cols).to_numpy().astype('float64')
    y = pool['label'].to_numpy().astype(int)
    groups = pool['subject_id'].to_numpy()

    outer_subject_ids: list[str] = []
    outer_loso = LeaveOneGroupOut()
    for _, test_idx in outer_loso.split(X, y, groups):
        outer_subject_ids.append(str(groups[test_idx][0]))

    return {
        'condition': condition,
        'namespace': namespace,
        'paths': paths,
        'df': df,
        'partition': partition,
        'protocol_manifest': protocol_manifest,
        'preprocessing_manifest': preprocessing_manifest,
        'feature_cols': feature_cols,
        'control_A': list(partition['control_A']),
        'candidate_strategy_policy': policy,
        'aggregation_rule': str(protocol_manifest['aggregation_rule']),
        'subject_probability_threshold': float(
            protocol_manifest['subject_probability_threshold']
        ),
        'tie_break_rule': str(protocol_manifest['tie_break_rule']),
        'feature_matrix_hash': feature_matrix_hash,
        'partition_hash': partition_hash,
        'protocol_manifest_hash': protocol_manifest_hash,
        'preprocessing_manifest_hash': sha256_file(preprocessing_manifest_path),
        'pool_subjects': int(pool.n_unique('subject_id')),
        'pool_strides': int(pool.height),
        'X': X,
        'y': y,
        'groups': groups,
        'outer_subject_ids': outer_subject_ids,
    }


def _resolve_outer_fold_indices(
    *,
    X: Any,
    y: Any,
    groups: Any,
    outer_fold_index: int,
    outer_subject_id: str,
) -> tuple[Any, Any]:
    from sklearn.model_selection import LeaveOneGroupOut

    outer_loso = LeaveOneGroupOut()
    for fold_idx, (train_idx, test_idx) in enumerate(outer_loso.split(X, y, groups)):
        if fold_idx != outer_fold_index:
            continue
        actual_subject_id = str(groups[test_idx][0])
        if actual_subject_id != outer_subject_id:
            raise ValueError(
                'Outer-fold subject mismatch: '
                f'expected {outer_subject_id}, got {actual_subject_id}.'
            )
        return train_idx, test_idx
    raise IndexError(
        f'Unable to resolve outer fold {outer_fold_index} for subject '
        f'{outer_subject_id!r}.'
    )


def _selected_svm_fold_specs(
    context: dict[str, Any],
    *,
    svm_fold_indices: list[int] | None = None,
) -> list[tuple[int, str]]:
    if svm_fold_indices is None:
        return list(enumerate(context['outer_subject_ids']))

    specs: list[tuple[int, str]] = []
    n_folds = len(context['outer_subject_ids'])
    for fold_idx in svm_fold_indices:
        if fold_idx >= n_folds:
            raise IndexError(
                f'SVM fold index {fold_idx} is out of range for condition '
                f"{context['condition']!r} (n_folds={n_folds})."
            )
        specs.append((fold_idx, context['outer_subject_ids'][fold_idx]))
    return specs


def _validate_payload_digest(payload: dict[str, Any]) -> None:
    from v4_provenance import canonical_payload_sha256

    expected = payload.get('payload_sha256')
    if not isinstance(expected, str) or not expected:
        raise ValueError('Artifact payload is missing payload_sha256.')
    actual = canonical_payload_sha256(payload)
    if actual != expected:
        raise ValueError('Artifact payload_sha256 mismatch.')


def _write_json_with_digest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from v4_provenance import atomic_write_json, canonical_payload_sha256

    materialized = dict(payload)
    materialized['payload_sha256'] = canonical_payload_sha256(materialized)
    atomic_write_json(path, materialized)
    return materialized


def _validate_model_and_trace_artifacts(
    *,
    classifier_payload: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    from v4_provenance import sha256_file

    full_source_model_relpath = classifier_payload.get('full_source_model_path')
    full_source_model_sha256 = classifier_payload.get('full_source_model_sha256')
    if not isinstance(full_source_model_relpath, str) or not isinstance(
        full_source_model_sha256,
        str,
    ):
        raise ValueError('Classifier payload is missing full-source model provenance.')
    full_source_model_path = _resolve_stored_path(
        full_source_model_relpath,
        default_parent=paths['models_dir'],
    )
    if not full_source_model_path.exists():
        raise FileNotFoundError(
            f'Full-source model artifact is missing: {full_source_model_path}'
        )
    if sha256_file(full_source_model_path) != full_source_model_sha256:
        raise ValueError(
            f'Full-source model SHA-256 mismatch for {full_source_model_path}.'
        )

    full_trace_path_text = classifier_payload.get('full_source_selection_trace_path')
    full_trace_sha = classifier_payload.get('full_source_selection_trace_sha256')
    if not isinstance(full_trace_path_text, str) or not isinstance(full_trace_sha, str):
        raise ValueError('Classifier payload is missing full-source selection-trace provenance.')
    full_trace_path = _resolve_stored_path(
        full_trace_path_text,
        default_parent=paths['results_dir'],
    )
    if not full_trace_path.exists():
        raise FileNotFoundError(
            f'Full-source selection trace is missing: {full_trace_path}'
        )
    if sha256_file(full_trace_path) != full_trace_sha:
        raise ValueError(
            f'Full-source selection trace SHA-256 mismatch for {full_trace_path}.'
        )

    for fold_detail in classifier_payload.get('outer_fold_selection_trace', []):
        model_relpath = fold_detail.get('fold_model_relpath')
        model_sha = fold_detail.get('fold_model_sha256')
        if not isinstance(model_relpath, str) or not isinstance(model_sha, str):
            raise ValueError('Outer-fold trace is missing fold-model provenance.')
        fold_model_path = _resolve_stored_path(
            model_relpath,
            default_parent=paths['models_dir'],
        )
        if not fold_model_path.exists():
            raise FileNotFoundError(f'Fold model artifact is missing: {fold_model_path}')
        if sha256_file(fold_model_path) != model_sha:
            raise ValueError(
                f'Fold model SHA-256 mismatch for {fold_model_path}.'
            )

        trace_relpath = fold_detail.get('candidate_trace_relpath')
        trace_sha = fold_detail.get('candidate_trace_sha256')
        if not isinstance(trace_relpath, str) or not isinstance(trace_sha, str):
            raise ValueError('Outer-fold trace is missing candidate-trace provenance.')
        trace_path = _resolve_stored_path(
            trace_relpath,
            default_parent=paths['results_dir'],
        )
        if not trace_path.exists():
            raise FileNotFoundError(f'Candidate trace sidecar is missing: {trace_path}')
        if sha256_file(trace_path) != trace_sha:
            raise ValueError(
                f'Candidate trace SHA-256 mismatch for {trace_path}.'
            )


def _validate_classifier_result_payload(
    *,
    classifier_payload: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    required_classifier_keys = (
        'f1_macro',
        'subject_primary_f1_macro',
        'selected_imbalance_strategy',
        'candidate_imbalance_strategies',
        'outer_fold_selection_trace',
        'full_source_selection_trace',
        'full_source_model_path',
        'full_source_model_sha256',
    )
    missing = [key for key in required_classifier_keys if key not in classifier_payload]
    if missing:
        raise ValueError(
            f'Classifier payload is missing required keys: {missing}'
        )
    _validate_model_and_trace_artifacts(
        classifier_payload=classifier_payload,
        paths=paths,
    )


def _validate_single_classifier_shard_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    clf_name: str,
) -> None:
    _validate_payload_digest(payload)
    if payload.get('schema_version') != CLASSIFIER_SHARD_SCHEMA_VERSION:
        raise ValueError('Classifier shard schema_version mismatch.')
    if payload.get('condition') != context['condition']:
        raise ValueError('Classifier shard condition mismatch.')
    if payload.get('subject_aggregation_rule') != context['aggregation_rule']:
        raise ValueError('Classifier shard aggregation_rule mismatch.')
    if float(payload.get('subject_probability_threshold')) != context['subject_probability_threshold']:
        raise ValueError('Classifier shard subject_probability_threshold mismatch.')
    if payload.get('tie_break_rule') != context['tie_break_rule']:
        raise ValueError('Classifier shard tie_break_rule mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_hash']:
        raise ValueError('Classifier shard feature_matrix_hash mismatch.')
    if payload.get('partition_hash') != context['partition_hash']:
        raise ValueError('Classifier shard partition_hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_hash']:
        raise ValueError('Classifier shard protocol_manifest_hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_hash']:
        raise ValueError('Classifier shard preprocessing_manifest_hash mismatch.')
    if payload.get('candidate_strategy_policy') != {
        clf_name: list(context['candidate_strategy_policy'][clf_name])
    }:
        raise ValueError('Classifier shard candidate_strategy_policy mismatch.')

    classifiers = payload.get('classifiers')
    if not isinstance(classifiers, dict) or set(classifiers) != {clf_name}:
        raise ValueError(
            f'Classifier shard for {context["condition"]}:{clf_name} must contain '
            'exactly one classifier entry.'
        )
    _validate_classifier_result_payload(
        classifier_payload=classifiers[clf_name],
        paths=context['paths'],
    )


def _validate_final_condition_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
) -> None:
    _validate_payload_digest(payload)
    if payload.get('schema_version') != FINAL_RESULTS_SCHEMA_VERSION:
        raise ValueError('Final within-condition payload schema_version mismatch.')
    if payload.get('condition') != context['condition']:
        raise ValueError('Final within-condition payload condition mismatch.')
    if payload.get('subject_aggregation_rule') != context['aggregation_rule']:
        raise ValueError('Final within-condition payload aggregation_rule mismatch.')
    if float(payload.get('subject_probability_threshold')) != context['subject_probability_threshold']:
        raise ValueError('Final within-condition payload subject_probability_threshold mismatch.')
    if payload.get('tie_break_rule') != context['tie_break_rule']:
        raise ValueError('Final within-condition payload tie_break_rule mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_hash']:
        raise ValueError('Final within-condition payload feature_matrix_hash mismatch.')
    if payload.get('partition_hash') != context['partition_hash']:
        raise ValueError('Final within-condition payload partition_hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_hash']:
        raise ValueError('Final within-condition payload protocol_manifest_hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_hash']:
        raise ValueError('Final within-condition payload preprocessing_manifest_hash mismatch.')
    expected_policy = {
        clf_name: list(context['candidate_strategy_policy'][clf_name])
        for clf_name in CLASSIFIER_ORDER
    }
    if payload.get('candidate_strategy_policy') != expected_policy:
        raise ValueError('Final within-condition payload candidate_strategy_policy mismatch.')
    classifiers = payload.get('classifiers')
    if not isinstance(classifiers, dict) or set(classifiers) != set(CLASSIFIER_ORDER):
        raise ValueError('Final within-condition payload classifier set mismatch.')
    for clf_name in CLASSIFIER_ORDER:
        _validate_classifier_result_payload(
            classifier_payload=classifiers[clf_name],
            paths=context['paths'],
        )


def _validate_svm_outer_fold_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    outer_fold_index: int,
    outer_subject_id: str,
) -> None:
    from v4_provenance import sha256_file

    _validate_payload_digest(payload)
    if payload.get('schema_version') != SVM_OUTER_FOLD_SCHEMA_VERSION:
        raise ValueError('SVM outer-fold shard schema_version mismatch.')
    if payload.get('condition') != context['condition']:
        raise ValueError('SVM outer-fold shard condition mismatch.')
    if payload.get('classifier') != 'svm':
        raise ValueError('SVM outer-fold shard classifier mismatch.')
    if int(payload.get('outer_fold_index')) != outer_fold_index:
        raise ValueError('SVM outer-fold shard outer_fold_index mismatch.')
    if payload.get('held_out_subject_id') != outer_subject_id:
        raise ValueError('SVM outer-fold shard held_out_subject_id mismatch.')
    if payload.get('subject_aggregation_rule') != context['aggregation_rule']:
        raise ValueError('SVM outer-fold shard aggregation_rule mismatch.')
    if float(payload.get('subject_probability_threshold')) != context['subject_probability_threshold']:
        raise ValueError('SVM outer-fold shard subject_probability_threshold mismatch.')
    if payload.get('tie_break_rule') != context['tie_break_rule']:
        raise ValueError('SVM outer-fold shard tie_break_rule mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_hash']:
        raise ValueError('SVM outer-fold shard feature_matrix_hash mismatch.')
    if payload.get('partition_hash') != context['partition_hash']:
        raise ValueError('SVM outer-fold shard partition_hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_hash']:
        raise ValueError('SVM outer-fold shard protocol_manifest_hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_hash']:
        raise ValueError('SVM outer-fold shard preprocessing_manifest_hash mismatch.')
    if payload.get('candidate_strategy_policy') != list(
        context['candidate_strategy_policy']['svm']
    ):
        raise ValueError('SVM outer-fold shard candidate_strategy_policy mismatch.')

    required_keys = (
        'selected_imbalance_strategy',
        'selected_params',
        'candidate_rankings_full',
        'public_outer_fold_detail',
        'y_true',
        'y_pred',
        'y_prob',
        'subject_ids',
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            f'SVM outer-fold shard is missing required keys: {missing}'
        )

    y_true = payload['y_true']
    y_pred = payload['y_pred']
    y_prob = payload['y_prob']
    subject_ids = payload['subject_ids']
    n_values = {len(y_true), len(y_pred), len(y_prob), len(subject_ids)}
    if len(n_values) != 1:
        raise ValueError('SVM outer-fold shard arrays do not share the same length.')
    if any(subject_id != outer_subject_id for subject_id in subject_ids):
        raise ValueError('SVM outer-fold shard subject_ids do not match held-out subject.')

    fold_detail = payload['public_outer_fold_detail']
    if fold_detail.get('held_out_subject_id') != outer_subject_id:
        raise ValueError(
            'SVM outer-fold shard public_outer_fold_detail held_out_subject_id mismatch.'
        )
    model_relpath = fold_detail.get('fold_model_relpath')
    model_sha = fold_detail.get('fold_model_sha256')
    if not isinstance(model_relpath, str) or not isinstance(model_sha, str):
        raise ValueError('SVM outer-fold shard is missing fold-model provenance.')
    fold_model_path = _resolve_stored_path(
        model_relpath,
        default_parent=context['paths']['models_dir'],
    )
    if not fold_model_path.exists():
        raise FileNotFoundError(f'SVM fold model is missing: {fold_model_path}')
    if sha256_file(fold_model_path) != model_sha:
        raise ValueError(f'SVM fold model SHA-256 mismatch for {fold_model_path}.')


def _load_reusable_payload(
    path: Path,
    *,
    validator: Any,
    force_recompute_invalid: bool,
    label: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    try:
        validator(payload)
    except Exception as exc:
        if force_recompute_invalid:
            return None
        raise type(exc)(
            f'Invalid existing {label} at {path}: {exc}'
        ) from exc
    return payload


def _finalize_classifier_shard_output(
    output: dict[str, Any],
    *,
    context: dict[str, Any],
    clf_name: str,
    path: Path,
) -> dict[str, Any]:
    payload = dict(output)
    payload['schema_version'] = CLASSIFIER_SHARD_SCHEMA_VERSION
    payload['subject_probability_threshold'] = context['subject_probability_threshold']
    written = _write_json_with_digest(path, payload)
    _validate_single_classifier_shard_payload(
        written,
        context=context,
        clf_name=clf_name,
    )
    return written


def _build_svm_outer_fold_payload(
    *,
    context: dict[str, Any],
    outer_fold_index: int,
    outer_subject_id: str,
) -> dict[str, Any]:
    import numpy as np

    from train import (
        _configure_classifier_for_resampling,
        _get_fit_kwargs,
        _json_safe,
        _save_pipeline_artifact,
        _select_best_grouped_candidate,
        build_pipeline,
        get_classifier_configs,
    )

    configs = get_classifier_configs()
    svm_config = configs['svm']
    train_idx, test_idx = _resolve_outer_fold_indices(
        X=context['X'],
        y=context['y'],
        groups=context['groups'],
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
    )

    X_train = context['X'][train_idx]
    X_test = context['X'][test_idx]
    y_train = context['y'][train_idx]
    y_test = context['y'][test_idx]
    groups_train = context['groups'][train_idx]
    groups_test = context['groups'][test_idx]

    best_candidate, candidate_rankings = _select_best_grouped_candidate(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        clf_template=svm_config['clf'],
        param_grid=svm_config['param_grid'],
        classifier_name='svm',
        candidate_imbalance_strategies=context['candidate_strategy_policy']['svm'],
        subject_aggregation_rule=context['aggregation_rule'],
        tie_break_rule=context['tie_break_rule'],
    )

    clf_variant = _configure_classifier_for_resampling(
        'svm',
        svm_config['clf'],
        best_candidate['imbalance_strategy'],
    )
    pipeline = build_pipeline(
        'svm',
        clf_variant,
        imbalance_strategy=best_candidate['imbalance_strategy'],
    )
    pipeline.set_params(**best_candidate['params'])
    fit_kwargs = _get_fit_kwargs(
        'svm',
        y_train,
        best_candidate['imbalance_strategy'],
    )
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    fold_model_path = (
        context['paths']['models_dir']
        / 'within_folds'
        / f'{context["condition"]}_svm_fold_{outer_fold_index:02d}_{outer_subject_id}.joblib'
    )
    fold_model_sha256 = _save_pipeline_artifact(pipeline, fold_model_path)

    public_candidate_rankings = [
        {
            'rank': rank_idx + 1,
            'candidate_index': int(candidate['candidate_index']),
            'imbalance_strategy': candidate['imbalance_strategy'],
            'params': dict(candidate['params']),
            'inner_subject_f1': float(candidate['inner_subject_f1']),
            'inner_stride_f1': float(candidate['inner_stride_f1']),
            'inner_subject_log_loss': (
                float(candidate['inner_subject_log_loss'])
                if candidate.get('inner_subject_log_loss') is not None else None
            ),
        }
        for rank_idx, candidate in enumerate(candidate_rankings)
    ]

    return {
        'schema_version': SVM_OUTER_FOLD_SCHEMA_VERSION,
        'kind': 'svm_outer_fold_shard',
        'condition': context['condition'],
        'classifier': 'svm',
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'candidate_strategy_policy': list(context['candidate_strategy_policy']['svm']),
        'feature_cols': list(context['feature_cols']),
        'feature_matrix_file': DEFAULT_FEATURE_MATRIX_FILE,
        'feature_set_version': DEFAULT_FEATURE_SET_VERSION,
        'normalization': DEFAULT_NORMALIZATION,
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
        'selected_params': dict(best_candidate['params']),
        'selected_inner_subject_f1': float(best_candidate['inner_subject_f1']),
        'selected_inner_stride_f1': float(best_candidate['inner_stride_f1']),
        'selected_inner_subject_log_loss': (
            float(best_candidate['inner_subject_log_loss'])
            if best_candidate.get('inner_subject_log_loss') is not None else None
        ),
        'candidate_rankings_full': _json_safe(candidate_rankings),
        'public_outer_fold_detail': {
            'held_out_subject_id': str(groups_test[0]),
            'held_out_true_label': int(y_test[0]),
            'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
            'selected_params': dict(best_candidate['params']),
            'selected_inner_subject_f1': float(best_candidate['inner_subject_f1']),
            'selected_inner_stride_f1': float(best_candidate['inner_stride_f1']),
            'candidate_rankings': public_candidate_rankings,
            'fold_model_relpath': str(fold_model_path.relative_to(context['paths']['models_dir'])),
            'fold_model_sha256': fold_model_sha256,
        },
        'y_true': np.asarray(y_test, dtype=int).tolist(),
        'y_pred': np.asarray(y_pred, dtype=int).tolist(),
        'y_prob': np.asarray(y_prob, dtype=float).tolist(),
        'subject_ids': np.asarray(groups_test).tolist(),
    }


def _assemble_svm_classifier_shard_payload(
    *,
    context: dict[str, Any],
    fold_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import f1_score

    from train import (
        _build_within_condition_output,
        _fit_grouped_full_source_model,
        _json_safe,
        _save_pipeline_artifact,
        _save_selection_trace_sidecar,
        _strategy_to_legacy_label,
        _subject_level_metrics,
        _subject_primary_bootstrap_ci,
        _subject_resampled_stride_bootstrap_ci,
        get_classifier_configs,
        get_modal_params,
        get_modal_strategy,
    )

    y_true_all = np.concatenate([
        np.asarray(payload['y_true'], dtype=int)
        for payload in fold_payloads
    ])
    y_pred_all = np.concatenate([
        np.asarray(payload['y_pred'], dtype=int)
        for payload in fold_payloads
    ])
    y_prob_all = np.concatenate([
        np.asarray(payload['y_prob'], dtype=float)
        for payload in fold_payloads
    ])
    subject_ids_all = np.concatenate([
        np.asarray(payload['subject_ids'])
        for payload in fold_payloads
    ])
    fold_params = [dict(payload['selected_params']) for payload in fold_payloads]
    fold_best_scores = [
        float(payload['selected_inner_subject_f1']) for payload in fold_payloads
    ]
    fold_best_strategies = [
        str(payload['selected_imbalance_strategy']) for payload in fold_payloads
    ]
    outer_fold_details = [
        dict(payload['public_outer_fold_detail']) for payload in fold_payloads
    ]

    ci_lower, ci_upper, n_rejected, rejection_rate = _subject_resampled_stride_bootstrap_ci(
        y_true=y_true_all,
        y_pred=y_pred_all,
        subject_ids=subject_ids_all,
        rng=np.random.default_rng(42),
        n_resamples=10_000,
    )
    subject_metrics = _subject_level_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all,
        subject_ids=subject_ids_all,
        subject_aggregation_rule=context['aggregation_rule'],
    )
    subj_true = np.asarray(subject_metrics['y_true'], dtype=int)
    subj_pred = np.asarray(subject_metrics['y_pred'], dtype=int)
    subject_ci_lower, subject_ci_upper, subject_n_rejected, subject_rejection_rate = (
        _subject_primary_bootstrap_ci(
            y_true_subject=subj_true,
            y_pred_subject=subj_pred,
            rng=np.random.default_rng(4242),
            n_resamples=10_000,
        )
    )

    full_source = _fit_grouped_full_source_model(
        clf_name='svm',
        clf=get_classifier_configs()['svm']['clf'],
        param_grid=get_classifier_configs()['svm']['param_grid'],
        X=context['X'],
        y=context['y'],
        groups=context['groups'],
        candidate_imbalance_strategies=context['candidate_strategy_policy']['svm'],
        subject_aggregation_rule=context['aggregation_rule'],
        tie_break_rule=context['tie_break_rule'],
    )

    context['paths']['selection_trace_dir'].mkdir(parents=True, exist_ok=True)
    for outer_fold_index, (fold_payload, fold_detail) in enumerate(zip(
        fold_payloads,
        outer_fold_details,
    )):
        sidecar_path = context['paths']['selection_trace_dir'] / (
            f'{context["condition"]}_svm_fold_{outer_fold_index:02d}_candidate_trace.npz'
        )
        sidecar_metadata = {
            'kind': 'outer_fold_grouped_selection_trace',
            'condition': context['condition'],
            'classifier': 'svm',
            'outer_fold_index': outer_fold_index,
            'held_out_subject_id': fold_detail['held_out_subject_id'],
            'aggregation_rule': context['aggregation_rule'],
            'tie_break_rule': context['tie_break_rule'],
            'candidate_imbalance_strategies': list(
                context['candidate_strategy_policy']['svm']
            ),
        }
        sidecar_sha = _save_selection_trace_sidecar(
            sidecar_path=sidecar_path,
            candidates=fold_payload['candidate_rankings_full'],
            metadata=sidecar_metadata,
        )
        fold_detail['candidate_trace_relpath'] = str(sidecar_path)
        fold_detail['candidate_trace_sha256'] = sidecar_sha

    full_trace_path = context['paths']['selection_trace_dir'] / (
        f'{context["condition"]}_svm_full_source_candidate_trace.npz'
    )
    full_trace_metadata = {
        'kind': 'full_source_grouped_selection_trace',
        'condition': context['condition'],
        'classifier': 'svm',
        'aggregation_rule': context['aggregation_rule'],
        'tie_break_rule': context['tie_break_rule'],
        'candidate_imbalance_strategies': list(
            context['candidate_strategy_policy']['svm']
        ),
    }
    full_trace_sha = _save_selection_trace_sidecar(
        sidecar_path=full_trace_path,
        candidates=full_source['selection_trace'],
        metadata=full_trace_metadata,
    )

    full_source_model_path = context['paths']['models_dir'] / f'{context["condition"]}_svm.joblib'
    full_source_model_sha256 = _save_pipeline_artifact(
        full_source['pipeline'],
        full_source_model_path,
    )

    outer_modal_params = get_modal_params(fold_params, fold_best_scores)
    outer_modal_strategy = get_modal_strategy(fold_best_strategies, fold_best_scores)
    modal_key = tuple(sorted(outer_modal_params.items()))
    outer_modal_frequency = sum(
        1 for params in fold_params if tuple(sorted(params.items())) == modal_key
    )
    outer_modal_strategy_frequency = sum(
        1 for strategy in fold_best_strategies if strategy == outer_modal_strategy
    )

    classifier_result = {
        'f1_macro': round(float(f1_score(
            y_true_all,
            y_pred_all,
            average='macro',
        )), 6),
        'f1_macro_ci_lower': round(ci_lower, 6),
        'f1_macro_ci_upper': round(ci_upper, 6),
        'subject_resampled_stride_f1_ci_lower': round(ci_lower, 6),
        'subject_resampled_stride_f1_ci_upper': round(ci_upper, 6),
        'subject_primary_f1_ci_lower': round(subject_ci_lower, 6),
        'subject_primary_f1_ci_upper': round(subject_ci_upper, 6),
        'subject_primary_bootstrap_rejections': subject_n_rejected,
        'subject_primary_bootstrap_rejection_rate': round(subject_rejection_rate, 6),
        'subject_resampled_stride_bootstrap_rejections': n_rejected,
        'subject_resampled_stride_bootstrap_rejection_rate': round(rejection_rate, 6),
        'subject_primary_f1_macro': round(float(subject_metrics['f1_macro']), 6),
        'subject_aggregation_rule': context['aggregation_rule'],
        'tie_break_rule': context['tie_break_rule'],
        'modal_params': outer_modal_params,
        'modal_frequency': outer_modal_frequency,
        'modal_strategy': outer_modal_strategy,
        'modal_strategy_frequency': outer_modal_strategy_frequency,
        'subject_metrics': subject_metrics,
        'y_true': y_true_all.tolist(),
        'y_pred': y_pred_all.tolist(),
        'y_prob': np.asarray(y_prob_all, dtype=float).tolist(),
        'selected_resampling': _strategy_to_legacy_label(
            full_source['selected_imbalance_strategy']
        ),
        'selected_imbalance_strategy': full_source['selected_imbalance_strategy'],
        'candidate_imbalance_strategies': list(
            context['candidate_strategy_policy']['svm']
        ),
        'outer_fold_selection_trace': _json_safe(outer_fold_details),
        'full_source_selected_params': _json_safe(full_source['selected_params']),
        'full_source_selected_imbalance_strategy': full_source[
            'selected_imbalance_strategy'
        ],
        'full_source_selection_subject_f1': round(
            float(full_source['selection_subject_f1']),
            6,
        ),
        'full_source_selection_stride_f1': round(
            float(full_source['selection_stride_f1']),
            6,
        ),
        'full_source_selection_subject_log_loss': (
            round(float(full_source['selection_subject_log_loss']), 6)
            if full_source.get('selection_subject_log_loss') is not None else None
        ),
        'full_source_selection_trace': _json_safe(full_source['selection_trace']),
        'full_source_selection_trace_path': str(full_trace_path),
        'full_source_selection_trace_sha256': full_trace_sha,
        'full_source_model_path': str(full_source_model_path.relative_to(context['paths']['models_dir'])),
        'full_source_model_sha256': full_source_model_sha256,
    }

    output = _build_within_condition_output(
        condition=context['condition'],
        pool_subjects=context['pool_subjects'],
        pool_strides=context['pool_strides'],
        selected_feature_cols=list(context['feature_cols']),
        feature_matrix_file=DEFAULT_FEATURE_MATRIX_FILE,
        feature_set_version=DEFAULT_FEATURE_SET_VERSION,
        normalization=DEFAULT_NORMALIZATION,
        models_dir=str(context['paths']['models_dir']),
        clf_results={'svm': classifier_result},
    )
    output['schema_version'] = CLASSIFIER_SHARD_SCHEMA_VERSION
    output['candidate_strategy_policy'] = {
        'svm': list(context['candidate_strategy_policy']['svm'])
    }
    output['subject_aggregation_rule'] = context['aggregation_rule']
    output['subject_probability_threshold'] = context['subject_probability_threshold']
    output['tie_break_rule'] = context['tie_break_rule']
    output['feature_matrix_hash'] = context['feature_matrix_hash']
    output['partition_hash'] = context['partition_hash']
    output['protocol_manifest_hash'] = context['protocol_manifest_hash']
    output['preprocessing_manifest_hash'] = context['preprocessing_manifest_hash']
    return output


def _build_final_condition_payload(
    *,
    context: dict[str, Any],
    shard_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    base = shard_payloads[0]
    merged_classifiers: dict[str, Any] = {}
    merged_policy: dict[str, list[str]] = {}
    for clf_name, payload in zip(CLASSIFIER_ORDER, shard_payloads):
        merged_classifiers[clf_name] = payload['classifiers'][clf_name]
        merged_policy.update(payload['candidate_strategy_policy'])

    output = {
        key: base[key]
        for key in (
            'condition',
            'pool_subjects',
            'pool_strides',
            'feature_cols',
            'n_features',
            'feature_matrix_file',
            'feature_set_version',
            'normalization',
            'models_dir',
        )
    }
    output['schema_version'] = FINAL_RESULTS_SCHEMA_VERSION
    output['classifiers'] = merged_classifiers
    output['candidate_strategy_policy'] = merged_policy
    output['subject_aggregation_rule'] = base['subject_aggregation_rule']
    output['subject_probability_threshold'] = base['subject_probability_threshold']
    output['tie_break_rule'] = base['tie_break_rule']
    output['feature_matrix_hash'] = base['feature_matrix_hash']
    output['partition_hash'] = base['partition_hash']
    output['protocol_manifest_hash'] = base['protocol_manifest_hash']
    output['preprocessing_manifest_hash'] = base['preprocessing_manifest_hash']
    return output


def _status_of_artifact(
    *,
    path: Path,
    validator: Any,
) -> str:
    if not path.exists():
        return 'missing'
    try:
        validator(_load_json(path))
    except Exception:
        return 'invalid'
    return 'completed'


def _status_summary(statuses: dict[str, str]) -> dict[str, Any]:
    expected = len(statuses)
    completed = sum(status == 'completed' for status in statuses.values())
    missing = sum(status == 'missing' for status in statuses.values())
    invalid = sum(status == 'invalid' for status in statuses.values())
    return {
        'expected': expected,
        'completed': completed,
        'missing': missing,
        'invalid': invalid,
        'per_artifact': statuses,
    }


@app.function(
    cpu=2,
    memory=2048,
    timeout=1800,
    max_containers=3,
    volumes={'/results': volume},
    retries=1,
)
def describe_condition_context_remote(
    condition: str,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
) -> str:
    volume.reload()
    context = _load_validated_authoritative_context(
        condition,
        namespace=namespace,
    )
    return json.dumps({
        'condition': condition,
        'namespace': namespace,
        'outer_subject_ids': context['outer_subject_ids'],
        'pool_subjects': context['pool_subjects'],
        'pool_strides': context['pool_strides'],
    }, indent=2)


@app.function(
    cpu=4,
    memory=4096,
    timeout=1800,
    max_containers=3,
    volumes={'/results': volume},
    retries=1,
)
def collect_status_remote(
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    targets: str = '',
    svm_fold_indices: str = '',
) -> str:
    volume.reload()
    parsed_targets = _parse_targets(targets)
    svm_fold_filter = _parse_svm_fold_indices(svm_fold_indices)

    summary: dict[str, Any] = {
        'namespace': namespace,
        'results_dir': str(_namespace_paths(namespace)['results_dir']),
        'models_dir': str(_namespace_paths(namespace)['models_dir']),
        'per_condition': {},
    }
    for condition in CONDITIONS:
        condition_targets = [
            clf_name
            for target_condition, clf_name in parsed_targets
            if target_condition == condition
        ]
        if parsed_targets and not condition_targets:
            continue
        include_all_classifiers = not parsed_targets or any(
            clf_name is None for clf_name in condition_targets
        )
        requested_classifiers = set(CLASSIFIER_ORDER if include_all_classifiers else [
            clf_name for clf_name in condition_targets if clf_name is not None
        ])
        try:
            context = _load_validated_authoritative_context(
                condition,
                namespace=namespace,
            )
        except Exception as exc:
            summary['per_condition'][condition] = {
                'context_status': 'invalid',
                'error': str(exc),
            }
            continue

        condition_summary: dict[str, Any] = {
            'context_status': 'ok',
        }
        non_svm_statuses: dict[str, str] = {}
        for clf_name in NON_SVM_CLASSIFIERS:
            if clf_name not in requested_classifiers:
                continue
            path = _classifier_shard_path(context['paths'], condition, clf_name)
            non_svm_statuses[clf_name] = _status_of_artifact(
                path=path,
                validator=lambda payload, *, _context=context, _clf=clf_name: _validate_single_classifier_shard_payload(
                    payload,
                    context=_context,
                    clf_name=_clf,
                ),
            )
        condition_summary['non_svm_classifier_shards'] = _status_summary(non_svm_statuses)

        if 'svm' in requested_classifiers:
            svm_specs = _selected_svm_fold_specs(
                context,
                svm_fold_indices=svm_fold_filter,
            )
            svm_fold_statuses: dict[str, str] = {}
            for fold_idx, subject_id in svm_specs:
                path = _svm_outer_fold_path(context['paths'], condition, fold_idx, subject_id)
                status = _status_of_artifact(
                    path=path,
                    validator=lambda payload, *, _context=context, _fold_idx=fold_idx, _subject_id=subject_id: _validate_svm_outer_fold_payload(
                        payload,
                        context=_context,
                        outer_fold_index=_fold_idx,
                        outer_subject_id=_subject_id,
                    ),
                )
                svm_fold_statuses[f'{fold_idx}:{subject_id}'] = status
            condition_summary['svm_outer_folds'] = _status_summary(svm_fold_statuses)

            svm_shard_path = _classifier_shard_path(context['paths'], condition, 'svm')
            svm_shard_status = _status_of_artifact(
                path=svm_shard_path,
                validator=lambda payload, *, _context=context: _validate_single_classifier_shard_payload(
                    payload,
                    context=_context,
                    clf_name='svm',
                ),
            )
            condition_summary['svm_classifier_shard'] = _status_summary(
                {'svm': svm_shard_status}
            )
        else:
            condition_summary['svm_outer_folds'] = _status_summary({})
            condition_summary['svm_classifier_shard'] = _status_summary({})

        if include_all_classifiers:
            final_path = _full_results_path(context['paths'], condition)
            final_status = _status_of_artifact(
                path=final_path,
                validator=lambda payload, *, _context=context: _validate_final_condition_payload(
                    payload,
                    context=_context,
                ),
            )
            condition_summary['final_condition_result'] = _status_summary(
                {condition: final_status}
            )
        else:
            condition_summary['final_condition_result'] = _status_summary({})
        summary['per_condition'][condition] = condition_summary

    return json.dumps(summary, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=86400,
    max_containers=12,
    volumes={'/results': volume},
    retries=1,
)
def run_classifier_shard_remote(
    condition: str,
    clf_name: str,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    force_recompute_invalid: bool = False,
) -> str:
    from train import run_within_condition

    if clf_name not in NON_SVM_CLASSIFIERS:
        raise ValueError(
            f'run_classifier_shard_remote only supports {NON_SVM_CLASSIFIERS}; '
            f'got {clf_name!r}.'
        )

    volume.reload()
    context = _load_validated_authoritative_context(
        condition,
        namespace=namespace,
    )
    path = _classifier_shard_path(context['paths'], condition, clf_name)
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_single_classifier_shard_payload(
            payload,
            context=context,
            clf_name=clf_name,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=f'classifier shard {condition}:{clf_name}',
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'reason': 'existing_valid_classifier_shard',
            'condition': condition,
            'classifier': clf_name,
            'namespace': namespace,
            'remote_shard_path': str(path),
        }, indent=2)

    output = run_within_condition(
        condition=condition,
        df=context['df'],
        control_subjects=context['control_A'],
        results_dir=context['paths']['results_dir'],
        models_dir=context['paths']['models_dir'],
        feature_cols=context['feature_cols'],
        feature_matrix_file=DEFAULT_FEATURE_MATRIX_FILE,
        feature_set_version=DEFAULT_FEATURE_SET_VERSION,
        normalization=DEFAULT_NORMALIZATION,
        results_filename=_classifier_shard_relpath(condition, clf_name),
        classifier_names=[clf_name],
        candidate_imbalance_strategies=context['candidate_strategy_policy'],
        subject_aggregation_rule=context['aggregation_rule'],
        tie_break_rule=context['tie_break_rule'],
        feature_matrix_hash=context['feature_matrix_hash'],
        partition_hash=context['partition_hash'],
        protocol_manifest_hash=context['protocol_manifest_hash'],
        preprocessing_manifest_hash=context['preprocessing_manifest_hash'],
    )
    written = _finalize_classifier_shard_output(
        output,
        context=context,
        clf_name=clf_name,
        path=path,
    )
    volume.commit()

    return json.dumps({
        'status': 'completed',
        'condition': condition,
        'classifier': clf_name,
        'namespace': namespace,
        'remote_shard_path': str(path),
        'selected_imbalance_strategy': written['classifiers'][clf_name][
            'selected_imbalance_strategy'
        ],
    }, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=21600,
    max_containers=16,
    volumes={'/results': volume},
    retries=1,
)
def run_svm_outer_fold_remote(
    condition: str,
    outer_fold_index: int,
    outer_subject_id: str,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    force_recompute_invalid: bool = False,
) -> str:
    volume.reload()
    context = _load_validated_authoritative_context(
        condition,
        namespace=namespace,
    )
    path = _svm_outer_fold_path(
        context['paths'],
        condition,
        outer_fold_index,
        outer_subject_id,
    )
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_svm_outer_fold_payload(
            payload,
            context=context,
            outer_fold_index=outer_fold_index,
            outer_subject_id=outer_subject_id,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=f'SVM outer-fold shard {condition}:{outer_fold_index}:{outer_subject_id}',
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'reason': 'existing_valid_outer_fold_shard',
            'condition': condition,
            'classifier': 'svm',
            'namespace': namespace,
            'outer_fold_index': outer_fold_index,
            'held_out_subject_id': outer_subject_id,
            'remote_shard_path': str(path),
        }, indent=2)

    payload = _build_svm_outer_fold_payload(
        context=context,
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
    )
    written = _write_json_with_digest(path, payload)
    _validate_svm_outer_fold_payload(
        written,
        context=context,
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
    )
    volume.commit()

    return json.dumps({
        'status': 'completed',
        'condition': condition,
        'classifier': 'svm',
        'namespace': namespace,
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'remote_shard_path': str(path),
    }, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=43200,
    max_containers=3,
    volumes={'/results': volume},
    retries=1,
)
def assemble_svm_classifier_shard_remote(
    condition: str,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    force_recompute_invalid: bool = False,
) -> str:
    volume.reload()
    context = _load_validated_authoritative_context(
        condition,
        namespace=namespace,
    )
    shard_path = _classifier_shard_path(context['paths'], condition, 'svm')
    existing = _load_reusable_payload(
        shard_path,
        validator=lambda payload: _validate_single_classifier_shard_payload(
            payload,
            context=context,
            clf_name='svm',
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=f'SVM classifier shard {condition}',
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'reason': 'existing_valid_classifier_shard',
            'condition': condition,
            'classifier': 'svm',
            'namespace': namespace,
            'remote_shard_path': str(shard_path),
        }, indent=2)

    volume.reload()
    fold_payloads: list[dict[str, Any]] = []
    for outer_fold_index, outer_subject_id in enumerate(context['outer_subject_ids']):
        fold_path = _svm_outer_fold_path(
            context['paths'],
            condition,
            outer_fold_index,
            outer_subject_id,
        )
        payload = _load_reusable_payload(
            fold_path,
            validator=lambda candidate_payload, *, _context=context, _fold_idx=outer_fold_index, _subject=outer_subject_id: _validate_svm_outer_fold_payload(
                candidate_payload,
                context=_context,
                outer_fold_index=_fold_idx,
                outer_subject_id=_subject,
            ),
            force_recompute_invalid=False,
            label=f'SVM outer-fold shard {condition}:{outer_fold_index}:{outer_subject_id}',
        )
        if payload is None:
            raise FileNotFoundError(
                f'Missing SVM outer-fold shard required for assembly: {fold_path}'
            )
        fold_payloads.append(payload)

    output = _assemble_svm_classifier_shard_payload(
        context=context,
        fold_payloads=fold_payloads,
    )
    written = _write_json_with_digest(shard_path, output)
    _validate_single_classifier_shard_payload(
        written,
        context=context,
        clf_name='svm',
    )
    volume.commit()

    return json.dumps({
        'status': 'completed',
        'condition': condition,
        'classifier': 'svm',
        'namespace': namespace,
        'remote_shard_path': str(shard_path),
        'selected_imbalance_strategy': written['classifiers']['svm'][
            'selected_imbalance_strategy'
        ],
    }, indent=2)


@app.function(
    cpu=4,
    memory=4096,
    timeout=3600,
    max_containers=3,
    volumes={'/results': volume},
    retries=1,
)
def assemble_condition_remote(
    condition: str,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    force_recompute_invalid: bool = False,
) -> str:
    volume.reload()
    context = _load_validated_authoritative_context(
        condition,
        namespace=namespace,
    )
    out_path = _full_results_path(context['paths'], condition)
    existing = _load_reusable_payload(
        out_path,
        validator=lambda payload: _validate_final_condition_payload(
            payload,
            context=context,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=f'final within-condition payload {condition}',
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'reason': 'existing_valid_final_condition_payload',
            'condition': condition,
            'namespace': namespace,
            'remote_results_path': str(out_path),
        }, indent=2)

    volume.reload()
    shard_payloads: list[dict[str, Any]] = []
    for clf_name in CLASSIFIER_ORDER:
        shard_path = _classifier_shard_path(context['paths'], condition, clf_name)
        payload = _load_reusable_payload(
            shard_path,
            validator=lambda classifier_payload, *, _context=context, _clf=clf_name: _validate_single_classifier_shard_payload(
                classifier_payload,
                context=_context,
                clf_name=_clf,
            ),
            force_recompute_invalid=False,
            label=f'classifier shard {condition}:{clf_name}',
        )
        if payload is None:
            raise FileNotFoundError(
                f'Missing classifier shard required for assembly: {shard_path}'
            )
        shard_payloads.append(payload)

    output = _build_final_condition_payload(
        context=context,
        shard_payloads=shard_payloads,
    )
    written = _write_json_with_digest(out_path, output)
    _validate_final_condition_payload(written, context=context)
    volume.commit()

    return json.dumps({
        'status': 'completed',
        'condition': condition,
        'namespace': namespace,
        'remote_results_path': str(out_path),
        'classifiers': CLASSIFIER_ORDER,
    }, indent=2)


def _spawn_non_svm_batches(
    *,
    specs: list[tuple[str, str]],
    namespace: str,
    force_recompute_invalid: bool,
    max_in_flight: int,
) -> None:
    if not specs:
        return
    for batch in _chunked(specs, max_in_flight):
        run_classifier_shard_remote.spawn_map(
            [condition for condition, _ in batch],
            [clf_name for _, clf_name in batch],
            kwargs={
                'namespace': namespace,
                'force_recompute_invalid': force_recompute_invalid,
            },
        )


def _spawn_svm_batches(
    *,
    specs: list[tuple[str, int, str]],
    namespace: str,
    force_recompute_invalid: bool,
    max_in_flight: int,
) -> None:
    if not specs:
        return
    for batch in _chunked(specs, max_in_flight):
        run_svm_outer_fold_remote.spawn_map(
            [condition for condition, _, _ in batch],
            [fold_idx for _, fold_idx, _ in batch],
            [subject_id for _, _, subject_id in batch],
            kwargs={
                'namespace': namespace,
                'force_recompute_invalid': force_recompute_invalid,
            },
        )


@app.function(
    cpu=1,
    memory=2048,
    timeout=1800,
    max_containers=1,
    volumes={'/results': volume},
    retries=0,
)
def run_svm_smoke_validator_probe_remote(
    condition: str,
    fold_idx: int,
    outer_subject_id: str,
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
) -> str:
    import copy

    volume.reload()
    context = _load_validated_authoritative_context(
        condition,
        namespace=namespace,
    )
    fold_path = _svm_outer_fold_path(
        context['paths'],
        condition,
        fold_idx,
        outer_subject_id,
    )
    payload = _read_volume_json(_volume_api_relpath(fold_path))
    _validate_svm_outer_fold_payload(
        payload,
        context=context,
        outer_fold_index=fold_idx,
        outer_subject_id=outer_subject_id,
    )

    corrupted = copy.deepcopy(payload)
    corrupted['payload_sha256'] = 'corrupted-for-smoke-validation'

    rejection_error: str | None = None
    try:
        _validate_svm_outer_fold_payload(
            corrupted,
            context=context,
            outer_fold_index=fold_idx,
            outer_subject_id=outer_subject_id,
        )
    except Exception as exc:
        rejection_error = str(exc)

    if rejection_error is None:
        raise RuntimeError(
            'Smoke validation expected a corrupted SVM fold payload to fail closed.'
        )

    return json.dumps({
        'status': 'validated',
        'condition': condition,
        'classifier': 'svm',
        'namespace': namespace,
        'outer_fold_index': fold_idx,
        'held_out_subject_id': outer_subject_id,
        'remote_shard_path': str(fold_path),
        'validator_rejected_corruption': True,
        'rejection_error': rejection_error,
    }, indent=2)


@app.local_entrypoint()
def main(
    action: str = 'submit',
    targets: str = '',
    namespace: str = DEFAULT_RESULTS_NAMESPACE,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    svm_fold_indices: str = '',
    force_recompute_invalid: bool = False,
) -> None:
    parsed_targets = _parse_targets(targets)
    parsed_fold_indices = _parse_svm_fold_indices(svm_fold_indices)

    if action == 'status':
        print(
            json.dumps(
                json.loads(
                    collect_status_remote.remote(
                        namespace=namespace,
                        targets=targets,
                        svm_fold_indices=svm_fold_indices,
                    )
                ),
                indent=2,
            ),
            flush=True,
        )
        return

    if action == 'submit':
        submit_pairs = _expand_submit_pairs(parsed_targets)
        non_svm_specs = [
            (condition, clf_name)
            for condition, clf_name in submit_pairs
            if clf_name != 'svm'
        ]
        svm_specs: list[tuple[str, int, str]] = []
        for condition, clf_name in submit_pairs:
            if clf_name != 'svm':
                continue
            context = json.loads(
                describe_condition_context_remote.remote(
                    condition,
                    namespace=namespace,
                )
            )
            outer_subject_ids = context['outer_subject_ids']
            if parsed_fold_indices is None:
                svm_specs.extend(
                    (condition, fold_idx, outer_subject_id)
                    for fold_idx, outer_subject_id in enumerate(outer_subject_ids)
                )
            else:
                for fold_idx in parsed_fold_indices:
                    svm_specs.append((condition, fold_idx, outer_subject_ids[fold_idx]))

        print('Submitting authoritative Step 2 shards on Modal...', flush=True)
        print(
            f'Namespace={namespace}. Inputs are read from gait-results:/processed_v4. '
            f'Results are written to gait-results:/{namespace} and '
            f'gait-results:/{_results_namespace_to_models_namespace(namespace)}.',
            flush=True,
        )
        print(
            'Classifier-specific strategy policy, aggregation rule, probability '
            'threshold, and tie-break rule are loaded from the frozen protocol '
            'manifest and validated against the preprocessing manifest.',
            flush=True,
        )
        print(
            f'max_in_flight={max_in_flight} controls bounded spawn_map '
            'submission batches.',
            flush=True,
        )
        _spawn_non_svm_batches(
            specs=non_svm_specs,
            namespace=namespace,
            force_recompute_invalid=force_recompute_invalid,
            max_in_flight=max_in_flight,
        )
        _spawn_svm_batches(
            specs=svm_specs,
            namespace=namespace,
            force_recompute_invalid=force_recompute_invalid,
            max_in_flight=max_in_flight,
        )
        print('\nSubmission complete.', flush=True)
        print('Monitor with:', flush=True)
        print('  modal app list', flush=True)
        print('  modal app logs gait-transfer-training-v4 -f', flush=True)
        print('  modal run scripts/training/run_within_condition_modal.py --action status', flush=True)
        return

    if action == 'assemble':
        svm_conditions, final_conditions = _expand_assemble_requests(parsed_targets)
        if svm_conditions:
            print('Assembling SVM classifier shards...', flush=True)
            for condition in svm_conditions:
                result = json.loads(
                    assemble_svm_classifier_shard_remote.remote(
                        condition,
                        namespace=namespace,
                        force_recompute_invalid=force_recompute_invalid,
                    )
                )
                print(json.dumps(result, indent=2), flush=True)
        if final_conditions:
            print('\nAssembling final per-condition within-condition results...', flush=True)
            for condition in final_conditions:
                result = json.loads(
                    assemble_condition_remote.remote(
                        condition,
                        namespace=namespace,
                        force_recompute_invalid=force_recompute_invalid,
                    )
                )
                print(json.dumps(result, indent=2), flush=True)
        if not svm_conditions and not final_conditions:
            print('No assemble targets requested.', flush=True)
        return

    if action == 'smoke':
        if namespace == DEFAULT_RESULTS_NAMESPACE:
            raise SystemExit(
                'Smoke mode must use an isolated namespace, for example '
                '--namespace results_v4_smoke.'
            )
        submit_pairs = _expand_submit_pairs(parsed_targets)
        light_specs = [
            (condition, clf_name)
            for condition, clf_name in submit_pairs
            if clf_name in {'qda', 'dt', 'knn'}
        ]
        svm_targets = [
            (condition, clf_name)
            for condition, clf_name in submit_pairs
            if clf_name == 'svm'
        ]
        if not light_specs or not svm_targets:
            raise SystemExit(
                'Smoke mode requires at least one light-family target and one svm '
                'target, for example --targets "als:qda,hd:svm".'
            )
        if parsed_fold_indices is None:
            parsed_fold_indices = [0]

        light_condition, light_classifier = light_specs[0]
        print(
            f'Running isolated smoke in namespace={namespace}: '
            f'{light_condition}:{light_classifier} and selected svm folds.',
            flush=True,
        )
        first_light = json.loads(
            run_classifier_shard_remote.remote(
                light_condition,
                light_classifier,
                namespace=namespace,
                force_recompute_invalid=force_recompute_invalid,
            )
        )
        second_light = json.loads(
            run_classifier_shard_remote.remote(
                light_condition,
                light_classifier,
                namespace=namespace,
                force_recompute_invalid=force_recompute_invalid,
            )
        )
        print(json.dumps(first_light, indent=2), flush=True)
        print(json.dumps(second_light, indent=2), flush=True)

        for condition, _ in svm_targets:
            context = json.loads(
                describe_condition_context_remote.remote(
                    condition,
                    namespace=namespace,
                )
            )
            outer_subject_ids = context['outer_subject_ids']
            for fold_idx in parsed_fold_indices:
                outer_subject_id = outer_subject_ids[fold_idx]
                first_fold = json.loads(
                    run_svm_outer_fold_remote.remote(
                        condition,
                        fold_idx,
                        outer_subject_id,
                        namespace=namespace,
                        force_recompute_invalid=force_recompute_invalid,
                    )
                )
                second_fold = json.loads(
                    run_svm_outer_fold_remote.remote(
                        condition,
                        fold_idx,
                        outer_subject_id,
                        namespace=namespace,
                        force_recompute_invalid=force_recompute_invalid,
                    )
                )
                print(json.dumps(first_fold, indent=2), flush=True)
                print(json.dumps(second_fold, indent=2), flush=True)
                smoke_probe = json.loads(
                    run_svm_smoke_validator_probe_remote.remote(
                        condition,
                        fold_idx,
                        outer_subject_id,
                        namespace=namespace,
                    )
                )
                print(json.dumps(smoke_probe, indent=2), flush=True)

        status = json.loads(
            collect_status_remote.remote(
                namespace=namespace,
                targets=targets,
                svm_fold_indices=','.join(str(idx) for idx in parsed_fold_indices),
            )
        )
        print(json.dumps(status, indent=2), flush=True)
        return

    raise SystemExit(
        f"Unknown action '{action}'. Expected one of "
        "{'status', 'submit', 'assemble', 'smoke'}."
    )
