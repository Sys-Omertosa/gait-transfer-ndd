"""
Modal recovery runner for the publication-track v4 within-condition benchmark.

This companion runner is recovery-only. It stages fragmented authoritative
Step 2 recovery artifacts on the dedicated `gait-results-v4-recovery` Volume
without writing to the primary `gait-results` Volume or modifying the frozen
authoritative runner.

Usage:
    modal run scripts/training/run_within_condition_recovery_modal.py --action recovery-status --targets "pd:rf,pd:xgb,hd:rf,als:rf,als:lgbm"
    modal run --detach scripts/training/run_within_condition_recovery_modal.py --action recovery-submit --targets "pd:rf,pd:xgb,hd:rf,als:rf,als:lgbm"
    modal run scripts/training/run_within_condition_recovery_modal.py --action recovery-assemble-missing --targets "pd:rf,pd:xgb,hd:rf,als:rf,als:lgbm"
    modal run scripts/training/run_within_condition_recovery_modal.py --action recovery-promotion-manifest --targets "pd:rf,pd:xgb,hd:rf,als:rf,als:lgbm"
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import modal

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-training-v4-recovery', image=image)
recovery_volume = modal.Volume.from_name(
    'gait-results-v4-recovery',
    create_if_missing=False,
)

CONDITIONS = ('pd', 'hd', 'als')
RECOVERABLE_TARGETS = (
    ('pd', 'rf'),
    ('pd', 'xgb'),
    ('hd', 'rf'),
    ('als', 'rf'),
    ('als', 'lgbm'),
)
CLASSIFIER_ORDER = ('rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm')
DEFAULT_RESULTS_NAMESPACE = 'results_v4'
DEFAULT_PROCESSED_NAMESPACE = 'processed_v4'
DEFAULT_MODELS_NAMESPACE = 'models_v4'
DEFAULT_SUBMISSION_BATCH_SIZE = 64
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
OUTER_CANDIDATE_FRAGMENT_SCHEMA_VERSION = (
    'v4-within-condition-recovery-outer-candidate-v1'
)
OUTER_SELECTED_REFIT_SCHEMA_VERSION = (
    'v4-within-condition-recovery-outer-selected-refit-v1'
)
FULL_SOURCE_CANDIDATE_FRAGMENT_SCHEMA_VERSION = (
    'v4-within-condition-recovery-full-source-candidate-v1'
)
FULL_SOURCE_SELECTED_MODEL_SCHEMA_VERSION = (
    'v4-within-condition-recovery-full-source-selected-model-v1'
)
PROMOTION_MANIFEST_SCHEMA_VERSION = (
    'v4-within-condition-recovery-promotion-manifest-v1'
)
RECOVERY_RANDOM_SEED = 42


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _normalize_target_key(condition: str, clf_name: str) -> str:
    return f'{condition}:{clf_name}'


def _parse_recovery_targets(targets: str) -> list[tuple[str, str]]:
    if not targets.strip():
        raise SystemExit(
            'Recovery actions require an explicit non-empty --targets list, '
            'for example --targets "pd:rf,pd:xgb,hd:rf,als:rf,als:lgbm".'
        )

    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    allowed = set(RECOVERABLE_TARGETS)
    for raw_token in targets.split(','):
        token = raw_token.strip().lower()
        if not token:
            continue
        parts = token.split(':')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid recovery target '{raw_token}'. Expected condition:classifier."
            )
        condition, clf_name = parts
        pair = (condition, clf_name)
        if pair not in allowed:
            raise ValueError(
                f"Unsupported recovery target '{token}'. Expected one of "
                f'{sorted(_normalize_target_key(*item) for item in RECOVERABLE_TARGETS)}.'
            )
        if pair in seen:
            continue
        seen.add(pair)
        parsed.append(pair)

    if not parsed:
        raise SystemExit(
            'Recovery actions require at least one explicit recoverable target.'
        )
    return parsed


def _recovery_paths(
    *,
    volume_root: Path = Path('/results'),
) -> dict[str, Path]:
    processed_dir = volume_root / DEFAULT_PROCESSED_NAMESPACE
    results_dir = volume_root / DEFAULT_RESULTS_NAMESPACE
    models_dir = volume_root / DEFAULT_MODELS_NAMESPACE
    classifier_shard_dir = results_dir / 'classifier_shards'
    selection_trace_dir = results_dir / 'selection_traces'
    fragments_dir = results_dir / 'recovery_fragments'
    return {
        'processed_dir': processed_dir,
        'results_dir': results_dir,
        'models_dir': models_dir,
        'classifier_shard_dir': classifier_shard_dir,
        'selection_trace_dir': selection_trace_dir,
        'fragments_dir': fragments_dir,
    }


def _classifier_shard_relpath(condition: str, clf_name: str) -> str:
    return f'classifier_shards/{condition}_{clf_name}_results_v4_shard.json'


def _classifier_shard_path(
    paths: dict[str, Path],
    condition: str,
    clf_name: str,
) -> Path:
    return paths['results_dir'] / _classifier_shard_relpath(condition, clf_name)


def _target_fragment_root(
    paths: dict[str, Path],
    condition: str,
    clf_name: str,
) -> Path:
    return paths['fragments_dir'] / f'{condition}_{clf_name}'


def _outer_candidate_fragment_path(
    paths: dict[str, Path],
    condition: str,
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
    candidate_index: int,
    imbalance_strategy: str,
    candidate_hash: str,
) -> Path:
    return (
        _target_fragment_root(paths, condition, clf_name)
        / 'outer_candidates'
        / f'fold_{outer_fold_index:02d}_{outer_subject_id}'
        / (
            f'candidate_{candidate_index:03d}_{imbalance_strategy}_'
            f'{candidate_hash[:12]}.json'
        )
    )


def _outer_selected_refit_fragment_path(
    paths: dict[str, Path],
    condition: str,
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
) -> Path:
    return (
        _target_fragment_root(paths, condition, clf_name)
        / 'outer_selected_refits'
        / f'fold_{outer_fold_index:02d}_{outer_subject_id}.json'
    )


def _full_source_candidate_fragment_path(
    paths: dict[str, Path],
    condition: str,
    clf_name: str,
    candidate_index: int,
    imbalance_strategy: str,
    candidate_hash: str,
) -> Path:
    return (
        _target_fragment_root(paths, condition, clf_name)
        / 'full_source_candidates'
        / (
            f'candidate_{candidate_index:03d}_{imbalance_strategy}_'
            f'{candidate_hash[:12]}.json'
        )
    )


def _full_source_selected_model_fragment_path(
    paths: dict[str, Path],
    condition: str,
    clf_name: str,
) -> Path:
    return (
        _target_fragment_root(paths, condition, clf_name)
        / 'full_source_selected_model.json'
    )


def _recovery_promotion_manifest_path(paths: dict[str, Path]) -> Path:
    return paths['results_dir'] / 'recovery_promotion_manifest.json'


def _volume_relpath(path: Path) -> str:
    return path.relative_to(Path('/results')).as_posix()


def _resolve_stored_path(path_text: str, *, default_parent: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else default_parent / path


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError('submission_batch_size must be positive.')
    return [values[idx:idx + size] for idx in range(0, len(values), size)]


def _recovery_execution_identity() -> dict[str, str]:
    import train
    import v4_provenance
    from v4_provenance import canonical_payload_sha256, sha256_file

    recovery_runner_sha256 = sha256_file(Path(__file__))
    train_module_sha256 = sha256_file(Path(train.__file__).resolve())
    v4_provenance_module_sha256 = sha256_file(
        Path(v4_provenance.__file__).resolve()
    )
    recovery_execution_sha256 = canonical_payload_sha256({
        'recovery_runner_sha256': recovery_runner_sha256,
        'train_module_sha256': train_module_sha256,
        'v4_provenance_module_sha256': v4_provenance_module_sha256,
    })
    return {
        'recovery_runner_sha256': recovery_runner_sha256,
        'train_module_sha256': train_module_sha256,
        'v4_provenance_module_sha256': v4_provenance_module_sha256,
        'recovery_execution_sha256': recovery_execution_sha256,
    }


def _status_summary(statuses: dict[str, str]) -> dict[str, Any]:
    expected = len(statuses)
    completed = sum(status == 'completed' for status in statuses.values())
    missing = sum(status == 'missing' for status in statuses.values())
    invalid = sum(status == 'invalid' for status in statuses.values())
    missing_examples = [name for name, status in statuses.items() if status == 'missing'][:10]
    invalid_examples = [name for name, status in statuses.items() if status == 'invalid'][:10]
    return {
        'expected': expected,
        'completed': completed,
        'missing': missing,
        'invalid': invalid,
        'missing_examples': missing_examples,
        'invalid_examples': invalid_examples,
    }


def _validate_payload_digest(payload: dict[str, Any]) -> None:
    from v4_provenance import canonical_payload_sha256

    expected = payload.get('payload_sha256')
    if not isinstance(expected, str) or not expected:
        raise ValueError('Artifact payload is missing payload_sha256.')
    actual = canonical_payload_sha256(payload)
    if actual != expected:
        raise ValueError('Artifact payload_sha256 mismatch.')


def _write_json_with_digest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from v4_provenance import atomic_write_json, canonical_payload_sha256, utc_now_iso

    materialized = dict(payload)
    materialized['completed_at_utc'] = utc_now_iso()
    materialized['payload_sha256'] = canonical_payload_sha256(materialized)
    atomic_write_json(path, materialized)
    return materialized


def _atomic_save_pipeline_artifact(pipeline: Any, path: Path) -> str:
    import joblib

    from v4_provenance import sha256_file

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(pipeline, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return sha256_file(path)


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
    from v4_provenance import sha256_file

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
        if sha256_file(artifact_path) != preprocessing_manifest[hash_key]:
            raise ValueError(
                f'Preprocessing manifest hash mismatch for {artifact_path}.'
            )


def _load_validated_recovery_context(
    condition: str,
    *,
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

    paths = _recovery_paths(volume_root=volume_root)
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
            'Missing authoritative Step 1 artifacts on the recovery volume:\n'
            + '\n'.join(missing)
        )

    protocol_manifest = _load_json(protocol_manifest_path)
    _validate_protocol_manifest(protocol_manifest)
    policy = _normalize_strategy_policy(protocol_manifest)

    protocol_manifest_hash = sha256_file(protocol_manifest_path)
    feature_matrix_hash = sha256_file(features_path)
    partition_hash = sha256_file(partition_path)
    preprocessing_manifest_hash = sha256_file(preprocessing_manifest_path)
    execution_identity = _recovery_execution_identity()

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
        'preprocessing_manifest_hash': preprocessing_manifest_hash,
        'recovery_runner_sha256': execution_identity['recovery_runner_sha256'],
        'train_module_sha256': execution_identity['train_module_sha256'],
        'v4_provenance_module_sha256': execution_identity[
            'v4_provenance_module_sha256'
        ],
        'recovery_execution_sha256': execution_identity[
            'recovery_execution_sha256'
        ],
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


def _candidate_specs_for_target(
    context: dict[str, Any],
    clf_name: str,
) -> list[dict[str, Any]]:
    from sklearn.model_selection import ParameterGrid

    from train import get_classifier_configs
    from v4_provenance import canonical_payload_sha256

    configs = get_classifier_configs()
    if clf_name not in configs:
        raise KeyError(f'Unknown classifier {clf_name!r}.')
    param_grid = list(ParameterGrid(configs[clf_name]['param_grid']))
    strategies = context['candidate_strategy_policy'][clf_name]
    specs: list[dict[str, Any]] = []
    candidate_index = 0
    classifier_grid_hash = canonical_payload_sha256(configs[clf_name]['param_grid'])
    for imbalance_strategy in strategies:
        for params in param_grid:
            candidate_hash = canonical_payload_sha256({
                'classifier': clf_name,
                'imbalance_strategy': imbalance_strategy,
                'params': params,
            })
            specs.append({
                'candidate_index': candidate_index,
                'candidate_hash': candidate_hash,
                'imbalance_strategy': imbalance_strategy,
                'params': dict(params),
                'classifier_grid_hash': classifier_grid_hash,
            })
            candidate_index += 1
    return specs


def _candidate_spec_by_index(
    context: dict[str, Any],
    clf_name: str,
    candidate_index: int,
) -> dict[str, Any]:
    specs = _candidate_specs_for_target(context, clf_name)
    if candidate_index < 0 or candidate_index >= len(specs):
        raise IndexError(
            f'Candidate index {candidate_index} is out of range for {clf_name!r}.'
        )
    return specs[candidate_index]


def _candidate_ranking_public(candidate_rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rankings: list[dict[str, Any]] = []
    for rank_idx, candidate in enumerate(candidate_rankings):
        public_rankings.append({
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
        })
    return public_rankings


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


def _validate_common_fragment_identity(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    condition: str,
    clf_name: str,
    schema_version: str,
    stage: str,
    classifier_grid_hash: str,
) -> None:
    _validate_payload_digest(payload)
    if payload.get('schema_version') != schema_version:
        raise ValueError(f'{stage} fragment schema_version mismatch.')
    if payload.get('stage') != stage:
        raise ValueError(f'{stage} fragment stage mismatch.')
    if payload.get('condition') != condition:
        raise ValueError(f'{stage} fragment condition mismatch.')
    if payload.get('classifier') != clf_name:
        raise ValueError(f'{stage} fragment classifier mismatch.')
    if payload.get('subject_aggregation_rule') != context['aggregation_rule']:
        raise ValueError(f'{stage} fragment aggregation_rule mismatch.')
    if float(payload.get('subject_probability_threshold')) != context['subject_probability_threshold']:
        raise ValueError(f'{stage} fragment subject_probability_threshold mismatch.')
    if payload.get('tie_break_rule') != context['tie_break_rule']:
        raise ValueError(f'{stage} fragment tie_break_rule mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_hash']:
        raise ValueError(f'{stage} fragment feature_matrix_hash mismatch.')
    if payload.get('partition_hash') != context['partition_hash']:
        raise ValueError(f'{stage} fragment partition_hash mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_hash']:
        raise ValueError(f'{stage} fragment protocol_manifest_hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_hash']:
        raise ValueError(f'{stage} fragment preprocessing_manifest_hash mismatch.')
    if payload.get('recovery_runner_sha256') != context['recovery_runner_sha256']:
        raise ValueError(f'{stage} fragment recovery_runner_sha256 mismatch.')
    if payload.get('train_module_sha256') != context['train_module_sha256']:
        raise ValueError(f'{stage} fragment train_module_sha256 mismatch.')
    if (
        payload.get('v4_provenance_module_sha256')
        != context['v4_provenance_module_sha256']
    ):
        raise ValueError(
            f'{stage} fragment v4_provenance_module_sha256 mismatch.'
        )
    if payload.get('recovery_execution_sha256') != context['recovery_execution_sha256']:
        raise ValueError(f'{stage} fragment recovery_execution_sha256 mismatch.')
    if payload.get('classifier_grid_hash') != classifier_grid_hash:
        raise ValueError(f'{stage} fragment classifier_grid_hash mismatch.')
    if int(payload.get('seed')) != RECOVERY_RANDOM_SEED:
        raise ValueError(f'{stage} fragment seed mismatch.')
    if payload.get('candidate_strategy_policy') != list(
        context['candidate_strategy_policy'][clf_name]
    ):
        raise ValueError(f'{stage} fragment candidate_strategy_policy mismatch.')


def _validate_candidate_fragment_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    clf_name: str,
    expected_stage: str,
    expected_schema_version: str,
    candidate_spec: dict[str, Any],
    outer_fold_index: int | None = None,
    outer_subject_id: str | None = None,
) -> None:
    _validate_common_fragment_identity(
        payload,
        context=context,
        condition=context['condition'],
        clf_name=clf_name,
        schema_version=expected_schema_version,
        stage=expected_stage,
        classifier_grid_hash=candidate_spec['classifier_grid_hash'],
    )
    if int(payload.get('candidate_index')) != candidate_spec['candidate_index']:
        raise ValueError(f'{expected_stage} candidate_index mismatch.')
    if payload.get('candidate_hash') != candidate_spec['candidate_hash']:
        raise ValueError(f'{expected_stage} candidate_hash mismatch.')
    if payload.get('imbalance_strategy') != candidate_spec['imbalance_strategy']:
        raise ValueError(f'{expected_stage} imbalance_strategy mismatch.')
    if payload.get('params') != candidate_spec['params']:
        raise ValueError(f'{expected_stage} params mismatch.')
    if outer_fold_index is not None:
        if int(payload.get('outer_fold_index')) != outer_fold_index:
            raise ValueError(f'{expected_stage} outer_fold_index mismatch.')
        if payload.get('held_out_subject_id') != outer_subject_id:
            raise ValueError(f'{expected_stage} held_out_subject_id mismatch.')

    summary = payload.get('candidate_summary')
    if not isinstance(summary, dict):
        raise ValueError(f'{expected_stage} fragment is missing candidate_summary.')
    required_summary_keys = (
        'candidate_index',
        'imbalance_strategy',
        'params',
        'inner_subject_f1',
        'inner_stride_f1',
        'inner_subject_log_loss',
        'aggregation_rule',
        'tie_break_rule',
        'subject_ids',
        'subject_y_true',
        'subject_y_pred',
        'subject_scores',
    )
    missing = [key for key in required_summary_keys if key not in summary]
    if missing:
        raise ValueError(
            f'{expected_stage} candidate_summary is missing required keys: {missing}'
        )
    if int(summary.get('candidate_index')) != candidate_spec['candidate_index']:
        raise ValueError(f'{expected_stage} candidate_summary candidate_index mismatch.')
    if summary.get('imbalance_strategy') != candidate_spec['imbalance_strategy']:
        raise ValueError(
            f'{expected_stage} candidate_summary imbalance_strategy mismatch.'
        )
    if summary.get('params') != candidate_spec['params']:
        raise ValueError(f'{expected_stage} candidate_summary params mismatch.')


def _validate_outer_selected_refit_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
) -> None:
    from v4_provenance import sha256_file

    candidate_specs = _candidate_specs_for_target(context, clf_name)
    classifier_grid_hash = candidate_specs[0]['classifier_grid_hash']
    _validate_common_fragment_identity(
        payload,
        context=context,
        condition=context['condition'],
        clf_name=clf_name,
        schema_version=OUTER_SELECTED_REFIT_SCHEMA_VERSION,
        stage='outer_selected_refit',
        classifier_grid_hash=classifier_grid_hash,
    )
    if int(payload.get('outer_fold_index')) != outer_fold_index:
        raise ValueError('outer_selected_refit outer_fold_index mismatch.')
    if payload.get('held_out_subject_id') != outer_subject_id:
        raise ValueError('outer_selected_refit held_out_subject_id mismatch.')

    selected_index = int(payload.get('selected_candidate_index'))
    selected_spec = candidate_specs[selected_index]
    if payload.get('selected_candidate_hash') != selected_spec['candidate_hash']:
        raise ValueError('outer_selected_refit selected_candidate_hash mismatch.')
    if payload.get('selected_imbalance_strategy') != selected_spec['imbalance_strategy']:
        raise ValueError('outer_selected_refit selected_imbalance_strategy mismatch.')
    if payload.get('selected_params') != selected_spec['params']:
        raise ValueError('outer_selected_refit selected_params mismatch.')

    required_keys = (
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
            f'outer_selected_refit fragment is missing required keys: {missing}'
        )
    y_true = payload['y_true']
    y_pred = payload['y_pred']
    y_prob = payload['y_prob']
    subject_ids = payload['subject_ids']
    if len({len(y_true), len(y_pred), len(y_prob), len(subject_ids)}) != 1:
        raise ValueError('outer_selected_refit arrays do not share the same length.')
    if any(subject_id != outer_subject_id for subject_id in subject_ids):
        raise ValueError(
            'outer_selected_refit subject_ids do not match held-out subject.'
        )
    fold_detail = payload['public_outer_fold_detail']
    if fold_detail.get('held_out_subject_id') != outer_subject_id:
        raise ValueError(
            'outer_selected_refit public_outer_fold_detail held_out_subject_id mismatch.'
        )
    model_relpath = fold_detail.get('fold_model_relpath')
    model_sha = fold_detail.get('fold_model_sha256')
    if not isinstance(model_relpath, str) or not isinstance(model_sha, str):
        raise ValueError('outer_selected_refit is missing fold-model provenance.')
    fold_model_path = _resolve_stored_path(
        model_relpath,
        default_parent=context['paths']['models_dir'],
    )
    if not fold_model_path.exists():
        raise FileNotFoundError(
            f'outer_selected_refit fold model is missing: {fold_model_path}'
        )
    if sha256_file(fold_model_path) != model_sha:
        raise ValueError(
            f'outer_selected_refit fold model SHA-256 mismatch for {fold_model_path}.'
        )


def _validate_full_source_selected_model_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    clf_name: str,
) -> None:
    from v4_provenance import sha256_file

    candidate_specs = _candidate_specs_for_target(context, clf_name)
    classifier_grid_hash = candidate_specs[0]['classifier_grid_hash']
    _validate_common_fragment_identity(
        payload,
        context=context,
        condition=context['condition'],
        clf_name=clf_name,
        schema_version=FULL_SOURCE_SELECTED_MODEL_SCHEMA_VERSION,
        stage='full_source_selected_model',
        classifier_grid_hash=classifier_grid_hash,
    )

    selected_index = int(payload.get('selected_candidate_index'))
    selected_spec = candidate_specs[selected_index]
    if payload.get('selected_candidate_hash') != selected_spec['candidate_hash']:
        raise ValueError('full_source_selected_model selected_candidate_hash mismatch.')
    if payload.get('selected_imbalance_strategy') != selected_spec['imbalance_strategy']:
        raise ValueError(
            'full_source_selected_model selected_imbalance_strategy mismatch.'
        )
    if payload.get('selected_params') != selected_spec['params']:
        raise ValueError('full_source_selected_model selected_params mismatch.')

    selection_trace = payload.get('selection_trace')
    if not isinstance(selection_trace, list) or not selection_trace:
        raise ValueError('full_source_selected_model is missing selection_trace.')

    model_relpath = payload.get('full_source_model_path')
    model_sha = payload.get('full_source_model_sha256')
    if not isinstance(model_relpath, str) or not isinstance(model_sha, str):
        raise ValueError(
            'full_source_selected_model is missing full-source model provenance.'
        )
    full_source_model_path = _resolve_stored_path(
        model_relpath,
        default_parent=context['paths']['models_dir'],
    )
    if not full_source_model_path.exists():
        raise FileNotFoundError(
            f'full_source_selected_model artifact is missing: {full_source_model_path}'
        )
    if sha256_file(full_source_model_path) != model_sha:
        raise ValueError(
            'full_source_selected_model SHA-256 mismatch for '
            f'{full_source_model_path}.'
        )


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
        raise ValueError(
            'Classifier payload is missing full-source selection-trace provenance.'
        )
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
            f'Full-source selection trace SHA-256 mismatch for {full_trace_path}'
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


def _validate_classifier_shard_payload(
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
    if payload.get('recovery_runner_sha256') != context['recovery_runner_sha256']:
        raise ValueError('Classifier shard recovery_runner_sha256 mismatch.')
    if payload.get('train_module_sha256') != context['train_module_sha256']:
        raise ValueError('Classifier shard train_module_sha256 mismatch.')
    if (
        payload.get('v4_provenance_module_sha256')
        != context['v4_provenance_module_sha256']
    ):
        raise ValueError(
            'Classifier shard v4_provenance_module_sha256 mismatch.'
        )
    if payload.get('recovery_execution_sha256') != context['recovery_execution_sha256']:
        raise ValueError('Classifier shard recovery_execution_sha256 mismatch.')
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


def _build_outer_candidate_fragment_payload(
    *,
    context: dict[str, Any],
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
    candidate_spec: dict[str, Any],
) -> dict[str, Any]:
    from train import _select_best_grouped_candidate, get_classifier_configs

    train_idx, _ = _resolve_outer_fold_indices(
        X=context['X'],
        y=context['y'],
        groups=context['groups'],
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
    )
    X_train = context['X'][train_idx]
    y_train = context['y'][train_idx]
    groups_train = context['groups'][train_idx]
    clf_config = get_classifier_configs()[clf_name]
    singleton_grid = {
        key: [value]
        for key, value in candidate_spec['params'].items()
    }
    best_candidate, _ = _select_best_grouped_candidate(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        clf_template=clf_config['clf'],
        param_grid=singleton_grid,
        classifier_name=clf_name,
        candidate_imbalance_strategies=(candidate_spec['imbalance_strategy'],),
        subject_aggregation_rule=context['aggregation_rule'],
        tie_break_rule=context['tie_break_rule'],
    )
    best_candidate['candidate_index'] = int(candidate_spec['candidate_index'])
    return {
        'schema_version': OUTER_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        'stage': 'outer_candidate',
        'condition': context['condition'],
        'classifier': clf_name,
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'candidate_index': int(candidate_spec['candidate_index']),
        'candidate_hash': candidate_spec['candidate_hash'],
        'imbalance_strategy': candidate_spec['imbalance_strategy'],
        'params': dict(candidate_spec['params']),
        'candidate_strategy_policy': list(context['candidate_strategy_policy'][clf_name]),
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'recovery_runner_sha256': context['recovery_runner_sha256'],
        'train_module_sha256': context['train_module_sha256'],
        'v4_provenance_module_sha256': context['v4_provenance_module_sha256'],
        'recovery_execution_sha256': context['recovery_execution_sha256'],
        'classifier_grid_hash': candidate_spec['classifier_grid_hash'],
        'seed': RECOVERY_RANDOM_SEED,
        'candidate_summary': best_candidate,
    }


def _build_full_source_candidate_fragment_payload(
    *,
    context: dict[str, Any],
    clf_name: str,
    candidate_spec: dict[str, Any],
) -> dict[str, Any]:
    from train import _select_best_grouped_candidate, get_classifier_configs

    clf_config = get_classifier_configs()[clf_name]
    singleton_grid = {
        key: [value]
        for key, value in candidate_spec['params'].items()
    }
    best_candidate, _ = _select_best_grouped_candidate(
        X_train=context['X'],
        y_train=context['y'],
        groups_train=context['groups'],
        clf_template=clf_config['clf'],
        param_grid=singleton_grid,
        classifier_name=clf_name,
        candidate_imbalance_strategies=(candidate_spec['imbalance_strategy'],),
        subject_aggregation_rule=context['aggregation_rule'],
        tie_break_rule=context['tie_break_rule'],
    )
    best_candidate['candidate_index'] = int(candidate_spec['candidate_index'])
    return {
        'schema_version': FULL_SOURCE_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        'stage': 'full_source_candidate',
        'condition': context['condition'],
        'classifier': clf_name,
        'candidate_index': int(candidate_spec['candidate_index']),
        'candidate_hash': candidate_spec['candidate_hash'],
        'imbalance_strategy': candidate_spec['imbalance_strategy'],
        'params': dict(candidate_spec['params']),
        'candidate_strategy_policy': list(context['candidate_strategy_policy'][clf_name]),
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'recovery_runner_sha256': context['recovery_runner_sha256'],
        'train_module_sha256': context['train_module_sha256'],
        'v4_provenance_module_sha256': context['v4_provenance_module_sha256'],
        'recovery_execution_sha256': context['recovery_execution_sha256'],
        'classifier_grid_hash': candidate_spec['classifier_grid_hash'],
        'seed': RECOVERY_RANDOM_SEED,
        'candidate_summary': best_candidate,
    }


def _sorted_candidate_summaries(
    candidate_payloads: list[dict[str, Any]],
    *,
    tie_break_rule: str,
) -> list[dict[str, Any]]:
    from train import _candidate_sort_key

    candidates = [dict(payload['candidate_summary']) for payload in candidate_payloads]
    return sorted(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            tie_break_rule=tie_break_rule,
        ),
    )


def _build_outer_selected_refit_payload(
    *,
    context: dict[str, Any],
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
    candidate_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    from train import (
        _configure_classifier_for_resampling,
        _get_fit_kwargs,
        _json_safe,
        build_pipeline,
        get_classifier_configs,
    )

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
    groups_test = context['groups'][test_idx]

    ranked_candidates = _sorted_candidate_summaries(
        candidate_payloads,
        tie_break_rule=context['tie_break_rule'],
    )
    best_candidate = ranked_candidates[0]
    clf_config = get_classifier_configs()[clf_name]
    clf_variant = _configure_classifier_for_resampling(
        clf_name,
        clf_config['clf'],
        best_candidate['imbalance_strategy'],
    )
    pipeline = build_pipeline(
        clf_name,
        clf_variant,
        imbalance_strategy=best_candidate['imbalance_strategy'],
    )
    pipeline.set_params(**best_candidate['params'])
    fit_kwargs = _get_fit_kwargs(
        clf_name,
        y_train,
        best_candidate['imbalance_strategy'],
    )
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fold_model_path = (
        context['paths']['models_dir']
        / 'within_folds'
        / f'{context["condition"]}_{clf_name}_fold_{outer_fold_index:02d}_{outer_subject_id}.joblib'
    )
    fold_model_sha256 = _atomic_save_pipeline_artifact(pipeline, fold_model_path)

    return {
        'schema_version': OUTER_SELECTED_REFIT_SCHEMA_VERSION,
        'stage': 'outer_selected_refit',
        'condition': context['condition'],
        'classifier': clf_name,
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'candidate_strategy_policy': list(context['candidate_strategy_policy'][clf_name]),
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'recovery_runner_sha256': context['recovery_runner_sha256'],
        'train_module_sha256': context['train_module_sha256'],
        'v4_provenance_module_sha256': context['v4_provenance_module_sha256'],
        'recovery_execution_sha256': context['recovery_execution_sha256'],
        'classifier_grid_hash': candidate_payloads[0]['classifier_grid_hash'],
        'seed': RECOVERY_RANDOM_SEED,
        'selected_candidate_index': int(best_candidate['candidate_index']),
        'selected_candidate_hash': next(
            payload['candidate_hash']
            for payload in candidate_payloads
            if int(payload['candidate_index']) == int(best_candidate['candidate_index'])
        ),
        'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
        'selected_params': dict(best_candidate['params']),
        'selected_inner_subject_f1': float(best_candidate['inner_subject_f1']),
        'selected_inner_stride_f1': float(best_candidate['inner_stride_f1']),
        'selected_inner_subject_log_loss': (
            float(best_candidate['inner_subject_log_loss'])
            if best_candidate.get('inner_subject_log_loss') is not None else None
        ),
        'candidate_rankings_full': _json_safe(ranked_candidates),
        'public_outer_fold_detail': {
            'held_out_subject_id': str(groups_test[0]),
            'held_out_true_label': int(y_test[0]),
            'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
            'selected_params': dict(best_candidate['params']),
            'selected_inner_subject_f1': float(best_candidate['inner_subject_f1']),
            'selected_inner_stride_f1': float(best_candidate['inner_stride_f1']),
            'selected_inner_subject_log_loss': (
                float(best_candidate['inner_subject_log_loss'])
                if best_candidate.get('inner_subject_log_loss') is not None else None
            ),
            'candidate_rankings': _candidate_ranking_public(ranked_candidates),
            'fold_model_relpath': str(
                fold_model_path.relative_to(context['paths']['models_dir'])
            ),
            'fold_model_sha256': fold_model_sha256,
        },
        'y_true': np.asarray(y_test, dtype=int).tolist(),
        'y_pred': np.asarray(y_pred, dtype=int).tolist(),
        'y_prob': np.asarray(y_prob, dtype=float).tolist(),
        'subject_ids': np.asarray(groups_test).tolist(),
    }


def _build_full_source_selected_model_payload(
    *,
    context: dict[str, Any],
    clf_name: str,
    candidate_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    from train import (
        _configure_classifier_for_resampling,
        _get_fit_kwargs,
        _json_safe,
        build_pipeline,
        get_classifier_configs,
    )

    ranked_candidates = _sorted_candidate_summaries(
        candidate_payloads,
        tie_break_rule=context['tie_break_rule'],
    )
    best_candidate = ranked_candidates[0]
    clf_config = get_classifier_configs()[clf_name]
    clf_variant = _configure_classifier_for_resampling(
        clf_name,
        clf_config['clf'],
        best_candidate['imbalance_strategy'],
    )
    pipeline = build_pipeline(
        clf_name,
        clf_variant,
        imbalance_strategy=best_candidate['imbalance_strategy'],
    )
    pipeline.set_params(**best_candidate['params'])
    fit_kwargs = _get_fit_kwargs(
        clf_name,
        context['y'],
        best_candidate['imbalance_strategy'],
    )
    pipeline.fit(context['X'], context['y'], **fit_kwargs)

    full_source_model_path = (
        context['paths']['models_dir'] / f'{context["condition"]}_{clf_name}.joblib'
    )
    full_source_model_sha256 = _atomic_save_pipeline_artifact(
        pipeline,
        full_source_model_path,
    )

    return {
        'schema_version': FULL_SOURCE_SELECTED_MODEL_SCHEMA_VERSION,
        'stage': 'full_source_selected_model',
        'condition': context['condition'],
        'classifier': clf_name,
        'candidate_strategy_policy': list(context['candidate_strategy_policy'][clf_name]),
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'recovery_runner_sha256': context['recovery_runner_sha256'],
        'train_module_sha256': context['train_module_sha256'],
        'v4_provenance_module_sha256': context['v4_provenance_module_sha256'],
        'recovery_execution_sha256': context['recovery_execution_sha256'],
        'classifier_grid_hash': candidate_payloads[0]['classifier_grid_hash'],
        'seed': RECOVERY_RANDOM_SEED,
        'selected_candidate_index': int(best_candidate['candidate_index']),
        'selected_candidate_hash': next(
            payload['candidate_hash']
            for payload in candidate_payloads
            if int(payload['candidate_index']) == int(best_candidate['candidate_index'])
        ),
        'selected_imbalance_strategy': str(best_candidate['imbalance_strategy']),
        'selected_params': dict(best_candidate['params']),
        'selection_subject_f1': float(best_candidate['inner_subject_f1']),
        'selection_stride_f1': float(best_candidate['inner_stride_f1']),
        'selection_subject_log_loss': (
            float(best_candidate['inner_subject_log_loss'])
            if best_candidate.get('inner_subject_log_loss') is not None else None
        ),
        'selection_trace': _json_safe(ranked_candidates),
        'full_source_model_path': str(
            full_source_model_path.relative_to(context['paths']['models_dir'])
        ),
        'full_source_model_sha256': full_source_model_sha256,
    }


def _build_classifier_shard_payload(
    *,
    context: dict[str, Any],
    clf_name: str,
    outer_refit_payloads: list[dict[str, Any]],
    full_source_selected_payload: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import f1_score

    from train import (
        _build_within_condition_output,
        _json_safe,
        _save_selection_trace_sidecar,
        _strategy_to_legacy_label,
        _subject_level_metrics,
        _subject_primary_bootstrap_ci,
        _subject_resampled_stride_bootstrap_ci,
        get_modal_params,
        get_modal_strategy,
    )

    outer_refit_payloads = sorted(
        outer_refit_payloads,
        key=lambda payload: int(payload['outer_fold_index']),
    )
    y_true_all = np.concatenate([
        np.asarray(payload['y_true'], dtype=int)
        for payload in outer_refit_payloads
    ])
    y_pred_all = np.concatenate([
        np.asarray(payload['y_pred'], dtype=int)
        for payload in outer_refit_payloads
    ])
    y_prob_all = np.concatenate([
        np.asarray(payload['y_prob'], dtype=float)
        for payload in outer_refit_payloads
    ])
    subject_ids_all = np.concatenate([
        np.asarray(payload['subject_ids'])
        for payload in outer_refit_payloads
    ])
    fold_params = [
        dict(payload['selected_params']) for payload in outer_refit_payloads
    ]
    fold_best_scores = [
        float(payload['selected_inner_subject_f1'])
        for payload in outer_refit_payloads
    ]
    fold_best_strategies = [
        str(payload['selected_imbalance_strategy'])
        for payload in outer_refit_payloads
    ]
    outer_fold_details = [
        dict(payload['public_outer_fold_detail']) for payload in outer_refit_payloads
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

    context['paths']['selection_trace_dir'].mkdir(parents=True, exist_ok=True)
    for outer_fold_index, (fold_payload, fold_detail) in enumerate(zip(
        outer_refit_payloads,
        outer_fold_details,
    )):
        sidecar_path = context['paths']['selection_trace_dir'] / (
            f'{context["condition"]}_{clf_name}_fold_{outer_fold_index:02d}_candidate_trace.npz'
        )
        sidecar_metadata = {
            'kind': 'outer_fold_grouped_selection_trace',
            'condition': context['condition'],
            'classifier': clf_name,
            'outer_fold_index': outer_fold_index,
            'held_out_subject_id': fold_detail['held_out_subject_id'],
            'aggregation_rule': context['aggregation_rule'],
            'tie_break_rule': context['tie_break_rule'],
            'candidate_imbalance_strategies': list(
                context['candidate_strategy_policy'][clf_name]
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
        f'{context["condition"]}_{clf_name}_full_source_candidate_trace.npz'
    )
    full_trace_metadata = {
        'kind': 'full_source_grouped_selection_trace',
        'condition': context['condition'],
        'classifier': clf_name,
        'aggregation_rule': context['aggregation_rule'],
        'tie_break_rule': context['tie_break_rule'],
        'candidate_imbalance_strategies': list(
            context['candidate_strategy_policy'][clf_name]
        ),
    }
    full_trace_sha = _save_selection_trace_sidecar(
        sidecar_path=full_trace_path,
        candidates=full_source_selected_payload['selection_trace'],
        metadata=full_trace_metadata,
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
            full_source_selected_payload['selected_imbalance_strategy']
        ),
        'selected_imbalance_strategy': full_source_selected_payload[
            'selected_imbalance_strategy'
        ],
        'candidate_imbalance_strategies': list(
            context['candidate_strategy_policy'][clf_name]
        ),
        'outer_fold_selection_trace': _json_safe(outer_fold_details),
        'full_source_selected_params': _json_safe(
            full_source_selected_payload['selected_params']
        ),
        'full_source_selected_imbalance_strategy': full_source_selected_payload[
            'selected_imbalance_strategy'
        ],
        'full_source_selection_subject_f1': round(
            float(full_source_selected_payload['selection_subject_f1']),
            6,
        ),
        'full_source_selection_stride_f1': round(
            float(full_source_selected_payload['selection_stride_f1']),
            6,
        ),
        'full_source_selection_subject_log_loss': (
            round(float(full_source_selected_payload['selection_subject_log_loss']), 6)
            if full_source_selected_payload.get('selection_subject_log_loss') is not None
            else None
        ),
        'full_source_selection_trace': _json_safe(
            full_source_selected_payload['selection_trace']
        ),
        'full_source_selection_trace_path': str(full_trace_path),
        'full_source_selection_trace_sha256': full_trace_sha,
        'full_source_model_path': full_source_selected_payload['full_source_model_path'],
        'full_source_model_sha256': full_source_selected_payload['full_source_model_sha256'],
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
        clf_results={clf_name: classifier_result},
    )
    output['schema_version'] = CLASSIFIER_SHARD_SCHEMA_VERSION
    output['candidate_strategy_policy'] = {
        clf_name: list(context['candidate_strategy_policy'][clf_name])
    }
    output['subject_aggregation_rule'] = context['aggregation_rule']
    output['subject_probability_threshold'] = context['subject_probability_threshold']
    output['tie_break_rule'] = context['tie_break_rule']
    output['feature_matrix_hash'] = context['feature_matrix_hash']
    output['partition_hash'] = context['partition_hash']
    output['protocol_manifest_hash'] = context['protocol_manifest_hash']
    output['preprocessing_manifest_hash'] = context['preprocessing_manifest_hash']
    output['recovery_runner_sha256'] = context['recovery_runner_sha256']
    output['train_module_sha256'] = context['train_module_sha256']
    output['v4_provenance_module_sha256'] = context[
        'v4_provenance_module_sha256'
    ]
    output['recovery_execution_sha256'] = context['recovery_execution_sha256']
    return output


def _scientific_identity_digest(
    *,
    context: dict[str, Any],
    clf_name: str,
) -> str:
    from train import get_classifier_configs
    from v4_provenance import canonical_payload_sha256

    configs = get_classifier_configs()
    return canonical_payload_sha256({
        'condition': context['condition'],
        'classifier': clf_name,
        'aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'candidate_strategy_policy': list(context['candidate_strategy_policy'][clf_name]),
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'recovery_runner_sha256': context['recovery_runner_sha256'],
        'train_module_sha256': context['train_module_sha256'],
        'v4_provenance_module_sha256': context['v4_provenance_module_sha256'],
        'recovery_execution_sha256': context['recovery_execution_sha256'],
        'classifier_grid_hash': (
            _candidate_specs_for_target(context, clf_name)[0]['classifier_grid_hash']
        ),
        'seed': RECOVERY_RANDOM_SEED,
        'methodology_version': context['protocol_manifest']['methodology_version'],
        'schema_version': CLASSIFIER_SHARD_SCHEMA_VERSION,
        'param_grid': configs[clf_name]['param_grid'],
    })


@app.function(
    cpu=2,
    memory=2048,
    timeout=1800,
    max_containers=2,
    volumes={'/results': recovery_volume},
    retries=1,
)
def describe_recovery_target_context_remote(
    condition: str,
    clf_name: str,
) -> str:
    recovery_volume.reload()
    context = _load_validated_recovery_context(condition)
    candidate_specs = _candidate_specs_for_target(context, clf_name)
    return json.dumps({
        'condition': condition,
        'classifier': clf_name,
        'outer_subject_ids': context['outer_subject_ids'],
        'candidate_count': len(candidate_specs),
        'candidate_strategy_policy': list(context['candidate_strategy_policy'][clf_name]),
    }, indent=2)


@app.function(
    cpu=4,
    memory=4096,
    timeout=1800,
    max_containers=2,
    volumes={'/results': recovery_volume},
    retries=1,
)
def collect_recovery_status_remote(targets: str) -> str:
    recovery_volume.reload()
    parsed_targets = _parse_recovery_targets(targets)
    summary: dict[str, Any] = {
        'results_dir': str(_recovery_paths()['results_dir']),
        'models_dir': str(_recovery_paths()['models_dir']),
        'targets': [_normalize_target_key(*target) for target in parsed_targets],
        'per_target': {},
    }

    for condition, clf_name in parsed_targets:
        key = _normalize_target_key(condition, clf_name)
        try:
            context = _load_validated_recovery_context(condition)
        except Exception as exc:
            summary['per_target'][key] = {
                'context_status': 'invalid',
                'error': str(exc),
            }
            continue

        candidate_specs = _candidate_specs_for_target(context, clf_name)
        outer_candidate_statuses: dict[str, str] = {}
        outer_refit_statuses: dict[str, str] = {}
        for outer_fold_index, outer_subject_id in enumerate(context['outer_subject_ids']):
            for candidate_spec in candidate_specs:
                path = _outer_candidate_fragment_path(
                    context['paths'],
                    condition,
                    clf_name,
                    outer_fold_index,
                    outer_subject_id,
                    candidate_spec['candidate_index'],
                    candidate_spec['imbalance_strategy'],
                    candidate_spec['candidate_hash'],
                )
                artifact_key = (
                    f'fold_{outer_fold_index:02d}:{outer_subject_id}:'
                    f'candidate_{candidate_spec["candidate_index"]:03d}:'
                    f'{candidate_spec["imbalance_strategy"]}'
                )
                outer_candidate_statuses[artifact_key] = _status_of_artifact(
                    path=path,
                    validator=lambda payload, *, _context=context, _clf=clf_name, _spec=candidate_spec, _fold_idx=outer_fold_index, _subject=outer_subject_id: _validate_candidate_fragment_payload(
                        payload,
                        context=_context,
                        clf_name=_clf,
                        expected_stage='outer_candidate',
                        expected_schema_version=OUTER_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
                        candidate_spec=_spec,
                        outer_fold_index=_fold_idx,
                        outer_subject_id=_subject,
                    ),
                )

            refit_path = _outer_selected_refit_fragment_path(
                context['paths'],
                condition,
                clf_name,
                outer_fold_index,
                outer_subject_id,
            )
            outer_refit_statuses[f'fold_{outer_fold_index:02d}:{outer_subject_id}'] = _status_of_artifact(
                path=refit_path,
                validator=lambda payload, *, _context=context, _clf=clf_name, _fold_idx=outer_fold_index, _subject=outer_subject_id: _validate_outer_selected_refit_payload(
                    payload,
                    context=_context,
                    clf_name=_clf,
                    outer_fold_index=_fold_idx,
                    outer_subject_id=_subject,
                ),
            )

        full_source_candidate_statuses: dict[str, str] = {}
        for candidate_spec in candidate_specs:
            path = _full_source_candidate_fragment_path(
                context['paths'],
                condition,
                clf_name,
                candidate_spec['candidate_index'],
                candidate_spec['imbalance_strategy'],
                candidate_spec['candidate_hash'],
            )
            full_source_candidate_statuses[
                f'candidate_{candidate_spec["candidate_index"]:03d}:{candidate_spec["imbalance_strategy"]}'
            ] = _status_of_artifact(
                path=path,
                validator=lambda payload, *, _context=context, _clf=clf_name, _spec=candidate_spec: _validate_candidate_fragment_payload(
                    payload,
                    context=_context,
                    clf_name=_clf,
                    expected_stage='full_source_candidate',
                    expected_schema_version=FULL_SOURCE_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
                    candidate_spec=_spec,
                ),
            )

        full_source_selected_path = _full_source_selected_model_fragment_path(
            context['paths'],
            condition,
            clf_name,
        )
        final_shard_path = _classifier_shard_path(
            context['paths'],
            condition,
            clf_name,
        )

        summary['per_target'][key] = {
            'context_status': 'ok',
            'outer_candidate_fragments': _status_summary(outer_candidate_statuses),
            'outer_selected_refit_fragments': _status_summary(outer_refit_statuses),
            'full_source_candidate_fragments': _status_summary(full_source_candidate_statuses),
            'full_source_selected_model': _status_summary({
                'full_source_selected_model': _status_of_artifact(
                    path=full_source_selected_path,
                    validator=lambda payload, *, _context=context, _clf=clf_name: _validate_full_source_selected_model_payload(
                        payload,
                        context=_context,
                        clf_name=_clf,
                    ),
                )
            }),
            'final_classifier_shard': _status_summary({
                key: _status_of_artifact(
                    path=final_shard_path,
                    validator=lambda payload, *, _context=context, _clf=clf_name: _validate_classifier_shard_payload(
                        payload,
                        context=_context,
                        clf_name=_clf,
                    ),
                )
            }),
        }

    return json.dumps(summary, indent=2)


@app.function(
    cpu=1,
    memory=2048,
    timeout=21600,
    max_containers=24,
    volumes={'/results': recovery_volume},
    retries=1,
)
def run_outer_candidate_fragment_remote(
    condition: str,
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
    candidate_index: int,
    force_recompute_invalid: bool = False,
) -> str:
    recovery_volume.reload()
    context = _load_validated_recovery_context(condition)
    candidate_spec = _candidate_spec_by_index(context, clf_name, candidate_index)
    path = _outer_candidate_fragment_path(
        context['paths'],
        condition,
        clf_name,
        outer_fold_index,
        outer_subject_id,
        candidate_spec['candidate_index'],
        candidate_spec['imbalance_strategy'],
        candidate_spec['candidate_hash'],
    )
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_candidate_fragment_payload(
            payload,
            context=context,
            clf_name=clf_name,
            expected_stage='outer_candidate',
            expected_schema_version=OUTER_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
            candidate_spec=candidate_spec,
            outer_fold_index=outer_fold_index,
            outer_subject_id=outer_subject_id,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=(
            f'outer candidate fragment '
            f'{condition}:{clf_name}:{outer_fold_index}:{outer_subject_id}:{candidate_index}'
        ),
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'stage': 'outer_candidate',
            'condition': condition,
            'classifier': clf_name,
            'outer_fold_index': outer_fold_index,
            'held_out_subject_id': outer_subject_id,
            'candidate_index': candidate_index,
            'remote_fragment_path': str(path),
        }, indent=2)

    payload = _build_outer_candidate_fragment_payload(
        context=context,
        clf_name=clf_name,
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
        candidate_spec=candidate_spec,
    )
    written = _write_json_with_digest(path, payload)
    _validate_candidate_fragment_payload(
        written,
        context=context,
        clf_name=clf_name,
        expected_stage='outer_candidate',
        expected_schema_version=OUTER_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        candidate_spec=candidate_spec,
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
    )
    recovery_volume.commit()
    return json.dumps({
        'status': 'completed',
        'stage': 'outer_candidate',
        'condition': condition,
        'classifier': clf_name,
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'candidate_index': candidate_index,
        'remote_fragment_path': str(path),
    }, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=21600,
    max_containers=8,
    volumes={'/results': recovery_volume},
    retries=1,
)
def run_outer_selected_refit_fragment_remote(
    condition: str,
    clf_name: str,
    outer_fold_index: int,
    outer_subject_id: str,
    force_recompute_invalid: bool = False,
) -> str:
    recovery_volume.reload()
    context = _load_validated_recovery_context(condition)
    path = _outer_selected_refit_fragment_path(
        context['paths'],
        condition,
        clf_name,
        outer_fold_index,
        outer_subject_id,
    )
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_outer_selected_refit_payload(
            payload,
            context=context,
            clf_name=clf_name,
            outer_fold_index=outer_fold_index,
            outer_subject_id=outer_subject_id,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=(
            f'outer selected refit fragment '
            f'{condition}:{clf_name}:{outer_fold_index}:{outer_subject_id}'
        ),
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'stage': 'outer_selected_refit',
            'condition': condition,
            'classifier': clf_name,
            'outer_fold_index': outer_fold_index,
            'held_out_subject_id': outer_subject_id,
            'remote_fragment_path': str(path),
        }, indent=2)

    candidate_payloads: list[dict[str, Any]] = []
    for candidate_spec in _candidate_specs_for_target(context, clf_name):
        candidate_path = _outer_candidate_fragment_path(
            context['paths'],
            condition,
            clf_name,
            outer_fold_index,
            outer_subject_id,
            candidate_spec['candidate_index'],
            candidate_spec['imbalance_strategy'],
            candidate_spec['candidate_hash'],
        )
        candidate_payload = _load_reusable_payload(
            candidate_path,
            validator=lambda payload, *, _spec=candidate_spec: _validate_candidate_fragment_payload(
                payload,
                context=context,
                clf_name=clf_name,
                expected_stage='outer_candidate',
                expected_schema_version=OUTER_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
                candidate_spec=_spec,
                outer_fold_index=outer_fold_index,
                outer_subject_id=outer_subject_id,
            ),
            force_recompute_invalid=False,
            label=(
                f'outer candidate fragment '
                f'{condition}:{clf_name}:{outer_fold_index}:{outer_subject_id}:'
                f'{candidate_spec["candidate_index"]}'
            ),
        )
        if candidate_payload is None:
            raise FileNotFoundError(
                'Missing outer candidate fragment required for outer selected refit: '
                f'{candidate_path}'
            )
        candidate_payloads.append(candidate_payload)

    payload = _build_outer_selected_refit_payload(
        context=context,
        clf_name=clf_name,
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
        candidate_payloads=candidate_payloads,
    )
    written = _write_json_with_digest(path, payload)
    _validate_outer_selected_refit_payload(
        written,
        context=context,
        clf_name=clf_name,
        outer_fold_index=outer_fold_index,
        outer_subject_id=outer_subject_id,
    )
    recovery_volume.commit()
    return json.dumps({
        'status': 'completed',
        'stage': 'outer_selected_refit',
        'condition': condition,
        'classifier': clf_name,
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'remote_fragment_path': str(path),
    }, indent=2)


@app.function(
    cpu=1,
    memory=2048,
    timeout=21600,
    max_containers=24,
    volumes={'/results': recovery_volume},
    retries=1,
)
def run_full_source_candidate_fragment_remote(
    condition: str,
    clf_name: str,
    candidate_index: int,
    force_recompute_invalid: bool = False,
) -> str:
    recovery_volume.reload()
    context = _load_validated_recovery_context(condition)
    candidate_spec = _candidate_spec_by_index(context, clf_name, candidate_index)
    path = _full_source_candidate_fragment_path(
        context['paths'],
        condition,
        clf_name,
        candidate_spec['candidate_index'],
        candidate_spec['imbalance_strategy'],
        candidate_spec['candidate_hash'],
    )
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_candidate_fragment_payload(
            payload,
            context=context,
            clf_name=clf_name,
            expected_stage='full_source_candidate',
            expected_schema_version=FULL_SOURCE_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
            candidate_spec=candidate_spec,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=(
            f'full-source candidate fragment '
            f'{condition}:{clf_name}:{candidate_index}'
        ),
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'stage': 'full_source_candidate',
            'condition': condition,
            'classifier': clf_name,
            'candidate_index': candidate_index,
            'remote_fragment_path': str(path),
        }, indent=2)

    payload = _build_full_source_candidate_fragment_payload(
        context=context,
        clf_name=clf_name,
        candidate_spec=candidate_spec,
    )
    written = _write_json_with_digest(path, payload)
    _validate_candidate_fragment_payload(
        written,
        context=context,
        clf_name=clf_name,
        expected_stage='full_source_candidate',
        expected_schema_version=FULL_SOURCE_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        candidate_spec=candidate_spec,
    )
    recovery_volume.commit()
    return json.dumps({
        'status': 'completed',
        'stage': 'full_source_candidate',
        'condition': condition,
        'classifier': clf_name,
        'candidate_index': candidate_index,
        'remote_fragment_path': str(path),
    }, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=21600,
    max_containers=4,
    volumes={'/results': recovery_volume},
    retries=1,
)
def run_full_source_selected_model_remote(
    condition: str,
    clf_name: str,
    force_recompute_invalid: bool = False,
) -> str:
    recovery_volume.reload()
    context = _load_validated_recovery_context(condition)
    path = _full_source_selected_model_fragment_path(
        context['paths'],
        condition,
        clf_name,
    )
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_full_source_selected_model_payload(
            payload,
            context=context,
            clf_name=clf_name,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=f'full-source selected model fragment {condition}:{clf_name}',
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'stage': 'full_source_selected_model',
            'condition': condition,
            'classifier': clf_name,
            'remote_fragment_path': str(path),
        }, indent=2)

    candidate_payloads: list[dict[str, Any]] = []
    for candidate_spec in _candidate_specs_for_target(context, clf_name):
        candidate_path = _full_source_candidate_fragment_path(
            context['paths'],
            condition,
            clf_name,
            candidate_spec['candidate_index'],
            candidate_spec['imbalance_strategy'],
            candidate_spec['candidate_hash'],
        )
        candidate_payload = _load_reusable_payload(
            candidate_path,
            validator=lambda payload, *, _spec=candidate_spec: _validate_candidate_fragment_payload(
                payload,
                context=context,
                clf_name=clf_name,
                expected_stage='full_source_candidate',
                expected_schema_version=FULL_SOURCE_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
                candidate_spec=_spec,
            ),
            force_recompute_invalid=False,
            label=(
                f'full-source candidate fragment '
                f'{condition}:{clf_name}:{candidate_spec["candidate_index"]}'
            ),
        )
        if candidate_payload is None:
            raise FileNotFoundError(
                'Missing full-source candidate fragment required for selected-model fit: '
                f'{candidate_path}'
            )
        candidate_payloads.append(candidate_payload)

    payload = _build_full_source_selected_model_payload(
        context=context,
        clf_name=clf_name,
        candidate_payloads=candidate_payloads,
    )
    written = _write_json_with_digest(path, payload)
    _validate_full_source_selected_model_payload(
        written,
        context=context,
        clf_name=clf_name,
    )
    recovery_volume.commit()
    return json.dumps({
        'status': 'completed',
        'stage': 'full_source_selected_model',
        'condition': condition,
        'classifier': clf_name,
        'remote_fragment_path': str(path),
    }, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=21600,
    max_containers=4,
    volumes={'/results': recovery_volume},
    retries=1,
)
def assemble_recovery_classifier_shard_remote(
    condition: str,
    clf_name: str,
    force_recompute_invalid: bool = False,
) -> str:
    recovery_volume.reload()
    context = _load_validated_recovery_context(condition)
    path = _classifier_shard_path(context['paths'], condition, clf_name)
    existing = _load_reusable_payload(
        path,
        validator=lambda payload: _validate_classifier_shard_payload(
            payload,
            context=context,
            clf_name=clf_name,
        ),
        force_recompute_invalid=force_recompute_invalid,
        label=f'final recovery classifier shard {condition}:{clf_name}',
    )
    if existing is not None:
        return json.dumps({
            'status': 'skipped',
            'condition': condition,
            'classifier': clf_name,
            'remote_shard_path': str(path),
        }, indent=2)

    outer_refit_payloads: list[dict[str, Any]] = []
    for outer_fold_index, outer_subject_id in enumerate(context['outer_subject_ids']):
        refit_path = _outer_selected_refit_fragment_path(
            context['paths'],
            condition,
            clf_name,
            outer_fold_index,
            outer_subject_id,
        )
        payload = _load_reusable_payload(
            refit_path,
            validator=lambda candidate_payload, *, _fold_idx=outer_fold_index, _subject=outer_subject_id: _validate_outer_selected_refit_payload(
                candidate_payload,
                context=context,
                clf_name=clf_name,
                outer_fold_index=_fold_idx,
                outer_subject_id=_subject,
            ),
            force_recompute_invalid=False,
            label=f'outer selected refit fragment {condition}:{clf_name}:{outer_fold_index}:{outer_subject_id}',
        )
        if payload is None:
            raise FileNotFoundError(
                'Missing outer selected refit fragment required for assembly: '
                f'{refit_path}'
            )
        outer_refit_payloads.append(payload)

    full_source_selected_path = _full_source_selected_model_fragment_path(
        context['paths'],
        condition,
        clf_name,
    )
    full_source_selected_payload = _load_reusable_payload(
        full_source_selected_path,
        validator=lambda payload: _validate_full_source_selected_model_payload(
            payload,
            context=context,
            clf_name=clf_name,
        ),
        force_recompute_invalid=False,
        label=f'full-source selected model fragment {condition}:{clf_name}',
    )
    if full_source_selected_payload is None:
        raise FileNotFoundError(
            'Missing full-source selected model fragment required for assembly: '
            f'{full_source_selected_path}'
        )

    payload = _build_classifier_shard_payload(
        context=context,
        clf_name=clf_name,
        outer_refit_payloads=outer_refit_payloads,
        full_source_selected_payload=full_source_selected_payload,
    )
    written = _write_json_with_digest(path, payload)
    _validate_classifier_shard_payload(
        written,
        context=context,
        clf_name=clf_name,
    )
    recovery_volume.commit()
    return json.dumps({
        'status': 'completed',
        'condition': condition,
        'classifier': clf_name,
        'remote_shard_path': str(path),
    }, indent=2)


@app.function(
    cpu=2,
    memory=2048,
    timeout=1800,
    max_containers=1,
    volumes={'/results': recovery_volume},
    retries=1,
)
def write_recovery_promotion_manifest_remote(targets: str) -> str:
    from v4_provenance import sha256_file, canonical_payload_sha256

    recovery_volume.reload()
    parsed_targets = _parse_recovery_targets(targets)
    paths = _recovery_paths()
    manifest_path = _recovery_promotion_manifest_path(paths)
    promotable_targets: list[dict[str, Any]] = []
    incomplete_targets: list[dict[str, Any]] = []
    promotable_artifacts: list[dict[str, Any]] = []
    seen_artifact_keys: set[tuple[str, str, str]] = set()

    def add_promotable_artifact(
        *,
        target_key: str,
        kind: str,
        path: Path,
        sha256: str,
    ) -> None:
        relpath = _volume_relpath(path)
        artifact_key = (target_key, kind, relpath)
        if artifact_key in seen_artifact_keys:
            return
        seen_artifact_keys.add(artifact_key)
        promotable_artifacts.append({
            'target': target_key,
            'kind': kind,
            'path': relpath,
            'sha256': sha256,
        })

    for condition, clf_name in parsed_targets:
        context = _load_validated_recovery_context(condition)
        shard_path = _classifier_shard_path(context['paths'], condition, clf_name)
        payload = _load_reusable_payload(
            shard_path,
            validator=lambda candidate_payload: _validate_classifier_shard_payload(
                candidate_payload,
                context=context,
                clf_name=clf_name,
            ),
            force_recompute_invalid=False,
            label=f'final recovery classifier shard {condition}:{clf_name}',
        )
        if payload is None:
            incomplete_targets.append({
                'condition': condition,
                'classifier': clf_name,
                'completion_state': 'incomplete',
            })
            continue

        target_key = _normalize_target_key(condition, clf_name)
        classifier_payload = payload['classifiers'][clf_name]
        final_model_path = _resolve_stored_path(
            classifier_payload['full_source_model_path'],
            default_parent=context['paths']['models_dir'],
        )
        selection_trace_paths = [
            _resolve_stored_path(
                fold_detail['candidate_trace_relpath'],
                default_parent=context['paths']['results_dir'],
            )
            for fold_detail in classifier_payload['outer_fold_selection_trace']
        ]
        selection_trace_paths.append(
            _resolve_stored_path(
                classifier_payload['full_source_selection_trace_path'],
                default_parent=context['paths']['results_dir'],
            )
        )
        fold_model_paths = [
            _resolve_stored_path(
                fold_detail['fold_model_relpath'],
                default_parent=context['paths']['models_dir'],
            )
            for fold_detail in classifier_payload['outer_fold_selection_trace']
        ]
        files = [shard_path, final_model_path, *selection_trace_paths, *fold_model_paths]
        file_digests = {
            _volume_relpath(path): sha256_file(path)
            for path in files
        }
        add_promotable_artifact(
            target_key=target_key,
            kind='classifier_shard',
            path=shard_path,
            sha256=file_digests[_volume_relpath(shard_path)],
        )
        add_promotable_artifact(
            target_key=target_key,
            kind='full_source_model',
            path=final_model_path,
            sha256=file_digests[_volume_relpath(final_model_path)],
        )
        for fold_model_path in fold_model_paths:
            add_promotable_artifact(
                target_key=target_key,
                kind='outer_fold_model',
                path=fold_model_path,
                sha256=file_digests[_volume_relpath(fold_model_path)],
            )
        for trace_path in selection_trace_paths[:-1]:
            add_promotable_artifact(
                target_key=target_key,
                kind='outer_fold_candidate_trace',
                path=trace_path,
                sha256=file_digests[_volume_relpath(trace_path)],
            )
        add_promotable_artifact(
            target_key=target_key,
            kind='full_source_candidate_trace',
            path=selection_trace_paths[-1],
            sha256=file_digests[_volume_relpath(selection_trace_paths[-1])],
        )
        promotable_targets.append({
            'condition': condition,
            'classifier': clf_name,
            'completion_state': 'completed',
            'final_classifier_shard_path': _volume_relpath(shard_path),
            'final_model_path': _volume_relpath(final_model_path),
            'selection_trace_paths': [
                _volume_relpath(path) for path in selection_trace_paths
            ],
            'outer_fold_model_paths': [
                _volume_relpath(path) for path in fold_model_paths
            ],
            'model_metadata_sidecars': [],
            'file_sha256': file_digests,
            'protocol_manifest_sha256': context['protocol_manifest_hash'],
            'preprocessing_manifest_sha256': context['preprocessing_manifest_hash'],
            'recovery_runner_sha256': payload['recovery_runner_sha256'],
            'train_module_sha256': payload['train_module_sha256'],
            'v4_provenance_module_sha256': payload['v4_provenance_module_sha256'],
            'recovery_execution_sha256': payload['recovery_execution_sha256'],
            'scientific_identity_digest': _scientific_identity_digest(
                context=context,
                clf_name=clf_name,
            ),
        })

    sorted_promotable_artifacts = sorted(
        promotable_artifacts,
        key=lambda item: (item['target'], item['kind'], item['path']),
    )
    manifest_payload = {
        'schema_version': PROMOTION_MANIFEST_SCHEMA_VERSION,
        'results_namespace': DEFAULT_RESULTS_NAMESPACE,
        'models_namespace': DEFAULT_MODELS_NAMESPACE,
        'targets_requested': [_normalize_target_key(*target) for target in parsed_targets],
        'promotable_targets': promotable_targets,
        'promotable_artifacts': sorted_promotable_artifacts,
        'incomplete_targets': incomplete_targets,
        'promotion_digest': canonical_payload_sha256({
            'targets': promotable_targets,
            'artifacts': sorted_promotable_artifacts,
            'incomplete_targets': incomplete_targets,
        }),
    }
    written = _write_json_with_digest(manifest_path, manifest_payload)
    recovery_volume.commit()
    return json.dumps({
        'status': 'completed',
        'remote_manifest_path': str(manifest_path),
        'promotable_targets': [
            _normalize_target_key(target['condition'], target['classifier'])
            for target in promotable_targets
        ],
        'incomplete_targets': [
            _normalize_target_key(target['condition'], target['classifier'])
            for target in incomplete_targets
        ],
        'manifest_sha256': sha256_file(manifest_path),
        'promotion_digest': written['promotion_digest'],
    }, indent=2)


def _spawn_outer_candidate_batches(
    *,
    specs: list[tuple[str, str, int, str, int]],
    force_recompute_invalid: bool,
    submission_batch_size: int,
) -> None:
    if not specs:
        return
    for batch in _chunked(specs, submission_batch_size):
        run_outer_candidate_fragment_remote.spawn_map(
            [condition for condition, _, _, _, _ in batch],
            [clf_name for _, clf_name, _, _, _ in batch],
            [fold_idx for _, _, fold_idx, _, _ in batch],
            [subject_id for _, _, _, subject_id, _ in batch],
            [candidate_index for _, _, _, _, candidate_index in batch],
            kwargs={'force_recompute_invalid': force_recompute_invalid},
        )


def _spawn_outer_refit_batches(
    *,
    specs: list[tuple[str, str, int, str]],
    force_recompute_invalid: bool,
    submission_batch_size: int,
) -> None:
    if not specs:
        return
    for batch in _chunked(specs, submission_batch_size):
        run_outer_selected_refit_fragment_remote.spawn_map(
            [condition for condition, _, _, _ in batch],
            [clf_name for _, clf_name, _, _ in batch],
            [fold_idx for _, _, fold_idx, _ in batch],
            [subject_id for _, _, _, subject_id in batch],
            kwargs={'force_recompute_invalid': force_recompute_invalid},
        )


def _spawn_full_source_candidate_batches(
    *,
    specs: list[tuple[str, str, int]],
    force_recompute_invalid: bool,
    submission_batch_size: int,
) -> None:
    if not specs:
        return
    for batch in _chunked(specs, submission_batch_size):
        run_full_source_candidate_fragment_remote.spawn_map(
            [condition for condition, _, _ in batch],
            [clf_name for _, clf_name, _ in batch],
            [candidate_index for _, _, candidate_index in batch],
            kwargs={'force_recompute_invalid': force_recompute_invalid},
        )


def _spawn_full_source_selected_batches(
    *,
    specs: list[tuple[str, str]],
    force_recompute_invalid: bool,
    submission_batch_size: int,
) -> None:
    if not specs:
        return
    for batch in _chunked(specs, submission_batch_size):
        run_full_source_selected_model_remote.spawn_map(
            [condition for condition, _ in batch],
            [clf_name for _, clf_name in batch],
            kwargs={'force_recompute_invalid': force_recompute_invalid},
        )


@app.local_entrypoint()
def main(
    action: str = 'recovery-status',
    targets: str = '',
    submission_batch_size: int = DEFAULT_SUBMISSION_BATCH_SIZE,
    max_in_flight: int = 0,
    force_recompute_invalid: bool = False,
) -> None:
    if max_in_flight > 0:
        print(
            'Deprecated alias detected: --max-in-flight maps to '
            '--submission-batch-size for backward compatibility.',
            flush=True,
        )
        submission_batch_size = max_in_flight
    parsed_targets = _parse_recovery_targets(targets)

    if action == 'recovery-status':
        status = json.loads(
            collect_recovery_status_remote.remote(targets)
        )
        print(json.dumps(status, indent=2), flush=True)
        return

    if action == 'recovery-submit':
        status = json.loads(
            collect_recovery_status_remote.remote(targets)
        )
        outer_candidate_specs: list[tuple[str, str, int, str, int]] = []
        outer_refit_specs: list[tuple[str, str, int, str]] = []
        full_source_candidate_specs: list[tuple[str, str, int]] = []
        full_source_selected_specs: list[tuple[str, str]] = []

        for condition, clf_name in parsed_targets:
            target_key = _normalize_target_key(condition, clf_name)
            target_status = status['per_target'][target_key]
            if target_status['context_status'] != 'ok':
                raise SystemExit(
                    f'Recovery context for {target_key} is invalid: '
                    f'{target_status.get("error", "unknown error")}'
                )

            for stage_key in (
                'outer_candidate_fragments',
                'outer_selected_refit_fragments',
                'full_source_candidate_fragments',
                'full_source_selected_model',
                'final_classifier_shard',
            ):
                if (
                    target_status[stage_key]['invalid'] > 0
                    and not force_recompute_invalid
                ):
                    raise SystemExit(
                        f'Invalid persisted {stage_key} exist for {target_key}. '
                        'Rerun with --force-recompute-invalid to recompute them.'
                    )

            if target_status['final_classifier_shard']['completed'] == 1:
                print(
                    f'{target_key}: final classifier shard already completed on '
                    'the recovery volume.',
                    flush=True,
                )
                continue

            target_context = json.loads(
                describe_recovery_target_context_remote.remote(condition, clf_name)
            )
            outer_subject_ids = target_context['outer_subject_ids']
            candidate_count = int(target_context['candidate_count'])

            outer_candidate_stage = target_status['outer_candidate_fragments']
            if (
                outer_candidate_stage['completed'] < outer_candidate_stage['expected']
                or (force_recompute_invalid and outer_candidate_stage['invalid'] > 0)
            ):
                for outer_fold_index, outer_subject_id in enumerate(outer_subject_ids):
                    for candidate_index in range(candidate_count):
                        outer_candidate_specs.append(
                            (
                                condition,
                                clf_name,
                                outer_fold_index,
                                outer_subject_id,
                                candidate_index,
                            )
                        )
                continue

            outer_refit_stage = target_status['outer_selected_refit_fragments']
            if (
                outer_refit_stage['completed'] < outer_refit_stage['expected']
                or (force_recompute_invalid and outer_refit_stage['invalid'] > 0)
            ):
                for outer_fold_index, outer_subject_id in enumerate(outer_subject_ids):
                    outer_refit_specs.append(
                        (condition, clf_name, outer_fold_index, outer_subject_id)
                    )
                continue

            full_source_candidate_stage = target_status['full_source_candidate_fragments']
            if (
                full_source_candidate_stage['completed']
                < full_source_candidate_stage['expected']
                or (force_recompute_invalid and full_source_candidate_stage['invalid'] > 0)
            ):
                for candidate_index in range(candidate_count):
                    full_source_candidate_specs.append(
                        (condition, clf_name, candidate_index)
                    )
                continue

            full_source_selected_stage = target_status['full_source_selected_model']
            if (
                full_source_selected_stage['completed']
                < full_source_selected_stage['expected']
                or (force_recompute_invalid and full_source_selected_stage['invalid'] > 0)
            ):
                full_source_selected_specs.append((condition, clf_name))
                continue

            print(
                f'{target_key}: all recovery fragments are complete; run '
                '--action recovery-assemble-missing next.',
                flush=True,
            )

        print(
            'Submitting recovery fragments on gait-results-v4-recovery with '
            f'submission_batch_size={submission_batch_size}.',
            flush=True,
        )
        print(
            'submission_batch_size controls spawn_map queue-submission batches; '
            'actual concurrent worker count is enforced by each remote '
            'function max_containers setting.',
            flush=True,
        )
        print(
            'Do not launch recovery-submit again while the previous detached '
            'recovery App is active. Re-run recovery-status only after the '
            'current recovery App invocation drains.',
            flush=True,
        )
        _spawn_outer_candidate_batches(
            specs=outer_candidate_specs,
            force_recompute_invalid=force_recompute_invalid,
            submission_batch_size=submission_batch_size,
        )
        _spawn_outer_refit_batches(
            specs=outer_refit_specs,
            force_recompute_invalid=force_recompute_invalid,
            submission_batch_size=submission_batch_size,
        )
        _spawn_full_source_candidate_batches(
            specs=full_source_candidate_specs,
            force_recompute_invalid=force_recompute_invalid,
            submission_batch_size=submission_batch_size,
        )
        _spawn_full_source_selected_batches(
            specs=full_source_selected_specs,
            force_recompute_invalid=force_recompute_invalid,
            submission_batch_size=submission_batch_size,
        )
        print('Recovery submission complete.', flush=True)
        print('Monitor with:', flush=True)
        print('  modal app list', flush=True)
        print('  modal app logs gait-transfer-training-v4-recovery -f', flush=True)
        print(
            '  modal run scripts/training/run_within_condition_recovery_modal.py '
            f'--action recovery-status --targets "{targets}"',
            flush=True,
        )
        return

    if action == 'recovery-assemble-missing':
        print('Assembling completed recovery classifier shards...', flush=True)
        for condition, clf_name in parsed_targets:
            result = json.loads(
                assemble_recovery_classifier_shard_remote.remote(
                    condition,
                    clf_name,
                    force_recompute_invalid=force_recompute_invalid,
                )
            )
            print(json.dumps(result, indent=2), flush=True)
        return

    if action == 'recovery-promotion-manifest':
        result = json.loads(
            write_recovery_promotion_manifest_remote.remote(targets)
        )
        print(json.dumps(result, indent=2), flush=True)
        return

    raise SystemExit(
        f"Unknown action '{action}'. Expected one of "
        "{'recovery-status', 'recovery-submit', 'recovery-assemble-missing', "
        "'recovery-promotion-manifest'}."
    )
