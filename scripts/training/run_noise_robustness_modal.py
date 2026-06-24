"""
Modal runner for Step 5: full noise robustness and sensitivity analysis.

Worker decomposition is preserved:
  - 3 within-condition robustness jobs
  - 6 cross-condition robustness jobs
  - 1 final aggregation pass

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


CONDITIONS = ('pd', 'hd', 'als')
DIRECTIONS = (
    ('pd', 'hd'),
    ('hd', 'pd'),
    ('pd', 'als'),
    ('als', 'pd'),
    ('hd', 'als'),
    ('als', 'hd'),
)
SECTION_NAMES = ('noise', 'featperm', 'corruption', 'conformal')
SECTION_SCHEMA_VERSION = 'v4-step5-section-v1'
SECTION_REPORTING_LABELS = {
    'noise': 'gaussian_feature_space_stress_test',
    'featperm': 'engineered_feature_permutation_sensitivity',
    'corruption': 'structured_corruption_suite',
    'conformal': 'exploratory_conformal_diagnostics',
}
FINAL_SCHEMA_VERSIONS = {
    'noise': 'v4-step5-noise-final-v1',
    'featperm': 'v4-step5-feature-sensitivity-final-v1',
    'subject': 'v4-step5-subject-sensitivity-final-v1',
    'corruption': 'v4-step5-corruption-final-v1',
    'conformal': 'v4-step5-conformal-final-v1',
}

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-noise-robustness', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _within_section_api_path(
    *,
    execution_id: str,
    condition: str,
    section: str,
) -> str:
    return (
        f'results_v4/downstream_runs/{execution_id}/step5/sections/'
        f'within_{condition}_{section}.json'
    )


def _cross_section_api_path(
    *,
    execution_id: str,
    source: str,
    target: str,
    section: str,
) -> str:
    return (
        f'results_v4/downstream_runs/{execution_id}/step5/sections/'
        f'cross_{source}_to_{target}_{section}.json'
    )


def _final_paths() -> dict[str, Path]:
    return {
        'noise': Path('/results/results_v4/noise_robustness_v4.json'),
        'featperm': Path('/results/results_v4/feature_sensitivity_v4.json'),
        'subject': Path('/results/results_v4/subject_sensitivity_v4.json'),
        'corruption': Path('/results/results_v4/corruption_robustness_v4.json'),
        'conformal': Path('/results/results_v4/conformal_v4.json'),
    }


def _validate_section_payload(
    *,
    payload: dict[str, Any],
    scope: str,
    logical_key: str,
    section: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    from v4_downstream import validate_payload_digest

    validate_payload_digest(payload)
    if payload.get('schema_version') != SECTION_SCHEMA_VERSION:
        raise ValueError('Unexpected Step 5 section schema version.')
    if payload.get('scope') != scope:
        raise ValueError('Step 5 scope mismatch.')
    if payload.get('logical_key') != logical_key:
        raise ValueError('Step 5 logical key mismatch.')
    if payload.get('section') != section:
        raise ValueError('Step 5 section mismatch.')
    if payload.get('protocol_manifest_hash') != context['protocol_manifest_sha256']:
        raise ValueError('Step 5 protocol hash mismatch.')
    if payload.get('preprocessing_manifest_hash') != context['preprocessing_manifest_sha256']:
        raise ValueError('Step 5 preprocessing hash mismatch.')
    if payload.get('feature_matrix_hash') != context['feature_matrix_sha256']:
        raise ValueError('Step 5 feature hash mismatch.')
    if payload.get('partition_hash') != context['partition_sha256']:
        raise ValueError('Step 5 partition hash mismatch.')
    if payload.get('downstream_execution_manifest_hash') != context['downstream_manifest_sha256']:
        raise ValueError('Step 5 downstream manifest hash mismatch.')
    if payload.get('downstream_execution_id') != context['downstream_execution_id']:
        raise ValueError('Step 5 downstream execution id mismatch.')
    return payload['section_payload']


def _validate_sigma_zero_against_authoritative(
    *,
    noise_payload: dict[str, Any],
    context: dict[str, Any],
    step3_payload: dict[str, Any],
) -> None:
    for condition in CONDITIONS:
        authoritative = context['step2_payloads'][condition]
        observed = noise_payload['within'][condition]
        for clf_name in observed:
            sigma_zero = float(observed[clf_name]['0.0'][0])
            expected = float(authoritative['classifiers'][clf_name]['f1_macro'])
            if abs(sigma_zero - expected) > 5e-7:
                raise ValueError(
                    f'Step 5 within sigma=0 mismatch for {condition}/{clf_name}: '
                    f'{sigma_zero} != {expected}'
                )
    for source, target in DIRECTIONS:
        direction_key = f'{source}_to_{target}'
        authoritative = step3_payload[direction_key]['classifiers']
        observed = noise_payload['cross'][direction_key]
        for scale_name, scale_payload in observed.items():
            for clf_name, sigma_payload in scale_payload.items():
                sigma_zero = float(sigma_payload['0.0'][0])
                expected = float(authoritative[clf_name]['f1_macro'])
                if abs(sigma_zero - expected) > 5e-7:
                    raise ValueError(
                        f'Step 5 cross sigma=0 mismatch for {direction_key}/{scale_name}/{clf_name}: '
                        f'{sigma_zero} != {expected}'
                    )


def _validate_final_output_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    from v4_downstream import validate_downstream_final_payload

    data = validate_downstream_final_payload(
        payload=payload,
        context=context,
        schema_version=FINAL_SCHEMA_VERSIONS[key],
    )
    if set(data) != {'within', 'cross'}:
        raise ValueError(f'Unexpected Step 5 final data keys for {key}.')
    return data


@app.function(
    cpu=16,
    memory=16384,
    timeout=86400,
    volumes={'/results': volume},
    retries=2,
)
def run_within_robustness(condition: str) -> str:
    import polars as pl

    import robustness as rb  # type: ignore[import-not-found]
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
            'scripts/training/run_noise_robustness_modal.py',
            'src/robustness.py',
        ),
        repo_root=None,
    )
    step3_payload = load_json(context['authoritative_results_dir'] / 'cross_condition_results_v4.json')
    validate_step3_results_payload(
        payload=step3_payload,
        output_namespace='results_v4',
        expected_directions=DIRECTIONS,
        context=context,
    )

    features_path = Path('/results/processed_v4/gait_features_v4.csv')
    partition_path = Path('/results/processed_v4/control_partition_v4.json')
    df = pl.read_csv(str(features_path))
    partition = _load_json(partition_path)
    control_a = partition['control_A']
    feature_cols = get_feature_cols('v4')
    within_results = context['step2_payloads'][condition]
    prefit_folds = rb.fit_within_condition_folds(
        condition,
        df,
        control_a,
        within_results,
        feature_cols=feature_cols,
        feature_matrix_hash=context['feature_matrix_sha256'],
        partition_hash=context['partition_sha256'],
        protocol_manifest_hash=context['protocol_manifest_sha256'],
        preprocessing_manifest_hash=context['preprocessing_manifest_sha256'],
    )
    baseline = {
        clf_name: float(within_results['classifiers'][clf_name]['f1_macro'])
        for clf_name in rb.CLF_ORDER
    }

    results: dict[str, str] = {}
    for section in SECTION_NAMES:
        section_path = api_path_to_mount_path(
            _within_section_api_path(
                execution_id=context['downstream_execution_id'],
                condition=condition,
                section=section,
            )
        )
        existing_valid = False
        if section_path.exists():
            try:
                existing_payload = _load_json(section_path)
                _validate_section_payload(
                    payload=existing_payload,
                    scope='within',
                    logical_key=condition,
                    section=section,
                    context=context,
                )
                existing_valid = True
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                existing_valid = False
        if existing_valid:
            results[section] = 'skipped_valid_existing'
            continue

        if section == 'noise':
            section_payload = rb.evaluate_noise_sweep_within(
                condition,
                df,
                control_a,
                within_results,
                feature_cols=feature_cols,
                prefit_folds=prefit_folds,
            )
        elif section == 'featperm':
            section_payload = rb.permutation_importance_within(
                condition,
                df,
                control_a,
                within_results,
                baseline,
                feature_cols=feature_cols,
                prefit_folds=prefit_folds,
            )
        elif section == 'corruption':
            section_payload = rb.evaluate_corruption_sweep_within(
                condition,
                df,
                control_a,
                within_results,
                feature_cols=feature_cols,
                prefit_folds=prefit_folds,
            )
        elif section == 'conformal':
            section_payload = rb.evaluate_conformal_within(
                condition,
                df,
                control_a,
                within_results,
                feature_cols=feature_cols,
                prefit_folds=prefit_folds,
            )
        else:
            raise ValueError(f'Unknown Step 5 section: {section}')

        payload = {
            'schema_version': SECTION_SCHEMA_VERSION,
            'scope': 'within',
            'logical_key': condition,
            'section': section,
            'reporting_label': SECTION_REPORTING_LABELS[section],
            'downstream_execution_id': context['downstream_execution_id'],
            'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
            'protocol_manifest_hash': context['protocol_manifest_sha256'],
            'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
            'feature_matrix_hash': context['feature_matrix_sha256'],
            'partition_hash': context['partition_sha256'],
            'section_payload': section_payload,
        }
        write_payload_json(section_path, payload)
        volume.commit()
        results[section] = 'completed'

    return json.dumps({'condition': condition, 'results': results}, indent=2)


@app.function(
    cpu=16,
    memory=16384,
    timeout=86400,
    volumes={'/results': volume},
    retries=2,
)
def run_cross_robustness(source: str, target: str) -> str:
    import joblib
    import polars as pl

    import robustness as rb  # type: ignore[import-not-found]
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
            'scripts/training/run_noise_robustness_modal.py',
            'src/robustness.py',
        ),
        repo_root=None,
    )
    step3_payload = load_json(context['authoritative_results_dir'] / 'cross_condition_results_v4.json')
    validate_step3_results_payload(
        payload=step3_payload,
        output_namespace='results_v4',
        expected_directions=DIRECTIONS,
        context=context,
    )

    features_path = Path('/results/processed_v4/gait_features_v4.csv')
    partition_path = Path('/results/processed_v4/control_partition_v4.json')
    df = pl.read_csv(str(features_path))
    partition = _load_json(partition_path)
    control_a = partition['control_A']
    control_b = partition['control_B']
    feature_cols = get_feature_cols('v4')
    direction_key = f'{source}_to_{target}'
    direction_results = step3_payload[direction_key]

    loaded_models: dict[str, object] = {}
    for clf_name in rb.CLF_ORDER:
        model_entry = context['step2_model_inventory'][f'{source}:{clf_name}']
        loaded_models[clf_name] = joblib.load(model_entry['resolved_path'])

    baseline = {
        clf_name: float(direction_results['classifiers'][clf_name]['f1_macro'])
        for clf_name in rb.CLF_ORDER
        if clf_name in direction_results['classifiers']
    }

    results: dict[str, str] = {}
    for section in SECTION_NAMES:
        section_path = api_path_to_mount_path(
            _cross_section_api_path(
                execution_id=context['downstream_execution_id'],
                source=source,
                target=target,
                section=section,
            )
        )
        existing_valid = False
        if section_path.exists():
            try:
                existing_payload = _load_json(section_path)
                _validate_section_payload(
                    payload=existing_payload,
                    scope='cross',
                    logical_key=direction_key,
                    section=section,
                    context=context,
                )
                existing_valid = True
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                existing_valid = False
        if existing_valid:
            results[section] = 'skipped_valid_existing'
            continue

        if section == 'noise':
            section_payload = rb.evaluate_noise_sweep_cross(
                source,
                target,
                df,
                control_a,
                control_b,
                context['authoritative_models_dir'],
                feature_cols=feature_cols,
                loaded_models=loaded_models,
            )
        elif section == 'featperm':
            section_payload = rb.permutation_importance_cross(
                source,
                target,
                df,
                control_a,
                control_b,
                context['authoritative_models_dir'],
                baseline,
                feature_cols=feature_cols,
                loaded_models=loaded_models,
            )
        elif section == 'corruption':
            section_payload = rb.evaluate_corruption_sweep_cross(
                source,
                target,
                df,
                control_a,
                control_b,
                context['authoritative_models_dir'],
                feature_cols=feature_cols,
                loaded_models=loaded_models,
            )
        elif section == 'conformal':
            section_payload = rb.evaluate_conformal_cross(
                source,
                target,
                df,
                control_a,
                control_b,
                context['authoritative_models_dir'],
                feature_cols=feature_cols,
                loaded_models=loaded_models,
            )
        else:
            raise ValueError(f'Unknown Step 5 section: {section}')

        payload = {
            'schema_version': SECTION_SCHEMA_VERSION,
            'scope': 'cross',
            'logical_key': direction_key,
            'section': section,
            'reporting_label': SECTION_REPORTING_LABELS[section],
            'downstream_execution_id': context['downstream_execution_id'],
            'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
            'protocol_manifest_hash': context['protocol_manifest_sha256'],
            'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
            'feature_matrix_hash': context['feature_matrix_sha256'],
            'partition_hash': context['partition_sha256'],
            'section_payload': section_payload,
        }
        write_payload_json(section_path, payload)
        volume.commit()
        results[section] = 'completed'

    return json.dumps({'direction_key': direction_key, 'results': results}, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=3600,
    volumes={'/results': volume},
    retries=1,
)
def submit_step5_remote() -> str:
    from v4_downstream import build_authoritative_step12_context

    volume.reload()
    context = build_authoritative_step12_context(
        volume_root='/results',
        require_downstream_manifest=True,
        required_downstream_files=('scripts/training/run_noise_robustness_modal.py',),
        repo_root=None,
    )
    submitted_within: list[str] = []
    for condition in CONDITIONS:
        needs_submit = False
        for section in SECTION_NAMES:
            path = Path('/results') / _within_section_api_path(
                execution_id=context['downstream_execution_id'],
                condition=condition,
                section=section,
            )
            if not path.exists():
                needs_submit = True
                break
            try:
                payload = _load_json(path)
                _validate_section_payload(
                    payload=payload,
                    scope='within',
                    logical_key=condition,
                    section=section,
                    context=context,
                )
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                needs_submit = True
                break
        if needs_submit:
            run_within_robustness.spawn(condition=condition)
            submitted_within.append(condition)

    submitted_cross: list[str] = []
    for source, target in DIRECTIONS:
        needs_submit = False
        for section in SECTION_NAMES:
            path = Path('/results') / _cross_section_api_path(
                execution_id=context['downstream_execution_id'],
                source=source,
                target=target,
                section=section,
            )
            if not path.exists():
                needs_submit = True
                break
            try:
                payload = _load_json(path)
                _validate_section_payload(
                    payload=payload,
                    scope='cross',
                    logical_key=f'{source}_to_{target}',
                    section=section,
                    context=context,
                )
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                needs_submit = True
                break
        if needs_submit:
            run_cross_robustness.spawn(source=source, target=target)
            submitted_cross.append(f'{source}_to_{target}')

    return json.dumps(
        {
            'status': 'submitted',
            'downstream_execution_id': context['downstream_execution_id'],
            'within_jobs': submitted_within,
            'cross_jobs': submitted_cross,
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
        required_downstream_files=('scripts/training/run_noise_robustness_modal.py',),
        repo_root=None,
    )
    status = {
        'within': {},
        'cross': {},
        'final_outputs': {},
    }
    for condition in CONDITIONS:
        per_section: dict[str, str] = {}
        for section in SECTION_NAMES:
            path = Path('/results') / _within_section_api_path(
                execution_id=context['downstream_execution_id'],
                condition=condition,
                section=section,
            )
            if not path.exists():
                per_section[section] = 'missing'
                continue
            try:
                payload = _load_json(path)
                _validate_section_payload(
                    payload=payload,
                    scope='within',
                    logical_key=condition,
                    section=section,
                    context=context,
                )
                per_section[section] = 'completed'
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                per_section[section] = 'invalid'
        status['within'][condition] = per_section
    for source, target in DIRECTIONS:
        direction_key = f'{source}_to_{target}'
        per_section = {}
        for section in SECTION_NAMES:
            path = Path('/results') / _cross_section_api_path(
                execution_id=context['downstream_execution_id'],
                source=source,
                target=target,
                section=section,
            )
            if not path.exists():
                per_section[section] = 'missing'
                continue
            try:
                payload = _load_json(path)
                _validate_section_payload(
                    payload=payload,
                    scope='cross',
                    logical_key=direction_key,
                    section=section,
                    context=context,
                )
                per_section[section] = 'completed'
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                per_section[section] = 'invalid'
        status['cross'][direction_key] = per_section
    for key, path in _final_paths().items():
        if not path.exists():
            status['final_outputs'][key] = 'missing'
            continue
        try:
            payload = _load_json(path)
            _validate_final_output_payload(
                payload=payload,
                context=context,
                key=key,
            )
            status['final_outputs'][key] = 'completed'
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            status['final_outputs'][key] = 'invalid'
    return json.dumps(status, indent=2)


@app.function(
    cpu=2,
    memory=4096,
    timeout=3600,
    volumes={'/results': volume},
    retries=1,
)
def assemble_step5_remote() -> str:
    import polars as pl

    import robustness as rb  # type: ignore[import-not-found]
    from features import get_feature_cols
    from v4_downstream import (
        DIRECTIONS,
        build_authoritative_step12_context,
        build_downstream_final_payload,
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
            'scripts/training/run_noise_robustness_modal.py',
            'src/robustness.py',
        ),
        repo_root=None,
    )
    step3_payload = load_json(context['authoritative_results_dir'] / 'cross_condition_results_v4.json')
    validate_step3_results_payload(
        payload=step3_payload,
        output_namespace='results_v4',
        expected_directions=DIRECTIONS,
        context=context,
    )

    noise_out: dict[str, Any] = {'within': {}, 'cross': {}}
    feat_out: dict[str, Any] = {'within': {}, 'cross': {}}
    corr_out: dict[str, Any] = {'within': {}, 'cross': {}}
    conf_out: dict[str, Any] = {'within': {}, 'cross': {}}

    for condition in CONDITIONS:
        for section, destination in (
            ('noise', noise_out['within']),
            ('featperm', feat_out['within']),
            ('corruption', corr_out['within']),
            ('conformal', conf_out['within']),
        ):
            path = Path('/results') / _within_section_api_path(
                execution_id=context['downstream_execution_id'],
                condition=condition,
                section=section,
            )
            if not path.exists():
                raise FileNotFoundError(f'Missing Step 5 within section: {path}')
            payload = _load_json(path)
            destination[condition] = _validate_section_payload(
                payload=payload,
                scope='within',
                logical_key=condition,
                section=section,
                context=context,
            )

    for source, target in DIRECTIONS:
        direction_key = f'{source}_to_{target}'
        for section, destination in (
            ('noise', noise_out['cross']),
            ('featperm', feat_out['cross']),
            ('corruption', corr_out['cross']),
            ('conformal', conf_out['cross']),
        ):
            path = Path('/results') / _cross_section_api_path(
                execution_id=context['downstream_execution_id'],
                source=source,
                target=target,
                section=section,
            )
            if not path.exists():
                raise FileNotFoundError(f'Missing Step 5 cross section: {path}')
            payload = _load_json(path)
            destination[direction_key] = _validate_section_payload(
                payload=payload,
                scope='cross',
                logical_key=direction_key,
                section=section,
                context=context,
            )

    _validate_sigma_zero_against_authoritative(
        noise_payload=noise_out,
        context=context,
        step3_payload=step3_payload,
    )

    feature_cols = get_feature_cols('v4')
    df = pl.read_csv(str(Path('/results/processed_v4/gait_features_v4.csv')))
    partition = _load_json(Path('/results/processed_v4/control_partition_v4.json'))
    subj_out = rb.build_subject_sensitivity_json(
        CONDITIONS,
        df,
        partition['control_A'],
        context['step2_payloads'],
        step3_payload,
        feature_cols=feature_cols,
    )

    final_paths = _final_paths()
    final_payloads = {
        'noise': build_downstream_final_payload(
            schema_version=FINAL_SCHEMA_VERSIONS['noise'],
            context=context,
            data=noise_out,
            extra_fields={
                'step': 'step5',
                'reporting_label': SECTION_REPORTING_LABELS['noise'],
            },
        ),
        'featperm': build_downstream_final_payload(
            schema_version=FINAL_SCHEMA_VERSIONS['featperm'],
            context=context,
            data=feat_out,
            extra_fields={
                'step': 'step5',
                'reporting_label': SECTION_REPORTING_LABELS['featperm'],
            },
        ),
        'subject': build_downstream_final_payload(
            schema_version=FINAL_SCHEMA_VERSIONS['subject'],
            context=context,
            data=subj_out,
            extra_fields={
                'step': 'step5',
                'reporting_label': 'subject_level_replay_sensitivity',
            },
        ),
        'corruption': build_downstream_final_payload(
            schema_version=FINAL_SCHEMA_VERSIONS['corruption'],
            context=context,
            data=corr_out,
            extra_fields={
                'step': 'step5',
                'reporting_label': SECTION_REPORTING_LABELS['corruption'],
            },
        ),
        'conformal': build_downstream_final_payload(
            schema_version=FINAL_SCHEMA_VERSIONS['conformal'],
            context=context,
            data=conf_out,
            extra_fields={
                'step': 'step5',
                'reporting_label': SECTION_REPORTING_LABELS['conformal'],
            },
        ),
    }
    for key, path in final_paths.items():
        write_payload_json(path, final_payloads[key])
    volume.commit()

    return json.dumps(
        {
            'status': 'completed',
            'downstream_execution_id': context['downstream_execution_id'],
            'paths': {key: str(path) for key, path in final_paths.items()},
        },
        indent=2,
    )


@app.local_entrypoint()
def main(action: str = 'submit') -> None:
    if action == 'submit':
        print(submit_step5_remote.remote(), flush=True)
        return
    if action == 'status':
        print(collect_status_remote.remote(), flush=True)
        return
    if action == 'assemble':
        print(assemble_step5_remote.remote(), flush=True)
        return
    raise SystemExit(
        f'Unknown action {action!r}. Expected one of: submit, status, assemble.'
    )
