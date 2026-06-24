"""
Grouped-selector fit-count and timing profiler for the v4 launch gate.

Local mode:
    python scripts/verification/profile_v4_grouped_selector.py

Detached Modal mode:
    modal run --detach scripts/verification/profile_v4_grouped_selector.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover
    modal = None

def _is_dir_without_raising(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _infer_repo_and_src_roots() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()

    for candidate in (script_path.parent, *script_path.parents):
        src_root = candidate / 'src'
        if _is_dir_without_raising(src_root):
            return candidate, src_root

    modal_root = Path('/root')
    modal_src = modal_root / 'src'
    if _is_dir_without_raising(modal_src):
        return modal_root, modal_src

    raise RuntimeError(
        'Unable to locate src directory from '
        f'script_path={script_path}'
    )


REPO_ROOT, SRC_ROOT = _infer_repo_and_src_roots()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features import build_feature_matrix, get_feature_cols  # type: ignore
from train import (  # type: ignore
    _configure_classifier_for_resampling,
    _get_fit_kwargs,
    build_pipeline,
    candidate_strategies_for_classifier,
    get_classifier_configs,
)
from v4_provenance import atomic_write_json, ensure_v4_preflight_scaffold  # type: ignore

CONDITIONS = ('pd', 'hd', 'als')
OUTER_SUBJECTS = {'pd': 23, 'hd': 27, 'als': 21}
CLASSIFIER_SPECIFIC_POLICY = {
    **{
        clf_name: ('synthetic', 'balanced', 'raw')
        for clf_name in ('rf', 'svm', 'dt', 'xgb', 'lgbm')
    },
    'knn': ('synthetic', 'raw'),
    'qda': ('synthetic', 'raw'),
}
TIMING_REPRESENTATIVES = {
    'rf': ('pd', 'raw'),
    'knn': ('pd', 'raw'),
    'svm': ('pd', 'synthetic'),
    'dt': ('hd', 'raw'),
    'qda': ('hd', 'raw'),
    'xgb': ('hd', 'synthetic'),
    'lgbm': ('als', 'synthetic'),
}
REMOTE_OUTPUT_NAME = 'grouped_selector_profile.json'
REMOTE_GET_PATH = f'results_v4_preflight/{REMOTE_OUTPUT_NAME}'


def _reference_artifacts(
    *,
    processed_dir: Path,
    allow_build: bool,
) -> tuple[pl.DataFrame, dict[str, list[str]], dict[str, str]]:
    features_path = processed_dir / 'gait_features_v4_reference.csv'
    partition_path = processed_dir / 'control_partition_v4_reference.json'
    manifest_path = processed_dir / 'preprocessing_manifest_v4_reference.json'
    if not (features_path.exists() and partition_path.exists() and manifest_path.exists()):
        if not allow_build:
            raise FileNotFoundError(
                'Missing preflight Step 1 artifacts on the Modal volume. '
                'Run the detached Step 1 preflight diagnostic first.'
            )
        feature_df, partition = build_feature_matrix(
            processed_dir=processed_dir,
            output_filename=features_path.name,
            feature_cols=get_feature_cols('v4'),
            feature_set_version='v4',
            filter_strategy='v3',
            control_partition_version='v4',
            partition_output_filename=partition_path.name,
            manifest_filename=manifest_path.name,
            robust_mad_multiplier=3.0,
        )
        return feature_df, partition, {
            'features_path': str(features_path),
            'partition_path': str(partition_path),
            'manifest_path': str(manifest_path),
        }
    with open(partition_path) as f:
        partition = json.load(f)
    return pl.read_csv(features_path), partition, {
        'features_path': str(features_path),
        'partition_path': str(partition_path),
        'manifest_path': str(manifest_path),
    }


def _fit_count_breakdown_for_condition(n_subjects: int, candidate_count: int) -> dict[str, int]:
    outer_selection_fits = candidate_count * n_subjects * (n_subjects - 1)
    outer_selected_model_refits = n_subjects
    full_source_retuning_fits = candidate_count * n_subjects
    final_full_source_fit = 1
    return {
        'outer_selection_fits': outer_selection_fits,
        'outer_selected_model_refits': outer_selected_model_refits,
        'full_source_retuning_fits': full_source_retuning_fits,
        'final_full_source_fit': final_full_source_fit,
        'total_fits': (
            outer_selection_fits
            + outer_selected_model_refits
            + full_source_retuning_fits
            + final_full_source_fit
        ),
    }


def _candidate_count_summary() -> dict[str, Any]:
    configs = get_classifier_configs()
    per_classifier = {}
    for clf_name, config in configs.items():
        n_params = 1
        for values in config['param_grid'].values():
            n_params *= len(values)
        strategies = list(candidate_strategies_for_classifier(clf_name, CLASSIFIER_SPECIFIC_POLICY))
        n_candidates = n_params * len(strategies)
        per_classifier[clf_name] = {
            'n_param_configs': n_params,
            'strategies': strategies,
            'n_candidates': n_candidates,
            'fit_counts': {
                condition: _fit_count_breakdown_for_condition(n_subjects, n_candidates)
                for condition, n_subjects in OUTER_SUBJECTS.items()
            },
        }
    return per_classifier


def _measure_candidate_timing(
    *,
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
) -> dict[str, Any]:
    feature_cols = get_feature_cols('v4')
    control_a = partition['control_A']
    configs = get_classifier_configs()
    timings: dict[str, Any] = {}
    for clf_name, (condition, strategy) in TIMING_REPRESENTATIVES.items():
        pool = feature_df.filter(
            (pl.col('condition') == condition) | pl.col('subject_id').is_in(control_a)
        )
        X = pool.select(feature_cols).to_numpy().astype(np.float64)
        y = pool['label'].to_numpy().astype(int)
        groups = pool['subject_id'].to_numpy().astype(str)
        clf_template = clone(configs[clf_name]['clf'])
        params = {
            key: values[0]
            for key, values in configs[clf_name]['param_grid'].items()
        }
        start = time.perf_counter()
        inner = LeaveOneGroupOut()
        n_inner_fits = 0
        for train_idx, test_idx in inner.split(X, y, groups):
            del test_idx
            clf_variant = _configure_classifier_for_resampling(
                clf_name,
                clone(clf_template),
                strategy,
            )
            pipeline = build_pipeline(
                clf_name,
                clf_variant,
                imbalance_strategy=strategy,
            )
            pipeline.set_params(**params)
            fit_kwargs = _get_fit_kwargs(clf_name, y[train_idx], strategy)
            pipeline.fit(X[train_idx], y[train_idx], **fit_kwargs)
            n_inner_fits += 1
        elapsed = time.perf_counter() - start
        n_candidates = _candidate_count_summary()[clf_name]['n_candidates']
        projected_condition = OUTER_SUBJECTS[condition]
        projected_seconds = (projected_condition + 1) * n_candidates * elapsed
        if clf_name in {'svm'}:
            sharding = 'condition×classifier×outer_fold'
        elif clf_name in {'xgb', 'lgbm', 'rf'}:
            sharding = 'condition×classifier'
        else:
            sharding = 'condition×classifier'
        timings[clf_name] = {
            'representative_condition': condition,
            'representative_strategy': strategy,
            'measured_inner_grouped_candidate_seconds': round(float(elapsed), 6),
            'n_inner_fits_measured': int(n_inner_fits),
            'projected_condition_critical_path_seconds_estimate': round(float(projected_seconds), 2),
            'projection_basis': (
                'Estimate derived from one representative candidate and imbalance '
                'strategy, then scaled by the full candidate count for the '
                'representative condition.'
            ),
            'recommended_modal_sharding': sharding,
            'recommended_timeout_seconds': int(max(7200, round(projected_seconds * 2.5))),
        }
    return timings


def build_grouped_selector_profile(
    *,
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
) -> dict[str, Any]:
    candidate_counts = _candidate_count_summary()
    timings = _measure_candidate_timing(feature_df=feature_df, partition=partition)
    return {
        'outer_subjects': OUTER_SUBJECTS,
        'candidate_strategy_policy': CLASSIFIER_SPECIFIC_POLICY,
        'candidate_counts': candidate_counts,
        'timing_profile': timings,
        'projected_main_step2_critical_path_seconds': {
            clf_name: profile['projected_condition_critical_path_seconds_estimate']
            for clf_name, profile in timings.items()
        },
    }


def run_local() -> dict[str, Any]:
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    processed_dir = Path(scaffold['processed_preflight'])
    results_dir = Path(scaffold['results_preflight'])
    feature_df, partition, input_paths = _reference_artifacts(
        processed_dir=processed_dir,
        allow_build=True,
    )
    report = build_grouped_selector_profile(feature_df=feature_df, partition=partition)
    report['input_paths'] = input_paths
    output_path = results_dir / REMOTE_OUTPUT_NAME
    atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2))
    print(f'\nWrote {output_path}')
    return {'report_path': str(output_path), 'report': report}


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version='3.12')
        .pip_install_from_requirements('requirements-core.txt')
        .env({'PYTHONPATH': '/root/src'})
        .add_local_dir('src', remote_path='/root/src')
    )
    app = modal.App('gait-transfer-v4-preflight-selector-profile', image=image)
    volume = modal.Volume.from_name('gait-results', create_if_missing=True)

    @app.function(
        cpu=16,
        memory=16384,
        timeout=21600,
        volumes={'/results': volume},
        retries=1,
    )
    def run_grouped_selector_profile_remote() -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        results_dir = Path('/results/results_v4_preflight')
        results_dir.mkdir(parents=True, exist_ok=True)
        feature_df, partition, input_paths = _reference_artifacts(
            processed_dir=processed_dir,
            allow_build=False,
        )
        report = build_grouped_selector_profile(feature_df=feature_df, partition=partition)
        report['input_paths'] = input_paths
        output_path = results_dir / REMOTE_OUTPUT_NAME
        atomic_write_json(output_path, report)
        volume.commit()
        summary = {
            'remote_output_path': str(output_path),
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{REMOTE_GET_PATH} '
                f'experiments/results/v4_preflight/{REMOTE_OUTPUT_NAME}'
            ),
            'timed_classifiers': list(report['timing_profile']),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.local_entrypoint()
    def main() -> None:
        print('Submitting detached Modal grouped-selector profile.', flush=True)
        print('Required remote inputs:', flush=True)
        print('  gait-results:/processed_v4_preflight/gait_features_v4_reference.csv', flush=True)
        print('  gait-results:/processed_v4_preflight/control_partition_v4_reference.json', flush=True)
        print('Output JSON:', flush=True)
        print(f'  gait-results:/{REMOTE_GET_PATH}', flush=True)
        print('Download command:', flush=True)
        print(
            '  modal volume get gait-results '
            f'{REMOTE_GET_PATH} '
            f'experiments/results/v4_preflight/{REMOTE_OUTPUT_NAME}',
            flush=True,
        )
        run_grouped_selector_profile_remote.spawn()


if __name__ == '__main__':
    run_local()
