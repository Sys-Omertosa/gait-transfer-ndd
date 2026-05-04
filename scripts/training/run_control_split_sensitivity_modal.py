"""
Modal control-split sensitivity runner for the publication-track v4 layer.

This script is not part of the main v3 rerun. It evaluates whether the
direction ordering and sign pattern of the main v3 transfer findings remain
stable across alternative near-optimal balanced control partitions.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal
import math

CONDITIONS = ['pd', 'hd', 'als']
DIRECTIONS = [
    ('pd', 'hd'),
    ('hd', 'pd'),
    ('pd', 'als'),
    ('als', 'pd'),
    ('hd', 'als'),
    ('als', 'hd'),
]
CLF_ORDER = ['rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm']
MAX_PARTITIONS = 3

VOLUME_FEATURES_PATH = '/results/processed_v3/gait_features_v3.csv'
VOLUME_CANDIDATES_PATH = '/results/processed_v3/control_partition_candidates_v3.json'
VOLUME_MAIN_PD_PATH = '/results/results_v3/pd_results_v3.json'
VOLUME_MAIN_HD_PATH = '/results/results_v3/hd_results_v3.json'
VOLUME_MAIN_ALS_PATH = '/results/results_v3/als_results_v3.json'
VOLUME_MAIN_CROSS_PATH = '/results/results_v3/cross_condition_results_v3.json'
VOLUME_RESULTS_ROOT = '/results/results_v3/control_split_sensitivity'
VOLUME_MODELS_ROOT = '/results/models_v3/control_split_sensitivity'

# Volume-API paths for use from the local entrypoint.
# volume.read_file() and volume.batch_upload() address paths
# relative to the volume root, not the /results/ mount point.
_API_CANDIDATES_PATH = 'processed_v3/control_partition_candidates_v3.json'
_API_MAIN_PD_PATH = 'results_v3/pd_results_v3.json'
_API_MAIN_HD_PATH = 'results_v3/hd_results_v3.json'
_API_MAIN_ALS_PATH = 'results_v3/als_results_v3.json'
_API_MAIN_CROSS_PATH = 'results_v3/cross_condition_results_v3.json'
_API_RESULTS_ROOT = 'results_v3/control_split_sensitivity'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-sensitivity', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)


def _partition_key(partition_index: int) -> str:
    return f'partition_{partition_index}'


def _partition_results_dir(partition_index: int) -> str:
    return f'{VOLUME_RESULTS_ROOT}/{_partition_key(partition_index)}'


def _partition_models_dir(partition_index: int) -> str:
    return f'{VOLUME_MODELS_ROOT}/{_partition_key(partition_index)}'


def _within_results_filename(condition: str, partition_index: int) -> str:
    return f'{condition}_results_v3_{_partition_key(partition_index)}.json'


def _cross_partial_filename(
    partition_index: int,
    source_condition: str,
    target_condition: str,
) -> str:
    return (
        f'{source_condition}_to_{target_condition}_'
        f'cross_results_v3_{_partition_key(partition_index)}.json'
    )


def _cross_combined_filename(partition_index: int) -> str:
    return f'cross_condition_results_v3_{_partition_key(partition_index)}.json'


def _read_volume_bytes(path: str) -> bytes:
    return b''.join(volume.read_file(path))


def _read_volume_json(path: str) -> dict:
    return json.loads(_read_volume_bytes(path).decode())


def _write_volume_json(path: str, payload: dict) -> None:
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(payload, indent=2).encode()), path)


def _volume_path_exists(path: str) -> bool:
    try:
        first_chunk = next(iter(volume.read_file(path)))
        return len(first_chunk) >= 0
    except (FileNotFoundError, StopIteration):
        return False


def _best_within_f1(within_result: dict) -> float:
    return max(
        float(clf_out['f1_macro'])
        for clf_out in within_result['classifiers'].values()
    )


def _direction_summary(
    *,
    within_best_f1_by_source: dict[str, float],
    cross_results: dict[str, dict],
) -> tuple[dict[str, dict], list[float], list[str]]:
    directions_summary: dict[str, dict] = {}
    mean_delta_vector: list[float] = []
    sign_vector: list[str] = []

    for source_condition, target_condition in DIRECTIONS:
        direction_key = f'{source_condition}_to_{target_condition}'
        direction_out = cross_results[direction_key]
        per_classifier_f1 = {
            clf_name: float(direction_out['classifiers'][clf_name]['f1_macro'])
            for clf_name in CLF_ORDER
        }
        mean_cross_f1 = sum(per_classifier_f1.values()) / len(CLF_ORDER)
        mean_delta_f1 = float(within_best_f1_by_source[source_condition] - mean_cross_f1)
        direction_sign = '+' if mean_delta_f1 > 0 else '-'

        directions_summary[direction_key] = {
            'mean_delta_f1': round(mean_delta_f1, 6),
            'direction_sign': direction_sign,
            'per_classifier_f1': {
                clf_name: round(f1_val, 6)
                for clf_name, f1_val in per_classifier_f1.items()
            },
        }
        mean_delta_vector.append(mean_delta_f1)
        sign_vector.append(direction_sign)

    return directions_summary, mean_delta_vector, sign_vector


@app.function(
    cpu=16,
    memory=12288,
    timeout=86400,
    volumes={'/results': volume},
    retries=1,
)
def run_partition_condition(
    partition_index: int,
    condition: str,
    control_a: list[str],
    control_b: list[str],
) -> str:
    import json
    from pathlib import Path as _Path

    import polars as pl

    from features import get_feature_cols
    from train import run_within_condition

    del control_b

    features_path = _Path(VOLUME_FEATURES_PATH)
    if not features_path.exists():
        raise FileNotFoundError(
            'Missing v3 feature matrix on the Modal volume. '
            'Run scripts/training/run_preprocessing_modal.py first.'
        )

    results_dir = _Path(_partition_results_dir(partition_index))
    existing_path = results_dir / _within_results_filename(
        condition, partition_index
    )
    if existing_path.exists():
        print(
            f'  [{condition} p{partition_index}] within-condition '
            f'results already exist, skipping recomputation.',
            flush=True,
        )
        existing = json.loads(existing_path.read_text())
        selected_arms = {
            clf_name: clf_out['selected_imbalance_strategy']
            for clf_name, clf_out in existing['classifiers'].items()
        }
        return json.dumps({
            'partition_index': partition_index,
            'condition': condition,
            'selected_arms': selected_arms,
            'output_path': str(existing_path),
        })

    df = pl.read_csv(str(features_path))
    feature_cols = get_feature_cols('v3')
    results_dir.mkdir(parents=True, exist_ok=True)

    output = run_within_condition(
        condition=condition,
        df=df,
        control_subjects=control_a,
        results_dir=results_dir,
        feature_cols=feature_cols,
        feature_matrix_file='v3/gait_features_v3.csv',
        feature_set_version='v3',
        normalization='none',
        results_filename=_within_results_filename(condition, partition_index),
        imbalance_arms=('synthetic', 'balanced', 'raw'),
        selection_arms=('synthetic', 'balanced'),
    )

    selected_arms = {
        clf_name: clf_out['selected_imbalance_strategy']
        for clf_name, clf_out in output['classifiers'].items()
    }
    return json.dumps({
        'partition_index': partition_index,
        'condition': condition,
        'selected_arms': selected_arms,
        'output_path': str(results_dir / _within_results_filename(condition, partition_index)),
    })


@app.function(
    cpu=16,
    memory=24576,
    timeout=86400,
    volumes={'/results': volume},
    retries=2,
)
def run_partition_direction(
    partition_index: int,
    source_condition: str,
    target_condition: str,
    control_a: list[str],
    control_b: list[str],
) -> str:
    import json
    import time as _time
    from pathlib import Path as _Path

    import polars as pl

    from features import get_feature_cols
    from train import run_cross_condition

    features_path = _Path(VOLUME_FEATURES_PATH)
    if not features_path.exists():
        raise FileNotFoundError(
            'Missing v3 feature matrix on the Modal volume. '
            'Run scripts/training/run_preprocessing_modal.py first.'
        )

    results_dir = _Path(_partition_results_dir(partition_index))
    models_dir = _Path(_partition_models_dir(partition_index))
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    source_within_path = results_dir / _within_results_filename(
        source_condition, partition_index
    )
    max_wait_seconds = 300
    poll_interval = 10
    waited = 0
    while not source_within_path.exists() and waited < max_wait_seconds:
        print(
            f'  [{source_condition}->{target_condition} p{partition_index}] '
            f'waiting for within-condition results '
            f'({waited}s elapsed)...',
            flush=True,
        )
        _time.sleep(poll_interval)
        waited += poll_interval
    if not source_within_path.exists():
        raise FileNotFoundError(
            f'Missing within-condition results for {source_condition} in '
            f'{_partition_key(partition_index)} after waiting '
            f'{max_wait_seconds}s. Check the within-condition job logs.'
        )

    df = pl.read_csv(str(features_path))
    source_results = json.loads(source_within_path.read_text())
    feature_cols = get_feature_cols('v3')

    output = run_cross_condition(
        source_condition=source_condition,
        target_condition=target_condition,
        df=df,
        control_a=control_a,
        control_b=control_b,
        source_results=source_results,
        results_dir=results_dir,
        models_dir=models_dir,
        feature_cols=feature_cols,
        feature_matrix_file='v3/gait_features_v3.csv',
        feature_set_version='v3',
        normalization='none',
    )

    partial_path = results_dir / _cross_partial_filename(
        partition_index,
        source_condition,
        target_condition,
    )
    with open(partial_path, 'w') as f:
        json.dump(output, f, indent=2)

    return json.dumps({
        'partition_index': partition_index,
        'direction_key': f'{source_condition}_to_{target_condition}',
        'output_path': str(partial_path),
        'result': output,
    })


@app.local_entrypoint()
def main() -> None:
    from scipy.stats import spearmanr

    repo_root = Path(__file__).resolve().parents[2]
    local_summary_dir = repo_root / 'experiments' / 'results' / 'v3' / 'control_split_sensitivity'
    local_summary_dir.mkdir(parents=True, exist_ok=True)

    candidates_all = _read_volume_json(_API_CANDIDATES_PATH)
    candidates = candidates_all[:MAX_PARTITIONS]
    if not candidates:
        raise RuntimeError(
            'No control partition candidates found on the Modal volume. '
            'Run scripts/training/run_preprocessing_modal.py first.'
        )

    missing_main_paths = [
        path for path in (
            _API_MAIN_PD_PATH,
            _API_MAIN_HD_PATH,
            _API_MAIN_ALS_PATH,
            _API_MAIN_CROSS_PATH,
        )
        if not _volume_path_exists(path)
    ]
    if missing_main_paths:
        raise FileNotFoundError(
            'Missing authoritative v3 result artifacts required for stability '
            f'comparison: {missing_main_paths}'
        )

    print('\n' + '=' * 88)
    print('Control-Split Sensitivity Candidates')
    print('=' * 88)
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f'partition_{idx} | score={candidate["score"]:.6f} | '
            f'age_delta={candidate["age_delta_years"]:.3f} | '
            f'speed_delta={candidate["gait_speed_delta_m_per_s"]:.6f}',
            flush=True,
        )
        print(f'  control_A={candidate["control_A"]}', flush=True)
        print(f'  control_B={candidate["control_B"]}', flush=True)
    print('=' * 88 + '\n')

    within_futures = {
        (idx, condition): run_partition_condition.spawn(
            partition_index=idx,
            condition=condition,
            control_a=candidate['control_A'],
            control_b=candidate['control_B'],
        )
        for idx, candidate in enumerate(candidates, start=1)
        for condition in CONDITIONS
    }

    print(f'Launching {len(within_futures)} within-condition sensitivity jobs...', flush=True)
    pending_within = list(within_futures.items())
    while pending_within:
        for i, ((partition_index, condition), future) in enumerate(pending_within):
            try:
                result = json.loads(future.get(timeout=5))
                print(
                    f'[within] {condition.upper()} complete for '
                    f'{_partition_key(partition_index)}',
                    flush=True,
                )
                print(f'  selected arms: {result["selected_arms"]}', flush=True)
                pending_within.pop(i)
                break
            except TimeoutError:
                continue

    print('\nAll within-condition sensitivity jobs complete.\n', flush=True)

    cross_futures = {
        (idx, source_condition, target_condition): run_partition_direction.spawn(
            partition_index=idx,
            source_condition=source_condition,
            target_condition=target_condition,
            control_a=candidate['control_A'],
            control_b=candidate['control_B'],
        )
        for idx, candidate in enumerate(candidates, start=1)
        for source_condition, target_condition in DIRECTIONS
    }

    print(f'Launching {len(cross_futures)} cross-condition sensitivity jobs...', flush=True)
    cross_results_by_partition: dict[int, dict[str, dict]] = {
        idx: {}
        for idx in range(1, len(candidates) + 1)
    }
    pending_cross = list(cross_futures.items())
    while pending_cross:
        for i, ((partition_index, source_condition, target_condition), future) in enumerate(pending_cross):
            try:
                result = json.loads(future.get(timeout=5))
                direction_key = result['direction_key']
                cross_results_by_partition[partition_index][direction_key] = result['result']
                print(
                    f'[cross] {direction_key} complete for '
                    f'{_partition_key(partition_index)}',
                    flush=True,
                )
                pending_cross.pop(i)
                break
            except TimeoutError:
                continue

    print('\nAll cross-condition sensitivity jobs complete.\n', flush=True)

    for partition_index, direction_map in cross_results_by_partition.items():
        if len(direction_map) != len(DIRECTIONS):
            raise RuntimeError(
                f'Partition {_partition_key(partition_index)} completed with '
                f'{len(direction_map)} directions instead of {len(DIRECTIONS)}.'
            )
        combined_path = (
            f'{_API_RESULTS_ROOT}/{_partition_key(partition_index)}/'
            f'{_cross_combined_filename(partition_index)}'
        )
        _write_volume_json(combined_path, direction_map)

    main_within_by_condition = {
        'pd': _read_volume_json(_API_MAIN_PD_PATH),
        'hd': _read_volume_json(_API_MAIN_HD_PATH),
        'als': _read_volume_json(_API_MAIN_ALS_PATH),
    }
    main_cross = _read_volume_json(_API_MAIN_CROSS_PATH)
    main_within_best = {
        condition: _best_within_f1(within_result)
        for condition, within_result in main_within_by_condition.items()
    }
    _, main_delta_vector, main_sign_vector = _direction_summary(
        within_best_f1_by_source=main_within_best,
        cross_results=main_cross,
    )

    partitions_summary: list[dict] = []
    for idx, candidate in enumerate(candidates, start=1):
        partition_results_dir_api = (
            f'{_API_RESULTS_ROOT}/{_partition_key(idx)}'
        )
        within_by_condition = {
            condition: _read_volume_json(
                f'{partition_results_dir_api}/{_within_results_filename(condition, idx)}'
            )
            for condition in CONDITIONS
        }
        partition_cross = _read_volume_json(
            f'{partition_results_dir_api}/{_cross_combined_filename(idx)}'
        )
        partition_within_best = {
            condition: _best_within_f1(within_result)
            for condition, within_result in within_by_condition.items()
        }
        direction_summary, partition_delta_vector, partition_sign_vector = _direction_summary(
            within_best_f1_by_source=partition_within_best,
            cross_results=partition_cross,
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
            'partition_index': idx,
            'control_A': candidate['control_A'],
            'control_B': candidate['control_B'],
            'age_delta_years': candidate['age_delta_years'],
            'gait_speed_delta_m_per_s': candidate['gait_speed_delta_m_per_s'],
            'score': candidate['score'],
            'directions': direction_summary,
            'stability_check': {
                'n_directions_same_sign': same_sign_count,
                'ordering_rank_correlation': (
                    None if rho is None else round(rho, 6)
                ),
            },
        })

    summary = {
        'max_partitions_requested': MAX_PARTITIONS,
        'n_partitions_evaluated': len(candidates),
        'main_authoritative': {
            'within_best_f1_by_source': {
                condition: round(f1_val, 6)
                for condition, f1_val in main_within_best.items()
            },
            'direction_mean_delta_f1': {
                f'{source_condition}_to_{target_condition}': round(main_delta_vector[idx], 6)
                for idx, (source_condition, target_condition) in enumerate(DIRECTIONS)
            },
            'direction_signs': {
                f'{source_condition}_to_{target_condition}': main_sign_vector[idx]
                for idx, (source_condition, target_condition) in enumerate(DIRECTIONS)
            },
        },
        'partitions': partitions_summary,
    }

    volume_summary_path = f'{_API_RESULTS_ROOT}/sensitivity_summary_v3.json'
    _write_volume_json(volume_summary_path, summary)

    local_summary_path = local_summary_dir / 'sensitivity_summary_v3.json'
    with open(local_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    sign_header = ' | '.join(
        f'{source}->{target}'
        for source, target in DIRECTIONS
    )
    print('\n' + '=' * 88)
    print('Sensitivity Sign Summary')
    print('=' * 88)
    print(f'partition | {sign_header} | same_sign', flush=True)
    for partition_summary in partitions_summary:
        signs = ' | '.join(
            partition_summary['directions'][f'{source}_to_{target}']['direction_sign']
            for source, target in DIRECTIONS
        )
        same_sign = partition_summary['stability_check']['n_directions_same_sign']
        print(
            f'{_partition_key(partition_summary["partition_index"]):>10} | '
            f'{signs} | {same_sign}/6',
            flush=True,
        )
    print('=' * 88)
    print(f'Volume summary: {volume_summary_path}', flush=True)
    print(f'Local summary : {local_summary_path}', flush=True)
