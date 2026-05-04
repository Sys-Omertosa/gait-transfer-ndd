"""
Modal runner for publication-track v3 SHAP transfer diagnosis.

Runs three Modal containers in parallel, one per source condition. Each
container processes both target directions for that source serially, so the
within-condition SHAP explanation can be computed once and reused safely.

All prerequisites are read from the shared Modal volume:
  /results/processed_v3/gait_features_v3.csv
  /results/processed_v3/control_partition_v3.json
  /results/models_v3/*.joblib

Outputs are written to:
  /results/shap_v3/*.npz
  /results/results_v3/shap_results_v3.json

Usage:
    modal run scripts/training/run_shap_modal.py
"""

import io
import json
import time
from pathlib import Path

import modal

# ── Container image ───────────────────────────────────────────────────────────
# Replicates the image from run_within_condition_modal.py exactly.
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-transfer-shap', image=image)

# ── Persistent volume for results and .npz files ──────────────────────────────
volume = modal.Volume.from_name('gait-results', create_if_missing=True)


# ── Per-source SHAP function ──────────────────────────────────────────────────
@app.function(
    cpu=16,
    memory=20480,
    timeout=86400,
    volumes={'/results': volume},
    retries=2,
)
def run_source_group(
    source_condition: str,
) -> str:
    """
    Compute SHAP values and δj for both target directions of one source condition.
    """
    import json
    import os
    import time
    from pathlib import Path as _Path

    import polars as pl

    from explain import run_shap_for_direction
    from features import get_feature_cols

    processed_dir = _Path('/results/processed_v3')
    results_dir = _Path('/results/results_v3')
    features_path = processed_dir / 'gait_features_v3.csv'
    partition_path = processed_dir / 'control_partition_v3.json'
    models_dir = _Path('/results/models_v3')
    shap_dir = _Path('/results/shap_v3')
    results_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists() or not partition_path.exists():
        raise FileNotFoundError(
            'Missing Step 1 v3 artifacts on the Modal volume. '
            'Run scripts/training/run_preprocessing_modal.py first.'
        )
    if not models_dir.exists():
        raise FileNotFoundError(
            'Missing v3 model directory on the Modal volume. '
            'Run scripts/training/run_cross_condition_modal.py first.'
        )

    df = pl.read_csv(str(features_path))
    with open(partition_path) as f:
        partition = json.load(f)
    control_a: list[str] = partition['control_A']
    control_b: list[str] = partition['control_B']
    feature_cols = get_feature_cols('v3')

    os.makedirs(shap_dir, exist_ok=True)

    target_map = {
        'pd': ['hd', 'als'],
        'hd': ['pd', 'als'],
        'als': ['pd', 'hd'],
    }
    group_results: dict[str, dict] = {}

    print(f'Container starting: source={source_condition}', flush=True)
    t0 = time.time()

    for target_condition in target_map[source_condition]:
        direction_key = f'{source_condition}_to_{target_condition}'
        group_results[direction_key] = run_shap_for_direction(
            source_condition=source_condition,
            target_condition=target_condition,
            df=df,
            control_a=control_a,
            control_b=control_b,
            models_dir=models_dir,
            shap_dir=shap_dir,
            feature_cols=feature_cols,
            feature_set_version='v3',
            reuse_within=True,
            stability_n_resamples=200,
        )

    elapsed = time.time() - t0
    print(
        f'Container source={source_condition} complete in {elapsed:.0f}s',
        flush=True,
    )

    return json.dumps(group_results)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    """
    Submit all three source groups in parallel and collect results.

    Each source-group container computes two target directions serially so the
    shared within-condition SHAP explanation can be reused safely. Partial
    results are written to the Modal volume after each source group completes.
    """
    repo_root = Path(__file__).resolve().parents[2]
    local_results_dir = repo_root / 'experiments' / 'results' / 'v3'
    local_results_dir.mkdir(parents=True, exist_ok=True)

    source_conditions = ['pd', 'hd', 'als']

    print(
        f'Launching {len(source_conditions)} source groups in parallel on Modal...', flush=True)
    print('Each container: 16 CPU, 20480 MB RAM.', flush=True)
    print('Each source-group container reuses within-condition SHAP across 2 targets.', flush=True)
    print('Inputs are read from gait-results:/processed_v3 and gait-results:/models_v3.', flush=True)
    print()

    futures = {
        source_condition: run_source_group.spawn(source_condition=source_condition)
        for source_condition in source_conditions
    }

    accumulated: dict = {}
    t_start = time.time()

    pending = list(futures.items())
    while pending:
        for i, (source_condition, future) in enumerate(pending):
            try:
                group_results = json.loads(future.get(timeout=5))
                accumulated.update(group_results)
                elapsed_so_far = time.time() - t_start
                with volume.batch_upload(force=True) as batch:
                    batch.put_file(
                        io.BytesIO(json.dumps(accumulated, indent=2).encode()),
                        '/results/results_v3/shap_results_v3_partial.json',
                    )
                print(f'\n{"="*60}', flush=True)
                print(
                    f'Completed source group: {source_condition}  '
                    f'({elapsed_so_far:.0f}s elapsed, {len(accumulated)}/6 done)',
                    flush=True,
                )
                print(f'{"="*60}', flush=True)
                pending.pop(i)
                break
            except TimeoutError:
                continue
        else:
            time.sleep(10)

    out_volume_path = '/results/results_v3/shap_results_v3.json'
    with volume.batch_upload(force=True) as batch:
        batch.put_file(
            io.BytesIO(json.dumps(accumulated, indent=2).encode()),
            out_volume_path,
        )

    out_local_path = local_results_dir / 'shap_results_v3.json'
    with open(out_local_path, 'w') as f:
        json.dump(accumulated, f, indent=2)

    total_elapsed = time.time() - t_start
    print(
        f'\nAll 3 source groups (6 directions) complete in {total_elapsed:.0f}s',
        flush=True,
    )
    print(f'Results written locally to {out_local_path}', flush=True)
    print(f'Results written to Modal volume at {out_volume_path}', flush=True)
    print('\nDownload .npz files:', flush=True)
    print('  modal volume ls gait-results shap_v3/', flush=True)
