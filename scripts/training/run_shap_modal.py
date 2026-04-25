"""
Modal runner for SHAP transfer-failure diagnosis.

Runs three Modal containers in parallel — one per source condition. Each
container processes both target directions for that source serially, so the
within-condition SHAP explanation can be computed once and reused safely.

Parallelization rationale:
  KernelExplainer classifiers (SVM, QDA, KNN) dominate cost. Grouping by
  source condition eliminates duplicate within-condition SHAP work that would
  otherwise be recomputed by two separate containers sharing the same source.

Within-condition .npz caching:
  reuse_within=True is safe here because each source condition is handled by
  exactly one container. The shared within-condition .npz files are written
  once per source/classifier and then reused for the second target direction.

Modal allocation: cpu=16, memory=20480 per container. KernelExplainer is
parallelised inside src/explain.py across CPU workers, so cpu=16 remains a
reasonable balance between wall time and Modal credit consumption.

Usage (from repo root with venv active):
    modal run scripts/training/run_shap_modal.py

Download results after completion:
    modal volume get gait-results shap_results_v2.json
    # .npz files in shap_v2/ subdirectory (gitignored locally):
    modal volume get gait-results shap_v2/pd_rf_within.npz  # example
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
    gait_features_csv: bytes,
    control_partition_json: bytes,
    source_condition: str,
) -> str:
    """
    Compute SHAP values and δj for both target directions of one source condition.

    Loads the feature matrix and control partition from bytes (passed in by the
    local entrypoint), constructs pools, calls run_shap_for_direction() twice,
    and writes .npz files to /results/shap_v2/ on the Modal volume.

    Returns a JSON string mapping both direction keys for the given source
    condition to their per-classifier SHAP summaries.
    """
    import json
    import os
    import tempfile
    import time

    import polars as pl

    from explain import run_shap_for_direction

    # Write data files to a temp directory accessible within the container.
    tmp = tempfile.mkdtemp()
    features_path = os.path.join(tmp, 'gait_features.csv')
    partition_path = os.path.join(tmp, 'control_partition.json')

    with open(features_path, 'wb') as f:
        f.write(gait_features_csv)
    with open(partition_path, 'wb') as f:
        f.write(control_partition_json)

    df = pl.read_csv(features_path)
    with open(partition_path) as f:
        partition = json.load(f)
    control_a: list[str] = partition['control_A']
    control_b: list[str] = partition['control_B']

    # The current authoritative v2 Modal volume keeps model files at this path.
    models_dir = '/results/models_v2'

    # .npz files are written to /results/shap_v2/ on the volume.
    shap_dir = '/results/shap_v2'
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
            reuse_within=True,
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

    print('Reading data files...', flush=True)
    features_bytes = (
        repo_root / 'data/processed/v2/gait_features_v2.csv').read_bytes()
    partition_bytes = (
        repo_root / 'data/processed/control_partition.json').read_bytes()

    source_conditions = ['pd', 'hd', 'als']

    print(
        f'Launching {len(source_conditions)} source groups in parallel on Modal...', flush=True)
    print('Each container: 16 CPU, 20480 MB RAM.', flush=True)
    print('Each source-group container reuses within-condition SHAP across 2 targets.', flush=True)
    print()

    futures = {
        source_condition: run_source_group.spawn(
            gait_features_csv=features_bytes,
            control_partition_json=partition_bytes,
            source_condition=source_condition,
        )
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
                        '/results/shap_results_v2_partial.json',
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

    # Write shap_results_v2.json to volume.
    out_volume_path = '/results/shap_results_v2.json'
    with volume.batch_upload(force=True) as batch:
        batch.put_file(
            io.BytesIO(json.dumps(accumulated, indent=2).encode()),
            out_volume_path,
        )

    # Write shap_results_v2.json locally.
    out_local_path = repo_root / 'experiments/results/v2/shap_results_v2.json'
    out_local_path.parent.mkdir(parents=True, exist_ok=True)
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
    print('  modal volume ls gait-results shap_v2/', flush=True)
