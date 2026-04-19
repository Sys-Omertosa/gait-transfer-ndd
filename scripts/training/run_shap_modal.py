"""
Modal runner for SHAP transfer-failure diagnosis.

Runs all six transfer directions in parallel — one Modal container per
direction. Each container calls run_shap_for_direction() for its assigned
(source, target) pair, computing SHAP values for all 7 classifiers and
writing .npz files to the Modal volume at /results/shap/.

Parallelization rationale:
  KernelExplainer classifiers (SVM, QDA, KNN) dominate cost at ~60+ minutes
  each per direction. Each direction runs its 7 classifiers serially within
  its container (KernelExplainer is single-threaded). With 6 parallel
  containers the total wall time equals the slowest single direction rather
  than the sum of all directions.

Within-condition .npz caching:
  Each container computes within-condition SHAP independently (reuse_within=False
  is passed to run_shap_for_direction). This deliberately avoids the race condition
  that would arise if two containers sharing the same source condition (e.g., PD as
  source for both pd->hd and pd->als) simultaneously tried to write and read the
  same within-condition file from the shared Modal volume. The within-condition
  computation is negligible relative to KernelExplainer cost (~minutes for tree
  classifiers, tens of minutes for kernel classifiers), so the duplication is
  acceptable. The local sequential runner (run_shap_local.py) retains reuse_within=True
  since it processes directions one at a time with no concurrency risk.

Modal allocation: cpu=16, memory=16384 per container (fixed for all Modal
runs in this project). KernelExplainer is single-threaded (Python loop) so
cpu=16 primarily benefits the sklearn distance computations inside KNN's
predict_proba calls.

Usage (from repo root with venv active):
    modal run scripts/training/run_shap_modal.py

Download results after completion:
    modal volume get gait-results shap_results.json
    # .npz files in shap/ subdirectory (gitignored locally):
    modal volume get gait-results shap/pd_rf_within.npz  # example
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


# ── Per-direction SHAP function ───────────────────────────────────────────────
@app.function(
    cpu=16,
    memory=16384,
    timeout=86400,      # 24-hour ceiling; each direction expected ~2-4 hours
    volumes={'/results': volume},
)
def run_direction(
    gait_features_csv: bytes,
    control_partition_json: bytes,
    source_condition: str,
    target_condition: str,
) -> str:
    """
    Compute SHAP values and δj for all 7 classifiers for one transfer direction.

    Loads the feature matrix and control partition from bytes (passed in by the
    local entrypoint), constructs pools, calls run_shap_for_direction(), and
    writes .npz files to /results/shap/ on the Modal volume.

    Returns the per-classifier δj results as a JSON string for accumulation
    in the local entrypoint.
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

    # Model files were written to the volume root by run_cross_condition_modal.py.
    models_dir = '/results'

    # .npz files are written to /results/shap/ on the volume.
    shap_dir = '/results/shap'
    os.makedirs(shap_dir, exist_ok=True)

    direction_key = f'{source_condition}_to_{target_condition}'
    print(f'Container starting: {direction_key}', flush=True)
    t0 = time.time()

    result = run_shap_for_direction(
        source_condition=source_condition,
        target_condition=target_condition,
        df=df,
        control_a=control_a,
        control_b=control_b,
        models_dir=models_dir,
        shap_dir=shap_dir,
        reuse_within=False,   # each container computes within-condition independently
                              # to eliminate the race condition that would arise if two
                              # containers sharing the same source tried to share a file
    )

    elapsed = time.time() - t0
    print(f'Container {direction_key} complete in {elapsed:.0f}s', flush=True)

    return json.dumps(result)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    """
    Submit all six directions in parallel and collect results.

    All six containers are spawned simultaneously via .spawn(). The local
    process then collects each result as it completes. After all six finish,
    the accumulated dict is written to shap_results.json both on the Modal
    volume and locally at experiments/results/shap_results.json.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent

    print('Reading data files...', flush=True)
    features_bytes = (repo_root / 'data/processed/gait_features.csv').read_bytes()
    partition_bytes = (repo_root / 'data/processed/control_partition.json').read_bytes()

    directions = [
        ('pd',  'hd'),
        ('hd',  'pd'),
        ('pd',  'als'),
        ('als', 'pd'),
        ('hd',  'als'),
        ('als', 'hd'),
    ]

    print(f'Launching {len(directions)} directions in parallel on Modal...', flush=True)
    print('Each container: 16 physical CPU cores, 16 GB RAM.', flush=True)
    print('KernelExplainer classifiers (SVM, QDA, KNN) dominate per-container time.', flush=True)
    print('Wall time ≈ slowest single direction (not sum of all).', flush=True)
    print()

    # Spawn all six simultaneously without blocking.
    futures = {
        f'{src}_to_{tgt}': run_direction.spawn(
            gait_features_csv=features_bytes,
            control_partition_json=partition_bytes,
            source_condition=src,
            target_condition=tgt,
        )
        for src, tgt in directions
    }

    accumulated: dict = {}
    t_start = time.time()

    # Poll all futures until all six are done. Each .get(timeout=5) either returns
    # the result (container finished) or raises a TimeoutError (still running).
    # This lets us print each direction as it completes rather than waiting for all
    # six to finish before printing any — important for a multi-hour run where the
    # terminal would otherwise appear stuck.
    pending = list(futures.items())
    while pending:
        for i, (direction_key, future) in enumerate(pending):
            try:
                result = json.loads(future.get(timeout=5))
                accumulated[direction_key] = result
                elapsed_so_far = time.time() - t_start
                print(f'\n{"="*60}', flush=True)
                print(
                    f'Completed: {direction_key}  '
                    f'({elapsed_so_far:.0f}s elapsed, {len(accumulated)}/6 done)',
                    flush=True,
                )
                print(f'{"="*60}', flush=True)
                pending.pop(i)
                break  # restart the outer while loop immediately
            except TimeoutError:
                continue
        else:
            # All futures in this sweep were still running — sleep before next sweep
            time.sleep(10)

    # Write shap_results.json to volume.
    out_volume_path = '/results/shap_results.json'
    with volume.batch_upload() as batch:
        batch.put_file(
            io.BytesIO(json.dumps(accumulated, indent=2).encode()),
            out_volume_path,
        )

    # Write shap_results.json locally.
    out_local_path = repo_root / 'experiments/results/shap_results.json'
    out_local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_local_path, 'w') as f:
        json.dump(accumulated, f, indent=2)

    total_elapsed = time.time() - t_start
    print(f'\nAll six directions complete in {total_elapsed:.0f}s', flush=True)
    print(f'Results written locally to {out_local_path}', flush=True)
    print(f'Results written to Modal volume at {out_volume_path}', flush=True)
    print('\nDownload .npz files:', flush=True)
    print('  modal volume ls gait-results shap/', flush=True)
