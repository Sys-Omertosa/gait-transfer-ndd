"""
Modal runner for the publication-track v3 zero-shot transfer benchmark.

Runs all six transfer directions sequentially in a single Modal container.
The container reads the Step 1 and Step 2 artifacts from the shared Modal
volume, writes model files to /results/models_v3/, and writes the combined
cross-condition JSON to /results/results_v3/.

Usage:
    modal run scripts/training/run_cross_condition_modal.py
"""

import json
import modal
from pathlib import Path

# ── Container image ───────────────────────────────────────────────────────────
# Replicates the image from run_within_condition_modal.py exactly.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements-core.txt")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gait-transfer-cross-condition", image=image)

# ── Persistent volume for results and model files ─────────────────────────────
volume = modal.Volume.from_name("gait-results", create_if_missing=True)


@app.function(
    cpu=16,
    memory=24576,
    timeout=86400,
    volumes={"/results": volume},
    retries=2,
)
def run_all_directions() -> str:
    """
    Run all six cross-condition transfer directions in one container.
    """
    import json
    import time
    from pathlib import Path as _Path

    import polars as pl

    from features import get_feature_cols
    from train import run_cross_condition

    processed_dir = _Path('/results/processed_v3')
    results_dir = _Path('/results/results_v3')
    models_dir = _Path('/results/models_v3')
    features_path = processed_dir / 'gait_features_v3.csv'
    partition_path = processed_dir / 'control_partition_v3.json'
    results_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        str(path)
        for path in (
            features_path,
            partition_path,
            results_dir / 'pd_results_v3.json',
            results_dir / 'hd_results_v3.json',
            results_dir / 'als_results_v3.json',
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            'Missing v3 prerequisites on the Modal volume. '
            'Run preprocessing and within-condition Modal steps first. '
            f'Missing: {missing}'
        )

    df = pl.read_csv(str(features_path))
    with open(partition_path) as f:
        partition = json.load(f)
    control_a = partition["control_A"]
    control_b = partition["control_B"]
    feature_cols = get_feature_cols('v3')

    source_results = {
        'pd': json.loads((results_dir / 'pd_results_v3.json').read_text()),
        'hd': json.loads((results_dir / 'hd_results_v3.json').read_text()),
        'als': json.loads((results_dir / 'als_results_v3.json').read_text()),
    }

    directions = [
        ("pd",  "hd"),
        ("hd",  "pd"),
        ("pd",  "als"),
        ("als", "pd"),
        ("hd",  "als"),
        ("als", "hd"),
    ]

    accumulated: dict = {}
    t_total_start = time.time()

    for source_cond, target_cond in directions:
        direction_key = f"{source_cond}_to_{target_cond}"
        print(f"{'='*60}", flush=True)
        print(
            f"Direction: {source_cond.upper()} -> {target_cond.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        t_dir_start = time.time()

        result = run_cross_condition(
            source_condition=source_cond,
            target_condition=target_cond,
            df=df,
            control_a=control_a,
            control_b=control_b,
            source_results=source_results[source_cond],
            results_dir=results_dir,
            models_dir=models_dir,
            feature_cols=feature_cols,
            feature_matrix_file='v3/gait_features_v3.csv',
            feature_set_version='v3',
            normalization='none',
        )

        elapsed = time.time() - t_dir_start
        accumulated[direction_key] = result
        partial_path = results_dir / 'cross_condition_results_v3_partial.json'
        with open(partial_path, "w") as f:
            json.dump(accumulated, f, indent=2)
        print(
            f"\nDirection {direction_key} complete in {elapsed:.0f}s", flush=True)
        print(flush=True)

    out_path = results_dir / 'cross_condition_results_v3.json'
    with open(out_path, "w") as f:
        json.dump(accumulated, f, indent=2)

    total_elapsed = time.time() - t_total_start
    print(f"All six directions complete in {total_elapsed:.0f}s", flush=True)
    print(f"Results written to Modal volume: {out_path}", flush=True)

    return json.dumps(accumulated, indent=2)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Submit the cross-condition job to Modal and stream output."""
    repo_root = Path(__file__).resolve().parents[2]
    local_results_dir = repo_root / 'experiments' / 'results' / 'v3'
    local_results_dir.mkdir(parents=True, exist_ok=True)

    print("Submitting cross-condition job to Modal...")
    print("Single container: 16 CPU, 24576 MB RAM.")
    print("Inputs are read from gait-results:/processed_v3 and gait-results:/results_v3.")
    print()

    result_json = run_all_directions.remote()
    result = json.loads(result_json)
    out_path = local_results_dir / 'cross_condition_results_v3.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print("\nJob complete. Download results:")
    print("  modal volume get gait-results results_v3/cross_condition_results_v3.json")
    print("  # Model files (21 total):")
    for src in ["pd", "hd", "als"]:
        for clf in ["rf", "knn", "svm", "dt", "qda", "xgb", "lgbm"]:
            print(f"  modal volume get gait-results models_v3/{src}_{clf}.joblib")
    print(f'Local copy written to {out_path}')
