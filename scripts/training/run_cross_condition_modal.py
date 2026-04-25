"""
Modal runner for zero-shot cross-condition transfer evaluation.

Runs all six transfer directions sequentially in a single Modal container:
    pd->hd, hd->pd, pd->als, als->pd, hd->als, als->hd

A single container (rather than six parallel containers) is used because the
dominant cost is model fitting (7 classifiers x 3 source conditions = 21 fits),
not per-direction evaluation. Parallelising would require re-fitting the same
source models multiple times across containers. Sequential execution in one
container allows model file caching: the pd_rf.joblib fitted for pd->hd is
reused for pd->als without refitting.

Modal allocation: cpu=16, memory=24576.

Usage (from repo root with venv active):
    modal run scripts/training/run_cross_condition_modal.py

Download results after completion:
    modal volume get gait-results cross_condition_results_v2.json
    modal volume get gait-results models_v2/pd_rf.joblib  # and all other model files
"""

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


# ── Single-container cross-condition function ─────────────────────────────────
@app.function(
    cpu=16,
    memory=24576,
    timeout=86400,     # 24-hour ceiling; all 21 fits expected in ~3-4 hours
    volumes={"/results": volume},
    retries=2,
)
def run_all_directions(
    gait_features_csv: bytes,
    control_partition_json: bytes,
    pd_results_json: bytes,
    hd_results_json: bytes,
    als_results_json: bytes,
) -> str:
    """
    Run all six cross-condition transfer directions in one container.

    Source models (21 total: 7 classifiers x 3 source conditions) are saved to
    /results/models_v2/{source}_{clf}.joblib after fitting. Subsequent
    directions sharing the same source condition load from disk rather than
    refitting, so each source model is fitted exactly once regardless of how
    many target directions share that source.

    Returns the accumulated results dict as a JSON string (also written to
    /results/cross_condition_results_v2.json on the Modal volume).
    """
    import json
    import os
    import tempfile
    import time

    import polars as pl

    from train import run_cross_condition

    # ── Write data files to temp location ────────────────────────────────────
    tmp = tempfile.mkdtemp()

    features_path = os.path.join(tmp, "gait_features.csv")
    with open(features_path, "wb") as f:
        f.write(gait_features_csv)

    partition_path = os.path.join(tmp, "control_partition.json")
    with open(partition_path, "wb") as f:
        f.write(control_partition_json)

    df = pl.read_csv(features_path)
    with open(partition_path) as f:
        partition = json.load(f)
    control_a = partition["control_A"]
    control_b = partition["control_B"]

    source_results = {
        "pd":  json.loads(pd_results_json),
        "hd":  json.loads(hd_results_json),
        "als": json.loads(als_results_json),
    }

    # Model files persist on the volume; run_cross_condition() loads from disk
    # if {source}_{clf}.joblib already exists, so pd_rf.joblib is fitted once
    # for pd->hd and reused for pd->als without any code change here.
    models_dir = "/results/models_v2"

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
            results_dir="/results",
            models_dir=models_dir,
            feature_matrix_file='v2/gait_features_v2.csv',
        )

        elapsed = time.time() - t_dir_start
        accumulated[direction_key] = result
        partial_path = "/results/cross_condition_results_v2_partial.json"
        with open(partial_path, "w") as f:
            json.dump(accumulated, f, indent=2)
        print(
            f"\nDirection {direction_key} complete in {elapsed:.0f}s", flush=True)
        print(flush=True)

    # ── Write single output file to Modal volume ──────────────────────────────
    out_path = "/results/cross_condition_results_v2.json"
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

    print("Reading data files...", flush=True)
    features_bytes = (
        repo_root / "data/processed/v2/gait_features_v2.csv").read_bytes()
    partition_bytes = (
        repo_root / "data/processed/control_partition.json").read_bytes()
    pd_bytes = (repo_root / "experiments/results/v2/pd_results_v2.json").read_bytes()
    hd_bytes = (repo_root / "experiments/results/v2/hd_results_v2.json").read_bytes()
    als_bytes = (
        repo_root / "experiments/results/v2/als_results_v2.json").read_bytes()

    print("Submitting cross-condition job to Modal...")
    print("Single container: 16 CPU, 24576 MB RAM.")
    print("21 source models will be fitted (7 classifiers x 3 source conditions).")
    print("Each source model is fitted once and cached for reuse across directions.")
    print()

    result_json = run_all_directions.remote(
        gait_features_csv=features_bytes,
        control_partition_json=partition_bytes,
        pd_results_json=pd_bytes,
        hd_results_json=hd_bytes,
        als_results_json=als_bytes,
    )

    print("\nJob complete. Download results:")
    print("  modal volume get gait-results cross_condition_results_v2.json")
    print("  # Model files (21 total):")
    for src in ["pd", "hd", "als"]:
        for clf in ["rf", "knn", "svm", "dt", "qda", "xgb", "lgbm"]:
            print(f"  modal volume get gait-results models_v2/{src}_{clf}.joblib")
