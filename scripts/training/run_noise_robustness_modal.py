"""
Modal runner for Step 5: full noise robustness and sensitivity analysis.

Runs the full (non-reduced) sweep defined in src/robustness.py:
  - SIGMA_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
  - N_NOISE_REPEATS = 30
  - all 7 classifiers

Writes to Modal volume (/results):
  - noise_robustness.json
  - feature_sensitivity.json
  - subject_sensitivity.json

Usage (from repo root with venv active):
    modal run scripts/training/run_noise_robustness_modal.py
"""

from __future__ import annotations

import modal
from pathlib import Path


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements-core.txt")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gait-transfer-noise-robustness", image=image)
volume = modal.Volume.from_name("gait-results", create_if_missing=True)


@app.function(
    cpu=16,
    memory=16384,
    timeout=86400,
    volumes={"/results": volume},
)
def run_step5_full(
    gait_features_csv: bytes,
    control_partition_json: bytes,
    pd_results_json: bytes,
    hd_results_json: bytes,
    als_results_json: bytes,
    cross_condition_results_json: bytes,
) -> str:
    import json
    import tempfile
    import time
    from pathlib import Path as _Path

    import polars as pl

    import robustness as rb  # type: ignore[import-not-found]

    t0 = time.time()
    tmp_dir = _Path(tempfile.mkdtemp())

    features_path = tmp_dir / "gait_features.csv"
    partition_path = tmp_dir / "control_partition.json"
    with open(features_path, "wb") as f:
        f.write(gait_features_csv)
    with open(partition_path, "wb") as f:
        f.write(control_partition_json)

    df = pl.read_csv(features_path)
    with open(partition_path) as f:
        partition = json.load(f)
    control_a = partition["control_A"]
    control_b = partition["control_B"]

    within_by: dict[str, dict] = {
        "pd": json.loads(pd_results_json),
        "hd": json.loads(hd_results_json),
        "als": json.loads(als_results_json),
    }
    cross_results: dict = json.loads(cross_condition_results_json)

    results_dir = _Path("/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Step 5 full run started (Modal).", flush=True)
    print(f"SIGMA_LEVELS={rb.SIGMA_LEVELS}, repeats={rb.N_NOISE_REPEATS}", flush=True)

    # Noise sweeps
    noise_out: dict = {"within": {}, "cross": {}}
    print("Noise sweeps (within-condition)...", flush=True)
    for cond in ("pd", "hd", "als"):
        print(f"  within {cond.upper()} ...", flush=True)
        noise_out["within"][cond] = rb.evaluate_noise_sweep_within(
            cond, df, control_a, within_by[cond]
        )

    print("Noise sweeps (cross-condition)...", flush=True)
    for direction_key in cross_results:
        src, tgt = rb.direction_key_to_pair(direction_key)
        print(f"  cross {direction_key} ...", flush=True)
        noise_out["cross"][direction_key] = rb.evaluate_noise_sweep_cross(
            src, tgt, df, control_a, control_b, results_dir
        )

    noise_path = results_dir / "noise_robustness.json"
    with open(noise_path, "w") as f:
        json.dump(noise_out, f, indent=2)
    print(f"Wrote {noise_path}", flush=True)

    # Feature permutation sensitivity
    feat_out: dict = {"within": {}, "cross": {}}
    print("Feature permutation sensitivity (within-condition)...", flush=True)
    for cond in ("pd", "hd", "als"):
        baseline = {
            c: float(within_by[cond]["classifiers"][c]["f1_macro"])
            for c in rb.CLF_ORDER
        }
        print(f"  within {cond.upper()} ...", flush=True)
        feat_out["within"][cond] = rb.permutation_importance_within(
            cond, df, control_a, within_by[cond], baseline
        )

    print("Feature permutation sensitivity (cross-condition)...", flush=True)
    for direction_key, dr in cross_results.items():
        src, tgt = rb.direction_key_to_pair(direction_key)
        baseline = {
            c: float(dr["classifiers"][c]["f1_macro"])
            for c in rb.CLF_ORDER
            if c in dr["classifiers"]
        }
        print(f"  cross {direction_key} ...", flush=True)
        feat_out["cross"][direction_key] = rb.permutation_importance_cross(
            src, tgt, df, control_a, control_b, results_dir, baseline
        )

    feat_path = results_dir / "feature_sensitivity.json"
    with open(feat_path, "w") as f:
        json.dump(feat_out, f, indent=2)
    print(f"Wrote {feat_path}", flush=True)

    # Per-subject sensitivity from stored predictions
    print("Per-subject sensitivity...", flush=True)
    subj_out = rb.build_subject_sensitivity_json(
        ("pd", "hd", "als"), df, control_a, within_by, cross_results
    )
    subj_path = results_dir / "subject_sensitivity.json"
    with open(subj_path, "w") as f:
        json.dump(subj_out, f, indent=2)
    print(f"Wrote {subj_path}", flush=True)

    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed:.0f}s", flush=True)

    summary = {
        "noise_robustness_path": str(noise_path),
        "feature_sensitivity_path": str(feat_path),
        "subject_sensitivity_path": str(subj_path),
        "elapsed_seconds": round(elapsed, 2),
        "sigma_levels": list(rb.SIGMA_LEVELS),
        "repeats": rb.N_NOISE_REPEATS,
    }
    return json.dumps(summary, indent=2)


@app.local_entrypoint()
def main():
    repo_root = Path(__file__).resolve().parent.parent.parent

    print("Reading Step 5 inputs from local repository...", flush=True)
    features_bytes = (repo_root / "data/processed/gait_features.csv").read_bytes()
    partition_bytes = (
        repo_root / "data/processed/control_partition.json"
    ).read_bytes()
    pd_bytes = (repo_root / "experiments/results/pd_results.json").read_bytes()
    hd_bytes = (repo_root / "experiments/results/hd_results.json").read_bytes()
    als_bytes = (repo_root / "experiments/results/als_results.json").read_bytes()
    cross_bytes = (
        repo_root / "experiments/results/cross_condition_results.json"
    ).read_bytes()

    print("Submitting full Step 5 run to Modal...", flush=True)
    print("No reduced sweep mode: full sigma grid and 30 repeats enabled.", flush=True)
    print(flush=True)

    summary_json = run_step5_full.remote(
        gait_features_csv=features_bytes,
        control_partition_json=partition_bytes,
        pd_results_json=pd_bytes,
        hd_results_json=hd_bytes,
        als_results_json=als_bytes,
        cross_condition_results_json=cross_bytes,
    )

    print("Step 5 run complete.", flush=True)
    print(summary_json, flush=True)
    print("\nDownload outputs:", flush=True)
    print("  modal volume get gait-results noise_robustness.json", flush=True)
    print("  modal volume get gait-results feature_sensitivity.json", flush=True)
    print("  modal volume get gait-results subject_sensitivity.json", flush=True)
