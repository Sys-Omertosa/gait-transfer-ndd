"""
Modal runner for within-condition baseline training.

Runs all three conditions (pd, hd, als) in parallel on separate Modal
containers. Each container executes the same src/train.py logic used locally,
so the Modal rerun stays aligned with the current Step 2 feature set, grids,
ablation behavior, and JSON schema.

When SMOTE is enabled inside the delegated training pipeline, it augments the
minority control class defined by Control Group A. Control Group B remains
disjoint and reserved for downstream cross-condition evaluation.

Usage (from repo root with venv active):
    modal run scripts/training/run_within_condition_modal.py

Download results after completion:
    modal volume get gait-results pd_results_v2.json
    modal volume get gait-results hd_results_v2.json
    modal volume get gait-results als_results_v2.json
"""

from pathlib import Path

import modal

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements-core.txt")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gait-transfer-training", image=image)

# ── Persistent volume for results ─────────────────────────────────────────────
volume = modal.Volume.from_name("gait-results", create_if_missing=True)


# ── Per-condition training function ───────────────────────────────────────────
@app.function(
    cpu=16,
    memory=12288,
    timeout=86400,
    volumes={"/results": volume},
    retries=1,
)
def run_condition(
    gait_features_csv: bytes,
    control_partition_json: bytes,
    condition: str,
) -> str:
    """
    Train all classifiers for one condition and return the results JSON string.

    The function delegates to src/train.run_within_condition() so the Modal run
    matches the local Step 2 implementation exactly, including the source-side
    SMOTE/no-SMOTE ablation over the minority control class in Control Group A.
    """
    import json
    import os
    import tempfile

    import polars as pl

    from train import run_within_condition

    tmp = tempfile.mkdtemp()
    features_path = os.path.join(tmp, "gait_features.csv")
    partition_path = os.path.join(tmp, "control_partition.json")

    with open(features_path, "wb") as f:
        f.write(gait_features_csv)
    with open(partition_path, "wb") as f:
        f.write(control_partition_json)

    df = pl.read_csv(features_path)
    with open(partition_path) as f:
        partition = json.load(f)

    output = run_within_condition(
        condition=condition,
        df=df,
        control_subjects=partition["control_A"],
        results_dir="/results",
        feature_matrix_file="v2/gait_features_v2.csv",
        results_filename=f"{condition}_results_v2.json",
    )
    return json.dumps(output, indent=2)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Launch all three conditions in parallel and collect results."""
    repo_root = Path(__file__).resolve().parents[2]
    features_bytes = (
        repo_root / "data/processed/v2/gait_features_v2.csv").read_bytes()
    partition_bytes = (
        repo_root / "data/processed/control_partition.json").read_bytes()

    conditions = ["pd", "hd", "als"]

    print("Launching all three conditions in parallel on Modal...")
    print("Each condition: 16 CPU, 12288 MB RAM, separate container.")
    print()

    futures = {
        condition: run_condition.spawn(
            gait_features_csv=features_bytes,
            control_partition_json=partition_bytes,
            condition=condition,
        )
        for condition in conditions
    }

    for condition, future in futures.items():
        print(f"\n{'=' * 60}")
        print(f"Results - {condition.upper()}")
        print(f"{'=' * 60}")
        print(future.get())

    print("\nAll conditions complete.")
    print("Download results:")
    for condition in conditions:
        print(f"  modal volume get gait-results {condition}_results_v2.json")
