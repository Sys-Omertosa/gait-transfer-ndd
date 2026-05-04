"""
Modal runner for the publication-track v3 within-condition benchmark.

Runs all three conditions (pd, hd, als) in parallel on separate Modal
containers. Each container reads the Step 1 artifacts from the shared Modal
volume and writes its result JSON to /results/results_v3/.

Usage:
    modal run scripts/training/run_within_condition_modal.py
"""

import json
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
def run_condition(condition: str) -> str:
    """
    Train all classifiers for one condition and return the results JSON string.
    """
    import json
    from pathlib import Path as _Path

    import polars as pl

    from features import get_feature_cols
    from train import run_within_condition

    processed_dir = _Path('/results/processed_v3')
    results_dir = _Path('/results/results_v3')
    features_path = processed_dir / 'gait_features_v3.csv'
    partition_path = processed_dir / 'control_partition_v3.json'

    if not features_path.exists() or not partition_path.exists():
        raise FileNotFoundError(
            'Missing Step 1 v3 artifacts on the Modal volume. '
            'Run scripts/training/run_preprocessing_modal.py first.'
        )

    df = pl.read_csv(str(features_path))
    with open(partition_path) as f:
        partition = json.load(f)
    feature_cols = get_feature_cols('v3')

    output = run_within_condition(
        condition=condition,
        df=df,
        control_subjects=partition["control_A"],
        results_dir=results_dir,
        feature_cols=feature_cols,
        feature_matrix_file='v3/gait_features_v3.csv',
        feature_set_version='v3',
        normalization='none',
        results_filename=f'{condition}_results_v3.json',
        imbalance_arms=('synthetic', 'balanced', 'raw'),
        selection_arms=('synthetic', 'balanced'),
    )
    return json.dumps(output, indent=2)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Launch all three conditions in parallel and collect results."""
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / 'experiments' / 'results' / 'v3'
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["pd", "hd", "als"]

    print("Launching all three conditions in parallel on Modal...")
    print("Each condition: 16 CPU, 12288 MB RAM, separate container.")
    print("Inputs are read from gait-results:/processed_v3 on the Modal volume.")
    print()

    futures = {
        condition: run_condition.spawn(condition=condition)
        for condition in conditions
    }

    pending = list(futures.items())
    while pending:
        for idx, (condition, future) in enumerate(pending):
            try:
                result = json.loads(future.get(timeout=5))
                out_path = results_dir / f'{condition}_results_v3.json'
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                selected_arms = {
                    clf_name: clf_out['selected_imbalance_strategy']
                    for clf_name, clf_out in result['classifiers'].items()
                }
                print(f"\n{'=' * 60}")
                print(f"Results - {condition.upper()}")
                print(f"{'=' * 60}")
                print(f"Selected imbalance arms: {selected_arms}")
                print(f'Local copy written to {out_path}')
                pending.pop(idx)
                break
            except TimeoutError:
                continue

    print("\nAll conditions complete.")
    print("Download results:")
    for condition in conditions:
        print(f"  modal volume get gait-results results_v3/{condition}_results_v3.json")
