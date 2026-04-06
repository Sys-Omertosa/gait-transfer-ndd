"""
Modal runner for within-condition baseline training.

Runs all three conditions (pd, hd, als) in parallel on separate Modal
containers. Each container is allocated 16 physical CPU cores and 32 GB RAM.
Total wall time equals the duration of the slowest single condition.

Differences from the local run (src/train.py):
  - LightGBM num_leaves grid includes {31, 63} — safe on 32 GB RAM
  - LightGBM n_jobs remains 1 — prevents OpenMP thread over-subscription
    inside GridSearchCV's parallel workers regardless of container size
  - All three conditions run simultaneously on separate containers

Usage (from repo root with venv active):
    modal run scripts/run_modal_training.py

Download results after completion:
    modal volume get gait-results pd_results.json
    modal volume get gait-results hd_results.json
    modal volume get gait-results als_results.json
"""

import modal
from pathlib import Path

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
    memory=16384,
    timeout=86400,     # 24-hour ceiling, far beyond the ~2.5 hour estimate
    volumes={"/results": volume},
)
def run_condition(
    gait_features_csv: bytes,
    control_partition_json: bytes,
    condition: str,
) -> str:
    """
    Train all 7 classifiers for one condition and return results as JSON.

    Replicates run_within_condition() from src/train.py with a Modal-specific
    LightGBM configuration: num_leaves ∈ {31, 63} is restored because 32 GB
    RAM eliminates the memory explosion that forced num_leaves ∈ {31} locally.
    LightGBM n_jobs=1 is kept to prevent OpenMP thread over-subscription inside
    GridSearchCV's parallel workers.
    """
    import os
    import json
    import time
    import tempfile
    import numpy as np
    import polars as pl
    from datetime import datetime
    from pathlib import Path as _Path
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    from features import ALL_FEATURE_COLS
    from train import (
        get_classifier_configs,
        build_pipeline,
        run_nested_loso,
        get_modal_params,
        _params_to_key,
    )

    # ── Write data files to temp location ────────────────────────────────────
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
    control_a = partition["control_A"]

    # ── Construct training pool ───────────────────────────────────────────────
    pool = df.filter(
        (pl.col("condition") == condition) |
        pl.col("subject_id").is_in(control_a)
    )
    pool_subjects = pool.n_unique("subject_id")
    pool_strides = pool.shape[0]

    X = pool.select(ALL_FEATURE_COLS).to_numpy().astype(np.float64)
    y = pool["label"].to_numpy().astype(int)
    groups = pool["subject_id"].to_numpy()

    # ── Classifier configs with Modal-specific LightGBM override ─────────────
    configs = get_classifier_configs()

    configs["rf"] = (
        RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
        ),
        configs["rf"][1],  # keep original grid unchanged
    )

    configs["xgb"] = (
        XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        ),
        configs["xgb"][1],  # keep original grid unchanged
    )

    configs["lgbm"] = (
        LGBMClassifier(
            random_state=42,
            n_jobs=1,           # prevents OpenMP over-subscription
            verbose=-1,
        ),
        {
            "clf__n_estimators":  [100, 200],
            "clf__max_depth":     [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.3],
            "clf__num_leaves":    [31, 63],   # full grid
        },
    )

    # ── Run each classifier ───────────────────────────────────────────────────
    clf_results = {}

    for clf_name, (clf, param_grid) in configs.items():
        pipeline = build_pipeline(clf_name, clf)

        t_start = time.time()
        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {clf_name:<6}  started: {start_ts}", flush=True)

        loso_out = run_nested_loso(X, y, groups, pipeline, param_grid)
        elapsed = time.time() - t_start
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        modal_params = get_modal_params(
            loso_out["fold_params"],
            loso_out["fold_best_scores"],
        )
        modal_key = _params_to_key(modal_params)
        modal_frequency = sum(
            1 for p in loso_out["fold_params"]
            if _params_to_key(p) == modal_key
        )

        f1_stored = round(float(loso_out["f1_macro"]), 6)
        y_true_list = loso_out["y_true_all"].tolist()
        y_pred_list = loso_out["y_pred_all"].tolist()

        # Verify the stored lists reproduce the stored F1 exactly.
        from sklearn.metrics import f1_score as _f1
        assert abs(_f1(y_true_list, y_pred_list, average="macro") - f1_stored) < 1e-6, \
            f"{clf_name}: F1 mismatch between stored value and prediction lists"

        clf_results[clf_name] = {
            "f1_macro":        f1_stored,
            "modal_params":    modal_params,
            "modal_frequency": modal_frequency,
            "y_true":          y_true_list,
            "y_pred":          y_pred_list,
        }

        print(
            f"  {clf_name:<6}  F1={loso_out['f1_macro']:.4f}  "
            f"modal={modal_params}  "
            f"start={start_ts}  end={end_ts}  ({elapsed:.0f}s)",
            flush=True,
        )

    # ── Assemble and persist output ───────────────────────────────────────────
    output = {
        "condition":     condition,
        "pool_subjects": pool_subjects,
        "pool_strides":  pool_strides,
        "classifiers":   clf_results,
    }

    results_dir = _Path("/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{condition}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return json.dumps(output, indent=2)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Launch all three conditions in parallel and collect results."""
    repo_root = Path(__file__).parent.parent.parent
    features_bytes = (
        repo_root / "data/processed/gait_features.csv"
    ).read_bytes()
    partition_bytes = (
        repo_root / "data/processed/control_partition.json"
    ).read_bytes()

    conditions = ["pd", "hd", "als"]

    print("Launching all three conditions in parallel on Modal...")
    print("Each condition: 16 physical CPU cores, 16 GB RAM, separate container.")
    print()

    # .spawn() submits all three simultaneously without blocking.
    # Modal provisions three independent containers on separate machines.
    futures = {
        condition: run_condition.spawn(
            gait_features_csv=features_bytes,
            control_partition_json=partition_bytes,
            condition=condition,
        )
        for condition in conditions
    }

    # Collect results as each condition finishes.
    for condition, future in futures.items():
        print(f"\n{'='*60}")
        print(f"Results — {condition.upper()}")
        print(f"{'='*60}")
        result_json = future.get()
        print(result_json)

    print("\nAll conditions complete.")
    print("Download results:")
    for condition in conditions:
        print(f"  modal volume get gait-results {condition}_results.json")
