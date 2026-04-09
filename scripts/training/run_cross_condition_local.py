"""
Local sequential runner for zero-shot cross-condition transfer evaluation.

Runs all six transfer directions sequentially on local hardware:
    pd->hd, hd->pd, pd->als, als->pd, hd->als, als->hd

Each direction calls run_cross_condition() from src/train.py, which trains
once on the full source pool (source condition + Control A) using modal
hyperparameters from within-condition LOSO-CV, then evaluates on the full
target pool (target condition + Control B) with no retraining.

Results from all six directions are accumulated into a single output dict
and written to experiments/results/cross_condition_results.json after all
directions complete.

Usage (from repo root with venv active):
    python scripts/training/run_cross_condition_local.py

    # Or with logging:
    mkdir -p logs
    python scripts/training/run_cross_condition_local.py 2>&1 | tee logs/cross_condition_local.log
"""

import json
import sys
import time
from pathlib import Path

import polars as pl

# Resolve repo root regardless of working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from train import run_cross_condition  # noqa: E402

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
MODELS_DIR  = REPO_ROOT / "experiments" / "models"

DIRECTIONS = [
    ("pd",  "hd"),
    ("hd",  "pd"),
    ("pd",  "als"),
    ("als", "pd"),
    ("hd",  "als"),
    ("als", "hd"),
]

SOURCE_JSON = {
    "pd":  RESULTS_DIR / "pd_results.json",
    "hd":  RESULTS_DIR / "hd_results.json",
    "als": RESULTS_DIR / "als_results.json",
}


def main() -> None:
    # ── Load shared inputs ────────────────────────────────────────────────────
    print("Loading feature matrix and control partition...", flush=True)
    df = pl.read_csv(REPO_ROOT / "data" / "processed" / "gait_features.csv")
    with open(REPO_ROOT / "data" / "processed" / "control_partition.json") as f:
        partition = json.load(f)
    control_a = partition["control_A"]
    control_b = partition["control_B"]

    # ── Load Step 2 within-condition results ──────────────────────────────────
    source_results: dict[str, dict] = {}
    for cond, path in SOURCE_JSON.items():
        with open(path) as f:
            source_results[cond] = json.load(f)
        print(f"  Loaded {path.name}", flush=True)
    print(flush=True)

    # ── Run all six transfer directions ───────────────────────────────────────
    accumulated: dict[str, dict] = {}
    t_total_start = time.time()

    for source_cond, target_cond in DIRECTIONS:
        direction_key = f"{source_cond}_to_{target_cond}"
        print(f"{'='*60}", flush=True)
        print(f"Direction: {source_cond.upper()} -> {target_cond.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        t_dir_start = time.time()

        result = run_cross_condition(
            source_condition=source_cond,
            target_condition=target_cond,
            df=df,
            control_a=control_a,
            control_b=control_b,
            source_results=source_results[source_cond],
            results_dir=RESULTS_DIR,
            models_dir=MODELS_DIR,
        )

        elapsed = time.time() - t_dir_start
        accumulated[direction_key] = result
        print(f"\nDirection {direction_key} complete in {elapsed:.0f}s", flush=True)
        print(flush=True)

    # ── Write single output file after all directions complete ────────────────
    out_path = RESULTS_DIR / "cross_condition_results.json"
    with open(out_path, "w") as f:
        json.dump(accumulated, f, indent=2)

    total_elapsed = time.time() - t_total_start
    print(f"{'='*60}", flush=True)
    print(f"All six directions complete in {total_elapsed:.0f}s", flush=True)
    print(f"Results written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
