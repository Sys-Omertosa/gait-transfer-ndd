"""
Sequential local runner for SHAP transfer-failure diagnosis.

Runs all six transfer directions in order:
    pd->hd, hd->pd, pd->als, als->pd, hd->als, als->hd

For each direction, calls run_shap_for_direction() from src/explain.py,
which computes SHAP values for all 7 classifiers, writes per-(source, clf)
.npz files to experiments/shap/v2_local/, and returns per-classifier δj
results.

After all six directions complete, the accumulated results are written to
experiments/results/v2/shap_results_v2_local.json following the same structure
and commit convention as cross_condition_results_v2.json. Local outputs are
kept separate from Modal outputs to avoid accidental overwrite when Modal
volume downloads are synced back into the repository.

TreeExplainer classifiers (RF, DT, XGB, LGB) complete in seconds to
tens of seconds per classifier per direction. KernelExplainer classifiers
(SVM, QDA, KNN) each take tens of minutes per direction. Total estimated
local wall time: several hours (dominated by KernelExplainer).

For Modal-accelerated execution with parallel directions, use:
    modal run scripts/training/run_shap_modal.py

Usage (from repo root with venv active):
    python scripts/training/run_shap_local.py
"""

import json
import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from explain import run_shap_for_direction  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH   = REPO_ROOT / 'data/processed/v2/gait_features_v2.csv'
PARTITION_PATH  = REPO_ROOT / 'data/processed/control_partition.json'
MODELS_DIR      = REPO_ROOT / 'experiments/models/v2'
SHAP_DIR        = REPO_ROOT / 'experiments/shap/v2_local'
RESULTS_DIR     = REPO_ROOT / 'experiments/results/v2'

DIRECTIONS = [
    ('pd',  'hd'),
    ('hd',  'pd'),
    ('pd',  'als'),
    ('als', 'pd'),
    ('hd',  'als'),
    ('als', 'hd'),
]


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FEATURES_PATH)
    with open(PARTITION_PATH) as f:
        partition = json.load(f)
    control_a: list[str] = partition['control_A']
    control_b: list[str] = partition['control_B']

    accumulated: dict = {}
    t_total = time.time()

    for source_cond, target_cond in DIRECTIONS:
        direction_key = f'{source_cond}_to_{target_cond}'
        print(f'\n{"="*60}', flush=True)
        print(f'Direction: {source_cond.upper()} -> {target_cond.upper()}', flush=True)
        print(f'{"="*60}', flush=True)
        t_dir = time.time()

        direction_result = run_shap_for_direction(
            source_condition=source_cond,
            target_condition=target_cond,
            df=df,
            control_a=control_a,
            control_b=control_b,
            models_dir=MODELS_DIR,
            shap_dir=SHAP_DIR,
        )

        elapsed = time.time() - t_dir
        accumulated[direction_key] = direction_result
        print(f'\nDirection {direction_key} complete in {elapsed:.0f}s', flush=True)

        # Write a partial results file after each direction so that a crash does
        # not lose completed work during a multi-hour run.
        partial_path = RESULTS_DIR / 'shap_results_v2_local_partial.json'
        with open(partial_path, 'w') as f:
            json.dump(accumulated, f, indent=2)
        print(f'Partial results saved ({len(accumulated)}/6 directions)', flush=True)

    # Rename the final partial file to the canonical output name.
    partial_path = RESULTS_DIR / 'shap_results_v2_local_partial.json'
    out_path = RESULTS_DIR / 'shap_results_v2_local.json'
    partial_path.rename(out_path)

    total_elapsed = time.time() - t_total
    print(f'\nAll six directions complete in {total_elapsed:.0f}s', flush=True)
    print(f'Results written to {out_path}', flush=True)


if __name__ == '__main__':
    main()
