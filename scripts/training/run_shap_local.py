"""
Sequential local runner for publication-track v3 SHAP transfer diagnosis.

Runs all six transfer directions in order:
    pd->hd, hd->pd, pd->als, als->pd, hd->als, als->hd

For each direction, calls run_shap_for_direction() from src/explain.py,
which computes SHAP values for all 7 classifiers, writes per-(source, clf)
.npz files to experiments/shap/v3/, and returns per-classifier δj results,
family-level summaries, and stability diagnostics.

After all six directions complete, the accumulated results are written to
experiments/results/v3/shap_results_v3.json.

Usage:
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
from features import get_feature_cols  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH   = REPO_ROOT / 'data/processed/v3/gait_features_v3.csv'
PARTITION_PATH  = REPO_ROOT / 'data/processed/v3/control_partition_v3.json'
MODELS_DIR      = REPO_ROOT / 'experiments/models/v3'
SHAP_DIR        = REPO_ROOT / 'experiments/shap/v3'
RESULTS_DIR     = REPO_ROOT / 'experiments/results/v3'

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
    feature_cols = get_feature_cols('v3')

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
            feature_cols=feature_cols,
            feature_set_version='v3',
            stability_n_resamples=200,
        )

        elapsed = time.time() - t_dir
        accumulated[direction_key] = direction_result
        print(f'\nDirection {direction_key} complete in {elapsed:.0f}s', flush=True)

        # Write a partial results file after each direction so that a crash does
        # not lose completed work during a multi-hour run.
        partial_path = RESULTS_DIR / 'shap_results_v3_partial.json'
        with open(partial_path, 'w') as f:
            json.dump(accumulated, f, indent=2)
        print(f'Partial results saved ({len(accumulated)}/6 directions)', flush=True)

    partial_path = RESULTS_DIR / 'shap_results_v3_partial.json'
    out_path = RESULTS_DIR / 'shap_results_v3.json'
    partial_path.rename(out_path)

    total_elapsed = time.time() - t_total
    print(f'\nAll six directions complete in {total_elapsed:.0f}s', flush=True)
    print(f'Results written to {out_path}', flush=True)


if __name__ == '__main__':
    main()
