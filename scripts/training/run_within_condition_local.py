"""
Runner script for the publication-track v3 within-condition benchmark.

Executes nested LOSO-CV with GridSearchCV tuning for all 7 classifiers across
all three disease conditions (pd, hd, als) in sequence. Results are saved to
experiments/results/v3/{condition}_results_v3.json.

Usage:
    python scripts/training/run_within_condition_local.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import polars as pl

from features import get_feature_cols
from train import run_within_condition

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[2]
PROCESSED_ROOT  = REPO_ROOT / 'data' / 'processed' / 'v3'
RESULTS_DIR     = REPO_ROOT / 'experiments' / 'results' / 'v3'

CONDITIONS = ['pd', 'hd', 'als']


def main() -> None:
    df = pl.read_csv(str(PROCESSED_ROOT / 'gait_features_v3.csv'))
    feature_cols = get_feature_cols('v3')

    with open(PROCESSED_ROOT / 'control_partition_v3.json') as f:
        partition = json.load(f)
    control_a = partition['control_A']

    total_start = time.time()
    result_paths: list[Path] = []

    for condition in CONDITIONS:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n{"="*60}', flush=True)
        print(f'Condition: {condition.upper()}  |  Started: {ts}', flush=True)
        print(f'{"="*60}', flush=True)

        t0 = time.time()
        run_within_condition(
            condition,
            df,
            control_a,
            RESULTS_DIR,
            feature_cols=feature_cols,
            feature_matrix_file='v3/gait_features_v3.csv',
            feature_set_version='v3',
            normalization='none',
            results_filename=f'{condition}_results_v3.json',
            imbalance_arms=('synthetic', 'balanced', 'raw'),
            selection_arms=('synthetic', 'balanced'),
        )
        elapsed = time.time() - t0

        out_path = RESULTS_DIR / f'{condition}_results_v3.json'
        result_paths.append(out_path)
        print(
            f'\n{condition.upper()} complete in {elapsed:.0f}s  ->  {out_path}',
            flush=True,
        )

    total_elapsed = time.time() - total_start
    print(f'\n{"="*60}', flush=True)
    print(f'All conditions complete.  Total wall time: {total_elapsed:.0f}s', flush=True)
    print('Result files:', flush=True)
    for p in result_paths:
        print(f'  {p}', flush=True)
    print(f'{"="*60}', flush=True)


if __name__ == '__main__':
    main()
