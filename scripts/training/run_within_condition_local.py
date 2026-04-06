"""
Runner script for the full within-condition baseline training.

Executes nested LOSO-CV with GridSearchCV tuning for all 7 classifiers across
all three disease conditions (pd, hd, als) in sequence. Results are saved to
experiments/results/{condition}_results.json.

Usage:
    python scripts/run_step2_training.py

    To run in the background and monitor progress in real time:
        nohup python scripts/run_step2_training.py > training_log.txt 2>&1 &
        tail -f training_log.txt
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import polars as pl

from train import run_within_condition

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
PROCESSED    = REPO_ROOT / 'data' / 'processed'
RESULTS_DIR  = REPO_ROOT / 'experiments' / 'results'

CONDITIONS = ['pd', 'hd', 'als']


def main() -> None:
    df = pl.read_csv(str(PROCESSED / 'gait_features.csv'))

    with open(PROCESSED / 'control_partition.json') as f:
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
        run_within_condition(condition, df, control_a, RESULTS_DIR)
        elapsed = time.time() - t0

        out_path = RESULTS_DIR / f'{condition}_results.json'
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
