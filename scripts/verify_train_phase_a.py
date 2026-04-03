"""
Verification script for src/train.py Phase A.

Constructs the PD training pool (PD + Control A), runs run_nested_loso with
a reduced RF grid (n_estimators in {100, 200}, max_depth in {None, 10}) to
keep runtime under 5 minutes, and prints:
  - Pool construction stats
  - Aggregate F1 macro (single scalar over concatenated predictions)
  - Modal best params
  - Total wall time
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from features import ALL_FEATURE_COLS
from train import build_pipeline, run_nested_loso, get_modal_params

# ── Load artifacts ────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

df = pl.read_csv(str(PROCESSED_DIR / 'gait_features.csv'))

with open(PROCESSED_DIR / 'control_partition.json') as f:
    partition = json.load(f)
control_a = partition['control_A']

# ── Construct PD training pool ────────────────────────────────────────────────
pool = df.filter(
    (pl.col('condition') == 'pd') | pl.col('subject_id').is_in(control_a)
)

n_subjects = pool.n_unique('subject_id')
n_strides  = pool.shape[0]
n_disease  = pool.filter(pl.col('label') == 1).shape[0]
n_control  = pool.filter(pl.col('label') == 0).shape[0]

print('PD Training Pool:')
print(f'  Subjects : {n_subjects}  (PD=15, Control A=8)')
print(f'  Strides  : {n_strides}')
print(f'  Disease  : {n_disease}  |  Control: {n_control}  |  Ratio: {n_disease/n_control:.2f}:1')
print()

# ── Numpy handoff ─────────────────────────────────────────────────────────────
X      = pool.select(ALL_FEATURE_COLS).to_numpy().astype(np.float64)
y      = pool['label'].to_numpy().astype(int)
groups = pool['subject_id'].to_numpy()

# ── Build RF pipeline ─────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
pipeline = build_pipeline('rf', clf)

# ── Reduced grid for verification run ────────────────────────────────────────
param_grid = {
    'clf__n_estimators':     [100, 200],
    'clf__max_depth':        [None, 10],
    'clf__min_samples_leaf': [1],
}

print('Running nested LOSO (reduced grid: 2x2x1 = 4 combinations)...')
print('This may take a few minutes.\n')

t0 = time.time()
results = run_nested_loso(X, y, groups, pipeline, param_grid)
elapsed = time.time() - t0

# ── Results ───────────────────────────────────────────────────────────────────
modal = get_modal_params(results['fold_params'], results['fold_best_scores'])

print(f'Aggregate F1 macro : {results["f1_macro"]:.4f}')
print(f'Modal best params  : {modal}')
print()
print(f'Wall time: {elapsed:.1f}s')
