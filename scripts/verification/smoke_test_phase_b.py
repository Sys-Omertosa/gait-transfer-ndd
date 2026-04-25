"""
Smoke test for get_classifier_configs() in src/train.py Phase B.

Checks:
  - All 7 classifiers present
  - SVM and KNN have no n_jobs at classifier level
  - RF, XGBoost, LightGBM have n_jobs=-1
  - SVM has probability=True
  - QDA is QuadraticDiscriminantAnalysis, not LinearDiscriminantAnalysis

Does NOT run any training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC

from train import get_classifier_configs

configs = get_classifier_configs()

EXPECTED = ['rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm']
NEED_NJOBS = {'rf', 'xgb'}
NO_NJOBS   = {'knn', 'svm'}

passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = '') -> None:
    global passed, failed
    status = 'PASS' if condition else 'FAIL'
    msg = f'[{status}] {name}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1

# ── All 7 classifiers present ─────────────────────────────────────────────────
for name in EXPECTED:
    check(f'{name} present', name in configs)

# ── n_jobs checks ─────────────────────────────────────────────────────────────
for name in NEED_NJOBS:
    clf, _ = configs[name]
    has = getattr(clf, 'n_jobs', None) == -1
    check(f'{name} has n_jobs=-1', has, f'n_jobs={getattr(clf, "n_jobs", "NOT SET")}')

for name in NO_NJOBS:
    clf, _ = configs[name]
    has_njobs = hasattr(clf, 'n_jobs') and clf.n_jobs is not None
    check(f'{name} has no n_jobs', not has_njobs,
          f'n_jobs={getattr(clf, "n_jobs", "not set")}')

# ── SVM probability=True ──────────────────────────────────────────────────────
svm_clf, _ = configs['svm']
check('svm probability=True', isinstance(svm_clf, SVC) and svm_clf.probability is True,
      f'probability={svm_clf.probability}')

# ── QDA is QuadraticDiscriminantAnalysis ──────────────────────────────────────
qda_clf, _ = configs['qda']
check('qda is QDA not LDA', isinstance(qda_clf, QuadraticDiscriminantAnalysis),
      type(qda_clf).__name__)
check('qda is not LDA',     not isinstance(qda_clf, LinearDiscriminantAnalysis),
      type(qda_clf).__name__)

# ── LGBM has n_jobs=1 ──────────────────────────────────────────────────────
lgbm_clf, _ = configs['lgbm']
check('lgbm has n_jobs=1', getattr(lgbm_clf, 'n_jobs', None) == 1,
      f'n_jobs={getattr(lgbm_clf, "n_jobs", "NOT SET")}')

# ── Param grid key prefix checks (clf__) ─────────────────────────────────────
for name in EXPECTED:
    _, grid = configs[name]
    all_prefixed = all(k.startswith('clf__') for k in grid)
    check(f'{name} grid keys use clf__ prefix', all_prefixed,
          str(list(grid.keys())[:2]))

# ── Grid sizes match GUIDELINE.md ────────────────────────────────────────────
def grid_size(grid: dict) -> int:
    result = 1
    for v in grid.values():
        result *= len(v)
    return result

expected_sizes = {
    'rf':   3 * 3 * 3,   # 27
    'knn':  4 * 2 * 2,   # 16
    'svm':  4 * 4,        # 16
    'dt':   4 * 3 * 2,   # 24
    'qda':  3,            # 3
    'xgb':  2 * 3 * 3,   # 18
    'lgbm': 2 * 3 * 3 * 1, # 18
}
for name, expected in expected_sizes.items():
    _, grid = configs[name]
    actual = grid_size(grid)
    check(f'{name} grid size = {expected}', actual == expected,
          f'got {actual}')

# ── class_weight checks ───────────────────────────────────────────────────────
BALANCED_WEIGHT = {'rf', 'svm', 'dt'}
NO_CLASS_WEIGHT  = {'knn', 'qda', 'xgb', 'lgbm'}

for name in BALANCED_WEIGHT:
    clf, _ = configs[name]
    val = getattr(clf, 'class_weight', None)
    check(f'{name} class_weight=balanced', val == 'balanced',
          f'class_weight={val}')

for name in NO_CLASS_WEIGHT:
    clf, _ = configs[name]
    val = getattr(clf, 'class_weight', 'NOT SET')
    check(f'{name} has no class_weight', val == 'NOT SET' or val is None,
          f'class_weight={val}')

# ── use_label_encoder removed from XGBoost ───────────────────────────────────
xgb_clf, _ = configs['xgb']
check('xgb has no use_label_encoder', not hasattr(xgb_clf, 'use_label_encoder'))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f'Results: {passed} passed, {failed} failed')
