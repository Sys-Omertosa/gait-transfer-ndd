"""
src/train.py

Within-condition baseline training utilities for gait classifier evaluation.

Functions implemented here:
  build_pipeline          -- constructs the correct ImbPipeline per classifier
  run_nested_loso         -- outer LOSO evaluation with inner GridSearchCV tuning
  get_modal_params        -- extracts modal best params from per-fold records
  get_classifier_configs  -- returns all 7 (clf, param_grid) configurations
  run_within_condition    -- orchestrates the full experiment for one condition

SMOTE convention: Mittal et al. (Frontiers Robotics AI, 2025) and Chu et al.
(MDPI Entropy, 2020) both apply SMOTE on training folds only on PhysioNet NDD
gait data. This module follows the same convention via ImbPipeline.

Feature scaling: Khera et al. (Scientific Reports, 2025) apply normalization on
GAITNDD features to address scale disparity across gait measurements. Here,
StandardScaler is applied for SVM and KNN only -- distance- and margin-based
classifiers whose kernels and distance metrics are distorted by unequal feature
scales. Tree-based methods and QDA are scale-invariant and receive no scaler.

F1 evaluation: each outer LOSO fold holds out one subject, who belongs to
exactly one class. Per-fold F1 macro is mathematically undefined over a
single-class test set. The correct approach is to accumulate predictions across
all folds and compute a single aggregate F1 macro over the full vector.
"""

from __future__ import annotations

import collections
import json
import time
import warnings
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import ALL_FEATURE_COLS

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ── Shared helpers ───────────────────────────────────────────────────────────

def _params_to_key(d: dict) -> tuple:
    """Convert a params dict to a hashable tuple for counting and comparison."""
    return tuple(sorted(d.items()))


# ── Classifiers that require feature scaling ──────────────────────────────────
# SVM and KNN are sensitive to feature scale: the RBF kernel and Euclidean/
# Manhattan distances are dominated by large-valued features without scaling.
# The 14 gait features span very different ranges (times ~0.3-2s vs
# percentages ~18-75%), so StandardScaler is required for these two only.
_SCALE_REQUIRED = {'svm', 'knn'}


def build_pipeline(classifier_name: str, clf: Any) -> ImbPipeline:
    """
    Construct an ImbPipeline for a given classifier.

    For SVM and KNN (distance- and margin-based classifiers sensitive to
    feature scale), the pipeline is:
        StandardScaler -> SMOTE -> classifier

    For all other classifiers (RF, DT, XGBoost, LightGBM, QDA):
        SMOTE -> classifier

    SMOTE is placed inside the pipeline so that it operates only on the
    training fold within each LOSO-CV iteration, never on the held-out test
    subject. This prevents synthetic samples from leaking information about
    the test subject into training.

    The clf argument should already be instantiated with all desired fixed
    parameters (e.g. class_weight, random_state, n_jobs). This function does
    not set any parameters on clf -- it only wraps it in the correct pipeline.

    Args:
        classifier_name: One of 'rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm'.
                         Case-insensitive.
        clf: An instantiated sklearn-compatible classifier object.

    Returns:
        ImbPipeline with steps appropriate for the given classifier.
    """
    name = classifier_name.lower()
    smote = SMOTE(random_state=42)

    if name in _SCALE_REQUIRED:
        steps = [
            ('scaler', StandardScaler()),
            ('smote', smote),
            ('clf', clf),
        ]
    else:
        steps = [
            ('smote', smote),
            ('clf', clf),
        ]

    return ImbPipeline(steps)


def run_nested_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: ImbPipeline,
    param_grid: dict[str, list],
) -> dict[str, Any]:
    """
    Outer LOSO-CV loop with inner GridSearchCV on the training fold only.

    For each outer fold (one held-out subject):
      1. Split pool into (n-1)-subject training set and 1-subject test set.
      2. Fit a fresh GridSearchCV with LeaveOneGroupOut on the training set,
         scoring by f1_macro. The inner splitter sees only training-fold
         subjects -- the held-out test subject is never visible during tuning.
      3. Predict on the held-out subject with the best estimator.
      4. Accumulate predictions and record best params and inner CV score.

    F1 is computed once over the full concatenated prediction vector after all
    outer folds complete. Each fold's test set contains strides from exactly
    one subject, who belongs to exactly one class. Computing F1 macro
    per-fold over a single-class test set is undefined; the aggregate
    computation over all folds is the scientifically correct evaluation.

    Args:
        X:          Feature matrix, shape (n_strides, n_features), float64.
        y:          Binary label vector, shape (n_strides,), int.
        groups:     Subject-ID array, shape (n_strides,), str or object.
                    Used as the group variable for LeaveOneGroupOut.
        pipeline:   ImbPipeline produced by build_pipeline().
        param_grid: Dict mapping pipeline step param names to value lists.
                    Keys must use the double-underscore convention, e.g.
                    {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10]}.

    Returns:
        Dict with keys:
          'f1_macro'         -- aggregate F1 macro over all folds (scalar)
          'y_true_all'       -- concatenated ground truth across all folds
          'y_pred_all'       -- concatenated predictions across all folds
          'fold_params'      -- list of per-fold best param dicts
          'fold_best_scores' -- list of per-fold inner CV best_score_ values
    """
    outer_loso = LeaveOneGroupOut()

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    fold_params: list[dict] = []
    fold_best_scores: list[float] = []

    for train_idx, test_idx in outer_loso.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Inner GridSearchCV uses a fresh LOSO splitter restricted to the
        # training fold. This ensures the held-out test subject's strides
        # never influence hyperparameter selection.
        inner_loso = LeaveOneGroupOut()
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_loso,
            scoring='f1_macro',
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train, groups=groups_train)

        y_pred = grid.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        fold_params.append(grid.best_params_)
        fold_best_scores.append(float(grid.best_score_))

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)

    return {
        'f1_macro':         f1_score(y_true_concat, y_pred_concat, average='macro'),
        'y_true_all':       y_true_concat,
        'y_pred_all':       y_pred_concat,
        'fold_params':      fold_params,
        'fold_best_scores': fold_best_scores,
    }


def get_modal_params(
    fold_params: list[dict],
    fold_scores: list[float] | None = None,
) -> dict:
    """
    Return the modal (most frequently selected) parameter combination.

    Each element of fold_params is the best_params_ dict from one outer LOSO
    fold. The mode is the combination that appears most often across folds.

    Tiebreaker: when two or more combinations appear equally often, the one
    with the higher mean inner-CV score (from fold_scores) across the folds
    where it was selected is returned. If fold_scores is not provided, the
    first-encountered tied combination is returned (deterministic because fold
    order is fixed by LeaveOneGroupOut).

    Args:
        fold_params: List of per-fold best param dicts from run_nested_loso.
        fold_scores: Optional list of per-fold inner CV best_score_ values,
                     same length as fold_params. Used for tiebreaking.

    Returns:
        The modal parameter dict.
    """
    keys = [_params_to_key(p) for p in fold_params]
    counter: collections.Counter = collections.Counter(keys)
    top_count = counter.most_common(1)[0][1]

    tied = [k for k, c in counter.items() if c == top_count]

    if len(tied) == 1 or fold_scores is None:
        return dict(tied[0])

    # Break ties by mean inner CV score across folds where each combination
    # was selected. Higher score wins.
    best_key = max(
        tied,
        key=lambda k: float(np.mean([
            s for key, s in zip(keys, fold_scores) if key == k
        ])),
    )
    return dict(best_key)


# ── Classifier configurations ─────────────────────────────────────────────────

def get_classifier_configs() -> dict[str, tuple[Any, dict[str, list]]]:
    """
    Return instantiated classifiers and their tuning grids for all 7 classifiers.

    Each value is a (clf_instance, param_grid) tuple. The param_grid keys use
    the clf__ prefix for ImbPipeline compatibility.

    Parallelism is handled at the GridSearchCV level.

    SVM is instantiated with probability=True, which is required for SHAP
    KernelExplainer to compute probability-based explanations.

    Returns:
        Dict mapping classifier name to (clf_instance, param_grid).
    """
    configs: dict[str, tuple[Any, dict[str, list]]] = {
        'rf': (
            RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=1,
            ),
            {
                'clf__n_estimators':    [100, 200, 500],
                'clf__max_depth':       [None, 10, 20],
                'clf__min_samples_leaf': [1, 2, 5],
            },
        ),
        'knn': (
            KNeighborsClassifier(),
            {
                'clf__n_neighbors': [3, 5, 7, 11],
                'clf__weights':     ['uniform', 'distance'],
                'clf__metric':      ['euclidean', 'manhattan'],
            },
        ),
        'svm': (
            SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42,
            ),
            {
                'clf__C':     [0.1, 1, 10, 100],
                'clf__gamma': ['scale', 'auto', 0.01, 0.1],
            },
        ),
        'dt': (
            DecisionTreeClassifier(
                class_weight='balanced',
                random_state=42,
            ),
            {
                'clf__max_depth':        [None, 5, 10, 20],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__criterion':        ['gini', 'entropy'],
            },
        ),
        'qda': (
            QuadraticDiscriminantAnalysis(),
            {
                'clf__reg_param': [0.0, 0.1, 0.5],
            },
        ),
        'xgb': (
            XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                n_jobs=1,
            ),
            {
                'clf__n_estimators':  [100, 200],
                'clf__max_depth':     [3, 5, 7],
                'clf__learning_rate': [0.01, 0.1, 0.3],
            },
        ),
        'lgbm': (
            LGBMClassifier(
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
            {
                'clf__n_estimators':  [100, 200],
                'clf__max_depth':     [3, 5, 7],
                'clf__learning_rate': [0.01, 0.1, 0.3],
                'clf__num_leaves':    [31],
            },
        ),
    }
    return configs


# ── Full within-condition orchestration ───────────────────────────────────────

def run_within_condition(
    condition: str,
    df: pl.DataFrame,
    control_a: list[str],
    results_dir: str | Path,
) -> dict[str, Any]:
    """
    Orchestrate the full within-condition experiment for one disease condition.

    Constructs the binary training pool (disease strides + Control Group A
    strides), runs nested LOSO-CV with GridSearchCV tuning for all 7
    classifiers, and saves results to a JSON file.

    Modal hyperparameters are extracted after all outer folds complete and
    serve as the fixed parameter set for the zero-shot cross-condition
    evaluation, where no target-condition data is available for tuning.

    The JSON output contains f1_macro, modal_params, and modal_frequency per
    classifier. The raw prediction arrays (y_true_all, y_pred_all) are not
    persisted because they are only needed transiently for the confusion matrix
    in the notebook.

    Args:
        condition:   One of 'pd', 'hd', 'als'. Must match the condition column
                     in df.
        df:          Full feature DataFrame from data/processed/gait_features.csv.
        control_a:   List of Control Group A subject IDs.
        results_dir: Directory where {condition}_results.json will be written.

    Returns:
        The results dict that was written to JSON.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Construct binary training pool ────────────────────────────────────────
    pool = df.filter(
        (pl.col('condition') == condition) | pl.col('subject_id').is_in(control_a)
    )
    pool_subjects = pool.n_unique('subject_id')
    pool_strides  = pool.shape[0]

    X      = pool.select(ALL_FEATURE_COLS).to_numpy().astype(np.float64)
    y      = pool['label'].to_numpy().astype(int)
    groups = pool['subject_id'].to_numpy()

    # ── Run each classifier ───────────────────────────────────────────────────
    clf_results: dict[str, dict] = {}

    for clf_name, (clf, param_grid) in get_classifier_configs().items():
        pipeline = build_pipeline(clf_name, clf)

        t_start = time.time()
        start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'  {clf_name:<6}  started: {start_ts}', flush=True)
        loso_out = run_nested_loso(X, y, groups, pipeline, param_grid)
        elapsed = time.time() - t_start
        end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        modal = get_modal_params(
            loso_out['fold_params'],
            loso_out['fold_best_scores'],
        )

        # Count how many outer folds selected the modal combination.
        modal_key = _params_to_key(modal)
        modal_frequency = sum(
            1 for p in loso_out['fold_params'] if _params_to_key(p) == modal_key
        )

        clf_results[clf_name] = {
            'f1_macro':       round(float(loso_out['f1_macro']), 6),
            'modal_params':   modal,
            'modal_frequency': modal_frequency,
        }

        print(
            f'  {clf_name:<6}  F1={loso_out["f1_macro"]:.4f}  '
            f'modal={modal}  start={start_ts}  end={end_ts}  ({elapsed:.0f}s)',
            flush=True,
        )

    # ── Assemble and save output ───────────────────────────────────────────────
    output = {
        'condition':    condition,
        'pool_subjects': pool_subjects,
        'pool_strides':  pool_strides,
        'classifiers':  clf_results,
    }

    out_path = results_dir / f'{condition}_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output
