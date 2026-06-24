"""
Lightweight equivalence check for RF recovery-only parallel execution.

Verifies that an RF configuration with n_jobs=1 and n_jobs=-1 yields
equivalent predictions, probabilities, macro-F1, and selected candidate /
strategy under the same fixed-seed diagnostic flow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score


def _infer_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError(f'Unable to locate repository root from {script_path}')


REPO_ROOT = _infer_repo_root()
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import get_feature_cols  # type: ignore
from scripts.verification.test_v4_subject_aggregation import (  # type: ignore
    _evaluate_candidate_outputs,
    _fit_outer_selected_candidate,
    _select_best_for_rule,
    _subject_level_arrays,
    get_classifier_configs,
)


def _build_fixture() -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
    feature_cols = get_feature_cols('v4')
    samples_per_subject = 8
    disease_subjects = [f'pd_{idx:02d}' for idx in range(1, 7)]
    control_subjects = [f'co_{idx:02d}' for idx in range(1, 7)]
    subject_ids: list[str] = []
    conditions: list[str] = []
    labels: list[int] = []
    for subject_id in disease_subjects:
        subject_ids.extend([subject_id] * samples_per_subject)
        conditions.extend(['pd'] * samples_per_subject)
        labels.extend([1] * samples_per_subject)
    for subject_id in control_subjects:
        subject_ids.extend([subject_id] * samples_per_subject)
        conditions.extend(['control'] * samples_per_subject)
        labels.extend([0] * samples_per_subject)

    X, _ = make_classification(
        n_samples=len(subject_ids),
        n_features=len(feature_cols),
        n_informative=min(10, len(feature_cols)),
        n_redundant=min(2, max(0, len(feature_cols) - 10)),
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.0,
        random_state=42,
    )
    data = {
        'subject_id': subject_ids,
        'condition': conditions,
        'label': labels,
    }
    for idx, col in enumerate(feature_cols):
        data[col] = X[:, idx].astype(float)
    return pl.DataFrame(data), feature_cols, disease_subjects, control_subjects


def _rf_template(n_jobs: int):
    configs = get_classifier_configs()
    template = clone(configs['rf']['clf'])
    template.set_params(n_jobs=n_jobs, random_state=42)
    return template, configs['rf']['param_grid']


def main() -> None:
    feature_df, feature_cols, disease_subjects, control_subjects = _build_fixture()
    rf_n1, rf_grid = _rf_template(n_jobs=1)
    rf_all, _ = _rf_template(n_jobs=-1)
    X = feature_df.select(feature_cols).to_numpy().astype(np.float64)
    y = feature_df['label'].to_numpy().astype(int)
    groups = feature_df['subject_id'].to_numpy().astype(str)

    held_out_subject = disease_subjects[0]
    test_mask = groups == held_out_subject
    train_mask = ~test_mask

    fixed_params = {
        key: values[0]
        for key, values in rf_grid.items()
    }
    comparison_params = {
        key: values[min(1, len(values) - 1)]
        for key, values in rf_grid.items()
    }

    outer_n1 = _fit_outer_selected_candidate(
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        groups_test=groups[test_mask],
        clf_name='rf',
        clf_template=rf_n1,
        params=fixed_params,
        imbalance_strategy='raw',
        rule='mean_probability',
    )
    outer_all = _fit_outer_selected_candidate(
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        groups_test=groups[test_mask],
        clf_name='rf',
        clf_template=rf_all,
        params=fixed_params,
        imbalance_strategy='raw',
        rule='mean_probability',
    )

    prediction_equality = (
        outer_n1['y_pred_subject'] == outer_all['y_pred_subject']
        and outer_n1['y_true_subject'] == outer_all['y_true_subject']
    )
    probability_difference = abs(
        float(outer_n1['subject_score']) - float(outer_all['subject_score'])
    )

    strategies = ('synthetic', 'balanced', 'raw')
    candidate_payloads_n1 = []
    candidate_payloads_all = []
    for params in (fixed_params, comparison_params):
        for strategy in strategies:
            outputs_n1 = _evaluate_candidate_outputs(
                X_train=X,
                y_train=y,
                groups_train=groups,
                clf_name='rf',
                clf_template=rf_n1,
                params=params,
                imbalance_strategy=strategy,
            )
            outputs_all = _evaluate_candidate_outputs(
                X_train=X,
                y_train=y,
                groups_train=groups,
                clf_name='rf',
                clf_template=rf_all,
                params=params,
                imbalance_strategy=strategy,
            )
            candidate_payloads_n1.append({
                'imbalance_strategy': strategy,
                'params': params,
                **outputs_n1,
            })
            candidate_payloads_all.append({
                'imbalance_strategy': strategy,
                'params': params,
                **outputs_all,
            })

    best_n1, _ = _select_best_for_rule(
        candidate_payloads=candidate_payloads_n1,
        rule='mean_probability',
    )
    best_all, _ = _select_best_for_rule(
        candidate_payloads=candidate_payloads_all,
        rule='mean_probability',
    )

    reference_outputs_n1 = candidate_payloads_n1[0]
    reference_outputs_all = candidate_payloads_all[0]
    row_pred_equal = np.array_equal(
        reference_outputs_n1['row_pred'],
        reference_outputs_all['row_pred'],
    )
    row_prob_diff = float(np.max(np.abs(
        reference_outputs_n1['row_prob'] - reference_outputs_all['row_prob']
    )))

    subj_true_n1, subj_pred_n1, _, _ = _subject_level_arrays(
        y_true=reference_outputs_n1['row_true'],
        y_pred=reference_outputs_n1['row_pred'],
        y_prob=reference_outputs_n1['row_prob'],
        subject_ids=reference_outputs_n1['row_subjects'],
        decision_scores=reference_outputs_n1['row_decision'],
        subject_aggregation_rule='mean_probability',
    )
    subj_true_all, subj_pred_all, _, _ = _subject_level_arrays(
        y_true=reference_outputs_all['row_true'],
        y_pred=reference_outputs_all['row_pred'],
        y_prob=reference_outputs_all['row_prob'],
        subject_ids=reference_outputs_all['row_subjects'],
        decision_scores=reference_outputs_all['row_decision'],
        subject_aggregation_rule='mean_probability',
    )
    f1_n1 = float(f1_score(subj_true_n1, subj_pred_n1, average='macro'))
    f1_all = float(f1_score(subj_true_all, subj_pred_all, average='macro'))

    report = {
        'prediction_equality': bool(prediction_equality and row_pred_equal),
        'maximum_absolute_probability_difference': max(probability_difference, row_prob_diff),
        'macro_f1_n_jobs_1': f1_n1,
        'macro_f1_n_jobs_all': f1_all,
        'macro_f1_equal': bool(np.isclose(f1_n1, f1_all, atol=1e-12)),
        'selected_params_equal': best_n1['params'] == best_all['params'],
        'selected_strategy_equal': (
            best_n1['imbalance_strategy'] == best_all['imbalance_strategy']
        ),
        'selected_params_n_jobs_1': best_n1['params'],
        'selected_params_n_jobs_all': best_all['params'],
        'selected_strategy_n_jobs_1': best_n1['imbalance_strategy'],
        'selected_strategy_n_jobs_all': best_all['imbalance_strategy'],
    }
    report['pass'] = bool(
        report['prediction_equality']
        and report['maximum_absolute_probability_difference'] <= 1e-12
        and report['macro_f1_equal']
        and report['selected_params_equal']
        and report['selected_strategy_equal']
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
