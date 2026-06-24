"""
Cheap equivalence regression for fragmented subject-aggregation recovery.

Confirms that the recovery-only fragmented execution graph produces the same
final shard payload as the existing monolithic evaluator for one RF fixture and
one SVM fixture.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.datasets import make_classification


def _infer_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError(f'Unable to infer repository root from {script_path}')


REPO_ROOT = _infer_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from features import get_feature_cols  # type: ignore
from scripts.verification.test_v4_subject_aggregation import (  # type: ignore
    _build_fragment_context_from_objects,
    _evaluate_condition_classifier,
    _fit_outer_selected_candidate,
    _outer_result_relpath,
    _outer_train_test_arrays,
    _rf_template_for_stage,
    _run_fragmented_target_local,
    _validate_fragment_payload,
    _validate_shard_payload,
)


def _make_fixture(condition: str) -> tuple[pl.DataFrame, dict[str, list[str]], dict[str, dict]]:
    feature_cols = get_feature_cols('v4')
    disease_subjects = [f'{condition}_{idx:02d}' for idx in range(3)]
    control_subjects = [f'co_{idx:02d}' for idx in range(3)]
    samples_per_subject = 8
    subject_ids: list[str] = []
    conditions: list[str] = []
    labels: list[int] = []
    for subject_id in disease_subjects:
        subject_ids.extend([subject_id] * samples_per_subject)
        conditions.extend([condition] * samples_per_subject)
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
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=1.2,
        random_state=42 if condition == 'pd' else 43,
    )
    data = {
        'subject_id': subject_ids,
        'condition': conditions,
        'label': labels,
    }
    for idx, col in enumerate(feature_cols):
        data[col] = X[:, idx].astype(float)
    feature_df = pl.DataFrame(data)
    partition = {
        'control_A': control_subjects,
        'control_B': [],
    }
    v3_within = {
        'pd': {
            'classifiers': {
                'rf': {
                    'modal_params': {
                        'clf__n_estimators': 100,
                        'clf__max_depth': None,
                        'clf__max_features': 'sqrt',
                        'clf__min_samples_leaf': 1,
                    }
                }
            }
        },
        'hd': {
            'classifiers': {
                'svm': {
                    'modal_params': {
                        'clf__C': 1.0,
                        'clf__gamma': 0.01,
                    }
                }
            }
        },
        'als': {},
    }
    return feature_df, partition, v3_within


def _expected_outer_prediction(
    *,
    context: dict,
    clf_name: str,
    outer_fold_subject: str,
    selected_candidate_key: str,
    selected_strategy: str,
    rule: str,
) -> dict:
    selected = json.loads(selected_candidate_key)
    assert selected['imbalance_strategy'] == selected_strategy
    split = _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject)
    if clf_name == 'rf':
        clf_template = _rf_template_for_stage(context, n_jobs=2)
    else:
        clf_template = clone(context['configs']['svm']['clf'])
    outer_prediction = _fit_outer_selected_candidate(
        X_train=split['X_train'],
        y_train=split['y_train'],
        X_test=split['X_test'],
        y_test=split['y_test'],
        groups_test=split['groups_test'],
        clf_name=clf_name,
        clf_template=clf_template,
        params=selected['params'],
        imbalance_strategy=selected_strategy,
        rule=rule,
    )
    outer_prediction['outer_fold_subject'] = outer_fold_subject
    return outer_prediction


def _fragmented_outer_subject_map(
    *,
    context: dict,
    storage_root: Path,
) -> dict[str, dict[str, dict]]:
    mapped: dict[str, dict[str, dict]] = {
        rule: {} for rule in context['rules']
    }
    for outer_fold_subject in context['selected_outer_subjects']:
        relpath = _outer_result_relpath(
            condition=context['condition'],
            clf_name=context['clf_name'],
            outer_fold_subject=outer_fold_subject,
        )
        fragment = json.loads((storage_root / relpath).read_text())
        _validate_fragment_payload(
            fragment,
            expected_identity={
                'scope': 'full' if context['scope'] == 'full' else 'full',
                'condition': context['condition'],
                'classifier': context['clf_name'],
                'fragment_kind': f'{context["clf_name"]}_outer_result',
                'outer_fold_subject': outer_fold_subject,
                'full_source_stage': False,
                'candidate_index': None,
                'candidate_hash': None,
                'imbalance_strategy': None,
                'inner_held_out_subject': None,
                'aggregation_rules': list(context['rules']),
                'feature_matrix_hash': context['feature_matrix_hash'],
                'partition_hash': context['partition_hash'],
                'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
                'v3_source_result_hash': context['v3_source_result_hash'],
                'grid_hash': context['grid_hash'],
                'tie_break_rule': 'stride_macro_f1_then_lexicographic',
                'code_schema_version': 'subject-aggregation-fragment-v1',
                'random_seed': 42,
            },
            required_keys=('per_rule',),
        )
        for rule in context['rules']:
            per_rule = fragment['per_rule'][rule]
            mapped[rule][outer_fold_subject] = {
                'selected_candidate_key': per_rule['selected_candidate_key'],
                'selected_strategy': per_rule['selected_imbalance_strategy'],
                'top_tie_count': per_rule['top_tie_count'],
                'outer_prediction': per_rule['outer_prediction'],
            }
    return mapped


def _assert_payload_equivalent(
    lhs: dict,
    rhs: dict,
    *,
    clf_name: str,
    context: dict,
    storage_root: Path,
) -> None:
    assert lhs['condition'] == rhs['condition']
    assert lhs['classifier'] == rhs['classifier'] == clf_name
    _validate_shard_payload(
        rhs,
        expected_condition=lhs['condition'],
        expected_classifier=clf_name,
    )
    assert lhs['report']['candidate_strategy_policy'] == rhs['report']['candidate_strategy_policy']
    assert lhs['report']['param_candidates'] == rhs['report']['param_candidates']
    assert lhs['report']['candidate_count_per_outer_fold'] == rhs['report']['candidate_count_per_outer_fold']

    fragmented_subject_map = _fragmented_outer_subject_map(
        context=context,
        storage_root=storage_root,
    )

    for rule, lhs_rule in lhs['report']['rule_summary'].items():
        rhs_rule = rhs['report']['rule_summary'][rule]
        lhs_subject_map = {}
        rhs_subject_map = {}
        for idx, subject_id in enumerate(lhs_rule['outer_fold_subjects']):
            lhs_outer_prediction = _expected_outer_prediction(
                context=context,
                clf_name=clf_name,
                outer_fold_subject=subject_id,
                selected_candidate_key=lhs_rule['outer_fold_selected_candidate_keys'][idx],
                selected_strategy=lhs_rule['outer_fold_selected_strategies'][idx],
                rule=rule,
            )
            lhs_subject_map[subject_id] = {
                'selected_candidate_key': lhs_rule['outer_fold_selected_candidate_keys'][idx],
                'selected_strategy': lhs_rule['outer_fold_selected_strategies'][idx],
                'top_tie_count': lhs_rule['outer_fold_top_tie_counts'][idx],
                'outer_prediction': lhs_outer_prediction,
            }
            rhs_subject_map[subject_id] = fragmented_subject_map[rule][subject_id]

        assert lhs_subject_map.keys() == rhs_subject_map.keys()
        for subject_id in lhs_subject_map:
            lhs_entry = lhs_subject_map[subject_id]
            rhs_entry = rhs_subject_map[subject_id]
            assert lhs_entry['selected_candidate_key'] == rhs_entry['selected_candidate_key']
            assert lhs_entry['selected_strategy'] == rhs_entry['selected_strategy']
            assert lhs_entry['top_tie_count'] == rhs_entry['top_tie_count']
            lhs_prediction = lhs_entry['outer_prediction']
            rhs_prediction = rhs_entry['outer_prediction']
            assert lhs_prediction['subject_id'] == rhs_prediction['subject_id']
            assert lhs_prediction['y_true_subject'] == rhs_prediction['y_true_subject']
            assert lhs_prediction['y_pred_subject'] == rhs_prediction['y_pred_subject']
            assert np.isclose(
                lhs_prediction['subject_score'],
                rhs_prediction['subject_score'],
                atol=1e-12,
            )
        assert lhs_rule['full_source_selection'] == rhs_rule['full_source_selection']
        assert np.isclose(
            lhs_rule['outer_subject_f1_macro'],
            rhs_rule['outer_subject_f1_macro'],
            atol=1e-12,
        )
    assert lhs['source_best_scores_by_rule'] == rhs['source_best_scores_by_rule']
    assert lhs['report']['prediction_disagreement_vs_mean_probability'] == rhs['report']['prediction_disagreement_vs_mean_probability']
    assert lhs['report']['selection_disagreements_vs_mean_probability'] == rhs['report']['selection_disagreements_vs_mean_probability']
    assert lhs['report']['strategy_disagreements_vs_mean_probability'] == rhs['report']['strategy_disagreements_vs_mean_probability']
    assert lhs['report']['full_source_selection_changes_vs_mean_probability'] == rhs['report']['full_source_selection_changes_vs_mean_probability']


def main() -> None:
    summaries = []
    with tempfile.TemporaryDirectory(prefix='v4-fragmented-equiv-') as tmpdir:
        storage_root = Path(tmpdir)

        for condition, clf_name in [('pd', 'rf'), ('hd', 'svm')]:
            feature_df, partition, v3_within = _make_fixture(condition)
            monolithic = _evaluate_condition_classifier(
                feature_df=feature_df,
                partition=partition,
                condition=condition,
                clf_name=clf_name,
                scope='quick',
                v3_within=v3_within,
            )
            context = _build_fragment_context_from_objects(
                feature_df=feature_df,
                partition=partition,
                condition=condition,
                clf_name=clf_name,
                v3_within=v3_within,
                scope='quick',
            )
            fragmented = _run_fragmented_target_local(
                context=context,
                storage_root=storage_root,
                force=True,
            )['payload']
            _assert_payload_equivalent(
                monolithic,
                fragmented,
                clf_name=clf_name,
                context=context,
                storage_root=storage_root,
            )
            summaries.append({
                'condition': condition,
                'classifier': clf_name,
                'status': 'pass',
            })

    print(json.dumps({
        'status': 'pass',
        'summaries': summaries,
    }, indent=2))


if __name__ == '__main__':
    main()
