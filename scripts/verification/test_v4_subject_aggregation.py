"""
Real-data subject-aggregation diagnostic for the grouped v4 selector.

Local mode:
    python scripts/verification/test_v4_subject_aggregation.py --scope quick

Detached Modal mode:
    modal run --detach scripts/verification/test_v4_subject_aggregation.py --scope full --action submit
    modal run --detach scripts/verification/test_v4_subject_aggregation.py --scope full --action assemble
    modal run --detach scripts/verification/test_v4_subject_aggregation.py --scope full --action recover-missing --targets "pd:rf,hd:rf,als:rf,hd:svm"
    modal run --detach scripts/verification/test_v4_subject_aggregation.py --scope full --action fragmented-submit --targets "pd:rf,hd:rf,als:rf,hd:svm" --submission-batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover
    modal = None


def _is_dir_without_raising(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _infer_repo_and_src_roots() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()

    for candidate in (script_path.parent, *script_path.parents):
        src_root = candidate / 'src'
        if _is_dir_without_raising(src_root):
            return candidate, src_root

    modal_root = Path('/root')
    modal_src = modal_root / 'src'
    if _is_dir_without_raising(modal_src):
        return modal_root, modal_src

    raise RuntimeError(
        'Unable to locate src directory from '
        f'script_path={script_path}'
    )


REPO_ROOT, SRC_ROOT = _infer_repo_and_src_roots()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features import build_feature_matrix, get_feature_cols  # type: ignore
from train import (  # type: ignore
    DEFAULT_TIE_BREAK_RULE,
    _candidate_sort_key,
    _configure_classifier_for_resampling,
    _get_fit_kwargs,
    _subject_level_arrays,
    _subject_level_log_loss,
    build_pipeline,
    get_classifier_configs,
)
from v4_provenance import (  # type: ignore
    atomic_write_json,
    ensure_v4_preflight_scaffold,
    sha256_file,
    sha256_text,
    utc_now_iso,
)

CONDITIONS = ('pd', 'hd', 'als')
RULES = ('mean_probability', 'median_probability', 'majority_vote', 'mean_decision_score')
CLF_ORDER = ('rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm')
SCOPES = ('quick', 'full')
MAX_OUTER_FOLDS_PER_LABEL = 2
DIAGNOSTIC_CLASSIFIERS_BY_CONDITION = {
    'pd': ('rf', 'knn', 'svm'),
    'hd': ('dt', 'qda', 'xgb'),
    'als': ('svm', 'lgbm'),
}
STRATEGY_POLICY = {
    'rf': ('synthetic', 'balanced', 'raw'),
    'svm': ('synthetic', 'balanced', 'raw'),
    'dt': ('synthetic', 'balanced', 'raw'),
    'xgb': ('synthetic', 'balanced', 'raw'),
    'lgbm': ('synthetic', 'balanced', 'raw'),
    'knn': ('synthetic', 'raw'),
    'qda': ('synthetic', 'raw'),
}
REMOTE_RESULTS_DIR = '/results/results_v4_preflight'
REMOTE_V3_RESULTS_DIR = '/results/results_v3'
REMOTE_SHARD_DIR = 'results_v4_preflight/subject_aggregation_shards'
REMOTE_FRAGMENT_DIR = 'results_v4_preflight/subject_aggregation_fragments'
FRAGMENT_SCHEMA_VERSION = 'subject-aggregation-fragment-v1'
FRAGMENT_RANDOM_SEED = 42
DEFAULT_FRAGMENT_MAX_IN_FLIGHT = 64
RECOVERY_TARGETS_DEFAULT = (
    ('pd', 'rf'),
    ('hd', 'rf'),
    ('als', 'rf'),
    ('hd', 'svm'),
)


def _report_output_name(scope: str) -> str:
    return (
        'subject_aggregation_diagnostic.json'
        if scope == 'full'
        else 'subject_aggregation_diagnostic_quick.json'
    )


def _report_remote_relpath(scope: str) -> str:
    return f'results_v4_preflight/{_report_output_name(scope)}'


def _shard_remote_relpath(scope: str, condition: str, clf_name: str) -> str:
    return f'{REMOTE_SHARD_DIR}/{scope}_{condition}_{clf_name}.json'


def _fragment_target_prefix(condition: str, clf_name: str) -> str:
    return f'{REMOTE_FRAGMENT_DIR}/{condition}_{clf_name}'


def _supported_rules_for_classifier(clf_name: str) -> tuple[str, ...]:
    if clf_name == 'svm':
        return RULES
    return tuple(rule for rule in RULES if rule != 'mean_decision_score')


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def _candidate_hash(params: dict[str, Any]) -> str:
    return sha256_text(_stable_json(params))[:16]


def _grid_hash(
    *,
    clf_name: str,
    param_candidates: list[dict[str, Any]],
    strategy_policy: tuple[str, ...] | list[str],
) -> str:
    return sha256_text(_stable_json({
        'classifier': clf_name,
        'param_candidates': param_candidates,
        'candidate_strategy_policy': list(strategy_policy),
        'rules': list(_supported_rules_for_classifier(clf_name)),
        'tie_break_rule': DEFAULT_TIE_BREAK_RULE,
    }))


def _rf_outer_candidate_relpath(
    *,
    condition: str,
    outer_fold_subject: str,
    imbalance_strategy: str,
    candidate_hash: str,
) -> str:
    return (
        f'{_fragment_target_prefix(condition, "rf")}/outer_candidates/'
        f'{outer_fold_subject}/{imbalance_strategy}__{candidate_hash}.json'
    )


def _rf_outer_result_relpath(*, condition: str, outer_fold_subject: str) -> str:
    return f'{_fragment_target_prefix(condition, "rf")}/outer_results/{outer_fold_subject}.json'


def _rf_full_source_candidate_relpath(
    *,
    condition: str,
    imbalance_strategy: str,
    candidate_hash: str,
) -> str:
    return (
        f'{_fragment_target_prefix(condition, "rf")}/full_source_candidates/'
        f'{imbalance_strategy}__{candidate_hash}.json'
    )


def _svm_inner_fragment_relpath(
    *,
    condition: str,
    outer_fold_subject: str,
    stage_name: str,
    inner_held_out_subject: str,
    imbalance_strategy: str,
    candidate_hash: str,
) -> str:
    return (
        f'{_fragment_target_prefix(condition, "svm")}/{stage_name}/'
        f'{outer_fold_subject}/{imbalance_strategy}__{candidate_hash}__'
        f'inner_{inner_held_out_subject}.json'
    )


def _svm_candidate_summary_relpath(
    *,
    condition: str,
    outer_fold_subject: str | None,
    stage_name: str,
    imbalance_strategy: str,
    candidate_hash: str,
) -> str:
    if outer_fold_subject is None:
        return (
            f'{_fragment_target_prefix(condition, "svm")}/{stage_name}/'
            f'{imbalance_strategy}__{candidate_hash}.json'
        )
    return (
        f'{_fragment_target_prefix(condition, "svm")}/{stage_name}/'
        f'{outer_fold_subject}/{imbalance_strategy}__{candidate_hash}.json'
    )


def _outer_result_relpath(*, condition: str, clf_name: str, outer_fold_subject: str) -> str:
    if clf_name == 'rf':
        return _rf_outer_result_relpath(
            condition=condition,
            outer_fold_subject=outer_fold_subject,
        )
    return f'{_fragment_target_prefix(condition, clf_name)}/outer_results/{outer_fold_subject}.json'


def _full_source_selection_relpath(*, condition: str, clf_name: str) -> str:
    return f'{_fragment_target_prefix(condition, clf_name)}/full_source_selection.json'


def _recovery_target_kind(condition: str, clf_name: str) -> str:
    if clf_name == 'rf':
        return 'rf'
    if condition == 'hd' and clf_name == 'svm':
        return 'svm_deep'
    raise ValueError(
        f'Unsupported fragmented recovery target {condition}:{clf_name}. '
        f'Expected one of {RECOVERY_TARGETS_DEFAULT}.'
    )


def _required_shard_rules_for_classifier(clf_name: str) -> tuple[str, ...]:
    return _supported_rules_for_classifier(clf_name)


def _required_shard_report_keys() -> tuple[str, ...]:
    return (
        'n_outer_folds_total',
        'n_outer_folds_sampled',
        'outer_fold_subjects_sampled',
        'candidate_strategy_policy',
        'param_candidates',
        'candidate_count_per_outer_fold',
        'used_reduced_grid',
        'used_reduced_outer_folds',
        'rule_summary',
        'prediction_disagreement_vs_mean_probability',
        'selection_disagreements_vs_mean_probability',
        'strategy_disagreements_vs_mean_probability',
        'full_source_selection_changes_vs_mean_probability',
    )


def _validate_shard_payload(
    payload: dict[str, Any],
    *,
    expected_condition: str,
    expected_classifier: str,
) -> None:
    if payload.get('condition') != expected_condition:
        raise ValueError(
            f'Shard condition mismatch: expected {expected_condition}, '
            f'got {payload.get("condition")!r}.'
        )
    if payload.get('classifier') != expected_classifier:
        raise ValueError(
            f'Shard classifier mismatch: expected {expected_classifier}, '
            f'got {payload.get("classifier")!r}.'
        )
    report = payload.get('report')
    if not isinstance(report, dict):
        raise ValueError('Shard payload is missing top-level "report" object.')
    missing = [
        key for key in _required_shard_report_keys()
        if key not in report
    ]
    if missing:
        raise ValueError(
            f'Shard payload for {expected_condition}/{expected_classifier} is missing '
            f'required report keys: {missing}'
        )
    if 'source_best_scores_by_rule' not in payload:
        raise ValueError(
            f'Shard payload for {expected_condition}/{expected_classifier} is missing '
            '"source_best_scores_by_rule".'
        )


def _parse_targets(targets: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_target in [part.strip() for part in targets.split(',') if part.strip()]:
        if ':' not in raw_target:
            raise SystemExit(
                f'Invalid target {raw_target!r}. Expected condition:classifier, '
                'for example "pd:rf".'
            )
        condition, clf_name = [part.strip() for part in raw_target.split(':', 1)]
        if condition not in CONDITIONS:
            raise SystemExit(
                f'Invalid condition {condition!r}. Expected one of {CONDITIONS}.'
            )
        if clf_name not in CLF_ORDER:
            raise SystemExit(
                f'Invalid classifier {clf_name!r}. Expected one of {CLF_ORDER}.'
            )
        key = (condition, clf_name)
        if key not in seen:
            parsed.append(key)
            seen.add(key)
    if not parsed:
        raise SystemExit(
            'No valid recovery targets were provided. Example: --targets "pd:rf,hd:svm"'
        )
    return parsed


def _apply_execution_overrides(
    *,
    clf_name: str,
    clf_template: Any,
    execution_overrides: dict[str, Any] | None,
) -> Any:
    if not execution_overrides:
        return clf_template
    if clf_name == 'rf' and execution_overrides.get('rf_n_jobs') is not None:
        overridden = clone(clf_template)
        overridden.set_params(
            n_jobs=int(execution_overrides['rf_n_jobs']),
            random_state=42,
        )
        return overridden
    return clf_template


def _reference_artifacts(
    *,
    processed_dir: Path,
    allow_build: bool,
) -> tuple[pl.DataFrame, dict[str, list[str]], dict[str, str]]:
    features_path = processed_dir / 'gait_features_v4_reference.csv'
    partition_path = processed_dir / 'control_partition_v4_reference.json'
    manifest_path = processed_dir / 'preprocessing_manifest_v4_reference.json'
    if not (features_path.exists() and partition_path.exists() and manifest_path.exists()):
        if not allow_build:
            raise FileNotFoundError(
                'Missing preflight Step 1 artifacts on the Modal volume. '
                'Run the detached Step 1 preflight diagnostic first.'
            )
        feature_df, partition = build_feature_matrix(
            processed_dir=processed_dir,
            output_filename=features_path.name,
            feature_cols=get_feature_cols('v4'),
            feature_set_version='v4',
            filter_strategy='v3',
            control_partition_version='v4',
            partition_output_filename=partition_path.name,
            manifest_filename=manifest_path.name,
            robust_mad_multiplier=3.0,
        )
        return feature_df, partition, {
            'features_path': str(features_path),
            'partition_path': str(partition_path),
            'manifest_path': str(manifest_path),
        }
    with open(partition_path) as f:
        partition = json.load(f)
    return pl.read_csv(features_path), partition, {
        'features_path': str(features_path),
        'partition_path': str(partition_path),
        'manifest_path': str(manifest_path),
    }


def _load_v3_within_results(
    results_v3_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    base = results_v3_dir or (REPO_ROOT / 'experiments' / 'results' / 'v3')
    missing = [
        condition for condition in CONDITIONS
        if not (base / f'{condition}_results_v3.json').exists()
    ]
    if missing:
        if str(base).startswith('/results/'):
            raise FileNotFoundError(
                'Missing v3 within-condition JSONs on the Modal volume at '
                f'{base}. Upload them first with:\n'
                '  modal volume put gait-results experiments/results/v3/pd_results_v3.json results_v3/pd_results_v3.json\n'
                '  modal volume put gait-results experiments/results/v3/hd_results_v3.json results_v3/hd_results_v3.json\n'
                '  modal volume put gait-results experiments/results/v3/als_results_v3.json results_v3/als_results_v3.json'
            )
        raise FileNotFoundError(
            f'Missing local v3 within-condition JSONs under {base}: {missing}'
        )
    return {
        condition: json.loads((base / f'{condition}_results_v3.json').read_text())
        for condition in CONDITIONS
    }


def _param_candidates(
    *,
    condition: str,
    clf_name: str,
    param_grid: dict[str, list[Any]],
    v3_within: dict[str, dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    full_grid = [dict(params) for params in ParameterGrid(param_grid)]
    if scope == 'full':
        return full_grid

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(params: dict[str, Any]) -> None:
        key = json.dumps(params, sort_keys=True, separators=(',', ':'))
        if key not in seen:
            selected.append(dict(params))
            seen.add(key)

    prior = v3_within[condition]['classifiers'][clf_name]['modal_params']
    add(prior)
    if full_grid:
        for idx in (len(full_grid) // 2, 0, len(full_grid) - 1):
            add(full_grid[idx])
            if len(selected) >= 2:
                break
    return selected


def _candidate_key(candidate_summary: dict[str, Any]) -> str:
    return json.dumps(
        {
            'imbalance_strategy': candidate_summary['imbalance_strategy'],
            'params': candidate_summary['params'],
        },
        sort_keys=True,
        separators=(',', ':'),
    )


def _classifiers_for_condition(condition: str, scope: str) -> tuple[str, ...]:
    if scope == 'full':
        return CLF_ORDER
    return DIAGNOSTIC_CLASSIFIERS_BY_CONDITION[condition]


def _select_outer_subjects(
    *,
    pool: pl.DataFrame,
    condition: str,
    control_a: list[str],
    scope: str,
) -> list[str]:
    disease_subjects = sorted(
        pool.filter(pl.col('condition') == condition)['subject_id']
        .unique()
        .to_list()
    )
    control_subjects = sorted(
        pool.filter(pl.col('subject_id').is_in(control_a))['subject_id']
        .unique()
        .to_list()
    )
    if scope == 'full':
        return disease_subjects + control_subjects

    def pick(subject_ids: list[str]) -> list[str]:
        if len(subject_ids) <= MAX_OUTER_FOLDS_PER_LABEL:
            return subject_ids
        positions = np.linspace(
            0,
            len(subject_ids) - 1,
            num=MAX_OUTER_FOLDS_PER_LABEL,
            dtype=int,
        )
        return [subject_ids[idx] for idx in positions]

    return pick(disease_subjects) + pick(control_subjects)


def _scores_by_rule_from_row_outputs(
    *,
    row_true: np.ndarray,
    row_pred: np.ndarray,
    row_prob: np.ndarray,
    row_subjects: np.ndarray,
    row_decision: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    scores_by_rule: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        if rule == 'mean_decision_score' and row_decision is None:
            continue
        subj_true, subj_pred, subj_scores, subj_ids = _subject_level_arrays(
            y_true=row_true,
            y_pred=row_pred,
            y_prob=row_prob,
            subject_ids=row_subjects,
            decision_scores=row_decision,
            subject_aggregation_rule=rule,
        )
        scores_by_rule[rule] = {
            'inner_subject_f1': round(float(f1_score(subj_true, subj_pred, average='macro')), 6),
            'inner_stride_f1': round(float(f1_score(row_true, row_pred, average='macro')), 6),
            'inner_subject_log_loss': _subject_level_log_loss(subj_true, subj_scores),
            'subject_ids': subj_ids,
            'subject_y_true': subj_true.tolist(),
            'subject_y_pred': subj_pred.tolist(),
            'subject_scores': np.round(subj_scores, 6).tolist(),
        }
    return scores_by_rule


def _evaluate_candidate_outputs(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    clf_name: str,
    clf_template: Any,
    params: dict[str, Any],
    imbalance_strategy: str,
) -> dict[str, Any]:
    inner_loso = LeaveOneGroupOut()
    row_true_all: list[np.ndarray] = []
    row_pred_all: list[np.ndarray] = []
    row_prob_all: list[np.ndarray] = []
    row_subjects_all: list[np.ndarray] = []
    row_decision_all: list[np.ndarray] = []
    has_decision = True

    for inner_train_idx, inner_test_idx in inner_loso.split(X_train, y_train, groups_train):
        clf_variant = _configure_classifier_for_resampling(
            clf_name,
            clone(clf_template),
            imbalance_strategy,
        )
        pipeline = build_pipeline(
            clf_name,
            clf_variant,
            imbalance_strategy=imbalance_strategy,
        )
        pipeline.set_params(**params)

        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_test = X_train[inner_test_idx]
        y_inner_test = y_train[inner_test_idx]
        groups_inner_test = groups_train[inner_test_idx]

        fit_kwargs = _get_fit_kwargs(clf_name, y_inner_train, imbalance_strategy)
        pipeline.fit(X_inner_train, y_inner_train, **fit_kwargs)

        row_true_all.append(y_inner_test)
        row_pred_all.append(np.asarray(pipeline.predict(X_inner_test), dtype=int))
        row_prob_all.append(np.asarray(pipeline.predict_proba(X_inner_test)[:, 1], dtype=float))
        row_subjects_all.append(groups_inner_test.astype(str))
        if hasattr(pipeline, 'decision_function'):
            row_decision_all.append(
                np.asarray(pipeline.decision_function(X_inner_test), dtype=float)
            )
        else:
            has_decision = False

    row_true = np.concatenate(row_true_all)
    row_pred = np.concatenate(row_pred_all)
    row_prob = np.concatenate(row_prob_all)
    row_subjects = np.concatenate(row_subjects_all)
    row_decision = (
        np.concatenate(row_decision_all) if has_decision and row_decision_all else None
    )

    return {
        'row_true': row_true,
        'row_pred': row_pred,
        'row_prob': row_prob,
        'row_subjects': row_subjects,
        'row_decision': row_decision,
        'scores_by_rule': _scores_by_rule_from_row_outputs(
            row_true=row_true,
            row_pred=row_pred,
            row_prob=row_prob,
            row_subjects=row_subjects,
            row_decision=row_decision,
        ),
    }


def _select_best_for_rule(
    *,
    candidate_payloads: list[dict[str, Any]],
    rule: str,
) -> tuple[dict[str, Any], int]:
    candidate_summaries = []
    for idx, payload in enumerate(candidate_payloads):
        rule_scores = payload['scores_by_rule'].get(rule)
        if rule_scores is None:
            continue
        candidate_summaries.append({
            'candidate_index': idx,
            'imbalance_strategy': payload['imbalance_strategy'],
            'params': payload['params'],
            'inner_subject_f1': rule_scores['inner_subject_f1'],
            'inner_stride_f1': rule_scores['inner_stride_f1'],
            'inner_subject_log_loss': rule_scores['inner_subject_log_loss'],
        })
    ranked = sorted(
        candidate_summaries,
        key=lambda item: _candidate_sort_key(item, tie_break_rule=DEFAULT_TIE_BREAK_RULE),
    )
    top_subject_f1 = ranked[0]['inner_subject_f1']
    top_tie_count = sum(
        1
        for candidate in ranked
        if float(candidate['inner_subject_f1']) == float(top_subject_f1)
    )
    return ranked[0], top_tie_count


def _fit_outer_selected_candidate(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    clf_name: str,
    clf_template: Any,
    params: dict[str, Any],
    imbalance_strategy: str,
    rule: str,
) -> dict[str, Any]:
    clf_variant = _configure_classifier_for_resampling(
        clf_name,
        clone(clf_template),
        imbalance_strategy,
    )
    pipeline = build_pipeline(
        clf_name,
        clf_variant,
        imbalance_strategy=imbalance_strategy,
    )
    pipeline.set_params(**params)
    fit_kwargs = _get_fit_kwargs(clf_name, y_train, imbalance_strategy)
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred = np.asarray(pipeline.predict(X_test), dtype=int)
    y_prob = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float)
    decision_scores = (
        np.asarray(pipeline.decision_function(X_test), dtype=float)
        if hasattr(pipeline, 'decision_function') else None
    )
    subj_true, subj_pred, subj_scores, subj_ids = _subject_level_arrays(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        subject_ids=groups_test.astype(str),
        decision_scores=decision_scores,
        subject_aggregation_rule=rule,
    )
    return {
        'subject_id': subj_ids[0],
        'y_true_subject': int(subj_true[0]),
        'y_pred_subject': int(subj_pred[0]),
        'subject_score': round(float(subj_scores[0]), 6),
    }


def _build_shard_payload_from_components(
    *,
    condition: str,
    clf_name: str,
    n_outer_folds_total: int,
    selected_outer_subjects: list[str],
    param_candidates: list[dict[str, Any]],
    strategy_policy: tuple[str, ...] | list[str],
    used_reduced_grid: bool,
    used_reduced_outer_folds: bool,
    outer_predictions_by_rule: dict[str, list[dict[str, Any]]],
    selection_keys_by_rule: dict[str, list[str]],
    selection_strategies_by_rule: dict[str, list[str]],
    tie_counts_by_rule: dict[str, list[int]],
    full_source_selection: dict[str, Any],
) -> dict[str, Any]:
    rule_summary: dict[str, Any] = {}
    source_best_scores_by_rule: dict[str, float] = {}
    for rule in RULES:
        if rule == 'mean_decision_score' and clf_name != 'svm':
            continue
        preds = outer_predictions_by_rule[rule]
        y_true_subject = np.asarray([row['y_true_subject'] for row in preds], dtype=int)
        y_pred_subject = np.asarray([row['y_pred_subject'] for row in preds], dtype=int)
        subject_f1 = round(float(f1_score(y_true_subject, y_pred_subject, average='macro')), 6)
        rule_summary[rule] = {
            'outer_subject_f1_macro': subject_f1,
            'outer_fold_subjects': [row['outer_fold_subject'] for row in preds],
            'outer_fold_selected_candidate_keys': selection_keys_by_rule[rule],
            'outer_fold_selected_strategies': selection_strategies_by_rule[rule],
            'outer_fold_top_tie_counts': tie_counts_by_rule[rule],
            'full_source_selection': full_source_selection[rule],
        }
        source_best_scores_by_rule[rule] = subject_f1

    reference_predictions = outer_predictions_by_rule['mean_probability']
    prediction_disagreement_vs_mean_probability: dict[str, Any] = {}
    selection_disagreements_vs_mean_probability: dict[str, Any] = {}
    strategy_disagreements_vs_mean_probability: dict[str, Any] = {}
    full_source_selection_changes_vs_mean_probability: dict[str, bool] = {}
    for rule in RULES:
        if rule == 'mean_probability':
            continue
        if rule == 'mean_decision_score' and clf_name != 'svm':
            continue
        prediction_disagreement_vs_mean_probability[rule] = round(float(np.mean([
            int(ref_row['y_pred_subject'] != cmp_row['y_pred_subject'])
            for ref_row, cmp_row in zip(
                reference_predictions,
                outer_predictions_by_rule[rule],
                strict=True,
            )
        ])), 6)
        selection_disagreements_vs_mean_probability[rule] = int(sum(
            ref_key != cmp_key
            for ref_key, cmp_key in zip(
                selection_keys_by_rule['mean_probability'],
                selection_keys_by_rule[rule],
                strict=True,
            )
        ))
        strategy_disagreements_vs_mean_probability[rule] = int(sum(
            ref_strategy != cmp_strategy
            for ref_strategy, cmp_strategy in zip(
                selection_strategies_by_rule['mean_probability'],
                selection_strategies_by_rule[rule],
                strict=True,
            )
        ))
        full_source_selection_changes_vs_mean_probability[rule] = bool(
            full_source_selection[rule]['selected_candidate_key']
            != full_source_selection['mean_probability']['selected_candidate_key']
        )

    return {
        'condition': condition,
        'classifier': clf_name,
        'report': {
            'n_outer_folds_total': int(n_outer_folds_total),
            'n_outer_folds_sampled': int(len(selected_outer_subjects)),
            'outer_fold_subjects_sampled': list(selected_outer_subjects),
            'candidate_strategy_policy': list(strategy_policy),
            'param_candidates': param_candidates,
            'candidate_count_per_outer_fold': int(
                len(list(strategy_policy)) * len(param_candidates)
            ),
            'used_reduced_grid': bool(used_reduced_grid),
            'used_reduced_outer_folds': bool(used_reduced_outer_folds),
            'rule_summary': rule_summary,
            'prediction_disagreement_vs_mean_probability': (
                prediction_disagreement_vs_mean_probability
            ),
            'selection_disagreements_vs_mean_probability': (
                selection_disagreements_vs_mean_probability
            ),
            'strategy_disagreements_vs_mean_probability': (
                strategy_disagreements_vs_mean_probability
            ),
            'full_source_selection_changes_vs_mean_probability': (
                full_source_selection_changes_vs_mean_probability
            ),
        },
        'source_best_scores_by_rule': source_best_scores_by_rule,
    }


def _evaluate_condition_classifier(
    *,
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
    condition: str,
    clf_name: str,
    scope: str,
    v3_within: dict[str, dict[str, Any]],
    execution_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control_a = partition['control_A']
    feature_cols = get_feature_cols('v4')
    configs = get_classifier_configs()
    clf_template = _apply_execution_overrides(
        clf_name=clf_name,
        clf_template=configs[clf_name]['clf'],
        execution_overrides=execution_overrides,
    )

    pool = feature_df.filter(
        (pl.col('condition') == condition) | pl.col('subject_id').is_in(control_a)
    )
    X = pool.select(feature_cols).to_numpy().astype(np.float64)
    y = pool['label'].to_numpy().astype(int)
    groups = pool['subject_id'].to_numpy().astype(str)
    outer = LeaveOneGroupOut()
    selected_outer_subjects = _select_outer_subjects(
        pool=pool,
        condition=condition,
        control_a=control_a,
        scope=scope,
    )
    param_candidates = _param_candidates(
        condition=condition,
        clf_name=clf_name,
        param_grid=configs[clf_name]['param_grid'],
        v3_within=v3_within,
        scope=scope,
    )

    outer_predictions_by_rule: dict[str, list[dict[str, Any]]] = {
        rule: [] for rule in RULES
    }
    selection_keys_by_rule: dict[str, list[str]] = {rule: [] for rule in RULES}
    selection_strategies_by_rule: dict[str, list[str]] = {rule: [] for rule in RULES}
    tie_counts_by_rule: dict[str, list[int]] = {rule: [] for rule in RULES}

    for outer_train_idx, outer_test_idx in outer.split(X, y, groups):
        held_out_subject = str(groups[outer_test_idx][0])
        if held_out_subject not in selected_outer_subjects:
            continue
        X_train = X[outer_train_idx]
        y_train = y[outer_train_idx]
        groups_train = groups[outer_train_idx]
        X_test = X[outer_test_idx]
        y_test = y[outer_test_idx]
        groups_test = groups[outer_test_idx]

        candidate_payloads: list[dict[str, Any]] = []
        for strategy in STRATEGY_POLICY[clf_name]:
            for params in param_candidates:
                outputs = _evaluate_candidate_outputs(
                    X_train=X_train,
                    y_train=y_train,
                    groups_train=groups_train,
                    clf_name=clf_name,
                    clf_template=clf_template,
                    params=params,
                    imbalance_strategy=strategy,
                )
                candidate_payloads.append({
                    'imbalance_strategy': strategy,
                    'params': params,
                    **outputs,
                })

        for rule in RULES:
            if rule == 'mean_decision_score' and clf_name != 'svm':
                continue
            best_candidate, top_tie_count = _select_best_for_rule(
                candidate_payloads=candidate_payloads,
                rule=rule,
            )
            selection_keys_by_rule[rule].append(_candidate_key(best_candidate))
            selection_strategies_by_rule[rule].append(best_candidate['imbalance_strategy'])
            tie_counts_by_rule[rule].append(int(top_tie_count))
            outer_prediction = _fit_outer_selected_candidate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                groups_test=groups_test,
                clf_name=clf_name,
                clf_template=clf_template,
                params=best_candidate['params'],
                imbalance_strategy=best_candidate['imbalance_strategy'],
                rule=rule,
            )
            outer_prediction['outer_fold_subject'] = held_out_subject
            outer_predictions_by_rule[rule].append(outer_prediction)

    full_source_candidate_payloads: list[dict[str, Any]] = []
    for strategy in STRATEGY_POLICY[clf_name]:
        for params in param_candidates:
            outputs = _evaluate_candidate_outputs(
                X_train=X,
                y_train=y,
                groups_train=groups,
                clf_name=clf_name,
                clf_template=clf_template,
                params=params,
                imbalance_strategy=strategy,
            )
            full_source_candidate_payloads.append({
                'imbalance_strategy': strategy,
                'params': params,
                **outputs,
            })

    full_source_selection: dict[str, Any] = {}
    for rule in RULES:
        if rule == 'mean_decision_score' and clf_name != 'svm':
            continue
        best_candidate, top_tie_count = _select_best_for_rule(
            candidate_payloads=full_source_candidate_payloads,
            rule=rule,
        )
        full_source_selection[rule] = {
            'selected_candidate_key': _candidate_key(best_candidate),
            'selected_imbalance_strategy': best_candidate['imbalance_strategy'],
            'selected_params': best_candidate['params'],
            'top_tie_count': int(top_tie_count),
        }

    payload = _build_shard_payload_from_components(
        condition=condition,
        clf_name=clf_name,
        n_outer_folds_total=int(len(np.unique(groups))),
        selected_outer_subjects=selected_outer_subjects,
        param_candidates=param_candidates,
        strategy_policy=STRATEGY_POLICY[clf_name],
        used_reduced_grid=(scope == 'quick'),
        used_reduced_outer_folds=(scope == 'quick'),
        outer_predictions_by_rule=outer_predictions_by_rule,
        selection_keys_by_rule=selection_keys_by_rule,
        selection_strategies_by_rule=selection_strategies_by_rule,
        tie_counts_by_rule=tie_counts_by_rule,
        full_source_selection=full_source_selection,
    )
    if execution_overrides is not None:
        payload['execution_overrides'] = dict(execution_overrides)
    return payload


def _assemble_subject_aggregation_report(
    *,
    shard_payloads: list[dict[str, Any]],
    scope: str,
    input_paths: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        'scope': scope,
        'grid_policy': 'full_real_data' if scope == 'full' else 'reduced_real_data',
        'fold_sampling_policy': (
            {
                'name': 'full_outer_loso',
                'warning': None,
            }
            if scope == 'full' else
            {
                'name': 'deterministic_class_balanced_outer_subset',
                'max_outer_folds_per_label': MAX_OUTER_FOLDS_PER_LABEL,
                'warning': (
                    'Quick mode uses a deterministic representative fold subset rather than '
                    'full outer LOSO and must not be used to lock the final aggregation rule.'
                ),
            }
        ),
        'conditions_represented': list(CONDITIONS),
        'classifiers_represented': (
            list(CLF_ORDER)
            if scope == 'full' else
            sorted({
                clf_name
                for clf_names in DIAGNOSTIC_CLASSIFIERS_BY_CONDITION.values()
                for clf_name in clf_names
            })
        ),
        'diagnostic_classifier_subset_by_condition': (
            {condition: list(CLF_ORDER) for condition in CONDITIONS}
            if scope == 'full' else
            {
                condition: list(clf_names)
                for condition, clf_names in DIAGNOSTIC_CLASSIFIERS_BY_CONDITION.items()
            }
        ),
        'rules_requested': list(RULES),
        'rule_support': {
            'mean_probability': list(CLF_ORDER),
            'median_probability': list(CLF_ORDER),
            'majority_vote': list(CLF_ORDER),
            'mean_decision_score': ['svm'],
        },
        'per_condition': {condition: {} for condition in CONDITIONS},
        'source_best_classifier_by_rule': {},
        'input_paths': input_paths,
        'remaining_uncertainty': [],
    }
    if scope == 'quick':
        report['remaining_uncertainty'].extend([
            'Quick mode uses a restricted classifier subset per condition.',
            'Quick mode uses reduced parameter contrasts and reduced outer-fold coverage.',
        ])

    source_best_scores_by_rule: dict[str, dict[str, dict[str, float]]] = {
        condition: {rule: {} for rule in RULES}
        for condition in CONDITIONS
    }
    for shard in shard_payloads:
        condition = shard['condition']
        clf_name = shard['classifier']
        report['per_condition'][condition][clf_name] = shard['report']
        for rule, score in shard['source_best_scores_by_rule'].items():
            source_best_scores_by_rule[condition][rule][clf_name] = float(score)

    for condition in CONDITIONS:
        report['source_best_classifier_by_rule'][condition] = {}
        for rule, scores in source_best_scores_by_rule[condition].items():
            if not scores:
                continue
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            report['source_best_classifier_by_rule'][condition][rule] = {
                'best_classifier': ordered[0][0],
                'scores': {clf_name: round(score, 6) for clf_name, score in ordered},
                'is_restricted_subset': scope != 'full',
            }

    tie_frequency_by_rule: dict[str, float] = {}
    prediction_disagreement_by_rule: dict[str, float] = {}
    selection_disagreement_by_rule: dict[str, float] = {}
    strategy_disagreement_by_rule: dict[str, float] = {}
    full_source_selection_changes_by_rule: dict[str, dict[str, dict[str, bool]]] = {}
    source_best_changes_by_rule: dict[str, dict[str, bool]] = {}

    for rule in RULES:
        tie_flags: list[int] = []
        pred_disagreements: list[float] = []
        selection_changes = 0
        strategy_changes = 0
        selection_total = 0
        strategy_total = 0
        for condition, condition_report in report['per_condition'].items():
            source_best_changes_by_rule.setdefault(condition, {})
            full_source_selection_changes_by_rule.setdefault(condition, {})
            mean_best = (
                report['source_best_classifier_by_rule'][condition]
                .get('mean_probability', {})
                .get('best_classifier')
            )
            rule_best = (
                report['source_best_classifier_by_rule'][condition]
                .get(rule, {})
                .get('best_classifier')
            )
            if rule != 'mean_probability' and rule_best is not None and mean_best is not None:
                source_best_changes_by_rule[condition][rule] = (rule_best != mean_best)
            for clf_name, clf_report in condition_report.items():
                rule_summary = clf_report['rule_summary'].get(rule)
                if rule_summary is None:
                    continue
                tie_flags.extend(
                    int(tie_count > 1)
                    for tie_count in rule_summary['outer_fold_top_tie_counts']
                )
                if rule != 'mean_probability':
                    pred_val = clf_report['prediction_disagreement_vs_mean_probability'].get(rule)
                    if pred_val is not None:
                        pred_disagreements.append(float(pred_val))
                    selection_changes += int(
                        clf_report['selection_disagreements_vs_mean_probability'].get(rule, 0)
                    )
                    strategy_changes += int(
                        clf_report['strategy_disagreements_vs_mean_probability'].get(rule, 0)
                    )
                    selection_total += len(rule_summary['outer_fold_selected_candidate_keys'])
                    strategy_total += len(rule_summary['outer_fold_selected_strategies'])
                    full_source_selection_changes_by_rule[condition].setdefault(clf_name, {})
                    full_source_selection_changes_by_rule[condition][clf_name][rule] = bool(
                        clf_report['full_source_selection_changes_vs_mean_probability'].get(rule, False)
                    )
        if tie_flags:
            tie_frequency_by_rule[rule] = round(float(np.mean(tie_flags)), 6)
        if rule != 'mean_probability':
            prediction_disagreement_by_rule[rule] = (
                round(float(np.mean(pred_disagreements)), 6) if pred_disagreements else 0.0
            )
            selection_disagreement_by_rule[rule] = round(
                float(selection_changes / max(selection_total, 1)),
                6,
            )
            strategy_disagreement_by_rule[rule] = round(
                float(strategy_changes / max(strategy_total, 1)),
                6,
            )

    if scope == 'full':
        mean_vs_majority_source_best_changed = any(
            source_best_changes_by_rule.get(condition, {}).get('majority_vote', False)
            for condition in CONDITIONS
        )
        recommended_rule = (
            'majority_vote'
            if mean_vs_majority_source_best_changed else
            'mean_probability'
        )
    else:
        recommended_rule = 'restricted_scope_no_global_lock'
        report['remaining_uncertainty'].append(
            'Quick-mode source-best recommendations are restricted and must not be treated as global.'
        )

    report['summary'] = {
        'tie_frequency_by_rule': tie_frequency_by_rule,
        'prediction_disagreement_vs_mean_probability': prediction_disagreement_by_rule,
        'selection_disagreement_vs_mean_probability': selection_disagreement_by_rule,
        'strategy_disagreement_vs_mean_probability': strategy_disagreement_by_rule,
        'full_source_selection_changes_vs_mean_probability': full_source_selection_changes_by_rule,
        'source_best_classifier_changes_vs_mean_probability': source_best_changes_by_rule,
        'recommended_primary_aggregation_rule': recommended_rule,
    }
    return report


def build_subject_aggregation_report(
    *,
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
    scope: str,
    results_v3_dir: Path | None = None,
) -> dict[str, Any]:
    if scope not in SCOPES:
        raise ValueError(f'Unsupported scope {scope!r}; expected one of {SCOPES}.')
    v3_within = _load_v3_within_results(results_v3_dir)
    shard_payloads: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for clf_name in _classifiers_for_condition(condition, scope):
            shard_payloads.append(
                _evaluate_condition_classifier(
                    feature_df=feature_df,
                    partition=partition,
                    condition=condition,
                    clf_name=clf_name,
                    scope=scope,
                    v3_within=v3_within,
                )
            )
    return _assemble_subject_aggregation_report(
        shard_payloads=shard_payloads,
        scope=scope,
        input_paths={},
    )


def run_local(scope: str = 'quick') -> dict[str, Any]:
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    processed_dir = Path(scaffold['processed_preflight'])
    results_dir = Path(scaffold['results_preflight'])
    df, partition, input_paths = _reference_artifacts(
        processed_dir=processed_dir,
        allow_build=True,
    )
    report = build_subject_aggregation_report(
        feature_df=df,
        partition=partition,
        scope=scope,
        results_v3_dir=REPO_ROOT / 'experiments' / 'results' / 'v3',
    )
    report['input_paths'] = input_paths
    output_path = results_dir / _report_output_name(scope)
    atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2))
    print(f'\nWrote {output_path}')
    return {'report_path': str(output_path), 'report': report}


def _feature_df_sha256(feature_df: pl.DataFrame) -> str:
    return sha256_text(feature_df.write_csv())


def _partition_sha256(partition: dict[str, Any]) -> str:
    return sha256_text(_stable_json(partition))


def _build_fragment_context_from_objects(
    *,
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
    condition: str,
    clf_name: str,
    v3_within: dict[str, dict[str, Any]],
    scope: str = 'full',
    features_path: str = 'in_memory_features.csv',
    partition_path: str = 'in_memory_partition.json',
    manifest_path: str = 'in_memory_manifest.json',
    feature_matrix_hash: str | None = None,
    partition_hash: str | None = None,
    preprocessing_manifest_hash: str | None = None,
    v3_source_result_hash: str | None = None,
) -> dict[str, Any]:
    feature_cols = get_feature_cols('v4')
    configs = get_classifier_configs()
    strategy_policy = STRATEGY_POLICY[clf_name]
    param_candidates = _param_candidates(
        condition=condition,
        clf_name=clf_name,
        param_grid=configs[clf_name]['param_grid'],
        v3_within=v3_within,
        scope=scope,
    )
    control_a = partition['control_A']
    pool = feature_df.filter(
        (pl.col('condition') == condition) | pl.col('subject_id').is_in(control_a)
    )
    X = pool.select(feature_cols).to_numpy().astype(np.float64)
    y = pool['label'].to_numpy().astype(int)
    groups = pool['subject_id'].to_numpy().astype(str)
    selected_outer_subjects = _select_outer_subjects(
        pool=pool,
        condition=condition,
        control_a=control_a,
        scope=scope,
    )
    return {
        'feature_df': feature_df,
        'partition': partition,
        'condition': condition,
        'clf_name': clf_name,
        'scope': scope,
        'feature_cols': feature_cols,
        'configs': configs,
        'strategy_policy': strategy_policy,
        'param_candidates': param_candidates,
        'grid_hash': _grid_hash(
            clf_name=clf_name,
            param_candidates=param_candidates,
            strategy_policy=strategy_policy,
        ),
        'pool': pool,
        'X': X,
        'y': y,
        'groups': groups,
        'selected_outer_subjects': selected_outer_subjects,
        'n_outer_folds_total': int(len(np.unique(groups))),
        'feature_matrix_hash': feature_matrix_hash or _feature_df_sha256(feature_df),
        'partition_hash': partition_hash or _partition_sha256(partition),
        'preprocessing_manifest_hash': (
            preprocessing_manifest_hash or
            sha256_text(_stable_json({'features_path': features_path, 'partition_path': partition_path}))
        ),
        'v3_source_result_hash': (
            v3_source_result_hash or
            sha256_text(_stable_json(v3_within[condition]))
        ),
        'features_path': features_path,
        'partition_path': partition_path,
        'manifest_path': manifest_path,
        'rules': _supported_rules_for_classifier(clf_name),
    }


def _build_fragment_context_from_local_artifacts(
    *,
    condition: str,
    clf_name: str,
) -> dict[str, Any]:
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    processed_dir = Path(scaffold['processed_preflight'])
    feature_df, partition, input_paths = _reference_artifacts(
        processed_dir=processed_dir,
        allow_build=False,
    )
    v3_within = _load_v3_within_results(REPO_ROOT / 'experiments' / 'results' / 'v3')
    return _build_fragment_context_from_objects(
        feature_df=feature_df,
        partition=partition,
        condition=condition,
        clf_name=clf_name,
        v3_within=v3_within,
        scope='full',
        features_path=input_paths['features_path'],
        partition_path=input_paths['partition_path'],
        manifest_path=input_paths['manifest_path'],
        feature_matrix_hash=sha256_file(input_paths['features_path']),
        partition_hash=sha256_file(input_paths['partition_path']),
        preprocessing_manifest_hash=sha256_file(input_paths['manifest_path']),
        v3_source_result_hash=sha256_file(
            REPO_ROOT / 'experiments' / 'results' / 'v3' / f'{condition}_results_v3.json'
        ),
    )


def _build_fragment_context_from_remote_artifacts(
    *,
    processed_dir: Path,
    results_v3_dir: Path,
    condition: str,
    clf_name: str,
) -> dict[str, Any]:
    feature_df, partition, input_paths = _reference_artifacts(
        processed_dir=processed_dir,
        allow_build=False,
    )
    v3_within = _load_v3_within_results(results_v3_dir)
    return _build_fragment_context_from_objects(
        feature_df=feature_df,
        partition=partition,
        condition=condition,
        clf_name=clf_name,
        v3_within=v3_within,
        scope='full',
        features_path=input_paths['features_path'],
        partition_path=input_paths['partition_path'],
        manifest_path=input_paths['manifest_path'],
        feature_matrix_hash=sha256_file(input_paths['features_path']),
        partition_hash=sha256_file(input_paths['partition_path']),
        preprocessing_manifest_hash=sha256_file(input_paths['manifest_path']),
        v3_source_result_hash=sha256_file(results_v3_dir / f'{condition}_results_v3.json'),
    )


def _candidate_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(context['param_candidates']):
        for imbalance_strategy in context['strategy_policy']:
            specs.append({
                'candidate_index': int(candidate_index),
                'candidate_hash': _candidate_hash(params),
                'params': params,
                'imbalance_strategy': imbalance_strategy,
            })
    return specs


def _subject_order_from_groups(groups: np.ndarray) -> list[str]:
    return list(dict.fromkeys(groups.astype(str).tolist()))


def _outer_train_test_arrays(
    context: dict[str, Any],
    *,
    outer_fold_subject: str,
) -> dict[str, Any]:
    groups = context['groups']
    test_mask = groups == outer_fold_subject
    if not np.any(test_mask):
        raise ValueError(
            f'Outer held-out subject {outer_fold_subject!r} is not present for '
            f'{context["condition"]}/{context["clf_name"]}.'
        )
    train_mask = ~test_mask
    return {
        'X_train': context['X'][train_mask],
        'y_train': context['y'][train_mask],
        'groups_train': groups[train_mask],
        'X_test': context['X'][test_mask],
        'y_test': context['y'][test_mask],
        'groups_test': groups[test_mask],
    }


def _fragment_identity(
    context: dict[str, Any],
    *,
    fragment_kind: str,
    outer_fold_subject: str | None = None,
    full_source_stage: bool = False,
    candidate_index: int | None = None,
    candidate_hash: str | None = None,
    imbalance_strategy: str | None = None,
    inner_held_out_subject: str | None = None,
) -> dict[str, Any]:
    return {
        'scope': 'full',
        'condition': context['condition'],
        'classifier': context['clf_name'],
        'fragment_kind': fragment_kind,
        'outer_fold_subject': outer_fold_subject,
        'full_source_stage': bool(full_source_stage),
        'candidate_index': candidate_index,
        'candidate_hash': candidate_hash,
        'imbalance_strategy': imbalance_strategy,
        'inner_held_out_subject': inner_held_out_subject,
        'aggregation_rules': list(context['rules']),
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'v3_source_result_hash': context['v3_source_result_hash'],
        'grid_hash': context['grid_hash'],
        'tie_break_rule': DEFAULT_TIE_BREAK_RULE,
        'code_schema_version': FRAGMENT_SCHEMA_VERSION,
        'random_seed': FRAGMENT_RANDOM_SEED,
    }


def _validate_fragment_payload(
    payload: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
    required_keys: tuple[str, ...],
) -> None:
    if payload.get('schema_version') != FRAGMENT_SCHEMA_VERSION:
        raise ValueError(
            f'Fragment schema mismatch: expected {FRAGMENT_SCHEMA_VERSION}, '
            f'got {payload.get("schema_version")!r}.'
        )
    identity = payload.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('Fragment is missing required identity object.')
    for key, expected_value in expected_identity.items():
        if identity.get(key) != expected_value:
            raise ValueError(
                f'Fragment identity mismatch for {key!r}: expected '
                f'{expected_value!r}, got {identity.get(key)!r}.'
            )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f'Fragment payload is missing required keys: {missing}')
    payload_sha256 = payload.get('payload_sha256')
    if not isinstance(payload_sha256, str) or not payload_sha256:
        raise ValueError('Fragment payload is missing required payload_sha256 digest.')
    materialized = dict(payload)
    materialized.pop('payload_sha256', None)
    expected_sha256 = sha256_text(_stable_json(materialized))
    if payload_sha256 != expected_sha256:
        raise ValueError(
            'Fragment payload digest mismatch: expected '
            f'{expected_sha256}, got {payload_sha256}.'
        )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _volume_listdir_or_empty(
    volume_obj: Any,
    prefix: str,
    *,
    missing_prefix_errors: tuple[type[BaseException], ...],
) -> list[Any]:
    try:
        return volume_obj.listdir(prefix, recursive=True)
    except missing_prefix_errors:
        return []


def _write_fragment_payload(path: Path, payload: dict[str, Any]) -> None:
    materialized = dict(payload)
    materialized['completed_at_utc'] = utc_now_iso()
    materialized['payload_sha256'] = sha256_text(_stable_json(materialized))
    atomic_write_json(path, materialized)


def _rf_template_for_stage(context: dict[str, Any], *, n_jobs: int) -> Any:
    clf_template = clone(context['configs']['rf']['clf'])
    clf_template.set_params(n_jobs=n_jobs, random_state=42)
    return clf_template


def _fragment_candidate_payloads_from_score_fragments(
    fragments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_payloads: list[dict[str, Any]] = []
    for fragment in fragments:
        candidate_payloads.append({
            'imbalance_strategy': fragment['imbalance_strategy'],
            'params': fragment['params'],
            'scores_by_rule': fragment['scores_by_rule'],
        })
    return candidate_payloads


def _run_rf_outer_candidate_fragment_local(
    *,
    context: dict[str, Any],
    outer_fold_subject: str,
    candidate_spec: dict[str, Any],
    storage_root: Path,
) -> dict[str, Any]:
    relpath = _rf_outer_candidate_relpath(
        condition=context['condition'],
        outer_fold_subject=outer_fold_subject,
        imbalance_strategy=candidate_spec['imbalance_strategy'],
        candidate_hash=candidate_spec['candidate_hash'],
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind='rf_outer_candidate',
        outer_fold_subject=outer_fold_subject,
        candidate_index=candidate_spec['candidate_index'],
        candidate_hash=candidate_spec['candidate_hash'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    split = _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject)
    outputs = _evaluate_candidate_outputs(
        X_train=split['X_train'],
        y_train=split['y_train'],
        groups_train=split['groups_train'],
        clf_name='rf',
        clf_template=_rf_template_for_stage(context, n_jobs=1),
        params=candidate_spec['params'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
    )
    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'params': candidate_spec['params'],
        'imbalance_strategy': candidate_spec['imbalance_strategy'],
        'scores_by_rule': outputs['scores_by_rule'],
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _run_rf_outer_result_fragment_local(
    *,
    context: dict[str, Any],
    outer_fold_subject: str,
    storage_root: Path,
) -> dict[str, Any]:
    relpath = _rf_outer_result_relpath(
        condition=context['condition'],
        outer_fold_subject=outer_fold_subject,
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind='rf_outer_result',
        outer_fold_subject=outer_fold_subject,
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=('per_rule',),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    fragments: list[dict[str, Any]] = []
    missing_relpaths: list[str] = []
    for candidate_spec in _candidate_specs(context):
        candidate_relpath = _rf_outer_candidate_relpath(
            condition=context['condition'],
            outer_fold_subject=outer_fold_subject,
            imbalance_strategy=candidate_spec['imbalance_strategy'],
            candidate_hash=candidate_spec['candidate_hash'],
        )
        candidate_path = storage_root / candidate_relpath
        fragment = _load_json_if_exists(candidate_path)
        if fragment is None:
            missing_relpaths.append(candidate_relpath)
            continue
        _validate_fragment_payload(
            fragment,
            expected_identity=_fragment_identity(
                context,
                fragment_kind='rf_outer_candidate',
                outer_fold_subject=outer_fold_subject,
                candidate_index=candidate_spec['candidate_index'],
                candidate_hash=candidate_spec['candidate_hash'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
            ),
            required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
        )
        fragments.append(fragment)
    if missing_relpaths:
        raise FileNotFoundError(
            f'Missing RF outer candidate fragments for {context["condition"]}/rf '
            f'held_out={outer_fold_subject}: {missing_relpaths[:5]}'
        )

    candidate_payloads = _fragment_candidate_payloads_from_score_fragments(fragments)
    split = _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject)
    per_rule: dict[str, Any] = {}
    for rule in _supported_rules_for_classifier('rf'):
        best_candidate, top_tie_count = _select_best_for_rule(
            candidate_payloads=candidate_payloads,
            rule=rule,
        )
        outer_prediction = _fit_outer_selected_candidate(
            X_train=split['X_train'],
            y_train=split['y_train'],
            X_test=split['X_test'],
            y_test=split['y_test'],
            groups_test=split['groups_test'],
            clf_name='rf',
            clf_template=_rf_template_for_stage(context, n_jobs=2),
            params=best_candidate['params'],
            imbalance_strategy=best_candidate['imbalance_strategy'],
            rule=rule,
        )
        outer_prediction['outer_fold_subject'] = outer_fold_subject
        per_rule[rule] = {
            'selected_candidate_key': _candidate_key(best_candidate),
            'selected_imbalance_strategy': best_candidate['imbalance_strategy'],
            'selected_params': best_candidate['params'],
            'top_tie_count': int(top_tie_count),
            'outer_prediction': outer_prediction,
        }

    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'per_rule': per_rule,
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _run_rf_full_source_candidate_fragment_local(
    *,
    context: dict[str, Any],
    candidate_spec: dict[str, Any],
    storage_root: Path,
) -> dict[str, Any]:
    relpath = _rf_full_source_candidate_relpath(
        condition=context['condition'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
        candidate_hash=candidate_spec['candidate_hash'],
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind='rf_full_source_candidate',
        full_source_stage=True,
        candidate_index=candidate_spec['candidate_index'],
        candidate_hash=candidate_spec['candidate_hash'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    outputs = _evaluate_candidate_outputs(
        X_train=context['X'],
        y_train=context['y'],
        groups_train=context['groups'],
        clf_name='rf',
        clf_template=_rf_template_for_stage(context, n_jobs=1),
        params=candidate_spec['params'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
    )
    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'params': candidate_spec['params'],
        'imbalance_strategy': candidate_spec['imbalance_strategy'],
        'scores_by_rule': outputs['scores_by_rule'],
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _run_svm_inner_fragment_local(
    *,
    context: dict[str, Any],
    outer_fold_subject: str,
    candidate_spec: dict[str, Any],
    inner_held_out_subject: str,
    full_source_stage: bool,
    storage_root: Path,
) -> dict[str, Any]:
    stage_name = 'full_source_inner' if full_source_stage else 'outer_inner'
    identity_kind = 'svm_full_source_inner' if full_source_stage else 'svm_outer_inner'
    relpath = _svm_inner_fragment_relpath(
        condition=context['condition'],
        outer_fold_subject=(
            '__full_source__'
            if full_source_stage else outer_fold_subject
        ),
        stage_name=stage_name,
        inner_held_out_subject=inner_held_out_subject,
        imbalance_strategy=candidate_spec['imbalance_strategy'],
        candidate_hash=candidate_spec['candidate_hash'],
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind=identity_kind,
        outer_fold_subject=None if full_source_stage else outer_fold_subject,
        full_source_stage=full_source_stage,
        candidate_index=candidate_spec['candidate_index'],
        candidate_hash=candidate_spec['candidate_hash'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
        inner_held_out_subject=inner_held_out_subject,
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=(
                    'params',
                    'imbalance_strategy',
                    'row_true',
                    'row_pred',
                    'row_prob',
                    'row_subjects',
                ),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    if full_source_stage:
        X_train = context['X']
        y_train = context['y']
        groups_train = context['groups']
    else:
        split = _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject)
        X_train = split['X_train']
        y_train = split['y_train']
        groups_train = split['groups_train']

    inner_mask = groups_train == inner_held_out_subject
    if not np.any(inner_mask):
        raise ValueError(
            f'Inner held-out subject {inner_held_out_subject!r} not present for '
            f'{context["condition"]}/svm outer={outer_fold_subject!r} full_source={full_source_stage}.'
        )
    inner_train_mask = ~inner_mask

    clf_variant = _configure_classifier_for_resampling(
        'svm',
        clone(context['configs']['svm']['clf']),
        candidate_spec['imbalance_strategy'],
    )
    pipeline = build_pipeline(
        'svm',
        clf_variant,
        imbalance_strategy=candidate_spec['imbalance_strategy'],
    )
    pipeline.set_params(**candidate_spec['params'])
    fit_kwargs = _get_fit_kwargs('svm', y_train[inner_train_mask], candidate_spec['imbalance_strategy'])
    pipeline.fit(X_train[inner_train_mask], y_train[inner_train_mask], **fit_kwargs)

    row_true = np.asarray(y_train[inner_mask], dtype=int)
    row_pred = np.asarray(pipeline.predict(X_train[inner_mask]), dtype=int)
    row_prob = np.asarray(pipeline.predict_proba(X_train[inner_mask])[:, 1], dtype=float)
    row_subjects = np.asarray(groups_train[inner_mask], dtype=str)
    row_decision = np.asarray(pipeline.decision_function(X_train[inner_mask]), dtype=float)
    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'params': candidate_spec['params'],
        'imbalance_strategy': candidate_spec['imbalance_strategy'],
        'row_true': row_true.tolist(),
        'row_pred': row_pred.tolist(),
        'row_prob': np.round(row_prob, 12).tolist(),
        'row_subjects': row_subjects.tolist(),
        'row_decision': np.round(row_decision, 12).tolist(),
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _run_svm_candidate_summary_fragment_local(
    *,
    context: dict[str, Any],
    outer_fold_subject: str | None,
    candidate_spec: dict[str, Any],
    full_source_stage: bool,
    storage_root: Path,
) -> dict[str, Any]:
    stage_name = (
        'full_source_candidate_summaries'
        if full_source_stage else
        'outer_candidate_summaries'
    )
    fragment_kind = (
        'svm_full_source_candidate_summary'
        if full_source_stage else
        'svm_outer_candidate_summary'
    )
    relpath = _svm_candidate_summary_relpath(
        condition=context['condition'],
        outer_fold_subject=outer_fold_subject,
        stage_name=stage_name,
        imbalance_strategy=candidate_spec['imbalance_strategy'],
        candidate_hash=candidate_spec['candidate_hash'],
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind=fragment_kind,
        outer_fold_subject=outer_fold_subject,
        full_source_stage=full_source_stage,
        candidate_index=candidate_spec['candidate_index'],
        candidate_hash=candidate_spec['candidate_hash'],
        imbalance_strategy=candidate_spec['imbalance_strategy'],
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    if full_source_stage:
        groups_for_order = context['groups']
        inner_subjects = _subject_order_from_groups(groups_for_order)
        outer_subject_key = '__full_source__'
    else:
        split = _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject or '')
        groups_for_order = split['groups_train']
        inner_subjects = _subject_order_from_groups(groups_for_order)
        outer_subject_key = outer_fold_subject or ''

    row_true_parts: list[np.ndarray] = []
    row_pred_parts: list[np.ndarray] = []
    row_prob_parts: list[np.ndarray] = []
    row_subject_parts: list[np.ndarray] = []
    row_decision_parts: list[np.ndarray] = []

    missing_relpaths: list[str] = []
    for inner_subject in inner_subjects:
        inner_relpath = _svm_inner_fragment_relpath(
            condition=context['condition'],
            outer_fold_subject=outer_subject_key,
            stage_name='full_source_inner' if full_source_stage else 'outer_inner',
            inner_held_out_subject=inner_subject,
            imbalance_strategy=candidate_spec['imbalance_strategy'],
            candidate_hash=candidate_spec['candidate_hash'],
        )
        inner_path = storage_root / inner_relpath
        fragment = _load_json_if_exists(inner_path)
        if fragment is None:
            missing_relpaths.append(inner_relpath)
            continue
        _validate_fragment_payload(
            fragment,
            expected_identity=_fragment_identity(
                context,
                fragment_kind=(
                    'svm_full_source_inner'
                    if full_source_stage else
                    'svm_outer_inner'
                ),
                outer_fold_subject=outer_fold_subject,
                full_source_stage=full_source_stage,
                candidate_index=candidate_spec['candidate_index'],
                candidate_hash=candidate_spec['candidate_hash'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
                inner_held_out_subject=inner_subject,
            ),
            required_keys=('params', 'imbalance_strategy', 'row_true', 'row_pred', 'row_prob', 'row_subjects'),
        )
        row_true_parts.append(np.asarray(fragment['row_true'], dtype=int))
        row_pred_parts.append(np.asarray(fragment['row_pred'], dtype=int))
        row_prob_parts.append(np.asarray(fragment['row_prob'], dtype=float))
        row_subject_parts.append(np.asarray(fragment['row_subjects'], dtype=str))
        row_decision_parts.append(np.asarray(fragment['row_decision'], dtype=float))

    if missing_relpaths:
        raise FileNotFoundError(
            f'Missing SVM inner fragments for {context["condition"]}/svm '
            f'outer={outer_fold_subject!r} full_source={full_source_stage}: {missing_relpaths[:5]}'
        )

    row_true = np.concatenate(row_true_parts)
    row_pred = np.concatenate(row_pred_parts)
    row_prob = np.concatenate(row_prob_parts)
    row_subjects = np.concatenate(row_subject_parts)
    row_decision = np.concatenate(row_decision_parts)
    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'params': candidate_spec['params'],
        'imbalance_strategy': candidate_spec['imbalance_strategy'],
        'scores_by_rule': _scores_by_rule_from_row_outputs(
            row_true=row_true,
            row_pred=row_pred,
            row_prob=row_prob,
            row_subjects=row_subjects,
            row_decision=row_decision,
        ),
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _run_svm_outer_result_fragment_local(
    *,
    context: dict[str, Any],
    outer_fold_subject: str,
    storage_root: Path,
) -> dict[str, Any]:
    relpath = _outer_result_relpath(
        condition=context['condition'],
        clf_name='svm',
        outer_fold_subject=outer_fold_subject,
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind='svm_outer_result',
        outer_fold_subject=outer_fold_subject,
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=('per_rule',),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    fragments: list[dict[str, Any]] = []
    missing_relpaths: list[str] = []
    for candidate_spec in _candidate_specs(context):
        rel = _svm_candidate_summary_relpath(
            condition=context['condition'],
            outer_fold_subject=outer_fold_subject,
            stage_name='outer_candidate_summaries',
            imbalance_strategy=candidate_spec['imbalance_strategy'],
            candidate_hash=candidate_spec['candidate_hash'],
        )
        fragment = _load_json_if_exists(storage_root / rel)
        if fragment is None:
            missing_relpaths.append(rel)
            continue
        _validate_fragment_payload(
            fragment,
            expected_identity=_fragment_identity(
                context,
                fragment_kind='svm_outer_candidate_summary',
                outer_fold_subject=outer_fold_subject,
                candidate_index=candidate_spec['candidate_index'],
                candidate_hash=candidate_spec['candidate_hash'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
            ),
            required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
        )
        fragments.append(fragment)
    if missing_relpaths:
        raise FileNotFoundError(
            f'Missing SVM outer candidate summaries for {context["condition"]}/svm '
            f'held_out={outer_fold_subject}: {missing_relpaths[:5]}'
        )

    candidate_payloads = _fragment_candidate_payloads_from_score_fragments(fragments)
    split = _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject)
    per_rule: dict[str, Any] = {}
    for rule in _supported_rules_for_classifier('svm'):
        best_candidate, top_tie_count = _select_best_for_rule(
            candidate_payloads=candidate_payloads,
            rule=rule,
        )
        outer_prediction = _fit_outer_selected_candidate(
            X_train=split['X_train'],
            y_train=split['y_train'],
            X_test=split['X_test'],
            y_test=split['y_test'],
            groups_test=split['groups_test'],
            clf_name='svm',
            clf_template=clone(context['configs']['svm']['clf']),
            params=best_candidate['params'],
            imbalance_strategy=best_candidate['imbalance_strategy'],
            rule=rule,
        )
        outer_prediction['outer_fold_subject'] = outer_fold_subject
        per_rule[rule] = {
            'selected_candidate_key': _candidate_key(best_candidate),
            'selected_imbalance_strategy': best_candidate['imbalance_strategy'],
            'selected_params': best_candidate['params'],
            'top_tie_count': int(top_tie_count),
            'outer_prediction': outer_prediction,
        }

    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'per_rule': per_rule,
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _run_full_source_selection_fragment_local(
    *,
    context: dict[str, Any],
    storage_root: Path,
) -> dict[str, Any]:
    relpath = _full_source_selection_relpath(
        condition=context['condition'],
        clf_name=context['clf_name'],
    )
    output_path = storage_root / relpath
    expected_identity = _fragment_identity(
        context,
        fragment_kind='full_source_selection',
        full_source_stage=True,
    )
    existing = _load_json_if_exists(output_path)
    if existing is not None:
        try:
            _validate_fragment_payload(
                existing,
                expected_identity=expected_identity,
                required_keys=('per_rule',),
            )
            return {'status': 'skipped_existing', 'relpath': relpath}
        except Exception:
            pass

    fragments: list[dict[str, Any]] = []
    missing_relpaths: list[str] = []
    for candidate_spec in _candidate_specs(context):
        if context['clf_name'] == 'rf':
            rel = _rf_full_source_candidate_relpath(
                condition=context['condition'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
                candidate_hash=candidate_spec['candidate_hash'],
            )
            fragment_kind = 'rf_full_source_candidate'
        else:
            rel = _svm_candidate_summary_relpath(
                condition=context['condition'],
                outer_fold_subject=None,
                stage_name='full_source_candidate_summaries',
                imbalance_strategy=candidate_spec['imbalance_strategy'],
                candidate_hash=candidate_spec['candidate_hash'],
            )
            fragment_kind = 'svm_full_source_candidate_summary'
        fragment = _load_json_if_exists(storage_root / rel)
        if fragment is None:
            missing_relpaths.append(rel)
            continue
        _validate_fragment_payload(
            fragment,
            expected_identity=_fragment_identity(
                context,
                fragment_kind=fragment_kind,
                full_source_stage=True,
                candidate_index=candidate_spec['candidate_index'],
                candidate_hash=candidate_spec['candidate_hash'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
            ),
            required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
        )
        fragments.append(fragment)
    if missing_relpaths:
        raise FileNotFoundError(
            f'Missing full-source candidate fragments for {context["condition"]}/{context["clf_name"]}: '
            f'{missing_relpaths[:5]}'
        )

    candidate_payloads = _fragment_candidate_payloads_from_score_fragments(fragments)
    per_rule: dict[str, Any] = {}
    for rule in context['rules']:
        best_candidate, top_tie_count = _select_best_for_rule(
            candidate_payloads=candidate_payloads,
            rule=rule,
        )
        per_rule[rule] = {
            'selected_candidate_key': _candidate_key(best_candidate),
            'selected_imbalance_strategy': best_candidate['imbalance_strategy'],
            'selected_params': best_candidate['params'],
            'top_tie_count': int(top_tie_count),
        }

    payload = {
        'schema_version': FRAGMENT_SCHEMA_VERSION,
        'identity': expected_identity,
        'per_rule': per_rule,
    }
    _write_fragment_payload(output_path, payload)
    return {'status': 'completed', 'relpath': relpath}


def _assemble_fragmented_full_shard_local(
    *,
    context: dict[str, Any],
    storage_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    output_path = storage_root / _shard_remote_relpath('full', context['condition'], context['clf_name'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        try:
            _validate_shard_payload(
                payload,
                expected_condition=context['condition'],
                expected_classifier=context['clf_name'],
            )
        except Exception as exc:
            if not force:
                raise ValueError(
                    f'Existing full shard at {output_path} is invalid and will not be '
                    'overwritten without force.'
                ) from exc
        else:
            if not force:
                return {'status': 'skipped_existing', 'path': str(output_path)}

    outer_predictions_by_rule: dict[str, list[dict[str, Any]]] = {
        rule: [] for rule in RULES
    }
    selection_keys_by_rule: dict[str, list[str]] = {rule: [] for rule in RULES}
    selection_strategies_by_rule: dict[str, list[str]] = {rule: [] for rule in RULES}
    tie_counts_by_rule: dict[str, list[int]] = {rule: [] for rule in RULES}

    for outer_fold_subject in context['selected_outer_subjects']:
        relpath = _outer_result_relpath(
            condition=context['condition'],
            clf_name=context['clf_name'],
            outer_fold_subject=outer_fold_subject,
        )
        fragment = _load_json_if_exists(storage_root / relpath)
        if fragment is None:
            raise FileNotFoundError(
                f'Missing outer result fragment for {context["condition"]}/{context["clf_name"]} '
                f'held_out={outer_fold_subject}: {relpath}'
            )
        _validate_fragment_payload(
            fragment,
            expected_identity=_fragment_identity(
                context,
                fragment_kind=f'{context["clf_name"]}_outer_result',
                outer_fold_subject=outer_fold_subject,
            ),
            required_keys=('per_rule',),
        )
        for rule in context['rules']:
            per_rule = fragment['per_rule'][rule]
            selection_keys_by_rule[rule].append(per_rule['selected_candidate_key'])
            selection_strategies_by_rule[rule].append(per_rule['selected_imbalance_strategy'])
            tie_counts_by_rule[rule].append(int(per_rule['top_tie_count']))
            outer_predictions_by_rule[rule].append(per_rule['outer_prediction'])

    full_source_fragment = _load_json_if_exists(
        storage_root / _full_source_selection_relpath(
            condition=context['condition'],
            clf_name=context['clf_name'],
        )
    )
    if full_source_fragment is None:
        raise FileNotFoundError(
            f'Missing full-source selection fragment for {context["condition"]}/{context["clf_name"]}.'
        )
    _validate_fragment_payload(
        full_source_fragment,
        expected_identity=_fragment_identity(
            context,
            fragment_kind='full_source_selection',
            full_source_stage=True,
        ),
        required_keys=('per_rule',),
    )
    payload = _build_shard_payload_from_components(
        condition=context['condition'],
        clf_name=context['clf_name'],
        n_outer_folds_total=context['n_outer_folds_total'],
        selected_outer_subjects=context['selected_outer_subjects'],
        param_candidates=context['param_candidates'],
        strategy_policy=context['strategy_policy'],
        used_reduced_grid=(context.get('scope') == 'quick'),
        used_reduced_outer_folds=(context.get('scope') == 'quick'),
        outer_predictions_by_rule=outer_predictions_by_rule,
        selection_keys_by_rule=selection_keys_by_rule,
        selection_strategies_by_rule=selection_strategies_by_rule,
        tie_counts_by_rule=tie_counts_by_rule,
        full_source_selection=full_source_fragment['per_rule'],
    )
    atomic_write_json(output_path, payload)
    return {'status': 'completed', 'path': str(output_path), 'payload': payload}


def _rf_outer_candidate_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for outer_fold_subject in context['selected_outer_subjects']:
        for candidate_spec in _candidate_specs(context):
            specs.append({
                'condition': context['condition'],
                'clf_name': 'rf',
                'outer_fold_subject': outer_fold_subject,
                **candidate_spec,
                'relpath': _rf_outer_candidate_relpath(
                    condition=context['condition'],
                    outer_fold_subject=outer_fold_subject,
                    imbalance_strategy=candidate_spec['imbalance_strategy'],
                    candidate_hash=candidate_spec['candidate_hash'],
                ),
            })
    return specs


def _rf_outer_result_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            'condition': context['condition'],
            'clf_name': 'rf',
            'outer_fold_subject': outer_fold_subject,
            'relpath': _rf_outer_result_relpath(
                condition=context['condition'],
                outer_fold_subject=outer_fold_subject,
            ),
        }
        for outer_fold_subject in context['selected_outer_subjects']
    ]


def _rf_full_source_candidate_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for candidate_spec in _candidate_specs(context):
        specs.append({
            'condition': context['condition'],
            'clf_name': 'rf',
            **candidate_spec,
            'relpath': _rf_full_source_candidate_relpath(
                condition=context['condition'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
                candidate_hash=candidate_spec['candidate_hash'],
            ),
        })
    return specs


def _svm_outer_inner_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for outer_fold_subject in context['selected_outer_subjects']:
        inner_subjects = _subject_order_from_groups(
            _outer_train_test_arrays(context, outer_fold_subject=outer_fold_subject)['groups_train']
        )
        for candidate_spec in _candidate_specs(context):
            for inner_held_out_subject in inner_subjects:
                specs.append({
                    'condition': context['condition'],
                    'clf_name': 'svm',
                    'outer_fold_subject': outer_fold_subject,
                    'inner_held_out_subject': inner_held_out_subject,
                    'full_source_stage': False,
                    **candidate_spec,
                    'relpath': _svm_inner_fragment_relpath(
                        condition=context['condition'],
                        outer_fold_subject=outer_fold_subject,
                        stage_name='outer_inner',
                        inner_held_out_subject=inner_held_out_subject,
                        imbalance_strategy=candidate_spec['imbalance_strategy'],
                        candidate_hash=candidate_spec['candidate_hash'],
                    ),
                })
    return specs


def _svm_outer_candidate_summary_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for outer_fold_subject in context['selected_outer_subjects']:
        for candidate_spec in _candidate_specs(context):
            specs.append({
                'condition': context['condition'],
                'clf_name': 'svm',
                'outer_fold_subject': outer_fold_subject,
                'full_source_stage': False,
                **candidate_spec,
                'relpath': _svm_candidate_summary_relpath(
                    condition=context['condition'],
                    outer_fold_subject=outer_fold_subject,
                    stage_name='outer_candidate_summaries',
                    imbalance_strategy=candidate_spec['imbalance_strategy'],
                    candidate_hash=candidate_spec['candidate_hash'],
                ),
            })
    return specs


def _svm_outer_result_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            'condition': context['condition'],
            'clf_name': 'svm',
            'outer_fold_subject': outer_fold_subject,
            'relpath': _outer_result_relpath(
                condition=context['condition'],
                clf_name='svm',
                outer_fold_subject=outer_fold_subject,
            ),
        }
        for outer_fold_subject in context['selected_outer_subjects']
    ]


def _svm_full_source_inner_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    inner_subjects = _subject_order_from_groups(context['groups'])
    for candidate_spec in _candidate_specs(context):
        for inner_held_out_subject in inner_subjects:
            specs.append({
                'condition': context['condition'],
                'clf_name': 'svm',
                'outer_fold_subject': None,
                'inner_held_out_subject': inner_held_out_subject,
                'full_source_stage': True,
                **candidate_spec,
                'relpath': _svm_inner_fragment_relpath(
                    condition=context['condition'],
                    outer_fold_subject='__full_source__',
                    stage_name='full_source_inner',
                    inner_held_out_subject=inner_held_out_subject,
                    imbalance_strategy=candidate_spec['imbalance_strategy'],
                    candidate_hash=candidate_spec['candidate_hash'],
                ),
            })
    return specs


def _svm_full_source_candidate_summary_stage_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for candidate_spec in _candidate_specs(context):
        specs.append({
            'condition': context['condition'],
            'clf_name': 'svm',
            'outer_fold_subject': None,
            'full_source_stage': True,
            **candidate_spec,
            'relpath': _svm_candidate_summary_relpath(
                condition=context['condition'],
                outer_fold_subject=None,
                stage_name='full_source_candidate_summaries',
                imbalance_strategy=candidate_spec['imbalance_strategy'],
                candidate_hash=candidate_spec['candidate_hash'],
            ),
        })
    return specs


def _stage_specs_for_context(context: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    target_kind = _recovery_target_kind(context['condition'], context['clf_name'])
    if target_kind == 'rf':
        return [
            ('rf_outer_candidates', _rf_outer_candidate_stage_specs(context)),
            ('rf_outer_results', _rf_outer_result_stage_specs(context)),
            ('rf_full_source_candidates', _rf_full_source_candidate_stage_specs(context)),
            ('full_source_selection', [{
                'condition': context['condition'],
                'clf_name': 'rf',
                'relpath': _full_source_selection_relpath(
                    condition=context['condition'],
                    clf_name='rf',
                ),
            }]),
        ]
    return [
        ('svm_outer_inner', _svm_outer_inner_stage_specs(context)),
        ('svm_outer_candidate_summaries', _svm_outer_candidate_summary_stage_specs(context)),
        ('svm_outer_results', _svm_outer_result_stage_specs(context)),
        ('svm_full_source_inner', _svm_full_source_inner_stage_specs(context)),
        ('svm_full_source_candidate_summaries', _svm_full_source_candidate_summary_stage_specs(context)),
        ('full_source_selection', [{
            'condition': context['condition'],
            'clf_name': 'svm',
            'relpath': _full_source_selection_relpath(
                condition=context['condition'],
                clf_name='svm',
            ),
        }]),
    ]


def _full_shard_local_path_for_context(context: dict[str, Any], storage_root: Path) -> Path:
    return storage_root / _shard_remote_relpath('full', context['condition'], context['clf_name'])


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[idx:idx + size] for idx in range(0, len(items), max(size, 1))]


def _run_fragmented_target_local(
    *,
    context: dict[str, Any],
    storage_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    target_kind = _recovery_target_kind(context['condition'], context['clf_name'])
    if target_kind == 'rf':
        for spec in _rf_outer_candidate_stage_specs(context):
            _run_rf_outer_candidate_fragment_local(
                context=context,
                outer_fold_subject=spec['outer_fold_subject'],
                candidate_spec=spec,
                storage_root=storage_root,
            )
        for spec in _rf_outer_result_stage_specs(context):
            _run_rf_outer_result_fragment_local(
                context=context,
                outer_fold_subject=spec['outer_fold_subject'],
                storage_root=storage_root,
            )
        for spec in _rf_full_source_candidate_stage_specs(context):
            _run_rf_full_source_candidate_fragment_local(
                context=context,
                candidate_spec=spec,
                storage_root=storage_root,
            )
        _run_full_source_selection_fragment_local(
            context=context,
            storage_root=storage_root,
        )
    else:
        for spec in _svm_outer_inner_stage_specs(context):
            _run_svm_inner_fragment_local(
                context=context,
                outer_fold_subject=spec['outer_fold_subject'],
                candidate_spec=spec,
                inner_held_out_subject=spec['inner_held_out_subject'],
                full_source_stage=False,
                storage_root=storage_root,
            )
        for spec in _svm_outer_candidate_summary_stage_specs(context):
            _run_svm_candidate_summary_fragment_local(
                context=context,
                outer_fold_subject=spec['outer_fold_subject'],
                candidate_spec=spec,
                full_source_stage=False,
                storage_root=storage_root,
            )
        for spec in _svm_outer_result_stage_specs(context):
            _run_svm_outer_result_fragment_local(
                context=context,
                outer_fold_subject=spec['outer_fold_subject'],
                storage_root=storage_root,
            )
        for spec in _svm_full_source_inner_stage_specs(context):
            _run_svm_inner_fragment_local(
                context=context,
                outer_fold_subject='__full_source__',
                candidate_spec=spec,
                inner_held_out_subject=spec['inner_held_out_subject'],
                full_source_stage=True,
                storage_root=storage_root,
            )
        for spec in _svm_full_source_candidate_summary_stage_specs(context):
            _run_svm_candidate_summary_fragment_local(
                context=context,
                outer_fold_subject=None,
                candidate_spec=spec,
                full_source_stage=True,
                storage_root=storage_root,
            )
        _run_full_source_selection_fragment_local(
            context=context,
            storage_root=storage_root,
        )
    return _assemble_fragmented_full_shard_local(
        context=context,
        storage_root=storage_root,
        force=force,
    )

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version='3.12')
        .pip_install_from_requirements('requirements-core.txt')
        .env({'PYTHONPATH': '/root/src'})
        .add_local_dir('src', remote_path='/root/src')
    )
    app = modal.App('gait-transfer-v4-preflight-aggregation', image=image)
    volume = modal.Volume.from_name('gait-results', create_if_missing=True)
    try:
        _MISSING_VOLUME_PREFIX_ERRORS = (
            FileNotFoundError,
            modal.exception.NotFoundError,
        )
    except AttributeError:
        _MISSING_VOLUME_PREFIX_ERRORS = (FileNotFoundError,)

    def _volume_relpath_exists(relpath: str) -> bool:
        try:
            first_chunk = next(iter(volume.read_file(relpath)))
            return len(first_chunk) >= 0
        except (FileNotFoundError, StopIteration):
            return False

    def _volume_list_paths(prefix: str) -> set[str]:
        entries = _volume_listdir_or_empty(
            volume,
            prefix,
            missing_prefix_errors=_MISSING_VOLUME_PREFIX_ERRORS,
        )
        paths: set[str] = set()
        for entry in entries:
            path = getattr(entry, 'path', None)
            if isinstance(path, str):
                paths.add(path)
            elif path is not None:
                paths.add(str(path))
        return paths

    def _volume_read_json(relpath: str) -> dict[str, Any] | None:
        try:
            raw = b''.join(volume.read_file(relpath))
        except FileNotFoundError:
            return None
        if not raw:
            return None
        return json.loads(raw.decode())

    @app.function(
        cpu=8,
        memory=12288,
        timeout=43200,
        volumes={'/results': volume},
        retries=1,
    )
    def run_subject_aggregation_quick_remote() -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        results_dir = Path(REMOTE_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        df, partition, input_paths = _reference_artifacts(
            processed_dir=processed_dir,
            allow_build=False,
        )
        report = build_subject_aggregation_report(
            feature_df=df,
            partition=partition,
            scope='quick',
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
        )
        report['input_paths'] = input_paths
        output_path = results_dir / _report_output_name('quick')
        atomic_write_json(output_path, report)
        volume.commit()
        summary = {
            'remote_output_path': str(output_path),
            'concise_report_summary': {
                'scope': report['scope'],
                'conditions': report['conditions_represented'],
                'classifier_families': report['classifiers_represented'],
                'recommended_rule': report['summary']['recommended_primary_aggregation_rule'],
            },
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{_report_remote_relpath("quick")} '
                f'experiments/results/v4_preflight/{_report_output_name("quick")}'
            ),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=8,
        memory=12288,
        timeout=43200,
        volumes={'/results': volume},
        retries=1,
    )
    def run_subject_aggregation_shard_remote(condition: str, clf_name: str, scope: str = 'full') -> str:
        if scope != 'full':
            raise ValueError('Sharded execution is only supported for full scope.')
        processed_dir = Path('/results/processed_v4_preflight')
        results_dir = Path(REMOTE_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        df, partition, _ = _reference_artifacts(
            processed_dir=processed_dir,
            allow_build=False,
        )
        v3_within = _load_v3_within_results(Path(REMOTE_V3_RESULTS_DIR))
        payload = _evaluate_condition_classifier(
            feature_df=df,
            partition=partition,
            condition=condition,
            clf_name=clf_name,
            scope='full',
            v3_within=v3_within,
        )
        output_path = Path('/results') / _shard_remote_relpath(scope, condition, clf_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, payload)
        volume.commit()
        summary = {
            'remote_output_path': str(output_path),
            'condition': condition,
            'classifier': clf_name,
            'concise_report_summary': {
                'n_outer_folds_total': payload['report']['n_outer_folds_total'],
                'n_outer_folds_sampled': payload['report']['n_outer_folds_sampled'],
            },
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{_shard_remote_relpath(scope, condition, clf_name)} '
                f'experiments/results/v4_preflight/subject_aggregation_shards/{scope}_{condition}_{clf_name}.json'
            ),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=16,
        memory=24576,
        timeout=86400,
        retries=2,
        nonpreemptible=True,
        volumes={'/results': volume},
    )
    def run_subject_aggregation_recovery_remote(condition: str, clf_name: str) -> str:
        if condition not in CONDITIONS:
            raise ValueError(f'Invalid recovery condition {condition!r}; expected one of {CONDITIONS}.')
        if clf_name not in CLF_ORDER:
            raise ValueError(f'Invalid recovery classifier {clf_name!r}; expected one of {CLF_ORDER}.')

        volume.reload()
        output_path = Path('/results') / _shard_remote_relpath('full', condition, clf_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            payload = json.loads(output_path.read_text())
            _validate_shard_payload(
                payload,
                expected_condition=condition,
                expected_classifier=clf_name,
            )
            summary = {
                'status': 'skipped_existing',
                'condition': condition,
                'classifier': clf_name,
                'remote_output_path': str(output_path),
            }
            print(json.dumps(summary, indent=2), flush=True)
            return json.dumps(summary, indent=2)

        execution_overrides = None
        if clf_name == 'rf':
            execution_overrides = {
                'rf_n_jobs': -1,
                'scope': 'recovery_only',
            }
            print(
                f'START recovery condition={condition} classifier={clf_name} rf_n_jobs=-1',
                flush=True,
            )
        else:
            print(
                f'START recovery condition={condition} classifier={clf_name}',
                flush=True,
            )
        processed_dir = Path('/results/processed_v4_preflight')
        results_dir = Path(REMOTE_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        df, partition, _ = _reference_artifacts(
            processed_dir=processed_dir,
            allow_build=False,
        )
        v3_within = _load_v3_within_results(Path(REMOTE_V3_RESULTS_DIR))
        payload = _evaluate_condition_classifier(
            feature_df=df,
            partition=partition,
            condition=condition,
            clf_name=clf_name,
            scope='full',
            v3_within=v3_within,
            execution_overrides=execution_overrides,
        )
        _validate_shard_payload(
            payload,
            expected_condition=condition,
            expected_classifier=clf_name,
        )
        atomic_write_json(output_path, payload)
        volume.commit()
        summary = {
            'status': 'completed',
            'condition': condition,
            'classifier': clf_name,
            'remote_output_path': str(output_path),
            'n_outer_folds_total': payload['report']['n_outer_folds_total'],
            'n_outer_folds_sampled': payload['report']['n_outer_folds_sampled'],
            'execution_overrides': payload.get('execution_overrides'),
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{_shard_remote_relpath("full", condition, clf_name)} '
                f'experiments/results/v4_preflight/subject_aggregation_shards/full_{condition}_{clf_name}.json'
            ),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=1,
        memory=2048,
        max_containers=24,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_rf_outer_candidate_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name='rf',
        )
        summary = _run_rf_outer_candidate_fragment_local(
            context=context,
            outer_fold_subject=spec['outer_fold_subject'],
            candidate_spec=spec,
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=2,
        memory=4096,
        max_containers=8,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_rf_outer_result_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name='rf',
        )
        summary = _run_rf_outer_result_fragment_local(
            context=context,
            outer_fold_subject=spec['outer_fold_subject'],
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=1,
        memory=2048,
        max_containers=24,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_rf_full_source_candidate_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name='rf',
        )
        summary = _run_rf_full_source_candidate_fragment_local(
            context=context,
            candidate_spec=spec,
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=1,
        memory=3072,
        max_containers=16,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_svm_inner_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name='svm',
        )
        summary = _run_svm_inner_fragment_local(
            context=context,
            outer_fold_subject=spec['outer_fold_subject'],
            candidate_spec=spec,
            inner_held_out_subject=spec['inner_held_out_subject'],
            full_source_stage=bool(spec['full_source_stage']),
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=1,
        memory=3072,
        max_containers=8,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_svm_candidate_summary_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name='svm',
        )
        summary = _run_svm_candidate_summary_fragment_local(
            context=context,
            outer_fold_subject=spec['outer_fold_subject'],
            candidate_spec=spec,
            full_source_stage=bool(spec['full_source_stage']),
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=2,
        memory=4096,
        max_containers=8,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_svm_outer_result_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name='svm',
        )
        summary = _run_svm_outer_result_fragment_local(
            context=context,
            outer_fold_subject=spec['outer_fold_subject'],
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=2,
        memory=4096,
        max_containers=4,
        timeout=21600,
        volumes={'/results': volume},
        retries=2,
    )
    def run_full_source_selection_fragment_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name=spec['clf_name'],
        )
        summary = _run_full_source_selection_fragment_local(
            context=context,
            storage_root=Path('/results'),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=2,
        memory=4096,
        max_containers=4,
        timeout=21600,
        volumes={'/results': volume},
        retries=1,
    )
    def assemble_fragmented_full_shard_remote(spec: dict[str, Any]) -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        context = _build_fragment_context_from_remote_artifacts(
            processed_dir=processed_dir,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
            condition=spec['condition'],
            clf_name=spec['clf_name'],
        )
        summary = _assemble_fragmented_full_shard_local(
            context=context,
            storage_root=Path('/results'),
            force=bool(spec.get('force', False)),
        )
        volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.function(
        cpu=4,
        memory=4096,
        timeout=21600,
        volumes={'/results': volume},
        retries=1,
    )
    def assemble_subject_aggregation_remote(scope: str = 'full') -> str:
        if scope != 'full':
            raise ValueError('Assembly is currently required only for full scope.')
        processed_dir = Path('/results/processed_v4_preflight')
        results_dir = Path(REMOTE_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        _, _, input_paths = _reference_artifacts(
            processed_dir=processed_dir,
            allow_build=False,
        )
        shard_payloads: list[dict[str, Any]] = []
        missing: list[str] = []
        for condition in CONDITIONS:
            for clf_name in CLF_ORDER:
                shard_path = Path('/results') / _shard_remote_relpath(scope, condition, clf_name)
                if not shard_path.exists():
                    missing.append(str(shard_path))
                    continue
                payload = json.loads(shard_path.read_text())
                _validate_shard_payload(
                    payload,
                    expected_condition=condition,
                    expected_classifier=clf_name,
                )
                shard_payloads.append(payload)
        if missing:
            raise FileNotFoundError(
                'Cannot assemble the full subject-aggregation diagnostic because '
                f'the following shard outputs are missing: {missing}'
            )
        report = _assemble_subject_aggregation_report(
            shard_payloads=shard_payloads,
            scope=scope,
            input_paths=input_paths,
        )
        output_path = results_dir / _report_output_name(scope)
        atomic_write_json(output_path, report)
        volume.commit()
        summary = {
            'remote_output_path': str(output_path),
            'concise_report_summary': {
                'scope': report['scope'],
                'conditions': report['conditions_represented'],
                'classifier_families': report['classifiers_represented'],
                'recommended_rule': report['summary']['recommended_primary_aggregation_rule'],
            },
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{_report_remote_relpath(scope)} '
                f'experiments/results/v4_preflight/{_report_output_name(scope)}'
            ),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    def _validate_remote_full_shard_if_present(context: dict[str, Any]) -> str:
        relpath = _shard_remote_relpath('full', context['condition'], context['clf_name'])
        payload = _volume_read_json(relpath)
        if payload is None:
            return 'missing'
        _validate_shard_payload(
            payload,
            expected_condition=context['condition'],
            expected_classifier=context['clf_name'],
        )
        return 'valid'

    def _target_contexts(targets: str) -> list[dict[str, Any]]:
        parsed = _parse_targets(targets)
        contexts: list[dict[str, Any]] = []
        for condition, clf_name in parsed:
            if (condition, clf_name) not in RECOVERY_TARGETS_DEFAULT:
                raise SystemExit(
                    f'Fragmented recovery only supports {RECOVERY_TARGETS_DEFAULT}; '
                    f'got {condition}:{clf_name}.'
                )
            contexts.append(
                _build_fragment_context_from_local_artifacts(
                    condition=condition,
                    clf_name=clf_name,
                )
            )
        return contexts

    def _submit_specs_in_batches(
        worker: Any,
        specs: list[dict[str, Any]],
        submission_batch_size: int,
    ) -> int:
        submitted = 0
        for batch in _chunked(specs, submission_batch_size):
            worker.spawn_map(batch)
            submitted += len(batch)
        return submitted

    def _stage_progress_for_context(
        context: dict[str, Any],
        existing_fragment_paths: set[str],
    ) -> list[dict[str, Any]]:
        progress: list[dict[str, Any]] = []
        for stage_name, specs in _stage_specs_for_context(context):
            expected = len(specs)
            completed = sum(int(spec['relpath'] in existing_fragment_paths) for spec in specs)
            progress.append({
                'stage_name': stage_name,
                'expected_fragments': expected,
                'completed_fragments': completed,
                'missing_fragments': expected - completed,
                'completion_fraction': round(completed / max(expected, 1), 6),
            })
        return progress

    @app.local_entrypoint()
    def main(
        scope: str = 'full',
        action: str = 'submit',
        targets: str = '',
        submission_batch_size: int = DEFAULT_FRAGMENT_MAX_IN_FLIGHT,
        max_in_flight: int = 0,
        force: bool = False,
    ) -> None:
        if scope not in SCOPES:
            raise SystemExit(f'Unsupported --scope {scope!r}; choose from {SCOPES}.')
        if scope == 'quick':
            if action != 'run':
                raise SystemExit('Quick scope supports only --action run.')
            print('Submitting detached Modal subject-aggregation diagnostic (quick scope).', flush=True)
            print('Required remote inputs:', flush=True)
            print('  gait-results:/processed_v4_preflight/gait_features_v4_reference.csv', flush=True)
            print('  gait-results:/processed_v4_preflight/control_partition_v4_reference.json', flush=True)
            print('  gait-results:/results_v3/pd_results_v3.json', flush=True)
            print('  gait-results:/results_v3/hd_results_v3.json', flush=True)
            print('  gait-results:/results_v3/als_results_v3.json', flush=True)
            print('Output JSON:', flush=True)
            print(f'  gait-results:/{_report_remote_relpath("quick")}', flush=True)
            print('Download command:', flush=True)
            print(
                '  modal volume get gait-results '
                f'{_report_remote_relpath("quick")} '
                f'experiments/results/v4_preflight/{_report_output_name("quick")}',
                flush=True,
            )
            run_subject_aggregation_quick_remote.spawn()
            return

        if action == 'fragmented-status':
            contexts = _target_contexts(targets)
            existing_fragment_paths = _volume_list_paths(REMOTE_FRAGMENT_DIR)
            summary: dict[str, Any] = {
                'fragment_namespace': REMOTE_FRAGMENT_DIR,
                'missing_full_shards': [],
                'per_target_progress': {},
            }
            for context in contexts:
                target_key = f'{context["condition"]}:{context["clf_name"]}'
                shard_status = _validate_remote_full_shard_if_present(context)
                if shard_status == 'missing':
                    summary['missing_full_shards'].append(target_key)
                progress = _stage_progress_for_context(context, existing_fragment_paths)
                summary['per_target_progress'][target_key] = {
                    'full_shard_status': shard_status,
                    'stages': progress,
                    'estimated_remaining_units': sum(
                        stage['missing_fragments'] for stage in progress
                    ),
                }
            print(json.dumps(summary, indent=2), flush=True)
            return

        if action == 'fragmented-submit':
            effective_submission_batch_size = int(submission_batch_size)
            legacy_alias_used = False
            if max_in_flight:
                effective_submission_batch_size = int(max_in_flight)
                legacy_alias_used = True
            if effective_submission_batch_size <= 0:
                raise SystemExit('--submission-batch-size must be positive.')
            contexts = _target_contexts(targets)
            existing_fragment_paths = _volume_list_paths(REMOTE_FRAGMENT_DIR)
            submissions: dict[str, list[dict[str, Any]]] = {
                'rf_outer_candidate': [],
                'rf_outer_result': [],
                'rf_full_source_candidate': [],
                'svm_inner': [],
                'svm_candidate_summary': [],
                'svm_outer_result': [],
                'full_source_selection': [],
            }
            skipped_targets: list[str] = []
            for context in contexts:
                target_key = f'{context["condition"]}:{context["clf_name"]}'
                shard_status = _validate_remote_full_shard_if_present(context)
                if shard_status == 'valid':
                    skipped_targets.append(target_key)
                    continue
                stage_specs = _stage_specs_for_context(context)
                for stage_name, specs in stage_specs:
                    missing_specs = [
                        spec for spec in specs
                        if spec['relpath'] not in existing_fragment_paths
                    ]
                    if not missing_specs:
                        continue
                    if stage_name == 'rf_outer_candidates':
                        submissions['rf_outer_candidate'].extend(missing_specs)
                    elif stage_name == 'rf_outer_results':
                        submissions['rf_outer_result'].extend(missing_specs)
                    elif stage_name == 'rf_full_source_candidates':
                        submissions['rf_full_source_candidate'].extend(missing_specs)
                    elif stage_name in {'svm_outer_inner', 'svm_full_source_inner'}:
                        submissions['svm_inner'].extend(missing_specs)
                    elif stage_name in {'svm_outer_candidate_summaries', 'svm_full_source_candidate_summaries'}:
                        submissions['svm_candidate_summary'].extend(missing_specs)
                    elif stage_name == 'svm_outer_results':
                        submissions['svm_outer_result'].extend(missing_specs)
                    elif stage_name == 'full_source_selection':
                        submissions['full_source_selection'].extend(missing_specs)
                    break

            total_submitted = 0
            total_submitted += _submit_specs_in_batches(
                run_rf_outer_candidate_fragment_remote,
                submissions['rf_outer_candidate'],
                effective_submission_batch_size,
            )
            total_submitted += _submit_specs_in_batches(
                run_rf_outer_result_fragment_remote,
                submissions['rf_outer_result'],
                effective_submission_batch_size,
            )
            total_submitted += _submit_specs_in_batches(
                run_rf_full_source_candidate_fragment_remote,
                submissions['rf_full_source_candidate'],
                effective_submission_batch_size,
            )
            total_submitted += _submit_specs_in_batches(
                run_svm_inner_fragment_remote,
                submissions['svm_inner'],
                effective_submission_batch_size,
            )
            total_submitted += _submit_specs_in_batches(
                run_svm_candidate_summary_fragment_remote,
                submissions['svm_candidate_summary'],
                effective_submission_batch_size,
            )
            total_submitted += _submit_specs_in_batches(
                run_svm_outer_result_fragment_remote,
                submissions['svm_outer_result'],
                effective_submission_batch_size,
            )
            total_submitted += _submit_specs_in_batches(
                run_full_source_selection_fragment_remote,
                submissions['full_source_selection'],
                effective_submission_batch_size,
            )

            summary = {
                'status': 'submitted',
                'fragment_namespace': REMOTE_FRAGMENT_DIR,
                'submission_batch_size': int(effective_submission_batch_size),
                'legacy_max_in_flight_alias_used': legacy_alias_used,
                'submission_batch_explanation': (
                    'submission_batch_size controls spawn_map queue-submission batches; '
                    'max_containers on each worker enforces the real autoscaling cap.'
                ),
                'skipped_existing_full_shards': skipped_targets,
                'submitted_counts': {
                    key: len(value)
                    for key, value in submissions.items()
                },
                'total_submitted': int(total_submitted),
                'monitor_command': 'modal app logs gait-transfer-v4-preflight-aggregation -f',
                'status_command': (
                    'venv/bin/modal run scripts/verification/test_v4_subject_aggregation.py '
                    f'--scope full --action fragmented-status --targets "{targets}"'
                ),
            }
            print(json.dumps(summary, indent=2), flush=True)
            return

        if action == 'fragmented-assemble-missing':
            contexts = _target_contexts(targets)
            submitted = 0
            for context in contexts:
                shard_status = _validate_remote_full_shard_if_present(context)
                if shard_status == 'valid' and not force:
                    continue
                assemble_fragmented_full_shard_remote.spawn({
                    'condition': context['condition'],
                    'clf_name': context['clf_name'],
                    'force': bool(force),
                })
                submitted += 1
            print(
                json.dumps({
                    'status': 'submitted',
                    'submitted_final_shard_assemblies': submitted,
                    'force': bool(force),
                    'fragment_namespace': REMOTE_FRAGMENT_DIR,
                }, indent=2),
                flush=True,
            )
            return

        if action == 'list-missing':
            missing_targets = []
            for condition in CONDITIONS:
                for clf_name in CLF_ORDER:
                    shard_relpath = _shard_remote_relpath('full', condition, clf_name)
                    if not _volume_relpath_exists(shard_relpath):
                        missing_targets.append(f'{condition}:{clf_name}')
            print('Missing full subject-aggregation shards:', flush=True)
            if missing_targets:
                for target in missing_targets:
                    print(f'  {target}', flush=True)
            else:
                print('  none', flush=True)
            return

        if action == 'submit':
            print('Submitting detached Modal subject-aggregation shards (full scope).', flush=True)
            print('Required remote inputs:', flush=True)
            print('  gait-results:/processed_v4_preflight/gait_features_v4_reference.csv', flush=True)
            print('  gait-results:/processed_v4_preflight/control_partition_v4_reference.json', flush=True)
            print('  gait-results:/results_v3/pd_results_v3.json', flush=True)
            print('  gait-results:/results_v3/hd_results_v3.json', flush=True)
            print('  gait-results:/results_v3/als_results_v3.json', flush=True)
            for condition in CONDITIONS:
                for clf_name in CLF_ORDER:
                    run_subject_aggregation_shard_remote.spawn(
                        condition=condition,
                        clf_name=clf_name,
                        scope='full',
                    )
            print('Submitted 21 condition×classifier shard jobs.', flush=True)
            print('After the shards finish, assemble the final report with:', flush=True)
            print(
                '  modal run --detach scripts/verification/test_v4_subject_aggregation.py '
                '--scope full --action assemble',
                flush=True,
            )
            return

        if action == 'recover-missing':
            recovery_targets = _parse_targets(targets)
            print('Submitting detached Modal subject-aggregation recovery jobs (full scope).', flush=True)
            print('Requested targets:', flush=True)
            for condition, clf_name in recovery_targets:
                print(f'  {condition}:{clf_name}', flush=True)
                run_subject_aggregation_recovery_remote.spawn(
                    condition=condition,
                    clf_name=clf_name,
                )
            return

        if action == 'assemble':
            print('Submitting detached Modal subject-aggregation assembly (full scope).', flush=True)
            print('Final output JSON:', flush=True)
            print(f'  gait-results:/{_report_remote_relpath("full")}', flush=True)
            print('Download command:', flush=True)
            print(
                '  modal volume get gait-results '
                f'{_report_remote_relpath("full")} '
                f'experiments/results/v4_preflight/{_report_output_name("full")}',
                flush=True,
            )
            assemble_subject_aggregation_remote.spawn(scope='full')
            return

        raise SystemExit(
            'Full scope supports --action submit, --action list-missing, '
            '--action recover-missing, --action fragmented-status, '
            '--action fragmented-submit, --action fragmented-assemble-missing, '
            'or --action assemble.'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=SCOPES, default='quick')
    args = parser.parse_args()
    run_local(scope=args.scope)
