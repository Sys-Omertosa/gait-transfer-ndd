"""
Cheap local validation for authoritative Step 2 context loading and stale-artifact rejection.

Usage:
    python scripts/verification/test_v4_authoritative_context_and_reuse.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import polars as pl
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

from features import (  # type: ignore
    build_per_stride_only_matrix,
    build_subject_level_matrix,
    get_feature_cols,
)
from scripts.training import run_within_condition_modal as runner  # type: ignore
from v4_provenance import (  # type: ignore
    atomic_write_json,
    canonical_payload_sha256,
    sha256_file,
)


def _with_digest(payload: dict) -> dict:
    materialized = copy.deepcopy(payload)
    materialized['payload_sha256'] = canonical_payload_sha256(materialized)
    return materialized


def _write_fixture(root: Path) -> tuple[dict, dict]:
    processed_dir = root / 'processed_v4'
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols('v4')
    disease_subjects = ['pd_01', 'pd_02']
    control_subjects = ['co_01', 'co_02']
    samples_per_subject = 6
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
        n_clusters_per_class=1,
        class_sep=1.2,
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
    feature_df = pl.DataFrame(data)

    features_path = processed_dir / 'gait_features_v4.csv'
    feature_df.write_csv(features_path)
    per_stride_only = build_per_stride_only_matrix(feature_df, feature_set_version='v4')
    per_stride_only_path = processed_dir / 'gait_features_v4_per_stride_only.csv'
    per_stride_only.write_csv(per_stride_only_path)
    subject_level = build_subject_level_matrix(feature_df, feature_set_version='v4')
    subject_level_path = processed_dir / 'gait_features_v4_subject_level.csv'
    subject_level.write_csv(subject_level_path)
    timing_path = processed_dir / 'gait_features_v4_timing_sensitivity.csv'
    feature_df.write_csv(timing_path)
    dfa_path = processed_dir / 'gait_features_v4_dfa_sensitivity.csv'
    feature_df.write_csv(dfa_path)
    no_dfa_path = processed_dir / 'gait_features_v4_no_dfa_sensitivity.csv'
    feature_df.drop('dfa_alpha_stride').write_csv(no_dfa_path)

    partition = {
        'control_A': control_subjects,
        'control_B': [],
    }
    partition_path = processed_dir / 'control_partition_v4.json'
    atomic_write_json(partition_path, partition)

    protocol_manifest_path = processed_dir / 'v4_protocol_manifest.json'
    protocol_manifest = {
        'approved': True,
        'schema_version': 'v4-protocol-manifest-v2',
        'methodology_version': 'v4-hardening',
        'robust_mad_multiplier': 3.0,
        'dfa_policy': 'concatenated',
        'aggregation_rule': 'mean_probability',
        'subject_probability_threshold': 0.5,
        'tie_break_rule': 'subject_probability_loss_then_lexicographic',
        'candidate_strategy_policy': {
            clf_name: list(strategies)
            for clf_name, strategies in runner.APPROVED_CANDIDATE_STRATEGY_POLICY.items()
        },
    }
    atomic_write_json(protocol_manifest_path, protocol_manifest)
    protocol_sha = sha256_file(protocol_manifest_path)

    preprocessing_manifest_path = processed_dir / 'preprocessing_manifest_v4.json'
    preprocessing_manifest = {
        'schema_version': 'v4-preprocessing-manifest-v2',
        'methodology_version': 'v4-hardening',
        'approved_protocol_manifest_required': True,
        'protocol_manifest_sha256': protocol_sha,
        'feature_matrix_sha256': sha256_file(features_path),
        'partition_sha256': sha256_file(partition_path),
        'robust_mad_multiplier': 3.0,
        'dfa_policy': 'concatenated',
        'aggregation_rule': 'mean_probability',
        'subject_probability_threshold': 0.5,
        'tie_break_rule': 'subject_probability_loss_then_lexicographic',
        'per_stride_only_matrix_path': str(per_stride_only_path),
        'per_stride_only_matrix_sha256': sha256_file(per_stride_only_path),
        'subject_level_matrix_path': str(subject_level_path),
        'subject_level_matrix_sha256': sha256_file(subject_level_path),
        'timing_sensitivity_matrix_path': str(timing_path),
        'timing_sensitivity_matrix_sha256': sha256_file(timing_path),
        'dfa_sensitivity_matrix_path': str(dfa_path),
        'dfa_sensitivity_matrix_sha256': sha256_file(dfa_path),
        'no_dfa_sensitivity_matrix_path': str(no_dfa_path),
        'no_dfa_sensitivity_matrix_sha256': sha256_file(no_dfa_path),
    }
    atomic_write_json(preprocessing_manifest_path, preprocessing_manifest)

    return {
        'processed_dir': processed_dir,
        'feature_df': feature_df,
        'feature_cols': feature_cols,
        'control_subjects': control_subjects,
    }, partition


def _write_dummy_file(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_file(path)


def _valid_classifier_shard_payload(context: dict, clf_name: str) -> dict:
    paths = context['paths']
    condition = context['condition']
    held_out_subject = context['outer_subject_ids'][0]
    full_model_path = paths['models_dir'] / f'{condition}_{clf_name}.joblib'
    full_model_sha = _write_dummy_file(full_model_path, b'full-model')
    full_trace_path = paths['selection_trace_dir'] / (
        f'{condition}_{clf_name}_full_source_candidate_trace.npz'
    )
    full_trace_sha = _write_dummy_file(full_trace_path, b'full-trace')
    fold_model_path = paths['models_dir'] / 'within_folds' / (
        f'{condition}_{clf_name}_fold_00_{held_out_subject}.joblib'
    )
    fold_model_sha = _write_dummy_file(fold_model_path, b'fold-model')
    candidate_trace_path = paths['selection_trace_dir'] / (
        f'{condition}_{clf_name}_fold_00_candidate_trace.npz'
    )
    candidate_trace_sha = _write_dummy_file(candidate_trace_path, b'candidate-trace')

    payload = {
        'schema_version': runner.CLASSIFIER_SHARD_SCHEMA_VERSION,
        'condition': condition,
        'pool_subjects': context['pool_subjects'],
        'pool_strides': context['pool_strides'],
        'feature_cols': list(context['feature_cols']),
        'n_features': len(context['feature_cols']),
        'feature_matrix_file': runner.DEFAULT_FEATURE_MATRIX_FILE,
        'feature_set_version': runner.DEFAULT_FEATURE_SET_VERSION,
        'normalization': runner.DEFAULT_NORMALIZATION,
        'models_dir': str(paths['models_dir']),
        'candidate_strategy_policy': {
            clf_name: list(context['candidate_strategy_policy'][clf_name])
        },
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'classifiers': {
            clf_name: {
                'f1_macro': 0.5,
                'subject_primary_f1_macro': 0.5,
                'selected_imbalance_strategy': context['candidate_strategy_policy'][clf_name][0],
                'candidate_imbalance_strategies': list(
                    context['candidate_strategy_policy'][clf_name]
                ),
                'outer_fold_selection_trace': [{
                    'held_out_subject_id': held_out_subject,
                    'fold_model_relpath': str(
                        fold_model_path.relative_to(paths['models_dir'])
                    ),
                    'fold_model_sha256': fold_model_sha,
                    'candidate_trace_relpath': str(candidate_trace_path),
                    'candidate_trace_sha256': candidate_trace_sha,
                }],
                'full_source_selection_trace': [{'rank': 1}],
                'full_source_selection_trace_path': str(full_trace_path),
                'full_source_selection_trace_sha256': full_trace_sha,
                'full_source_model_path': str(
                    full_model_path.relative_to(paths['models_dir'])
                ),
                'full_source_model_sha256': full_model_sha,
            }
        },
    }
    return _with_digest(payload)


def _valid_svm_fold_payload(context: dict) -> dict:
    paths = context['paths']
    condition = context['condition']
    outer_fold_index = 0
    outer_subject_id = context['outer_subject_ids'][0]
    fold_model_path = paths['models_dir'] / 'within_folds' / (
        f'{condition}_svm_fold_{outer_fold_index:02d}_{outer_subject_id}.joblib'
    )
    fold_model_sha = _write_dummy_file(fold_model_path, b'svm-fold-model')
    payload = {
        'schema_version': runner.SVM_OUTER_FOLD_SCHEMA_VERSION,
        'condition': condition,
        'classifier': 'svm',
        'outer_fold_index': outer_fold_index,
        'held_out_subject_id': outer_subject_id,
        'subject_aggregation_rule': context['aggregation_rule'],
        'subject_probability_threshold': context['subject_probability_threshold'],
        'tie_break_rule': context['tie_break_rule'],
        'candidate_strategy_policy': list(context['candidate_strategy_policy']['svm']),
        'feature_matrix_hash': context['feature_matrix_hash'],
        'partition_hash': context['partition_hash'],
        'protocol_manifest_hash': context['protocol_manifest_hash'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_hash'],
        'selected_imbalance_strategy': context['candidate_strategy_policy']['svm'][0],
        'selected_params': {'clf__C': 1.0, 'clf__gamma': 'scale'},
        'candidate_rankings_full': [{
            'imbalance_strategy': context['candidate_strategy_policy']['svm'][0],
            'params': {'clf__C': 1.0, 'clf__gamma': 'scale'},
        }],
        'public_outer_fold_detail': {
            'held_out_subject_id': outer_subject_id,
            'fold_model_relpath': str(fold_model_path.relative_to(paths['models_dir'])),
            'fold_model_sha256': fold_model_sha,
        },
        'y_true': [1, 1],
        'y_pred': [1, 0],
        'y_prob': [0.812345678901, 0.212345678901],
        'subject_ids': [outer_subject_id, outer_subject_id],
    }
    return _with_digest(payload)


def _assert_manifest_chain_mismatch_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix='v4-context-mismatch-') as tmpdir:
        root = Path(tmpdir)
        _write_fixture(root)
        preprocessing_manifest_path = root / 'processed_v4' / 'preprocessing_manifest_v4.json'
        preprocessing_manifest = json.loads(preprocessing_manifest_path.read_text())
        preprocessing_manifest['protocol_manifest_sha256'] = 'corrupted'
        atomic_write_json(preprocessing_manifest_path, preprocessing_manifest)
        try:
            runner._load_validated_authoritative_context(  # noqa: SLF001
                'pd',
                namespace='results_v4',
                volume_root=root,
            )
        except ValueError:
            return
        raise AssertionError('Expected manifest-chain mismatch rejection.')


def _assert_reload_hooks_present() -> None:
    text = (
        REPO_ROOT / 'scripts' / 'training' / 'run_within_condition_modal.py'
    ).read_text()
    for function_name in (
        'describe_condition_context_remote',
        'collect_status_remote',
        'run_classifier_shard_remote',
        'run_svm_outer_fold_remote',
        'assemble_svm_classifier_shard_remote',
        'assemble_condition_remote',
    ):
        marker = f'def {function_name}('
        start = text.index(marker)
        next_start = text.find('\n@app.function(', start + 1)
        block = text[start:] if next_start == -1 else text[start:next_start]
        if 'volume.reload()' not in block:
            raise AssertionError(f'{function_name} is missing volume.reload().')


def main() -> None:
    _assert_manifest_chain_mismatch_rejected()
    _assert_reload_hooks_present()

    with tempfile.TemporaryDirectory(prefix='v4-authoritative-context-') as tmpdir:
        root = Path(tmpdir)
        fixture, _ = _write_fixture(root)
        context = runner._load_validated_authoritative_context(  # noqa: SLF001
            'pd',
            namespace='results_v4',
            volume_root=root,
        )

        valid_shard = _valid_classifier_shard_payload(context, 'qda')
        runner._validate_single_classifier_shard_payload(  # noqa: SLF001
            valid_shard,
            context=context,
            clf_name='qda',
        )

        stale_tie_break = copy.deepcopy(valid_shard)
        stale_tie_break['tie_break_rule'] = 'stride_macro_f1_then_lexicographic'
        stale_tie_break = _with_digest(stale_tie_break)
        try:
            runner._validate_single_classifier_shard_payload(  # noqa: SLF001
                stale_tie_break,
                context=context,
                clf_name='qda',
            )
        except ValueError:
            pass
        else:
            raise AssertionError('Expected stale tie-break shard rejection.')

        stale_feature_hash = copy.deepcopy(valid_shard)
        stale_feature_hash['feature_matrix_hash'] = 'corrupted-feature-hash'
        stale_feature_hash = _with_digest(stale_feature_hash)
        try:
            runner._validate_single_classifier_shard_payload(  # noqa: SLF001
                stale_feature_hash,
                context=context,
                clf_name='qda',
            )
        except ValueError:
            pass
        else:
            raise AssertionError('Expected stale feature-hash shard rejection.')

        reusable_path = root / 'results_v4' / 'classifier_shards' / 'pd_qda_results_v4_shard.json'
        runner._write_json_with_digest(reusable_path, stale_tie_break)  # noqa: SLF001
        try:
            runner._load_reusable_payload(  # noqa: SLF001
                reusable_path,
                validator=lambda payload: runner._validate_single_classifier_shard_payload(  # noqa: SLF001
                    payload,
                    context=context,
                    clf_name='qda',
                ),
                force_recompute_invalid=False,
                label='qda stale shard',
            )
        except ValueError:
            pass
        else:
            raise AssertionError('Expected invalid reusable shard to fail closed.')
        forced = runner._load_reusable_payload(  # noqa: SLF001
            reusable_path,
            validator=lambda payload: runner._validate_single_classifier_shard_payload(  # noqa: SLF001
                payload,
                context=context,
                clf_name='qda',
            ),
            force_recompute_invalid=True,
            label='qda stale shard',
        )
        assert forced is None

        valid_svm_fold = _valid_svm_fold_payload(context)
        runner._validate_svm_outer_fold_payload(  # noqa: SLF001
            valid_svm_fold,
            context=context,
            outer_fold_index=0,
            outer_subject_id=context['outer_subject_ids'][0],
        )

        bad_svm_sha = copy.deepcopy(valid_svm_fold)
        bad_svm_sha['public_outer_fold_detail']['fold_model_sha256'] = 'corrupted'
        bad_svm_sha = _with_digest(bad_svm_sha)
        try:
            runner._validate_svm_outer_fold_payload(  # noqa: SLF001
                bad_svm_sha,
                context=context,
                outer_fold_index=0,
                outer_subject_id=context['outer_subject_ids'][0],
            )
        except ValueError:
            pass
        else:
            raise AssertionError('Expected SVM fold-model SHA rejection.')

        bad_svm_lengths = copy.deepcopy(valid_svm_fold)
        bad_svm_lengths['y_prob'] = bad_svm_lengths['y_prob'][:-1]
        bad_svm_lengths = _with_digest(bad_svm_lengths)
        try:
            runner._validate_svm_outer_fold_payload(  # noqa: SLF001
                bad_svm_lengths,
                context=context,
                outer_fold_index=0,
                outer_subject_id=context['outer_subject_ids'][0],
            )
        except ValueError:
            pass
        else:
            raise AssertionError('Expected SVM fold array-length rejection.')

        assert runner._parse_targets('als:qda,hd:svm') == [('als', 'qda'), ('hd', 'svm')]  # noqa: SLF001
        assert runner._parse_svm_fold_indices('0') == [0]  # noqa: SLF001
        canonical_paths = runner._namespace_paths('results_v4', volume_root=root)  # noqa: SLF001
        smoke_paths = runner._namespace_paths('results_v4_smoke', volume_root=root)  # noqa: SLF001
        assert smoke_paths['results_dir'] != canonical_paths['results_dir']
        assert smoke_paths['models_dir'] != canonical_paths['models_dir']
        assert smoke_paths['results_dir'].name == 'results_v4_smoke'
        assert smoke_paths['models_dir'].name == 'models_v4_smoke'

    print('v4 authoritative context and reuse checks passed.')


if __name__ == '__main__':
    main()
