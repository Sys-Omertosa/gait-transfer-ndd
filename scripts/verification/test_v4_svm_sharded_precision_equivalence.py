"""
Cheap precision-equivalence regression for the authoritative sharded SVM path.

Usage:
    MPLCONFIGDIR=/tmp/mpl python scripts/verification/test_v4_svm_sharded_precision_equivalence.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.datasets import make_classification
from sklearn.svm import SVC


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
import train  # type: ignore
from v4_provenance import atomic_write_json, sha256_file  # type: ignore


def _write_fixture(root: Path) -> tuple[pl.DataFrame, list[str]]:
    processed_dir = root / 'processed_v4'
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols('v4')
    disease_subjects = ['pd_01', 'pd_02', 'pd_03']
    control_subjects = ['co_01', 'co_02', 'co_03']
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
        class_sep=1.1,
        flip_y=0.0,
        random_state=123,
    )
    data = {
        'subject_id': subject_ids,
        'condition': conditions,
        'label': labels,
    }
    for idx, col in enumerate(feature_cols):
        data[col] = X[:, idx].astype(float)
    df = pl.DataFrame(data)

    features_path = processed_dir / 'gait_features_v4.csv'
    df.write_csv(features_path)
    per_stride_only_path = processed_dir / 'gait_features_v4_per_stride_only.csv'
    build_per_stride_only_matrix(df, feature_set_version='v4').write_csv(per_stride_only_path)
    subject_level_path = processed_dir / 'gait_features_v4_subject_level.csv'
    build_subject_level_matrix(df, feature_set_version='v4').write_csv(subject_level_path)
    timing_path = processed_dir / 'gait_features_v4_timing_sensitivity.csv'
    dfa_path = processed_dir / 'gait_features_v4_dfa_sensitivity.csv'
    no_dfa_path = processed_dir / 'gait_features_v4_no_dfa_sensitivity.csv'
    df.write_csv(timing_path)
    df.write_csv(dfa_path)
    df.drop('dfa_alpha_stride').write_csv(no_dfa_path)

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

    preprocessing_manifest_path = processed_dir / 'preprocessing_manifest_v4.json'
    preprocessing_manifest = {
        'schema_version': 'v4-preprocessing-manifest-v2',
        'methodology_version': 'v4-hardening',
        'approved_protocol_manifest_required': True,
        'protocol_manifest_sha256': sha256_file(protocol_manifest_path),
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

    return df, control_subjects


def _small_svm_configs() -> dict[str, dict]:
    return {
        'svm': {
            'clf': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
            ),
            'param_grid': {
                'clf__C': [1.0, 10.0],
                'clf__gamma': ['scale'],
            },
        }
    }


def _assert_outer_traces_equivalent(monolithic: dict, sharded: dict) -> None:
    mono_traces = monolithic['classifiers']['svm']['outer_fold_selection_trace']
    shard_traces = sharded['classifiers']['svm']['outer_fold_selection_trace']
    assert len(mono_traces) == len(shard_traces)
    for mono_trace, shard_trace in zip(mono_traces, shard_traces):
        assert mono_trace['held_out_subject_id'] == shard_trace['held_out_subject_id']
        assert mono_trace['selected_imbalance_strategy'] == shard_trace['selected_imbalance_strategy']
        assert mono_trace['selected_params'] == shard_trace['selected_params']


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='v4-svm-sharded-precision-') as tmpdir:
        root = Path(tmpdir)
        df, control_subjects = _write_fixture(root)
        context = runner._load_validated_authoritative_context(  # noqa: SLF001
            'pd',
            namespace='results_v4',
            volume_root=root,
        )

        original_get_configs = train.get_classifier_configs
        train.get_classifier_configs = _small_svm_configs
        try:
            fold_payloads = [
                runner._build_svm_outer_fold_payload(  # noqa: SLF001
                    context=context,
                    outer_fold_index=fold_idx,
                    outer_subject_id=outer_subject_id,
                )
                for fold_idx, outer_subject_id in enumerate(context['outer_subject_ids'])
            ]
            sharded_output = runner._assemble_svm_classifier_shard_payload(  # noqa: SLF001
                context=context,
                fold_payloads=fold_payloads,
            )

            mono_models_dir = root / 'mono_models'
            monolithic_result = train._evaluate_within_condition_classifier(  # noqa: SLF001
                condition='pd',
                clf_name='svm',
                clf=_small_svm_configs()['svm']['clf'],
                param_grid=_small_svm_configs()['svm']['param_grid'],
                X=context['X'],
                y=context['y'],
                groups=context['groups'],
                candidate_imbalance_strategies=context['candidate_strategy_policy']['svm'],
                models_dir=mono_models_dir,
                selection_trace_dir=None,
                subject_aggregation_rule=context['aggregation_rule'],
                tie_break_rule=context['tie_break_rule'],
            )
        finally:
            train.get_classifier_configs = original_get_configs

        mono_svm = monolithic_result
        shard_svm = sharded_output['classifiers']['svm']

        assert np.array_equal(np.asarray(mono_svm['y_true']), np.asarray(shard_svm['y_true']))
        assert np.array_equal(np.asarray(mono_svm['y_pred']), np.asarray(shard_svm['y_pred']))
        max_prob_delta = float(np.max(np.abs(
            np.asarray(mono_svm['y_prob'], dtype=float)
            - np.asarray(shard_svm['y_prob'], dtype=float)
        )))
        assert max_prob_delta <= 1e-6, max_prob_delta
        assert abs(float(mono_svm['f1_macro']) - float(shard_svm['f1_macro'])) <= 1e-9
        assert abs(
            float(mono_svm['subject_primary_f1_macro'])
            - float(shard_svm['subject_primary_f1_macro'])
        ) <= 1e-9
        assert abs(
            float(mono_svm['subject_metrics']['subject_log_loss'])
            - float(shard_svm['subject_metrics']['subject_log_loss'])
        ) <= 1e-6
        assert mono_svm['full_source_selected_params'] == shard_svm['full_source_selected_params']
        assert (
            mono_svm['full_source_selected_imbalance_strategy']
            == shard_svm['full_source_selected_imbalance_strategy']
        )
        assert mono_svm['modal_strategy'] == shard_svm['modal_strategy']
        mono_trace_wrapper = {'classifiers': {'svm': {'outer_fold_selection_trace': mono_svm['outer_fold_selection_trace']}}}
        shard_trace_wrapper = {'classifiers': {'svm': {'outer_fold_selection_trace': shard_svm['outer_fold_selection_trace']}}}
        _assert_outer_traces_equivalent(mono_trace_wrapper, shard_trace_wrapper)

    print('v4 sharded-vs-monolithic SVM precision equivalence passed.')


if __name__ == '__main__':
    main()
