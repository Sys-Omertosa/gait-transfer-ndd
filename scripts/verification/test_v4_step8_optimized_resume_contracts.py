"""
Cheap local validation for the optimized Step 8 shard-resume contract.

Usage:
    venv/bin/python scripts/verification/test_v4_step8_optimized_resume_contracts.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


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

from scripts.training import run_control_split_sensitivity_modal as runner  # type: ignore
from v4_downstream import (  # type: ignore
    build_authoritative_step12_context,
    canonical_payload_sha256,
    sha256_file,
)


EXECUTION_ID = 'v4d-20260611T022033+0000-85765ad9afd7'
OLD_DUMP_ROOT = (
    REPO_ROOT
    / 'modal_handoff'
    / 'step8_workspace_migration_20260612'
    / 'old_volume_dump'
)
OLD_RESULTS_ROOT = (
    OLD_DUMP_ROOT
    / 'results_v4'
    / 'control_split_sensitivity_runs'
    / EXECUTION_ID
    / 'near_optimal'
)
OLD_MODELS_ROOT = (
    OLD_DUMP_ROOT
    / 'models_v4'
    / 'control_split_sensitivity_runs'
    / EXECUTION_ID
    / 'near_optimal'
)


def _parse_partial_filename(path: Path) -> tuple[int, str, str]:
    stem = path.stem
    prefix, _, partition_suffix = stem.partition('_partial_v4_partition_')
    condition, clf_name = prefix.split('_', 1)
    return int(partition_suffix), condition, clf_name


def _near_optimal_partition_metadata() -> dict[int, dict]:
    main_partition = json.loads(
        (REPO_ROOT / 'data/processed/v4/control_partition_v4.json').read_text()
    )
    candidates = json.loads(
        (
            REPO_ROOT / 'data/processed/v4/control_partition_candidates_v4.json'
        ).read_text()
    )[: runner.MAX_PARTITIONS]
    return {
        idx: runner._candidate_metadata(  # type: ignore[attr-defined]
            candidate=candidate,
            candidate_family='near_optimal',
            partition_index=idx,
            main_partition=main_partition,
        )
        for idx, candidate in enumerate(candidates, start=1)
    }


def _validate_existing_completed_partials(context: dict, metadata_by_partition: dict[int, dict]) -> int:
    count = 0
    for partial_path in sorted(OLD_RESULTS_ROOT.glob('partition_*/*_partial_v4_partition_*.json')):
        partition_index, condition, clf_name = _parse_partial_filename(partial_path)
        payload = json.loads(partial_path.read_text())
        runner._validate_partial_within_payload(  # type: ignore[attr-defined]
            payload=payload,
            context=context,
            condition=condition,
            clf_name=clf_name,
            partition_metadata=metadata_by_partition[partition_index],
        )
        count += 1
    return count


def _copy_old_path_to_temp(temp_root: Path, old_absolute_results_path: str) -> Path:
    if not old_absolute_results_path.startswith('/results/'):
        raise ValueError(old_absolute_results_path)
    rel = old_absolute_results_path.removeprefix('/results/')
    src = OLD_DUMP_ROOT / rel
    dst = temp_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_old_model_to_temp(
    *,
    temp_models_partition_dir: Path,
    old_models_partition_dir: Path,
    relpath: str,
) -> Path:
    src = old_models_partition_dir / relpath
    dst = temp_models_partition_dir / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _slice_fold_predictions(
    *,
    classifier_payload: dict,
    groups: list[str],
) -> list[dict]:
    y_true_all = list(classifier_payload['y_true'])
    y_pred_all = list(classifier_payload['y_pred'])
    y_prob_all = list(classifier_payload['y_prob'])
    offset = 0
    fold_records: list[dict] = []
    for fold_index, fold_detail in enumerate(classifier_payload['outer_fold_selection_trace']):
        held_out_subject = fold_detail['held_out_subject_id']
        n_rows = sum(1 for group in groups if group == held_out_subject)
        if n_rows <= 0:
            raise RuntimeError(
                f'Unable to infer stride count for held-out subject {held_out_subject}.'
            )
        next_offset = offset + n_rows
        fold_records.append(
            {
                'outer_fold_index': fold_index,
                'held_out_subject_id': held_out_subject,
                'y_true': y_true_all[offset:next_offset],
                'y_pred': y_pred_all[offset:next_offset],
                'y_prob': y_prob_all[offset:next_offset],
                'subject_ids': [held_out_subject] * n_rows,
            }
        )
        offset = next_offset
    if offset != len(y_true_all):
        raise RuntimeError(
            f'Outer-fold slicing consumed {offset} predictions, expected {len(y_true_all)}.'
        )
    return fold_records


def _prepare_local_resume_fixture(
    *,
    context: dict,
    metadata_by_partition: dict[int, dict],
) -> tuple[Path, dict, dict]:
    temp_root = Path(tempfile.mkdtemp(prefix='step8-resume-'))
    (temp_root / 'processed_v4').mkdir(parents=True, exist_ok=True)
    for name in [
        'gait_features_v4.csv',
        'control_partition_v4.json',
        'control_partition_candidates_v4.json',
    ]:
        shutil.copy2(REPO_ROOT / 'data' / 'processed' / 'v4' / name, temp_root / 'processed_v4' / name)

    sample_partial_path = OLD_RESULTS_ROOT / 'partition_1' / 'als_knn_partial_v4_partition_1.json'
    sample_partial = json.loads(sample_partial_path.read_text())
    partition_index = 1
    condition = 'als'
    clf_name = 'knn'
    partition_metadata = metadata_by_partition[partition_index]
    X, y, groups, _, _, _ = runner._load_partition_condition_pool(  # type: ignore[attr-defined]
        features_path=temp_root / 'processed_v4' / 'gait_features_v4.csv',
        condition=condition,
        control_a=partition_metadata['control_A'],
    )
    del X, y
    fold_slices = _slice_fold_predictions(
        classifier_payload=sample_partial['classifiers'][clf_name],
        groups=list(groups),
    )

    temp_results_partition_dir = (
        temp_root
        / 'results_v4'
        / 'control_split_sensitivity_runs'
        / context['downstream_execution_id']
        / 'near_optimal'
        / 'partition_1'
    )
    temp_models_partition_dir = (
        temp_root
        / 'models_v4'
        / 'control_split_sensitivity_runs'
        / context['downstream_execution_id']
        / 'near_optimal'
        / 'partition_1'
    )
    temp_results_partition_dir.mkdir(parents=True, exist_ok=True)
    temp_models_partition_dir.mkdir(parents=True, exist_ok=True)

    classifier_payload = sample_partial['classifiers'][clf_name]
    for fold_record, fold_detail in zip(
        fold_slices,
        classifier_payload['outer_fold_selection_trace'],
        strict=True,
    ):
        temp_trace_path = _copy_old_path_to_temp(
            temp_root,
            fold_detail['candidate_trace_relpath'],
        )
        temp_fold_model_path = _copy_old_model_to_temp(
            temp_models_partition_dir=temp_models_partition_dir,
            old_models_partition_dir=OLD_MODELS_ROOT / 'partition_1',
            relpath=fold_detail['fold_model_relpath'],
        )
        shard_payload = {
            'schema_version': runner.WITHIN_FOLD_SHARD_SCHEMA_VERSION,
            'condition': condition,
            'classifier': clf_name,
            'candidate_family': 'near_optimal',
            'partition_index': partition_index,
            'partition_key': 'partition_1',
            'partition_hash': partition_metadata['partition_hash'],
            'control_A': list(partition_metadata['control_A']),
            'control_B': list(partition_metadata['control_B']),
            'models_dir': str(temp_models_partition_dir),
            'feature_matrix_hash': context['feature_matrix_sha256'],
            'protocol_manifest_hash': context['protocol_manifest_sha256'],
            'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
            'downstream_execution_id': context['downstream_execution_id'],
            'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
            'partition_metadata': partition_metadata,
            'outer_fold_index': fold_record['outer_fold_index'],
            'held_out_subject_id': fold_record['held_out_subject_id'],
            'held_out_true_label': fold_detail['held_out_true_label'],
            'selected_imbalance_strategy': fold_detail['selected_imbalance_strategy'],
            'selected_params': fold_detail['selected_params'],
            'selected_inner_subject_f1': fold_detail['selected_inner_subject_f1'],
            'selected_inner_stride_f1': fold_detail['selected_inner_stride_f1'],
            'selected_inner_subject_log_loss': fold_detail['selected_inner_subject_log_loss'],
            'candidate_rankings': fold_detail['candidate_rankings'],
            'candidate_imbalance_strategies': classifier_payload['candidate_imbalance_strategies'],
            'subject_aggregation_rule': sample_partial['subject_aggregation_rule'],
            'tie_break_rule': sample_partial['tie_break_rule'],
            'y_true': fold_record['y_true'],
            'y_pred': fold_record['y_pred'],
            'y_prob': fold_record['y_prob'],
            'subject_ids': fold_record['subject_ids'],
            'candidate_trace_relpath': str(temp_trace_path),
            'candidate_trace_sha256': sha256_file(temp_trace_path),
            'fold_model_relpath': fold_detail['fold_model_relpath'],
            'fold_model_sha256': sha256_file(temp_fold_model_path),
        }
        shard_payload['payload_sha256'] = canonical_payload_sha256(shard_payload)
        shard_path = temp_root / runner._within_fold_shard_api_path(  # type: ignore[attr-defined]
            execution_id=context['downstream_execution_id'],
            candidate_family='near_optimal',
            partition_index=partition_index,
            condition=condition,
            clf_name=clf_name,
            outer_fold_index=fold_record['outer_fold_index'],
            held_out_subject_id=fold_record['held_out_subject_id'],
        )
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(json.dumps(shard_payload, indent=2, sort_keys=True))

    temp_full_trace_path = _copy_old_path_to_temp(
        temp_root,
        classifier_payload['full_source_selection_trace_path'],
    )
    temp_full_model_path = _copy_old_model_to_temp(
        temp_models_partition_dir=temp_models_partition_dir,
        old_models_partition_dir=OLD_MODELS_ROOT / 'partition_1',
        relpath=classifier_payload['full_source_model_path'],
    )
    full_source_payload = {
        'schema_version': runner.WITHIN_FULL_SOURCE_SHARD_SCHEMA_VERSION,
        'condition': condition,
        'classifier': clf_name,
        'candidate_family': 'near_optimal',
        'partition_index': partition_index,
        'partition_key': 'partition_1',
        'partition_hash': partition_metadata['partition_hash'],
        'control_A': list(partition_metadata['control_A']),
        'control_B': list(partition_metadata['control_B']),
        'models_dir': str(temp_models_partition_dir),
        'feature_matrix_hash': context['feature_matrix_sha256'],
        'protocol_manifest_hash': context['protocol_manifest_sha256'],
        'preprocessing_manifest_hash': context['preprocessing_manifest_sha256'],
        'downstream_execution_id': context['downstream_execution_id'],
        'downstream_execution_manifest_hash': context['downstream_manifest_sha256'],
        'partition_metadata': partition_metadata,
        'selected_imbalance_strategy': classifier_payload['full_source_selected_imbalance_strategy'],
        'selected_params': classifier_payload['full_source_selected_params'],
        'selection_subject_f1': classifier_payload['full_source_selection_subject_f1'],
        'selection_stride_f1': classifier_payload['full_source_selection_stride_f1'],
        'selection_subject_log_loss': classifier_payload['full_source_selection_subject_log_loss'],
        'selection_trace': classifier_payload['full_source_selection_trace'],
        'selection_trace_path': str(temp_full_trace_path),
        'selection_trace_sha256': sha256_file(temp_full_trace_path),
        'model_path': classifier_payload['full_source_model_path'],
        'model_sha256': sha256_file(temp_full_model_path),
        'candidate_imbalance_strategies': classifier_payload['candidate_imbalance_strategies'],
        'subject_aggregation_rule': sample_partial['subject_aggregation_rule'],
        'tie_break_rule': sample_partial['tie_break_rule'],
    }
    full_source_payload['payload_sha256'] = canonical_payload_sha256(full_source_payload)
    full_source_path = temp_root / runner._within_full_source_shard_api_path(  # type: ignore[attr-defined]
        execution_id=context['downstream_execution_id'],
        candidate_family='near_optimal',
        partition_index=partition_index,
        condition=condition,
        clf_name=clf_name,
    )
    full_source_path.parent.mkdir(parents=True, exist_ok=True)
    full_source_path.write_text(json.dumps(full_source_payload, indent=2, sort_keys=True))

    return temp_root, partition_metadata, {'partition_index': partition_index, 'condition': condition, 'clf_name': clf_name}


def main() -> None:
    if not OLD_RESULTS_ROOT.exists() or not OLD_MODELS_ROOT.exists():
        raise SystemExit('Step 8 old-volume handoff dump is missing; cannot run optimized resume contracts test.')

    context = build_authoritative_step12_context(
        volume_root=REPO_ROOT,
        require_downstream_manifest=True,
        required_downstream_files=('scripts/training/run_control_split_sensitivity_modal.py',),
        repo_root=None,
    )
    metadata_by_partition = _near_optimal_partition_metadata()

    validated_count = _validate_existing_completed_partials(context, metadata_by_partition)
    assert validated_count == 27, validated_count
    print(f'PASS: validated {validated_count} existing completed Step 8 partials from the handoff dump.')

    with tempfile.TemporaryDirectory(prefix='step8-missing-') as tmpdir:
        temp_root = Path(tmpdir)
        shutil.copytree(REPO_ROOT / 'data' / 'processed' / 'v4', temp_root / 'processed_v4')
        target_root = (
            temp_root
            / 'results_v4'
            / 'control_split_sensitivity_runs'
            / context['downstream_execution_id']
            / 'near_optimal'
        )
        target_root.mkdir(parents=True, exist_ok=True)
        present: set[str] = set()
        for partial_path in sorted(OLD_RESULTS_ROOT.glob('partition_*/*_partial_v4_partition_*.json')):
            partition_index, condition, clf_name = _parse_partial_filename(partial_path)
            present.add(f'partition_{partition_index}:{condition}:{clf_name}')
            dest = target_root / f'partition_{partition_index}' / partial_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(partial_path, dest)

        missing_detected = sorted(
            f'{entry["partition_key"]}:{entry["condition"]}:{entry["classifier"]}'
            for entry in runner._iter_within_classifier_states(  # type: ignore[attr-defined]
                volume_root=temp_root,
                context=context,
                candidate_family='near_optimal',
            )
            if entry['state']['partial_state'] != 'completed'
        )
        all_expected = sorted(
            f'partition_{partition_index}:{condition}:{clf_name}'
            for partition_index in range(1, 4)
            for condition in runner.CONDITIONS
            for clf_name in runner.CLF_ORDER
        )
        expected_missing = sorted(set(all_expected) - present)
        assert missing_detected == expected_missing
        print(f'PASS: missing-classifier detection matched the expected complement ({len(expected_missing)} missing).')

    expected_fold_path = (
        f'results_v4/control_split_sensitivity_runs/{context["downstream_execution_id"]}/'
        'near_optimal/partition_1/within_shards/pd/rf/outer_fold_00_control1.json'
    )
    assert runner._within_fold_shard_api_path(  # type: ignore[attr-defined]
        execution_id=context['downstream_execution_id'],
        candidate_family='near_optimal',
        partition_index=1,
        condition='pd',
        clf_name='rf',
        outer_fold_index=0,
        held_out_subject_id='control1',
    ) == expected_fold_path
    print('PASS: within-fold shard paths are deterministic.')

    temp_root, partition_metadata, sample = _prepare_local_resume_fixture(
        context=context,
        metadata_by_partition=metadata_by_partition,
    )
    try:
        assembled = runner._assemble_classifier_partial_from_shards(  # type: ignore[attr-defined]
            volume_root=temp_root,
            context=context,
            candidate_family='near_optimal',
            partition_index=sample['partition_index'],
            partition_metadata=partition_metadata,
            condition=sample['condition'],
            clf_name=sample['clf_name'],
            control_a=partition_metadata['control_A'],
        )
        assert assembled['status'] == 'completed', assembled
        partial_path = temp_root / runner._within_partial_api_path(  # type: ignore[attr-defined]
            execution_id=context['downstream_execution_id'],
            candidate_family='near_optimal',
            partition_index=sample['partition_index'],
            condition=sample['condition'],
            clf_name=sample['clf_name'],
        )
        payload = json.loads(partial_path.read_text())
        runner._validate_partial_within_payload(  # type: ignore[attr-defined]
            payload=payload,
            context=context,
            condition=sample['condition'],
            clf_name=sample['clf_name'],
            partition_metadata=partition_metadata,
        )
        first_sha = sha256_file(partial_path)
        print('PASS: assembling from complete fold/full-source shards produced a valid Step 8 classifier partial.')

        second = runner._assemble_classifier_partial_from_shards(  # type: ignore[attr-defined]
            volume_root=temp_root,
            context=context,
            candidate_family='near_optimal',
            partition_index=sample['partition_index'],
            partition_metadata=partition_metadata,
            condition=sample['condition'],
            clf_name=sample['clf_name'],
            control_a=partition_metadata['control_A'],
        )
        assert second['status'] == 'skipped', second
        assert sha256_file(partial_path) == first_sha
        print('PASS: existing completed classifier partials are not overwritten on re-assembly.')
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    main()
