"""
Cheap resume and full-shard immutability regression for fragmented recovery.
"""

from __future__ import annotations

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

from features import get_feature_cols  # type: ignore
from scripts.verification.test_v4_subject_aggregation import (  # type: ignore
    _assemble_fragmented_full_shard_local,
    _build_fragment_context_from_objects,
    _fragment_identity,
    _rf_outer_candidate_stage_specs,
    _run_fragmented_target_local,
    _run_rf_outer_candidate_fragment_local,
    _validate_fragment_payload,
    _validate_shard_payload,
)


def _make_fixture() -> tuple[pl.DataFrame, dict[str, list[str]], dict[str, dict]]:
    feature_cols = get_feature_cols('v4')
    disease_subjects = [f'pd_{idx:02d}' for idx in range(3)]
    control_subjects = [f'co_{idx:02d}' for idx in range(3)]
    samples_per_subject = 8
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
        flip_y=0.0,
        class_sep=1.25,
        random_state=123,
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
        'hd': {},
        'als': {},
    }
    return feature_df, partition, v3_within


def main() -> None:
    feature_df, partition, v3_within = _make_fixture()
    context = _build_fragment_context_from_objects(
        feature_df=feature_df,
        partition=partition,
        condition='pd',
        clf_name='rf',
        v3_within=v3_within,
        scope='quick',
    )

    with tempfile.TemporaryDirectory(prefix='v4-fragment-resume-') as tmpdir:
        storage_root = Path(tmpdir)
        candidate_spec = _rf_outer_candidate_stage_specs(context)[0]

        first = _run_rf_outer_candidate_fragment_local(
            context=context,
            outer_fold_subject=candidate_spec['outer_fold_subject'],
            candidate_spec=candidate_spec,
            storage_root=storage_root,
        )
        second = _run_rf_outer_candidate_fragment_local(
            context=context,
            outer_fold_subject=candidate_spec['outer_fold_subject'],
            candidate_spec=candidate_spec,
            storage_root=storage_root,
        )
        assert first['status'] == 'completed'
        assert second['status'] == 'skipped_existing'

        fragment_path = storage_root / candidate_spec['relpath']
        fragment_payload = json.loads(fragment_path.read_text())
        fragment_payload['identity']['candidate_hash'] = 'corrupted'
        fragment_path.write_text(json.dumps(fragment_payload))
        third = _run_rf_outer_candidate_fragment_local(
            context=context,
            outer_fold_subject=candidate_spec['outer_fold_subject'],
            candidate_spec=candidate_spec,
            storage_root=storage_root,
        )
        repaired_payload = json.loads(fragment_path.read_text())
        _validate_fragment_payload(
            repaired_payload,
            expected_identity=_fragment_identity(
                context,
                fragment_kind='rf_outer_candidate',
                outer_fold_subject=candidate_spec['outer_fold_subject'],
                candidate_index=candidate_spec['candidate_index'],
                candidate_hash=candidate_spec['candidate_hash'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
            ),
            required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
        )
        assert third['status'] == 'completed'

        scientific_corruption = json.loads(fragment_path.read_text())
        scientific_corruption['scores_by_rule']['mean_probability']['inner_subject_f1'] = round(
            float(scientific_corruption['scores_by_rule']['mean_probability']['inner_subject_f1']) + 0.123456,
            6,
        )
        fragment_path.write_text(json.dumps(scientific_corruption))
        fourth = _run_rf_outer_candidate_fragment_local(
            context=context,
            outer_fold_subject=candidate_spec['outer_fold_subject'],
            candidate_spec=candidate_spec,
            storage_root=storage_root,
        )
        repaired_after_scientific_corruption = json.loads(fragment_path.read_text())
        _validate_fragment_payload(
            repaired_after_scientific_corruption,
            expected_identity=_fragment_identity(
                context,
                fragment_kind='rf_outer_candidate',
                outer_fold_subject=candidate_spec['outer_fold_subject'],
                candidate_index=candidate_spec['candidate_index'],
                candidate_hash=candidate_spec['candidate_hash'],
                imbalance_strategy=candidate_spec['imbalance_strategy'],
            ),
            required_keys=('params', 'imbalance_strategy', 'scores_by_rule'),
        )
        assert fourth['status'] == 'completed'

        assembled = _run_fragmented_target_local(
            context=context,
            storage_root=storage_root,
            force=True,
        )
        assert assembled['status'] == 'completed'

        full_shard_path = storage_root / 'results_v4_preflight' / 'subject_aggregation_shards' / 'full_pd_rf.json'
        skipped = _assemble_fragmented_full_shard_local(
            context=context,
            storage_root=storage_root,
            force=False,
        )
        assert skipped['status'] == 'skipped_existing'

        invalid_payload = json.loads(full_shard_path.read_text())
        invalid_payload.pop('report')
        full_shard_path.write_text(json.dumps(invalid_payload))
        failed_closed = False
        try:
            _assemble_fragmented_full_shard_local(
                context=context,
                storage_root=storage_root,
                force=False,
            )
        except ValueError:
            failed_closed = True
        assert failed_closed, 'Invalid existing full shard should fail closed without force.'

        repaired = _assemble_fragmented_full_shard_local(
            context=context,
            storage_root=storage_root,
            force=True,
        )
        _validate_shard_payload(
            repaired['payload'],
            expected_condition='pd',
            expected_classifier='rf',
        )

    print(json.dumps({
        'status': 'pass',
        'checks': {
            'resume_skip_valid_fragment': True,
            'recompute_invalid_fragment': True,
            'recompute_scientifically_corrupted_fragment': True,
            'full_shard_skip_without_force': True,
            'full_shard_fail_closed_when_invalid': True,
            'full_shard_force_repair': True,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
