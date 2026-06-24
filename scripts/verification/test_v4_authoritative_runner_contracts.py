"""
Static contract checks for the authoritative v4 Step 1 and Step 2 runners.

Usage:
    python scripts/verification/test_v4_authoritative_runner_contracts.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1_RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_preprocessing_modal.py'
STEP2_RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_within_condition_modal.py'
STEP3_RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_cross_condition_modal.py'
STEP4_RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_shap_modal.py'
STEP5_RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_noise_robustness_modal.py'
STEP8_RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_control_split_sensitivity_modal.py'
DOWNSTREAM_SNAPSHOT = REPO_ROOT / 'scripts' / 'setup' / 'snapshot_v4_step2_frozen_source.py'
DOWNSTREAM_FREEZER = REPO_ROOT / 'scripts' / 'setup' / 'freeze_v4_downstream_execution_manifest.py'
DOWNSTREAM_HELPERS = REPO_ROOT / 'src' / 'v4_downstream.py'
ROBUSTNESS_HELPERS = REPO_ROOT / 'src' / 'robustness.py'


def _assert_required_snippets(text: str, required: tuple[str, ...], *, label: str) -> None:
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise SystemExit(
            f'{label} is missing required snippets:\n' + '\n'.join(missing)
        )


def _assert_forbidden_snippets(text: str, forbidden: tuple[str, ...], *, label: str) -> None:
    present = [snippet for snippet in forbidden if snippet in text]
    if present:
        raise SystemExit(
            f'{label} contains forbidden snippets:\n' + '\n'.join(present)
        )


def main() -> None:
    step1_text = STEP1_RUNNER.read_text()
    _assert_required_snippets(
        step1_text,
        (
            'v4_protocol_manifest.json',
            'approved_protocol_manifest_required',
            'robust_mad_multiplier',
            'dfa_policy',
            'subject_probability_threshold',
            'tie_break_rule',
            'schema_version',
            'gait_features_v4_per_stride_only.csv',
            'gait_features_v4_subject_level.csv',
            'gait_features_v4_no_dfa_sensitivity.csv',
            'SUPPORTED_DFA_POLICY',
            'volume.commit()',
        ),
        label='run_preprocessing_modal.py',
    )

    step2_text = STEP2_RUNNER.read_text()
    _assert_required_snippets(
        step2_text,
        (
            'v4_protocol_manifest.json',
            'preprocessing_manifest_v4.json',
            'candidate_strategy_policy',
            "subject_aggregation_rule=context['aggregation_rule']",
            "tie_break_rule=context['tie_break_rule']",
            'subject_probability_threshold',
            'payload_sha256',
            'volume.reload()',
            'results_v4_smoke',
            'force_recompute_invalid',
            '--action status',
            '--action smoke',
            'namespace: str = DEFAULT_RESULTS_NAMESPACE',
            'max_in_flight: int = DEFAULT_MAX_IN_FLIGHT',
            "svm_fold_indices: str = ''",
            'spawn_map(',
            'run_svm_outer_fold_remote',
            'assemble_svm_classifier_shard_remote',
            'assemble_condition_remote',
            'describe_condition_context_remote',
            'classifier_shards/svm_outer_folds',
            'selection_traces',
            'volume.commit()',
        ),
        label='run_within_condition_modal.py',
    )
    _assert_forbidden_snippets(
        step2_text,
        (
            "candidate_imbalance_strategies=('synthetic', 'balanced', 'raw')",
            '"candidate_imbalance_strategies": ("synthetic", "balanced", "raw")',
        ),
        label='run_within_condition_modal.py',
    )

    step3_text = STEP3_RUNNER.read_text()
    _assert_required_snippets(
        step3_text,
        (
            '--output-namespace',
            'allow_refit=False',
            'build_authoritative_step12_context',
            'validate_step3_results_payload',
            'write_payload_json',
            'downstream_runs',
            'results_v4_smoke',
            'atomic_write_json',
            'cross_condition_results_v4_partial.json',
        ),
        label='run_cross_condition_modal.py',
    )

    step4_text = STEP4_RUNNER.read_text()
    _assert_required_snippets(
        step4_text,
        (
            '--action submit',
            '--action status',
            '--action assemble',
            'build_authoritative_step12_context',
            'build_downstream_final_payload',
            'validate_step3_results_payload',
            'downstream_execution_id',
            'downstream_execution_manifest_hash',
            'downstream_runs',
            'source_groups',
            'FINAL_SCHEMA_VERSION',
            'data',
            'volume.commit()',
        ),
        label='run_shap_modal.py',
    )

    step5_text = STEP5_RUNNER.read_text()
    _assert_required_snippets(
        step5_text,
        (
            '--action submit',
            '--action status',
            '--action assemble',
            'build_authoritative_step12_context',
            'build_downstream_final_payload',
            'validate_step3_results_payload',
            'fit_within_condition_folds',
            'gaussian_feature_space_stress_test',
            'engineered_feature_permutation_sensitivity',
            'downstream_runs',
            'FINAL_SCHEMA_VERSIONS',
            'volume.commit()',
        ),
        label='run_noise_robustness_modal.py',
    )
    _assert_required_snippets(
        ROBUSTNESS_HELPERS.read_text(),
        (
            'engineered_feature_masking',
            'engineered_feature_gain_bias_drift',
            'gaussian_feature_space_jitter',
            'evaluation_row_dropout',
            'benchmark_label_corruption',
            'hard_label_consensus_curve',
            'raw_set_stats',
            'post_fallback_set_stats',
        ),
        label='src/robustness.py',
    )

    step8_text = STEP8_RUNNER.read_text()
    _assert_required_snippets(
        step8_text,
        (
            '--action retune-submit-within',
            '--action retune-status',
            '--action retune-assemble-within',
            '--action retune-submit-cross',
            '--action retune-assemble',
            'allow_refit=False',
            'candidate_family',
            'downstream_execution_id',
            'downstream_execution_manifest_hash',
            'control_split_sensitivity_runs',
            'SOURCE_WITHIN_RETRY_ATTEMPTS',
            'write_payload_json',
            'is_main_partition',
            'is_role_reversal_of_main',
            'overlap_with_main_control_a',
            'independent_sensitivity_candidate',
            'volume.commit()',
        ),
        label='run_control_split_sensitivity_modal.py',
    )

    _assert_required_snippets(
        DOWNSTREAM_HELPERS.read_text(),
        (
            'DOWNSTREAM_MANIFEST_SCHEMA_VERSION',
            'derive_downstream_execution_id',
            'excluded_keys',
            'step2_model_hashes',
            'logical_path',
            'normalize_step2_model_logical_path',
            'build_downstream_final_payload',
            'dirty_tree',
        ),
        label='src/v4_downstream.py',
    )
    _assert_required_snippets(
        DOWNSTREAM_SNAPSHOT.read_text(),
        (
            'working_tree.patch',
            'untracked_files.tar.gz',
            'source_file_hashes.json',
            'snapshot_manifest.json',
            '--compare-current-surface',
        ),
        label='snapshot_v4_step2_frozen_source.py',
    )
    _assert_required_snippets(
        DOWNSTREAM_FREEZER.read_text(),
        (
            '--approved',
            '--allow-dirty-tree',
            'downstream_execution_id',
            'allow_overwrite_existing_manifest',
        ),
        label='freeze_v4_downstream_execution_manifest.py',
    )

    print('v4 authoritative runner contracts passed.')


if __name__ == '__main__':
    main()
