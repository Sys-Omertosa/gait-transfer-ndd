"""
Static Step 8 schedule and race-avoidance contract check.

Usage:
    python scripts/verification/test_v4_step8_schedule.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / 'scripts' / 'training' / 'run_control_split_sensitivity_modal.py'


def main() -> None:
    text = RUNNER.read_text()
    required_snippets = [
        '--action retune-submit-within',
        '--action retune-status',
        '--action retune-status-optimized',
        '--action retune-dry-run',
        '--action retune-diagnose-missing',
        '--action retune-submit-within-missing',
        '--action retune-submit-within-shards',
        '--action retune-assemble-within-missing',
        '--action retune-assemble-within',
        '--action retune-submit-cross',
        '--action retune-assemble',
        'allow_refit=False',
        'candidate_family',
        'control_split_sensitivity_runs',
        'within_shards',
        'downstream_execution_id',
        'downstream_execution_manifest_hash',
        'SOURCE_WITHIN_RETRY_ATTEMPTS',
        'retune-submit-cross requires assembled within-condition JSONs first',
        'is_main_partition',
        'is_role_reversal_of_main',
        'overlap_with_main_control_a',
        'independent_sensitivity_candidate',
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        raise SystemExit(
            'Step 8 runner is missing required hardening snippets:\n' + '\n'.join(missing)
        )

    submit_within_marker = 'def submit_retune_within_remote('
    assemble_within_missing_marker = 'def assemble_within_missing_remote('
    assemble_within_marker = 'def assemble_within_remote('
    submit_cross_marker = 'def submit_retune_cross_remote('
    assemble_final_marker = 'def assemble_retune_remote('
    submit_within_start = text.index(submit_within_marker)
    assemble_within_start = text.index(assemble_within_marker)
    assemble_within_missing_start = text.index(assemble_within_missing_marker)
    submit_cross_start = text.index(submit_cross_marker)
    assemble_final_start = text.index(assemble_final_marker)

    submit_within_block = text[submit_within_start:assemble_within_start]
    assemble_within_block = text[assemble_within_start:assemble_within_missing_start]
    assemble_within_missing_block = text[assemble_within_missing_start:submit_cross_start]
    submit_cross_block = text[submit_cross_start:assemble_final_start]
    assemble_final_block = text[assemble_final_start:]

    if 'run_partition_direction.spawn(' in submit_within_block:
        raise SystemExit('retune-submit-within must not spawn cross jobs.')
    if 'run_partition_condition_outer_fold_shard.spawn(' not in submit_within_block:
        raise SystemExit('retune-submit-within must submit within outer-fold shard jobs.')
    if 'run_partition_condition_full_source_shard.spawn(' not in submit_within_block:
        raise SystemExit('retune-submit-within must submit within full-source shard jobs.')
    if '_assemble_missing_within_partials(' not in assemble_within_missing_block:
        raise SystemExit('retune-assemble-within-missing must assemble missing classifier partials from shards.')
    if '_within_assembled_api_path(' not in submit_cross_block:
        raise SystemExit('retune-submit-cross must validate assembled within JSONs before spawn.')
    if 'run_partition_direction.spawn(' not in submit_cross_block:
        raise SystemExit('retune-submit-cross no longer submits cross-direction jobs.')
    if '_within_results_filename(condition, idx)' not in assemble_within_block:
        raise SystemExit('retune-assemble-within must write assembled condition JSONs.')
    if 'write_payload_json(assembled_path, assembled)' not in assemble_within_block:
        raise SystemExit('retune-assemble-within must persist assembled within payloads atomically.')
    if '_assemble_missing_within_partials(' not in assemble_within_block:
        raise SystemExit('retune-assemble-within must materialize ready classifier partials from shards first.')
    if 'Missing Step 8 cross partial' not in assemble_final_block and '_validate_cross_partial_payload' not in assemble_final_block:
        raise SystemExit('retune-assemble must fail closed on incomplete cross partials.')
    print('Step 8 schedule static contract passed.')


if __name__ == '__main__':
    main()
