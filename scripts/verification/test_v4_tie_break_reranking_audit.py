"""
Cheap offline reranking audit for grouped-selection trace sidecars.

Usage:
    python scripts/verification/test_v4_tie_break_reranking_audit.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import _candidate_sort_key  # type: ignore

AUTHORITATIVE_TRACE_DIR = REPO_ROOT / 'experiments' / 'results' / 'v4' / 'selection_traces'
PREFLIGHT_TRACE_ROOTS = (
    REPO_ROOT / 'experiments' / 'results' / 'v4_preflight',
    REPO_ROOT / 'data' / 'processed' / 'v4_preflight',
    REPO_ROOT / 'modal_handoff',
)
HANDOFF_FULL_SHARD_DIR = (
    REPO_ROOT
    / 'modal_handoff'
    / 'aggregation_workspace_transfer_20260606_180556'
    / 'subject_aggregation_shards'
)
TRACE_GLOB_PATTERNS = (
    '*candidate_trace*.npz',
    '*selection_trace*.npz',
)
ALTERNATIVE_TIE_BREAK_RULE = 'subject_probability_loss_then_lexicographic'
FAST_AUDIT_TRACE_LIMIT = 24


def _load_candidates(npz_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(npz_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload['metadata_json'][0]))
        candidates: list[dict[str, Any]] = []
        n_candidates = len(payload['candidate_index'])
        for idx in range(n_candidates):
            log_loss_value = float(payload['inner_subject_log_loss'][idx])
            candidates.append({
                'candidate_index': int(payload['candidate_index'][idx]),
                'imbalance_strategy': str(payload['imbalance_strategy'][idx]),
                'inner_subject_f1': float(payload['inner_subject_f1'][idx]),
                'inner_stride_f1': float(payload['inner_stride_f1'][idx]),
                'inner_subject_log_loss': (
                    None if np.isnan(log_loss_value) else log_loss_value
                ),
                'params': json.loads(str(payload['params_json'][idx])),
            })
    return metadata, candidates


def _winner_key(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        'imbalance_strategy': candidate['imbalance_strategy'],
        'params': candidate['params'],
        'inner_subject_f1': candidate['inner_subject_f1'],
        'inner_stride_f1': candidate['inner_stride_f1'],
        'inner_subject_log_loss': candidate['inner_subject_log_loss'],
    }


def _discover_trace_paths() -> dict[str, list[Path]]:
    authoritative = sorted(AUTHORITATIVE_TRACE_DIR.glob('*.npz')) if AUTHORITATIVE_TRACE_DIR.exists() else []

    if authoritative:
        return {
            'authoritative': authoritative,
            'preflight': [],
        }

    preflight_seen: set[Path] = set()
    for root in PREFLIGHT_TRACE_ROOTS:
        if not root.exists():
            continue
        for pattern in TRACE_GLOB_PATTERNS:
            for path in root.rglob(pattern):
                preflight_seen.add(path.resolve())
    preflight = sorted(preflight_seen)
    return {
        'authoritative': authoritative,
        'preflight': preflight,
    }


def _coverage_report(trace_paths: dict[str, list[Path]]) -> dict[str, Any]:
    handoff_full_shards = (
        sorted(HANDOFF_FULL_SHARD_DIR.glob('full_*.json'))
        if HANDOFF_FULL_SHARD_DIR.exists() else
        []
    )
    return {
        'status': 'incomplete_coverage',
        'alternative_tie_break_rule': ALTERNATIVE_TIE_BREAK_RULE,
        'searched_roots': {
            'authoritative': str(AUTHORITATIVE_TRACE_DIR),
            'preflight': [str(root) for root in PREFLIGHT_TRACE_ROOTS],
            'handoff_full_shards': str(HANDOFF_FULL_SHARD_DIR),
        },
        'trace_counts': {
            'authoritative': len(trace_paths['authoritative']),
            'preflight': len(trace_paths['preflight']),
            'total': len(trace_paths['authoritative']) + len(trace_paths['preflight']),
        },
        'handoff_full_shard_count': len(handoff_full_shards),
        'message': (
            'No candidate-trace sidecars were found locally, so the tie-break '
            'reranking audit cannot be completed from preflight evidence alone. '
            'The completed full subject-aggregation shards are present, but they do '
            'not contain enough candidate-level detail for offline reranking.'
        ),
    }


def main() -> None:
    trace_paths = _discover_trace_paths()
    all_traces = trace_paths['authoritative'] + trace_paths['preflight']
    if not all_traces:
        print(json.dumps(_coverage_report(trace_paths), indent=2))
        return

    full_audit = os.environ.get('V4_TIEBREAK_AUDIT_FULL') == '1'
    audited_traces = all_traces if full_audit else all_traces[:FAST_AUDIT_TRACE_LIMIT]

    summaries: list[dict[str, Any]] = []
    changed = 0
    changed_outer_fold_winners = 0
    changed_outer_fold_strategies = 0
    changed_full_source_winners = 0

    for trace_path in audited_traces:
        metadata, candidates = _load_candidates(trace_path)
        current_rule = str(metadata.get('tie_break_rule'))
        current_winner = sorted(
            candidates,
            key=lambda candidate: _candidate_sort_key(
                candidate,
                tie_break_rule=current_rule,
            ),
        )[0]
        alternative_winner = sorted(
            candidates,
            key=lambda candidate: _candidate_sort_key(
                candidate,
                tie_break_rule=ALTERNATIVE_TIE_BREAK_RULE,
            ),
        )[0]
        current_key = _winner_key(current_winner)
        alternative_key = _winner_key(alternative_winner)
        changed_winner = current_key != alternative_key
        if changed_winner:
            changed += 1

        kind = str(metadata.get('kind'))
        if changed_winner:
            if kind == 'outer_fold_grouped_selection_trace':
                changed_outer_fold_winners += 1
                if current_key['imbalance_strategy'] != alternative_key['imbalance_strategy']:
                    changed_outer_fold_strategies += 1
            elif kind == 'full_source_grouped_selection_trace':
                changed_full_source_winners += 1

        summaries.append({
            'trace': str(trace_path),
            'trace_source': (
                'authoritative'
                if trace_path in trace_paths['authoritative'] else
                'preflight'
            ),
            'kind': metadata.get('kind'),
            'condition': metadata.get('condition'),
            'classifier': metadata.get('classifier'),
            'outer_fold_index': metadata.get('outer_fold_index'),
            'held_out_subject_id': metadata.get('held_out_subject_id'),
            'current_tie_break_rule': current_rule,
            'alternative_tie_break_rule': ALTERNATIVE_TIE_BREAK_RULE,
            'winner_changed': changed_winner,
            'strategy_changed': (
                current_key['imbalance_strategy'] != alternative_key['imbalance_strategy']
            ),
            'current_winner': current_key,
            'alternative_winner': alternative_key,
        })

    print(json.dumps({
        'status': 'ok',
        'alternative_tie_break_rule': ALTERNATIVE_TIE_BREAK_RULE,
        'full_audit': full_audit,
        'searched_roots': {
            'authoritative': str(AUTHORITATIVE_TRACE_DIR),
            'preflight': [str(root) for root in PREFLIGHT_TRACE_ROOTS],
            'handoff_full_shards': str(HANDOFF_FULL_SHARD_DIR),
        },
        'trace_counts': {
            'authoritative': len(trace_paths['authoritative']),
            'preflight': len(trace_paths['preflight']),
            'total': len(all_traces),
            'audited': len(audited_traces),
        },
        'handoff_full_shard_count': (
            len(sorted(HANDOFF_FULL_SHARD_DIR.glob('full_*.json')))
            if HANDOFF_FULL_SHARD_DIR.exists() else 0
        ),
        'coverage_note': (
            'Fast local validation mode audited only a prefix of the available traces. '
            'Set V4_TIEBREAK_AUDIT_FULL=1 to force a full offline reranking sweep.'
            if not full_audit and len(audited_traces) < len(all_traces)
            else 'All discovered traces were audited.'
        ),
        'n_changed_winners': changed,
        'changed_outer_fold_winners': changed_outer_fold_winners,
        'changed_outer_fold_strategies': changed_outer_fold_strategies,
        'changed_full_source_winners': changed_full_source_winners,
        'summaries': summaries,
    }, indent=2))


if __name__ == '__main__':
    main()
