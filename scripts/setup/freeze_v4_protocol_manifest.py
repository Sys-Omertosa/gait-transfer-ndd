"""
Freeze the immutable v4 protocol manifest after diagnostic approval.

This script is intentionally not called automatically.

Usage:
    python scripts/setup/freeze_v4_protocol_manifest.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import DEFAULT_SUBJECT_AGGREGATION_RULE, DEFAULT_TIE_BREAK_RULE, get_classifier_configs
from v4_provenance import (
    atomic_write_json,
    code_hash,
    collect_package_versions,
    ensure_v4_scaffold,
    sha256_file,
    utc_now_iso,
)

SUPPORTED_AGGREGATION_RULES = (
    'mean_probability',
    'median_probability',
    'majority_vote',
    'mean_decision_score',
)
SUPPORTED_TIE_BREAK_RULES = (
    'stride_macro_f1_then_lexicographic',
    'subject_probability_loss_then_lexicographic',
    'lexicographic_only',
)
SUPPORTED_DFA_POLICIES = (
    'concatenated',
    'longest_contiguous_segment',
    'segmented_summary',
)
PROTOCOL_MANIFEST_SCHEMA_VERSION = 'v4-protocol-manifest-v2'
SUPPORTED_METHODOLOGY_VERSIONS = ('v4-hardening',)
FIXED_SUBJECT_PROBABILITY_THRESHOLD = 0.5
PRAGMATIC_DFA_DESCRIPTOR_NOTE = (
    'pragmatic_dfa_derived_descriptor_from_discontinuous_stride_series'
)


def _git_head() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregation-rule', required=True, choices=SUPPORTED_AGGREGATION_RULES)
    parser.add_argument('--tie-break-rule', required=True, choices=SUPPORTED_TIE_BREAK_RULES)
    parser.add_argument('--robust-mad-multiplier', type=float, required=True)
    parser.add_argument('--dfa-policy', required=True, choices=SUPPORTED_DFA_POLICIES)
    parser.add_argument('--approved', action='store_true')
    parser.add_argument(
        '--allow-dirty-tree',
        action='store_true',
        help='Development-only override for freezing from a dirty git working tree.',
    )
    parser.add_argument(
        '--allow-overwrite-existing-manifest',
        action='store_true',
        help='Development-only override for replacing an existing protocol manifest.',
    )
    parser.add_argument('--methodology-version', default='v4-hardening')
    args = parser.parse_args()

    if not args.approved:
        raise SystemExit(
            'Refusing to freeze the v4 protocol manifest without --approved. '
            'Run the diagnostics, review the evidence, and pass --approved only after user sign-off.'
        )
    if args.robust_mad_multiplier <= 0:
        raise SystemExit(
            '--robust-mad-multiplier must be positive for the v4 protocol freeze.'
        )
    if args.methodology_version not in SUPPORTED_METHODOLOGY_VERSIONS:
        raise SystemExit(
            f"Unsupported methodology version '{args.methodology_version}'. "
            f'Expected one of {SUPPORTED_METHODOLOGY_VERSIONS}.'
        )
    if _git_dirty() and not args.allow_dirty_tree:
        raise SystemExit(
            'Refusing to freeze the v4 protocol manifest from a dirty git working tree. '
            'Commit or stash the working tree first, or pass --allow-dirty-tree only '
            'for a deliberate development-only override.'
        )

    scaffold = ensure_v4_scaffold(REPO_ROOT)
    manifest_path = REPO_ROOT / 'data' / 'processed' / 'v4' / 'v4_protocol_manifest.json'
    if manifest_path.exists() and not args.allow_overwrite_existing_manifest:
        raise FileExistsError(
            f'Protocol manifest already exists at {manifest_path}. '
            'Refusing to overwrite it without --allow-overwrite-existing-manifest.'
        )
    source_files = [
        REPO_ROOT / 'requirements-core.txt',
        REPO_ROOT / 'src' / 'train.py',
        REPO_ROOT / 'src' / 'robustness.py',
        REPO_ROOT / 'src' / 'explain.py',
        REPO_ROOT / 'src' / 'features.py',
        REPO_ROOT / 'src' / 'preprocessing.py',
        REPO_ROOT / 'src' / 'v4_provenance.py',
        REPO_ROOT / 'scripts' / 'training' / 'run_preprocessing_modal.py',
        REPO_ROOT / 'scripts' / 'training' / 'run_within_condition_modal.py',
        REPO_ROOT / 'scripts' / 'training' / 'run_cross_condition_modal.py',
        REPO_ROOT / 'scripts' / 'training' / 'run_shap_modal.py',
        REPO_ROOT / 'scripts' / 'training' / 'run_noise_robustness_modal.py',
        REPO_ROOT / 'scripts' / 'training' / 'run_control_split_sensitivity_modal.py',
    ]
    candidate_strategy_policy = {
        'rf': ['synthetic', 'balanced', 'raw'],
        'svm': ['synthetic', 'balanced', 'raw'],
        'dt': ['synthetic', 'balanced', 'raw'],
        'xgb': ['synthetic', 'balanced', 'raw'],
        'lgbm': ['synthetic', 'balanced', 'raw'],
        'knn': ['synthetic', 'raw'],
        'qda': ['synthetic', 'raw'],
    }
    manifest = {
        'schema_version': PROTOCOL_MANIFEST_SCHEMA_VERSION,
        'created_at_utc': utc_now_iso(),
        'approved': True,
        'git_commit': _git_head(),
        'dirty_tree': _git_dirty(),
        'code_hash': code_hash(source_files, base_dir=REPO_ROOT),
        'package_versions': collect_package_versions(),
        'random_seeds': {'global': 42},
        'classifier_grids': {
            clf_name: config['param_grid']
            for clf_name, config in get_classifier_configs().items()
        },
        'candidate_strategy_policy': candidate_strategy_policy,
        'conformal_policy': 'exploratory_only',
        'step3_primary_endpoint': 'subject_level_macro_f1',
        'aggregation_rule': args.aggregation_rule,
        'subject_probability_threshold': FIXED_SUBJECT_PROBABILITY_THRESHOLD,
        'tie_break_rule': args.tie_break_rule,
        'robust_mad_multiplier': float(args.robust_mad_multiplier),
        'dfa_policy': args.dfa_policy,
        'dfa_descriptor_interpretation': PRAGMATIC_DFA_DESCRIPTOR_NOTE,
        'namespace_map': scaffold,
        'methodology_version': args.methodology_version,
    }
    atomic_write_json(manifest_path, manifest)
    print(f'Wrote {manifest_path}')
    print(f'SHA256 {sha256_file(manifest_path)}')


if __name__ == '__main__':
    main()
