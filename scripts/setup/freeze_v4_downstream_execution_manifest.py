"""
Freeze the immutable downstream-execution manifest after Step 1/2 completion.

Usage:
    python scripts/setup/freeze_v4_downstream_execution_manifest.py --approved
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v4_downstream import collect_downstream_manifest_payload  # type: ignore
from v4_provenance import atomic_write_json, sha256_file  # type: ignore

SNAPSHOT_DIR = (
    REPO_ROOT
    / 'logs'
    / 'v4_authoritative'
    / 'step2_final'
    / 'frozen_step1_step2_source_snapshot'
)
MANIFEST_PATH = REPO_ROOT / 'data' / 'processed' / 'v4' / 'v4_downstream_execution_manifest.json'
DOWNSTREAM_CODE_PATHS: tuple[str, ...] = (
    'src/v4_downstream.py',
    'scripts/setup/snapshot_v4_step2_frozen_source.py',
    'scripts/setup/freeze_v4_downstream_execution_manifest.py',
    'scripts/training/run_cross_condition_modal.py',
    'scripts/training/run_shap_modal.py',
    'src/explain.py',
    'scripts/training/run_noise_robustness_modal.py',
    'src/robustness.py',
    'notebooks/06_pca_kmeans.ipynb',
    'scripts/training/run_control_split_sensitivity_modal.py',
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--approved', action='store_true')
    parser.add_argument('--allow-dirty-tree', action='store_true')
    parser.add_argument(
        '--allow-overwrite-existing-manifest',
        action='store_true',
        help='Development-only override for replacing an existing downstream manifest.',
    )
    args = parser.parse_args()

    if not args.approved:
        raise SystemExit(
            'Refusing to freeze the downstream execution manifest without --approved.'
        )
    if not SNAPSHOT_DIR.exists():
        raise FileNotFoundError(
            f'Missing required pre-edit snapshot at {SNAPSHOT_DIR}.'
        )

    payload = collect_downstream_manifest_payload(
        repo_root=REPO_ROOT,
        snapshot_dir=SNAPSHOT_DIR,
        code_paths=DOWNSTREAM_CODE_PATHS,
    )
    if payload['dirty_tree'] and not args.allow_dirty_tree:
        raise SystemExit(
            'Refusing to freeze the downstream manifest from a dirty working tree. '
            'Pass --allow-dirty-tree only after preserving the pre-edit snapshot.'
        )

    if MANIFEST_PATH.exists():
        existing_sha256 = sha256_file(MANIFEST_PATH)
        candidate_path = MANIFEST_PATH.with_suffix('.candidate.json')
        atomic_write_json(candidate_path, payload)
        candidate_sha256 = sha256_file(candidate_path)
        candidate_path.unlink()
        if existing_sha256 == candidate_sha256:
            print(f'Downstream manifest already matches {MANIFEST_PATH}')
            print(f'SHA256 {existing_sha256}')
            return
        if not args.allow_overwrite_existing_manifest:
            raise FileExistsError(
                f'Downstream manifest already exists at {MANIFEST_PATH} with a different payload. '
                'Refusing to overwrite without --allow-overwrite-existing-manifest.'
            )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MANIFEST_PATH, payload)
    print(f'Wrote {MANIFEST_PATH}')
    print(f'SHA256 {sha256_file(MANIFEST_PATH)}')
    print(f"downstream_execution_id {payload['downstream_execution_id']}")


if __name__ == '__main__':
    main()
