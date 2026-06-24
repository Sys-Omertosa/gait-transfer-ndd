"""
Cheap verification for frozen Step 1/2 snapshot semantics after downstream edits.

Usage:
    python scripts/verification/test_v4_snapshot_surface_contract.py
"""

from __future__ import annotations

import importlib.util
import json
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_HELPER = (
    REPO_ROOT / 'scripts' / 'setup' / 'snapshot_v4_step2_frozen_source.py'
)


def _load_helper_module():
    spec = importlib.util.spec_from_file_location(
        'snapshot_v4_step2_frozen_source',
        SNAPSHOT_HELPER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load {SNAPSHOT_HELPER}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _build_mock_surface(root: Path, helper) -> None:
    for relpath in (
        list(helper.PROTOCOL_CODE_HASH_INPUTS)
        + list(helper.ARTIFACT_PATHS)
        + [
            'src/extra_surface_file.py',
            'scripts/setup/extra_surface_file.py',
            'scripts/training/extra_surface_file.py',
            'scripts/verification/extra_surface_file.py',
        ]
    ):
        _write_text(root / relpath, f'mock:{relpath}\n')


def _write_snapshot(snapshot_dir: Path, repo_root: Path, helper) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _write_text(snapshot_dir / 'base_git_head.txt', 'deadbeef\n')
    _write_text(snapshot_dir / 'git_status_porcelain.txt', ' M src/example.py\n')
    _write_text(snapshot_dir / 'working_tree.patch', 'diff --git a/x b/x\n')
    _write_text(
        snapshot_dir / 'working_tree.patch.sha256',
        f"{helper._sha256_file(snapshot_dir / 'working_tree.patch')}  working_tree.patch\n",
    )
    with tarfile.open(snapshot_dir / 'untracked_files.tar.gz', 'w:gz'):
        pass
    _write_text(
        snapshot_dir / 'untracked_files.tar.gz.sha256',
        (
            f"{helper._sha256_file(snapshot_dir / 'untracked_files.tar.gz')}  "
            'untracked_files.tar.gz\n'
        ),
    )
    _write_text(
        snapshot_dir / 'source_file_hashes.json',
        json.dumps(
            helper._hash_entries(repo_root=repo_root),
            indent=2,
            sort_keys=True,
        ),
    )
    _write_text(
        snapshot_dir / 'snapshot_manifest.json',
        json.dumps(
            helper._snapshot_manifest_payload(
                snapshot_dir=snapshot_dir,
                include_source_file_hashes_sha256=False,
            ),
            indent=2,
            sort_keys=True,
        ),
    )


def main() -> None:
    helper = _load_helper_module()
    original_hash_entries = helper._hash_entries

    with tempfile.TemporaryDirectory(prefix='v4-snapshot-contract-') as tmpdir:
        repo_root = Path(tmpdir) / 'repo'
        repo_root.mkdir(parents=True, exist_ok=True)
        _build_mock_surface(repo_root, helper)
        snapshot_dir = repo_root / 'logs' / 'snapshot'
        _write_snapshot(snapshot_dir, repo_root, helper)

        helper.SNAPSHOT_DIR = snapshot_dir
        helper._verify_snapshot(compare_current_surface=False)

        _write_text(repo_root / 'src' / 'extra_surface_file.py', 'post-edit mutation\n')
        helper._verify_snapshot(compare_current_surface=False)

        helper._hash_entries = (
            lambda *, repo_root=repo_root: original_hash_entries(repo_root=repo_root)
        )
        try:
            helper._verify_snapshot(compare_current_surface=True)
        except SystemExit as exc:
            message = str(exc)
            assert 'differs from the current repository surface' in message, message
        else:
            raise AssertionError('Expected --compare-current-surface to detect the edit.')

    print(json.dumps({
        'status': 'pass',
        'checks': {
            'default_verify_after_source_edits': True,
            'compare_current_surface_detects_difference': True,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
