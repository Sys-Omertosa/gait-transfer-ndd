"""
Create or verify the frozen pre-edit Step 1/2 source snapshot.

Usage:
    python scripts/setup/snapshot_v4_step2_frozen_source.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = (
    REPO_ROOT
    / 'logs'
    / 'v4_authoritative'
    / 'step2_final'
    / 'frozen_step1_step2_source_snapshot'
)
SNAPSHOT_TARGETS = (
    'base_git_head.txt',
    'git_status_porcelain.txt',
    'working_tree.patch',
    'working_tree.patch.sha256',
    'untracked_files.tar.gz',
    'untracked_files.tar.gz.sha256',
    'source_file_hashes.json',
    'snapshot_manifest.json',
)
PROTOCOL_CODE_HASH_INPUTS = (
    'requirements-core.txt',
    'src/train.py',
    'src/robustness.py',
    'src/explain.py',
    'src/features.py',
    'src/preprocessing.py',
    'src/v4_provenance.py',
    'scripts/training/run_preprocessing_modal.py',
    'scripts/training/run_within_condition_modal.py',
    'scripts/training/run_cross_condition_modal.py',
    'scripts/training/run_shap_modal.py',
    'scripts/training/run_noise_robustness_modal.py',
    'scripts/training/run_control_split_sensitivity_modal.py',
)
ARTIFACT_PATHS = (
    'data/processed/v4/v4_protocol_manifest.json',
    'data/processed/v4/preprocessing_manifest_v4.json',
    'experiments/results/v4/pd_results_v4.json',
    'experiments/results/v4/hd_results_v4.json',
    'experiments/results/v4/als_results_v4.json',
    'experiments/results/v4/v4_artifact_index.json',
    'logs/v4_authoritative/step2_final/all_local_artifacts.sha256',
)
SURFACE_ROOTS = (
    'src',
    'scripts/setup',
    'scripts/training',
    'scripts/verification',
)
REQUIRED_SNAPSHOT_MANIFEST_KEYS = (
    'base_git_head',
    'git_status_porcelain_path',
    'working_tree_patch_path',
    'working_tree_patch_sha256',
    'untracked_archive_path',
    'untracked_archive_sha256',
    'source_file_hashes_path',
    'protocol_code_hash_inputs',
    'artifact_paths',
)
OPTIONAL_SNAPSHOT_MANIFEST_KEYS = (
    'surface_roots',
    'source_file_hashes_sha256',
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_manifest_payload(
    *,
    snapshot_dir: Path = SNAPSHOT_DIR,
    include_source_file_hashes_sha256: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'base_git_head': (snapshot_dir / 'base_git_head.txt').read_text().strip(),
        'git_status_porcelain_path': 'git_status_porcelain.txt',
        'working_tree_patch_path': 'working_tree.patch',
        'working_tree_patch_sha256': _sha256_file(
            snapshot_dir / 'working_tree.patch'
        ),
        'untracked_archive_path': 'untracked_files.tar.gz',
        'untracked_archive_sha256': _sha256_file(
            snapshot_dir / 'untracked_files.tar.gz'
        ),
        'source_file_hashes_path': 'source_file_hashes.json',
        'protocol_code_hash_inputs': list(PROTOCOL_CODE_HASH_INPUTS),
        'artifact_paths': list(ARTIFACT_PATHS),
        'surface_roots': list(SURFACE_ROOTS),
    }
    if include_source_file_hashes_sha256:
        payload['source_file_hashes_sha256'] = _sha256_file(
            snapshot_dir / 'source_file_hashes.json'
        )
    return payload


def _hash_entries(*, repo_root: Path = REPO_ROOT) -> list[dict[str, object]]:
    paths_to_hash: list[Path] = []
    for root in SURFACE_ROOTS:
        root_path = repo_root / root
        if root_path.exists():
            paths_to_hash.extend(sorted(p for p in root_path.rglob('*') if p.is_file()))

    seen: set[str] = set()
    entries: list[dict[str, object]] = []
    for rel in (
        list(PROTOCOL_CODE_HASH_INPUTS)
        + list(ARTIFACT_PATHS)
        + [path.relative_to(repo_root).as_posix() for path in paths_to_hash]
    ):
        if rel in seen:
            continue
        seen.add(rel)
        path = repo_root / rel
        if path.exists() and path.is_file():
            entries.append({
                'path': rel,
                'sha256': _sha256_file(path),
                'size_bytes': path.stat().st_size,
            })
    return entries


def _write_snapshot() -> None:
    existing = [name for name in SNAPSHOT_TARGETS if (SNAPSHOT_DIR / name).exists()]
    if existing:
        raise SystemExit(
            f'Refusing to overwrite existing snapshot files: {existing}'
        )
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    (SNAPSHOT_DIR / 'base_git_head.txt').write_text(
        subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        + '\n'
    )
    (SNAPSHOT_DIR / 'git_status_porcelain.txt').write_text(
        subprocess.check_output(
            ['git', 'status', '--porcelain=v1'],
            cwd=REPO_ROOT,
            text=True,
        )
    )
    (SNAPSHOT_DIR / 'working_tree.patch').write_bytes(
        subprocess.check_output(
            ['git', 'diff', '--binary', '--', '.'],
            cwd=REPO_ROOT,
        )
    )
    (SNAPSHOT_DIR / 'working_tree.patch.sha256').write_text(
        f'{_sha256_file(SNAPSHOT_DIR / "working_tree.patch")}  working_tree.patch\n'
    )

    untracked_raw = subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard', '-z'],
        cwd=REPO_ROOT,
    )
    untracked = [
        Path(item.decode())
        for item in untracked_raw.split(b'\0')
        if item
    ]
    with tarfile.open(SNAPSHOT_DIR / 'untracked_files.tar.gz', 'w:gz') as tf:
        for relpath in untracked:
            tf.add(REPO_ROOT / relpath, arcname=relpath.as_posix())
    (SNAPSHOT_DIR / 'untracked_files.tar.gz.sha256').write_text(
        f'{_sha256_file(SNAPSHOT_DIR / "untracked_files.tar.gz")}  '
        'untracked_files.tar.gz\n'
    )

    (SNAPSHOT_DIR / 'source_file_hashes.json').write_text(
        json.dumps(_hash_entries(), indent=2, sort_keys=True)
    )
    (SNAPSHOT_DIR / 'snapshot_manifest.json').write_text(
        json.dumps(
            _snapshot_manifest_payload(
                include_source_file_hashes_sha256=True,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def _verify_snapshot(*, compare_current_surface: bool) -> None:
    missing = [name for name in SNAPSHOT_TARGETS if not (SNAPSHOT_DIR / name).exists()]
    if missing:
        raise SystemExit(f'Snapshot is incomplete. Missing files: {missing}')

    stored_manifest = json.loads((SNAPSHOT_DIR / 'snapshot_manifest.json').read_text())
    expected_manifest = _snapshot_manifest_payload(
        snapshot_dir=SNAPSHOT_DIR,
        include_source_file_hashes_sha256='source_file_hashes_sha256' in stored_manifest,
    )
    unexpected_keys = sorted(
        set(stored_manifest)
        - set(REQUIRED_SNAPSHOT_MANIFEST_KEYS)
        - set(OPTIONAL_SNAPSHOT_MANIFEST_KEYS)
    )
    if unexpected_keys:
        raise SystemExit(
            'snapshot_manifest.json contains unexpected keys: '
            f'{unexpected_keys}'
        )
    for key in REQUIRED_SNAPSHOT_MANIFEST_KEYS:
        if stored_manifest.get(key) != expected_manifest.get(key):
            raise SystemExit(
                f'snapshot_manifest.json does not match the current snapshot files for {key}.'
            )
    for key in OPTIONAL_SNAPSHOT_MANIFEST_KEYS:
        if key in stored_manifest and stored_manifest.get(key) != expected_manifest.get(key):
            raise SystemExit(
                f'snapshot_manifest.json does not match the current snapshot files for optional key {key}.'
            )

    stored_entries = json.loads((SNAPSHOT_DIR / 'source_file_hashes.json').read_text())
    if not isinstance(stored_entries, list) or not all(
        isinstance(entry, dict)
        and isinstance(entry.get('path'), str)
        and isinstance(entry.get('sha256'), str)
        and isinstance(entry.get('size_bytes'), int)
        for entry in stored_entries
    ):
        raise SystemExit('source_file_hashes.json is not a valid frozen hash inventory.')
    if compare_current_surface and stored_entries != _hash_entries():
        raise SystemExit(
            'source_file_hashes.json differs from the current repository surface.'
        )

    patch_digest_line = (
        SNAPSHOT_DIR / 'working_tree.patch.sha256'
    ).read_text().strip()
    patch_digest = patch_digest_line.split()[0] if patch_digest_line else ''
    if patch_digest != _sha256_file(SNAPSHOT_DIR / 'working_tree.patch'):
        raise SystemExit('working_tree.patch.sha256 does not match working_tree.patch.')

    untracked_digest_line = (
        SNAPSHOT_DIR / 'untracked_files.tar.gz.sha256'
    ).read_text().strip()
    untracked_digest = untracked_digest_line.split()[0] if untracked_digest_line else ''
    if untracked_digest != _sha256_file(SNAPSHOT_DIR / 'untracked_files.tar.gz'):
        raise SystemExit(
            'untracked_files.tar.gz.sha256 does not match untracked_files.tar.gz.'
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Verify the existing snapshot without creating one.',
    )
    parser.add_argument(
        '--compare-current-surface',
        action='store_true',
        help='Also compare the frozen source hash inventory against the current repository surface.',
    )
    args = parser.parse_args()

    if SNAPSHOT_DIR.exists() and any((SNAPSHOT_DIR / name).exists() for name in SNAPSHOT_TARGETS):
        _verify_snapshot(compare_current_surface=args.compare_current_surface)
        print(f'Verified {SNAPSHOT_DIR}')
        return

    if args.verify_only:
        raise SystemExit(f'Snapshot does not exist yet at {SNAPSHOT_DIR}.')

    _write_snapshot()
    _verify_snapshot(compare_current_surface=args.compare_current_surface)
    print(f'Created {SNAPSHOT_DIR}')


if __name__ == '__main__':
    main()
