"""
Create the canonical local v4 scaffold and seed the empty artifact index.

Usage:
    python scripts/setup/scaffold_v4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v4_provenance import (
    atomic_write_json,
    artifact_index_default,
    ensure_v4_preflight_scaffold,
    ensure_v4_scaffold,
)


def main() -> None:
    scaffold = ensure_v4_scaffold(REPO_ROOT)
    preflight_scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    artifact_index_path = REPO_ROOT / 'experiments' / 'results' / 'v4' / 'v4_artifact_index.json'
    if not artifact_index_path.exists():
        atomic_write_json(artifact_index_path, artifact_index_default(REPO_ROOT))

    print('v4 scaffold ready:')
    for key, path in scaffold.items():
        print(f'  {key}: {path}')
    for key, path in preflight_scaffold.items():
        print(f'  {key}: {path}')
    print(f'  artifact_index: {artifact_index_path}')


if __name__ == '__main__':
    main()
