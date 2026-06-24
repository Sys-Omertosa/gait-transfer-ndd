"""
Shared v4 provenance, scaffold, and atomic-write helpers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RUNTIME_PACKAGES: tuple[str, ...] = (
    'numpy',
    'polars',
    'scikit-learn',
    'imbalanced-learn',
    'xgboost',
    'lightgbm',
    'shap',
    'joblib',
    'modal',
)


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest for an in-memory payload."""
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    """Return the SHA-256 hex digest for a UTF-8 string."""
    return sha256_bytes(payload.encode())


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file on disk."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Stable SHA-256 digest for a NumPy array's contiguous raw bytes."""
    contiguous = np.ascontiguousarray(arr)
    return sha256_bytes(contiguous.tobytes())


def canonical_json_dumps(payload: Any, *, indent: int = 2) -> str:
    """Return deterministic JSON text suitable for hashing or persistence."""
    return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=True)


def _drop_keys_recursive(payload: Any, excluded_keys: set[str]) -> Any:
    """Return a deep copy of payload with excluded dict keys removed recursively."""
    if isinstance(payload, dict):
        return {
            key: _drop_keys_recursive(value, excluded_keys)
            for key, value in payload.items()
            if key not in excluded_keys
        }
    if isinstance(payload, list):
        return [_drop_keys_recursive(value, excluded_keys) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_drop_keys_recursive(value, excluded_keys) for value in payload)
    return payload


def canonical_payload_sha256(
    payload: Any,
    *,
    exclude_keys: tuple[str, ...] = ('payload_sha256',),
) -> str:
    """
    Compute a deterministic SHA-256 digest for JSON-serializable payload content.

    Excluded keys are removed recursively before canonical serialization so the
    digest can be persisted inside the payload itself.
    """
    normalized = _drop_keys_recursive(payload, set(exclude_keys))
    return sha256_text(canonical_json_dumps(normalized, indent=2))


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write bytes atomically via a temp file and os.replace()."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f'.{target.name}.',
        suffix='.tmp',
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)


def atomic_write_text(path: str | Path, payload: str) -> None:
    """Write UTF-8 text atomically."""
    atomic_write_bytes(path, payload.encode())


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON atomically with deterministic key ordering."""
    atomic_write_text(path, canonical_json_dumps(payload, indent=indent))


def collect_package_versions(
    package_names: tuple[str, ...] = DEFAULT_RUNTIME_PACKAGES,
) -> dict[str, str]:
    """Collect installed package versions without failing on missing optional packages."""
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = 'missing'
    return versions


def _normalized_repo_relative_path(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[Path, str]:
    """
    Resolve a path under repo_root and return its normalized repo-relative POSIX path.

    Machine-specific absolute checkout prefixes are never included in the logical
    hash path. Paths outside repo_root are rejected explicitly.
    """
    root = Path(repo_root).resolve()
    candidate = Path(path)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        logical_path = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f'code_hash() only accepts files inside repo_root={root}; got {resolved}'
        ) from exc
    return resolved, logical_path


def code_hash(paths: list[str | Path], *, base_dir: str | Path | None = None) -> str:
    """Hash repository-relative paths plus file contents."""
    root = (
        Path(base_dir).resolve()
        if base_dir is not None else
        Path(__file__).resolve().parents[1]
    )
    digest = hashlib.sha256()
    normalized_paths: list[tuple[Path, str]] = []
    for raw_path in paths:
        resolved, logical_path = _normalized_repo_relative_path(
            raw_path,
            repo_root=root,
        )
        normalized_paths.append((resolved, logical_path))
    for resolved, logical_path in sorted(normalized_paths, key=lambda item: item[1]):
        digest.update(logical_path.encode())
        with open(resolved, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_v4_scaffold(repo_root: str | Path) -> dict[str, str]:
    """Create the canonical local v4 directory layout if it does not already exist."""
    root = Path(repo_root)
    scaffold = {
        'processed': root / 'data' / 'processed' / 'v4',
        'results': root / 'experiments' / 'results' / 'v4',
        'models': root / 'experiments' / 'models' / 'v4',
        'shap': root / 'experiments' / 'shap' / 'v4',
        'figures': root / 'report' / 'figures' / 'v4',
        'tables': root / 'report' / 'tables' / 'v4',
    }
    for path in scaffold.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(path) for key, path in scaffold.items()}


def ensure_v4_preflight_scaffold(repo_root: str | Path) -> dict[str, str]:
    """Create the lightweight local preflight directory layout used before freeze."""
    root = Path(repo_root)
    scaffold = {
        'processed_preflight': root / 'data' / 'processed' / 'v4_preflight',
        'results_preflight': root / 'experiments' / 'results' / 'v4_preflight',
        'models_preflight': root / 'experiments' / 'models' / 'v4_preflight',
        'shap_preflight': root / 'experiments' / 'shap' / 'v4_preflight',
    }
    for path in scaffold.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(path) for key, path in scaffold.items()}


def artifact_index_default(repo_root: str | Path) -> dict[str, Any]:
    """Return the empty canonical artifact-index structure."""
    return {
        'schema_version': 'v4-artifact-index',
        'updated_at_utc': utc_now_iso(),
        'repo_root': str(Path(repo_root).resolve()),
        'artifacts': [],
    }


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    """Load JSON from disk when present, otherwise return None."""
    json_path = Path(path)
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def upsert_artifact_index_entry(
    *,
    index_path: str | Path,
    entry: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Insert or replace an artifact-index entry by logical artifact key."""
    index = load_json_if_exists(index_path)
    if index is None:
        index = artifact_index_default(repo_root)

    logical_key = entry['logical_key']
    updated_entries = [
        existing
        for existing in index.get('artifacts', [])
        if existing.get('logical_key') != logical_key
    ]
    updated_entries.append(entry)
    index['artifacts'] = sorted(updated_entries, key=lambda item: item['logical_key'])
    index['updated_at_utc'] = utc_now_iso()
    atomic_write_json(index_path, index)
    return index
