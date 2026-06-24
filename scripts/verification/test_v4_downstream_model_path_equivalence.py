"""
Cheap local check for logical Step 2 model inventory equivalence across local and Modal roots.

Usage:
    python scripts/verification/test_v4_downstream_model_path_equivalence.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v4_downstream import (  # type: ignore
    CLF_ORDER,
    CONDITIONS,
    normalize_step2_model_hashes,
    normalize_step2_model_logical_path,
    validate_step2_full_source_models,
)
from v4_provenance import sha256_file  # type: ignore


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_file(path)


def _payloads_for_root(models_dir: Path, *, style: str) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for condition in CONDITIONS:
        classifiers: dict[str, dict] = {}
        for clf_name in CLF_ORDER:
            relpath = f'{condition}_{clf_name}.joblib'
            model_path = models_dir / relpath
            model_sha = _write_bytes(model_path, f'{condition}:{clf_name}'.encode())
            if style == 'relative':
                stored_path = relpath
            elif style == 'local':
                stored_path = str(model_path)
            elif style == 'remote':
                stored_path = f'/results/models_v4/{relpath}'
            else:
                raise ValueError(style)
            classifiers[clf_name] = {
                'full_source_model_path': stored_path,
                'full_source_model_sha256': model_sha,
            }
        payloads[condition] = {'classifiers': classifiers}
    return payloads


def _logical_inventory(inventory: dict[str, dict]) -> dict[str, tuple[str, str]]:
    return {
        logical_key: (entry['logical_path'], entry['sha256'])
        for logical_key, entry in inventory.items()
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='v4-model-paths-') as tmpdir:
        root = Path(tmpdir)
        local_models_dir = root / 'local_root' / 'experiments' / 'models' / 'v4'
        remote_models_dir = root / 'remote_root' / 'models_v4'

        local_inventory = validate_step2_full_source_models(
            payloads=_payloads_for_root(local_models_dir, style='local'),
            models_dir=local_models_dir,
        )
        remote_inventory = validate_step2_full_source_models(
            payloads=_payloads_for_root(remote_models_dir, style='remote'),
            models_dir=remote_models_dir,
        )
        relative_inventory = validate_step2_full_source_models(
            payloads=_payloads_for_root(local_models_dir, style='relative'),
            models_dir=local_models_dir,
        )

        assert _logical_inventory(local_inventory) == _logical_inventory(remote_inventory)
        assert _logical_inventory(local_inventory) == _logical_inventory(relative_inventory)

        normalized = normalize_step2_model_hashes({
            'pd:rf': {
                'logical_path': 'pd_rf.joblib',
                'sha256': local_inventory['pd:rf']['sha256'],
            },
            'pd:knn': {
                'path': f'/results/models_v4/pd_knn.joblib',
                'sha256': local_inventory['pd:knn']['sha256'],
            },
            'pd:svm': {
                'path': str(local_models_dir / 'pd_svm.joblib'),
                'sha256': local_inventory['pd:svm']['sha256'],
            },
        })
        assert normalized['pd:rf']['logical_path'] == 'pd_rf.joblib'
        assert normalized['pd:knn']['logical_path'] == 'pd_knn.joblib'
        assert normalized['pd:svm']['logical_path'] == 'pd_svm.joblib'

        try:
            normalize_step2_model_logical_path('../escape.joblib')
        except ValueError:
            pass
        else:
            raise AssertionError('Expected path-escape rejection for relative .. path.')

    print(json.dumps({
        'status': 'pass',
        'checks': {
            'local_remote_logical_inventory_equivalence': True,
            'relative_local_remote_path_normalization': True,
            'path_escape_rejection': True,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
