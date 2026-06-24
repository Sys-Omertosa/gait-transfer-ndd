"""
Cheap Step 3 payload-digest round-trip contract check.

Usage:
    python scripts/verification/test_v4_step3_digest_roundtrip.py
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
    validate_payload_digest,
    validate_step3_results_payload,
    write_payload_json,
)
from v4_provenance import atomic_write_json  # type: ignore


def _context() -> dict[str, str]:
    return {
        'protocol_manifest_sha256': 'protocol-sha',
        'preprocessing_manifest_sha256': 'preproc-sha',
        'feature_matrix_sha256': 'feature-sha',
        'partition_sha256': 'partition-sha',
        'downstream_manifest_sha256': 'downstream-sha',
        'downstream_execution_id': 'v4d-test-digest',
    }


def _step3_payload() -> dict[str, object]:
    return {
        'protocol_manifest_hash': 'protocol-sha',
        'preprocessing_manifest_hash': 'preproc-sha',
        'feature_matrix_hash': 'feature-sha',
        'partition_hash': 'partition-sha',
        'downstream_execution_manifest_hash': 'downstream-sha',
        'downstream_execution_id': 'v4d-test-digest',
        'output_namespace': 'results_v4_smoke',
        'pd_to_hd': {'rf': {'f1_macro': 0.5}},
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='v4-step3-digest-') as tmpdir:
        root = Path(tmpdir)
        remote_like_path = root / 'cross_condition_results_v4.json'
        local_like_path = root / 'cross_condition_results_v4_local.json'

        materialized = write_payload_json(remote_like_path, _step3_payload())
        validate_payload_digest(materialized)

        loaded_remote = json.loads(remote_like_path.read_text())
        validate_payload_digest(loaded_remote)
        assert loaded_remote == materialized

        atomic_write_json(local_like_path, materialized)
        loaded_local = json.loads(local_like_path.read_text())
        validate_payload_digest(loaded_local)
        assert loaded_local == materialized

        validate_step3_results_payload(
            payload=loaded_remote,
            output_namespace='results_v4_smoke',
            expected_directions=(('pd', 'hd'),),
            context=_context(),
        )

    print(json.dumps({
        'status': 'pass',
        'checks': {
            'remote_materialized_matches_returned_payload': True,
            'local_persisted_matches_returned_payload': True,
            'step3_digest_validation_passes': True,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
