"""
Cheap validation for Step 4 and Step 5 canonical downstream result envelopes.

Usage:
    python scripts/verification/test_v4_downstream_final_envelopes.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from scripts.training import run_noise_robustness_modal as step5  # type: ignore
from scripts.training import run_shap_modal as step4  # type: ignore
from v4_downstream import build_downstream_final_payload, write_payload_json  # type: ignore


def _context() -> dict[str, str]:
    return {
        'protocol_manifest_sha256': 'protocol-sha',
        'preprocessing_manifest_sha256': 'preproc-sha',
        'feature_matrix_sha256': 'feature-sha',
        'partition_sha256': 'partition-sha',
        'downstream_manifest_sha256': 'downstream-sha',
        'downstream_execution_id': 'v4d-envelope-test',
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='v4-final-envelopes-') as tmpdir:
        root = Path(tmpdir)

        step4_data = {
            'pd_to_hd': {'rf': {'delta_j': [0.1]}},
            'pd_to_als': {'rf': {'delta_j': [0.2]}},
            'hd_to_pd': {'rf': {'delta_j': [0.3]}},
            'hd_to_als': {'rf': {'delta_j': [0.4]}},
            'als_to_pd': {'rf': {'delta_j': [0.5]}},
            'als_to_hd': {'rf': {'delta_j': [0.6]}},
        }
        step4_payload = build_downstream_final_payload(
            schema_version=step4.FINAL_SCHEMA_VERSION,
            context=_context(),
            data=step4_data,
            extra_fields={'step': 'step4'},
        )
        step4_path = root / 'shap_results_v4.json'
        write_payload_json(step4_path, step4_payload)
        validated_step4 = step4._validate_final_step4_payload(  # noqa: SLF001
            payload=json.loads(step4_path.read_text()),
            context=_context(),
        )
        assert set(validated_step4) == set(step4_data)

        step5_payload = build_downstream_final_payload(
            schema_version=step5.FINAL_SCHEMA_VERSIONS['noise'],
            context=_context(),
            data={
                'within': {'pd': {'rf': {'0.0': [0.9]}}},
                'cross': {'pd_to_hd': {'source_relative': {'rf': {'0.0': [0.8]}}}},
            },
            extra_fields={'step': 'step5'},
        )
        step5_path = root / 'noise_robustness_v4.json'
        write_payload_json(step5_path, step5_payload)
        validated_step5 = step5._validate_final_output_payload(  # noqa: SLF001
            payload=json.loads(step5_path.read_text()),
            context=_context(),
            key='noise',
        )
        assert set(validated_step5) == {'within', 'cross'}

    print(json.dumps({
        'status': 'pass',
        'checks': {
            'step4_final_envelope_validation': True,
            'step5_final_envelope_validation': True,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
