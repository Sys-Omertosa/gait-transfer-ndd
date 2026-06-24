"""
Prepare, but do not launch, the isolated v4 smoke-test namespace contract.

Usage:
    python scripts/verification/test_v4_modal_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    payload = {
        'processed_namespace': 'processed_v4_smoke',
        'results_namespace': 'results_v4_smoke',
        'models_namespace': 'models_v4_smoke',
        'shap_namespace': 'shap_v4_smoke',
        'local_root': str(REPO_ROOT),
        'notes': [
            'Prepare only; do not launch Modal automatically.',
            'Use the smoke namespace to validate path isolation and artifact assembly.',
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
