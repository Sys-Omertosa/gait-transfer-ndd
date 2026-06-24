"""
Exact Step 5 replay contract smoke check.

Usage:
    python scripts/verification/test_v4_step5_replay.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if __name__ == '__main__':
    runpy.run_path(
        str(REPO_ROOT / 'scripts' / 'verification' / 'test_v4_artifact_identity.py'),
        run_name='__main__',
    )
