"""
Static contract checks for the v4 hardening pass.

Usage:
    python scripts/verification/test_v4_static_contracts.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import preprocessing  # type: ignore
import robustness  # type: ignore
import train  # type: ignore
from v4_provenance import code_hash, ensure_v4_preflight_scaffold, ensure_v4_scaffold


def main() -> None:
    scaffold = ensure_v4_scaffold(REPO_ROOT)
    preflight_scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)

    run_cross_sig = inspect.signature(train.run_cross_condition)
    fit_replay_sig = inspect.signature(robustness.fit_within_condition_folds)

    assert run_cross_sig.parameters['allow_refit'].default is False
    assert fit_replay_sig.parameters['allow_approximate_refit'].default is False

    bounds = preprocessing._scaled_mad_bounds(  # noqa: SLF001 - intentional contract check
        values=[1.0, 2.0, 3.0, 4.0],
        robust_mad_multiplier=3.0,
    )
    assert bounds is not None

    rel_hash = code_hash(
        [Path('src') / 'v4_provenance.py'],
        base_dir=REPO_ROOT,
    )
    abs_hash = code_hash(
        [REPO_ROOT / 'src' / 'v4_provenance.py'],
        base_dir=REPO_ROOT,
    )
    assert rel_hash == abs_hash
    try:
        code_hash([Path('/tmp') / 'outside_repo.txt'], base_dir=REPO_ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError('code_hash() should reject files outside repo_root.')

    shap_runner = (REPO_ROOT / 'scripts' / 'training' / 'run_shap_modal.py').read_text()
    assert "batch.put_file(" in shap_runner
    assert "'results_v4/shap_results_v4.json'" in shap_runner
    assert 'v4_protocol_manifest.json' in shap_runner
    assert 'preprocessing_manifest_v4.json' in shap_runner
    assert 'volume.commit()' in shap_runner

    sensitivity_runner = (
        REPO_ROOT / 'scripts' / 'training' / 'run_control_split_sensitivity_modal.py'
    ).read_text()
    assert 'models_dir=models_dir' in sensitivity_runner
    assert 'allow_refit=False' in sensitivity_runner

    freeze_script = (
        REPO_ROOT / 'scripts' / 'setup' / 'freeze_v4_protocol_manifest.py'
    ).read_text()
    for required_flag in (
        '--aggregation-rule',
        '--tie-break-rule',
        '--robust-mad-multiplier',
        '--dfa-policy',
        '--approved',
        '--allow-dirty-tree',
    ):
        assert required_flag in freeze_script

    print('v4 static contracts passed.')
    for key, value in scaffold.items():
        print(f'  scaffold[{key}] = {value}')
    for key, value in preflight_scaffold.items():
        print(f'  preflight_scaffold[{key}] = {value}')


if __name__ == '__main__':
    main()
