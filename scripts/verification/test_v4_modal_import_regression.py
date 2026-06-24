"""
Flat-staged Modal import regression checks for detached v4 diagnostics.

Usage:
    python scripts/verification/test_v4_modal_import_regression.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_CLASS = type(Path())

SCRIPT_PATHS = (
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_step1_sensitivity.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_subject_aggregation.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_svc_probability_diagnostics.py',
    REPO_ROOT / 'scripts' / 'verification' / 'profile_v4_grouped_selector.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_step6_views.py',
)


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to import {script_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _root_src_permission_denied_patch():
    original_is_dir = PATH_CLASS.is_dir

    def patched_is_dir(self):  # type: ignore[no-untyped-def]
        if self == Path('/root/src'):
            raise PermissionError('[Errno 13] Permission denied: \'/root/src\'')
        return original_is_dir(self)

    with mock.patch.object(PATH_CLASS, 'is_dir', patched_is_dir):
        yield


def _import_local_checkout(script_path: Path) -> None:
    module = _load_module(script_path, f'local_import_{script_path.stem}')
    assert Path(module.REPO_ROOT) == REPO_ROOT
    assert Path(module.SRC_ROOT) == REPO_ROOT / 'src'

    with _root_src_permission_denied_patch():
        assert module._is_dir_without_raising(Path('/root/src')) is False
        repo_root, src_root = module._infer_repo_and_src_roots()
        assert Path(repo_root) == REPO_ROOT
        assert Path(src_root) == REPO_ROOT / 'src'


def _import_from_flat_stage(script_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='v4_modal_stage_') as tmpdir:
        staged_root = Path(tmpdir)
        staged_src = staged_root / 'src'
        staged_script = staged_root / script_path.name
        staged_src.symlink_to(REPO_ROOT / 'src', target_is_directory=True)
        shutil.copy2(script_path, staged_script)
        old_sys_path = list(sys.path)
        try:
            module = _load_module(staged_script, f'flat_stage_import_{script_path.stem}')
            assert Path(module.REPO_ROOT) == staged_root
            assert Path(module.SRC_ROOT) == staged_src
        finally:
            sys.path[:] = old_sys_path


def main() -> None:
    for script_path in SCRIPT_PATHS:
        _import_local_checkout(script_path)
        print(f'Local checkout import OK: {script_path.name}')
        _import_from_flat_stage(script_path)
        print(f'Flat-staged import OK: {script_path.name}')


if __name__ == '__main__':
    main()
