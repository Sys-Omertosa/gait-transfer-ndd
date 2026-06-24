"""
Static Modal path validation for v4 runners.

Usage:
    python scripts/verification/test_v4_modal_paths.py
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from scripts.verification.test_v4_subject_aggregation import _volume_listdir_or_empty  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS = [
    REPO_ROOT / 'scripts' / 'training' / 'run_cross_condition_modal.py',
    REPO_ROOT / 'scripts' / 'training' / 'run_shap_modal.py',
    REPO_ROOT / 'scripts' / 'training' / 'run_noise_robustness_modal.py',
    REPO_ROOT / 'scripts' / 'training' / 'run_control_split_sensitivity_modal.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_step1_sensitivity.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_subject_aggregation.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_svc_probability_diagnostics.py',
    REPO_ROOT / 'scripts' / 'verification' / 'profile_v4_grouped_selector.py',
    REPO_ROOT / 'scripts' / 'verification' / 'test_v4_step6_views.py',
]


class _FakeModalNotFoundError(Exception):
    pass


class _FakeVolume:
    def __init__(self, response: object) -> None:
        self.response = response

    def listdir(self, prefix: str, recursive: bool = False) -> object:
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _assert_missing_prefix_returns_empty() -> None:
    returned = _volume_listdir_or_empty(
        _FakeVolume(_FakeModalNotFoundError('No such file or directory')),
        'results_v4_preflight/subject_aggregation_fragments',
        missing_prefix_errors=(FileNotFoundError, _FakeModalNotFoundError),
    )
    if returned != []:
        raise AssertionError(f'Expected [] for missing prefix, got {returned!r}')


def _assert_unrelated_exception_propagates() -> None:
    try:
        _volume_listdir_or_empty(
            _FakeVolume(RuntimeError('boom')),
            'results_v4_preflight/subject_aggregation_fragments',
            missing_prefix_errors=(FileNotFoundError, _FakeModalNotFoundError),
        )
    except RuntimeError:
        return
    raise AssertionError('Expected unrelated RuntimeError to propagate.')


def _assert_valid_existing_entries_returned() -> None:
    entries = [
        SimpleNamespace(path='results_v4_preflight/subject_aggregation_fragments/pd_rf/a.json'),
        SimpleNamespace(path='results_v4_preflight/subject_aggregation_fragments/pd_rf/b.json'),
    ]
    returned = _volume_listdir_or_empty(
        _FakeVolume(entries),
        'results_v4_preflight/subject_aggregation_fragments',
        missing_prefix_errors=(FileNotFoundError, _FakeModalNotFoundError),
    )
    if returned != entries:
        raise AssertionError('Expected existing entries to be returned unchanged.')


def main() -> None:
    violations: list[str] = []
    pattern = re.compile(r"batch\.put_file\([^)]*,\s*['\"](/results/[^'\"]+)['\"]", re.MULTILINE)
    forbidden_outputs = (
        '/results/processed_v4/',
        '/results/results_v4/',
        '/results/models_v4/',
        '/results/shap_v4/',
    )
    for runner in RUNNERS:
        text = runner.read_text()
        for match in pattern.finditer(text):
            violations.append(f'{runner}: {match.group(1)}')
        if 'scripts/verification/' in str(runner):
            for forbidden in forbidden_outputs:
                if forbidden in text:
                    violations.append(f'{runner}: forbidden canonical output path {forbidden}')

    if violations:
        raise SystemExit(
            'Found Modal path contract violations:\n'
            + '\n'.join(violations)
        )

    _assert_missing_prefix_returns_empty()
    _assert_unrelated_exception_propagates()
    _assert_valid_existing_entries_returned()

    print('v4 Modal path validation passed.')


if __name__ == '__main__':
    main()
