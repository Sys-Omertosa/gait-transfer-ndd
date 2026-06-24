"""
Cheap static contract check for fragmented subject-aggregation recovery workers.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = REPO_ROOT / 'scripts' / 'verification' / 'test_v4_subject_aggregation.py'

EXPECTED = {
    'run_rf_outer_candidate_fragment_remote': {
        'cpu': 1,
        'memory': 2048,
        'max_containers': 24,
        'nonpreemptible': False,
    },
    'run_rf_outer_result_fragment_remote': {
        'cpu': 2,
        'memory': 4096,
        'max_containers': 8,
        'nonpreemptible': False,
    },
    'run_rf_full_source_candidate_fragment_remote': {
        'cpu': 1,
        'memory': 2048,
        'max_containers': 24,
        'nonpreemptible': False,
    },
    'run_svm_inner_fragment_remote': {
        'cpu': 1,
        'memory': 3072,
        'max_containers': 16,
        'nonpreemptible': False,
    },
    'run_svm_candidate_summary_fragment_remote': {
        'cpu': 1,
        'memory': 3072,
        'max_containers': 8,
        'nonpreemptible': False,
    },
    'run_svm_outer_result_fragment_remote': {
        'cpu': 2,
        'memory': 4096,
        'max_containers': 8,
        'nonpreemptible': False,
    },
    'run_full_source_selection_fragment_remote': {
        'cpu': 2,
        'memory': 4096,
        'max_containers': 4,
        'nonpreemptible': False,
    },
    'assemble_fragmented_full_shard_remote': {
        'cpu': 2,
        'memory': 4096,
        'max_containers': 4,
        'nonpreemptible': False,
    },
}

LITERAL_KEYS = {
    'cpu',
    'memory',
    'max_containers',
    'timeout',
    'retries',
    'nonpreemptible',
}


def _function_kwargs(tree: ast.AST) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in EXPECTED:
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute) or func.attr != 'function':
                continue
            kwargs: dict[str, object] = {}
            for keyword in decorator.keywords:
                if keyword.arg is None:
                    continue
                if keyword.arg not in LITERAL_KEYS:
                    continue
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            found[node.name] = kwargs
            break
    return found


def main() -> None:
    tree = ast.parse(TARGET_FILE.read_text())
    found = _function_kwargs(tree)
    if set(found) != set(EXPECTED):
        missing = sorted(set(EXPECTED) - set(found))
        extra = sorted(set(found) - set(EXPECTED))
        raise AssertionError(
            f'Fragmented worker contract mismatch. missing={missing} extra={extra}'
        )
    for fn_name, expected in EXPECTED.items():
        kwargs = found[fn_name]
        for key, expected_value in expected.items():
            actual = kwargs.get(key, False if key == 'nonpreemptible' else None)
            if actual != expected_value:
                raise AssertionError(
                    f'{fn_name} expected {key}={expected_value!r}, got {actual!r}'
                )
    print(json.dumps({
        'status': 'pass',
        'validated_workers': sorted(EXPECTED),
    }, indent=2))


if __name__ == '__main__':
    main()
