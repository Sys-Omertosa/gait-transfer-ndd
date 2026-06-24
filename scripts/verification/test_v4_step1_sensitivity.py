"""
Real-data Step 1 MAD and segmented-DFA preflight diagnostics for v4.

Usage:
    python scripts/verification/test_v4_step1_sensitivity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - optional local import
    modal = None

def _is_dir_without_raising(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _infer_repo_and_src_roots() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()

    for candidate in (script_path.parent, *script_path.parents):
        src_root = candidate / 'src'
        if _is_dir_without_raising(src_root):
            return candidate, src_root

    modal_root = Path('/root')
    modal_src = modal_root / 'src'
    if _is_dir_without_raising(modal_src):
        return modal_root, modal_src

    raise RuntimeError(
        'Unable to locate src directory from '
        f'script_path={script_path}'
    )


REPO_ROOT, SRC_ROOT = _infer_repo_and_src_roots()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import robustness as rb  # type: ignore
from features import (  # type: ignore
    DEFAULT_DATA_DIR,
    _dfa_alpha_from_stride_sequence,
    build_feature_matrix,
    build_per_stride_only_matrix,
    build_subject_level_matrix,
    get_feature_cols,
)
from preprocessing import load_raw_data  # type: ignore
from v4_provenance import atomic_write_json, ensure_v4_preflight_scaffold  # type: ignore

CONDITIONS = ('pd', 'hd', 'als')
MAD_MULTIPLIERS = (2.5, 3.0, 3.5, 4.0)
DFA_MIN_SEGMENT_LENGTH = 100
REMOTE_RAW_DATA_DIR = '/results/raw/gait-in-neurodegenerative-disease-database-1.0.0'
REMOTE_PROCESSED_DIR = '/results/processed_v4_preflight'
REMOTE_RESULTS_DIR = '/results/results_v4_preflight'
REMOTE_V3_RESULTS_DIR = '/results/results_v3'
REMOTE_REPORT_NAME = 'step1_preflight_report.json'
REMOTE_REPORT_GET_PATH = f'results_v4_preflight/{REMOTE_REPORT_NAME}'


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xr = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    yr = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(xr, yr)[0, 1])


def _json_load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _mad_tag(multiplier: float) -> str:
    return str(multiplier).replace('.', '_')


def _reference_paths(
    *,
    processed_dir: Path,
    results_dir: Path,
) -> dict[str, Path]:
    return {
        'processed_dir': processed_dir,
        'results_dir': results_dir,
        'reference_features': processed_dir / 'gait_features_v4_reference.csv',
        'reference_partition': processed_dir / 'control_partition_v4_reference.json',
        'reference_manifest': processed_dir / 'preprocessing_manifest_v4_reference.json',
        'per_stride_only': processed_dir / 'gait_features_v4_per_stride_only_reference.csv',
        'subject_level': processed_dir / 'gait_features_v4_subject_level_reference.csv',
        'report': results_dir / REMOTE_REPORT_NAME,
    }


def _build_or_load_reference_matrix(
    multiplier: float,
    *,
    processed_dir: Path,
    results_dir: Path,
    data_dir: Path,
    allow_build: bool,
) -> tuple[pl.DataFrame, dict[str, list[str]], dict[str, Any], dict[str, str]]:
    paths = _reference_paths(processed_dir=processed_dir, results_dir=results_dir)
    processed_dir = paths['processed_dir']
    tag = _mad_tag(multiplier)
    if math.isclose(multiplier, 3.0):
        features_path = paths['reference_features']
        partition_path = paths['reference_partition']
        manifest_path = paths['reference_manifest']
        output_filename = features_path.name
        partition_filename = partition_path.name
        manifest_filename = manifest_path.name
    else:
        features_path = processed_dir / f'gait_features_v4_mad_{tag}.csv'
        partition_path = processed_dir / f'control_partition_v4_mad_{tag}.json'
        manifest_path = processed_dir / f'preprocessing_manifest_v4_mad_{tag}.json'
        output_filename = features_path.name
        partition_filename = partition_path.name
        manifest_filename = manifest_path.name

    if features_path.exists() and partition_path.exists() and manifest_path.exists():
        return (
            pl.read_csv(features_path),
            _json_load(partition_path),
            _json_load(manifest_path),
            {
                'features_path': str(features_path),
                'partition_path': str(partition_path),
                'manifest_path': str(manifest_path),
            },
        )

    if not allow_build:
        raise FileNotFoundError(
            'Missing preflight Step 1 reference artifacts. '
            'Run the detached Step 1 preflight diagnostic first.'
        )

    feature_df, partition = build_feature_matrix(
        data_dir=data_dir,
        processed_dir=processed_dir,
        output_filename=output_filename,
        feature_cols=get_feature_cols('v4'),
        feature_set_version='v4',
        filter_strategy='v3',
        control_partition_version='v4',
        partition_output_filename=partition_filename,
        manifest_filename=manifest_filename,
        robust_mad_multiplier=multiplier,
    )
    manifest = _json_load(manifest_path)
    return (
        feature_df,
        partition,
        manifest,
        {
            'features_path': str(features_path),
            'partition_path': str(partition_path),
            'manifest_path': str(manifest_path),
        },
    )


def _materialize_reference_companions(
    feature_df: pl.DataFrame,
    *,
    processed_dir: Path,
    results_dir: Path,
) -> dict[str, str]:
    paths = _reference_paths(processed_dir=processed_dir, results_dir=results_dir)
    per_stride_only = build_per_stride_only_matrix(feature_df)
    subject_level = build_subject_level_matrix(feature_df)
    per_stride_only.write_csv(paths['per_stride_only'])
    subject_level.write_csv(paths['subject_level'])
    return {
        'per_stride_only_path': str(paths['per_stride_only']),
        'subject_level_path': str(paths['subject_level']),
        'per_stride_only_shape': [int(per_stride_only.height), int(per_stride_only.width)],
        'subject_level_shape': [int(subject_level.height), int(subject_level.width)],
    }


def _segments(subject_df: pl.DataFrame) -> list[np.ndarray]:
    raw_idx = subject_df['raw_stride_index'].to_numpy()
    elapsed = subject_df['elapsed_s'].to_numpy()
    left_stride = subject_df['left_stride_s'].to_numpy()
    segments: list[list[float]] = []
    current: list[float] = []
    prev_idx = None
    prev_elapsed = None
    for idx, elapsed_s, stride in zip(raw_idx, elapsed, left_stride, strict=True):
        is_contiguous = (
            prev_idx is None
            or (int(idx) == int(prev_idx) + 1 and float(elapsed_s) > float(prev_elapsed))
        )
        if not is_contiguous and current:
            segments.append(current)
            current = []
        current.append(float(stride))
        prev_idx = idx
        prev_elapsed = elapsed_s
    if current:
        segments.append(current)
    return [np.asarray(segment, dtype=np.float64) for segment in segments]


def _compute_dfa_policy_values(feature_df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    per_subject: dict[str, dict[str, Any]] = {}
    for subject_df in feature_df.partition_by('subject_id', maintain_order=True):
        subject_id = str(subject_df['subject_id'][0])
        condition = str(subject_df['condition'][0])
        concat_alpha = float(subject_df['dfa_alpha_stride'][0])
        segments = _segments(subject_df)
        eligible_segments = [segment for segment in segments if len(segment) >= DFA_MIN_SEGMENT_LENGTH]
        longest_segment = max(segments, key=len)
        longest_alpha = None
        if len(longest_segment) >= DFA_MIN_SEGMENT_LENGTH:
            longest_alpha = float(_dfa_alpha_from_stride_sequence(longest_segment)[0])
        segmented_alpha = None
        if eligible_segments:
            alphas = np.asarray(
                [_dfa_alpha_from_stride_sequence(segment)[0] for segment in eligible_segments],
                dtype=float,
            )
            weights = np.asarray([len(segment) for segment in eligible_segments], dtype=float)
            segmented_alpha = float(np.average(alphas, weights=weights))
        per_subject[subject_id] = {
            'subject_id': subject_id,
            'condition': condition,
            'concat_alpha': round(concat_alpha, 6),
            'longest_alpha': None if longest_alpha is None else round(longest_alpha, 6),
            'segmented_alpha': None if segmented_alpha is None else round(segmented_alpha, 6),
            'n_segments': int(len(segments)),
            'segment_lengths': [int(len(segment)) for segment in segments],
            'eligible_segment_lengths': [int(len(segment)) for segment in eligible_segments],
            'eligible_longest': bool(longest_alpha is not None),
            'eligible_segmented': bool(segmented_alpha is not None),
        }
    return per_subject


def _policy_summary(per_subject: dict[str, dict[str, Any]], policy_key: str) -> dict[str, Any]:
    if policy_key == 'concat_alpha':
        eligible_subjects = list(per_subject)
        excluded_subjects: list[str] = []
    else:
        eligible_subjects = [
            subject_id
            for subject_id, payload in per_subject.items()
            if payload[policy_key] is not None
        ]
        excluded_subjects = [
            subject_id
            for subject_id, payload in per_subject.items()
            if payload[policy_key] is None
        ]
    by_condition: dict[str, int] = {}
    excluded_by_condition: dict[str, int] = {}
    for subject_id in eligible_subjects:
        condition = str(per_subject[subject_id]['condition'])
        by_condition[condition] = by_condition.get(condition, 0) + 1
    for subject_id in excluded_subjects:
        condition = str(per_subject[subject_id]['condition'])
        excluded_by_condition[condition] = excluded_by_condition.get(condition, 0) + 1
    return {
        'eligible_subject_ids': eligible_subjects,
        'excluded_subject_ids': excluded_subjects,
        'eligible_subject_count': int(len(eligible_subjects)),
        'excluded_subject_count': int(len(excluded_subjects)),
        'eligible_by_condition': by_condition,
        'excluded_by_condition': excluded_by_condition,
    }


def _apply_dfa_policy(
    feature_df: pl.DataFrame,
    per_subject: dict[str, dict[str, Any]],
    policy: str,
) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    if policy == 'concatenated':
        return feature_df, _policy_summary(per_subject, 'concat_alpha')

    policy_key = 'longest_alpha' if policy == 'longest_contiguous_segment' else 'segmented_alpha'
    summary = _policy_summary(per_subject, policy_key)
    if not summary['eligible_subject_ids']:
        return None, summary

    replacement_rows = [
        {
            'subject_id': subject_id,
            'dfa_alpha_stride': float(per_subject[subject_id][policy_key]),
        }
        for subject_id in summary['eligible_subject_ids']
    ]
    replacement_df = pl.DataFrame(replacement_rows)
    kept = (
        feature_df
        .filter(pl.col('subject_id').is_in(summary['eligible_subject_ids']))
        .drop('dfa_alpha_stride')
        .join(replacement_df, on='subject_id', how='left')
    )
    ordered = kept.select(feature_df.columns)
    return ordered, summary


def _load_v3_within_results(
    results_v3_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    base = results_v3_dir or (REPO_ROOT / 'experiments' / 'results' / 'v3')
    missing = [
        condition for condition in CONDITIONS
        if not (base / f'{condition}_results_v3.json').exists()
    ]
    if missing:
        if str(base).startswith('/results/'):
            raise FileNotFoundError(
                'Missing v3 within-condition JSONs on the Modal volume at '
                f'{base}. Upload them first with:\n'
                '  modal volume put gait-results experiments/results/v3/pd_results_v3.json results_v3/pd_results_v3.json\n'
                '  modal volume put gait-results experiments/results/v3/hd_results_v3.json results_v3/hd_results_v3.json\n'
                '  modal volume put gait-results experiments/results/v3/als_results_v3.json results_v3/als_results_v3.json'
            )
        raise FileNotFoundError(
            f'Missing local v3 within-condition JSONs under {base}: {missing}'
        )
    return {
        condition: _json_load(base / f'{condition}_results_v3.json')
        for condition in CONDITIONS
    }


def _evaluate_source_orderings(
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
    *,
    results_v3_dir: Path | None = None,
) -> dict[str, Any]:
    v3_within = _load_v3_within_results(results_v3_dir)
    control_a = partition['control_A']
    feature_cols = get_feature_cols('v4')
    results: dict[str, Any] = {}

    for condition in CONDITIONS:
        pool = feature_df.filter(
            (pl.col('condition') == condition) | pl.col('subject_id').is_in(control_a)
        )
        if pool.n_unique('subject_id') < 3 or len(np.unique(pool['label'].to_numpy())) < 2:
            results[condition] = {'status': 'not_evaluable', 'reason': 'insufficient_subjects_or_classes'}
            continue
        X = pool.select(feature_cols).to_numpy().astype(np.float64)
        y = pool['label'].to_numpy().astype(int)
        groups = pool['subject_id'].to_numpy()
        feature_std = np.zeros(len(feature_cols), dtype=np.float64)

        ranked_v3 = sorted(
            v3_within[condition]['classifiers'].items(),
            key=lambda item: float(item[1]['f1_macro']),
            reverse=True,
        )
        evaluated_clfs = [clf_name for clf_name, _ in ranked_v3[:2]]
        scores: dict[str, float] = {}
        for clf_name in evaluated_clfs:
            clf_res = v3_within[condition]['classifiers'][clf_name]
            fitted_folds = rb.loso_fit_all_folds_fixed(
                X,
                y,
                groups,
                clf_name,
                clf_res['modal_params'],
                imbalance_strategy=clf_res['selected_imbalance_strategy'],
            )
            score = rb.loso_predict_from_fitted(
                fitted_folds,
                X,
                y,
                np.random.default_rng(42),
                0.0,
                feature_std,
            )
            scores[clf_name] = round(float(score), 6)
        ordering = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results[condition] = {
            'status': 'ok',
            'evaluated_classifiers': evaluated_clfs,
            'scores': scores,
            'ordering': [clf_name for clf_name, _ in ordering],
            'best_classifier': ordering[0][0],
        }
    return results


def _feature_shift_ranking(
    baseline_subject_df: pl.DataFrame,
    comparison_subject_df: pl.DataFrame,
) -> list[dict[str, Any]]:
    merged = baseline_subject_df.join(
        comparison_subject_df,
        on='subject_id',
        how='inner',
        suffix='_cmp',
    )
    shifts: list[dict[str, Any]] = []
    for feature_name in get_feature_cols('v4'):
        base = merged[feature_name].to_numpy().astype(np.float64)
        cmp = merged[f'{feature_name}_cmp'].to_numpy().astype(np.float64)
        baseline_sd = float(np.std(base)) or 1.0
        shift = abs(float(np.mean(cmp) - np.mean(base))) / baseline_sd
        shifts.append({
            'feature': feature_name,
            'standardized_mean_shift': round(shift, 6),
        })
    shifts.sort(key=lambda item: item['standardized_mean_shift'], reverse=True)
    return shifts


def build_step1_preflight_report(
    *,
    processed_dir: Path,
    results_dir: Path,
    data_dir: Path,
    allow_build: bool,
    results_v3_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    paths = _reference_paths(processed_dir=processed_dir, results_dir=results_dir)
    raw = load_raw_data(str(data_dir))
    raw_rows_by_subject = {
        str(row['subject_id']): int(row['len'])
        for row in raw.group_by('subject_id').len().to_dicts()
    }
    raw_rows_by_condition = {
        str(row['condition']): int(row['len'])
        for row in raw.group_by('condition').len().to_dicts()
    }

    mad_frames: dict[str, pl.DataFrame] = {}
    mad_partitions: dict[str, dict[str, list[str]]] = {}
    mad_manifests: dict[str, dict[str, Any]] = {}
    mad_paths: dict[str, dict[str, str]] = {}

    for multiplier in MAD_MULTIPLIERS:
        tag = _mad_tag(multiplier)
        feature_df, partition, manifest, output_paths = _build_or_load_reference_matrix(
            multiplier,
            processed_dir=processed_dir,
            results_dir=results_dir,
            data_dir=data_dir,
            allow_build=allow_build,
        )
        mad_frames[tag] = feature_df
        mad_partitions[tag] = partition
        mad_manifests[tag] = manifest
        mad_paths[tag] = output_paths

    baseline_df = mad_frames['3_0']
    baseline_partition = mad_partitions['3_0']
    baseline_subject_df = build_subject_level_matrix(baseline_df)
    companion_paths = _materialize_reference_companions(
        baseline_df,
        processed_dir=processed_dir,
        results_dir=results_dir,
    )

    mad_report: dict[str, Any] = {}
    baseline_orderings = _evaluate_source_orderings(
        baseline_df,
        baseline_partition,
        results_v3_dir=results_v3_dir,
    )
    for multiplier in MAD_MULTIPLIERS:
        tag = _mad_tag(multiplier)
        feature_df = mad_frames[tag]
        manifest = mad_manifests[tag]
        subject_df = build_subject_level_matrix(feature_df)
        filter_stats = manifest.get('filter_stats', {})
        per_subject_rows = {
            str(row['subject_id']): int(row['len'])
            for row in feature_df.group_by('subject_id').len().to_dicts()
        }
        removed_rows_by_subject = {
            subject_id: int(raw_rows_by_subject.get(subject_id, 0) - per_subject_rows.get(subject_id, 0))
            for subject_id in sorted(raw_rows_by_subject)
        }
        condition_rows = feature_df.group_by('condition').len().sort('condition').to_dicts()
        final_rows_by_condition = {
            str(row['condition']): int(row['len']) for row in condition_rows
        }
        removed_rows_by_condition = {
            condition: int(raw_rows_by_condition.get(condition, 0) - final_rows_by_condition.get(condition, 0))
            for condition in sorted(raw_rows_by_condition)
        }
        feature_shifts = _feature_shift_ranking(baseline_subject_df, subject_df)
        orderings = _evaluate_source_orderings(
            feature_df,
            mad_partitions[tag],
            results_v3_dir=results_v3_dir,
        )
        ordering_changes = {
            condition: {
                'baseline_best': baseline_orderings.get(condition, {}).get('best_classifier'),
                'current_best': orderings.get(condition, {}).get('best_classifier'),
                'changed': (
                    baseline_orderings.get(condition, {}).get('best_classifier')
                    != orderings.get(condition, {}).get('best_classifier')
                ) if (
                    baseline_orderings.get(condition, {}).get('status') == 'ok'
                    and orderings.get(condition, {}).get('status') == 'ok'
                ) else None,
            }
            for condition in CONDITIONS
        }
        subjects_crossing_dfa_threshold = sorted([
            subject_id
            for subject_id, n_rows in per_subject_rows.items()
            if n_rows < DFA_MIN_SEGMENT_LENGTH
        ])
        robust_subject_stats = filter_stats.get('per_subject_robust_stats', [])
        disproportionate = sorted(
            robust_subject_stats,
            key=lambda item: item.get('rows_removed_robust', 0),
            reverse=True,
        )[:5]
        mad_report[str(multiplier)] = {
            'paths': mad_paths[tag],
            'rows': int(feature_df.height),
            'subjects': int(feature_df.n_unique('subject_id')),
            'rows_by_condition': condition_rows,
            'removed_rows_by_condition': removed_rows_by_condition,
            'removed_rows_by_subject': removed_rows_by_subject,
            'disproportionately_affected_subjects': disproportionate,
            'subjects_crossing_dfa_eligibility': subjects_crossing_dfa_threshold,
            'feature_rank_shifts_top10': feature_shifts[:10],
            'source_model_orderings': orderings,
            'source_model_ordering_changes_vs_3_0': ordering_changes,
            'filter_stats': {
                'raw_rows': filter_stats.get('raw_rows'),
                'rows_after_hard_filter': filter_stats.get('rows_after_hard_filter'),
                'rows_after_robust_filter': filter_stats.get('rows_after_robust_filter'),
                'rows_removed_hard_filter': filter_stats.get('rows_removed_hard_filter'),
                'rows_removed_robust_filter': filter_stats.get('rows_removed_robust_filter'),
                'subjects_skipped_robust_filter': filter_stats.get('subjects_skipped_robust_filter', []),
                'robust_mad_multiplier': filter_stats.get('robust_mad_multiplier'),
            },
        }

    dfa_values = _compute_dfa_policy_values(baseline_df)
    concat_subjects = [payload['subject_id'] for payload in dfa_values.values()]
    concat_alpha = [float(payload['concat_alpha']) for payload in dfa_values.values()]
    longest_alpha = [
        float(payload['longest_alpha']) for payload in dfa_values.values()
        if payload['longest_alpha'] is not None
    ]
    segmented_alpha = [
        float(payload['segmented_alpha']) for payload in dfa_values.values()
        if payload['segmented_alpha'] is not None
    ]
    paired_longest = [
        (float(payload['concat_alpha']), float(payload['longest_alpha']))
        for payload in dfa_values.values()
        if payload['longest_alpha'] is not None
    ]
    paired_segmented = [
        (float(payload['concat_alpha']), float(payload['segmented_alpha']))
        for payload in dfa_values.values()
        if payload['segmented_alpha'] is not None
    ]
    longest_df, longest_summary = _apply_dfa_policy(
        baseline_df,
        dfa_values,
        'longest_contiguous_segment',
    )
    segmented_df, segmented_summary = _apply_dfa_policy(
        baseline_df,
        dfa_values,
        'segmented_summary',
    )
    dfa_orderings = {
        'concatenated': baseline_orderings,
        'longest_contiguous_segment': (
            {'status': 'not_evaluable'} if longest_df is None
            else _evaluate_source_orderings(
                longest_df,
                baseline_partition,
                results_v3_dir=results_v3_dir,
            )
        ),
        'segmented_summary': (
            {'status': 'not_evaluable'} if segmented_df is None
            else _evaluate_source_orderings(
                segmented_df,
                baseline_partition,
                results_v3_dir=results_v3_dir,
            )
        ),
    }
    dfa_report = {
        'gap_definition': (
            'A new segment starts when retained raw_stride_index is not consecutive '
            'or elapsed_s is non-increasing.'
        ),
        'minimum_eligible_segment_length': DFA_MIN_SEGMENT_LENGTH,
        'per_subject': list(dfa_values.values()),
        'concatenated_policy': _policy_summary(dfa_values, 'concat_alpha'),
        'longest_contiguous_segment_policy': longest_summary,
        'segmented_summary_policy': segmented_summary,
        'concat_vs_longest_spearman': (
            None if not paired_longest
            else round(_spearman(
                [a for a, _ in paired_longest],
                [b for _, b in paired_longest],
            ) or 0.0, 6)
        ),
        'concat_vs_segmented_spearman': (
            None if not paired_segmented
            else round(_spearman(
                [a for a, _ in paired_segmented],
                [b for _, b in paired_segmented],
            ) or 0.0, 6)
        ),
        'max_abs_alpha_shift_longest': (
            None if not paired_longest
            else round(max(abs(a - b) for a, b in paired_longest), 6)
        ),
        'max_abs_alpha_shift_segmented': (
            None if not paired_segmented
            else round(max(abs(a - b) for a, b in paired_segmented), 6)
        ),
        'source_model_orderings': dfa_orderings,
    }

    report = {
        'mad_reference_multiplier': 3.0,
        'mad_diagnostic': mad_report,
        'dfa_diagnostic': dfa_report,
        'reference_paths': {
            'baseline_features': mad_paths['3_0']['features_path'],
            'baseline_partition': mad_paths['3_0']['partition_path'],
            'baseline_manifest': mad_paths['3_0']['manifest_path'],
            **companion_paths,
        },
        'notes': [
            'Diagnostics are written only under data/processed/v4_preflight and experiments/results/v4_preflight.',
            'The lightweight downstream ablation freezes v3 modal params and selected imbalance strategies to test ordering sensitivity without full retuning.',
            'No authoritative v4 Step 1 artifacts were written during this pass.',
        ],
    }

    atomic_write_json(paths['report'], report)
    return report, paths['report']

def run_local() -> dict[str, Any]:
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    processed_dir = Path(scaffold['processed_preflight'])
    results_dir = Path(scaffold['results_preflight'])
    report, report_path = build_step1_preflight_report(
        processed_dir=processed_dir,
        results_dir=results_dir,
        data_dir=DEFAULT_DATA_DIR,
        allow_build=True,
        results_v3_dir=REPO_ROOT / 'experiments' / 'results' / 'v3',
    )
    print(json.dumps(report, indent=2))
    print(f'\nWrote {report_path}')
    return {'report_path': str(report_path), 'report': report}


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version='3.12')
        .pip_install_from_requirements('requirements-core.txt')
        .env({'PYTHONPATH': '/root/src'})
        .add_local_dir('src', remote_path='/root/src')
    )
    app = modal.App('gait-transfer-v4-preflight-step1', image=image)
    volume = modal.Volume.from_name('gait-results', create_if_missing=True)

    @app.function(
        cpu=16,
        memory=24576,
        timeout=43200,
        volumes={'/results': volume},
        retries=1,
    )
    def run_step1_preflight_remote() -> str:
        raw_data_dir = Path(REMOTE_RAW_DATA_DIR)
        if not raw_data_dir.exists() or not any(raw_data_dir.glob('*.ts')):
            raise FileNotFoundError(
                'Missing raw GAITNDD dataset on the Modal volume at '
                f'{REMOTE_RAW_DATA_DIR}. Upload it first with:\n'
                '  modal volume put gait-results '
                'data/raw/gait-in-neurodegenerative-disease-database-1.0.0 '
                'raw/gait-in-neurodegenerative-disease-database-1.0.0'
            )
        processed_dir = Path(REMOTE_PROCESSED_DIR)
        results_dir = Path(REMOTE_RESULTS_DIR)
        processed_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        report, report_path = build_step1_preflight_report(
            processed_dir=processed_dir,
            results_dir=results_dir,
            data_dir=raw_data_dir,
            allow_build=True,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
        )
        volume.commit()
        summary = {
            'remote_output_path': str(report_path),
            'remote_processed_dir': str(processed_dir),
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{REMOTE_REPORT_GET_PATH} '
                'experiments/results/v4_preflight/step1_preflight_report.json'
            ),
            'mad_variants': list(report['mad_diagnostic'].keys()),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.local_entrypoint()
    def main() -> None:
        print('Submitting detached Modal Step 1 preflight diagnostic.', flush=True)
        print('Prerequisite remote raw-data path:', flush=True)
        print(f'  gait-results:/raw/gait-in-neurodegenerative-disease-database-1.0.0', flush=True)
        print('Required remote v3 comparison inputs:', flush=True)
        print('  gait-results:/results_v3/pd_results_v3.json', flush=True)
        print('  gait-results:/results_v3/hd_results_v3.json', flush=True)
        print('  gait-results:/results_v3/als_results_v3.json', flush=True)
        print('If missing, upload with:', flush=True)
        print(
            '  modal volume put gait-results '
            'data/raw/gait-in-neurodegenerative-disease-database-1.0.0 '
            'raw/gait-in-neurodegenerative-disease-database-1.0.0',
            flush=True,
        )
        print(
            '  modal volume put gait-results experiments/results/v3/pd_results_v3.json results_v3/pd_results_v3.json',
            flush=True,
        )
        print(
            '  modal volume put gait-results experiments/results/v3/hd_results_v3.json results_v3/hd_results_v3.json',
            flush=True,
        )
        print(
            '  modal volume put gait-results experiments/results/v3/als_results_v3.json results_v3/als_results_v3.json',
            flush=True,
        )
        print('Primary output JSON:', flush=True)
        print(f'  gait-results:/{REMOTE_REPORT_GET_PATH}', flush=True)
        print('Download command:', flush=True)
        print(
            '  modal volume get gait-results '
            f'{REMOTE_REPORT_GET_PATH} '
            'experiments/results/v4_preflight/step1_preflight_report.json',
            flush=True,
        )
        run_step1_preflight_remote.spawn()


if __name__ == '__main__':
    run_local()
