"""
Real-data SVC probability and aggregation diagnostic for the v4 preflight pass.

Local mode:
    python scripts/verification/test_v4_svc_probability_diagnostics.py

Detached Modal mode:
    modal run --detach scripts/verification/test_v4_svc_probability_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import LeaveOneGroupOut

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

from features import build_feature_matrix, get_feature_cols  # type: ignore
from train import (  # type: ignore
    _configure_classifier_for_resampling,
    _get_fit_kwargs,
    _subject_level_arrays,
    build_pipeline,
    get_classifier_configs,
)
from v4_provenance import atomic_write_json, ensure_v4_preflight_scaffold  # type: ignore

CONDITIONS = ('pd', 'hd', 'als')
STRATEGIES = ('synthetic', 'balanced', 'raw')
REMOTE_FEATURES_PATH = '/results/processed_v4_preflight/gait_features_v4_reference.csv'
REMOTE_PARTITION_PATH = '/results/processed_v4_preflight/control_partition_v4_reference.json'
REMOTE_MANIFEST_PATH = '/results/processed_v4_preflight/preprocessing_manifest_v4_reference.json'
REMOTE_RESULTS_DIR = '/results/results_v4_preflight'
REMOTE_V3_RESULTS_DIR = '/results/results_v3'
REMOTE_OUTPUT_NAME = 'svc_probability_diagnostic.json'
REMOTE_GET_PATH = f'results_v4_preflight/{REMOTE_OUTPUT_NAME}'


def _reference_artifacts(
    *,
    processed_dir: Path,
    allow_build: bool,
) -> tuple[pl.DataFrame, dict[str, list[str]], dict[str, str]]:
    features_path = processed_dir / 'gait_features_v4_reference.csv'
    partition_path = processed_dir / 'control_partition_v4_reference.json'
    manifest_path = processed_dir / 'preprocessing_manifest_v4_reference.json'
    if not (features_path.exists() and partition_path.exists() and manifest_path.exists()):
        if not allow_build:
            raise FileNotFoundError(
                'Missing preflight Step 1 artifacts on the Modal volume. '
                'Run the detached Step 1 preflight diagnostic first.'
            )
        feature_df, partition = build_feature_matrix(
            processed_dir=processed_dir,
            output_filename=features_path.name,
            feature_cols=get_feature_cols('v4'),
            feature_set_version='v4',
            filter_strategy='v3',
            control_partition_version='v4',
            partition_output_filename=partition_path.name,
            manifest_filename=manifest_path.name,
            robust_mad_multiplier=3.0,
        )
        return feature_df, partition, {
            'features_path': str(features_path),
            'partition_path': str(partition_path),
            'manifest_path': str(manifest_path),
        }
    with open(partition_path) as f:
        partition = json.load(f)
    return pl.read_csv(features_path), partition, {
        'features_path': str(features_path),
        'partition_path': str(partition_path),
        'manifest_path': str(manifest_path),
    }


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
        condition: json.loads((base / f'{condition}_results_v3.json').read_text())
        for condition in CONDITIONS
    }


def _reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        if hi < 1.0:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            out.append({
                'bin_start': round(float(lo), 3),
                'bin_end': round(float(hi), 3),
                'count': 0,
                'mean_predicted_probability': None,
                'observed_rate': None,
            })
            continue
        out.append({
            'bin_start': round(float(lo), 3),
            'bin_end': round(float(hi), 3),
            'count': int(np.sum(mask)),
            'mean_predicted_probability': round(float(np.mean(y_prob[mask])), 6),
            'observed_rate': round(float(np.mean(y_true[mask])), 6),
        })
    return out


def build_svc_probability_report(
    *,
    feature_df: pl.DataFrame,
    partition: dict[str, list[str]],
    results_v3_dir: Path | None = None,
) -> dict[str, Any]:
    control_a = partition['control_A']
    feature_cols = get_feature_cols('v4')
    v3_within = _load_v3_within_results(results_v3_dir)
    svm_template = get_classifier_configs()['svm']['clf']

    report: dict[str, Any] = {
        'conditions_represented': list(CONDITIONS),
        'strategies_represented': list(STRATEGIES),
        'per_condition': {},
    }

    for condition in CONDITIONS:
        pool = feature_df.filter(
            (pl.col('condition') == condition) | pl.col('subject_id').is_in(control_a)
        )
        X = pool.select(feature_cols).to_numpy().astype(np.float64)
        y = pool['label'].to_numpy().astype(int)
        groups = pool['subject_id'].to_numpy().astype(str)
        params = v3_within[condition]['classifiers']['svm']['modal_params']
        outer = LeaveOneGroupOut()
        condition_report: dict[str, Any] = {
            'modal_params_used': params,
            'n_outer_folds': int(len(np.unique(groups))),
            'strategies': {},
        }

        for strategy in STRATEGIES:
            row_true_all: list[np.ndarray] = []
            row_pred_all: list[np.ndarray] = []
            row_prob_all: list[np.ndarray] = []
            row_subject_all: list[np.ndarray] = []
            row_decision_all: list[np.ndarray] = []

            for train_idx, test_idx in outer.split(X, y, groups):
                clf_variant = _configure_classifier_for_resampling(
                    'svm',
                    clone(svm_template),
                    strategy,
                )
                pipeline = build_pipeline('svm', clf_variant, imbalance_strategy=strategy)
                pipeline.set_params(**params)
                fit_kwargs = _get_fit_kwargs('svm', y[train_idx], strategy)
                pipeline.fit(X[train_idx], y[train_idx], **fit_kwargs)

                row_true_all.append(y[test_idx])
                row_pred_all.append(np.asarray(pipeline.predict(X[test_idx]), dtype=int))
                row_prob_all.append(np.asarray(pipeline.predict_proba(X[test_idx])[:, 1], dtype=float))
                row_subject_all.append(groups[test_idx].astype(str))
                row_decision_all.append(np.asarray(pipeline.decision_function(X[test_idx]), dtype=float))

            y_true = np.concatenate(row_true_all)
            y_pred = np.concatenate(row_pred_all)
            y_prob = np.concatenate(row_prob_all)
            subject_ids = np.concatenate(row_subject_all)
            decision_scores = np.concatenate(row_decision_all)

            mean_prob = _subject_level_arrays(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                subject_ids=subject_ids,
                decision_scores=decision_scores,
                subject_aggregation_rule='mean_probability',
            )
            majority_vote = _subject_level_arrays(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                subject_ids=subject_ids,
                decision_scores=decision_scores,
                subject_aggregation_rule='majority_vote',
            )
            mean_decision = _subject_level_arrays(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                subject_ids=subject_ids,
                decision_scores=decision_scores,
                subject_aggregation_rule='mean_decision_score',
            )

            mean_prob_pred = np.asarray(mean_prob[1], dtype=int)
            majority_pred = np.asarray(majority_vote[1], dtype=int)
            mean_decision_pred = np.asarray(mean_decision[1], dtype=int)

            condition_report['strategies'][strategy] = {
                'brier_score_stride': round(float(brier_score_loss(y_true, y_prob)), 6),
                'log_loss_stride': round(
                    float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
                    6,
                ),
                'reliability_bins_stride': _reliability_bins(y_true, y_prob),
                'mean_probability_vs_majority_vote_subject_disagreement': round(
                    float(np.mean(mean_prob_pred != majority_pred)),
                    6,
                ),
                'mean_probability_vs_mean_decision_subject_disagreement': round(
                    float(np.mean(mean_prob_pred != mean_decision_pred)),
                    6,
                ),
                'mean_probability_subject_scores': np.round(
                    np.asarray(mean_prob[2], dtype=float),
                    6,
                ).tolist(),
                'mean_decision_subject_scores': np.round(
                    np.asarray(mean_decision[2], dtype=float),
                    6,
                ).tolist(),
            }

        synthetic = condition_report['strategies']['synthetic']
        balanced = condition_report['strategies']['balanced']
        raw = condition_report['strategies']['raw']
        mean_non_synthetic_log_loss = float(np.mean([
            balanced['log_loss_stride'],
            raw['log_loss_stride'],
        ]))
        synthetic_vs_non_synthetic_gap = synthetic['log_loss_stride'] - mean_non_synthetic_log_loss
        if (
            synthetic_vs_non_synthetic_gap > 0.05
            or synthetic['mean_probability_vs_majority_vote_subject_disagreement'] > 0.15
        ):
            condition_report['recalibration_recommendation'] = 'separate_design_review_recommended'
        else:
            condition_report['recalibration_recommendation'] = 'no_clear_need_from_preflight'

        report['per_condition'][condition] = condition_report

    return report


def run_local() -> dict[str, Any]:
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    processed_dir = Path(scaffold['processed_preflight'])
    results_dir = Path(scaffold['results_preflight'])
    df, partition, input_paths = _reference_artifacts(
        processed_dir=processed_dir,
        allow_build=True,
    )
    report = build_svc_probability_report(feature_df=df, partition=partition)
    report['input_paths'] = input_paths
    output_path = results_dir / REMOTE_OUTPUT_NAME
    atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2))
    print(f'\nWrote {output_path}')
    return {'report_path': str(output_path), 'report': report}


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version='3.12')
        .pip_install_from_requirements('requirements-core.txt')
        .env({'PYTHONPATH': '/root/src'})
        .add_local_dir('src', remote_path='/root/src')
    )
    app = modal.App('gait-transfer-v4-preflight-svc', image=image)
    volume = modal.Volume.from_name('gait-results', create_if_missing=True)

    @app.function(
        cpu=12,
        memory=12288,
        timeout=21600,
        volumes={'/results': volume},
        retries=1,
    )
    def run_svc_probability_remote() -> str:
        processed_dir = Path('/results/processed_v4_preflight')
        results_dir = Path(REMOTE_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        df, partition, input_paths = _reference_artifacts(
            processed_dir=processed_dir,
            allow_build=False,
        )
        report = build_svc_probability_report(
            feature_df=df,
            partition=partition,
            results_v3_dir=Path(REMOTE_V3_RESULTS_DIR),
        )
        report['input_paths'] = input_paths
        output_path = results_dir / REMOTE_OUTPUT_NAME
        atomic_write_json(output_path, report)
        volume.commit()
        summary = {
            'remote_output_path': str(output_path),
            'modal_volume_get_command': (
                'modal volume get gait-results '
                f'{REMOTE_GET_PATH} '
                f'experiments/results/v4_preflight/{REMOTE_OUTPUT_NAME}'
            ),
            'conditions': report['conditions_represented'],
            'strategies': report['strategies_represented'],
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.local_entrypoint()
    def main() -> None:
        print('Submitting detached Modal SVC preflight diagnostic.', flush=True)
        print('Required remote inputs:', flush=True)
        print(f'  gait-results:/{REMOTE_FEATURES_PATH.removeprefix("/results/")}', flush=True)
        print(f'  gait-results:/{REMOTE_PARTITION_PATH.removeprefix("/results/")}', flush=True)
        print('  gait-results:/results_v3/pd_results_v3.json', flush=True)
        print('  gait-results:/results_v3/hd_results_v3.json', flush=True)
        print('  gait-results:/results_v3/als_results_v3.json', flush=True)
        print('Output JSON:', flush=True)
        print(f'  gait-results:/{REMOTE_GET_PATH}', flush=True)
        print('Download command:', flush=True)
        print(
            '  modal volume get gait-results '
            f'{REMOTE_GET_PATH} '
            f'experiments/results/v4_preflight/{REMOTE_OUTPUT_NAME}',
            flush=True,
        )
        run_svc_probability_remote.spawn()


if __name__ == '__main__':
    run_local()
