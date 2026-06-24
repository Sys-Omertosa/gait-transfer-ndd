"""
Subject-level Step 6 cohort-view diagnostic for the v4 preflight pass.

Local mode:
    python scripts/verification/test_v4_step6_views.py

Detached Modal mode:
    modal run --detach scripts/verification/test_v4_step6_views.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local non-Modal unit path
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

from features import build_subject_level_matrix, get_feature_cols  # type: ignore
from v4_provenance import atomic_write_json, ensure_v4_preflight_scaffold  # type: ignore

LOCAL_OUTPUT_NAME = 'step6_views_diagnostic.json'
REMOTE_OUTPUT_NAME = 'step6_views_diagnostic.json'
REMOTE_SUBJECT_MATRIX = '/results/processed_v4_preflight/gait_features_v4_subject_level_reference.csv'
REMOTE_FEATURE_MATRIX = '/results/processed_v4_preflight/gait_features_v4_reference.csv'
REMOTE_RESULTS_DIR = '/results/results_v4_preflight'
REMOTE_GET_PATH = f'results_v4_preflight/{REMOTE_OUTPUT_NAME}'


def _load_subject_level_df_local() -> tuple[pl.DataFrame, dict[str, str], dict[str, list[str]] | None]:
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    processed_dir = Path(scaffold['processed_preflight'])
    subject_matrix_path = processed_dir / 'gait_features_v4_subject_level_reference.csv'
    partition_path = processed_dir / 'control_partition_v4_reference.json'
    partition = None
    if partition_path.exists():
        partition = json.loads(partition_path.read_text())
    if subject_matrix_path.exists():
        return pl.read_csv(subject_matrix_path), {
            'subject_matrix_path': str(subject_matrix_path),
        }, partition

    feature_candidates = [
        processed_dir / 'gait_features_v4_reference.csv',
        REPO_ROOT / 'data' / 'processed' / 'v4' / 'gait_features_v4.csv',
        REPO_ROOT / 'data' / 'processed' / 'v3' / 'gait_features_v3.csv',
    ]
    for candidate in feature_candidates:
        if candidate.exists():
            return build_subject_level_matrix(
                pl.read_csv(candidate),
                feature_set_version='v4',
            ), {
                'feature_matrix_path': str(candidate),
                'subject_matrix_path': 'derived_in_memory',
            }, partition
    raise FileNotFoundError(
        'No local preflight/canonical feature matrix found for Step 6 preparation.'
    )


def _load_subject_level_df_remote() -> tuple[pl.DataFrame, dict[str, str], dict[str, list[str]]]:
    subject_matrix_path = Path(REMOTE_SUBJECT_MATRIX)
    partition_path = Path('/results/processed_v4_preflight/control_partition_v4_reference.json')
    if not partition_path.exists():
        raise FileNotFoundError(
            'Missing /results/processed_v4_preflight/control_partition_v4_reference.json on the '
            'Modal volume. Run the detached Step 1 preflight diagnostic first.'
        )
    partition = json.loads(partition_path.read_text())
    if subject_matrix_path.exists():
        return pl.read_csv(subject_matrix_path), {
            'subject_matrix_path': str(subject_matrix_path),
        }, partition
    feature_matrix_path = Path(REMOTE_FEATURE_MATRIX)
    if feature_matrix_path.exists():
        return build_subject_level_matrix(
            pl.read_csv(feature_matrix_path),
            feature_set_version='v4',
        ), {
            'feature_matrix_path': str(feature_matrix_path),
            'subject_matrix_path': 'derived_in_memory_from_preflight_reference',
        }, partition
    raise FileNotFoundError(
        'Missing /results/processed_v4_preflight/gait_features_v4_reference.csv on the Modal '
        'volume. Run the detached Step 1 preflight diagnostic first.'
    )


def _view_df(
    df: pl.DataFrame,
    view_name: str,
    *,
    partition: dict[str, list[str]] | None,
) -> tuple[pl.DataFrame, list[int]]:
    if view_name == 'disease_only':
        view = df.filter(pl.col('condition') != 'control')
        labels = [0 if c == 'pd' else 1 if c == 'hd' else 2 for c in view['condition']]
        return view, labels
    if view_name == 'all_subjects':
        view = df
        labels = [
            0 if c == 'control'
            else 1 if c == 'pd'
            else 2 if c == 'hd'
            else 3
            for c in view['condition']
        ]
        return view, labels
    if view_name == 'pathological_vs_control':
        view = df
        labels = view['label'].to_list()
        return view, labels
    if view_name == 'control_role_audit':
        view = df.filter(pl.col('condition') == 'control')
        if partition is None:
            raise ValueError('Control-role audit requires a control partition.')
        labels = [
            0 if sid in set(partition['control_A']) else 1
            for sid in view['subject_id']
        ]
        return view, labels
    raise ValueError(view_name)


def _seed_stability(X_pca: np.ndarray, k: int) -> float:
    reference = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X_pca)
    aris = []
    for seed in range(5):
        labels = KMeans(n_clusters=k, n_init=20, random_state=100 + seed).fit_predict(X_pca)
        aris.append(adjusted_rand_score(reference, labels))
    return round(float(np.mean(aris)), 6)


def build_step6_views_report(
    subject_df: pl.DataFrame,
    *,
    partition: dict[str, list[str]] | None,
) -> dict[str, Any]:
    feature_cols = get_feature_cols('v4')
    report: dict[str, Any] = {}
    for view_name in (
        'disease_only',
        'all_subjects',
        'pathological_vs_control',
        'control_role_audit',
    ):
        view_df, labels = _view_df(subject_df, view_name, partition=partition)
        if view_df.height < 3:
            continue
        X = view_df.select(feature_cols).to_numpy().astype(np.float64)
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(5, X_std.shape[1], X_std.shape[0]))
        X_pca = pca.fit_transform(X_std)
        k_metrics = {}
        for k in range(2, min(10, len(X_std) - 1) + 1):
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            cluster_labels = km.fit_predict(X_pca)
            k_metrics[str(k)] = {
                'silhouette': round(float(silhouette_score(X_pca, cluster_labels)), 6),
                'inertia': round(float(km.inertia_), 6),
                'calinski_harabasz': round(
                    float(calinski_harabasz_score(X_pca, cluster_labels)),
                    6,
                ),
                'davies_bouldin': round(
                    float(davies_bouldin_score(X_pca, cluster_labels)),
                    6,
                ),
                'ari_against_labels': round(
                    float(adjusted_rand_score(labels, cluster_labels)),
                    6,
                ),
                'seed_stability_mean_ari': _seed_stability(X_pca, k),
            }
        report[view_name] = {
            'n_subjects': int(view_df.height),
            'feature_cols': feature_cols,
            'pca_explained_variance_ratio': np.round(
                pca.explained_variance_ratio_,
                6,
            ).tolist(),
            'k_metrics': k_metrics,
        }
    return report


def run_local() -> dict[str, Any]:
    subject_df, input_paths, partition = _load_subject_level_df_local()
    report = build_step6_views_report(subject_df, partition=partition)
    report['input_paths'] = input_paths
    scaffold = ensure_v4_preflight_scaffold(REPO_ROOT)
    output_path = Path(scaffold['results_preflight']) / LOCAL_OUTPUT_NAME
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
    app = modal.App('gait-transfer-v4-preflight-step6', image=image)
    volume = modal.Volume.from_name('gait-results', create_if_missing=True)

    @app.function(
        cpu=8,
        memory=8192,
        timeout=7200,
        volumes={'/results': volume},
        retries=1,
    )
    def run_step6_views_remote() -> str:
        from pathlib import Path as _Path

        subject_df, input_paths, partition = _load_subject_level_df_remote()
        report = build_step6_views_report(subject_df, partition=partition)
        report['input_paths'] = input_paths
        results_dir = _Path(REMOTE_RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
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
            'views': list(report.keys()),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return json.dumps(summary, indent=2)

    @app.local_entrypoint()
    def main() -> None:
        print('Submitting detached Modal Step 6 preflight diagnostic.', flush=True)
        print('Prerequisite remote input:', flush=True)
        print(
            '  gait-results:/processed_v4_preflight/gait_features_v4_subject_level_reference.csv',
            flush=True,
        )
        print('Primary output:', flush=True)
        print(f'  gait-results:/{REMOTE_GET_PATH}', flush=True)
        print('Download command:', flush=True)
        print(
            '  modal volume get gait-results '
            f'{REMOTE_GET_PATH} '
            f'experiments/results/v4_preflight/{REMOTE_OUTPUT_NAME}',
            flush=True,
        )
        run_step6_views_remote.spawn()


if __name__ == '__main__':
    run_local()
