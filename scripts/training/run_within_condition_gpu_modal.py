"""
Modal GPU runner for within-condition baseline training.

Runs all three conditions (pd, hd, als) in parallel on Modal GPU containers and
writes *_results_v2_gpu.json outputs with the same schema as the CPU Step 2
runner. GPU acceleration is attempted for XGBoost and LightGBM; when a GPU
backend is unavailable or fails at fit time, the classifier falls back to the
current CPU implementation automatically.

Random forest remains on CPU because the current authoritative RF design uses
class_weight='balanced', and we do not have a clean cuML-equivalent path that
preserves those semantics without algorithmic drift.

Usage (from repo root with venv active):
    modal run scripts/training/run_within_condition_gpu_modal.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class XGBGpuFallbackClassifier(BaseEstimator, ClassifierMixin):
    """Try GPU XGBoost first, then fall back to the CPU hist implementation."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        eval_metric: str = 'logloss',
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _build_model(self, *, use_gpu: bool) -> XGBClassifier:
        common_kwargs = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'eval_metric': self.eval_metric,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }
        if use_gpu:
            return XGBClassifier(device='cuda', **common_kwargs)
        return XGBClassifier(tree_method='hist', **common_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        try:
            self.model_ = self._build_model(use_gpu=True)
            self.model_.fit(X, y, **fit_kwargs)
            self.backend_ = 'gpu'
        except Exception as exc:
            print(
                f'[xgb gpu fallback] GPU fit failed ({exc!r}); using CPU hist backend.',
                flush=True,
            )
            self.model_ = self._build_model(use_gpu=False)
            self.model_.fit(X, y, **fit_kwargs)
            self.backend_ = 'cpu'

        self.classes_ = getattr(self.model_, 'classes_', np.unique(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X))


class LightGBMGpuFallbackClassifier(BaseEstimator, ClassifierMixin):
    """Try GPU LightGBM first, then fall back to CPU LightGBM."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 1.0,
        feature_fraction: float = 1.0,
        min_child_samples: int = 20,
        subsample_freq: int = 1,
        class_weight: str | None = None,
        random_state: int = 42,
        n_jobs: int = 1,
        verbose: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.feature_fraction = feature_fraction
        self.min_child_samples = min_child_samples
        self.subsample_freq = subsample_freq
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _build_model(self, *, use_gpu: bool) -> LGBMClassifier:
        common_kwargs = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'subsample': self.subsample,
            'feature_fraction': self.feature_fraction,
            'min_child_samples': self.min_child_samples,
            'subsample_freq': self.subsample_freq,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
        }
        if use_gpu:
            return LGBMClassifier(device='cuda', **common_kwargs)
        return LGBMClassifier(**common_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        try:
            self.model_ = self._build_model(use_gpu=True)
            self.model_.fit(X, y, **fit_kwargs)
            self.backend_ = 'gpu'
        except Exception as exc:
            print(
                f'[lgbm gpu fallback] GPU fit failed ({exc!r}); using CPU backend.',
                flush=True,
            )
            self.model_ = self._build_model(use_gpu=False)
            self.model_.fit(X, y, **fit_kwargs)
            self.backend_ = 'cpu'

        self.classes_ = getattr(self.model_, 'classes_', np.unique(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X))


def build_gpu_classifier_configs() -> dict[str, dict[str, Any]]:
    """Return classifier configs in GPU-first order for the v2 GPU runner."""
    from train import get_classifier_configs

    base = get_classifier_configs()
    gpu_configs: dict[str, dict[str, Any]] = {}

    gpu_configs['xgb'] = {
        'clf': XGBGpuFallbackClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        ),
        'param_grid': base['xgb']['param_grid'],
    }
    gpu_configs['lgbm'] = {
        'clf': LightGBMGpuFallbackClassifier(
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        'param_grid': base['lgbm']['param_grid'],
    }

    for clf_name in ['rf', 'knn', 'svm', 'dt', 'qda']:
        gpu_configs[clf_name] = base[clf_name]

    return gpu_configs


# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "cmake")
    .add_local_file(
        "requirements-core.txt",
        remote_path="/root/requirements-core.txt",
        copy=True,
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install -r /root/requirements-core.txt",
        (
            "bash -lc '"
            "git clone --depth 1 --recursive https://github.com/microsoft/LightGBM /tmp/LightGBM && "
            "cd /tmp/LightGBM && mkdir -p build && cd build && "
            "cmake -DUSE_CUDA=ON .. && make -j$(nproc) && "
            "cd ../python-package && python -m pip install ."
            " || true'"
        ),
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gait-transfer-training-gpu", image=image)
volume = modal.Volume.from_name("gait-results", create_if_missing=True)


@app.function(
    gpu="A10",
    cpu=16,
    memory=16384,
    timeout=86400,
    volumes={"/results": volume},
    retries=1,
)
def run_condition_gpu(
    gait_features_csv: bytes,
    control_partition_json: bytes,
    condition: str,
) -> str:
    """Run one within-condition experiment with GPU-capable classifier fallbacks."""
    import json
    import os
    import tempfile

    import polars as pl

    from train import run_within_condition

    tmp = tempfile.mkdtemp()
    features_path = os.path.join(tmp, "gait_features.csv")
    partition_path = os.path.join(tmp, "control_partition.json")

    with open(features_path, "wb") as f:
        f.write(gait_features_csv)
    with open(partition_path, "wb") as f:
        f.write(control_partition_json)

    df = pl.read_csv(features_path)
    with open(partition_path) as f:
        partition = json.load(f)

    output = run_within_condition(
        condition=condition,
        df=df,
        control_subjects=partition["control_A"],
        results_dir="/results",
        feature_matrix_file="v2/gait_features_v2.csv",
        results_filename=f"{condition}_results_v2_gpu.json",
        classifier_configs=build_gpu_classifier_configs(),
    )
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main():
    """Launch all three GPU Step 2 conditions in parallel and collect results."""
    repo_root = Path(__file__).resolve().parents[2]
    features_bytes = (
        repo_root / "data/processed/v2/gait_features_v2.csv").read_bytes()
    partition_bytes = (
        repo_root / "data/processed/control_partition.json").read_bytes()

    conditions = ["pd", "hd", "als"]

    print("Launching all three conditions in parallel on Modal GPU...", flush=True)
    print(
        "Each condition: 1x A10, 16 CPU, 16384 MB RAM. GPU: XGB, LightGBM. CPU: RF, KNN, SVM, DT, QDA.",
        flush=True,
    )
    print()

    futures = {
        condition: run_condition_gpu.spawn(
            gait_features_csv=features_bytes,
            control_partition_json=partition_bytes,
            condition=condition,
        )
        for condition in conditions
    }

    for condition, future in futures.items():
        print(f"\n{'=' * 60}")
        print(f"GPU Results - {condition.upper()}")
        print(f"{'=' * 60}")
        print(future.get())

    print("\nAll GPU conditions complete.")
    print("Download results:")
    for condition in conditions:
        print(
            f"  modal volume get gait-results {condition}_results_v2_gpu.json")
