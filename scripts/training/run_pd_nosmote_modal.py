"""
Modal runner for the targeted PD no-SMOTE sensitivity evaluation.

This script computes a single-arm no-SMOTE artifact for the PD source pool and
writes `pd_results_v2_nosmote.json` to the Modal results volume. The emitted
JSON intentionally omits `f1_macro_smote` and `f1_macro_no_smote` because it is
not a winner-selection artifact; it records only the no-SMOTE arm.
This sensitivity run was motivated by the discovery that SMOTE augments the
control class (the minority) in all three within-condition pools, not the
disease class as originally assumed; the initial PD ablation omission was
therefore based on an incorrect class-imbalance assessment.

If a classifier is later manually promoted into the authoritative
`experiments/results/v2/pd_results_v2.json`, preserve schema consistency with the HD and ALS
artifacts by writing:
  - `f1_macro_smote` from the current authoritative PD SMOTE result
  - `f1_macro_no_smote` from this script's `f1_macro`
  - `selected_resampling = "no_smote"`
  - authoritative `modal_params`, `y_true`, and `y_pred` from the no-SMOTE run

Usage (from repo root with venv active):
    modal run scripts/training/run_pd_nosmote_modal.py

Download results after completion:
    modal volume get gait-results pd_results_v2_nosmote.json
"""

from pathlib import Path

import modal

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements-core.txt")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gait-transfer-pd-nosmote", image=image)

# ── Persistent volume for results ─────────────────────────────────────────────
volume = modal.Volume.from_name("gait-results", create_if_missing=True)


@app.function(
    cpu=16,
    memory=12288,
    timeout=86400,
    volumes={"/results": volume},
    retries=1,
)
def run_pd_no_smote(
    gait_features_csv: bytes,
    control_partition_json: bytes,
) -> str:
    """
    Run the PD no-SMOTE sensitivity evaluation and return the results JSON.

    PD is paired with the 8 Control Group A subjects exactly as in the standard
    within-condition source pool, but no synthetic control interpolation is
    applied in this targeted sensitivity run.
    """
    import json
    import os
    import tempfile

    import numpy as np
    import polars as pl

    from train import (
        ALL_FEATURE_COLS,
        DEFAULT_FEATURE_MATRIX_FILE,
        DEFAULT_FEATURE_SET_VERSION,
        DEFAULT_NORMALIZATION,
        _build_within_condition_output,
        _evaluate_within_condition_classifier,
        get_classifier_configs,
    )

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

    control_a = partition["control_A"]
    selected_feature_cols = list(ALL_FEATURE_COLS)

    pool = df.filter(
        (pl.col('condition') == 'pd') |
        pl.col('subject_id').is_in(control_a)
    )
    pool_subjects = pool.n_unique('subject_id')
    pool_strides = pool.shape[0]

    X = pool.select(selected_feature_cols).to_numpy().astype(np.float64)
    y = pool['label'].to_numpy().astype(int)
    groups = pool['subject_id'].to_numpy()

    clf_results: dict[str, dict] = {}
    partial_path = Path("/results/pd_results_v2_nosmote_partial.json")

    for clf_name, config in get_classifier_configs().items():
        result = _evaluate_within_condition_classifier(
            clf_name=clf_name,
            clf=config['clf'],
            param_grid=config['param_grid'],
            X=X,
            y=y,
            groups=groups,
            use_smote=False,
        )

        clf_results[clf_name] = {
            'f1_macro': result['f1_macro'],
            'f1_macro_ci_lower': result['f1_macro_ci_lower'],
            'f1_macro_ci_upper': result['f1_macro_ci_upper'],
            'modal_params': result['modal_params'],
            'modal_frequency': result['modal_frequency'],
            'y_true': result['y_true'],
            'y_pred': result['y_pred'],
            'selected_resampling': 'no_smote',
        }

        # This single-arm artifact intentionally omits paired SMOTE/no-SMOTE
        # fields; those are added only if a classifier is manually promoted
        # into the authoritative pd_results_v2.json artifact. The omitted SMOTE
        # arm would correspond to synthetic control interpolation within the
        # Control Group A stride space.
        partial_output = _build_within_condition_output(
            condition='pd',
            pool_subjects=pool_subjects,
            pool_strides=pool_strides,
            selected_feature_cols=selected_feature_cols,
            feature_matrix_file=DEFAULT_FEATURE_MATRIX_FILE,
            feature_set_version=DEFAULT_FEATURE_SET_VERSION,
            normalization=DEFAULT_NORMALIZATION,
            clf_results=clf_results,
        )
        with open(partial_path, "w") as f:
            json.dump(partial_output, f, indent=2)

    output = _build_within_condition_output(
        condition='pd',
        pool_subjects=pool_subjects,
        pool_strides=pool_strides,
        selected_feature_cols=selected_feature_cols,
        feature_matrix_file=DEFAULT_FEATURE_MATRIX_FILE,
        feature_set_version=DEFAULT_FEATURE_SET_VERSION,
        normalization=DEFAULT_NORMALIZATION,
        clf_results=clf_results,
    )

    out_path = Path("/results/pd_results_v2_nosmote.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main():
    """Launch the targeted PD no-SMOTE sensitivity evaluation on Modal."""
    repo_root = Path(__file__).resolve().parents[2]
    features_bytes = (repo_root / "data/processed/v2/gait_features_v2.csv").read_bytes()
    partition_bytes = (
        repo_root / "data/processed/control_partition.json").read_bytes()

    print("Launching PD no-SMOTE sensitivity run on Modal...")
    print("Single container: 16 CPU, 12288 MB RAM.")
    print()
    print(
        run_pd_no_smote.remote(
            gait_features_csv=features_bytes,
            control_partition_json=partition_bytes,
        )
    )
    print("\nRun complete.")
    print("Download results:")
    print("  modal volume get gait-results pd_results_v2_nosmote.json")
