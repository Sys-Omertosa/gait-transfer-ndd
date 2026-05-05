"""
Modal runner for Step 5: full noise robustness and sensitivity analysis.

Runs the full sweep defined in src/robustness.py against the publication-track
v3 artifacts already stored on the shared Modal volume.

Writes to Modal volume (/results/results_v3):
  - noise_robustness_v3.json
  - feature_sensitivity_v3.json
  - subject_sensitivity_v3.json
  - corruption_robustness_v3.json
  - conformal_v3.json

Usage (from repo root with venv active):
    modal run scripts/training/run_noise_robustness_modal.py
"""

from __future__ import annotations

from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements-core.txt")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gait-transfer-noise-robustness", image=image)
volume = modal.Volume.from_name("gait-results", create_if_missing=True)


@app.function(
    cpu=16,
    memory=16384,
    timeout=86400,
    volumes={"/results": volume},
)
def run_step5_full() -> str:
    import json
    import time
    from pathlib import Path as _Path

    import polars as pl

    import robustness as rb  # type: ignore[import-not-found]
    from features import get_feature_cols

    t0 = time.time()
    features_path = _Path('/results/processed_v3/gait_features_v3.csv')
    partition_path = _Path('/results/processed_v3/control_partition_v3.json')
    results_dir = _Path('/results/results_v3')
    models_dir = _Path('/results/models_v3')

    missing = [
        str(path)
        for path in (
            features_path,
            partition_path,
            results_dir / 'pd_results_v3.json',
            results_dir / 'hd_results_v3.json',
            results_dir / 'als_results_v3.json',
            results_dir / 'cross_condition_results_v3.json',
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            'Missing v3 prerequisites on the Modal volume. '
            'Run preprocessing, within-condition, and cross-condition steps first. '
            f'Missing: {missing}'
        )

    df = pl.read_csv(str(features_path))
    with open(partition_path) as f:
        partition = json.load(f)
    control_a = partition["control_A"]
    control_b = partition["control_B"]
    feature_cols = get_feature_cols('v3')

    within_by: dict[str, dict] = {
        "pd": json.loads((results_dir / "pd_results_v3.json").read_text()),
        "hd": json.loads((results_dir / "hd_results_v3.json").read_text()),
        "als": json.loads((results_dir / "als_results_v3.json").read_text()),
    }
    cross_results: dict = json.loads(
        (results_dir / "cross_condition_results_v3.json").read_text()
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    print("Step 5 full run started (Modal).", flush=True)
    print(f"SIGMA_LEVELS={rb.SIGMA_LEVELS}, repeats={rb.N_NOISE_REPEATS}", flush=True)

    noise_out: dict = {"within": {}, "cross": {}}
    print("Noise sweeps (within-condition)...", flush=True)
    for cond in ("pd", "hd", "als"):
        print(f"  within {cond.upper()} ...", flush=True)
        noise_out["within"][cond] = rb.evaluate_noise_sweep_within(
            cond, df, control_a, within_by[cond], feature_cols=feature_cols
        )

    print("Noise sweeps (cross-condition)...", flush=True)
    for direction_key in cross_results:
        src, tgt = rb.direction_key_to_pair(direction_key)
        print(f"  cross {direction_key} ...", flush=True)
        noise_out["cross"][direction_key] = rb.evaluate_noise_sweep_cross(
            src, tgt, df, control_a, control_b, models_dir, feature_cols=feature_cols
        )

    noise_path = results_dir / "noise_robustness_v3.json"
    with open(noise_path, "w") as f:
        json.dump(noise_out, f, indent=2)
    print(f"Wrote {noise_path}", flush=True)

    feat_out: dict = {"within": {}, "cross": {}}
    print("Feature permutation sensitivity (within-condition)...", flush=True)
    for cond in ("pd", "hd", "als"):
        baseline = {
            c: float(within_by[cond]["classifiers"][c]["f1_macro"])
            for c in rb.CLF_ORDER
        }
        print(f"  within {cond.upper()} ...", flush=True)
        feat_out["within"][cond] = rb.permutation_importance_within(
            cond, df, control_a, within_by[cond], baseline, feature_cols=feature_cols
        )

    print("Feature permutation sensitivity (cross-condition)...", flush=True)
    for direction_key, dr in cross_results.items():
        src, tgt = rb.direction_key_to_pair(direction_key)
        baseline = {
            c: float(dr["classifiers"][c]["f1_macro"])
            for c in rb.CLF_ORDER
            if c in dr["classifiers"]
        }
        print(f"  cross {direction_key} ...", flush=True)
        feat_out["cross"][direction_key] = rb.permutation_importance_cross(
            src, tgt, df, control_a, control_b, models_dir, baseline,
            feature_cols=feature_cols
        )

    feat_path = results_dir / "feature_sensitivity_v3.json"
    with open(feat_path, "w") as f:
        json.dump(feat_out, f, indent=2)
    print(f"Wrote {feat_path}", flush=True)

    print("Per-subject sensitivity...", flush=True)
    subj_out = rb.build_subject_sensitivity_json(
        ("pd", "hd", "als"), df, control_a, within_by, cross_results,
        feature_cols=feature_cols
    )
    subj_path = results_dir / "subject_sensitivity_v3.json"
    with open(subj_path, "w") as f:
        json.dump(subj_out, f, indent=2)
    print(f"Wrote {subj_path}", flush=True)

    print("Structured corruption benchmark...", flush=True)
    corr_out: dict = {"within": {}, "cross": {}}
    for cond in ("pd", "hd", "als"):
        print(f"  within {cond.upper()} ...", flush=True)
        corr_out["within"][cond] = rb.evaluate_corruption_sweep_within(
            cond, df, control_a, within_by[cond], feature_cols=feature_cols
        )
    for direction_key in cross_results:
        src, tgt = rb.direction_key_to_pair(direction_key)
        print(f"  cross {direction_key} ...", flush=True)
        corr_out["cross"][direction_key] = rb.evaluate_corruption_sweep_cross(
            src, tgt, df, control_a, control_b, models_dir, feature_cols=feature_cols
        )
    corr_path = results_dir / "corruption_robustness_v3.json"
    with open(corr_path, "w") as f:
        json.dump(corr_out, f, indent=2)
    print(f"Wrote {corr_path}", flush=True)

    print("Split conformal diagnostics...", flush=True)
    conf_out: dict = {"within": {}, "cross": {}}
    for cond in ("pd", "hd", "als"):
        print(f"  within {cond.upper()} ...", flush=True)
        conf_out["within"][cond] = rb.evaluate_conformal_within(
            cond, df, control_a, within_by[cond], feature_cols=feature_cols
        )
    for direction_key in cross_results:
        src, tgt = rb.direction_key_to_pair(direction_key)
        print(f"  cross {direction_key} ...", flush=True)
        conf_out["cross"][direction_key] = rb.evaluate_conformal_cross(
            src, tgt, df, control_a, control_b, models_dir, feature_cols=feature_cols
        )
    conf_path = results_dir / "conformal_v3.json"
    with open(conf_path, "w") as f:
        json.dump(conf_out, f, indent=2)
    print(f"Wrote {conf_path}", flush=True)

    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed:.0f}s", flush=True)

    summary = {
        "noise_robustness_path": str(noise_path),
        "feature_sensitivity_path": str(feat_path),
        "subject_sensitivity_path": str(subj_path),
        "corruption_robustness_path": str(corr_path),
        "conformal_path": str(conf_path),
        "elapsed_seconds": round(elapsed, 2),
        "sigma_levels": list(rb.SIGMA_LEVELS),
        "repeats": rb.N_NOISE_REPEATS,
    }
    return json.dumps(summary, indent=2)


@app.local_entrypoint()
def main():
    print("Submitting full Step 5 run to Modal...", flush=True)
    print("Inputs are read from gait-results:/processed_v3 and gait-results:/results_v3.")
    print("No reduced sweep mode: full sigma grid and 30 repeats enabled.", flush=True)
    print(flush=True)

    summary_json = run_step5_full.remote()

    print("Step 5 run complete.", flush=True)
    print(summary_json, flush=True)
    print("\nDownload outputs:", flush=True)
    print("  modal volume get gait-results results_v3/noise_robustness_v3.json", flush=True)
    print("  modal volume get gait-results results_v3/feature_sensitivity_v3.json", flush=True)
    print("  modal volume get gait-results results_v3/subject_sensitivity_v3.json", flush=True)
    print("  modal volume get gait-results results_v3/corruption_robustness_v3.json", flush=True)
    print("  modal volume get gait-results results_v3/conformal_v3.json", flush=True)
