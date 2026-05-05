"""
Local runner for Step 5: noise robustness and sensitivity analysis.

Writes:
  experiments/results/v3/noise_robustness_v3.json
  experiments/results/v3/feature_sensitivity_v3.json
  experiments/results/v3/subject_sensitivity_v3.json
  experiments/results/v3/corruption_robustness_v3.json
  experiments/results/v3/conformal_v3.json

Requires existing Step 1–3 artifacts (gait_features.csv, control partition,
pd/hd/als_results.json). Cross-condition sections require
cross_condition_results.json and experiments/models/*.joblib from Step 3.

Usage (from repo root, venv active):
    python scripts/training/run_noise_robustness_local.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import robustness as rb  # noqa: E402
from features import get_feature_cols  # noqa: E402

RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "v3"
MODELS_DIR = REPO_ROOT / "experiments" / "models" / "v3"
CONDITIONS = ("pd", "hd", "als")


def main() -> None:
    t0 = time.time()
    print("Loading v3 gait_features and control partition...", flush=True)
    df = pl.read_csv(REPO_ROOT / "data" / "processed" / "v3" / "gait_features_v3.csv")
    with open(REPO_ROOT / "data" / "processed" / "v3" / "control_partition_v3.json") as f:
        partition = json.load(f)
    control_a = partition["control_A"]
    control_b = partition["control_B"]
    feature_cols = get_feature_cols('v3')

    within_by: dict[str, dict] = {}
    for cond in CONDITIONS:
        path = RESULTS_DIR / f"{cond}_results_v3.json"
        with open(path) as f:
            within_by[cond] = json.load(f)
        print(f"  Loaded {path.name}", flush=True)

    cross_path = RESULTS_DIR / "cross_condition_results_v3.json"
    cross_results: dict | None = None
    if cross_path.exists():
        with open(cross_path) as f:
            cross_results = json.load(f)
        print(f"  Loaded {cross_path.name}", flush=True)
    else:
        print(
            f"  Skip cross-condition blocks: {cross_path.name} not found",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Noise sweeps ─────────────────────────────────────────────────────────
    print("\nNoise sweeps (within-condition)...", flush=True)
    noise_out: dict = {"within": {}, "cross": {}}
    for cond in CONDITIONS:
        print(f"  within {cond.upper()} ...", flush=True)
        noise_out["within"][cond] = rb.evaluate_noise_sweep_within(
            cond, df, control_a, within_by[cond], feature_cols=feature_cols
        )

    if cross_results:
        print("Noise sweeps (cross-condition)...", flush=True)
        for direction_key in cross_results:
            src, tgt = rb.direction_key_to_pair(direction_key)
            print(f"  cross {direction_key} ...", flush=True)
            noise_out["cross"][direction_key] = rb.evaluate_noise_sweep_cross(
                src, tgt, df, control_a, control_b, MODELS_DIR, feature_cols=feature_cols
            )

    noise_path = RESULTS_DIR / "noise_robustness_v3.json"
    with open(noise_path, "w") as f:
        json.dump(noise_out, f, indent=2)
    print(f"Wrote {noise_path}", flush=True)

    # ── Feature permutation (F1 drop vs baseline) ────────────────────────────
    print("\nFeature permutation sensitivity...", flush=True)
    feat_out: dict = {"within": {}, "cross": {}}
    for cond in CONDITIONS:
        print(f"  within {cond.upper()} ...", flush=True)
        baseline = {
            c: float(within_by[cond]["classifiers"][c]["f1_macro"])
            for c in rb.CLF_ORDER
        }
        feat_out["within"][cond] = rb.permutation_importance_within(
            cond, df, control_a, within_by[cond], baseline, feature_cols=feature_cols
        )

    if cross_results:
        for direction_key, dr in cross_results.items():
            src, tgt = rb.direction_key_to_pair(direction_key)
            print(f"  cross {direction_key} ...", flush=True)
            baseline = {
                c: float(dr["classifiers"][c]["f1_macro"])
                for c in rb.CLF_ORDER
                if c in dr["classifiers"]
            }
            feat_out["cross"][direction_key] = rb.permutation_importance_cross(
                src, tgt, df, control_a, control_b, MODELS_DIR, baseline,
                feature_cols=feature_cols
            )

    feat_path = RESULTS_DIR / "feature_sensitivity_v3.json"
    with open(feat_path, "w") as f:
        json.dump(feat_out, f, indent=2)
    print(f"Wrote {feat_path}", flush=True)

    # ── Per-subject accuracy (from stored predictions) ───────────────────────
    print("\nPer-subject sensitivity (no retraining)...", flush=True)
    subj_out = rb.build_subject_sensitivity_json(
        CONDITIONS, df, control_a, within_by, cross_results,
        feature_cols=feature_cols
    )
    subj_path = RESULTS_DIR / "subject_sensitivity_v3.json"
    with open(subj_path, "w") as f:
        json.dump(subj_out, f, indent=2)
    print(f"Wrote {subj_path}", flush=True)

    # ── Structured corruption benchmark ────────────────────────────────────────
    print("\nStructured corruption benchmark...", flush=True)
    corr_out: dict = {"within": {}, "cross": {}}
    for cond in CONDITIONS:
        print(f"  within {cond.upper()} ...", flush=True)
        corr_out["within"][cond] = rb.evaluate_corruption_sweep_within(
            cond, df, control_a, within_by[cond], feature_cols=feature_cols
        )

    if cross_results:
        for direction_key in cross_results:
            src, tgt = rb.direction_key_to_pair(direction_key)
            print(f"  cross {direction_key} ...", flush=True)
            corr_out["cross"][direction_key] = rb.evaluate_corruption_sweep_cross(
                src, tgt, df, control_a, control_b, MODELS_DIR, feature_cols=feature_cols
            )
    corr_path = RESULTS_DIR / "corruption_robustness_v3.json"
    with open(corr_path, "w") as f:
        json.dump(corr_out, f, indent=2)
    print(f"Wrote {corr_path}", flush=True)

    # ── Split conformal diagnostics ────────────────────────────────────────────
    print("\nSplit conformal diagnostics...", flush=True)
    conf_out: dict = {"within": {}, "cross": {}}
    for cond in CONDITIONS:
        print(f"  within {cond.upper()} ...", flush=True)
        conf_out["within"][cond] = rb.evaluate_conformal_within(
            cond, df, control_a, within_by[cond], feature_cols=feature_cols
        )
    if cross_results:
        for direction_key in cross_results:
            src, tgt = rb.direction_key_to_pair(direction_key)
            print(f"  cross {direction_key} ...", flush=True)
            conf_out["cross"][direction_key] = rb.evaluate_conformal_cross(
                src, tgt, df, control_a, control_b, MODELS_DIR, feature_cols=feature_cols
            )
    conf_path = RESULTS_DIR / "conformal_v3.json"
    with open(conf_path, "w") as f:
        json.dump(conf_out, f, indent=2)
    print(f"Wrote {conf_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
