"""
app/app.py

Step 8 — Streamlit demo for interactive gait-based NDD inference.

Three tabs driven entirely off the artefacts in experiments/results/ and the
preprocessed data/processed/gait_features.csv:

    1. Within-condition inference  — pick a cohort (PD/HD/ALS), classify one
       stride with the cohort's modal-RF pipeline, see SHAP contributions and
       PCA placement.
    2. Cross-condition transfer    — pick source and target cohorts, run the
       source RF on the target stride, and show the stored ΔF1 degradation.
    3. Noise robustness            — perturb an input stride with Gaussian
       noise at a chosen σ, visualise how predicted P(disease) drifts.

No raw .ts re-processing, no Step 2–5 sweep is re-run. RF pipelines are re-fit
once per session using the modal hyper-parameters stored in
experiments/results/{cond}_results.json and cached via @st.cache_resource.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Resolve project paths and make src/ importable. The app is launched as
#   streamlit run app/app.py
# from the project root, so Path(__file__).resolve().parents[1] is the root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from features import ALL_FEATURE_COLS  # noqa: E402
from train import build_pipeline  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

CONDITIONS = ("pd", "hd", "als")
COND_LABEL = {"pd": "Parkinson's disease", "hd": "Huntington's disease", "als": "ALS"}
COND_SHORT = {"pd": "PD", "hd": "HD", "als": "ALS"}
DATA_CSV = ROOT / "data" / "processed" / "gait_features.csv"
RESULTS_DIR = ROOT / "experiments" / "results"
PARTITION_JSON = ROOT / "data" / "processed" / "control_partition.json"

FEATURE_HELP = {
    "left_stride_s":       "Left-foot stride duration (s)",
    "right_stride_s":      "Right-foot stride duration (s)",
    "left_swing_s":        "Left-foot swing phase duration (s)",
    "right_swing_s":       "Right-foot swing phase duration (s)",
    "left_swing_pct":      "Left swing as % of stride",
    "right_swing_pct":     "Right swing as % of stride",
    "left_stance_s":       "Left-foot stance phase duration (s)",
    "right_stance_s":      "Right-foot stance phase duration (s)",
    "left_stance_pct":     "Left stance as % of stride",
    "right_stance_pct":    "Right stance as % of stride",
    "double_support_s":    "Double-support phase duration (s)",
    "double_support_pct":  "Double-support as % of stride",
    "asymmetry_index":     "|L-R| stride / mean stride (engineered, PD marker)",
    "cv_stride":           "Per-subject stride-time CV (engineered, HD marker)",
}


# ── Cached data & model loaders ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    """Full preprocessed feature matrix (14,753 × 17)."""
    return pd.read_csv(DATA_CSV)


@st.cache_data(show_spinner=False)
def load_partition() -> dict[str, list[str]]:
    with PARTITION_JSON.open() as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_within_results() -> dict[str, dict]:
    return {
        c: json.loads((RESULTS_DIR / f"{c}_results.json").read_text())
        for c in CONDITIONS
    }


@st.cache_data(show_spinner=False)
def load_cross_results() -> dict:
    return json.loads((RESULTS_DIR / "cross_condition_results.json").read_text())


@st.cache_data(show_spinner=False)
def load_noise_results() -> dict:
    path = RESULTS_DIR / "noise_robustness.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def cohort_feature_stats() -> dict[str, pd.DataFrame]:
    """Per-cohort p1/median/p99 for the 14 features, used for slider ranges."""
    df = load_features()
    out: dict[str, pd.DataFrame] = {}
    partition = load_partition()
    control_a = partition["control_A"]
    for cond in CONDITIONS:
        pool = df[(df["condition"] == cond) | (df["subject_id"].isin(control_a))]
        stats = pool[list(ALL_FEATURE_COLS)].agg(
            lambda s: pd.Series({
                "p1":     float(np.percentile(s, 1)),
                "median": float(np.median(s)),
                "p99":    float(np.percentile(s, 99)),
            })
        ).T
        out[cond] = stats
    return out


@st.cache_resource(show_spinner="Fitting cohort RF models (one-time, ~3 s total)…")
def fit_rf(condition: str):
    """Fit the Step-2 within-condition RF on the full (disease + Control A) pool."""
    df = load_features()
    partition = load_partition()
    control_a = partition["control_A"]
    pool = df[(df["condition"] == condition) | (df["subject_id"].isin(control_a))]
    X = pool[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float64)
    y = pool["label"].to_numpy(dtype=int)

    params = load_within_results()[condition]["classifiers"]["rf"]["modal_params"]
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
    pipe = build_pipeline("rf", rf)
    pipe.set_params(**params)
    pipe.fit(X, y)
    return pipe


@st.cache_resource(show_spinner=False)
def tree_explainer(condition: str):
    """SHAP TreeExplainer on the RF classifier inside the cached pipeline."""
    pipe = fit_rf(condition)
    clf = pipe.named_steps["clf"]
    return shap.TreeExplainer(clf)


@st.cache_resource(show_spinner=False)
def fit_pca_projection():
    """Step-6-identical scaler + PCA(2) fit on all 14,753 strides."""
    df = load_features()
    X = df[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2, random_state=42).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    return scaler, pca, X_pca, df[["subject_id", "condition", "label"]].reset_index(drop=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stride_CI_from_y(y_true: list[int], y_pred: list[int], n_boot: int = 1000) -> tuple[float, float]:
    """Stride-level 95% CI on F1 macro, bootstrapped from the stored predictions."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(42)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[i] = f1_score(yt[idx], yp[idx], average="macro")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


@st.cache_data(show_spinner=False)
def within_f1_with_ci(condition: str) -> tuple[float, float, float]:
    clf = load_within_results()[condition]["classifiers"]["rf"]
    lo, hi = _stride_CI_from_y(clf["y_true"], clf["y_pred"])
    return float(clf["f1_macro"]), lo, hi


# ── Input-mode helpers ───────────────────────────────────────────────────────

def _pick_subject_stride(df: pd.DataFrame, condition: str, key: str) -> pd.DataFrame:
    """Dropdown → subject → stride index slider. Returns a 1-row frame."""
    partition = load_partition()
    control_a = partition["control_A"]
    pool = df[(df["condition"] == condition) | (df["subject_id"].isin(control_a))]
    subjects = sorted(pool["subject_id"].unique())
    default = next((s for s in subjects if s.startswith(("park", "hunt", "als"))), subjects[0])
    subj = st.selectbox(
        "Subject",
        subjects,
        index=subjects.index(default),
        key=f"{key}_subj",
        help=f"Choose a subject from the {COND_SHORT[condition]} cohort (disease or Control A).",
    )
    strides = pool[pool["subject_id"] == subj].reset_index(drop=True)
    idx = st.slider(
        "Stride index",
        min_value=0,
        max_value=len(strides) - 1,
        value=0,
        key=f"{key}_idx",
    )
    st.caption(f"{len(strides)} strides available for **{subj}** (label={int(strides['label'].iloc[0])}).")
    return strides.iloc[[idx]].reset_index(drop=True)


def _slider_inputs(condition: str, key: str) -> pd.DataFrame:
    """14 feature sliders with cohort-median defaults, [p1, p99] ranges."""
    stats = cohort_feature_stats()[condition]
    cols = st.columns(2)
    values: dict[str, float] = {}
    for i, feat in enumerate(ALL_FEATURE_COLS):
        lo, med, hi = float(stats.loc[feat, "p1"]), float(stats.loc[feat, "median"]), float(stats.loc[feat, "p99"])
        if hi - lo < 1e-6:
            hi = lo + 1e-3
        step = max((hi - lo) / 200.0, 1e-6)
        with cols[i % 2]:
            values[feat] = st.slider(
                feat,
                min_value=float(round(lo, 4)),
                max_value=float(round(hi, 4)),
                value=float(round(med, 4)),
                step=float(round(step, 6)),
                help=FEATURE_HELP.get(feat, feat),
                key=f"{key}_{feat}",
            )
    row = {**values, "subject_id": "manual_input", "condition": condition, "label": -1}
    return pd.DataFrame([row])


def _csv_upload(key: str) -> pd.DataFrame | None:
    """Accept a CSV with the 14 engineered feature columns (multi-stride OK)."""
    up = st.file_uploader(
        "Upload a pre-engineered feature CSV (14 columns: "
        + ", ".join(ALL_FEATURE_COLS) + ")",
        type=["csv"],
        key=f"{key}_csv",
    )
    if up is None:
        st.info("Upload a CSV or switch to a different input mode.")
        return None
    try:
        data = pd.read_csv(up)
    except Exception as err:
        st.error(f"Could not parse CSV: {err}")
        return None
    missing = [c for c in ALL_FEATURE_COLS if c not in data.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None
    st.success(f"Loaded **{len(data)}** strides from `{up.name}`.")
    return data[list(ALL_FEATURE_COLS)].copy()


def collect_input(condition: str, key: str) -> pd.DataFrame | None:
    """
    Unified input-mode selector: subject / sliders / CSV.
    Returns a DataFrame with at least the 14 feature columns, or None if
    the user hasn't supplied valid input yet.
    """
    mode = st.radio(
        "Input mode",
        ["Pick an existing stride", "Sliders (manual entry)", "Upload feature CSV"],
        horizontal=True,
        key=f"{key}_mode",
    )
    df = load_features()
    if mode == "Pick an existing stride":
        return _pick_subject_stride(df, condition, key)
    if mode == "Sliders (manual entry)":
        return _slider_inputs(condition, key)
    return _csv_upload(key)


# ── Rendering helpers ────────────────────────────────────────────────────────

def render_prediction_card(condition: str, X: np.ndarray, title: str = "Prediction"):
    pipe = fit_rf(condition)
    proba = pipe.predict_proba(X)
    if len(X) == 1:
        p_disease = float(proba[0, 1])
        pred = int(pipe.predict(X)[0])
        label = "Disease" if pred == 1 else "Healthy control"
        col_a, col_b = st.columns(2)
        col_a.metric(f"{title} — predicted class", label)
        col_b.metric("P(disease)", f"{p_disease:.3f}")
    else:
        p_disease = proba[:, 1]
        preds = pipe.predict(X)
        frac = float(np.mean(preds == 1))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"{title} — strides", f"{len(X)}")
        col_b.metric("mean P(disease)", f"{p_disease.mean():.3f}")
        col_c.metric("fraction predicted Disease", f"{frac:.2%}")
    return proba


def render_shap_waterfall(condition: str, x_row: np.ndarray):
    """Horizontal bar chart of the top-5 SHAP contributors for a single stride."""
    explainer = tree_explainer(condition)
    raw = explainer.shap_values(x_row)

    # shap returns different shapes depending on version: list[(n,14),(n,14)] for
    # older APIs, or a single (n,14,2) ndarray for newer ones. Normalise to
    # the class-1 contributions.
    if isinstance(raw, list):
        vals = np.asarray(raw[1])[0]
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            vals = arr[0, :, 1]
        else:
            vals = arr[0]

    order = np.argsort(np.abs(vals))[::-1][:5]
    top_feats = [ALL_FEATURE_COLS[i] for i in order]
    top_vals = [float(vals[i]) for i in order]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in top_vals]

    fig = go.Figure(
        go.Bar(
            x=top_vals,
            y=top_feats,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in top_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="SHAP top-5 contributors (positive ⇒ pushes prediction toward Disease)",
        xaxis_title="SHAP value (class 1)",
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pca_scatter(x_row: np.ndarray, stride_condition: str | None = None):
    """PC1/PC2 scatter of all 14,753 strides, with the user's stride highlighted."""
    scaler, pca, X_pca, meta = fit_pca_projection()
    user_pc = pca.transform(scaler.transform(x_row))[0]

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "condition": meta["condition"],
        "group": np.where(meta["label"] == 1, "Disease", "Control"),
    })
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="condition", opacity=0.25,
        color_discrete_map={"pd": "#1f77b4", "hd": "#2ca02c", "als": "#ff7f0e", "control": "#7f7f7f"},
        height=420,
    )
    fig.add_trace(go.Scatter(
        x=[user_pc[0]], y=[user_pc[1]],
        mode="markers", marker=dict(size=18, color="red", symbol="star", line=dict(color="black", width=2)),
        name="Your stride",
    ))
    fig.update_layout(
        title=f"PCA projection — cohort distribution vs your stride" + (f" ({COND_SHORT.get(stride_condition, '')})" if stride_condition else ""),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cross_banner(source: str, target: str):
    """ΔF1 degradation banner pulled from cross_condition_results.json."""
    within = load_within_results()
    cross = load_cross_results()
    within_f1 = float(within[source]["classifiers"]["rf"]["f1_macro"])
    key = f"{source}_to_{target}"
    if key not in cross:
        st.warning(f"No stored cross-condition run for {source} → {target}.")
        return
    pair = cross[key]["classifiers"]["rf"]
    cross_f1 = float(pair["f1_macro"])
    delta = cross_f1 - within_f1
    ci_lo = float(pair["f1_macro_ci_lower"])
    ci_hi = float(pair["f1_macro_ci_upper"])
    pval = float(pair["permutation_p_value"])
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(f"Within-{COND_SHORT[source]} F1", f"{within_f1:.3f}")
    col_b.metric(f"{COND_SHORT[source]} → {COND_SHORT[target]} F1", f"{cross_f1:.3f}", delta=f"{delta:+.3f}")
    col_c.metric("Stride 95% CI", f"[{ci_lo:.2f}, {ci_hi:.2f}]")
    col_d.metric("Permutation p-value", f"{pval:.3f}")
    if delta < -0.10:
        st.warning(
            f"This model **loses {abs(delta):.0%} macro-F1** when transferred from "
            f"{COND_SHORT[source]} to {COND_SHORT[target]}. Treat any individual "
            f"prediction accordingly."
        )


def render_noise_curve(condition: str, x_row: np.ndarray, sigma_max: float, n_mc: int = 100):
    """
    Live P(disease) vs σ, Monte-Carlo over n_mc perturbations per σ grid point,
    overlaid with the reference F1-vs-σ curve from noise_robustness.json.
    """
    pipe = fit_rf(condition)
    rng = np.random.default_rng(42)
    scaler, _, _, _ = fit_pca_projection()
    feat_std = np.sqrt(scaler.var_)  # per-feature std on the full 14,753-stride pool

    sigmas = np.linspace(0.0, max(sigma_max, 0.05), 11)
    mean_p = np.empty_like(sigmas)
    std_p = np.empty_like(sigmas)
    for i, s in enumerate(sigmas):
        if s == 0:
            p = pipe.predict_proba(x_row)[0, 1]
            mean_p[i], std_p[i] = p, 0.0
            continue
        noise = rng.normal(scale=s * feat_std, size=(n_mc, len(ALL_FEATURE_COLS)))
        X_noisy = x_row + noise
        probs = pipe.predict_proba(X_noisy)[:, 1]
        mean_p[i], std_p[i] = float(probs.mean()), float(probs.std())

    ref = load_noise_results().get("within", {}).get(condition, {}).get("rf", {})
    ref_sigma = sorted(float(k) for k in ref)
    ref_mean = [float(np.mean(ref[f"{s:g}"])) if f"{s:g}" in ref else float(np.mean(ref[str(s)])) for s in ref_sigma]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Your stride — P(disease) vs σ (mean ± 1 std over MC)",
                        f"Reference cohort F1 vs σ ({COND_SHORT[condition]} RF)"),
    )
    fig.add_trace(go.Scatter(
        x=sigmas, y=mean_p + std_p, mode="lines", line=dict(width=0), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sigmas, y=mean_p - std_p, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.2)", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sigmas, y=mean_p, mode="lines+markers", name="P(disease)",
        line=dict(color="#1f77b4", width=2),
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", row=1, col=1)
    fig.update_xaxes(title="σ (feature-std multiplier)", row=1, col=1)
    fig.update_yaxes(title="P(disease)", range=[0, 1], row=1, col=1)

    if ref_sigma:
        fig.add_trace(go.Scatter(
            x=ref_sigma, y=ref_mean, mode="lines+markers", name="Cohort F1",
            line=dict(color="#2ca02c", width=2),
        ), row=1, col=2)
        fig.update_xaxes(title="σ", row=1, col=2)
        fig.update_yaxes(title="F1 macro", range=[0, 1], row=1, col=2)
    else:
        fig.add_annotation(text="noise_robustness.json unavailable", xref="x2", yref="y2",
                           x=0.5, y=0.5, showarrow=False, row=1, col=2)

    fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────

def tab_within():
    st.subheader("Within-condition inference")
    st.markdown(
        "Select a cohort (PD / HD / ALS) and supply a stride. The cohort's "
        "Step-2 Random Forest (trained on disease strides + Control Group A) "
        "predicts disease vs control, and SHAP shows which of the 14 gait "
        "features drove the decision."
    )

    condition = st.radio("Cohort", list(CONDITIONS), horizontal=True,
                         format_func=lambda c: COND_SHORT[c], key="within_cond")
    f1, lo, hi = within_f1_with_ci(condition)
    st.caption(f"Cohort LOSO F1 (RF): **{f1:.3f}**  —  stride-level 95% CI **[{lo:.3f}, {hi:.3f}]**")

    inp = collect_input(condition, key="within")
    if inp is None:
        return
    X = inp[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float64)

    render_prediction_card(condition, X)

    st.divider()
    # Use the first stride for the per-stride explanations.
    st.markdown("#### Local explanation (first stride)")
    render_shap_waterfall(condition, X[[0]])

    st.markdown("#### PCA placement in cohort feature space")
    render_pca_scatter(X[[0]], stride_condition=condition)

    if len(X) > 1:
        st.markdown("#### Per-stride predictions")
        pipe = fit_rf(condition)
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)[:, 1]
        st.dataframe(pd.DataFrame({
            "stride": np.arange(len(X)),
            "P(disease)": np.round(probs, 4),
            "predicted": np.where(preds == 1, "Disease", "Control"),
        }), hide_index=True, use_container_width=True)


def tab_cross():
    st.subheader("Cross-condition transfer")
    st.markdown(
        "Pick a **source** cohort to train on and a **target** cohort to "
        "predict. This reproduces Step 3's zero-shot transfer protocol: the "
        "RF fits on source + Control A strides and predicts on a stride from "
        "the target cohort. The expected F1 drop (ΔF1) is pulled from the "
        "stored cross-condition sweep."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        source = st.radio("Train on (source)", list(CONDITIONS), horizontal=True,
                          format_func=lambda c: COND_SHORT[c], key="cross_src")
    with col_b:
        target_opts = [c for c in CONDITIONS if c != source]
        target = st.radio("Predict on (target)", target_opts, horizontal=True,
                          format_func=lambda c: COND_SHORT[c], key="cross_tgt")

    render_cross_banner(source, target)

    st.markdown(f"#### Input stride ({COND_SHORT[target]} feature space)")
    inp = collect_input(target, key="cross")
    if inp is None:
        return
    X = inp[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float64)

    render_prediction_card(source, X, title=f"{COND_SHORT[source]}-RF on {COND_SHORT[target]} stride")

    st.divider()
    st.markdown(f"#### Local explanation via {COND_SHORT[source]} RF")
    render_shap_waterfall(source, X[[0]])

    st.markdown("#### PCA placement (all cohorts)")
    render_pca_scatter(X[[0]], stride_condition=target)


def tab_noise():
    st.subheader("Noise robustness")
    st.markdown(
        "Explore how additive Gaussian noise on the 14 gait features degrades "
        "the cohort RF's confidence for a chosen stride. Per-feature noise "
        "scale is σ × pooled feature std (same convention as the Step-5 "
        "sweep). The right panel shows the **reference** F1 curve across "
        "the full cohort from `noise_robustness.json`."
    )

    condition = st.radio("Cohort", list(CONDITIONS), horizontal=True,
                         format_func=lambda c: COND_SHORT[c], key="noise_cond")
    inp = collect_input(condition, key="noise")
    if inp is None:
        return
    X = inp[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float64)

    sigma_max = st.slider(
        "Maximum σ (feature-std multiplier)",
        min_value=0.05, max_value=1.0, value=0.5, step=0.05, key="noise_sigma_max",
        help="Sweep from σ = 0 up to this value; P(disease) is averaged over 100 MC draws per σ.",
    )

    render_prediction_card(condition, X[[0]], title="Clean-stride baseline")
    render_noise_curve(condition, X[[0]], sigma_max=sigma_max)


# ── Main ─────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("### Gait Transfer — NDD Demo")
        st.caption(
            "Interactive inference for the paper "
            "*Cross-condition transferability of stride-based classifiers "
            "for Parkinson's, Huntington's, and ALS*."
        )
        st.divider()
        st.markdown("**Models cached once per session**")
        st.caption(
            "Each cohort RF is re-fit from the Step-2 modal hyper-parameters "
            "(stored in `experiments/results/{cond}_results.json`) on first "
            "load, then cached — no re-training or re-processing runs on "
            "subsequent interactions."
        )
        st.divider()
        within = load_within_results()
        st.markdown("**Within-condition F1 (RF)**")
        for c in CONDITIONS:
            st.write(f"- {COND_SHORT[c]}: `{within[c]['classifiers']['rf']['f1_macro']:.3f}`")
        st.divider()
        st.caption(
            "⚠ **Research demo only** — predictions are based on a small "
            "(~63-subject) benchmark cohort and are not validated for "
            "clinical use."
        )


def main():
    st.set_page_config(
        page_title="Gait Transfer NDD Demo",
        layout="wide",
        page_icon="🦶" if False else None,
    )
    st.title("Gait-based NDD classifier — interactive inference")
    st.markdown(
        "This app demonstrates the Step-2 within-condition classifiers and "
        "the Step-3 zero-shot cross-condition transfer of this project, plus "
        "the Step-5 noise-robustness behaviour, all driven by the published "
        "results in `experiments/results/`."
    )
    _sidebar()

    tabs = st.tabs([
        "1. Within-condition inference",
        "2. Cross-condition transfer",
        "3. Noise robustness",
    ])
    with tabs[0]:
        tab_within()
    with tabs[1]:
        tab_cross()
    with tabs[2]:
        tab_noise()

    st.divider()
    st.caption(
        "Models: Random Forest pipelines (SMOTE → RF) from `src/train.build_pipeline`, "
        "fit in-process from modal hyper-parameters in "
        "`experiments/results/{cond}_results.json`. SHAP: `shap.TreeExplainer` on the "
        "underlying RF classifier. PCA: `StandardScaler → PCA(n_components=2)` fit on "
        "all 14,753 strides (identical to Step 6)."
    )


if __name__ == "__main__":
    main()
