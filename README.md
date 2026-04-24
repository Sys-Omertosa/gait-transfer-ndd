# Gait-based NDD classifier — cross-condition transfer study

Stride-level classifiers for Parkinson's (PD), Huntington's (HD), and ALS built
on the PhysioNet GAITNDD dataset, with a focus on **zero-shot cross-condition
transfer** and a matching interactive demo.

The repository follows the 8-step workflow documented in
[CONTRIBUTING.md](CONTRIBUTING.md):

| Step | Deliverable                                                       |
|------|-------------------------------------------------------------------|
| 1    | Data ingestion, cleaning, feature engineering (`src/features.py`) |
| 2    | Within-condition LOSO-CV training (`src/train.py` / notebook 02)  |
| 3    | Cross-condition zero-shot transfer (notebook 03)                  |
| 4    | SHAP-based transfer-failure diagnosis (notebook 04, `src/explain.py`) |
| 5    | Noise and sensitivity robustness (`src/robustness.py`)            |
| 6    | PCA + K-Means unsupervised analysis (notebook 06)                 |
| 7    | Publication-quality figures (notebook 07)                         |
| 8    | **Streamlit demo for interactive inference (`app/app.py`)**       |

## Quick start

```bash
# 1. Create a Python 3.12 virtualenv and install pinned deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt   # or requirements.txt for the full env

# 2. Reproduce the preprocessed feature matrix (one-off, ~10 s)
python -c "from src.features import build_feature_matrix; build_feature_matrix('data/raw', 'data/processed')"
```

The heavy training / SHAP / noise sweeps live under
[scripts/training/](scripts/training/) and write JSON artefacts into
[experiments/results/](experiments/results/). Those JSONs are the canonical
inputs for notebooks 02 – 07 and for the demo.

## Run the demo

The Step-8 demo at [app/app.py](app/app.py) is a single-file Streamlit app that
makes every result in [experiments/results/](experiments/results/) interactive:

```bash
streamlit run app/app.py
```

On first load the app fits one Random Forest per cohort from the modal
hyper-parameters stored in `experiments/results/{cond}_results.json` (~3 s
total) and caches the pipelines, SHAP `TreeExplainer`s, and the Step-6 PCA
projection via `@st.cache_resource`. No training sweep, grid search, or raw
`.ts` re-processing is triggered by any user interaction.

The app exposes three tabs:

1. **Within-condition inference** — pick PD / HD / ALS, supply a stride via
   one of three input modes, and see the predicted class, P(disease), the
   cohort's LOSO F1 with a stride-level 95 % CI, a SHAP top-5 waterfall
   showing which of the 14 gait features drove the decision, and the stride's
   placement in the cohort's PC1/PC2 space.
2. **Cross-condition transfer** — pick a source cohort to train on and a
   target cohort to predict. A banner surfaces the stored within-F1,
   cross-F1, ΔF1, stride 95 % CI, and permutation p-value for that pair from
   `cross_condition_results.json`, with an automatic calibration warning when
   the transfer drop exceeds 10 points of macro-F1.
3. **Noise robustness** — σ slider (feature-std multiplier) sweeps Gaussian
   noise over a chosen stride with 100 Monte-Carlo draws per σ, overlaid with
   the reference F1-vs-σ curve for the cohort pulled from
   `noise_robustness.json`.

Every tab supports three input modes — pick an existing subject + stride,
manual sliders (pre-filled with cohort median, ranges clipped to [p1, p99]),
or upload a pre-engineered 14-column CSV (multi-stride supported).

The app is a **research demo only** and is not validated for clinical use;
see the disclaimer in the sidebar.

## Layout

```
app/                   Streamlit demo (Step 8)
data/processed/        gait_features.csv, control_partition.json
experiments/results/   Step 2–5 JSON artefacts (inputs to all notebooks + app)
notebooks/             Steps 0–7 Jupyter notebooks (+ PDFs under notebooks/pdf/)
report/figures/        Generated figures (Steps 2–6)
report/figures/paper/  Publication-quality IEEE-column figures (Step 7)
report/tables/         Master results table (CSV + LaTeX, Step 7)
scripts/training/      Local and Modal runners for Steps 2–5
src/                   Library modules (preprocessing, features, train, explain, robustness)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for member attributions and dependency
conventions, and [PROGRESS.md](PROGRESS.md) for a chronological log of every
step's implementation.
