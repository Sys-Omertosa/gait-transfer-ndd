# Gait Transfer Across Neurological Conditions

This repository contains the full research codebase for a gait-timing study on
Parkinson's disease, Huntington's disease, and amyotrophic lateral sclerosis
using the PhysioNet Neurodegenerative Disease Gait database. The project is not
just an interactive demo. It includes the end-to-end experimental pipeline for
data cleaning, feature engineering, within-condition benchmarking, zero-shot
cross-condition transfer, SHAP-based transfer-failure diagnosis, robustness
analysis, unsupervised geometric analysis, paper generation, and a Streamlit
inference app.

The central question is simple but clinically important: if a model learns to
separate pathological gait from healthy gait in one neurological condition, does
that timing knowledge transfer to a different condition without retraining?

The repository is intended to be public-facing and reproducible. Raw GAITNDD
source files are not committed, private planning material remains outside the
public code path, and the current authoritative experiment outputs live under
the `v2/` artifact directories described below.

## Project Scope

The repository implements an eight-step workflow:

1. Raw gait timing ingestion, artifact filtering, label assignment, and control
   partitioning.
2. Within-condition disease-versus-control training under subject-level
   leave-one-subject-out cross-validation.
3. Zero-shot cross-condition transfer across all six directed disease pairs.
4. SHAP-based feature attribution analysis and the `delta_j` importance-shift
   diagnostic.
5. Robustness analysis, including Gaussian noise, structured corruption,
   feature sensitivity, subject sensitivity, and conformal diagnostics.
6. PCA-based geometry analysis and K-means clustering.
7. Publication-quality figures, summary tables, and LaTeX paper artifacts.
8. A Streamlit application for interactive inspection of stored predictions and
   optional live explanation mode.

## What Is In This Repository

### Core research modules

- `src/preprocessing.py`
  - loads the raw `.ts` files
  - removes artifact rows using the project filters
  - creates the fixed `Control A` and `Control B` partition
- `src/features.py`
  - builds the processed feature matrix
  - computes the 17-feature set used in the current experiment path
  - includes `asymmetry_index`, `stride_asymmetry_signed`, `cv_stride`,
    `cv_swing`, and `dfa_alpha_stride`
- `src/train.py`
  - defines classifier configurations
  - runs nested subject-level LOSO-CV
  - handles SMOTE ablation and modal-parameter selection
  - evaluates zero-shot cross-condition transfer
- `src/explain.py`
  - computes SHAP values
  - assigns the appropriate explainer per classifier family
  - implements the `delta_j` transfer-failure metric
- `src/robustness.py`
  - evaluates Gaussian noise robustness
  - computes feature and subject sensitivity
  - runs structured corruption sweeps and conformal diagnostics

### Notebooks

The notebooks are the narrative layer of the project. Reusable logic lives in
`src/`, while notebooks inspect, summarize, and visualize outputs.

- `notebooks/00_data_exploration.ipynb`
- `notebooks/01_preprocessing.ipynb`
- `notebooks/02_within_condition.ipynb`
- `notebooks/03_cross_condition.ipynb`
- `notebooks/04_shap_diagnosis.ipynb`
- `notebooks/05_noise_robustness.ipynb`
- `notebooks/06_pca_kmeans.ipynb`
- `notebooks/07_final_figures.ipynb`

### Result artifacts

The repository contains versioned artifacts under:

- `data/processed/`
  - `v1/gait_features_v1.csv`
  - `v2/gait_features_v2.csv`
  - `control_partition.json`
- `experiments/results/`
  - `v1/` and `v2/` JSON outputs
- `experiments/models/`
  - persisted per-condition trained models for `v1` and `v2`
- `report/figures/`
  - generated PDF and PNG figures
- `report/tables/`
  - summary tables exported from the final notebooks

At the moment, the primary and authoritative experimental path in the codebase
is `v2`. That includes:

- `data/processed/v2/`
- `experiments/results/v2/`
- `experiments/models/v2/`
- `report/figures/v2/`
- `report/tables/v2/`

Earlier `v1` artifacts are preserved for provenance and comparison. Additional
improvements, along with mitigation of several currently known limitations, are
actively underway.

## Data and Experimental Design

The project uses the PhysioNet Neurodegenerative Disease Gait database. Raw
files are expected under:

```text
data/raw/gait-in-neurodegenerative-disease-database-1.0.0/
```

These raw files are not committed to the repository. The processed outputs are
constructed from them.

The current experiment path uses:

- 63 usable subjects after exclusions
- three disease cohorts: PD, HD, ALS
- a fixed disjoint healthy-control split:
  - `Control A` for source and within-condition training pools
  - `Control B` for transfer evaluation pools
- a 17-feature timing-based representation
- subject-level LOSO-CV as the primary evaluation protocol

This design is intentional. It prevents leakage from having the same subject
appear in both training and evaluation folds, and it makes the transfer results
more defensible than stride-level random splits.

## Repository Layout

```text
gait-transfer-ndd/
├── README.md
├── requirements.txt
├── requirements-core.txt
├── CONTRIBUTING.md
├── PROGRESS.md
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
│   ├── models/
│   └── results/
├── notebooks/
├── report/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── training/
│   └── verification/
├── src/
└── tests/
```

## Environment Setup

The repository currently uses a plain pip environment, not a `pyproject.toml`
workflow.

### Full environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Core research environment

If you do not need every notebook-side dependency, use the smaller direct
dependency list:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

## Rebuilding the Processed Feature Matrix

The full Step 1 feature-matrix builder is implemented in
`src/features.py::build_feature_matrix`.

From the repository root:

```bash
source venv/bin/activate
python -c "from src.features import build_feature_matrix; build_feature_matrix()"
```

This will:

- load all raw `.ts` files
- apply the artifact filters
- assign binary labels
- save the fixed control partition
- compute the engineered features
- write the processed CSV to `data/processed/v2/gait_features_v2.csv`

## Reproducing the Main Experimental Stages

The shortest reproducibility path for the current paper-facing results is:

1. rebuild or verify `data/processed/v2/gait_features_v2.csv`
2. run the Step 2 through Step 5 local runners, or use the committed
   authoritative `experiments/results/v2/` outputs
3. open notebooks `03` through `07` for analysis, figure generation, and
   manuscript-facing exports

If you are reproducing the published repository state rather than rerunning the
entire experiment chain, start from the committed `v2` artifacts.

### Step 2: within-condition baselines

```bash
source venv/bin/activate
python scripts/training/run_within_condition_local.py
```

### Step 3: cross-condition transfer

```bash
source venv/bin/activate
python scripts/training/run_cross_condition_local.py
```

### Step 4: SHAP and `delta_j`

```bash
source venv/bin/activate
python scripts/training/run_shap_local.py
```

### Step 5: robustness and sensitivity

```bash
source venv/bin/activate
python scripts/training/run_noise_robustness_local.py
```

### Modal-backed training paths

The repository also includes Modal runners for heavier sweeps:

- `scripts/training/run_within_condition_modal.py`
- `scripts/training/run_cross_condition_modal.py`
- `scripts/training/run_shap_modal.py`
- `scripts/training/run_noise_robustness_modal.py`

These are useful when local runtime becomes restrictive, especially for SHAP
and broader robustness evaluation.

### Notebook-driven analysis and figure generation

After the core result JSONs exist, the remaining analysis is primarily notebook
driven:

- `notebooks/06_pca_kmeans.ipynb` for PCA and clustering
- `notebooks/07_final_figures.ipynb` for final figures and summary tables

## Streamlit Application

The app lives at `app/app.py`.

Run it with:

```bash
source venv/bin/activate
streamlit run app/app.py
```

The app is designed around stored research artifacts first. In its default
mode, it reads the processed feature matrix and the saved `v2` JSON outputs,
then exposes three interactive views:

- within-condition inference
- cross-condition transfer inspection
- noise robustness inspection

It also supports an optional explain mode, which performs one-time cached
Random Forest fitting per cohort for live SHAP and live noise simulation.

This app is a research-facing interface, not a clinical product.

## Figures, Tables, and Paper Assets

The paper source currently lives in:

- `report/main.tex`
- `report/references.bib`

Generated visual assets are organized by version under:

- `report/figures/v1/`
- `report/figures/v2/`

Summary tables exported from the final notebooks are stored under:

- `report/tables/v2/`

These reporting exports are different from the processed gait-feature CSVs.
They are manuscript-facing summaries, not training inputs.

## Current Project Status

The repository currently reflects a stable `v2` research path and an active
improvement path in parallel.

- The `v2` folders contain the current authoritative results used by the paper,
  the app, and the exported figures and tables.
- Several limitations identified during the study, including feature-space
  redundancy and additional robustness refinements, are being addressed in
  ongoing follow-up work.
- Placeholder `v3/` directories exist as staging space for future iterations,
  not as a completed result line.

## Important Implementation Details

Several design choices matter for interpreting the repository correctly:

- Subject-level LOSO-CV is the default evaluation design.
- SMOTE, when enabled, is applied inside the training pipeline only.
- In the within-condition pools, control is the minority class, so SMOTE
  augments healthy control strides rather than disease strides.
- The app prefers stored predictions and stored result JSONs rather than
  rerunning the experimental pipeline on user interaction.
- The `v1`, `v2`, and `v3` directories are versioned experiment paths and
  artifact namespaces, not separate projects.

## Current Limitations

This repository reflects an active research workflow, so a few caveats are
worth stating clearly:

- Some reporting artifacts are versioned exports produced by notebooks and may
  use internal naming conventions that are more useful for the team than for
  outside readers.
- `context/` is intentionally private and gitignored because it contains working
  notes, papers, and planning documents rather than public-facing code.
- The repository includes some large but intentional research artifacts, most
  notably processed feature matrices and cross-condition JSON result files,
  because they are part of the reproducibility story for the current study.
- The codebase is research-grade and reproducible, but it is not packaged as a
  general-purpose Python library.
- The Streamlit app is suitable for exploratory inspection, not deployment in a
  clinical environment.

## Suggested Reading Order

If you are new to the repository, this sequence works well:

1. Read `src/preprocessing.py` and `src/features.py`.
2. Inspect `src/train.py` for the main training and transfer logic.
3. Read `src/explain.py` for SHAP and `delta_j`.
4. Read `src/robustness.py` for Step 5 extensions.
5. Open notebooks `02` through `07` in order.
6. Read `report/main.tex` for the current manuscript draft.

## Collaboration Notes

The repository has been developed collaboratively. The project structure,
progress log, and work split history are documented in:

- `CONTRIBUTING.md`
- `PROGRESS.md`

## Disclaimer

This repository is for research and educational use. It should not be used for
clinical diagnosis, treatment decisions, or patient-facing deployment without
substantial additional validation.
