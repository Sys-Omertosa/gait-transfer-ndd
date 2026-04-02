
## 1. Repository Structure
```
gait-transfer-ndd/
├── README.md                          # Public project description, reproduction instructions
├── .gitignore
├── requirements.txt                   # Full pinned environment (pip freeze output)
├── requirements-core.txt              # 15 direct dependencies with versions (human-readable)
├── PROGRESS.md                        # Shared chronological log — updated after every step
│
├── data/
│   ├── raw/                           # GAITNDD .ts/.hea/.let/.rit files — gitignored
│   └── processed/                     # Cleaned feature CSVs after preprocessing — gitignored
│
├── context/                           # Private reference — gitignored entirely
│   ├── GUIDELINE.md                   # This document
│   └── papers/                        # PDFs of all cited papers
│       └── SOTA Papers/               # The 4 primary SOTA comparison papers
│
├── src/                               # Reusable Python modules imported by notebooks
│   ├── preprocessing.py               # load_and_clean(), engineer_features(), partition_controls()
│   ├── features.py                    # asymmetry_index(), cv_stride()
│   ├── train.py                       # run_loso_cv(), smote_pipeline(), grid_search_loso()
│   ├── evaluate.py                    # build_degradation_table(), compute_metrics()
│   └── explain.py                     # compute_shap_values(), compute_delta_j()
│
├── notebooks/                         # One notebook per pipeline stage — narrative + outputs
│   ├── 00_data_exploration.ipynb      # Raw data inspection, distributions, subject counts
│   ├── 01_preprocessing.ipynb         # Pause filtering, feature engineering, verification
│   ├── 02_within_condition.ipynb      # LOSO-CV baselines for all 6 classifiers
│   ├── 03_cross_condition.ipynb       # Transfer experiments, degradation table (primary result)
│   ├── 04_shap_analysis.ipynb         # δj metric, per-classifier SHAP comparison
│   ├── 05_noise_robustness.ipynb      # Gaussian noise injection analysis
│   ├── 06_pca_kmeans.ipynb            # PCA geometric visualisation, K-Means cluster discovery
│   └── 07_final_figures.ipynb         # Publication-quality plots for the IEEE paper
│
├── experiments/                       # All outputs from runs
│   ├── models/                        # Saved .pkl model files — gitignored
│   ├── results/                       # Metric CSVs, degradation table — commit finals only
│   └── figures/                       # Plots — commit finals only
│
├── app/
│   └── app.py                         # Streamlit deployment demo (Step 8)
│
├── report/
│   ├── main.tex                       # IEEE paper (LaTeX)
│   ├── references.bib                 # BibTeX references
│   └── figures/                       # Copies of final figures for the paper
│
└── tests/
    └── test_preprocessing.py          # Sanity checks on key pipeline functions
```

**Key design principle:** Notebooks are the narrative layer — they import from `src/` and display results. All reusable logic lives in `src/` so it can be shared across notebooks without duplication. The processed feature matrix committed after Step 1 is the coordination artifact between the group members.

---

## 2. Work Split

### Dependency order and coordination
```
Step 1 (Member 1)
    ├── Step 2 (Member 1)
    |       ├── Step 3 (Member 1)
    |       |       └── Step 4 (Member 1)
    |       ├── Step 5 (Member 2)
    |       └── Step 8 (Member 2)
    |
    ├── Step 6 (Member 2)
    └── Step 7 (Member 2)
```

**Hard rule:** Member 1 commits `data/processed/gait_features.csv` and `data/processed/control_partition.json` before Member 2 begins any work.

**Steps 6 and 7** only require the feature matrix — Member 2 can begin these
immediately after Step 1 without waiting for Step 2.

**Steps 3 and 4** cannot begin until Step 2 is complete — Step 3 requires the
modal best hyperparameters and within-condition F1 scores produced by Step 2.

**Timeline:**
```
Member 1: Step 1 → Step 2 → Step 3 → Step 4
Member 2:    ↑ Steps 6, 7 → Step 5 → Step 8
```

### Work Split Assessment:

| Step                   | Best Owner | Reason                                              |
| ---------------------- | ---------- | --------------------------------------------------- |
| Step 1                 | Member 1   | Foundational — your work, your constraint knowledge |
| Step 2:                | Member 1   | All classifier pipelines                            |
| Step 3                 | Member 1   | Primary novel contribution                          |
| Step 4                 | Member 1   | Depends on Step 3, flows naturally                  |
| Step 5                 | Member 2   | Only needs Step 2 output                            |
| Step 6                 | Member 2   | Only needs Step 1 output — can start early          |
| Step 7                 | Member 2   | Same as Step 6                                      |
| Step 8                 | Member 2   | Only needs Step 2 RF model                          |

Work assignments are subject to change. This table reflects the current state.

---

## 3. Technology Stack

**Language:** Python 3.12.x

| Library | Role |
|---|---|
| `polars` | Primary data layer — loading, filtering, feature engineering |
| `pandas` | Compatibility layer for sklearn/shap/imbalanced-learn only |
| `numpy` | Array operations, noise injection |
| `scikit-learn` | All classifiers, LOSO-CV, GridSearchCV, PCA, KMeans |
| `imbalanced-learn` | SMOTE — inside LOSO pipeline only via `imblearn.pipeline.Pipeline` |
| `xgboost` | Gradient boosting classifier 1 |
| `lightgbm` | Gradient boosting classifier 2 |
| `shap` | SHAP values, δj metric, waterfall plots |
| `plotly` | Primary visualization library — all custom charts |
| `matplotlib` | SHAP built-in plots only — do not use for custom charts |
| `streamlit` | Deployment demo (Step 8) — minimal UI |
| `scipy` | Statistical tests |
| `tqdm` | Progress bars for LOSO loops |

**Polars→pandas handoff pattern (use at sklearn/shap boundary only):**
```python
X_np = gait_clean.select(feature_cols).to_numpy()
y = gait_clean['label'].to_numpy()
groups = gait_clean['subject_id'].to_numpy()
```

---