# Project Progress Log

## 2026-04-03 — Shahmeer — Within-condition training: Phase B (src/train.py complete, runner script)
- Implemented `get_classifier_configs()` with all 7 classifiers (RF, KNN, SVM, DT, QDA, XGBoost, LightGBM) and full hyperparameter grids matching GUIDELINE.md
- Implemented `run_within_condition()` which constructs the binary training pool, runs nested LOSO-CV for all classifiers, and saves results to `experiments/results/{condition}_results.json`
- Wrote `scripts/run_step2_training.py` — standalone runner for all three conditions in sequence, compatible with `nohup` and `tail -f` monitoring via `flush=True` throughout
- Smoke test: 37/37 checks pass (classifier presence, n_jobs, probability, class_weight, QDA type, grid sizes, clf__ key prefixes)

## 2026-04-03 — Shahmeer — Within-condition training: Phase A (src/train.py, three core functions)
- Implemented `build_pipeline()`, `run_nested_loso()`, `get_modal_params()` in `src/train.py`
- `build_pipeline()` applies StandardScaler before SMOTE for SVM and KNN; omits scaler for tree-based methods and QDA
- `run_nested_loso()` uses outer LOSO for evaluation and inner GridSearchCV with a fresh LOSO on the training fold only; F1 macro computed over the full concatenated prediction vector (not per-fold, which is undefined for single-class test sets)
- `get_modal_params()` uses fold_best_scores as tiebreaker when two combinations are equally frequent
- Verified on PD pool (23 subjects, 5738 strides, 1.78:1 ratio): aggregate F1 macro = 0.7784 with reduced 4-combination RF grid. Wall time 463s.

## 2026-04-02 — Shahmeer — Step 1: Preprocessing notebook (01_preprocessing.ipynb)
- Created notebook that imports `build_feature_matrix()` from `src/features.py` and runs the full Step 1 pipeline
- Cells display: feature matrix shape (14753x17), per-condition subject/stride counts, asymmetry_index stats, cv_stride per-subject stats, first 5 rows, file existence checks, CSV reload verification, null count confirmation
- Verification summary table confirms all Step 1 checkpoints pass

## 2026-04-02 — Shahmeer — Step 1: Data exploration notebook (00_data_exploration.ipynb)
- Created notebook documenting raw dataset inspection performed during pre-implementation audit
- Cells display: file counts per condition (64 total), raw file structure (13 tab-separated columns), total raw stride count (15,160) with per-subject breakdown, pause event distribution (276 rows with stride > 3.0s), hunt20 frozen sensor anomaly (all 238 rows), als5/als7 double_support_pct > 100 anomaly (131 rows)
- Summary records the two data quality issues that motivated the dual-filter design

## 2026-04-02 — Shahmeer — Step 1 Phase B: Feature engineering and full pipeline orchestration
- Implemented `compute_asymmetry_index()`, `compute_cv_stride()`, `build_feature_matrix()` in `src/features.py`
- `build_feature_matrix()` orchestrates the full Step 1 pipeline and writes both output artifacts to `data/processed/`
- Verified: final shape=(14753, 17), null values in feature columns=0, gait_features.csv written, control_partition.json written. asymmetry_index range [0.000, 0.929], mean=0.037. cv_stride per-subject range [0.018, 0.240], mean=0.076. 63 unique subjects (64 minus hunt20).

## 2026-04-02 — Shahmeer — Step 1 Phase A: Data loading, filtering, labelling, and control partitioning
- Implemented `load_raw_data()`, `filter_pause_events()`, `assign_labels()`, `partition_controls()` in `src/preprocessing.py`
- Pre-implementation audit identified two discrepancies from GUIDELINE.md: (1) the stride-only filter removes 276 rows (not 484), and (2) an additional 131 rows have physically impossible double_support_pct > 100 due to sensor malfunction. Both GUIDELINE.md and CLAUDE.md updated accordingly. hunt20 entirely removed by stride filter (right-foot sensor frozen), leaving 19 usable HD subjects.
- Verified: raw=15,160, removed by stride filter=276, removed by pct filter=131, clean=14,753. Subject counts: PD=15, HD=19 usable, ALS=13, Control=16. hunt20 rows in clean data=0. control_partition.json written.

## 2026-03-30 — Shahmeer
- Initialized repository and directory structure
- Setup independent google drive for relevant papers and GUIDELINE.md
- Downloaded dataset