# Project Progress Log

## 2026-04-06 — Shahmeer — Within-condition notebook: figures and PDF (notebooks/02_within_condition.ipynb)
- Refactored `notebooks/02_within_condition.ipynb` to use stored predictions from result JSON files instead of re-running LOSO-CV with fixed modal params (`eval_loso_fixed` removed entirely)
- Fixed `FIGURES_DIR` from `reports/figures/` (wrong) to `report/figures/` (correct)
- All five paper figures saved to `report/figures/`: `f1_within_condition_heatmap`, `modal_frequency_heatmap`, `cm_pd_rf`, `cm_hd_rf`, `cm_als_knn` (PDF + PNG each)
- Notebook executes top-to-bottom without errors

## 2026-04-06 — Shahmeer — Within-condition training: Modal re-run, y_true/y_pred persistence
- Updated `src/train.py` and `scripts/training/run_within_condition_modal.py` to persist `y_true` and `y_pred` as flat Python lists in each classifier's result JSON entry; added `< 1e-6` F1 round-trip assertion at write time
- Re-ran all three conditions on Modal (cpu=16, memory=16384) to produce updated `experiments/results/{pd,hd,als}_results.json` with stored prediction arrays
- LightGBM num_leaves grid confirmed {31, 63} on Modal; num_leaves=31 selected in all conditions; only change from previous run is LightGBM PD F1: 0.7211 → 0.7251
- All other classifiers bit-for-bit identical between local and Modal runs, confirming full determinism across platforms

## 2026-04-04 — Shahmeer — Within-condition training: local run, Modal run, results analysis
- Ran full within-condition training locally for all three conditions and all 7 classifiers using `scripts/run_step2_training.py`; results saved to `experiments/results/{pd,hd,als}_results.json`
- Identified nested OpenMP parallelism bug (n_jobs=-1 at both GridSearchCV and classifier level under gVisor); fixed by setting n_jobs=1 on RF, XGBoost, and LightGBM at classifier level; n_jobs=-1 retained at GridSearchCV level only; zero effect on accuracy
- Re-ran on Modal (cpu=16, memory=16384) with expanded LightGBM num_leaves grid {31, 63} to validate the num_leaves=31 constraint; confirmed num_leaves=31 selected in all three conditions, validating original grid choice
- Authoritative results (Modal run): PD best RF F1=0.7777, HD best RF F1=0.8995, ALS best KNN F1=0.8758

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