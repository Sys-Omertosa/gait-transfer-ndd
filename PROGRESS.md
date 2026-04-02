# Project Progress Log

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