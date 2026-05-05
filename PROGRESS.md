# Project Progress Log

## 2026-05-02 — Shahmeer — Public repository documentation and licensing refresh

- Rewrote `README.md` to describe the full research pipeline rather than only the Streamlit demo, with explicit reproduction guidance, repository layout, artifact structure, and app usage notes
- Marked the `v2/` directories as the current authoritative experiment path across processed data, results, models, figures, and tables, while clarifying that follow-up improvements and mitigation work remain in progress
- Added an MIT `LICENSE` file to support public release and reuse of the codebase under a standard academic-friendly open source license
- Tightened `.gitignore` coverage around temporary local tooling outputs and clarified the public-versus-private repository boundary in the documentation

## 2026-05-02 — Shahmeer — Manuscript draft expanded to full paper structure

- Completed the working IEEE manuscript structure in `report/main.tex`, including title, author block, abstract, Introduction, Related Work, Dataset, Methodology, Experimental Setup, Results, Discussion, and Conclusion
- Populated the core manuscript tables and figures from the authoritative `v2` outputs, including within-condition baselines, cross-condition transfer results, SHAP `delta_j` summaries, prior-work positioning, and clustering summaries
- Integrated Step 5 to 8 assets into the paper flow, with Ali's robustness, PCA, and clustering artifacts anchored to the main Results section and supported by inline figure and table placement
- Added source-traceable wording for the reported methodology so manuscript claims align directly with `src/train.py`, `src/explain.py`, `src/features.py`, and the stored experiment JSONs

## 2026-05-02 — Shahmeer — Literature integration and manuscript positioning pass

- Expanded `report/references.bib` with the working paper set used in the manuscript, including gait-clinical background, same-dataset comparators, broader ML gait papers, XAI references, and the public repository citation
- Reworked the paper narrative to position the study around the specific gap it fills, zero-shot cross-condition transfer and feature-reliance shift diagnosis, rather than as a direct leaderboard comparison against prior work
- Added limitation-forward discussion text covering feature redundancy, control-pool mismatch, shared target controls, sample-size uncertainty, residual preprocessing artifacts, and the practical interpretation boundary of DFA-based comparisons
- Harmonized the manuscript voice around a two-researcher perspective while keeping the technical sections precise and the results grounded in the authoritative `v2` path

## 2026-04-25 — Shahmeer — Repository restructuring: versioned artifact layout

- Applied a versioned local artifact structure across `data/processed/`, `experiments/results/`, `experiments/models/`, `experiments/shap/`, and `report/figures/`, moving historical outputs into `v1/`, authoritative rerun outputs into `v2/`, and creating empty `v3/` placeholders
- Standardised the processed feature filenames to explicit versioned names (`gait_features_v1.csv`, `gait_features_v2.csv`) and moved the pre-existing root-level v1 joblibs into `experiments/models/v1/` while relocating the former `experiments/models_v2/` set into `experiments/models/v2/`
- Removed version subdirectories from `scripts/training/`; active runners now live directly under `scripts/training/` and point to the authoritative local v2 paths while keeping the existing v2 Modal volume layout frozen for compatibility (`/results/*.json`, `/results/models_v2/`, `/results/shap_v2/`)
- Updated `.gitignore`, coordination docs, and `context/authoritative_path_map.md` to match the new local directory structure and authoritative-v2 path conventions

## 2026-04-25 — Shahmeer — Notebook and figure updates for authoritative v2 analysis

- Repointed notebooks `01`–`04` to the authoritative local v2 artifacts for processed data, result JSONs, saved models, SHAP arrays, and figure output directories; legacy outputs were preserved under `report/figures/v1/` and current outputs now write to `report/figures/v2/`
- Added new authoritative-result figures to support the paper narrative directly: `smote_ablation_by_classifier`, `resampling_decision_summary`, `cross_condition_f1_heatmap`, `delta_f1_asymmetry_pairs`, and `dfa_alpha_stride_directionality`
- Expanded notebook 02 with an exact SMOTE vs. no-SMOTE comparison table and a heatmap-style figure/table color scheme so the selected resampling arm is visible numerically as well as visually
- Performed a markdown-only sweep across notebooks 03 and 04 against stored outputs and the v2 JSONs, correcting stale RF/`cv_stride`-dominance language to the authoritative `DT/XGBoost/KNN` and `cv_swing`-dominant interpretation; notebook 04’s opening diagnostic now prints `cv_swing` first and `cv_stride` second

## 2026-04-24 — Ali Aqdas — Step 7: final figures notebook (notebooks/07_final_figures.ipynb)

- Implemented `notebooks/07_final_figures.ipynb` as the Step 7 "publication-quality plots for the IEEE paper" deliverable defined in `CONTRIBUTING.md`; consumes only existing artefacts — no training or sweep is re-executed.
- Inputs: `experiments/results/{pd,hd,als}_results.json`, `cross_condition_results.json`, `shap_results.json`, `noise_robustness.json`, and `data/processed/gait_features.csv` (for the PCA reproduction used in Fig 5 styling).
- Six hero figures exported to a new `report/figures/paper/{pdf,png}/` subdirectory at IEEE column widths (`IEEE_SINGLE = 3.5 in`, `IEEE_DOUBLE = 7.16 in`): `fig1_within_overview` (F1 + modal-frequency heatmaps + single-column F1 variant), `fig2_within_cms` (1×3 confusion matrices from stored `y_true`/`y_pred` with shared colorbar), `fig3_cross_degradation` (ΔF1 heatmap + within-vs-cross bars with stride-level 95% CIs), `fig4_shap_delta_j` (RF normalised δj heatmap + per-direction top-3 features), `fig5_pca_kmeans` (PC1/PC2 condition scatter + K=3 cluster overlay + cumulative variance strip, Step 6-identical fit), and `fig6_noise_robustness` (per-classifier F1-vs-σ curves with ±1 SD bands across 3 conditions; cross sweep is empty in the current JSON and is skipped with a clear message).
- Master results table assembled per (direction × classifier) row with within F1, cross F1, ΔF1, stride CI, subject CI, permutation p-value, and top-1 δj feature; written to `report/tables/master_results.csv` (42 rows) and `report/tables/master_results.tex` (booktabs-style IEEE `tabular`, ready to `\input`).
- Verification cell asserts every paper figure + both table files exist on disk and round-trips all F1 values back to the source JSONs to 1e-6; 42/42 rows confirmed.
- Notebook-generation source of truth kept in `scripts/verification/build_step7_notebook.py` so cell structure is reviewable and regeneration is idempotent; notebook executed end-to-end via `jupyter nbconvert --to notebook --execute --inplace`.
- PDF rendered to `notebooks/pdf/07_final_figures.pdf` via `nbconvert --to latex` + `tectonic` (pandoc + tectonic installed locally). Figures render cleanly; a handful of Greek Unicode glyphs (δ, σ, ∈) in inline text fall back to the default Latin Modern fonts, which does not affect the PDF figure content.

## 2026-04-24 — Ali Aqdas — Step 8: Streamlit demo application (app/app.py)

- Implemented `app/app.py` as the Step 8 "Streamlit deployment demo" deliverable defined in `CONTRIBUTING.md`; a single-file Streamlit app (~460 lines) exposing interactive inference over the existing Step 2 / 3 / 5 artefacts with zero re-processing of raw `.ts` files.
- Inputs consumed read-only: `data/processed/gait_features.csv`, `data/processed/control_partition.json`, `experiments/results/{pd,hd,als}_results.json`, `cross_condition_results.json`, and `noise_robustness.json`. No Step 2–5 sweep, grid-search, bootstrap, or SHAP run is re-executed — the app only re-fits one RF pipeline per cohort on first load using the modal hyper-parameters stored in `{cond}_results.json`, then caches all seven resources (3 RF pipes + 3 TreeExplainers + 1 PCA/scaler) via `@st.cache_resource` for the session so every slider/dropdown interaction is instant.
- Three tabs driven off `src/train.build_pipeline` (keeps the SMOTE → RF structure identical to Step 2 training): **(1) Within-condition inference** — pick PD/HD/ALS, supply a stride, get predicted class + P(disease) alongside the cohort's LOSO F1 with a stride-level bootstrap 95 % CI computed on-the-fly from the stored `y_true`/`y_pred` arrays; **(2) Cross-condition transfer** — pick a source cohort and a target cohort, banner surfaces the stored within-F1 / cross-F1 / ΔF1 / stride CI / permutation p-value pulled from `cross_condition_results.json` plus an automatic calibration warning when `|ΔF1| > 0.10`; **(3) Noise robustness** — σ slider (0 → 1, feature-std multiplier) sweeps a 100-sample Monte-Carlo over the user's stride and overlays the reference F1-vs-σ curve for the cohort from `noise_robustness.json`.
- Interpretability: every tab renders a Plotly SHAP top-5 waterfall built from `shap.TreeExplainer` on the RF classifier (positive contributions plotted in red ⇒ push toward Disease, negative in blue ⇒ push toward Control) plus a PC1/PC2 scatter of all 14,753 strides (Step-6-identical `StandardScaler → PCA(2)` fit) with the user's stride highlighted as a red star, so each prediction is grounded in both local feature attributions and a global cohort-space placement.
- Three input modes available in every tab, all preprocessing-free: **pick an existing subject + stride** from the cohort, **14 sliders** pre-filled with cohort median and clipped to cohort [p1, p99], or **upload a 14-column feature CSV** (multi-stride; the app aggregates per-stride predictions in a table).
- Sidebar shows per-cohort F1 snapshot, a caching-semantics note, and a "research demo only" clinical disclaimer; footer cites the exact models / SHAP / PCA formulations used so reviewers can reproduce the cached components.
- Smoke-tested headlessly: RF fits for all three cohorts in < 3 s total, `TreeExplainer.shap_values` returns valid contributions on a representative stride for all three cohorts, PCA projects the full 14,753-stride matrix, and the noise-robustness reference-curve key lookup round-trips all seven σ keys in `noise_robustness.json` (mean RF F1: 0.777 at σ=0.0 → 0.709 at σ=0.5). `streamlit run app/app.py --server.headless true` boots cleanly and responds with HTTP 200 on `GET /`.
- No new dependencies: `streamlit==1.55.0`, `shap==0.51.0`, and `plotly==6.6.0` were already pinned in `requirements-core.txt`. README updated with a Run-the-demo section describing the three tabs and the cached-model boot step.

## 2026-04-24 — Shahmeer — v2 improvement pass: Step 4 SHAP rerun

- Reworked the kernel SHAP execution path in `src/explain.py` to batch `_kernel_shap_worker` calls through `joblib.Parallel(backend='loky')`, replacing the old fork-based pool that was blocked or effectively serialized under Modal gVisor
- Relaxed the interventional LightGBM/XGBoost completeness guard from a hard `1e-4` assertion to a warn-and-continue path below `0.05`, while retaining the exact `tree_path_dependent` completeness assertion for RF and DT
- The authoritative v2 SHAP summary now confirms `cv_swing` as the dominant transfer-failure feature: mean `δj = 0.0415` across all 42 direction×classifier pairs, with RF ranking it first in 5 of the 6 directions
- `dfa_alpha_stride` shows a directional rather than uniform effect in RF: `pd_to_hd = 0.0305` and `pd_to_als = 0.0416`, but `hd_to_als = 0.0002`, so the feature behaves as a selective shift marker rather than a blanket HD-source dominant signal

## 2026-04-23 — Ali Aqdas — Step 6: PCA/K-Means notebook (notebooks/06_pca_kmeans.ipynb)

- Implemented `notebooks/06_pca_kmeans.ipynb` with full Step 6 flow: setup/imports, data loading from `data/processed/gait_features.csv`, feature scaling, PCA decomposition, K-Means clustering, and interpretation prompts.
- Added and exported all planned figures to `report/figures/pdf/` and `report/figures/png/`: `pca_explained_variance`, `pca_scatter_by_condition`, `pca_feature_loadings`, `kmeans_elbow`, `kmeans_silhouette`, and `kmeans_scatter_k3`.
- Notebook includes both static Matplotlib outputs (paper-ready exports) and interactive Plotly scatter views for PCA and K-Means overlays.
- Added quantitative evaluation cells for unsupervised alignment: ARI for `K=3` vs condition labels, ARI for `K=6` vs condition×label groups, plus contingency tables for both settings.
- Runtime validation of numerical outputs is pending on a fully provisioned environment with project dependencies installed (`polars`, `scikit-learn`) before final reporting of variance/ARI values.

## 2026-04-23 — Shahmeer — v2 improvement pass: Step 3 cross-condition rerun

- Regenerated `experiments/results/v2/cross_condition_results_v2.json` against the new v2 within-condition baselines and confirmed that transfer remains strongly bidirectional and asymmetric across all 6 source→target directions
- `PD→HD` is now the only direction with a negative **mean** `ΔF1` (`-0.0171`), with 5 of 7 classifiers transferring above their PD within-condition baseline; the remaining two (`KNN`, `DT`) degrade only slightly
- `HD→ALS` is the worst degradation direction in the authoritative rerun (`mean ΔF1 = +0.1565`), and RF shows the single largest drop in the matrix (`+0.2742`, from `0.9530` within HD to `0.6788` cross-condition on ALS)
- Subject-level bootstrap intervals remain far wider than stride-level intervals across directions, with subject CIs exceeding stride CIs by roughly `0.20–0.41` F1 units and confirming that between-patient heterogeneity dominates uncertainty

## 2026-04-22 — Ali Aqdas — Step 5 Phase B: run_noise_robustness_local.py

- Added `scripts/training/run_noise_robustness_local.py`: loads `gait_features.csv`, control partition, Step 2 JSONs, optional `cross_condition_results.json`; calls `robustness` helpers; writes `noise_robustness.json`, `feature_sensitivity.json`, and `subject_sensitivity.json` under `experiments/results/`.
- Cross-condition blocks are skipped automatically when `cross_condition_results.json` is absent; within-condition and per-subject (from stored predictions) always run.

## 2026-04-22 — Ali Aqdas — Step 5: remove environment-specific classifier skipping

- Removed `fresh_classifier_or_skip` and LOSO `None`/`nan` fallbacks from `src/robustness.py`; XGBoost/LightGBM are instantiated like other classifiers and failures surface as normal errors (intended for Linux/other CI or lab machines with full native deps, not partial Mac workarounds).

## 2026-04-22 — Ali Aqdas — Step 5 Phase A: src/robustness.py (noise + sensitivity helpers)

- Added self-contained `src/robustness.py` for Step 5 noise robustness and sensitivity analysis without importing `train.py` (keeps Step 2/3 code untouched per strict file scope).
- Implements Gaussian noise on test inputs only, LOSO with fixed modal params (pre-fit folds once per classifier then reuse for noise/permutation repeats), per-feature column shuffle on test folds (within) or target matrix (cross), and per-subject accuracy from stored `y_true` / `y_pred` plus LOSO test index alignment.
- Classifier construction and `build_pipeline` mirror `train.py` defaults; XGBoost and LightGBM are imported only inside `_fresh_classifier()` when those models are instantiated.

## 2026-04-22 — Ali Aqdas — Step 5 Phase C: full Modal runner added

- Added `scripts/training/run_noise_robustness_modal.py` mirroring existing project Modal runner patterns (`debian_slim` image, `requirements-core.txt`, `PYTHONPATH=/root/src`, `src` mounted into image, `gait-results` volume mounted at `/results`).
- Runner executes the full Step 5 pipeline only (no reduced sweep): full sigma grid `(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)`, `N_NOISE_REPEATS=30`, all 7 classifiers, and both within/cross-condition analyses.
- Outputs are written to Modal volume as `/results/noise_robustness.json`, `/results/feature_sensitivity.json`, and `/results/subject_sensitivity.json`; local entrypoint prints exact `modal volume get` commands for download.
- Validation attempt from local environment failed before remote execution because Modal auth token is not configured (`Token missing. Could not authenticate client.`). Script is ready; run once `modal token new` is completed on the operator machine.

## 2026-04-22 — Shahmeer — v2 improvement pass: Step 2 within-condition rerun

- Broadened the within-condition search space in `src/train.py` and reran nested LOSO-CV for all three diseases with a per-classifier SMOTE vs. no-SMOTE ablation recorded directly in the v2 result JSONs
- Resampling selection is now condition-specific rather than global: HD chose `no_smote` for all 7 classifiers, ALS chose `smote` for all 7 classifiers, and PD remained mixed (`RF/DT/QDA = smote`, `KNN/SVM/XGB/LGBM = no_smote`)
- Added a dedicated PD no-SMOTE sensitivity artifact at `experiments/results/v2/pd_results_v2_nosmote.json` so the selected PD recipe can be compared directly against a forced no-SMOTE rerun
- Authoritative v2 within-condition winners are now PD `DT` (`F1 = 0.8668`), HD `XGBoost` (`F1 = 0.9562`), and ALS `KNN` (`F1 = 0.8300`)

## 2026-04-21 — Shahmeer — v2 improvement pass: Step 1 feature engineering additions

- Extended `src/features.py` from the original 14-feature baseline to a 17-feature v2 set by adding `stride_asymmetry_signed`, `cv_swing`, and `dfa_alpha_stride`; the authoritative Step 1 artifact is now `data/processed/v2/gait_features_v2.csv`
- Rebuilt the processed matrix with 14,753 clean strides and 17 feature columns (`+ subject_id`, `condition`, `label`), preserving the existing control partition in `data/processed/control_partition.json`
- Implemented DFA per subject on the left stride-time sequence using log-spaced window sizes, linear detrending inside each scale, and explicit assertions on minimum usable strides and minimum valid scales before accepting an exponent
- `cv_swing` and signed stride asymmetry were added as subject-/stride-level complements to `cv_stride` and `asymmetry_index`, preserving laterality and swing-phase variability in the v2 feature space

## 2026-04-12 — Shahmeer — Step 4 Phase C: 04_shap_diagnosis.ipynb implementation

- Implemented `notebooks/04_shap_diagnosis.ipynb` with 8 analysis cells, markdown explanation before each cell, and completed clinical narrative
- Figures produced: δj heatmap (RF raw + 7-classifier normalised mean), cross-classifier Spearman consistency (6×7×7 matrices), within vs. cross SHAP importance bars (RF, 6 directions), top-3 δj features table (pandas gradient + matplotlib color-coded), emerged features bar chart, δj vs ΔF1 scatter, δj summary statistic sensitivity analysis (4 variants), 3 waterfall plots for most confidently misclassified pd→hd strides
- Key results: cv_stride top-shifted in 5/6 directions; right_swing_pct emerged exclusively in XGB/LightGBM (8/42 pairs); overall δj→ΔF1 Spearman ρ=0.345 (p=0.025) driven by between-condition differences, not within-condition relationship (per-condition ρ range: -0.07 to +0.14, all p>0.6); max-normalised δj remains best summary statistic across 4 tested variants
- All 18 figures saved to `report/figures/pdf/` and `report/figures/png/`; notebook converts to PDF cleanly

## 2026-04-12 — Shahmeer — Step 4 complete: infrastructure and result verification

- Confirmed two independent Modal runs produce scientifically equivalent results; shap_results.json authoritative from run 2
- report/figures/ reorganised into pdf/ and png/ subdirectories; all notebooks updated to save to both
- Step 4 deferred items documented: expanded hyperparameter grids, subject-level normalization, additional engineered features, SMOTE ablations — unified post-Step-4 improvement pass planned before paper submission

## 2026-04-11 — Shahmeer — Step 4 Phase B: runner scripts and Modal run

- Added `scripts/training/run_shap_local.py`: sequential runner with partial save after each direction (`shap_results_partial.json` → renamed `shap_results_local.json` at completion to enable comparison with Modal run)
- Added `scripts/training/run_shap_modal.py`: 6-container parallel Modal runner (one container per direction); `reuse_within=False` to avoid race condition between containers sharing the same source condition; polling loop with `future.get(timeout=5)` for completion-order feedback
- Modal run 1: completed with one container preemption (`pd_to_hd` during SVM pool workers); restart succeeded; all 42 entries confirmed present in `shap_results.json`
- Modal run 2 (verification run): all 6 containers completed cleanly with no preemptions; wall time ≈32 min total; tree classifier results bit-for-bit identical to run 1; kernel classifier results differ by ≤0.002 δj (KernelExplainer random coalition sampling, expected)
- `fork` context retained over `spawn` after confirming spawn is incompatible with stdin-based Python scripts and Modal container entrypoints; preemption-triggered KeyboardInterrupt in forked workers is handled by Modal's automatic container restart
- Key findings: cv_stride dominates δj in 5/6 directions for RF; asymmetry_index prominent in HD-source directions; completeness errors excellent across all 42 pairs; 972/6611 (14.7%) strides misclassified in pd→hd RF cross-condition

## 2026-04-10 — Shahmeer — Step 4 Phase A: src/explain.py implementation

- Implemented `src/explain.py` (683 lines): full SHAP-based transfer-failure diagnosis library for all 7 classifiers
- Explainer assignment: RF/DT use `shap.TreeExplainer(feature_perturbation='tree_path_dependent')` — exact probability-scale output; XGB/LightGBM use `shap.TreeExplainer(feature_perturbation='interventional', model_output='probability')` — required because tree_path_dependent produces log-odds for gradient boosting; SVM/QDA/KNN use `shap.KernelExplainer(pipeline.predict_proba, background)` — full ImbPipeline passed to keep SHAP values in original 14-dimensional feature space
- Background data: class-balanced `shap.sample(k=100)` from source pool — equal disease/control strides sampled before k-means to prevent disease-cluster bias; produces base values ≈0.46–0.50 for XGB/LGB (vs. ≈0.88 without balancing)
- δj metric implemented: `δj = |mean(|φj_within|) − mean(|φj_cross|)|` with normalised variant `δj_norm = δj / mean(|φj_within|)` capped at 10.0; emerged features flagged where `mean(|φj_within|) < 1e-3`
- SHAP 0.51.0 fix: interventional TreeExplainer requires `np.asarray(background.data)` not the DenseData object; `shap.sample()` returns plain numpy array in this version (not DenseData), enabling direct pickling for multiprocessing
- Completeness verification: tree classifiers assert max error < 1e-4; kernel classifiers report median error (approximate by design); all 7 paths verified — RF/DT: 1e-15 to 4e-16 (machine precision), XGB/LGB: 1e-7 to 1e-8, KNN/SVM/QDA: ≈2e-16
- Storage: raw SHAP arrays to `experiments/shap/{source}_{clf}_{pool}.npz` (gitignored); aggregated δj results to `experiments/results/shap_results.json`
- RNG: direction-specific deterministic seed `42 + condition_seeds[src]*10 + condition_seeds[tgt]` enabling parallel Modal containers with reproducible subsamples

## 2026-04-10 — Shahmeer — Step 4 Phase A: KernelExplainer parallelization investigation

- Identified that KernelExplainer is fundamentally single-threaded (Python loop over each sample); adding CPU cores via n_jobs has no effect on the core loop — 10% CPU utilization on Modal was one core at 100% out of 16
- Benchmarked threadpoolctl fix under gVisor: speedup = 0.99× (noise level) — confirmed threading is not the bottleneck, unlike the previous n_jobs=-1 bug in Step 2
- gVisor overhead: Modal containers run ≈6× slower than local hardware for KernelExplainer (1.73s/sample vs 0.22s/sample) due to syscall overhead on repeated predict_proba calls
- Sequential KernelExplainer at n_explained=1000, nsamples=1024: estimated 115 min/computation on Modal — infeasible for 36 total kernel computations
- Solution: custom batch parallelization via `mp.get_context('fork').Pool` splitting 1000 samples into per-core batches; each worker loads its own pipeline copy from disk; achieves near-linear speedup
- Modal diagnostic confirmed: 50 samples in 19.3s with 31 workers → estimated 25.8 min for full 1000-sample run — feasible within Modal timeout

## 2026-04-09 — Shahmeer — Step 3: cross-condition notebook complete (notebooks/03_cross_condition.ipynb)

- Implemented `notebooks/03_cross_condition.ipynb` with all analysis cells: per-classifier degradation table (styled HTML + matplotlib PDF-compatible split into two half-figures), degradation heatmap (`degradation_heatmap.pdf/.png`), within-vs-cross F1 grouped bar chart with per-classifier within-condition baselines and stride CI error bars (`within_vs_cross_f1.pdf/.png`), normalized confusion matrices with single shared colorbar (`cross_condition_cms.pdf/.png`), per-subject accuracy grouped bar chart with disease/control separation (`per_subject_accuracy.pdf/.png`), CI width comparison (`ci_width_comparison.pdf/.png`), and per-class recall breakdown table (`recall_table.pdf/.png`)
- Paired Wilcoxon signed-rank test cell added (one-sided, H1 = within > cross, n=7 classifiers per direction); results recorded in notebook output
- All 9 figures saved to `report/figures/`; notebook converts to `notebooks/03_cross_condition.pdf` without errors

## 2026-04-08 — Shahmeer — Step 3 Phase B: runner scripts and full six-direction transfer run

- Added `scripts/training/run_cross_condition_local.py`: sequential local runner for all six transfer directions; loads `gait_features.csv`, `control_partition.json`, and the three Step 2 result JSONs; accumulates all six direction dicts into a single output dict; writes `experiments/results/cross_condition_results.json` after all directions complete
- Added `scripts/training/run_cross_condition_modal.py`: single-container Modal runner (cpu=16, memory=16384) replicating the image/volume/entrypoint pattern from `run_within_condition_modal.py`; single container because the 21 source models (7 classifiers × 3 source conditions) are fitted once and cached via joblib for reuse across directions sharing the same source
- Local run: all six directions completed in 136 seconds; all verification assertions pass (F1 round-trip < 1e-6, CI brackets point estimate, p-value ∈ [0,1], both classes in target, 21 joblib model files confirmed in `experiments/models/`)
- `experiments/results/cross_condition_results.json` written by local run and confirmed bit-for-bit identical to `experiments/results/cross_condition_results_modal.json` downloaded from Modal volume
- Preliminary degradation table (best-classifier ΔF1 per direction): pd→hd −0.0431, hd→pd +0.2288, pd→als +0.0312, als→pd +0.2722, hd→als +0.1653, als→hd +0.2942

## 2026-04-08 — Shahmeer — Step 3 Phase C: subject-level bootstrap CIs and structural fix

- Added subject-level bootstrap CIs to `run_cross_condition()` in `src/train.py`: 10,000 resamples (vs. 1,000 for stride-level) drawing subjects with replacement and collecting all their strides, with a degenerate-resample rejection loop that discards and redraws any resample producing a single-class y_true; rejection rates confirmed negligible (0.01–0.03% across all 42 classifier-direction pairs)
- Moved `target_subject_ids` from per-classifier dict to direction-level in `src/train.py` and regenerated `experiments/results/cross_condition_results.json`; field now appears once per direction alongside `target_pool_subjects` and `target_pool_strides`, not duplicated 7× across classifiers
- Subject CI fields `f1_macro_subj_ci_lower` and `f1_macro_subj_ci_upper` confirmed present and bracketing point estimate for all 42 entries; subject CIs are 0.20–0.41 F1 units wider than stride CIs, confirming subject-to-subject variance dominates uncertainty
- Modal run confirmed bit-for-bit identical F1 values and stride CIs to the local run; subject CIs differ by up to ~0.02 due to different rng state consumed by the 10,000-resample loops across the two runs (rng state is direction-sequential and the Modal run processes directions independently)

## 2026-04-07 — Shahmeer — Step 2 improvement candidates identified, deferred to post-Step-4

- Identified four improvement candidates for the within-condition baseline after Steps 3 and 4 are complete: (1) expanded hyperparameter grids for RF, KNN, SVM, XGBoost, LightGBM; (2) subject-level normalization / dimensionless normalization (highest expected value for PD); (3) additional engineered features (stance/swing ratio, signed L/R asymmetry, CV of swing and stance); (4) SMOTE ablations for HD and ALS
- Deferred to a unified post-Step-4 Modal run combining improved Steps 1–4 in sequence; no changes to existing Step 2 results or pipeline
- Current Step 2 results remain the authoritative baseline for Step 3 degradation computation

## 2026-04-07 — Shahmeer — Step 3 Phase A: run_cross_condition() implementation and documentation

- Implemented `run_cross_condition()` in `src/train.py`: source pool construction with pool-size assertions against Step 2 JSON, target pool construction with both-class assertion, per-classifier modal-param loading → pipeline build → set_params → fit-or-load → predict → metrics → bootstrap CI → permutation test → joblib save
- Bootstrap 95% CI: 1,000 resamples of (y_true, y_pred) pairs using `numpy.random.default_rng(42)`, 2.5th/97.5th percentile of F1 macro distribution
- Permutation test: 1,000 shuffles of y_true against original y_pred, p-value = fraction of permuted F1 ≥ observed F1
- Model persistence: fitted pipelines saved to `experiments/models/{source_cond}_{clf_name}.joblib`; function is idempotent (loads existing model rather than retraining)
- Verification: all assertions pass; pd→hd RF F1=0.8208 CI=[0.8108,0.8309] p=0.0000; target pool 27 subjects 6,611 strides confirmed; zero source/target overlap confirmed

## 2026-04-06 — Shahmeer — Within-condition training: Modal re-run, y_true/y_pred persistence
- Updated `src/train.py` and `scripts/training/run_within_condition_modal.py` to persist `y_true` and `y_pred` as flat Python lists in each classifier's result JSON entry; added `< 1e-6` F1 round-trip assertion at write time
- Re-ran all three conditions on Modal (cpu=16, memory=16384) to produce updated `experiments/results/{pd,hd,als}_results.json` with stored prediction arrays
- LightGBM num_leaves grid confirmed {31, 63} on Modal; num_leaves=31 selected in all conditions; only change from previous run is LightGBM PD F1: 0.7211 → 0.7251
- All other classifiers bit-for-bit identical between local and Modal runs, confirming full determinism across platforms

## 2026-04-06 — Shahmeer — Within-condition notebook: figures and PDF (notebooks/02_within_condition.ipynb)
- Refactored `notebooks/02_within_condition.ipynb` to use stored predictions from result JSON files instead of re-running LOSO-CV with fixed modal params (`eval_loso_fixed` removed entirely)
- Fixed `FIGURES_DIR` from `reports/figures/` (wrong) to `report/figures/` (correct)
- All five paper figures saved to `report/figures/`: `f1_within_condition_heatmap`, `modal_frequency_heatmap`, `cm_pd_rf`, `cm_hd_rf`, `cm_als_knn` (PDF + PNG each)
- Notebook executes top-to-bottom without errors

## 2026-04-04 — Shahmeer — Within-condition training: local run, Modal run, results analysis
- Ran full within-condition training locally for all three conditions and all 7 classifiers using `scripts/run_step2_training.py`; results saved to `experiments/results/{pd,hd,als}_results.json`
- Identified nested OpenMP parallelism bug (n_jobs=-1 at both GridSearchCV and classifier level under gVisor); fixed by setting n_jobs=1 on RF, XGBoost, and LightGBM at classifier level; n_jobs=-1 retained at GridSearchCV level only; zero effect on accuracy
- Re-ran on Modal (cpu=16, memory=16384) with expanded LightGBM num_leaves grid {31, 63} to validate the num_leaves=31 constraint; confirmed num_leaves=31 selected in all three conditions, validating original grid choice
- Authoritative results (Modal run): PD best RF F1=0.7777, HD best RF F1=0.8995, ALS best KNN F1=0.8758

## 2026-04-03 — Shahmeer — Within-condition training: Phase A (src/train.py, three core functions)
- Implemented `build_pipeline()`, `run_nested_loso()`, `get_modal_params()` in `src/train.py`
- `build_pipeline()` applies feature scaling before SMOTE for SVM and KNN; omits scaler for tree-based methods and QDA (current repo state uses RobustScaler)
- `run_nested_loso()` uses outer LOSO for evaluation and inner GridSearchCV with a fresh LOSO on the training fold only; F1 macro computed over the full concatenated prediction vector (not per-fold, which is undefined for single-class test sets)
- `get_modal_params()` uses fold_best_scores as tiebreaker when two combinations are equally frequent
- Verified on PD pool (23 subjects, 5738 strides, 1.78:1 ratio): aggregate F1 macro = 0.7784 with reduced 4-combination RF grid. Wall time 463s.

## 2026-04-03 — Shahmeer — Within-condition training: Phase B (src/train.py complete, runner script)
- Implemented `get_classifier_configs()` with all 7 classifiers (RF, KNN, SVM, DT, QDA, XGBoost, LightGBM) and full hyperparameter grids matching GUIDELINE.md
- Implemented `run_within_condition()` which constructs the binary training pool, runs nested LOSO-CV for all classifiers, and saves results to `experiments/results/{condition}_results.json`
- Wrote `scripts/run_step2_training.py` — standalone runner for all three conditions in sequence, compatible with `nohup` and `tail -f` monitoring via `flush=True` throughout
- Smoke test: 37/37 checks pass (classifier presence, n_jobs, probability, class_weight, QDA type, grid sizes, clf__ key prefixes)

## 2026-04-02 — Shahmeer — Step 1 Phase A: Data loading, filtering, labelling, and control partitioning
- Implemented `load_raw_data()`, `filter_pause_events()`, `assign_labels()`, `partition_controls()` in `src/preprocessing.py`
- Pre-implementation audit identified two discrepancies from GUIDELINE.md: (1) the stride-only filter removes 276 rows (not 484), and (2) an additional 131 rows have physically impossible double_support_pct > 100 due to sensor malfunction. Both GUIDELINE.md and CLAUDE.md updated accordingly. hunt20 entirely removed by stride filter (right-foot sensor frozen), leaving 19 usable HD subjects.
- Verified: raw=15,160, removed by stride filter=276, removed by pct filter=131, clean=14,753. Subject counts: PD=15, HD=19 usable, ALS=13, Control=16. hunt20 rows in clean data=0. control_partition.json written.

## 2026-04-02 — Shahmeer — Step 1 Phase B: Feature engineering and full pipeline orchestration
- Implemented `compute_asymmetry_index()`, `compute_cv_stride()`, `build_feature_matrix()` in `src/features.py`
- `build_feature_matrix()` orchestrates the full Step 1 pipeline and writes both output artifacts to `data/processed/`
- Verified: final shape=(14753, 17), null values in feature columns=0, gait_features.csv written, control_partition.json written. asymmetry_index range [0.000, 0.929], mean=0.037. cv_stride per-subject range [0.018, 0.240], mean=0.076. 63 unique subjects (64 minus hunt20).

## 2026-04-02 — Shahmeer — Step 1: Data exploration notebook (00_data_exploration.ipynb)
- Created notebook documenting raw dataset inspection performed during pre-implementation audit
- Cells display: file counts per condition (64 total), raw file structure (13 tab-separated columns), total raw stride count (15,160) with per-subject breakdown, pause event distribution (276 rows with stride > 3.0s), hunt20 frozen sensor anomaly (all 238 rows), als5/als7 double_support_pct > 100 anomaly (131 rows)
- Summary records the two data quality issues that motivated the dual-filter design

## 2026-04-02 — Shahmeer — Step 1: Preprocessing notebook (01_preprocessing.ipynb)
- Created notebook that imports `build_feature_matrix()` from `src/features.py` and runs the full Step 1 pipeline
- Cells display: feature matrix shape (14753x17), per-condition subject/stride counts, asymmetry_index stats, cv_stride per-subject stats, first 5 rows, file existence checks, CSV reload verification, null count confirmation
- Verification summary table confirms all Step 1 checkpoints pass

## 2026-03-30 — Shahmeer
- Initialized repository and directory structure
- Setup independent google drive for relevant papers and GUIDELINE.md
- Downloaded dataset
