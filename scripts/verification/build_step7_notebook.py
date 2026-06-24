"""
Build notebooks/07_final_figures.ipynb (Step 7) for the authoritative v4 line.

Step 7 is the manuscript-facing synthesis notebook. It curates existing
step-specific v4 figures and tables, writes paper figure/table manifests, and
adds final reporting guidance without regenerating the main analysis figures.

Usage:
    python scripts/verification/build_step7_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


OUT = Path(__file__).resolve().parents[2] / "notebooks" / "07_final_figures.ipynb"


def md(text: str, cell_id: str) -> dict:
    body = dedent(text).strip("\n")
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in body.split("\n")],
    }


def code(text: str, cell_id: str) -> dict:
    body = dedent(text).strip("\n")
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in body.split("\n")],
    }


cells: list[dict] = []

cells.append(md(
    """
# Step 7: Final Results Synthesis and Paper Figure Selection

This notebook assembles the manuscript-facing figure and table set from the
authoritative v4 outputs. It focuses on empirical evidence and final reporting
choices: which existing step-specific figures are strongest for the main paper,
which belong in the supplement, and which tables should carry exact reported
values.

The study-design schematic will be handled directly in the manuscript source,
so this notebook concentrates on the finalized analytical figures and tables.
    """,
    "step7-md-intro",
))


cells.append(code(
    """
    import sys
    from pathlib import Path

    import pandas as pd
    from IPython.display import Image, display


    def find_repo_root() -> Path:
        candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        for base in candidates:
            if (base / "data" / "processed" / "v4" / "v4_protocol_manifest.json").exists():
                return base
        raise RuntimeError("Could not locate repository root from the current working directory.")


    REPO_ROOT = find_repo_root()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))

    FIG_PDF_ROOT = REPO_ROOT / "report" / "figures" / "v4" / "pdf"
    FIG_PNG_ROOT = REPO_ROOT / "report" / "figures" / "v4" / "png"
    TABLE_DIR = REPO_ROOT / "report" / "tables" / "v4"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    DIRECTIONS = ["pd_to_hd", "hd_to_pd", "pd_to_als", "als_to_pd", "hd_to_als", "als_to_hd"]
    DIR_LABEL = {d: d.replace("_to_", "→").upper() for d in DIRECTIONS}
    CLF_LABEL = {
        "rf": "RF",
        "knn": "KNN",
        "svm": "SVM",
        "dt": "DT",
        "qda": "QDA",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
    }


    def figure_paths(step: str, stem: str) -> tuple[Path, Path]:
        return FIG_PDF_ROOT / step / f"{stem}.pdf", FIG_PNG_ROOT / step / f"{stem}.png"


    def display_existing_figure(step: str, stem: str, width: int = 780) -> None:
        pdf_path, png_path = figure_paths(step, stem)
        assert png_path.exists(), f"Missing PNG figure: {png_path}"
        assert pdf_path.exists(), f"Missing PDF figure: {pdf_path}"
        print(f"Displayed figure: {png_path.relative_to(REPO_ROOT)}")
        print(f"Vector source: {pdf_path.relative_to(REPO_ROOT)}")
        display(Image(filename=str(png_path), width=width))


    def save_table_csv(df: pd.DataFrame, name: str) -> Path:
        path = TABLE_DIR / name
        df.to_csv(path, index=False)
        print(f"Saved table: {path.relative_to(REPO_ROOT)}")
        return path


    def save_table_latex(
        df: pd.DataFrame,
        name: str,
        caption: str,
        label: str,
        *,
        longtable: bool = False,
    ) -> Path:
        path = TABLE_DIR / name
        latex = df.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label,
            longtable=longtable,
        )
        path.write_text(latex)
        print(f"Saved table: {path.relative_to(REPO_ROOT)}")
        return path


    FIGURE_SPECS = [
        {
            "tier": "main",
            "step": "step0",
            "figure_stem": "step0_authoritative_filtering_cascade",
            "recommended_caption": "Authoritative v4 filtering cascade from raw stride rows to the retained cohort.",
            "claim_supported": "The retained dataset is auditable from raw rows to the frozen analytical matrix.",
            "why_included": "Cohort accounting is essential for a publication-track biomedical ML study.",
            "caveat": "Descriptive only; not a performance result.",
        },
        {
            "tier": "main",
            "step": "step1",
            "figure_stem": "step1_control_split_summary",
            "recommended_caption": "Frozen Control A / Control B split used for transfer and control-partition sensitivity analyses.",
            "claim_supported": "Downstream transfer and Step 8 sensitivity rely on an explicit disjoint-control design.",
            "why_included": "This is the clearest visual anchor for the control-partition methodology.",
            "caveat": "Small cohorts still permit sensitivity to alternate near-optimal control splits.",
        },
        {
            "tier": "main",
            "step": "step2",
            "figure_stem": "f1_within_condition_heatmap",
            "recommended_caption": "Subject-level within-condition macro-F1 across classifiers and source conditions.",
            "claim_supported": "The source-condition benchmark is strong enough to make downstream transfer interpretable.",
            "why_included": "This is the cleanest subject-primary overview of Step 2.",
            "caveat": "ALS still requires small-sample caution despite its strong scores.",
        },
        {
            "tier": "main",
            "step": "step2",
            "figure_stem": "step2_subject_stride_leaders",
            "recommended_caption": "Condition-level leaders with subject-primary and stride-level companion metrics.",
            "claim_supported": "The paper can report exact Step 2 leaders without hard-coded manuscript values.",
            "why_included": "The panel complements the heatmap with exact condition-level numbers.",
            "caveat": "Nested winners are protocol-specific, not universal classifier rankings.",
        },
        {
            "tier": "main",
            "step": "step3",
            "figure_stem": "degradation_heatmap",
            "recommended_caption": "Subject-level matched degradation by transfer direction and classifier.",
            "claim_supported": "Transfer is measurable but usually degraded relative to within-source subject baselines.",
            "why_included": "Matched degradation is the central Step 3 comparison lens.",
            "caveat": "Direction-level means are uniformly positive, but individual classifier-direction cells still vary.",
        },
        {
            "tier": "main",
            "step": "step3",
            "figure_stem": "cross_condition_f1_heatmap",
            "recommended_caption": "Absolute subject-level zero-shot transfer performance by direction and classifier.",
            "claim_supported": "Transfer success is directional and classifier-dependent rather than uniform.",
            "why_included": "It complements matched degradation with the absolute transfer-performance surface.",
            "caveat": "Absolute F1 should still be interpreted alongside the within-source baseline.",
        },
        {
            "tier": "main",
            "step": "step3",
            "figure_stem": "step3_subject_confusion_matrices",
            "recommended_caption": "Subject-level normalized confusion matrices for the best classifier in each transfer direction.",
            "claim_supported": "The strongest transfer directions remain interpretable in terms of control specificity and target-disease recall.",
            "why_included": "This panel adds error-structure information that the heatmaps do not show directly.",
            "caveat": "The matrices summarize only the direction-level leaders, not every classifier.",
        },
        {
            "tier": "extended_main",
            "step": "step3",
            "figure_stem": "step3_direction_intervals",
            "recommended_caption": "Direction-level subject-F1 intervals for the best classifier in each transfer direction.",
            "claim_supported": "The strongest transfer directions still carry non-trivial uncertainty.",
            "why_included": "Useful when the manuscript wants explicit interval context beyond heatmaps.",
            "caveat": "Descriptive intervals only; not a primary direction-level inferential test.",
        },
        {
            "tier": "extended_main",
            "step": "step3",
            "figure_stem": "step3_bidirectional_asymmetry",
            "recommended_caption": "Bidirectional asymmetry of subject-level transfer between disease pairs.",
            "claim_supported": "Source-to-target direction matters and cannot be treated as interchangeable.",
            "why_included": "This is the most direct visualization of the asymmetry claim.",
            "caveat": "Asymmetry should be interpreted within this repaired v4 protocol, not as a universal disease ordering.",
        },
        {
            "tier": "main",
            "step": "step4",
            "figure_stem": "step4_delta_j_rf_heatmap",
            "recommended_caption": "Random-forest SHAP reliance shifts under transfer.",
            "claim_supported": "Transfer alters feature reliance in concrete, direction-specific ways for a representative strong model family.",
            "why_included": "It provides a model-specific diagnostic anchor for the broader SHAP story.",
            "caveat": "Model-specific SHAP movement should not be generalized without the consensus views.",
        },
        {
            "tier": "main",
            "step": "step4",
            "figure_stem": "step4_delta_j_normalized_consensus_heatmap",
            "recommended_caption": "Normalized consensus SHAP reliance shifts across classifiers and transfer directions.",
            "claim_supported": "Several timing-structure features recur as transfer-shift signals across model families.",
            "why_included": "This is the strongest cross-classifier Step 4 panel.",
            "caveat": "Diagnostic only; not evidence of causal gait biomarkers.",
        },
        {
            "tier": "main",
            "step": "step4",
            "figure_stem": "step4_delta_j_spearman",
            "recommended_caption": "Direction-wise cross-classifier rank agreement of SHAP transfer-shift features.",
            "claim_supported": "Some directions show coherent diagnostic ranking across classifier families, whereas others are more fragmented.",
            "why_included": "It indicates when the transfer-diagnosis story is shared across models rather than model-specific.",
            "caveat": "Agreement patterns are descriptive and should not be overread as proof of a single latent mechanism.",
        },
        {
            "tier": "extended_main",
            "step": "step4",
            "figure_stem": "step4_family_delta_j_consensus",
            "recommended_caption": "Family-level consensus movement in feature reliance under transfer.",
            "claim_supported": "Variability and raw timing dominate the family-level transfer-diagnostic story.",
            "why_included": "This is the cleanest abstraction over the feature-level maps.",
            "caveat": "Family movement summarizes reliance shifts, not biology.",
        },
        {
            "tier": "extended_main",
            "step": "step4",
            "figure_stem": "step4_top_feature_recurrence",
            "recommended_caption": "Recurring top-ranked transfer-shift features across directions and classifiers.",
            "claim_supported": "A small set of timing-variability descriptors repeatedly reappears in the transfer-diagnostic layer.",
            "why_included": "Useful as a feature-level complement to the family summary.",
            "caveat": "Recurrence is descriptive rather than causal.",
        },
        {
            "tier": "main",
            "step": "step5",
            "figure_stem": "noise_curves_cross",
            "recommended_caption": "Cross-condition Gaussian feature-space stress curves across transfer directions.",
            "claim_supported": "Cross-condition performance deteriorates under increasing feature-space stress in a direction-dependent way.",
            "why_included": "This is the most compact robustness summary for transferred models.",
            "caveat": "Gaussian feature-space stress is not raw sensor simulation.",
        },
        {
            "tier": "main",
            "step": "step5",
            "figure_stem": "step5_feature_sensitivity_within",
            "recommended_caption": "Within-condition single-feature permutation sensitivity heatmap.",
            "claim_supported": "Within-condition models show selective reliance patterns that provide context for transfer sensitivity.",
            "why_included": "The within-condition perturbation surface is scientifically distinct from the cross-condition one.",
            "caveat": "Permutation sensitivity is a perturbation diagnostic rather than a causal attribution.",
        },
        {
            "tier": "main",
            "step": "step5",
            "figure_stem": "step5_feature_sensitivity_cross",
            "recommended_caption": "Cross-condition single-feature permutation sensitivity heatmap.",
            "claim_supported": "Transfer performance is especially sensitive to a small set of timing-variability features.",
            "why_included": "This figure links the Step 4 reliance-shift layer to a direct perturbation response surface.",
            "caveat": "Cross-condition sensitivity under permutation does not establish deployment fragility in the real world.",
        },
        {
            "tier": "main",
            "step": "step5",
            "figure_stem": "conformal_within",
            "recommended_caption": "Within-condition exploratory conformal coverage and set-size behavior.",
            "claim_supported": "Source-calibrated set behavior differs materially by method and alpha even before distribution shift.",
            "why_included": "It gives the matched-domain reference for the conformal diagnostics.",
            "caveat": "Descriptive only; not the primary performance endpoint.",
        },
        {
            "tier": "main",
            "step": "step5",
            "figure_stem": "conformal_cross",
            "recommended_caption": "Cross-condition exploratory conformal coverage and set-size behavior under shift.",
            "claim_supported": "Prediction-set behavior changes under transfer and should be framed as exploratory under domain shift.",
            "why_included": "It directly supports the caution that target-domain coverage is not guaranteed.",
            "caveat": "No formal target-domain conformal guarantee should be claimed here.",
        },
        {
            "tier": "extended_main",
            "step": "step5",
            "figure_stem": "step5_feature_family_sensitivity",
            "recommended_caption": "Feature-family permutation sensitivity across within and transfer settings.",
            "claim_supported": "Variability and raw timing families dominate the average transfer sensitivity story.",
            "why_included": "This is a strong higher-level complement to the feature heatmaps.",
            "caveat": "Family-level averages can hide direction-specific exceptions.",
        },
        {
            "tier": "extended_main",
            "step": "step5",
            "figure_stem": "step5_corruption_cross",
            "recommended_caption": "Structured cross-condition corruption response across severity levels.",
            "claim_supported": "Heavy benchmark label corruption and heavy feature-space jitter are among the most damaging average cross-condition stressors.",
            "why_included": "Adds a perturbation family that is distinct from Gaussian stress and feature permutation.",
            "caveat": "These are benchmark perturbations, not deployment-validated sensor-fault models.",
        },
        {
            "tier": "main",
            "step": "step6",
            "figure_stem": "kmeans_scatter_k3",
            "recommended_caption": "Exploratory K-means overlay in subject-level PCA space.",
            "claim_supported": "The subject-level feature space has visible structure, but that structure should remain exploratory rather than confirmatory.",
            "why_included": "This is the most visually direct Step 6 panel for qualitative geometry context.",
            "caveat": "Exploratory only; clustering does not prove disease separability or transfer validity.",
        },
        {
            "tier": "extended_main",
            "step": "step6",
            "figure_stem": "pca_scatter_by_condition",
            "recommended_caption": "Subject-level PCA projection colored by condition.",
            "claim_supported": "The condition geometry is visually structured but remains only qualitative context for the supervised results.",
            "why_included": "Useful if the supplement wants the raw PCA view alongside the K-means overlay.",
            "caveat": "Visual separation alone is not supervised evidence.",
        },
        {
            "tier": "main",
            "step": "step8",
            "figure_stem": "step8_sign_stability_matrix",
            "recommended_caption": "Sign stability of direction-level matched degradation across near-optimal control partitions.",
            "claim_supported": "Step 8 qualifies stability claims rather than proving invariance.",
            "why_included": "This is the clearest figure for the key Step 8 caveat.",
            "caveat": "Partition 3 reverses four of six direction-level signs.",
        },
        {
            "tier": "main",
            "step": "step8",
            "figure_stem": "step8_direction_degradation_ranges",
            "recommended_caption": "Direction-wise matched-degradation ranges across the main and alternate near-optimal control partitions.",
            "claim_supported": "Control-partition sensitivity affects both sign and effect-size scale.",
            "why_included": "It complements the sign matrix with direction-level magnitude movement.",
            "caveat": "Sensitivity includes movement in both within-source baselines and transfer outcomes.",
        },
        {
            "tier": "extended_main",
            "step": "step8",
            "figure_stem": "step8_within_source_baselines",
            "recommended_caption": "Best within-source subject-level baselines across near-optimal control partitions.",
            "claim_supported": "Some Step 8 movement is driven by baseline instability, not transfer behavior alone.",
            "why_included": "Important supporting context for the sensitivity verdict.",
            "caveat": "Baseline drift does not by itself imply transfer failure.",
        },
        {
            "tier": "extended_main",
            "step": "step8",
            "figure_stem": "step8_best_transfer_leaders_heatmap",
            "recommended_caption": "Best transfer leaders by direction and control partition.",
            "claim_supported": "The identity of the best transfer classifier is not completely fixed across alternate partitions.",
            "why_included": "Helpful if the supplement wants the classifier-family stability view explicitly.",
            "caveat": "Best-leader changes should be read alongside effect-size movement, not in isolation.",
        },
    ]

    TABLE_SPECS = [
        {
            "tier": "main",
            "table_file": "step2_subject_stride_leaders_v4.csv",
            "purpose": "Condition-level within-condition leaders with subject-primary and stride-level companions.",
            "main_columns": "Condition, Leader, Subject F1, Stride F1, Subject log loss, Strategy",
            "paper_use": "Exact Step 2 leader reporting.",
            "caveat": "Nested winners are protocol-specific rather than universal rankings.",
        },
        {
            "tier": "main",
            "table_file": "paper_main_results_summary_v4.csv",
            "purpose": "Full direction-by-classifier cross-condition transfer summary with subject-primary columns emphasized.",
            "main_columns": "Direction, Classifier, Subject F1, Within-source subject F1, Matched degradation (subject), Control specificity, Target recall, Strategy, Permutation p-value",
            "paper_use": "Primary Step 3 numerical reporting table.",
            "caveat": "Direction-level means are the safest headline summaries; not every classifier-direction row degrades.",
        },
        {
            "tier": "main",
            "table_file": "step5_noise_robustness_summary_v4.csv",
            "purpose": "Sigma-zero replay and key severity checkpoints for the Gaussian feature-space stress test.",
            "main_columns": "scope, classifier, f1@0, f1@0.10, f1@0.25, f1@0.50, sigma_at_10pct_drop",
            "paper_use": "Exact Step 5 checkpoint reporting.",
            "caveat": "This is a feature-space perturbation diagnostic, not a raw sensor benchmark.",
        },
        {
            "tier": "main",
            "table_file": "step8_direction_sensitivity_v4.csv",
            "purpose": "Direction-level matched-degradation sensitivity across alternate control partitions.",
            "main_columns": "Partition, Direction, Mean matched degradation, Sign, Best classifier, Best subject F1",
            "paper_use": "Exact Step 8 sensitivity reporting.",
            "caveat": "Sensitivity qualifies the main transfer story rather than replacing it.",
        },
        {
            "tier": "extended_main",
            "table_file": "step0_filter_flow.csv",
            "purpose": "Cohort filtering stages and retained stride-row counts.",
            "main_columns": "stage_label, rows",
            "paper_use": "Methods transparency for cohort accounting.",
            "caveat": "Descriptive only.",
        },
        {
            "tier": "extended_main",
            "table_file": "step1_control_partition_summary.csv",
            "purpose": "Frozen Control A / Control B membership.",
            "main_columns": "subject_id, group",
            "paper_use": "Methods or supplement table for control-reuse transparency.",
            "caveat": "Important for protocol clarity but not a results table.",
        },
        {
            "tier": "extended_main",
            "table_file": "step3_best_direction_recall_summary_v4.csv",
            "purpose": "Best classifier per transfer direction with recall and specificity context.",
            "main_columns": "Direction, Classifier, Subject F1, Delta subject, Control specificity, Target recall",
            "paper_use": "Compact Step 3 companion table if the manuscript wants the best-direction subset explicitly.",
            "caveat": "Best-direction rows do not summarize the full classifier surface.",
        },
        {
            "tier": "extended_main",
            "table_file": "step5_feature_family_sensitivity_v4.csv",
            "purpose": "Family-level permutation sensitivity across within and transfer settings.",
            "main_columns": "scope, domain, classifier, family, drop",
            "paper_use": "Step 5 companion table for the feature-family story.",
            "caveat": "Family averages compress heterogeneous direction-specific behavior.",
        },
        {
            "tier": "extended_main",
            "table_file": "step8_partition_overview_v4.csv",
            "purpose": "Partition-level Step 8 overview with overlap and role-reversal metadata.",
            "main_columns": "partition, role, overlap, independent_sensitivity_candidate, candidate_family",
            "paper_use": "Supplementary context for the control-partition design.",
            "caveat": "Structural metadata rather than a headline results table.",
        },
    ]

    LEGACY_DUPLICATE_TABLES = {
        "step7_within_leaders_v4.csv",
        "step7_within_leaders_v4.tex",
        "step7_cross_transfer_leaders_v4.csv",
        "step7_cross_transfer_leaders_v4.tex",
        "step7_master_transfer_table_v4.csv",
        "step7_master_transfer_table_v4.tex",
    }
    """,
    "step7-code-bootstrap",
))


cells.append(md(
    """
## Manuscript selection criteria

The manuscript should stay subject-primary, extensive enough to cover every
major result layer, and disciplined enough to avoid near-duplicate panels.

- Step-specific figures are preferred over regenerated summary surrogates when
  the existing figure already states the result clearly.
- Main-paper figures should support distinct claims: cohort accounting, frozen
  control design, within-condition benchmarking, zero-shot transfer,
  transfer-diagnostic evidence, stress testing and uncertainty behavior,
  exploratory geometry context, and control-partition sensitivity.
- Dense diagnostics still matter scientifically, but they move to
  `extended_main` or `supplement` when they repeat the same claim at finer
  granularity.
    """,
    "step7-md-policy",
))


cells.append(md(
    """
## Methods and cohort figures

The paper’s methods/result interface should begin with two concrete anchors:
the authoritative filtering cascade and the frozen Control A / Control B
partition. Together they make the retained cohort auditable and clarify why the
later transfer and sensitivity analyses can be interpreted as leakage-aware.
    """,
    "step7-md-methods",
))


cells.append(code(
    """
    display_existing_figure("step0", "step0_authoritative_filtering_cascade", width=800)
    display_existing_figure("step1", "step1_control_split_summary", width=800)
    """,
    "step7-code-methods",
))


cells.append(md(
    """
- **`step0_authoritative_filtering_cascade`** is a main paper figure because it
  shows the retained-row path from **15,160** raw stride rows to **13,765**
  authoritative v4 rows. It supports cohort transparency rather than a
  performance claim.
- **`step1_control_split_summary`** is also main because the later transfer
  protocol and Step 8 sensitivity analysis depend on a frozen **8-vs-8**
  Control A / Control B split. The main caveat is that a balanced split is not
  the same thing as a partition-invariant result.
    """,
    "step7-md-methods-note",
))


cells.append(md(
    """
## Step 2: within-condition benchmark

The Step 2 figures establish whether the source-condition models are strong
enough to support a downstream transfer analysis. The paper should therefore
show both the subject-level leaderboard surface and the compact leader summary.
    """,
    "step7-md-step2",
))


cells.append(code(
    """
    step2_leaders = pd.read_csv(TABLE_DIR / "step2_subject_stride_leaders_v4.csv")
    display_existing_figure("step2", "f1_within_condition_heatmap", width=780)
    display_existing_figure("step2", "step2_subject_stride_leaders", width=780)
    display(step2_leaders)
    """,
    "step7-code-step2",
))


cells.append(md(
    """
The subject-primary Step 2 story is crisp. **ALS** is the strongest source
condition and reaches a **QDA** leader score of **1.0000** subject-level
macro-F1; **HD** is led by **XGBoost** at **0.9571**; and **PD** is flatter,
with **DT** selected as the authoritative leader at **0.9042** subject-level
macro-F1. That combination matters for Step 3 because it means the transfer
analysis starts from strong, but not identical, within-source baselines. The
main caveat is that these leaders are optimal within the repaired nested
procedure only; they should not be described as universal classifier winners.
    """,
    "step7-md-step2-note",
))


cells.append(md(
    """
## Step 3: zero-shot cross-condition transfer

Step 3 is the core empirical result. The manuscript should keep both the
absolute transfer surface and the matched-degradation lens, then pair them with
one detailed diagnostic view and one exact-value table so that the main claim
does not rest on heatmaps alone.
    """,
    "step7-md-step3",
))


cells.append(code(
    """
    transfer_df = pd.read_csv(TABLE_DIR / "step3_cross_condition_summary_v4.csv")
    transfer_table = transfer_df.copy()
    transfer_table["Direction"] = transfer_table["direction"].map(DIR_LABEL)
    transfer_table["Classifier"] = transfer_table["classifier"].map(CLF_LABEL)
    paper_main_results_summary = transfer_table[
        [
            "Direction",
            "Classifier",
            "subject_f1",
            "within_subject_f1",
            "delta_subject",
            "stride_f1",
            "within_stride_f1",
            "delta_stride",
            "subject_ci_low",
            "subject_ci_high",
            "control_specificity_subject",
            "target_recall_subject",
            "selected_strategy",
            "permutation_p_value",
        ]
    ].rename(
        columns={
            "subject_f1": "Subject F1",
            "within_subject_f1": "Within-source subject F1",
            "delta_subject": "Matched degradation (subject)",
            "stride_f1": "Stride F1",
            "within_stride_f1": "Within-source stride F1",
            "delta_stride": "Matched degradation (stride)",
            "subject_ci_low": "Subject CI low",
            "subject_ci_high": "Subject CI high",
            "control_specificity_subject": "Control specificity",
            "target_recall_subject": "Target recall",
            "selected_strategy": "Strategy",
            "permutation_p_value": "Permutation p-value",
        }
    )
    save_table_csv(paper_main_results_summary, "paper_main_results_summary_v4.csv")
    save_table_latex(
        paper_main_results_summary.round(4),
        "paper_main_results_summary_v4.tex",
        "Cross-condition transfer summary with subject-primary columns emphasized.",
        "tab:paper_main_results_summary_v4",
        longtable=True,
    )

    styled_transfer = (
        paper_main_results_summary.style
        .format({
            "Subject F1": "{:.4f}",
            "Within-source subject F1": "{:.4f}",
            "Matched degradation (subject)": "{:+.4f}",
            "Stride F1": "{:.4f}",
            "Within-source stride F1": "{:.4f}",
            "Matched degradation (stride)": "{:+.4f}",
            "Subject CI low": "{:.4f}",
            "Subject CI high": "{:.4f}",
            "Control specificity": "{:.4f}",
            "Target recall": "{:.4f}",
            "Permutation p-value": "{:.4f}",
        })
        .background_gradient(
            subset=["Matched degradation (subject)"],
            cmap="RdYlBu_r",
            vmin=-0.15,
            vmax=0.40,
        )
        .background_gradient(
            subset=["Subject F1"],
            cmap="Blues",
            vmin=0.45,
            vmax=1.0,
        )
        .set_caption("Cross-condition transfer summary (subject-primary columns emphasized)")
    )

    display_existing_figure("step3", "degradation_heatmap", width=780)
    display_existing_figure("step3", "cross_condition_f1_heatmap", width=780)
    display_existing_figure("step3", "step3_subject_confusion_matrices", width=780)
    display_existing_figure("step3", "step3_direction_intervals", width=780)
    display_existing_figure("step3", "step3_bidirectional_asymmetry", width=780)
    display(styled_transfer)
    """,
    "step7-code-step3",
))


cells.append(md(
    """
The transfer figures support the central v4 claim directly. At the
direction-average level, **matched subject-level degradation is positive in all
six directions**, with the largest mean losses in **HD→ALS** (**0.1925**) and
**HD→PD** (**0.1735**), and the smallest in **PD→HD** (**0.0458**). The
absolute subject-F1 heatmap shows that transfer remains practically strong in
selected directions even when degradation is positive overall; the best
direction-level outcomes are **QDA** for **PD→ALS** (**0.9481**), **HD→ALS**
(**0.8929**), **PD→HD** (**0.8712**), and **HD→PD** (**0.8083**), with **DT**
leading **ALS→PD** (**0.8600**) and **LightGBM** leading **ALS→HD**
(**0.8052**). The confusion matrices add a complementary class-balance view:
for example, **ALS→HD** achieves perfect control specificity at the cost of
lower HD recall, whereas **HD→ALS** reaches full ALS recall with lower control
specificity. The key caveat remains explicit in the table: some individual
classifier-direction rows have near-zero or negative matched degradation even
though the direction-level means are uniformly positive.
    """,
    "step7-md-step3-note",
))


cells.append(md(
    """
## Step 4: SHAP-based transfer diagnostics

Step 4 should stay diagnostic rather than causal. The manuscript therefore
benefits from showing the model-specific random-forest map, the cross-classifier
consensus map, and the rank-agreement panel together, with family-level and
recurrence summaries available as supporting detail.
    """,
    "step7-md-step4",
))


cells.append(code(
    """
    top_features = pd.read_csv(TABLE_DIR / "step4_top_feature_recurrence_v4.csv")
    family_delta = pd.read_csv(TABLE_DIR / "step4_family_delta_j_long_v4.csv")
    family_means = (
        family_delta.groupby("family")["delta_j_family"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
        .rename("mean_delta_j_family")
        .reset_index()
    )

    display_existing_figure("step4", "step4_delta_j_rf_heatmap", width=780)
    display_existing_figure("step4", "step4_delta_j_normalized_consensus_heatmap", width=780)
    display_existing_figure("step4", "step4_delta_j_spearman", width=780)
    display_existing_figure("step4", "step4_family_delta_j_consensus", width=780)
    display_existing_figure("step4", "step4_top_feature_recurrence", width=780)
    display(top_features.head(10))
    display(family_means)
    """,
    "step7-code-step4",
))


cells.append(md(
    """
The Step 4 diagnostic layer is coherent but should still be described
conservatively. The consensus heatmap and recurrence table point repeatedly to
**`cv_stride`** (**33** top-3 appearances), **`cv_swing`** (**21**), and
**`dfa_alpha_stride`** (**19**) as recurring transfer-shift features. At the
family level, the mean movement is largest for **variability** (**0.0302**) and
**raw timing** (**0.0274**), with **fractal** movement still visible
(**0.0124**) and the remaining families smaller. The Spearman panel is useful
because it distinguishes directions where classifier families tell a similar
diagnostic story from directions where the feature rankings are more model
contingent. The important caveat is unchanged: these figures diagnose changes
in feature reliance under transfer, not disease-specific causal biomarkers.
    """,
    "step7-md-step4-note",
))


cells.append(md(
    """
## Step 5: stress testing, feature sensitivity, and exploratory conformal diagnostics

Step 5 covers several genuinely different questions, so the synthesis notebook
keeps more than one main figure here. The selected panels separate three claims:
how transfer degrades under Gaussian feature-space stress, which individual
features drive strong permutation sensitivity, and how exploratory conformal set
behavior changes between matched and shifted evaluation.
    """,
    "step7-md-step5",
))


cells.append(code(
    """
    noise_summary = pd.read_csv(TABLE_DIR / "step5_noise_robustness_summary_v4.csv")
    family_sensitivity = pd.read_csv(TABLE_DIR / "step5_feature_family_sensitivity_v4.csv")
    conformal_summary = pd.read_csv(TABLE_DIR / "step5_conformal_summary_v4.csv")
    conformal_methods = conformal_summary[conformal_summary["method"].isin(["aps", "lac"])].copy()
    conformal_agg = (
        conformal_methods
        .groupby(["scope", "method", "alpha"])[
            [
                "coverage_subject_level",
                "raw_mean_set_size",
                "post_mean_set_size",
                "raw_empty_rate",
                "post_empty_rate",
            ]
        ]
        .mean()
        .reset_index()
        .round(4)
    )
    family_drop_means = (
        family_sensitivity.groupby(["scope", "family"])["drop"]
        .mean()
        .reset_index()
        .sort_values(["scope", "drop"], ascending=[True, False])
        .round(4)
    )

    display_existing_figure("step5", "noise_curves_cross", width=780)
    display_existing_figure("step5", "step5_feature_sensitivity_within", width=780)
    display_existing_figure("step5", "step5_feature_sensitivity_cross", width=780)
    display_existing_figure("step5", "conformal_within", width=780)
    display_existing_figure("step5", "conformal_cross", width=780)
    display_existing_figure("step5", "step5_feature_family_sensitivity", width=780)
    display_existing_figure("step5", "step5_corruption_cross", width=780)
    display(noise_summary.head(12))
    display(family_drop_means)
    display(conformal_agg)
    """,
    "step7-code-step5",
))


cells.append(md(
    """
The Step 5 evidence is rich enough that several panels deserve main-paper
status. The cross-condition noise curves show that transferred performance is
stress-sensitive in a direction-dependent way rather than collapsing uniformly.
The feature-sensitivity heatmaps and family summaries sharpen that story: under
transfer, the largest average family-level drops are in **variability**
(**0.1685**) and **raw timing** (**0.1137**), whereas within-condition family
effects are generally smaller and more selective. The exploratory conformal
panels add an uncertainty-behavior layer. Averaged across domains, **APS**
retains larger mean set sizes and higher subject-level coverage than **LAC**
both within-condition and under transfer, while **LAC** becomes notably tighter
and more empty-set-prone before fallback under shift (raw empty rates rising to
about **0.34** at `alpha=0.20`). The final caveat is crucial: this entire step
is diagnostic stress testing. Gaussian feature-space perturbation is not raw
sensor noise, benchmark label corruption is not noisy-label training, and
cross-condition conformal behavior is exploratory under distribution shift.
    """,
    "step7-md-step5-note",
))


cells.append(md(
    """
## Step 6: exploratory geometry

Step 6 remains exploratory, but one geometry panel is still worth surfacing in
the synthesis notebook because it provides qualitative context for the
subject-level feature space without claiming supervised evidence.
    """,
    "step7-md-step6",
))


cells.append(code(
    """
    step6_summary = json.loads((REPO_ROOT / "experiments" / "results" / "v4" / "step6_subject_views_summary.json").read_text())
    display_existing_figure("step6", "kmeans_scatter_k3", width=780)
    display_existing_figure("step6", "pca_scatter_by_condition", width=780)
    display(pd.DataFrame(step6_summary["views"]).T[[
        "subject_count",
        "ari_against_relevant_labels",
        "seed_stability_mean_ari",
        "subject_bootstrap_stability_mean_ari",
    ]])
    """,
    "step7-code-step6",
))


cells.append(md(
    """
The exploratory geometry panels are visually informative but scientifically
limited. The **K-means overlay in PCA space** is the strongest Step 6 figure for
the manuscript because it shows that the subject-level feature space has
visible structure, yet the saved summary metrics keep the interpretation
honest: ARI against the relevant labels remains weak (**0.0260** for
`all_subjects`, **0.0135** for `disease_only`), and bootstrap stability is much
lower than seed-to-seed stability. In other words, the geometry is useful
context for transfer difficulty, but it does not prove disease separability or
replace the supervised transfer results.
    """,
    "step7-md-step6-note",
))


cells.append(md(
    """
## Step 8: control-partition sensitivity

Step 8 is now a required part of the final paper story because it tests whether
the main transfer pattern is stable when the control partition changes within
the near-optimal candidate family. The key framing is sensitivity, not
invariance.
    """,
    "step7-md-step8",
))


cells.append(code(
    """
    step8_sensitivity = pd.read_csv(TABLE_DIR / "step8_direction_sensitivity_v4.csv")
    step8_within = pd.read_csv(TABLE_DIR / "step8_within_baselines_v4.csv")
    display_existing_figure("step8", "step8_sign_stability_matrix", width=780)
    display_existing_figure("step8", "step8_direction_degradation_ranges", width=780)
    display_existing_figure("step8", "step8_within_source_baselines", width=780)
    display_existing_figure("step8", "step8_best_transfer_leaders_heatmap", width=780)
    display(step8_sensitivity)
    display(step8_within)
    """,
    "step7-code-step8",
))


cells.append(md(
    """
The Step 8 result is scientifically important precisely because it is mixed.
**Partition 2 preserves the direction-level matched-degradation signs**, but
**partition 3 reverses four of six signs**, so the control-partition sensitivity
analysis clearly **qualifies** the main transfer story rather than proving
invariance. The within-source baseline panel explains part of that sensitivity:
the best within-source **PD** baseline drops from **0.9042** in the main
partition to **0.7758** in partition 3, whereas **ALS** remains comparatively
stable. The safest paper wording is therefore that the v4 conclusions are
directionally informative but not fully insensitive to alternate near-optimal
control splits.
    """,
    "step7-md-step8-note",
))


cells.append(md(
    """
## Figure and table manifests

The manifest files below turn the notebook’s selection decisions into explicit
paper metadata. Legacy `step7/paper_fig*` image files are intentionally
excluded; the manuscript should cite the stronger step-specific source figures
directly.
    """,
    "step7-md-manifests",
))


cells.append(code(
    """
    def manifest_row(spec: dict) -> dict:
        pdf_path, png_path = figure_paths(spec["step"], spec["figure_stem"])
        return {
            **spec,
            "pdf_path": str(pdf_path.relative_to(REPO_ROOT)),
            "png_path": str(png_path.relative_to(REPO_ROOT)),
            "exists_pdf": pdf_path.exists(),
            "exists_png": png_path.exists(),
        }


    figure_manifest_df = pd.DataFrame([manifest_row(spec) for spec in FIGURE_SPECS])

    discovered_existing = []
    for png_path in sorted(FIG_PNG_ROOT.glob("step*/*.png")):
        step = png_path.parent.name
        stem = png_path.stem
        if step == "step7":
            continue
        if any((row["step"] == step and row["figure_stem"] == stem) for row in FIGURE_SPECS):
            continue
        pdf_path = FIG_PDF_ROOT / step / f"{stem}.pdf"
        discovered_existing.append({
            "tier": "supplement",
            "step": step,
            "figure_stem": stem,
            "recommended_caption": "Supplementary diagnostic figure retained for repository depth.",
            "claim_supported": "Provides detailed backup for a step-specific interpretation.",
            "why_included": "Non-redundant repository-facing detail.",
            "caveat": "Supplement only to avoid redundancy in the main paper.",
            "pdf_path": str(pdf_path.relative_to(REPO_ROOT)),
            "png_path": str(png_path.relative_to(REPO_ROOT)),
            "exists_pdf": pdf_path.exists(),
            "exists_png": png_path.exists(),
        })

    if discovered_existing:
        figure_manifest_df = pd.concat([figure_manifest_df, pd.DataFrame(discovered_existing)], ignore_index=True)

    figure_manifest_df = figure_manifest_df.sort_values(["tier", "step", "figure_stem"]).reset_index(drop=True)
    save_table_csv(figure_manifest_df, "paper_figure_manifest_v4.csv")
    save_table_latex(
        figure_manifest_df,
        "paper_figure_manifest_v4.tex",
        "Recommended v4 paper-figure manifest.",
        "tab:paper_figure_manifest_v4",
        longtable=True,
    )

    all_csv_tables = sorted(path.name for path in TABLE_DIR.glob("*.csv"))
    table_rows = list(TABLE_SPECS)
    seen_tables = {spec["table_file"] for spec in TABLE_SPECS}
    for table_name in all_csv_tables:
        if table_name in seen_tables or table_name in LEGACY_DUPLICATE_TABLES:
            continue
        if table_name.startswith("paper_"):
            continue
        table_rows.append({
            "tier": "supplement",
            "table_file": table_name,
            "purpose": "Supplementary numerical detail retained for repository and appendix use.",
            "main_columns": "See file columns.",
            "paper_use": "Supplement or appendix only.",
            "caveat": "Not every supplementary table should appear in the main manuscript.",
        })

    table_manifest_df = pd.DataFrame(table_rows).sort_values(["tier", "table_file"]).reset_index(drop=True)
    save_table_csv(table_manifest_df, "paper_table_manifest_v4.csv")
    save_table_latex(
        table_manifest_df,
        "paper_table_manifest_v4.tex",
        "Recommended v4 paper-table manifest.",
        "tab:paper_table_manifest_v4",
        longtable=True,
    )

    display(figure_manifest_df[["tier", "step", "figure_stem", "exists_pdf", "exists_png"]])
    display(table_manifest_df)
    """,
    "step7-code-manifests",
))


cells.append(md(
    """
## Final synthesis guidance for the manuscript

The curated figure and table set above supports a publication-facing story that
stays close to the frozen v4 evidence.

- **Main claim:** zero-shot abnormal-vs-control transfer is measurable under
  leakage-controlled subject-level validation, but it is directionally
  asymmetric and usually degraded relative to within-source subject baselines.
- **Safe Step 3 wording:** direction-level mean subject-level matched
  degradation is positive across all six directions, while individual
  classifier-direction pairs still vary.
- **Safe Step 4 wording:** SHAP supports a diagnostic reliance-shift story
  centered on variability, raw timing, and a small set of recurring
  timing-structure features; it does **not** establish causal biomarkers.
- **Safe Step 5 wording:** the robustness and conformal layers are diagnostic
  stress tests and exploratory uncertainty analyses, not raw-sensor realism or
  target-domain coverage guarantees.
- **Safe Step 6 wording:** PCA and K-means provide exploratory geometry context
  only.
- **Safe Step 8 wording:** alternate near-optimal control partitions preserve
  some conclusions but materially weaken others, because partition 3 reverses
  four of six direction-level matched-degradation signs.
- **Global caveat:** small subject counts remain the main limit on stronger
  inferential claims.
    """,
    "step7-md-final",
))


cells.append(code(
    """
    expected_step7_outputs = [
        TABLE_DIR / "paper_figure_manifest_v4.csv",
        TABLE_DIR / "paper_figure_manifest_v4.tex",
        TABLE_DIR / "paper_table_manifest_v4.csv",
        TABLE_DIR / "paper_table_manifest_v4.tex",
        TABLE_DIR / "paper_main_results_summary_v4.csv",
        TABLE_DIR / "paper_main_results_summary_v4.tex",
    ]
    for path in expected_step7_outputs:
        assert path.exists(), path

    for row in figure_manifest_df.itertuples():
        if row.tier in {"main", "extended_main"}:
            assert row.exists_pdf, row.pdf_path
            assert row.exists_png, row.png_path

    assert all(spec["step"] != "step7" for spec in FIGURE_SPECS)

    print("PASS: Step 7 synthesis notebook curated existing v4 figures and wrote the paper manifests.")
    """,
    "step7-code-verify",
))


payload = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote {OUT}")
