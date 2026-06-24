"""
Static notebook contract checks for the authoritative v4 notebook bundle.

Usage:
    python scripts/verification/test_v4_notebook_contracts.py
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def notebook_source(name: str) -> str:
    payload = json.loads((NOTEBOOK_DIR / name).read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))


def require_all(name: str, snippets: list[str], source: str, failures: list[str]) -> None:
    for snippet in snippets:
        if snippet not in source:
            failures.append(f"{name}: missing required snippet {snippet!r}")


def forbid_all(name: str, snippets: list[str], source: str, failures: list[str]) -> None:
    for snippet in snippets:
        if snippet in source:
            failures.append(f"{name}: forbidden stale snippet {snippet!r} still present")


def require_any(name: str, groups: list[list[str]], source: str, failures: list[str]) -> None:
    for group in groups:
        if not any(snippet in source for snippet in group):
            failures.append(f"{name}: missing any of {group!r}")


def require_bootstrap(name: str, source: str, failures: list[str]) -> None:
    require_all(
        name,
        [
            "find_repo_root",
            "REPO_ROOT = find_repo_root()",
            "if str(REPO_ROOT) not in sys.path",
        ],
        source,
        failures,
    )
    require_any(
        name,
        [
            ["if str(REPO_ROOT / 'src') not in sys.path", 'if str(REPO_ROOT / "src") not in sys.path'],
        ],
        source,
        failures,
    )


def main() -> None:
    failures: list[str] = []

    stale_v3 = [
        "results_v3",
        "gait_features_v3",
        "control_partition_v3",
        "preprocessing_manifest_v3",
        "cross_condition_results_v3.json",
        "shap_results_v3.json",
        "noise_robustness_v3.json",
        "feature_sensitivity_v3.json",
        "subject_sensitivity_v3.json",
        "corruption_robustness_v3.json",
        "conformal_v3.json",
    ]

    nb00 = notebook_source("00_data_exploration.ipynb")
    require_bootstrap("00_data_exploration.ipynb", nb00, failures)
    require_all(
        "00_data_exploration.ipynb",
        [
            "preprocessing_manifest_v4.json",
            "step0_raw_cohort_composition",
            "step0_authoritative_filtering_cascade",
            "step0_raw_cohort_summary.csv",
            "step0_filter_flow.csv",
            "filter_artifact_rows_v3",
            "implementation name only",
        ],
        nb00,
        failures,
    )

    nb01 = notebook_source("01_preprocessing.ipynb")
    require_bootstrap("01_preprocessing.ipynb", nb01, failures)
    forbid_all("01_preprocessing.ipynb", stale_v3, nb01, failures)
    require_all(
        "01_preprocessing.ipynb",
        [
            "data' / 'processed' / 'v4",
            "v4_protocol_manifest.json",
            "preprocessing_manifest_v4.json",
            "gait_features_v4.csv",
            "gait_features_v4_subject_level.csv",
            "gait_features_v4_per_stride_only.csv",
            "gait_features_v4_dfa_sensitivity.csv",
            "gait_features_v4_no_dfa_sensitivity.csv",
            "gait_features_v4_timing_sensitivity.csv",
            "control_partition_v4.json",
            "subject_probability_loss_then_lexicographic",
        ],
        nb01,
        failures,
    )

    nb02 = notebook_source("02_within_condition.ipynb")
    require_bootstrap("02_within_condition.ipynb", nb02, failures)
    forbid_all("02_within_condition.ipynb", stale_v3 + ["0.9026", "0.9546", "0.9692"], nb02, failures)
    require_all(
        "02_within_condition.ipynb",
        [
            "results' / 'v4",
            "subject_primary_f1_macro",
            "subject_log_loss",
            "subject_brier",
            "subject_ece10",
            "f1_within_condition_heatmap",
            "step2_subject_stride_leaders",
            "step2_best_classifier_confusion_matrices",
        ],
        nb02,
        failures,
    )

    nb03 = notebook_source("03_cross_condition.ipynb")
    require_bootstrap("03_cross_condition.ipynb", nb03, failures)
    forbid_all(
        "03_cross_condition.ipynb",
        stale_v3 + ["Wilcoxon", "Holm", "RF for PD/HD-source", "KNN for ALS-source"],
        nb03,
        failures,
    )
    require_all(
        "03_cross_condition.ipynb",
        [
            "cross_condition_results_v4.json",
            "delta_f1_subject",
            "within-condition source baseline - cross-condition transfer subject F1",
            "subject_primary_f1_ci_lower",
            "degradation_heatmap",
            "cross_condition_f1_heatmap",
            "step3_best_direction_heatmap",
            "step3_bidirectional_asymmetry",
            "step3_direction_intervals",
            "step3_subject_confusion_matrices",
        ],
        nb03,
        failures,
    )

    nb04 = notebook_source("04_shap_diagnosis.ipynb")
    require_bootstrap("04_shap_diagnosis.ipynb", nb04, failures)
    forbid_all("04_shap_diagnosis.ipynb", stale_v3, nb04, failures)
    require_all(
        "04_shap_diagnosis.ipynb",
        [
            "shap_results_v4.json",
            "CACHE_DIR = REPO_ROOT / 'experiments' / 'shap' / 'v4'",
            "['data']",
            "__consensus__",
            "delta_j_normalized",
            "step4_delta_j_rf_heatmap",
            "step4_delta_j_normalized_consensus_heatmap",
            "step4_family_delta_j_consensus",
            "delta_j_vs_degradation",
            "waterfall_pd_to_hd_case_",
            "diagnostic",
        ],
        nb04,
        failures,
    )

    nb05 = notebook_source("05_noise_robustness.ipynb")
    require_bootstrap("05_noise_robustness.ipynb", nb05, failures)
    forbid_all("05_noise_robustness.ipynb", stale_v3, nb05, failures)
    require_all(
        "05_noise_robustness.ipynb",
        [
            "noise_robustness_v4.json",
            "feature_sensitivity_v4.json",
            "subject_sensitivity_v4.json",
            "corruption_robustness_v4.json",
            "conformal_v4.json",
            "gaussian_feature_space_stress_test",
            "engineered_feature_masking",
            "engineered_feature_gain_bias_drift",
            "gaussian_feature_space_jitter",
            "evaluation_row_dropout",
            "benchmark_label_corruption",
            "hard_label_consensus_curve",
            "['data']",
        ],
        nb05,
        failures,
    )

    nb06 = notebook_source("06_pca_kmeans.ipynb")
    require_bootstrap("06_pca_kmeans.ipynb", nb06, failures)
    require_all(
        "06_pca_kmeans.ipynb",
        [
            "gait_features_v4_subject_level.csv",
            "step6_subject_views_summary.json",
            "disease_only",
            "all_subjects",
            "binary_pathological_vs_control",
            "control_a_vs_control_b",
            "subject_count",
            "active_feature_list",
            "standardization_rule",
            "pca_explained_variance",
            "pca_coordinates_path",
            "k_range_evaluated",
            "silhouette_score",
            "inertia",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "ari_against_relevant_labels",
            "seed_stability_mean_ari",
            "subject_bootstrap_stability_mean_ari",
            "cluster_composition",
            "cautious_interpretation_metadata",
            "protocol_manifest_sha256",
            "preprocessing_manifest_sha256",
            "downstream_execution_manifest_sha256",
            "subject_level_matrix_sha256",
            "notebook_source_sha256",
            "package_versions",
            "random_seeds",
        ],
        nb06,
        failures,
    )

    nb07 = notebook_source("07_final_figures.ipynb")
    require_bootstrap("07_final_figures.ipynb", nb07, failures)
    forbid_all("07_final_figures.ipynb", stale_v3 + ["placeholder", "paper_v4_fig"], nb07, failures)
    require_all(
        "07_final_figures.ipynb",
        [
            "# Step 7: Final Results Synthesis and Paper Figure Selection",
            "display_existing_figure",
            "paper_figure_manifest_v4.csv",
            "paper_figure_manifest_v4.tex",
            "paper_table_manifest_v4.csv",
            "paper_table_manifest_v4.tex",
            "paper_main_results_summary_v4.csv",
            "paper_main_results_summary_v4.tex",
            "step0_authoritative_filtering_cascade",
            "step1_control_split_summary",
            "f1_within_condition_heatmap",
            "step2_subject_stride_leaders",
            "degradation_heatmap",
            "cross_condition_f1_heatmap",
            "step3_subject_confusion_matrices",
            "step4_delta_j_rf_heatmap",
            "step4_delta_j_normalized_consensus_heatmap",
            "step4_delta_j_spearman",
            "noise_curves_cross",
            "step5_feature_sensitivity_within",
            "step5_feature_sensitivity_cross",
            "conformal_within",
            "conformal_cross",
            "kmeans_scatter_k3",
            "step8_sign_stability_matrix",
            "step8_direction_degradation_ranges",
            "study-design schematic will be handled directly in the manuscript source",
            "Cross-condition transfer summary (subject-primary columns emphasized)",
            "Manuscript selection criteria",
        ],
        nb07,
        failures,
    )
    require_any(
        "07_final_figures.ipynb",
        [
            ['FIG_PDF_ROOT = REPO_ROOT / "report" / "figures" / "v4" / "pdf"', "FIG_PDF_ROOT = REPO_ROOT / 'report' / 'figures' / 'v4' / 'pdf'"],
            ['FIG_PNG_ROOT = REPO_ROOT / "report" / "figures" / "v4" / "png"', "FIG_PNG_ROOT = REPO_ROOT / 'report' / 'figures' / 'v4' / 'png'"],
            ['TABLE_DIR = REPO_ROOT / "report" / "tables" / "v4"', "TABLE_DIR = REPO_ROOT / 'report' / 'tables' / 'v4'"],
        ],
        nb07,
        failures,
    )
    forbid_all(
        "07_final_figures.ipynb",
        [
            "save_step7_figure",
            "paper_fig1_study_design",
            "save_figure(fig, \"paper_fig2_within_condition\")",
            "save_figure(fig, \"paper_fig3_cross_transfer\")",
            "save_figure(fig, \"paper_fig4_transfer_diagnostics\")",
            "save_figure(fig, \"paper_fig5_robustness\")",
            "save_figure(fig, \"paper_fig6_control_split_sensitivity\")",
        ],
        nb07,
        failures,
    )

    nb08 = notebook_source("08_sensitivity_analysis.ipynb")
    require_bootstrap("08_sensitivity_analysis.ipynb", nb08, failures)
    forbid_all("08_sensitivity_analysis.ipynb", stale_v3 + ["control_split_sensitivity_diverse"], nb08, failures)
    require_all(
        "08_sensitivity_analysis.ipynb",
        [
            "control_split_sensitivity/sensitivity_summary_v4.json",
            "control_split_sensitivity_runs",
            "candidate_family",
            "is_main_partition",
            "is_role_reversal_of_main",
            "overlap_with_main_control_a",
            "independent_sensitivity_candidate",
            "pending",
        ],
        nb08,
        failures,
    )

    if failures:
        raise SystemExit("Notebook contract failures:\\n" + "\\n".join(failures))

    print(json.dumps(
        {
            "status": "pass",
            "checked_notebooks": [
                "00_data_exploration.ipynb",
                "01_preprocessing.ipynb",
                "02_within_condition.ipynb",
                "03_cross_condition.ipynb",
                "04_shap_diagnosis.ipynb",
                "05_noise_robustness.ipynb",
                "06_pca_kmeans.ipynb",
                "07_final_figures.ipynb",
                "08_sensitivity_analysis.ipynb",
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
