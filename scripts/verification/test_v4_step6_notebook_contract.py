"""
Static contract check for the v4 Step 6 subject-level notebook.

Usage:
    python scripts/verification/test_v4_step6_notebook_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / 'notebooks' / '06_pca_kmeans.ipynb'


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source_text = '\n'.join(
        ''.join(cell.get('source', []))
        for cell in notebook.get('cells', [])
    )
    required_snippets = [
        'gait_features_v4_subject_level.csv',
        'step6_subject_views_summary.json',
        'disease_only',
        'all_subjects',
        'binary_pathological_vs_control',
        'control_a_vs_control_b',
        'subject_count',
        'active_feature_list',
        'standardization_rule',
        'pca_explained_variance',
        'pca_coordinates_path',
        'k_range_evaluated',
        'silhouette_score',
        'inertia',
        'calinski_harabasz_score',
        'davies_bouldin_score',
        'ari_against_relevant_labels',
        'seed_stability_mean_ari',
        'subject_bootstrap_stability_mean_ari',
        'cluster_composition',
        'cautious_interpretation_metadata',
        'protocol_manifest_sha256',
        'preprocessing_manifest_sha256',
        'downstream_execution_manifest_sha256',
        'subject_level_matrix_sha256',
        'notebook_source_sha256',
        'package_versions',
        'random_seeds',
    ]
    missing = [snippet for snippet in required_snippets if snippet not in source_text]
    if missing:
        raise SystemExit(
            'Step 6 notebook is missing required subject-view contract snippets:\n'
            + '\n'.join(missing)
        )

    print(json.dumps({
        'status': 'pass',
        'views': [
            'disease_only',
            'all_subjects',
            'binary_pathological_vs_control',
            'control_a_vs_control_b',
        ],
        'summary_json': 'experiments/results/v4/step6_subject_views_summary.json',
    }, indent=2))


if __name__ == '__main__':
    main()
