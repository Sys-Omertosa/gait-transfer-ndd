"""
Corrected figures for the long-form preprint.

Five figures from the original manuscript either plotted a superseded quantity
or carried a label that misstated the endpoint. This script redraws them from
the frozen result envelopes and derived tables only. Nothing is refitted,
re-explained, re-permuted, or re-clustered; every plotted value is read from a
stored artifact.

    family_movement.pdf        family-level SHAP movement using total_movement
    noise_curves_cross.pdf     Gaussian stress labelled as stride-level macro-F1
    family_sensitivity.pdf     summed marginal and joint family permutation drops
    transfer_asymmetry.pdf     asymmetry under both direction summaries
    subject_geometry.pdf       subject-level PCA and K-means agreement across k

Run from the repository root:
    python report/figures/arxiv/make_arxiv_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
from figstyle import draw_heatmap  # noqa: E402

RESULTS = REPO / 'experiments' / 'results' / 'v4'
TABLES = REPO / 'report' / 'tables' / 'v4'
OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ONE_COL = 3.5
TWO_COL = 7.16
DIRECTIONS = ['pd_to_hd', 'hd_to_pd', 'pd_to_als', 'als_to_pd', 'hd_to_als', 'als_to_hd']
DIR_LABEL = {d: d.replace('_to_', '→').upper() for d in DIRECTIONS}
CLF = ['rf', 'knn', 'svm', 'dt', 'qda', 'xgb', 'lgbm']
CLF_LABEL = {'rf': 'RF', 'knn': 'KNN', 'svm': 'SVM', 'dt': 'DT', 'qda': 'QDA',
             'xgb': 'XGBoost', 'lgbm': 'LightGBM'}
FAMILIES = ['variability', 'raw_timing', 'fractal', 'phase_percentage', 'asymmetry']
FAMILY_LABEL = {f: f.replace('_', ' ') for f in FAMILIES}
FAMILY_SIZE = {'variability': 2, 'raw_timing': 7, 'fractal': 1,
               'phase_percentage': 3, 'asymmetry': 1}
COND_COLOR = {'pd': '#4C78A8', 'hd': '#F58518', 'als': '#54A24B', 'control': '#7F7F7F'}
COND_LABEL = {'pd': 'PD', 'hd': 'HD', 'als': 'ALS', 'control': 'Control'}
AXIS_GREY = '#8C959D'


def _style_axes(ax, *, left: bool = True) -> None:
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS_GREY)
        ax.spines[side].set_linewidth(0.7)
    if not left:
        ax.spines['left'].set_visible(False)
    ax.tick_params(length=3, pad=2)


def _save(fig, stem: str) -> None:
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'wrote {stem}.pdf')


# ── Family-level SHAP movement ───────────────────────────────────────────────
def family_movement() -> dict[str, float]:
    """
    Heatmap of mean family-level total reliance movement per transfer direction.

    total_movement sums the absolute feature-level changes in mean absolute SHAP
    magnitude inside a family, so opposing movements of two features in the same
    family cannot cancel. The earlier figure plotted the net shift, which allows
    that cancellation. Each cell averages the seven classifier families; the
    separate right-hand column averages all 42 classifier-direction pairs.

    Returns:
        Overall mean total movement per family, for cross-checking the text.
    """
    shap = json.loads((RESULTS / 'shap_results_v4.json').read_text())['data']
    grid = np.array([[np.mean([shap[d][k]['family_delta_j'][f]['total_movement'] for k in CLF])
                      for d in DIRECTIONS] for f in FAMILIES])
    overall = np.array([[np.mean([shap[d][k]['family_delta_j'][f]['total_movement']
                                  for d in DIRECTIONS for k in CLF])] for f in FAMILIES])

    cmap = sns.light_palette('#6C3483', as_cmap=True)
    norm = Normalize(vmin=0.0, vmax=float(grid.max()))
    left, cell_w, row_h, gap, cbar_gap, cbar_w = 1.25, 0.72, 0.30, 0.18, 0.14, 0.12
    top, title_h, xtick_h, bottom = 0.04, 0.30, 0.28, 0.04
    plot_h = row_h * len(FAMILIES)
    fig_w = left + cell_w * 7 + gap + cbar_gap + cbar_w + 0.42
    fig_h = top + title_h + plot_h + xtick_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    y0 = (bottom + xtick_h) / fig_h

    ax = fig.add_axes((left / fig_w, y0, cell_w * 6 / fig_w, plot_h / fig_h))
    draw_heatmap(ax, grid, cmap=cmap, norm=norm, fmt='{:.3f}', annot_size=8.0)
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_xticklabels([DIR_LABEL[d] for d in DIRECTIONS], fontsize=8.5)
    ax.set_yticks(np.arange(len(FAMILIES)) + 0.5)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=8.5)
    ax.tick_params(length=0, pad=3)

    ax_all = fig.add_axes(((left + cell_w * 6 + gap) / fig_w, y0, cell_w / fig_w, plot_h / fig_h))
    draw_heatmap(ax_all, overall, cmap=cmap, norm=norm, fmt='{:.4f}', annot_size=8.0)
    ax_all.set_xticks([0.5])
    ax_all.set_xticklabels(['All 42'], fontsize=8.5)
    ax_all.set_yticks([])
    ax_all.tick_params(length=0, pad=3)

    cax = fig.add_axes(((left + cell_w * 7 + gap + cbar_gap) / fig_w, y0, cbar_w / fig_w, plot_h / fig_h))
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cbar.ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor('#9AA0A6')

    fig.text((left + (cell_w * 7 + gap) / 2) / fig_w, (fig_h - top - title_h / 2) / fig_h,
             'Mean family-level total movement of SHAP reliance',
             ha='center', va='center', fontsize=10)
    _save(fig, 'family_movement')
    return {f: float(overall[i, 0]) for i, f in enumerate(FAMILIES)}


# ── Gaussian feature-space stress ────────────────────────────────────────────
def noise_curves_cross() -> None:
    """
    Cross-condition stride-level macro-F1 under pooled-relative Gaussian noise.

    The robustness suite scores raw stride rows without subject aggregation, so
    these curves are stride-level. Noise is scaled by the standard deviation of
    each feature over the pooled source and target rows. Each point averages the
    30 stored repetitions (a single evaluation at sigma = 0).
    """
    noise = json.loads((RESULTS / 'noise_robustness_v4.json').read_text())['data']['cross']
    sigma_keys = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.5']
    sigmas = [float(s) for s in sigma_keys]
    colors = {'rf': '#4C78A8', 'knn': '#F58518', 'svm': '#54A24B', 'dt': '#E45756',
              'qda': '#B279A2', 'xgb': '#72B7B2', 'lgbm': '#9D7660'}
    markers = {'rf': 'o', 'knn': 's', 'svm': '^', 'dt': 'D', 'qda': 'v', 'xgb': 'P', 'lgbm': 'X'}

    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL, 4.0), sharex=True, sharey=True)
    for idx, direction in enumerate(DIRECTIONS):
        ax = axes[idx // 3][idx % 3]
        block = noise[direction]['pooled_relative']
        for k in CLF:
            ax.plot(sigmas, [float(np.mean(block[k][s])) for s in sigma_keys], color=colors[k],
                    marker=markers[k], markersize=3.0, markeredgewidth=0.0, linewidth=1.1,
                    label=CLF_LABEL[k])
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(-0.02, 0.52)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(DIR_LABEL[direction], fontsize=9, pad=3)
        ax.grid(color='#E5E8EB', linewidth=0.6)
        ax.set_axisbelow(True)
        _style_axes(ax)
    fig.supylabel('Cross-condition stride-level macro-$F_1$', fontsize=9, x=0.015)
    fig.supxlabel('Noise $\\sigma$ (multiples of the pooled source–target feature standard deviation)',
                  fontsize=9, y=0.055)
    fig.suptitle('Cross-condition Gaussian feature-space stress', fontsize=10, y=0.985)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center', ncol=7,
               frameon=False, bbox_to_anchor=(0.5, -0.035), handlelength=1.6,
               handletextpad=0.5, columnspacing=1.3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.17, hspace=0.32, wspace=0.08)
    _save(fig, 'noise_curves_cross')


# ── Permutation sensitivity by family ────────────────────────────────────────
def family_sensitivity() -> dict[str, tuple[float, float]]:
    """
    Cross-condition permutation drops by family under two aggregation rules.

    The summed marginal value permutes each feature on its own and adds the
    resulting drops within a family. The joint value permutes all features of a
    family together with one shared row permutation. Both are drops in stride-
    level macro-F1 averaged over the 42 classifier-direction pairs. Family sizes
    are printed with the labels because a larger family can accumulate a larger
    summed drop. The within-condition values are not drawn: within-condition
    permutation shuffles rows inside a single held-out subject, where the three
    subject-level descriptors are constant, so their drops are zero by
    construction rather than by measurement.

    Returns:
        Mapping from family to (summed marginal drop, joint drop).
    """
    marginal = pd.read_csv(TABLES / 'step5_feature_family_sensitivity_v4.csv')
    marginal = marginal[marginal.scope == 'cross'].groupby('family')['drop'].mean()
    sens = json.loads((RESULTS / 'feature_sensitivity_v4.json').read_text())['data']['cross']
    joint = {f: float(np.mean([sens[d][k]['joint_family_permutation_drop'][f]
                               for d in DIRECTIONS for k in CLF])) for f in FAMILIES}

    order = sorted(FAMILIES, key=lambda f: -marginal[f])
    rows = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.35))
    series = (('Summed marginal (one feature at a time)', [marginal[f] for f in order], '#1F5A8C', -0.19),
              ('Joint (whole family permuted together)', [joint[f] for f in order], '#9EC3E6', 0.19))
    for label, values, color, offset in series:
        ax.barh(rows + offset, values, height=0.36, color=color, edgecolor='white',
                linewidth=0.5, label=label, zorder=2)
        for r, v in zip(rows, values):
            ax.text(v + 0.003, r + offset, f'{v:.3f}', va='center', ha='left', fontsize=6.8,
                    color='#2F2F2F', zorder=3)
    ax.set_yticks(rows)
    ax.set_yticklabels([f'{FAMILY_LABEL[f]} ({FAMILY_SIZE[f]})' for f in order], fontsize=8)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlim(0, 0.2)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel('Drop in cross-condition stride-level macro-$F_1$', fontsize=8.5, labelpad=3)
    ax.grid(axis='x', color='#E5E8EB', linewidth=0.6)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.42, 1.0), ncol=1, frameon=False,
              fontsize=7, handlelength=1.2, borderpad=0.0, labelspacing=0.25)
    _save(fig, 'family_sensitivity')
    return {f: (float(marginal[f]), joint[f]) for f in FAMILIES}


# ── Directional asymmetry ────────────────────────────────────────────────────
def transfer_asymmetry() -> dict[str, tuple[float, float]]:
    """
    Forward-minus-reverse matched degradation for each disease pair.

    Two direction summaries are compared: the unweighted mean over the seven
    classifier families and the retrospective direction leader, the classifier
    with the highest cross-condition subject-level macro-F1. A negative value
    means the forward direction degraded less. The two summaries disagree for
    HD and ALS, which is why both are shown.

    Returns:
        Mapping from pair label to (classifier-average gap, leader gap).
    """
    cross = json.loads((RESULTS / 'cross_condition_results_v4.json').read_text())

    def average(direction):
        return float(np.mean([cross[direction]['classifiers'][k]['delta_f1_subject'] for k in CLF]))

    def leader(direction):
        pool = cross[direction]['classifiers']
        best = max(CLF, key=lambda k: (pool[k]['subject_primary_f1_macro'], pool[k]['f1_macro']))
        return float(pool[best]['delta_f1_subject'])

    pairs = (('pd_to_hd', 'hd_to_pd'), ('pd_to_als', 'als_to_pd'), ('hd_to_als', 'als_to_hd'))
    labels = [f'{DIR_LABEL[a]} vs {DIR_LABEL[b]}' for a, b in pairs]
    avg_gap = [average(a) - average(b) for a, b in pairs]
    lead_gap = [leader(a) - leader(b) for a, b in pairs]

    rows = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.95))
    for label, values, color, offset in (('Classifier average', avg_gap, '#4C78A8', -0.18),
                                         ('Retrospective leader', lead_gap, '#F58518', 0.18)):
        ax.barh(rows + offset, values, height=0.34, color=color, edgecolor='white',
                linewidth=0.5, label=label, zorder=2)
        for r, v in zip(rows, values):
            pad = 0.004 if v >= 0 else -0.004
            ax.text(v + pad, r + offset, f'{v:+.3f}', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=6.8, color='#2F2F2F', zorder=3)
    ax.axvline(0.0, color='#5D6D7E', linewidth=0.9, zorder=3)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(len(pairs) - 0.45, -0.55)
    ax.set_xlim(-0.17, 0.08)
    ax.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05])
    ax.set_xlabel('Forward minus reverse matched degradation', fontsize=8.5, labelpad=3)
    _style_axes(ax, left=False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False,
              fontsize=7.2, handlelength=1.2, borderpad=0.0, columnspacing=1.0)
    _save(fig, 'transfer_asymmetry')
    return dict(zip(labels, zip(avg_gap, lead_gap)))


# ── Subject-level geometry ───────────────────────────────────────────────────
def subject_geometry() -> dict[str, float]:
    """
    Subject-level PCA projection and K-means label agreement across k.

    Panel (a) plots the stored first two principal-component coordinates of the
    63 subjects after z-scoring across subjects. Panel (b) plots the stored
    adjusted Rand index between K-means assignments and the reference labels for
    k = 2 to 10, and marks the k chosen by the silhouette criterion. The cluster
    assignments themselves are not stored, so no cluster overlay is drawn.

    Returns:
        Explained variance of the first two components, for cross-checking.
    """
    summary = json.loads((RESULTS / 'step6_subject_views_summary.json').read_text())['views']
    coords = pd.read_csv(TABLES / 'step6_all_subjects_pca_coordinates.csv')
    evr = summary['all_subjects']['pca_explained_variance']

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TWO_COL, 2.55),
                                     gridspec_kw={'width_ratios': [1.0, 1.15], 'wspace': 0.3})
    for cond in ('control', 'pd', 'hd', 'als'):
        block = coords[coords.label == cond]
        ax_a.scatter(block.pc1, block.pc2, s=16, color=COND_COLOR[cond], edgecolor='white',
                     linewidth=0.4, label=f'{COND_LABEL[cond]} ({len(block)})', zorder=2)
    ax_a.set_xlabel(f'PC1 ({evr[0] * 100:.1f}% of variance)', fontsize=8.5)
    ax_a.set_ylabel(f'PC2 ({evr[1] * 100:.1f}%)', fontsize=8.5)
    ax_a.set_title('(a) Subject-level PCA', fontsize=9, pad=4)
    ax_a.grid(color='#E5E8EB', linewidth=0.6)
    ax_a.set_axisbelow(True)
    _style_axes(ax_a)
    ax_a.legend(frameon=False, fontsize=7, loc='lower left', handletextpad=0.2, borderpad=0.2)

    views = (('disease_only', 'Disease only (PD, HD, ALS)', '#6C3483', 'o'),
             ('all_subjects', 'All subjects (four labels)', '#1F5A8C', 's'),
             ('binary_pathological_vs_control', 'Pathological vs control', '#B8860B', '^'))
    for key, label, color, marker in views:
        metrics = pd.read_csv(TABLES / f'step6_{key}_kmeans_metrics.csv')
        best_k = int(metrics.loc[metrics.silhouette_score.idxmax(), 'k'])
        ax_b.plot(metrics.k, metrics.ari_against_relevant_labels, color=color, marker=marker,
                  markersize=3.5, linewidth=1.1, label=label, zorder=2)
        best = metrics[metrics.k == best_k].iloc[0]
        ax_b.scatter([best_k], [best.ari_against_relevant_labels], s=62, facecolor='none',
                     edgecolor=color, linewidth=1.0, zorder=3)
    ax_b.axhline(0.0, color='#5D6D7E', linewidth=0.8, zorder=1)
    ax_b.set_xticks(range(2, 11))
    ax_b.set_xlabel('Number of K-means clusters $k$', fontsize=8.5)
    ax_b.set_ylabel('Adjusted Rand index', fontsize=8.5)
    ax_b.set_title('(b) K-means agreement with reference labels', fontsize=9, pad=4)
    ax_b.set_ylim(-0.14, 0.37)
    ax_b.grid(color='#E5E8EB', linewidth=0.6)
    ax_b.set_axisbelow(True)
    _style_axes(ax_b)
    ax_b.legend(frameon=False, fontsize=7, loc='upper right', handletextpad=0.4)
    _save(fig, 'subject_geometry')
    return {'pc1': evr[0], 'pc2': evr[1], 'pc1_pc2': evr[0] + evr[1]}


if __name__ == '__main__':
    print('family total movement:', {k: round(v, 4) for k, v in family_movement().items()})
    noise_curves_cross()
    print('family sensitivity (marginal, joint):',
          {k: tuple(round(x, 4) for x in v) for k, v in family_sensitivity().items()})
    print('asymmetry (average gap, leader gap):',
          {k: tuple(round(x, 4) for x in v) for k, v in transfer_asymmetry().items()})
    print('subject PCA:', {k: round(v, 4) for k, v in subject_geometry().items()})
