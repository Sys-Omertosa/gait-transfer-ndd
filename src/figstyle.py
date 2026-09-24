"""
Shared styling helpers for the manuscript figure set.

Every manuscript figure is authored at its final printed width and placed at 1:1
scale, so a type size chosen here is the type size that reaches the page. These
helpers keep the three things that must not drift between notebooks identical:
annotation legibility on coloured cells, the numeric format used for bounded
quantities, and the canonical ordering of the six directed transfers.
"""

from __future__ import annotations

import matplotlib.colors as mcolors

# ── Printed widths ───────────────────────────────────────────────────────────
# IEEEtran conference class: \columnwidth = 252 pt, \textwidth = 516 pt.
IEEE_ONE_COL = 3.5
IEEE_TWO_COL = 7.16

# Pure black rather than a soft near-black: the two inks meet at cell luminance
# 0.179, where each still reaches 4.58:1. A near-black such as #1A1A1A moves that
# meeting point to 0.202, where the best available ink only reaches 4.17:1, so
# some cell shade would necessarily fall below the readability threshold.
INK_DARK = '#000000'
INK_LIGHT = '#FFFFFF'

WCAG_AA_NORMAL = 4.5

# ── Canonical orderings ──────────────────────────────────────────────────────
# Pivoting a long frame on 'direction' yields alphabetical columns, which puts
# the transfers in an order that matches no other figure in the manuscript.
# Reindex against this sequence after every pivot.
DIRECTION_ORDER = [
    'pd_to_hd', 'hd_to_pd', 'pd_to_als', 'als_to_pd', 'hd_to_als', 'als_to_hd',
]
DIRECTION_LABEL = {name: name.replace('_to_', '→').upper() for name in DIRECTION_ORDER}

# Feature families are referenced by more than one manuscript figure, so the
# ordering and the colour assignment are fixed here rather than per notebook.
FAMILY_ORDER = ['variability', 'raw_timing', 'fractal', 'phase_percentage', 'asymmetry']
FAMILY_COLOR = {
    'variability': '#4C78A8',
    'raw_timing': '#F58518',
    'fractal': '#54A24B',
    'phase_percentage': '#B279A2',
    'asymmetry': '#8C6D4F',
}
FAMILY_LABEL = {name: name.replace('_', ' ') for name in FAMILY_ORDER}


def relative_luminance(color) -> float:
    """
    Relative luminance of a colour on linearised sRGB channels.

    Args:
        color: Any Matplotlib colour specification.
    Returns:
        Luminance in [0, 1] following the WCAG definition.
    """
    red, green, blue = mcolors.to_rgb(color)
    channels = [
        value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        for value in (red, green, blue)
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(first, second) -> float:
    """
    WCAG contrast ratio between two colours.

    Args:
        first: Any Matplotlib colour specification.
        second: Any Matplotlib colour specification.
    Returns:
        Ratio in [1, 21]; normal body text needs at least 4.5.
    """
    lighter = max(relative_luminance(first), relative_luminance(second))
    darker = min(relative_luminance(first), relative_luminance(second))
    return (lighter + 0.05) / (darker + 0.05)


def annotation_ink(facecolor) -> str:
    """
    Pick the annotation colour that maximises contrast against a filled cell.

    Seaborn's built-in rule switches to white text once cell luminance falls to
    0.408, but the crossover against its dark ink is near 0.220. Cells between
    those two points therefore receive white text at roughly 2.3:1 to 3.9:1,
    below the WCAG AA threshold for normal text. Choosing by measured contrast
    instead keeps every annotation legible regardless of the colour map.

    Args:
        facecolor: Any Matplotlib colour specification for the cell fill.
    Returns:
        Hex colour for the annotation text.
    """
    if contrast_ratio(facecolor, INK_DARK) >= contrast_ratio(facecolor, INK_LIGHT):
        return INK_DARK
    return INK_LIGHT


def draw_heatmap(
    ax,
    values,
    *,
    cmap,
    norm,
    fmt: str = '{:.3f}',
    annot_size: float = 8.0,
    linewidth: float = 0.8,
    linecolor: str = 'white',
    outline: str | None = '#9AA0A6',
):
    """
    Render a value matrix as an annotated heatmap on an existing axes.

    Cells are drawn as individual patches so that each annotation can take the
    ink that maximises its own contrast, rather than a single threshold applied
    to the whole matrix. The axes is left with data coordinates running from 0 to
    the column and row counts, so callers place ticks at index + 0.5.

    Args:
        ax: Target Matplotlib axes.
        values: Two-dimensional array of cell values, row-major.
        cmap: Colour map applied through norm.
        norm: Normalisation mapping values onto the colour map.
        fmt: Format string applied to each cell value.
        annot_size: Annotation font size in points.
        linewidth: Width of the separating grid lines.
        linecolor: Colour of the separating grid lines.
        outline: Colour of the border drawn around the block, or None.
    Returns:
        List of annotation Text artists in row-major order.
    """
    import numpy as np
    from matplotlib.patches import Rectangle

    grid = np.asarray(values, dtype=float)
    n_rows, n_cols = grid.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    texts = []
    for row in range(n_rows):
        for col in range(n_cols):
            face = cmap(norm(grid[row, col]))
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face,
                                   edgecolor=linecolor, linewidth=linewidth, zorder=2))
            texts.append(ax.text(
                col + 0.5, row + 0.5, fmt.format(grid[row, col]),
                ha='center', va='center', fontsize=annot_size,
                color=annotation_ink(face), zorder=3,
            ))
    if outline is not None:
        ax.add_patch(Rectangle((0, 0), n_cols, n_rows, facecolor='none',
                               edgecolor=outline, linewidth=0.7, zorder=4))
    return texts


def format_correlation(value: float) -> str:
    """
    Format a quantity bounded on [-1, 1] at two decimals without a leading zero.

    The units digit is informative only at the extremes, so suppressing it
    shortens the widest annotation by one glyph and keeps dense matrices inside
    their cell boundaries.

    Args:
        value: Correlation or other unit-bounded quantity.
    Returns:
        Annotation string using a typographic minus sign.
    """
    text = f'{value:.2f}'
    if text.startswith('0.'):
        return text[1:]
    if text.startswith('-0.'):
        return '−' + text[2:]
    return text.replace('-', '−')
