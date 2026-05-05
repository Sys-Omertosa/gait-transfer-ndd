# LaTeX Beamer presentation (IEEE-style)

This folder contains a **professional Beamer** slide deck for the gait transfer paper. It uses the **Madrid** theme with a conservative blue structure color suitable for engineering seminars and IEEE-flavored talks.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full presentation source |
| `Makefile` | Optional `make pdf` build |

## Figures

Slide figures are stored under **`presentation/figures/`** (copies of the v2 report PDFs) so the deck is self-contained. To refresh them from the project outputs:

```bash
cp ../report/figures/v2/pdf/f1_within_condition_heatmap.pdf \
   ../report/figures/v2/pdf/degradation_heatmap.pdf \
   ../report/figures/v2/pdf/delta_f1_asymmetry_pairs.pdf \
   ../report/figures/v2/pdf/delta_j_heatmap.pdf \
   ../report/figures/v2/pdf/ci_width_comparison.pdf \
   ../report/figures/v2/pdf/fig6_noise_robustness.pdf \
   ../report/figures/v2/pdf/fig5_pca_kmeans.pdf \
   figures/
```

## Build

From this directory:

```bash
cd presentation
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or:

```bash
make pdf
```

Requirements: a LaTeX distribution with **Beamer** (TeX Live, MacTeX, MiKTeX).

Output: `main.pdf`.

## Customization

- Edit title, authors, and institute in the preamble of `main.tex`.
- Add or remove `\begin{frame}...\end{frame}` blocks; figures are under `figures/`.
- **Aspect ratio:** the deck uses **4:3** (`aspectratio=43`) so content is not cropped on most projectors. For **16:9** screens only, set `\documentclass[aspectratio=169,10pt]{beamer}` in `main.tex`.
- If a figure still touches the footline, lower its height in `\presfig[0.7]{...}` (smaller fraction).
