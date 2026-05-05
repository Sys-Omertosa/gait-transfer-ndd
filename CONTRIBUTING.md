# Contributing

Thank you for your interest in contributing to this project.

This repository supports a research study on gait-timing transfer across
neurological conditions using the PhysioNet Neurodegenerative Disease Gait
database. Contributions are welcome, especially when they improve
reproducibility, methodological clarity, documentation quality, statistical
rigor, and robustness of the experimental pipeline.

## Scope of Contributions

Useful contributions include:

- bug fixes in preprocessing, training, explainability, robustness, or app code
- reproducibility improvements for the `v2` experiment path
- validation scripts, consistency checks, and artifact verification utilities
- documentation improvements for setup, interpretation, and replication
- notebook cleanup that improves clarity without changing scientific meaning
- additional tests for metric computation, data filtering, and result integrity
- careful extensions of the methodology, provided they preserve provenance and
  do not silently overwrite the current authoritative artifacts

Less useful contributions include:

- stylistic refactors with no research or maintenance benefit
- replacing committed authoritative outputs without a clear reproduction path
- adding unpublished claims to the paper or README without code-backed evidence
- committing raw private data, local environment files, or private working notes

## Before You Start

Please read:

- `README.md` for project scope and reproduction flow
- `PROGRESS.md` for the historical development path
- `report/main.tex` if your contribution touches manuscript-facing outputs
- specifications and docstrings in relevant code files

The current authoritative result path is `v2`. If your work builds on the
present experiment line, please treat these as the reference artifacts:

- `data/processed/v2/gait_features_v2.csv`
- `experiments/results/v2/`
- `experiments/models/v2/`
- `report/figures/v2/`
- `report/tables/v2/`

## Environment Setup

Use Python 3.12.

Full environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Smaller research environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

## Repository Principles

The repository is organized around a simple rule: reusable logic lives in
`src/`, while notebooks remain the narrative and visualization layer.

- Put reusable data-processing or modeling logic in `src/`
- Use notebooks to inspect, summarize, and export results
- Keep manuscript-facing artifacts in `report/`
- Keep runner scripts in `scripts/training/` or `scripts/verification/`

Please do not duplicate core logic across notebooks when it can live in a
shared module.

## Research Integrity Requirements

This project is research code, so contributions need stronger discipline than a
typical application repository.

- Do not modify reported metrics manually
- Do not overwrite authoritative outputs without recording how they were
  regenerated
- Do not change sign conventions, labels, or metric definitions silently
- Do not mix `v1` and `v2` artifacts inside the same analysis without stating it
- Do not introduce data leakage, especially across subject boundaries or control
  partitions

If you change experimental logic, please explain:

1. what changed
2. why it changed
3. which artifacts must be regenerated
4. whether prior manuscript text becomes stale

## Data and Privacy Rules

The repository intentionally does not commit raw GAITNDD source files.

Please do not commit:

- raw dataset copies under `data/raw/`
- local environment directories such as `venv/`
- any private planning files, like under `context/`
- temporary notebook or Jupyter cache files
- personal API keys, tokens, or secrets

If you add new ignores that improve repository hygiene, that is a welcome
contribution.

## Coding Guidelines

Please follow these conventions:

- prefer small, reviewable changes
- keep file and function names descriptive
- use ASCII by default unless a file already requires Unicode
- preserve existing result schema conventions in JSON outputs
- avoid unnecessary dependency additions
- prefer `rg` for fast text search and repository inspection

For Python:

- follow the existing code style in `src/`
- keep numerical behavior explicit and reproducible
- use fixed random seeds where the current pipeline expects them
- document non-obvious methodological choices in comments or docstrings

## Testing and Verification

If your contribution affects experimental results, please run the narrowest
relevant verification step you can and report what you checked.

Examples:

- preprocessing changes:
  - verify subject counts, stride counts, and control partition integrity
- training changes:
  - verify LOSO grouping, modal parameter selection, and result JSON structure
- SHAP changes:
  - verify explainer assignment and completeness behavior where applicable
- robustness changes:
  - verify output JSON keys and downstream notebook compatibility
- manuscript-facing changes:
  - verify figures, tables, or LaTeX references still resolve correctly

If you could not run a full validation, say so clearly in the pull request.

## Pull Request Expectations

Good pull requests usually include:

- a short description of the problem
- a clear summary of what changed
- the scope of affected files or artifacts
- any regeneration steps needed
- verification notes

If your contribution changes experimental outputs, please say whether it:

- preserves the current `v2` authoritative path
- creates a new experimental branch
- supersedes an existing artifact set

## Issues and Discussion

Issues are welcome for:

- reproducibility failures
- documentation gaps
- methodological concerns
- suspicious metrics or figure inconsistencies
- app behavior that diverges from stored results

When opening an issue, please include enough context for someone else to
reproduce the problem. Paths, commands, and exact error messages help.

## Suggested Contribution Areas

Researchers and external contributors may find these areas especially useful:

- better verification coverage around result JSON integrity
- lightweight tests for preprocessing and metric utilities
- clearer artifact provenance around exported paper tables
- manuscript-to-code consistency checks
- app usability improvements that do not alter the underlying science
- future experiment-path preparation for post-`v2` improvement work

## Citation and Attribution

If you use this repository in academic work, please cite the associated paper
or repository once the formal citation is finalized in the manuscript and
project metadata.

Please preserve author attribution in source files and documentation when
extending existing work.

## Final Note

We welcome careful contributions that make the repository easier to trust,
understand, and reproduce. In this project, correctness and provenance matter
at least as much as code style.
