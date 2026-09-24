# Contributing

Thanks for your interest in this project.

This repository supports a research study on zero-shot gait-timing transfer across Parkinson's disease, Huntington's disease, and ALS, using the PhysioNet Gait in Neurodegenerative Disease Database. Contributions are welcome, particularly those that improve reproducibility, methodological clarity, statistical rigour, verification coverage, and documentation quality.

Because this is research code with published numbers attached to it, correctness and provenance matter more here than style. The sections below describe how the experiment line is organised and what a contribution needs to respect.

---

## The authoritative experiment line is `v4`

`v1`, `v2`, and `v3` are earlier experiment lines. They are retained for provenance only. **All current results come from `v4`**, and the methodology changed materially between these lines, so older numbers are not comparable.

Treat these as the reference artifacts:

| Artifact | Path |
|---|---|
| Frozen feature matrix, partition, manifests | `data/processed/v4/` |
| Result envelopes | `experiments/results/v4/` |
| Figures | `report/figures/v4/` |
| Manuscript tables | `report/tables/v4/` |

Model binaries, SHAP caches, and run logs are generated locally and excluded from version control by size. They are not part of the committed evidence.

What changed between `v3` and `v4`, in case you touch the affected code:

- inner model selection pools predictions across all held-out inner subjects instead of scoring each inner fold independently
- imbalance handling became part of the grouped candidate space rather than a post-hoc choice
- transfer models are re-selected and refitted on the full source pool instead of reusing modal fold parameters
- permutation inference became subject-aware, with Monte Carlo p-values that cannot be zero
- SHAP backgrounds became source-specific and class-balanced
- family-level SHAP aggregation uses total movement, so opposing feature movements inside a family no longer cancel

If you find documentation or notebook text that still describes `v2` behaviour as current, correcting it is a useful contribution. `PROGRESS.md` is a historical log whose entries stop before the `v4` line; `README.md` is the current overview.

---

## Before you start

Please read:

- `README.md` for scope, protocol, results, and the reproduction path
- `report/main.tex` if your contribution touches manuscript-facing outputs
- the docstrings in the relevant `src/` module, which carry the methodological reasoning

---

## Environment

Python 3.12.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt     # or requirements.txt for the full pinned set
```

Modules inside `src/` import each other flat, so put `src` on the path rather than importing through the package root:

```bash
PYTHONPATH=src python -c "from features import get_feature_cols; print(len(get_feature_cols()))"
```

Raw GAITNDD data is not committed. Download it from PhysioNet and place it under `data/raw/gait-in-neurodegenerative-disease-database-1.0.0/` if you need to rebuild the feature matrix.

---

## Repository principles

Reusable logic lives in `src/`; notebooks are the narrative and visualisation layer.

- put reusable data-processing or modelling logic in `src/`
- use notebooks to inspect, summarise, and export results, not to hold core logic
- keep manuscript-facing artifacts in `report/`
- keep runners in `scripts/training/` and checks in `scripts/verification/`

Note which runner you are looking at. The `*_modal.py` runners produce the `v4` line and execute remotely; the `*_local.py` runners target the earlier `v3` line and do not reproduce `v4` artifacts.

---

## Methodological rules

These constraints are what make the reported numbers defensible. A change that breaks one of them is a scientific regression, not a refactor.

- **Subject-level evaluation is primary.** Metrics are computed after aggregating each subject's stride probabilities into one decision. Stride-level scores are companions and are labelled as such.
- **No subject spans training and evaluation.** Grouping is by subject at every level, including the inner selection loop.
- **The held-out subject is never seen during tuning.** Candidate selection happens entirely inside the outer training pool.
- **Control A trains, Control B scores.** The two healthy pools are disjoint and were fixed before any task was defined. Do not mix them, and do not re-derive the partition casually.
- **SMOTE is training-fold only.** It never touches evaluation rows. Control A is the minority class in every source pool, so it synthesises healthy strides.
- **Transfer is zero-shot.** No target labels, hyperparameters, thresholds, or calibration sets may enter model construction.
- **Matched degradation compares like with like.** Each classifier family is compared against its own within-condition estimate, never against the source condition's best model.
- **Seeds are fixed.** The protocol manifest records seed 42; keep stochastic behaviour reproducible.

Interpretation rules that apply to text as much as code:

- SHAP reliance shift describes model behaviour, not disease mechanism, and supports no causal claim
- direction leaders are identified after observing target labels and are not a selection rule
- the conformal analysis in the repository is exploratory and is deliberately excluded from reported evidence

---

## Artifact and provenance discipline

- do not edit reported metrics by hand
- do not overwrite frozen `v4` artifacts without recording how they were regenerated
- do not change sign conventions, label semantics, or metric definitions silently
- do not mix artifacts from different experiment lines in one analysis without saying so
- preserve the existing JSON schema keys; downstream notebooks, figures, and the verification suite read them
- manifest hashes tie results to the state that produced them. If you regenerate an artifact, regenerate its manifest through `scripts/setup/` rather than editing hashes

If you change experimental logic, state in the pull request:

1. what changed
2. why it changed
3. which artifacts must be regenerated
4. whether any manuscript text or figure becomes stale

---

## Verification

There is no CI and no pytest suite. `scripts/verification/` contains 33 standalone scripts that check artifact identity, envelope schemas, runner contracts, notebook wiring, and replay equivalence. Run them directly with `python`; each prints a pass line or exits with an error.

Run the narrowest relevant checks and report what you ran:

```bash
python scripts/verification/test_v4_artifact_identity.py
python scripts/verification/test_v4_downstream_final_envelopes.py
python scripts/verification/test_v4_static_contracts.py
python scripts/verification/test_v4_authoritative_runner_contracts.py
python scripts/verification/test_v4_notebook_contracts.py
```

By area:

| If you change | Verify |
|---|---|
| preprocessing | subject and stride counts, control partition integrity, preprocessing manifest |
| training or selection | subject grouping, selection trace, result envelope structure |
| SHAP | explainer assignment, background construction, completeness behaviour |
| robustness | output keys and downstream notebook compatibility |
| manuscript-facing outputs | figures, tables, and LaTeX references still resolve |

**Known state:** `test_v4_subject_aggregation_fragmented_equivalence.py` currently fails on a stale fixture that pins the superseded tie-break rule. The pipeline is correct and the fixture is out of date. If you see that failure, it is pre-existing; fixing the fixture is a welcome contribution.

If you could not run a full validation, say so plainly in the pull request.

---

## Data and privacy

Do not commit:

- raw dataset copies under `data/raw/`
- local environment directories such as `venv/`
- private planning material, including `context/`
- notebook checkpoints and temporary caches
- API keys, tokens, or credentials

Improvements to ignore rules that strengthen this boundary are welcome.

---

## Coding guidelines

- prefer small, reviewable changes
- follow the existing style in `src/`; keep names descriptive
- keep numerical behaviour explicit and reproducible
- document non-obvious methodological choices in docstrings, including the reason
- avoid unnecessary dependencies; the environment is pinned
- use ASCII unless a file already requires Unicode

---

## Pull requests and issues

A good pull request states the problem, what changed, which files and artifacts are affected, any regeneration steps, and what you verified. If it changes experimental outputs, say whether it preserves the `v4` line, opens a new experiment line, or supersedes an existing artifact set.

Issues are welcome for reproducibility failures, documentation gaps, methodological concerns, suspicious metrics or figures, and app behaviour that diverges from stored results. Include paths, commands, and exact error messages.

---

## Useful contribution areas

- migrating the Streamlit app in `app/` from the legacy artifact line to `v4`, so its numbers match the current results
- refreshing historical documentation that still describes `v2` as authoritative
- broader verification coverage over result envelopes and derived tables
- manuscript-to-artifact consistency checks
- regenerating the few notebook figures that still use the superseded family-aggregation convention
- research extensions listed in the README roadmap, such as external validation, a selection rule that never reads target labels, or shift-aware calibration with a held-out split

---

## Citation and attribution

The manuscript is a working draft and has not been submitted to or accepted by any venue, so there is no formal citation yet. If you build on this work in the meantime, cite the repository and preserve author attribution in source files and documentation.

---

## Final note

We welcome careful contributions that make this repository easier to trust, understand, and reproduce. Provenance and correctness come first; everything else is negotiable.
