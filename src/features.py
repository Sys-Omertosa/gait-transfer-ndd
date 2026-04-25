"""
src/features.py: Per-stride and per-subject feature engineering, and full Step 1 pipeline orchestration.
"""

from pathlib import Path

import numpy as np
import polars as pl

from preprocessing import (
    FEATURE_COLS,
    assign_labels,
    filter_pause_events,
    load_raw_data,
    partition_controls,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / 'data' / 'raw' / 'gait-in-neurodegenerative-disease-database-1.0.0'
DEFAULT_PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'
DEFAULT_FEATURES_FILENAME = 'v2/gait_features_v2.csv'

# Original 14-feature baseline used in the first complete experiment chain.
ORIGINAL_FEATURE_COLS: list[str] = FEATURE_COLS + [
    'asymmetry_index',
    'cv_stride',
]

# Candidate v2 feature set: original 14 + three additive features chosen for
# transfer-study relevance without introducing the strongest redundancies from
# the earlier 20-feature attempt.
V2_FEATURE_COLS: list[str] = ORIGINAL_FEATURE_COLS + [
    'stride_asymmetry_signed',
    'cv_swing',
    'dfa_alpha_stride',
]

# The default feature set used by the v2 rerun path.
ALL_FEATURE_COLS: list[str] = V2_FEATURE_COLS


# ── Functions ─────────────────────────────────────────────────────────────────

def compute_asymmetry_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the per-stride absolute asymmetry index as a new column.

    Formula (Khera et al., Scientific Reports, 2025):
        asymmetry_index = |left_stride_s - right_stride_s| / (0.5 * (left_stride_s + right_stride_s))

    This captures bilateral stride timing asymmetry, the dominant discriminating
    feature for PD (~81% predictive weight). It is computed independently for each
    stride row and requires no grouping.

    Args:
        df: Polars DataFrame containing 'left_stride_s' and 'right_stride_s'.
    Returns:
        DataFrame with an additional 'asymmetry_index' column (Float64).
    """
    return df.with_columns(
        (
            (pl.col('left_stride_s') - pl.col('right_stride_s')).abs()
            / (0.5 * (pl.col('left_stride_s') + pl.col('right_stride_s')))
        ).alias('asymmetry_index')
    )


def compute_stride_asymmetry_signed(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the per-stride signed stride asymmetry feature.

    Formula:
        stride_asymmetry_signed =
            (left_stride_s - right_stride_s) / (left_stride_s + right_stride_s + 1e-8)

    Unlike the absolute asymmetry index, this signed form preserves laterality:
    positive values indicate longer left strides, negative values indicate longer
    right strides.

    Args:
        df: Polars DataFrame containing left/right stride timings.
    Returns:
        DataFrame with an additional 'stride_asymmetry_signed' column.
    """
    return df.with_columns(
        (
            (pl.col('left_stride_s') - pl.col('right_stride_s'))
            / (pl.col('left_stride_s') + pl.col('right_stride_s') + 1e-8)
        ).alias('stride_asymmetry_signed'),
    )


def compute_cv_stride(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the per-subject coefficient of variation of stride timing as a new column.

    Formula:
        cv_stride = std(left_stride_s) / mean(left_stride_s)

    Computed once per subject over their entire clean stride sequence, then
    broadcast as a constant value to every stride row belonging to that subject.
    CV measures stride-to-stride variability, which is a neurological property
    of the subject rather than of a single stride. It is the dominant discriminating
    feature for HD (~82% predictive weight, Khera et al., Scientific Reports, 2025).

    Uses sample standard deviation (ddof=1), which is the standard for CV computation.

    Args:
        df: Polars DataFrame with 'subject_id' and 'left_stride_s' columns.
    Returns:
        DataFrame with an additional 'cv_stride' column (Float64).
    """
    cv_per_subject = (
        df
        .group_by('subject_id')
        .agg(
            (pl.col('left_stride_s').std() / pl.col('left_stride_s').mean())
            .alias('cv_stride')
        )
    )
    return df.join(cv_per_subject, on='subject_id', how='left')


def compute_cv_swing(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the per-subject coefficient of variation of left swing timing.

    Computed once per subject over the full clean stride sequence, then joined
    back as a constant feature for each of that subject's strides. The result is
    clipped to [0.0, 2.0] to guard against pathological outliers.

    Args:
        df: Polars DataFrame with 'subject_id' and 'left_swing_s' columns.
    Returns:
        DataFrame with an additional 'cv_swing' column (Float64).
    """
    cv_per_subject = (
        df
        .group_by('subject_id')
        .agg(
            (pl.col('left_swing_s').std() / pl.col('left_swing_s').mean())
            .clip(0.0, 2.0)
            .alias('cv_swing')
        )
    )
    return df.join(cv_per_subject, on='subject_id', how='left')


def _dfa_alpha_from_stride_sequence(
    sequence: np.ndarray,
    *,
    min_strides: int = 100,
    min_scales: int = 8,
) -> tuple[float, int]:
    """
    Estimate the DFA exponent alpha for one stride-time sequence.

    A conservative scale range is used to keep the estimate stable on the shorter
    GAITNDD stride sequences: log-spaced window sizes from 4 to floor(n/4),
    requiring at least 100 strides and at least 8 usable scales.

    Args:
        sequence: 1D array of clean left-stride durations for one subject.
        min_strides: Minimum stride count required for a valid estimate.
        min_scales: Minimum number of window scales required for a valid fit.
    Returns:
        Tuple of:
            - DFA alpha estimate.
            - Number of usable scales in the regression.
    """
    x = np.asarray(sequence, dtype=np.float64)
    n = len(x)
    if n < min_strides:
        raise ValueError(
            f'DFA requires at least {min_strides} strides, received {n}'
        )

    y = np.cumsum(x - x.mean())
    max_scale = n // 4
    scales = np.unique(
        np.floor(
            np.logspace(np.log10(4), np.log10(max_scale), num=10)
        ).astype(int)
    )
    scales = scales[(scales >= 4) & (scales <= max_scale)]

    fluctuation: list[float] = []
    used_scales: list[int] = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 4:
            continue

        trimmed = y[:n_segments * scale].reshape(n_segments, scale)
        t = np.arange(scale, dtype=np.float64)
        rms = np.empty(n_segments, dtype=np.float64)

        for idx, segment in enumerate(trimmed):
            coef = np.polyfit(t, segment, 1)
            trend = coef[0] * t + coef[1]
            rms[idx] = np.sqrt(np.mean((segment - trend) ** 2))

        f_scale = float(np.sqrt(np.mean(rms ** 2)))
        if f_scale > 0:
            used_scales.append(int(scale))
            fluctuation.append(f_scale)

    if len(used_scales) < min_scales:
        raise ValueError(
            f'DFA requires at least {min_scales} usable scales, received {len(used_scales)}'
        )

    alpha = float(
        np.polyfit(np.log(np.asarray(used_scales)), np.log(np.asarray(fluctuation)), 1)[0]
    )
    return alpha, len(used_scales)


def compute_dfa_alpha_stride(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the per-subject DFA exponent of the left stride-time sequence.

    DFA summarises long-range temporal correlation structure in stride timing,
    complementing variance-based features such as cv_stride with a dynamical
    measure of serial dependence across the full walking sequence.

    The estimate is computed once per subject from the clean left_stride_s
    sequence, then joined back as a constant feature for each stride.

    Args:
        df: Polars DataFrame containing 'subject_id' and 'left_stride_s'.
    Returns:
        DataFrame with an additional 'dfa_alpha_stride' column.
    """
    rows: list[dict[str, float | str | int]] = []
    for subject_id, seq in (
        df.group_by('subject_id', maintain_order=True)
        .agg(pl.col('left_stride_s'))
        .iter_rows()
    ):
        sequence = np.asarray(seq, dtype=np.float64)
        alpha, n_scales = _dfa_alpha_from_stride_sequence(sequence)
        rows.append({
            'subject_id': subject_id,
            'dfa_alpha_stride': alpha,
            '_dfa_stride_count': int(len(sequence)),
            '_dfa_n_scales': int(n_scales),
        })

    dfa_per_subject = pl.DataFrame(rows)
    assert dfa_per_subject.height == df.n_unique('subject_id'), (
        'DFA results are missing one or more subjects'
    )
    assert dfa_per_subject['_dfa_stride_count'].min() >= 100, (
        'At least one subject has fewer than 100 clean strides for DFA'
    )
    assert dfa_per_subject['_dfa_n_scales'].min() >= 8, (
        'At least one subject has fewer than 8 usable DFA scales'
    )

    joined = df.join(
        dfa_per_subject.select(['subject_id', 'dfa_alpha_stride']),
        on='subject_id',
        how='left',
    )
    assert joined['dfa_alpha_stride'].null_count() == 0, (
        'DFA feature is missing for one or more stride rows'
    )
    return joined


def build_feature_matrix(
    data_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    output_filename: str = DEFAULT_FEATURES_FILENAME,
    feature_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """
    Orchestrate the full Step 1 pipeline and write outputs to processed_dir.

    Pipeline order:
        1. load_raw_data                 -- load all 64 .ts files (15,160 raw strides)
        2. filter_pause_events           -- remove artifact rows (407 removed, 14,753 remain)
        3. assign_labels                 -- add binary label column (0=control, 1=pathological)
        4. partition_controls            -- define and save the 8/8 control partition JSON
        5. compute_asymmetry_index       -- add per-stride asymmetry_index column
        6. compute_stride_asymmetry_signed -- add per-stride signed asymmetry
        7. compute_cv_stride             -- add per-subject cv_stride column
        8. compute_cv_swing              -- add per-subject cv_swing column
        9. compute_dfa_alpha_stride      -- add per-subject DFA alpha column

    Outputs written to processed_dir:
        v2/gait_features_v2.csv -- full feature matrix (14,753 rows x 20 columns)
        control_partition.json  -- fixed Group A / Group B subject lists

    The saved CSV contains all 20 columns:
        17 feature columns + subject_id + condition + label

    Args:
        data_dir:      Path to the raw .ts file directory. Defaults to the repo's
                       GAITNDD raw-data folder.
        processed_dir: Path to write the feature matrix and control partition.
                       Defaults to the repo's data/processed directory.
        output_filename: Name of the feature-matrix CSV written to processed_dir.
        feature_cols: Optional explicit feature list to write. Defaults to
                      the v2 feature set defined in ALL_FEATURE_COLS.
    Returns:
        Tuple of:
            - Final Polars DataFrame (14,753 x 20 for the default v2 path).
            - Control partition dict with keys 'control_A' and 'control_B'.
    """
    data_path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    processed_path = (
        Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
    )
    processed_path.mkdir(parents=True, exist_ok=True)

    # Steps 1-2: load and filter (filter must be first operation on raw data)
    raw = load_raw_data(str(data_path))
    clean = filter_pause_events(raw)

    # Step 3: binary labels
    labeled = assign_labels(clean)

    # Step 4: save control partition (defines train/test split for all experiments)
    partition = partition_controls(str(processed_path / 'control_partition.json'))

    selected_feature_cols = list(feature_cols) if feature_cols is not None else ALL_FEATURE_COLS

    # Steps 5-9: feature engineering
    with_asym = compute_asymmetry_index(labeled)
    with_stride_features = compute_stride_asymmetry_signed(with_asym)
    with_cv_stride = compute_cv_stride(with_stride_features)
    with_cv_swing = compute_cv_swing(with_cv_stride)
    with_dfa = compute_dfa_alpha_stride(with_cv_swing)

    # Select final column order: features first, then metadata
    output_cols = selected_feature_cols + ['subject_id', 'condition', 'label']
    output = with_dfa.select(output_cols)

    output_path = processed_path / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(str(output_path))

    return output, partition
