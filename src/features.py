"""
src/features.py: Per-stride and per-subject feature engineering, and full Step 1 pipeline orchestration.
"""

from pathlib import Path

import polars as pl

from src.preprocessing import (
    FEATURE_COLS,
    load_raw_data,
    filter_pause_events,
    assign_labels,
    partition_controls,
)

# The full set of feature columns written to gait_features.csv.
# 12 raw features + 2 engineered features = 14 columns used as X in all models.
ALL_FEATURE_COLS: list[str] = FEATURE_COLS + ['asymmetry_index', 'cv_stride']


# ── Functions ─────────────────────────────────────────────────────────────────

def compute_asymmetry_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the per-stride asymmetry index as a new column.

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


def build_feature_matrix(data_dir: str, processed_dir: str) -> pl.DataFrame:
    """
    Orchestrate the full Step 1 pipeline and write outputs to processed_dir.

    Pipeline order:
        1. load_raw_data        -- load all 64 .ts files (15,160 raw strides)
        2. filter_pause_events  -- remove artifact rows (407 removed, 14,753 remain)
        3. assign_labels        -- add binary label column (0=control, 1=pathological)
        4. partition_controls   -- define and save the 8/8 control partition JSON
        5. compute_asymmetry_index -- add per-stride asymmetry_index column
        6. compute_cv_stride    -- add per-subject cv_stride column

    Outputs written to processed_dir:
        gait_features.csv       -- full feature matrix (14,753 rows x 17 columns)
        control_partition.json  -- fixed Group A / Group B subject lists

    The saved CSV contains all 17 columns:
        12 raw features + asymmetry_index + cv_stride + subject_id + condition + label

    Args:
        data_dir:      Path to the raw .ts file directory.
        processed_dir: Path to write gait_features.csv and control_partition.json.
    Returns:
        Final Polars DataFrame (14,753 x 17).
    """
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Steps 1-2: load and filter (filter must be first operation on raw data)
    raw = load_raw_data(data_dir)
    clean = filter_pause_events(raw)

    # Step 3: binary labels
    labeled = assign_labels(clean)

    # Step 4: save control partition (defines train/test split for all experiments)
    partition_controls(str(processed_path / 'control_partition.json'))

    # Steps 5-6: feature engineering
    with_asym = compute_asymmetry_index(labeled)
    with_cv   = compute_cv_stride(with_asym)

    # Select final column order: features first, then metadata
    output_cols = ALL_FEATURE_COLS + ['subject_id', 'condition', 'label']
    output = with_cv.select(output_cols)

    output.write_csv(str(processed_path / 'gait_features.csv'))

    return output
