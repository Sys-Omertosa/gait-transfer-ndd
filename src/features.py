"""
src/features.py: Per-stride and per-subject feature engineering, and full Step 1 pipeline orchestration.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    from src.preprocessing import (
        FEATURE_COLS,
        assign_labels,
        filter_artifact_rows_v3,
        filter_pause_events,
        load_raw_data,
        partition_controls,
        summarize_control_partition,
    )
    from src.v4_provenance import atomic_write_json
except ModuleNotFoundError:
    from preprocessing import (  # type: ignore
        FEATURE_COLS,
        assign_labels,
        filter_artifact_rows_v3,
        filter_pause_events,
        load_raw_data,
        partition_controls,
        summarize_control_partition,
    )
    from v4_provenance import atomic_write_json  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / 'data' / 'raw' / 'gait-in-neurodegenerative-disease-database-1.0.0'
DEFAULT_PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'
DEFAULT_FEATURES_FILENAME = 'v4/gait_features_v4.csv'

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

# Publication-track v3 feature set: remove exact percentage complements and the
# empirically negligible signed stride asymmetry feature.
V3_FEATURE_COLS: list[str] = [
    col for col in V2_FEATURE_COLS
    if col not in {'left_stance_pct', 'right_stance_pct', 'stride_asymmetry_signed'}
]

# The default feature set used by the current authoritative v4 path.
ALL_FEATURE_COLS: list[str] = V3_FEATURE_COLS
BROADCAST_RECORDING_FEATURE_COLS: list[str] = ['cv_stride', 'cv_swing', 'dfa_alpha_stride']


def get_feature_cols(feature_set_version: str = 'v4') -> list[str]:
    """Return the configured feature columns for the requested experiment version."""
    if feature_set_version == 'v1':
        return list(ORIGINAL_FEATURE_COLS)
    if feature_set_version == 'v2':
        return list(V2_FEATURE_COLS)
    if feature_set_version == 'v3':
        return list(V3_FEATURE_COLS)
    if feature_set_version == 'v4':
        return list(V3_FEATURE_COLS)
    raise ValueError(f"Unknown feature_set_version '{feature_set_version}'")


def get_per_stride_only_feature_cols(feature_set_version: str = 'v4') -> list[str]:
    """Feature columns excluding broadcast recording-derived subject features."""
    return [
        col for col in get_feature_cols(feature_set_version)
        if col not in BROADCAST_RECORDING_FEATURE_COLS
    ]


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
    scale_mode: str = 'pragmatic',
    custom_scales: np.ndarray | None = None,
) -> tuple[float, int]:
    """
    Estimate the DFA exponent alpha for one stride-time sequence.

    A conservative scale range is used to keep the estimate stable on the shorter
    GAITNDD stride sequences: log-spaced window sizes from 4 to floor(n/4),
    requiring at least 100 strides and at least 8 usable scales.

    The default `scale_mode='pragmatic'` reproduces the current v2/v3 DFA path.
    A custom scale array can be supplied for sensitivity studies, including a
    closer Hausdorff-style schedule once the final target protocol is locked.

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
    if custom_scales is not None:
        scales = np.asarray(custom_scales, dtype=int)
    elif scale_mode == 'pragmatic':
        scales = np.unique(
            np.floor(
                np.logspace(np.log10(4), np.log10(max_scale), num=10)
            ).astype(int)
        )
    else:
        raise ValueError(f"Unknown DFA scale_mode '{scale_mode}'")
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


def compute_dfa_alpha_stride(
    df: pl.DataFrame,
    *,
    dfa_scale_mode: str = 'pragmatic',
    dfa_custom_scales: np.ndarray | None = None,
) -> pl.DataFrame:
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
        alpha, n_scales = _dfa_alpha_from_stride_sequence(
            sequence,
            scale_mode=dfa_scale_mode,
            custom_scales=dfa_custom_scales,
        )
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
    *,
    feature_set_version: str = 'v4',
    filter_strategy: str = 'v3',
    control_partition_version: str = 'v4',
    control_partition: dict[str, list[str]] | None = None,
    partition_output_filename: str = 'control_partition.json',
    manifest_filename: str | None = None,
    dfa_scale_mode: str = 'pragmatic',
    dfa_custom_scales: np.ndarray | None = None,
    metadata_cols: list[str] | None = None,
    robust_mad_multiplier: float = 3.0,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """
    Orchestrate the full Step 1 pipeline and write outputs to processed_dir.

    Pipeline order:
        1. load_raw_data                  -- load all 64 .ts files (15,160 raw strides)
        2. filter_*                      -- artifact filtering (strategy-dependent)
        3. assign_labels                 -- add binary label column (0=control, 1=pathological)
        4. partition_controls            -- define and save the control partition JSON
        5. compute_asymmetry_index       -- add per-stride asymmetry_index column
        6. compute_stride_asymmetry_signed -- add per-stride signed asymmetry
        7. compute_cv_stride             -- add per-subject cv_stride column
        8. compute_cv_swing              -- add per-subject cv_swing column
        9. compute_dfa_alpha_stride      -- add per-subject DFA alpha column

    The saved CSV contains:
        len(feature_cols) feature columns + subject_id + condition + label

    Args:
        data_dir:      Path to the raw .ts file directory. Defaults to the repo's
                       GAITNDD raw-data folder.
        processed_dir: Path to write the feature matrix and control partition.
                       Defaults to the repo's data/processed directory.
        output_filename: Name of the feature-matrix CSV written to processed_dir.
        feature_cols: Optional explicit feature list to write. Defaults to the
                      feature list resolved from feature_set_version.
    Returns:
        Tuple of:
            - Final Polars DataFrame.
            - Control partition dict with keys 'control_A' and 'control_B'.
    """
    data_path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    processed_path = (
        Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
    )
    processed_path.mkdir(parents=True, exist_ok=True)

    # Steps 1-2: load and filter (filter must be first operation on raw data)
    raw = load_raw_data(str(data_path))

    selected_feature_cols = (
        list(feature_cols)
        if feature_cols is not None else
        get_feature_cols(feature_set_version)
    )

    # Steps 5-9: feature engineering
    filter_stats: dict[str, Any] | None = None
    if filter_strategy == 'v2':
        clean = filter_pause_events(raw)
    elif filter_strategy == 'v3':
        clean, filter_stats = filter_artifact_rows_v3(
            raw,
            robust_mad_multiplier=robust_mad_multiplier,
            return_stats=True,
        )
    else:
        raise ValueError(f"Unknown filter_strategy '{filter_strategy}'")

    # Step 3: binary labels
    labeled = assign_labels(clean)

    # Step 4: save control partition (defines train/test split for all experiments)
    partition = partition_controls(
        str(processed_path / partition_output_filename),
        version=control_partition_version,
        custom_partition=control_partition,
    )

    with_asym = compute_asymmetry_index(labeled)
    if 'stride_asymmetry_signed' in selected_feature_cols:
        with_stride_features = compute_stride_asymmetry_signed(with_asym)
    else:
        with_stride_features = with_asym
    with_cv_stride = compute_cv_stride(with_stride_features)
    with_cv_swing = compute_cv_swing(with_cv_stride)
    with_dfa = compute_dfa_alpha_stride(
        with_cv_swing,
        dfa_scale_mode=dfa_scale_mode,
        dfa_custom_scales=dfa_custom_scales,
    )

    # Select final column order: features first, then metadata
    selected_metadata_cols = (
        list(metadata_cols)
        if metadata_cols is not None
        else [col for col in ('elapsed_s', 'raw_stride_index') if col in with_dfa.columns]
    )
    output_cols = selected_feature_cols + selected_metadata_cols + ['subject_id', 'condition', 'label']
    output = with_dfa.select(output_cols)

    output_path = processed_path / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(str(output_path))

    if manifest_filename is not None:
        manifest_path = processed_path / manifest_filename
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        control_summary = summarize_control_partition(partition)
        manifest = {
            'feature_set_version': feature_set_version,
            'filter_strategy': filter_strategy,
            'feature_cols': selected_feature_cols,
            'n_features': len(selected_feature_cols),
            'output_filename': output_filename,
            'partition_output_filename': partition_output_filename,
            'metadata_cols': selected_metadata_cols,
            'raw_rows': int(raw.height),
            'final_rows': int(output.height),
            'n_subjects': int(output.n_unique('subject_id')),
            'control_partition': partition,
            'control_partition_summary': control_summary,
            'dfa_scale_mode': dfa_scale_mode,
            'robust_mad_multiplier': robust_mad_multiplier,
        }
        if filter_stats is not None:
            manifest['filter_stats'] = filter_stats
        atomic_write_json(manifest_path, manifest)

    return output, partition


def build_per_stride_only_matrix(
    feature_df: pl.DataFrame,
    *,
    feature_set_version: str = 'v4',
    metadata_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Build the per-stride-only companion matrix from a full feature dataframe."""
    selected_feature_cols = get_per_stride_only_feature_cols(feature_set_version)
    selected_metadata_cols = (
        list(metadata_cols)
        if metadata_cols is not None
        else [col for col in ('elapsed_s', 'raw_stride_index') if col in feature_df.columns]
    )
    return feature_df.select(
        selected_feature_cols + selected_metadata_cols + ['subject_id', 'condition', 'label']
    )


def build_subject_level_matrix(
    feature_df: pl.DataFrame,
    *,
    feature_set_version: str = 'v4',
) -> pl.DataFrame:
    """
    Build the subject-level companion matrix for Step 6 and interpretive sensitivity.

    Stride-level timing features are aggregated by subject mean. The three
    recording-derived features are retained directly; because they are already
    broadcast constants within subject, `first()` is equivalent to `mean()`.
    """
    feature_cols = get_feature_cols(feature_set_version)
    stride_like_cols = [col for col in feature_cols if col not in BROADCAST_RECORDING_FEATURE_COLS]
    agg_exprs = [pl.col(col).mean().alias(col) for col in stride_like_cols]
    agg_exprs.extend(
        pl.col(col).first().alias(col)
        for col in BROADCAST_RECORDING_FEATURE_COLS
        if col in feature_cols
    )
    agg_exprs.extend([
        pl.col('condition').first().alias('condition'),
        pl.col('label').first().alias('label'),
        pl.len().alias('n_strides'),
    ])
    return (
        feature_df
        .group_by('subject_id', maintain_order=True)
        .agg(agg_exprs)
        .select(['subject_id'] + feature_cols + ['condition', 'label', 'n_strides'])
    )
