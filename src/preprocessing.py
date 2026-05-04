"""
src/preprocessing.py: Data loading, artifact filtering, labelling, and control partitioning.
"""

from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# ── Constants ─────────────────────────────────────────────────────────────────

# The 12 gait timing features as ordered in the .ts files (columns 2–13).
# Column 1 (elapsed_s) is discarded during loading.
FEATURE_COLS: list[str] = [
    'left_stride_s',
    'right_stride_s',
    'left_swing_s',
    'right_swing_s',
    'left_swing_pct',
    'right_swing_pct',
    'left_stance_s',
    'right_stance_s',
    'left_stance_pct',
    'right_stance_pct',
    'double_support_s',
    'double_support_pct',
]

SECOND_FEATURE_COLS: list[str] = [
    'left_stride_s',
    'right_stride_s',
    'left_swing_s',
    'right_swing_s',
    'left_stance_s',
    'right_stance_s',
    'double_support_s',
]

PERCENT_FEATURE_COLS: list[str] = [
    'left_swing_pct',
    'right_swing_pct',
    'left_stance_pct',
    'right_stance_pct',
    'double_support_pct',
]

ROBUST_OUTLIER_COLS: list[str] = [
    'left_stride_s',
    'right_stride_s',
    'double_support_s',
]

# Filename prefix → condition label mapping.
_CONDITION_MAP: dict[str, str] = {
    'park':    'pd',
    'hunt':    'hd',
    'als':     'als',
    'control': 'control',
}

# Fixed 8/8 control partition. This partition never changes across any experiment.
# control1–8: training pools only.
# control9–16: cross-condition test sets only.
CONTROL_A: list[str] = [
    'control1', 'control2', 'control3', 'control4',
    'control5', 'control6', 'control7', 'control8',
]
CONTROL_B: list[str] = [
    'control9',  'control10', 'control11', 'control12',
    'control13', 'control14', 'control15', 'control16',
]

# Publication-track balanced 8/8 control partition. This split preserves a
# disjoint training/evaluation firewall while nearly eliminating the age/speed
# mismatch present in the legacy lexicographic partition.
CONTROL_A_V3: list[str] = [
    'control1', 'control2', 'control3', 'control4',
    'control6', 'control7', 'control11', 'control16',
]
CONTROL_B_V3: list[str] = [
    'control5', 'control8', 'control9', 'control10',
    'control12', 'control13', 'control14', 'control15',
]

DEFAULT_SUBJECT_DESCRIPTION = (
    Path(__file__).resolve().parent.parent
    / 'data'
    / 'raw'
    / 'gait-in-neurodegenerative-disease-database-1.0.0'
    / 'subject-description.txt'
)


def _subject_description_id_col(meta: pl.DataFrame) -> str:
    """Return the subject-ID column name for subject-description.txt."""
    for candidate in ('Unnamed: 0', ''):
        if candidate in meta.columns:
            return candidate
    raise ValueError(
        f'Could not locate subject-id column in subject-description.txt; '
        f'found columns: {meta.columns}'
    )


# ── Functions ─────────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str) -> pl.DataFrame:
    """
    Load all 64 .ts files from data_dir into a single Polars DataFrame.

    Each .ts file is tab-separated, 13 columns, no header:
      - Column 1: elapsed time in seconds (discarded, not a feature)
      - Columns 2–13: the 12 gait timing features

    subject_id is derived from the filename stem (e.g. 'park1' from 'park1.ts').
    condition is derived from the filename prefix:
      'park' → 'pd', 'hunt' → 'hd', 'als' → 'als', 'control' → 'control'

    Args:
        data_dir: Path to the directory containing the .ts files.
    Returns:
        Raw Polars DataFrame with columns: FEATURE_COLS + ['subject_id', 'condition'].
        All 15,160 raw strides (before any filtering).
    """
    frames: list[pl.DataFrame] = []

    for ts_file in sorted(Path(data_dir).glob('*.ts')):
        subject_id = ts_file.stem  # e.g. 'park1'

        condition = next(
            cond for prefix, cond in _CONDITION_MAP.items()
            if subject_id.startswith(prefix)
        )

        df = pl.read_csv(
            ts_file,
            separator='\t',
            has_header=False,
            new_columns=['elapsed_s'] + FEATURE_COLS,
            schema_overrides={col: pl.Float64 for col in [
                'elapsed_s'] + FEATURE_COLS},
        )

        df = df.drop('elapsed_s').with_columns([
            pl.lit(subject_id).alias('subject_id'),
            pl.lit(condition).alias('condition'),
        ])

        frames.append(df)

    return pl.concat(frames)


def filter_pause_events(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove artifact strides that are not valid gait cycles.

    Two filters applied in sequence:

    Filter 1 (pause events, stride interval > 3.0 s):
        Excludes rows where left_stride_s > 3.0 OR right_stride_s > 3.0.
        A normal walking stride falls between 0.3-2.5 s; 3.0 s is a conservative
        upper bound that retains even severely impaired gait (ALS).
        Removes 276 rows. Note: hunt20 loses all 238 strides; its right-foot
        sensor was frozen at 19.6-58.4 s throughout the recording, leaving 19
        usable HD subjects after filtering.

    Filter 2 (physically impossible percentage values, double_support_pct > 100):
        A percentage exceeding 100 is impossible regardless of pathology and
        indicates sensor malfunction producing corrupted derived values.
        Removes 131 additional rows (als5: 90, als7: 36, others: 5).

    Expected combined removal: 407 rows (2.7% of 15,160), leaving 14,753 strides.

    Reference: PhysioNet GAITNDD documentation; data explicitly noted as unfiltered.

    Args:
        df: Raw Polars DataFrame with all stride rows.
    Returns:
        Filtered DataFrame with artifact rows removed.
    """
    # Filter 1: pause events and sensor total failures
    after_stride_filter = df.filter(
        (pl.col('left_stride_s') <= 3.0) &
        (pl.col('right_stride_s') <= 3.0)
    )

    # Filter 2: physically impossible percentage values
    after_pct_filter = after_stride_filter.filter(
        pl.col('double_support_pct') <= 100.0
    )

    return after_pct_filter


def _scaled_mad_bounds(values: np.ndarray) -> tuple[float, float] | None:
    """
    Return median ± 3 * scaled MAD bounds for a 1D array.

    If the scaled MAD is zero or non-finite, returns None to indicate that no
    robust outlier filter should be applied to this variable.
    """
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scaled_mad = 1.4826 * mad
    if not np.isfinite(scaled_mad) or scaled_mad <= 0:
        return None
    width = 3.0 * scaled_mad
    return median - width, median + width


def filter_artifact_rows_v3(
    df: pl.DataFrame,
    *,
    min_subject_strides_for_robust: int = 100,
    return_stats: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    """
    Publication-track artifact filtering for the v3 rerun.

    Stage 1 applies hard physiological plausibility constraints:
      - all stride/swing/stance times must be strictly positive
      - double_support_s must be non-negative
      - left/right stride times must not exceed 3.0 seconds
      - all percentages must lie in [0, 100]

    Stage 2 applies a per-subject robust outlier filter on a small timing block
    (left/right stride and double support) using median ± 3 * scaled MAD. This
    is intended to suppress turn, pause, and acceleration artifacts that can
    disproportionately affect variability and DFA features. If the robust stage
    would leave a subject with fewer than `min_subject_strides_for_robust`
    strides, it is skipped for that subject and only the hard plausibility
    filter is retained.

    Args:
        df: Full raw stride dataframe.
        min_subject_strides_for_robust: Minimum retained strides required to
            keep the robust-filtered rows for a subject.
        return_stats: When True, also return a JSON-ready dict of filtering
            statistics for manifests and verification.

    Returns:
        Filtered Polars dataframe, and optionally filtering statistics.
    """
    working = df.with_row_index('__row_id')

    hard_mask = (
        (pl.col('left_stride_s') > 0.0) &
        (pl.col('right_stride_s') > 0.0) &
        (pl.col('left_swing_s') > 0.0) &
        (pl.col('right_swing_s') > 0.0) &
        (pl.col('left_stance_s') > 0.0) &
        (pl.col('right_stance_s') > 0.0) &
        (pl.col('double_support_s') >= 0.0) &
        (pl.col('left_stride_s') <= 3.0) &
        (pl.col('right_stride_s') <= 3.0) &
        (pl.col('left_swing_pct') >= 0.0) &
        (pl.col('left_swing_pct') <= 100.0) &
        (pl.col('right_swing_pct') >= 0.0) &
        (pl.col('right_swing_pct') <= 100.0) &
        (pl.col('left_stance_pct') >= 0.0) &
        (pl.col('left_stance_pct') <= 100.0) &
        (pl.col('right_stance_pct') >= 0.0) &
        (pl.col('right_stance_pct') <= 100.0) &
        (pl.col('double_support_pct') >= 0.0) &
        (pl.col('double_support_pct') <= 100.0)
    )
    hard_filtered = working.filter(hard_mask)

    kept_row_ids: list[int] = []
    skipped_subjects: list[str] = []
    robust_removed_rows = 0
    per_subject_stats: list[dict[str, Any]] = []

    for subject_df in hard_filtered.partition_by('subject_id', maintain_order=True):
        subject_id = subject_df['subject_id'][0]
        subject_pd = subject_df.to_pandas()
        robust_mask = np.ones(len(subject_pd), dtype=bool)

        for col in ROBUST_OUTLIER_COLS:
            bounds = _scaled_mad_bounds(subject_pd[col].to_numpy())
            if bounds is None:
                continue
            lower, upper = bounds
            robust_mask &= (
                (subject_pd[col].to_numpy() >= lower) &
                (subject_pd[col].to_numpy() <= upper)
            )

        kept_after_robust = int(robust_mask.sum())
        subject_total = int(len(subject_pd))
        subject_removed = subject_total - kept_after_robust

        if kept_after_robust < min_subject_strides_for_robust:
            skipped_subjects.append(subject_id)
            kept_row_ids.extend(subject_pd['__row_id'].tolist())
            per_subject_stats.append({
                'subject_id': subject_id,
                'rows_before_robust': subject_total,
                'rows_after_robust': subject_total,
                'rows_removed_robust': 0,
                'robust_filter_applied': False,
            })
            continue

        kept_ids = subject_pd.loc[robust_mask, '__row_id'].tolist()
        kept_row_ids.extend(kept_ids)
        robust_removed_rows += subject_removed
        per_subject_stats.append({
            'subject_id': subject_id,
            'rows_before_robust': subject_total,
            'rows_after_robust': kept_after_robust,
            'rows_removed_robust': subject_removed,
            'robust_filter_applied': True,
        })

    final = (
        hard_filtered
        .filter(pl.col('__row_id').is_in(kept_row_ids))
        .drop('__row_id')
    )

    if not return_stats:
        return final

    raw_rows = int(df.height)
    hard_rows = int(hard_filtered.height)
    final_rows = int(final.height)
    stats = {
        'raw_rows': raw_rows,
        'rows_after_hard_filter': hard_rows,
        'rows_after_robust_filter': final_rows,
        'rows_removed_hard_filter': raw_rows - hard_rows,
        'rows_removed_robust_filter': robust_removed_rows,
        'subjects_skipped_robust_filter': skipped_subjects,
        'negative_double_support_rows_removed': int(
            working.filter(pl.col('double_support_s') < 0.0).height
        ),
        'per_subject_robust_stats': per_subject_stats,
    }
    return final, stats


def assign_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a binary classification label column to the DataFrame.

    label = 1 for pathological subjects (pd, hd, als)
    label = 0 for healthy controls

    This labelling is used for all binary within-condition and cross-condition
    experiments. The label is condition-agnostic within the pathological class;
    multi-class identity is preserved via the 'condition' column.

    Args:
        df: Polars DataFrame with a 'condition' column.
    Returns:
        DataFrame with an additional integer 'label' column (0 or 1).
    """
    return df.with_columns(
        pl.when(pl.col('condition') == 'control')
        .then(0)
        .otherwise(1)
        .cast(pl.Int8)
        .alias('label')
    )


def get_control_partition(
    version: str = 'v2',
    *,
    custom_partition: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Return the requested fixed control partition."""
    if custom_partition is not None:
        return {
            'control_A': list(custom_partition['control_A']),
            'control_B': list(custom_partition['control_B']),
        }

    if version == 'v2':
        return {
            'control_A': list(CONTROL_A),
            'control_B': list(CONTROL_B),
        }
    if version == 'v3':
        return {
            'control_A': list(CONTROL_A_V3),
            'control_B': list(CONTROL_B_V3),
        }
    raise ValueError(f"Unknown control partition version '{version}'")


def partition_controls(
    output_path: str | None = None,
    *,
    version: str = 'v2',
    custom_partition: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Define and return the fixed 8/8 control subject partition.

    This partition is fixed before any training begins and never changes across
    any experiment.

    - control_A (control1–8): used in training pools for all within-condition
      LOSO-CV experiments. Paired with PD, HD, or ALS subjects as the healthy class.
    - control_B (control9–16): reserved exclusively for cross-condition test sets.
      Neither these subjects nor the target-condition pathological subjects have
      been seen during training, ensuring genuinely zero-shot evaluation for both
      classes.

    Args:
        output_path: If provided, saves the partition as JSON to this path.
    Returns:
        Dict with keys 'control_A' and 'control_B', each a list of subject IDs.
    """
    partition = get_control_partition(
        version, custom_partition=custom_partition)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(partition, f, indent=2)

    return partition


def summarize_control_partition(
    partition: dict[str, list[str]],
    *,
    subject_description_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Summarize demographic balance for a control partition using subject metadata.
    """
    metadata_path = (
        Path(subject_description_path)
        if subject_description_path is not None else DEFAULT_SUBJECT_DESCRIPTION
    )
    meta = pl.read_csv(str(metadata_path), separator='\t',
                       null_values=['MISSING'])
    subject_id_col = _subject_description_id_col(meta)
    controls = meta.filter(pl.col('GROUP') == 'control')

    summary: dict[str, Any] = {}
    for key in ('control_A', 'control_B'):
        subset = controls.filter(pl.col(subject_id_col).is_in(partition[key]))
        summary[key] = {
            'subjects': partition[key],
            'n_subjects': int(subset.height),
            'mean_age_years': float(subset['AGE(YRS)'].cast(pl.Float64).mean()),
            'mean_gait_speed_m_per_s': float(
                subset['GaitSpeed(m/sec)'].cast(pl.Float64).mean()
            ),
            'mean_height_m': float(subset['HEIGHT(meters)'].cast(pl.Float64).mean()),
            'mean_weight_kg': float(subset['Weight(kg)'].cast(pl.Float64).mean()),
            'male_count': int(subset.filter(pl.col('gender') == 'm').height),
            'female_count': int(subset.filter(pl.col('gender') == 'f').height),
        }

    summary['between_group_deltas'] = {
        'age_years': abs(
            summary['control_A']['mean_age_years'] -
            summary['control_B']['mean_age_years']
        ),
        'gait_speed_m_per_s': abs(
            summary['control_A']['mean_gait_speed_m_per_s'] -
            summary['control_B']['mean_gait_speed_m_per_s']
        ),
    }
    return summary


def enumerate_balanced_control_partitions(
    *,
    subject_description_path: str | Path | None = None,
    top_k: int = 5,
    require_one_male_each: bool = True,
) -> list[dict[str, Any]]:
    """
    Enumerate the top-k balanced 8/8 control partitions by age/speed balance.

    The objective is the standardized mean-difference score:

        |Δ age| / sd(age_all_controls) + |Δ speed| / sd(speed_all_controls)

    This is intended for control-split sensitivity analysis rather than for
    changing the authoritative v3 split, which is fixed separately.
    """
    metadata_path = (
        Path(subject_description_path)
        if subject_description_path is not None else DEFAULT_SUBJECT_DESCRIPTION
    )
    meta = pl.read_csv(str(metadata_path), separator='\t',
                       null_values=['MISSING'])
    subject_id_col = _subject_description_id_col(meta)
    controls = (
        meta.filter(pl.col('GROUP') == 'control')
        .with_columns([
            pl.col('AGE(YRS)').cast(pl.Float64),
            pl.col('GaitSpeed(m/sec)').cast(pl.Float64),
            pl.col('HEIGHT(meters)').cast(pl.Float64),
            pl.col('Weight(kg)').cast(pl.Float64),
            pl.col(subject_id_col).str.extract(
                r'(\d+)').cast(pl.Int64).alias('_subject_num'),
        ])
        .sort('_subject_num')
    )

    rows = controls.to_dicts()
    age_sd = float(controls['AGE(YRS)'].std(ddof=0))
    speed_sd = float(controls['GaitSpeed(m/sec)'].std(ddof=0))

    ranked: list[dict[str, Any]] = []
    for combo in combinations(range(len(rows)), 8):
        group_a_rows = [rows[i] for i in combo]
        group_b_rows = [row for idx, row in enumerate(
            rows) if idx not in combo]

        if require_one_male_each:
            male_a = sum(1 for row in group_a_rows if row['gender'] == 'm')
            male_b = sum(1 for row in group_b_rows if row['gender'] == 'm')
            if male_a != 1 or male_b != 1:
                continue

        age_a = float(np.mean([row['AGE(YRS)'] for row in group_a_rows]))
        age_b = float(np.mean([row['AGE(YRS)'] for row in group_b_rows]))
        speed_a = float(np.mean([row['GaitSpeed(m/sec)']
                        for row in group_a_rows]))
        speed_b = float(np.mean([row['GaitSpeed(m/sec)']
                        for row in group_b_rows]))

        score = (
            abs(age_a - age_b) / (age_sd + 1e-12) +
            abs(speed_a - speed_b) / (speed_sd + 1e-12)
        )
        ranked.append({
            'score': score,
            'control_A': sorted(
                [row[subject_id_col] for row in group_a_rows],
                key=lambda s: int(re.search(r'\d+', s).group()),
            ),
            'control_B': sorted(
                [row[subject_id_col] for row in group_b_rows],
                key=lambda s: int(re.search(r'\d+', s).group()),
            ),
            'age_delta_years': abs(age_a - age_b),
            'gait_speed_delta_m_per_s': abs(speed_a - speed_b),
        })

    ranked.sort(key=lambda item: item['score'])
    return ranked[:top_k]
