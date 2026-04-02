"""
src/preprocessing.py: Data loading, artifact filtering, labelling, and control partitioning.
"""

import json
from pathlib import Path

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
            schema_overrides={col: pl.Float64 for col in ['elapsed_s'] + FEATURE_COLS},
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


def partition_controls(output_path: str | None = None) -> dict[str, list[str]]:
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
    partition = {
        'control_A': CONTROL_A,
        'control_B': CONTROL_B,
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(partition, f, indent=2)

    return partition
