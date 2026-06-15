"""
Prepare traffic data for forecasting experiments.

Takes raw hourly_counts.csv and produces:
- data/train.csv  (2020-01-01 to 2025-12-31)
- data/test.csv   (2026-01-01 to 2026-06-14)
- data/series/    Per-direction, per-hour-of-day time series for R
- data/toto_contexts/  Pre-built context windows for Toto
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data")
SERIES_DIR = DATA_DIR / "series"
SERIES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"

# Toto context/horizon settings
# Context: 8704 hours = 272 patches × 32 (full year, matching R model context for fairness)
TOTO_CONTEXT_HOURS = 8704  # Must be divisible by Toto patch_size=32
TOTO_HORIZON_HOURS = 160  # Must be divisible by Toto patch_size=32


def load_and_clean():
    """Load raw CSV, keep only hourly data, parse properly."""
    df = pd.read_csv(DATA_DIR / "hourly_counts.csv", low_memory=False)

    # Keep only pure hourly rows (have :00 in hour label)
    hourly = df[df["hour"].str.contains(":00", na=False)].copy()

    # Convert count to numeric, coerce errors (some are "-")
    hourly["count"] = pd.to_numeric(hourly["count"], errors="coerce")
    hourly = hourly.dropna(subset=["count"])

    # Parse date
    hourly["date"] = pd.to_datetime(hourly["date"])

    # Extract hour of day as integer (0-23)
    hour_map = {
        "12:00 am": 0, "01:00 am": 1, "02:00 am": 2, "03:00 am": 3,
        "04:00 am": 4, "05:00 am": 5, "06:00 am": 6, "07:00 am": 7,
        "08:00 am": 8, "09:00 am": 9, "10:00 am": 10, "11:00 am": 11,
        "12:00 pm": 12, "01:00 pm": 13, "02:00 pm": 14, "03:00 pm": 15,
        "04:00 pm": 16, "05:00 pm": 17, "06:00 pm": 18, "07:00 pm": 19,
        "08:00 pm": 20, "09:00 pm": 21, "10:00 pm": 22, "11:00 pm": 23,
    }
    hourly["hour_of_day"] = hourly["hour"].map(hour_map)
    hourly = hourly.dropna(subset=["hour_of_day"])

    # Create datetime index
    hourly["datetime"] = hourly["date"] + hourly["hour_of_day"].apply(
        lambda h: timedelta(hours=h)
    )

    return hourly


def split_train_test(df):
    """Split into train and test sets."""
    train = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    test = df[df["date"] >= pd.Timestamp(TEST_START)].copy()
    return train, test


def prepare_r_series(df, direction="All directions"):
    """
    Prepare data for R forecasting.

    Creates:
    1. Full daily time series (24h per day) for the main direction
    2. Per-hour-of-day series (e.g., all 8am values) for hourly models
    3. Daily totals series

    Output filenames include the direction slug to avoid overwrites.
    """
    dir_data = df[df["direction"] == direction].sort_values("datetime")
    # Create a filename-safe slug from direction name
    slug = direction.lower().replace(" ", "_").replace("/", "_")

    # 1. Full hourly series (for SARIMA with frequency=24)
    hourly_series = dir_data.pivot_table(
        index="datetime", columns="hour_of_day", values="count"
    ).fillna(0)
    # Flatten back to single column: count per hour
    hourly_flat = dir_data[["datetime", "count"]].rename(
        columns={"count": "value"}
    )
    (SERIES_DIR / f"r_hourly_full_{slug}.csv").write_text(
        hourly_flat.to_csv(index=False)
    )

    # 2. Daily totals (for daily-level models)
    daily = dir_data.groupby("date").agg(
        total_count=("count", "sum"),
        peak_count=("count", "max"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    (SERIES_DIR / f"r_daily_totals_{slug}.csv").write_text(
        daily.to_csv(index=False)
    )

    # 3. Per-hour-of-day series (e.g., all 8am values across all days)
    # This is useful for modeling specific hours
    for h in range(24):
        h_data = dir_data[dir_data["hour_of_day"] == h][["date", "count"]].copy()
        h_data = h_data.rename(columns={"count": f"hour_{h:02d}"})
        h_data = h_data.sort_values("date")
        (SERIES_DIR / f"r_hour_{h:02d}_{slug}.csv").write_text(
            h_data.to_csv(index=False)
        )

    # 4. Wide format: each column is an hour of the day, rows are dates
    wide = dir_data.pivot_table(
        index="date", columns="hour_of_day", values="count"
    ).reset_index()
    wide.columns = ["date"] + [f"h{h:02d}" for h in range(24)]
    (SERIES_DIR / f"r_wide_hourly_{slug}.csv").write_text(
        wide.to_csv(index=False)
    )

    return hourly_flat, daily


def prepare_toto_contexts(df, direction="All directions"):
    """
    Prepare context windows for Toto forecasting.

    For each test date, create a sliding window context of TOTO_CONTEXT_HOURS
    hours ending just before the test period, with the next TOTO_HORIZON_HOURS
    hours as the target.
    """
    dir_data = df[df["direction"] == direction].sort_values("datetime")

    # Create complete hourly index (fill missing hours with 0)
    full_index = pd.date_range(
        start=dir_data["datetime"].min(),
        end=dir_data["datetime"].max(),
        freq="h"
    )
    series = dir_data.set_index("datetime")["count"].reindex(full_index, fill_value=0)

    # Find where test period starts
    test_start_dt = pd.Timestamp(TEST_START)

    # Build rolling windows: slide through test set
    windows = []
    test_idx = series.index.get_loc(test_start_dt) if test_start_dt in series.index else series.index.searchsorted(test_start_dt)

    for i in range(test_idx, len(series) - TOTO_HORIZON_HOURS + 1, 24):
        # Slide by 24 hours (forecast each day)
        context_start = i - TOTO_CONTEXT_HOURS
        if context_start < 0:
            continue

        context = series.iloc[context_start:i].values
        target = series.iloc[i:i + TOTO_HORIZON_HOURS].values

        windows.append({
            "context_start": series.index[context_start],
            "context_end": series.index[i - 1],
            "forecast_start": series.index[i],
            "forecast_end": series.index[i + TOTO_HORIZON_HOURS - 1],
            "context": context,
            "target": target,
        })

    # Save as numpy for fast loading
    toto_dir = DATA_DIR / "toto_contexts"
    toto_dir.mkdir(parents=True, exist_ok=True)

    for j, w in enumerate(windows):
        np.savez(
            toto_dir / f"window_{j:04d}.npz",
            context=w["context"],
            target=w["target"],
            forecast_start=str(w["forecast_start"]),
            forecast_end=str(w["forecast_end"]),
        )

    # Also save metadata
    meta = pd.DataFrame([
        {
            "window_id": j,
            "context_start": w["context_start"],
            "context_end": w["context_end"],
            "forecast_start": w["forecast_start"],
            "forecast_end": w["forecast_end"],
        }
        for j, w in enumerate(windows)
    ])
    (toto_dir / "metadata.csv").write_text(meta.to_csv(index=False))

    print(f"Created {len(windows)} Toto forecast windows")
    return windows


def prepare_toto_contexts_multivariate(df, directions=None):
    """
    Prepare multivariate context windows for Toto forecasting.

    Passes all directional lanes as separate variates so Toto's
    variate-wise attention can learn cross-lane correlations.

    Args:
        df: Full cleaned dataframe with all directions
        directions: List of direction names to include as variates
    """
    if directions is None:
        directions = [
            "All directions", "All Northbound", "All Southbound",
            "Ln 1 NB", "Center Turn Lane", "Ln 1 SB"
        ]

    # Build per-direction series aligned to the same hourly index
    full_index = pd.date_range(
        start=df["datetime"].min(),
        end=df["datetime"].max(),
        freq="h"
    )

    series_dict = {}
    for d in directions:
        dir_data = df[df["direction"] == d].sort_values("datetime")
        s = dir_data.set_index("datetime")["count"].reindex(full_index, fill_value=0)
        series_dict[d] = s

    # Stack into (n_hours, n_variates) array
    n_hours = len(full_index)
    n_var = len(directions)
    all_series = np.column_stack([series_dict[d].values for d in directions])

    test_start_dt = pd.Timestamp(TEST_START)
    test_idx = full_index.get_loc(test_start_dt) if test_start_dt in full_index else full_index.searchsorted(test_start_dt)

    # Build rolling windows
    windows = []
    for i in range(test_idx, n_hours - TOTO_HORIZON_HOURS + 1, 24):
        context_start = i - TOTO_CONTEXT_HOURS
        if context_start < 0:
            continue

        context = all_series[context_start:i, :]  # (context_hours, n_var)
        target = all_series[i:i + TOTO_HORIZON_HOURS, 0]  # "All directions" only for target

        windows.append({
            "context_start": full_index[context_start],
            "context_end": full_index[i - 1],
            "forecast_start": full_index[i],
            "forecast_end": full_index[i + TOTO_HORIZON_HOURS - 1],
            "context": context,  # (320, 6)
            "target": target,    # (160,)
        })

    # Save to separate directory
    toto_mv_dir = DATA_DIR / "toto_contexts_mv"
    toto_mv_dir.mkdir(parents=True, exist_ok=True)

    for j, w in enumerate(windows):
        np.savez(
            toto_mv_dir / f"window_{j:04d}.npz",
            context=w["context"],
            target=w["target"],
            forecast_start=str(w["forecast_start"]),
            forecast_end=str(w["forecast_end"]),
        )

    meta = pd.DataFrame([
        {
            "window_id": j,
            "context_start": w["context_start"],
            "context_end": w["context_end"],
            "forecast_start": w["forecast_start"],
            "forecast_end": w["forecast_end"],
        }
        for j, w in enumerate(windows)
    ])
    (toto_mv_dir / "metadata.csv").write_text(meta.to_csv(index=False))

    print(f"Created {len(windows)} multivariate Toto windows ({n_var} variates)")
    return windows


def main():
    print("Loading and cleaning data...")
    df = load_and_clean()
    print(f"  {len(df)} hourly records, {df['direction'].nunique()} directions")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nSplitting train/test...")
    train, test = split_train_test(df)
    (DATA_DIR / "train.csv").write_text(train.to_csv(index=False))
    (DATA_DIR / "test.csv").write_text(test.to_csv(index=False))
    print(f"  Train: {len(train)} records ({train['date'].min()} to {train['date'].max()})")
    print(f"  Test:  {len(test)} records ({test['date'].min()} to {test['date'].max()})")

    print("\nPreparing R series...")
    # Combine train+test for continuous series (R script does its own split)
    all_data = pd.concat([train, test], ignore_index=True)
    prepare_r_series(all_data, "All directions")
    # Also prepare for individual directions
    for direction in ["All Northbound", "All Southbound"]:
        dir_data = all_data[all_data["direction"] == direction]
        prepare_r_series(dir_data, direction)
    print("  R series saved to data/series/")

    print("\nPreparing Toto contexts (univariate)...")
    # Combine train + test for continuous series
    prepare_toto_contexts(df, "All directions")
    print("  Toto contexts saved to data/toto_contexts/")

    print("\nPreparing Toto contexts (multivariate, 6 lanes)...")
    prepare_toto_contexts_multivariate(df)
    print("  Toto MV contexts saved to data/toto_contexts_mv/")

    print("\nDone!")


if __name__ == "__main__":
    main()
