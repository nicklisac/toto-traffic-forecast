"""
Horizon stress-test analysis: Measure how MAE grows with forecast horizon depth.

Two analyses:
1. CPM Stress-Test (Toto-only): How error grows across all rolling windows at different horizons.
   Demonstrates CPM's parallel decoding — inference time flat regardless of horizon.
2. Single-Origin Comparison (R + Toto): All models fitted once on 8,760h context, no re-scoring.
   Uses Toto window_id == 0 (2026-01-01 origin) to match R model origin exactly.

Segments:
  - 0-167h (1-week-ahead)
  - 168-335h (2-weeks-ahead)
  - 336-503h (3-weeks-ahead)
  - 504-671h (4-weeks-ahead)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load actual test data for lookup beyond forecast window
actuals_df = pd.read_csv(DATA_DIR / "forecasts_r" / "actuals_hourly.csv")
actuals_df["datetime"] = pd.to_datetime(actuals_df["datetime"], format="mixed", utc=True)
actuals_df = actuals_df.set_index("datetime")
actuals_df.index = actuals_df.index.floor("h")

# Segment definitions (hour ranges)
SEGMENTS = {
    "0-167h\n(1 week)": (0, 168),
    "168-335h\n(2 weeks)": (168, 336),
    "336-503h\n(3 weeks)": (336, 504),
    "504-671h\n(4 weeks)": (504, 672),
}

SEGMENT_LABELS = ["0-167h (1 week)", "168-335h (2 weeks)",
                  "336-503h (3 weeks)", "504-671h (4 weeks)"]


# ---------------------------------------------------------------------------
# 1. CPM Stress-Test (Toto-only, all windows)
# ---------------------------------------------------------------------------

def load_toto_forecasts(horizon):
    """Load Toto forecasts for a given horizon.

    h160 is stored in data/forecasts_toto/toto_forecasts.csv (default).
    h320/h672 are in data/forecasts_toto_h{horizon}/toto_forecasts_h{horizon}.csv.
    """
    if horizon == 160:
        fpath = DATA_DIR / "forecasts_toto" / "toto_forecasts.csv"
    else:
        fpath = DATA_DIR / f"forecasts_toto_h{horizon}" / f"toto_forecasts_h{horizon}.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def compute_toto_segment_mae(df, horizon):
    """Compute MAE for each horizon segment, averaged across all windows.

    Allows partial segments: if horizon=160, the 0-167h segment uses hours 0-159.
    Looks up actuals from test data for hours beyond the forecast window's target.
    """
    segment_errors = {name: [] for name in SEGMENTS}

    for wid, window in df.groupby("window_id"):
        window = window.sort_values("datetime")
        forecast_start = window["datetime"].iloc[0]

        for seg_name, (start, end) in SEGMENTS.items():
            if start >= horizon:
                continue

            seg_end = min(end, horizon, len(window))
            errors = []
            for i in range(start, seg_end):
                fc = window.iloc[i]["forecast_median"]
                act = window.iloc[i]["actual"]

                # Look up actual from test data if NaN
                if np.isnan(act):
                    dt = window.iloc[i]["datetime"]
                    try:
                        act = actuals_df.loc[dt, "actual"]
                    except (KeyError, TypeError):
                        continue

                if not np.isnan(act) and not np.isnan(fc):
                    errors.append(abs(act - fc))

            if errors:
                segment_errors[seg_name].append(np.mean(errors))

    return {name: np.mean(errs) if errs else np.nan
            for name, errs in segment_errors.items()}


def plot_cpm_stress_test():
    """Plot Toto-only error growth by horizon depth (CPM stress-test)."""
    results = {}

    for horizon in [160, 320, 672]:
        df = load_toto_forecasts(horizon)
        if df is not None:
            results[horizon] = compute_toto_segment_mae(df, horizon)

    fig, ax = plt.subplots(figsize=(10, 5))

    seg_names = list(SEGMENTS.keys())
    x_pos = range(len(seg_names))
    colors = {160: "#2563eb", 320: "#7c3aed", 672: "#dc2626"}
    widths = 0.25

    for horizon, color in colors.items():
        if horizon not in results:
            continue
        vals = [results[horizon].get(seg, np.nan) for seg in seg_names]
        offset = (horizon - 320) / 320 * widths
        bars = ax.bar([x + offset for x in x_pos], vals, widths,
                      label=f"Toto h={horizon}h", color=color, alpha=0.8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(seg_names)
    ax.set_ylabel("MAE (vehicles/hour)")
    ax.set_title("Horizon Stress-Test: CPM Error Growth by Forecast Depth\n"
                  "(averaged across all rolling windows)")
    ax.legend(title="Total horizon")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "plots" / "horizon_stress_test.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Saved: horizon_stress_test.png")
    return results


# ---------------------------------------------------------------------------
# 2. Single-Origin Comparison (R + Toto, same origin)
# ---------------------------------------------------------------------------

def load_r_forecasts(horizon):
    """Load R model forecasts for a given horizon."""
    fpath = DATA_DIR / "forecasts_r" / f"horizon_stress_h{horizon}.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=True)
    return df


def compute_r_segment_mae(df, model_col, horizon):
    """Compute MAE for each horizon segment for an R model.

    Allows partial segments: if horizon=168, the 0-167h segment uses all 168 hours.
    """
    segment_errors = {}

    for seg_name, (start, end) in SEGMENTS.items():
        if start >= horizon:
            continue
        seg_end = min(end, horizon, len(df))
        fc = df[model_col].iloc[start:seg_end].values
        actual = df["actual"].iloc[start:seg_end].values
        mask = ~np.isnan(actual) & ~np.isnan(fc)
        if mask.any():
            segment_errors[seg_name] = np.mean(np.abs(actual[mask] - fc[mask]))

    return segment_errors


def compute_toto_single_origin_mae(horizon, window_id=7):
    """Compute MAE for Toto at a single origin (window_id=7, 2026-01-08).

    Window 7 starts at 2026-01-08, matching R models' origin.
    Context (2025-01-10 to 2025-12-31) includes Jan 1-2, 2025 holiday pattern.

    Looks up actuals from test data for hours beyond the forecast window's target.
    Allows partial segments when horizon < segment end.
    """
    df = load_toto_forecasts(horizon)
    if df is None:
        return {}

    # Filter to single origin
    window = df[df["window_id"] == window_id].sort_values("datetime")
    if window.empty:
        return {}

    forecast_start = window["datetime"].iloc[0]
    segment_errors = {}

    for seg_name, (start, end) in SEGMENTS.items():
        if start >= horizon:
            continue
        seg_end = min(end, horizon, len(window))

        errors = []
        for i in range(start, seg_end):
            fc = window.iloc[i]["forecast_median"]
            act = window.iloc[i]["actual"]

            # Look up actual from test data if NaN
            if np.isnan(act):
                dt = window.iloc[i]["datetime"]
                try:
                    act = actuals_df.loc[dt, "actual"]
                except (KeyError, TypeError):
                    continue

            if not np.isnan(act) and not np.isnan(fc):
                errors.append(abs(act - fc))

        if errors:
            segment_errors[seg_name] = np.mean(errors)

    return segment_errors


def compute_all_single_origin():
    """Compute segment MAE for all models at matching horizons, single origin."""
    # Map R horizons to Toto horizons (R uses 168/336/672, Toto uses 160/320/672)
    horizon_pairs = [
        ("r", 168, ("toto", 160)),
        ("r", 336, ("toto", 320)),
        ("r", 672, ("toto", 672)),
    ]

    results = {
        "SARIMA": {},
        "MSTL+ARIMA": {},
        "Naive": {},
        "Toto-2.5B": {},
    }

    r_model_cols = {
        "SARIMA": "sarima",
        "MSTL+ARIMA": "tbats",
        "Naive": "naive",
    }

    for r_label, r_horizon, (toto_label, toto_horizon) in horizon_pairs:
        # R models
        r_df = load_r_forecasts(r_horizon)
        if r_df is not None:
            for model_name, col in r_model_cols.items():
                seg_mae = compute_r_segment_mae(r_df, col, r_horizon)
                results[model_name].update(seg_mae)

        # Toto — single origin (window_id == 7, starts 2026-01-08)
        toto_seg_mae = compute_toto_single_origin_mae(toto_horizon, window_id=7)
        results["Toto-2.5B"].update(toto_seg_mae)

    return results


def plot_single_origin_comparison(results):
    """Plot R vs Toto error growth from single origin (Figure 5)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    seg_names = list(SEGMENTS.keys())
    x_pos = np.arange(len(seg_names))
    width = 0.2

    models = {
        "SARIMA": {"color": "#e74c3c", "marker": "o"},
        "MSTL+ARIMA": {"color": "#3498db", "marker": "s"},
        "Naive": {"color": "#2ecc71", "marker": "^"},
        "Toto-2.5B": {"color": "#9b59b6", "marker": "D"},
    }

    offsets = [-1.5, -0.5, 0.5, 1.5]

    for idx, (model, style) in enumerate(models.items()):
        if model not in results:
            continue
        vals = []
        for seg in seg_names:
            if seg in results[model]:
                vals.append(results[model][seg])
            else:
                vals.append(np.nan)

        offset = offsets[idx] * width
        ax.plot([x + offset for x in x_pos], vals, style["marker"],
                label=model, color=style["color"], linewidth=2, markersize=7)
        for x, v in zip(x_pos, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.0f}", (x + offset, v),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7, color=style["color"])

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(seg_names)
    ax.set_ylabel("MAE (vehicles/hour)")
    ax.set_title(
        "Error Growth by Horizon Depth:\n"
        "Autoregressive (SARIMA, MSTL+ARIMA, Naive) vs CPM (Toto-2.5B)\n"
        "All models fitted once on 8,760h context, no re-scoring",
        y=1.05
    )
    ax.legend(title="Model", loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "plots" / "horizon_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Saved: horizon_comparison.png")


def print_comparison_table(results):
    """Print MAE comparison table for paper."""
    seg_display = {
        "0-167h\n(1 week)": "1wk",
        "168-335h\n(2 weeks)": "2wk",
        "336-503h\n(3 weeks)": "3wk",
        "504-671h\n(4 weeks)": "4wk",
    }

    print("\nMAE by horizon segment (single origin, 2026-01-08):")
    print(f"{'Model':<15}", end="")
    for seg in seg_display:
        print(f"{seg_display[seg]:>10}", end="")
    print()

    for model in ["Toto-2.5B", "SARIMA", "MSTL+ARIMA", "Naive"]:
        if model not in results:
            continue
        print(f"{model:<15}", end="")
        for seg in seg_display:
            if seg in results[model]:
                print(f"{results[model][seg]:>10.1f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1. CPM Stress-Test (Toto-only, all rolling windows)")
    print("=" * 60)
    cpm_results = plot_cpm_stress_test()
    for horizon, segs in cpm_results.items():
        print(f"\n  Horizon={horizon}h:")
        for seg, mae in segs.items():
            seg_display = seg.replace("\n", " ")
            print(f"    {seg_display}: MAE={mae:.1f}" if not np.isnan(mae)
                  else f"    {seg_display}: N/A")

    print("\n" + "=" * 60)
    print("2. Single-Origin Comparison (R + Toto, 2026-01-08 origin)")
    print("=" * 60)
    comparison_results = compute_all_single_origin()
    plot_single_origin_comparison(comparison_results)
    print_comparison_table(comparison_results)
