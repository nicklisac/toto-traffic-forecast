"""
Evaluation framework: Compare R forecasts vs Toto-2.0 forecasts.

Produces metrics tables and comparison plots.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
R_DIR = DATA_DIR / "forecasts_r"
TOTO_DIR = DATA_DIR / "forecasts_toto"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(actual, forecast, name=""):
    """Compute forecasting metrics."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(forecast)
    actual, forecast = actual[mask], forecast[mask]

    if len(actual) == 0:
        return {}

    errors = actual - forecast
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / np.where(actual == 0, 1, actual)) * 100
    smape = np.mean(
        2 * abs_errors / np.where(actual + forecast == 0, 1, actual + forecast)
    ) * 100

    # Coverage: what % of actuals fall within prediction interval (if available)
    coverage = {}

    return {
        "model": name,
        "n": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE%": mape,
        "sMAPE%": smape,
        **coverage,
    }


def compute_calibration(actual, lower, upper, nominal=0.80):
    """Compute uncertainty calibration metrics.

    Parameters
    ----------
    actual : array-like
        Observed values.
    lower, upper : array-like
        Prediction interval bounds (e.g. q10, q90 for 80% PI).
    nominal : float
        Expected coverage rate (0.80 for 80% interval).

    Returns
    -------
    dict with coverage%, ECE, Winkler Score, and interval width stats.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    actual, lower, upper = actual[mask], lower[mask], upper[mask]

    if len(actual) == 0:
        return {}

    # Empirical coverage: fraction of actuals inside the interval
    inside = (actual >= lower) & (actual <= upper)
    empirical_cov = np.mean(inside) * 100

    # Empirical Coverage Error: |empirical - nominal|
    ece = abs(empirical_cov - nominal * 100)

    # Winkler Score: interval width, penalized for misses
    # Lower is better.  w_i = (upper - lower) + 2/(upper-lower) * (lower - actual)*{actual < lower} + 2/(upper-lower) * (actual - upper)*{actual > upper}
    width = upper - lower
    width = np.where(width == 0, 1e-6, width)  # avoid division by zero
    winkler = width.copy()
    below = actual < lower
    above = actual > upper
    winkler[below] += 2 * (lower[below] - actual[below])
    winkler[above] += 2 * (actual[above] - upper[above])
    mean_winkler = np.mean(winkler)

    return {
        "Coverage%": round(empirical_cov, 1),
        "ECE%": round(ece, 1),
        "Winkler": round(mean_winkler, 1),
        "Mean_Width": round(np.mean(width), 1),
    }


def load_r_forecasts(granularity="hourly"):
    """Load R forecast files."""
    forecasts = {}
    for model in ["sarima", "tbats", "naive"]:
        fpath = R_DIR / f"{model}_{granularity}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            forecasts[model] = df
    return forecasts


def load_toto_forecasts():
    """Load Toto forecast files, aligned to non-overlapping daily forecasts."""
    fpath = TOTO_DIR / "toto_forecasts.csv"
    if not fpath.exists():
        return None

    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Each window forecasts 160h starting from its forecast_start date.
    # Windows slide by 24h. Keep the FRESHEST forecast for each datetime
    # (highest window_id = shortest horizon = most recent context).
    df = df.sort_values(["datetime", "window_id"], ascending=[True, False])
    df = df.drop_duplicates(subset="datetime", keep="first")

    return df


def load_actuals(granularity="hourly"):
    """Load actual values."""
    fpath = R_DIR / f"actuals_{granularity}.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    return df


def compare_hourly():
    """Compare hourly forecasts."""
    print("\n=== Hourly Forecast Comparison ===\n")

    r_fcs = load_r_forecasts("hourly")
    toto_fc = load_toto_forecasts()
    actuals = load_actuals("hourly")

    if actuals is None:
        print("No actuals found, skipping hourly comparison")
        return {}

    actual_values = actuals["actual"].values
    metrics = []

    # R models
    for model_name, fc_df in r_fcs.items():
        fc_values = fc_df["forecast"].values[:len(actual_values)]
        m = compute_metrics(actual_values, fc_values, model_name.upper())
        metrics.append(m)
        print(f"  {model_name.upper():8s}: MAE={m['MAE']:8.1f}, RMSE={m['RMSE']:8.1f}, "
              f"MAPE={m['MAPE%']:6.1f}%, sMAPE={m['sMAPE%']:6.1f}%")

    # Toto
    if toto_fc is not None:
        # Toto forecasts by window, align with actuals
        toto_values = toto_fc["forecast_median"].values[:len(actual_values)]
        m = compute_metrics(actual_values, toto_values, "TOTO-2.5B")
        # Calibration: Toto has quantile outputs (q10/q90 = 80% PI)
        if "forecast_q10" in toto_fc.columns and "forecast_q90" in toto_fc.columns:
            cal = compute_calibration(
                actual_values,
                toto_fc["forecast_q10"].values[:len(actual_values)],
                toto_fc["forecast_q90"].values[:len(actual_values)],
                nominal=0.80,
            )
            m.update(cal)
        metrics.append(m)
        cal_str = ""
        if "Coverage%" in m:
            cal_str = f", Coverage={m['Coverage%']:.1f}%, ECE={m['ECE%']:.1f}%, Winkler={m['Winkler']:.0f}"
        print(f"  {'TOTO-2.5B':8s}: MAE={m['MAE']:8.1f}, RMSE={m['RMSE']:8.1f}, "
              f"MAPE={m['MAPE%']:6.1f}%, sMAPE={m['sMAPE%']:6.1f}%{cal_str}")

    return metrics


def compare_daily():
    """Compare daily forecasts."""
    print("\n=== Daily Forecast Comparison ===\n")

    r_fcs = load_r_forecasts("daily")
    actuals = load_actuals("daily")

    if actuals is None:
        print("No daily actuals found, skipping")
        return {}

    actual_values = actuals["actual"].values
    metrics = []

    for model_name, fc_df in r_fcs.items():
        fc_values = fc_df["forecast"].values[:len(actual_values)]
        m = compute_metrics(actual_values, fc_values, model_name.upper())
        metrics.append(m)
        print(f"  {model_name.upper():8s}: MAE={m['MAE']:8.1f}, RMSE={m['RMSE']:8.1f}, "
              f"MAPE={m['MAPE%']:6.1f}%, sMAPE={m['sMAPE%']:6.1f}%")

    # Toto: aggregate hourly forecasts to daily totals
    toto_fc = load_toto_forecasts()
    if toto_fc is not None:
        toto_fc["datetime"] = pd.to_datetime(toto_fc["datetime"], format="mixed")
        toto_fc["date"] = toto_fc["datetime"].dt.date
        toto_daily = toto_fc.groupby("date")["forecast_median"].sum()
        toto_daily_lower = toto_fc.groupby("date")["forecast_q10"].sum()
        toto_daily_upper = toto_fc.groupby("date")["forecast_q90"].sum()

        # Align with actual daily dates
        actual_dates = pd.to_datetime(actuals["datetime"], format="mixed").dt.date
        toto_values = np.array([toto_daily.get(d, 0) for d in actual_dates])
        m = compute_metrics(actual_values, toto_values, "TOTO-2.5B")
        # Calibration on daily intervals
        toto_lower = np.array([toto_daily_lower.get(d, 0) for d in actual_dates])
        toto_upper = np.array([toto_daily_upper.get(d, 0) for d in actual_dates])
        cal = compute_calibration(actual_values, toto_lower, toto_upper, nominal=0.80)
        m.update(cal)
        metrics.append(m)
        cal_str = f", Coverage={m['Coverage%']:.1f}%, ECE={m['ECE%']:.1f}%, Winkler={m['Winkler']:.0f}" if cal else ""
        print(f"  {'TOTO-2.5B':8s}: MAE={m['MAE']:8.1f}, RMSE={m['RMSE']:8.1f}, "
              f"MAPE={m['MAPE%']:6.1f}%, sMAPE={m['sMAPE%']:6.1f}%{cal_str}")

    return metrics


def plot_comparison():
    """Generate comparison plots."""
    print("\n=== Generating Plots ===\n")

    # 1. Hourly: actual vs forecasts (first 168 hours = 1 week)
    actuals = load_actuals("hourly")
    r_fcs = load_r_forecasts("hourly")
    toto_fc = load_toto_forecasts()

    if actuals is None:
        print("  No data for plots")
        return

    n = min(168, len(actuals))  # First week
    dates = pd.to_datetime(actuals["datetime"].head(n), format="mixed", dayfirst=False)
    actual_values = actuals["actual"].head(n).values

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates, actual_values, "k-", linewidth=1.5, label="Actual", alpha=0.8)

    colors = {"sarima": "#e74c3c", "tbats": "#3498db", "naive": "#2ecc71"}
    for model_name, fc_df in r_fcs.items():
        fc_values = fc_df["forecast"].values[:n]
        ax.plot(dates, fc_values, color=colors.get(model_name, "gray"),
                linewidth=1, label=model_name.upper(), alpha=0.7)

    if toto_fc is not None:
        toto_values = toto_fc["forecast_median"].values[:n]
        ax.plot(dates, toto_values, color="#9b59b6", linewidth=1,
                label="TOTO-2.5B", alpha=0.7, linestyle="--")

        # Add confidence interval
        q10 = toto_fc["forecast_q10"].values[:n]
        q90 = toto_fc["forecast_q90"].values[:n]
        ax.fill_between(dates, q10, q90, alpha=0.15, color="#9b59b6",
                        label="TOTO 80% CI")

    ax.set_xlabel("Date")
    ax.set_ylabel("Vehicle Count (per hour)")
    ax.set_title("Hourly Traffic Forecast: First Week of Test Period")
    ax.legend(fontsize=8, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "hourly_comparison.png", dpi=150)
    plt.close()
    print("  Saved: hourly_comparison.png")

    # 2. Daily pattern: average traffic by hour of day (actual vs forecast)
    # Load full Toto forecasts to get hourly breakdown
    if toto_fc is not None:
        toto_full = pd.read_csv(TOTO_DIR / "toto_forecasts.csv")
        toto_full["datetime"] = pd.to_datetime(toto_full["datetime"], format="mixed")
        toto_full["hour_of_day"] = toto_full["datetime"].dt.hour

        actual_full = load_actuals("hourly")
        if actual_full is not None:
            actual_full["datetime"] = pd.to_datetime(actual_full["datetime"], format="mixed")
            actual_full["hour_of_day"] = actual_full["datetime"].dt.hour

            # Average by hour of day
            actual_by_hour = actual_full.groupby("hour_of_day")["actual"].mean()
            toto_by_hour = toto_full.groupby("hour_of_day")["forecast_median"].mean()

            fig, ax = plt.subplots(figsize=(12, 4))
            hours = sorted(set(actual_by_hour.index) | set(toto_by_hour.index))
            ax.plot(hours, actual_by_hour.reindex(hours).values, "k-o",
                    label="Actual (avg)", markersize=4)

            # Plot R models (SARIMA, TBATS, Naive)
            r_fcs = load_r_forecasts("hourly")
            colors = {"sarima": "#e74c3c", "tbats": "#3498db", "naive": "#2ecc71"}
            markers = {"sarima": "^", "tbats": "v", "naive": "d"}
            linestyles = {"sarima": ":", "tbats": "-.", "naive": "--"}
            for model_name, fc_df in r_fcs.items():
                fc_df = fc_df.copy()
                fc_df["datetime"] = pd.to_datetime(fc_df["datetime"], format="mixed")
                fc_df["hour_of_day"] = fc_df["datetime"].dt.hour
                model_by_hour = fc_df.groupby("hour_of_day")["forecast"].mean()
                ax.plot(hours, model_by_hour.reindex(hours).values,
                        marker=markers.get(model_name, "o"),
                        linestyle=linestyles.get(model_name, "-"),
                        color=colors.get(model_name, "gray"),
                        label=f"{model_name.upper()} (avg)", markersize=4, alpha=0.7)

            ax.plot(hours, toto_by_hour.reindex(hours).values, "s--",
                    color="#9b59b6", label="TOTO-2.5B (avg)", markersize=4)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Avg Vehicle Count")
            ax.set_title("Average Traffic Pattern by Hour of Day")
            ax.set_xticks(range(0, 24, 2))
            ax.legend()
            plt.tight_layout()
            fig.savefig(PLOTS_DIR / "hourly_pattern.png", dpi=150)
            plt.close()
            print("  Saved: hourly_pattern.png")

    # 3. Metrics bar chart
    hourly_metrics = compare_hourly()
    daily_metrics = compare_daily()

    if hourly_metrics:
        hm = pd.DataFrame(hourly_metrics)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, metric in enumerate(["MAE", "MAPE%"]):
            ax = axes[idx]
            bars = ax.bar(hm["model"], hm[metric], color=["#e74c3c", "#3498db",
                                                           "#2ecc71", "#9b59b6"][:len(hm)])
            ax.set_xlabel("Model")
            ax.set_ylabel(metric + (" " if metric == "MAE" else ""))
            ax.set_title(f"Hourly Forecast {metric}")
            # Add value labels
            for bar, val in zip(bars, hm[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=9)
            ax.axhline(y=hm[metric].min(), color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "metrics_comparison.png", dpi=150)
        plt.close()
        print("  Saved: metrics_comparison.png")


def save_summary(hourly_metrics=None, daily_metrics=None):
    """Save final comparison summary."""
    print("\n=== Saving Summary ===\n")

    summary = {
        "experiment": "Traffic Forecasting: R vs Toto-2.0-2.5B",
        "site": "133119702600 (OGUNQUIT 02600)",
        "train_period": "2020-01-01 to 2025-12-31",
        "test_period": "2026-01-01 to 2026-06-14",
        "models": {
            "R_SARIMA": "auto.arima with frequency=24",
            "R_MSTL_ARIMA": "stlm with msts(seasonal.periods=c(24,168))",
            "R_Naive": "Last week same-hour",
            "Toto_2.5B": "Datadog Toto-2.0-2.5B, GPU (RTX 3090), 8704h context, 160h horizon",
        },
    }

    # Load Toto metrics if available
    toto_metrics_path = TOTO_DIR / "toto_metrics.json"
    if toto_metrics_path.exists():
        summary["toto_details"] = json.loads(toto_metrics_path.read_text())

    # Include calibration metrics
    if hourly_metrics:
        for m in hourly_metrics:
            if m.get("model") == "TOTO-2.5B" and "Coverage%" in m:
                summary["calibration_hourly"] = {
                    "nominal_coverage": 80,
                    "empirical_coverage": m["Coverage%"],
                    "ece": m["ECE%"],
                    "winkler_score": m["Winkler"],
                    "mean_interval_width": m["Mean_Width"],
                }
                break
    if daily_metrics:
        for m in daily_metrics:
            if m.get("model") == "TOTO-2.5B" and "Coverage%" in m:
                summary["calibration_daily"] = {
                    "nominal_coverage": 80,
                    "empirical_coverage": m["Coverage%"],
                    "ece": m["ECE%"],
                    "winkler_score": m["Winkler"],
                    "mean_interval_width": m["Mean_Width"],
                }
                break

    (DATA_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print("  Saved: comparison_summary.json")


def main():
    print("Traffic Forecast Evaluation")
    print("=" * 40)

    hourly_metrics = compare_hourly()
    daily_metrics = compare_daily()
    plot_comparison()
    save_summary(hourly_metrics, daily_metrics)

    print("\nDone! Results in data/plots/ and data/comparison_summary.json")


if __name__ == "__main__":
    main()
