"""
Compare error growth by horizon depth: R models (AR) vs Toto (CPM).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load actuals for lookup beyond 160h
actuals_df = pd.read_csv(DATA_DIR / "forecasts_r" / "actuals_hourly.csv")
actuals_df["datetime"] = pd.to_datetime(actuals_df["datetime"], format="mixed", utc=True)
actuals_df = actuals_df.set_index("datetime")

segments = {
    "0-167h\n(1 week)": (0, 168),
    "168-335h\n(2 weeks)": (168, 336),
    "336-503h\n(3 weeks)": (336, 504),
    "504-671h\n(4 weeks)": (504, 672),
}

# Load R model forecasts
r_results = {}
for model in ["sarima", "tbats", "naive"]:
    r_results[model] = {}
    for horizon in [168, 336, 672]:
        fpath = DATA_DIR / "forecasts_r" / f"horizon_stress_h{horizon}.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        for seg_name, (start, end) in segments.items():
            if end > horizon:
                continue
            fc = df[model][start:end].values
            actual = df["actual"][start:end].values
            mask = ~np.isnan(actual) & ~np.isnan(fc)
            if mask.any():
                mae = np.mean(np.abs(actual[mask] - fc[mask]))
                r_results[model][seg_name] = mae

# Load Toto forecasts -- single origin (window_id=0, starts 2026-01-01) to match R models
toto_results = {}
for horizon in [160, 320, 672]:
    # h160 is in data/forecasts_toto/, others in data/forecasts_toto_h{horizon}/
    if horizon == 160:
        fpath = DATA_DIR / "forecasts_toto" / "toto_forecasts.csv"
    else:
        fpath = DATA_DIR / f"forecasts_toto_h{horizon}" / f"toto_forecasts_h{horizon}.csv"
    if not fpath.exists():
        continue
    df = pd.read_csv(fpath)
    # Use window_id == 7 (forecast_start = 2026-01-08) to match R model origin exactly
    w = df[df["window_id"] == 7].sort_values("datetime")
    if len(w) == 0:
        continue

    for seg_name, (start, end) in segments.items():
        if start >= horizon:
            continue
        seg_end = min(end, horizon, len(w))
        fc = w["forecast_median"].iloc[start:seg_end].values
        actual = w["actual"].iloc[start:seg_end].values.copy()
        # Look up missing actuals from test data
        for i in range(len(actual)):
            if np.isnan(actual[i]):
                dt = w.iloc[start + i]["datetime"]
                try:
                    actual[i] = actuals_df.loc[dt, "actual"]
                except (KeyError, TypeError):
                    pass
        mask = ~np.isnan(actual) & ~np.isnan(fc)
        if mask.any():
            mae = np.mean(np.abs(actual[mask] - fc[mask]))
            if "Toto-2.5B" not in toto_results:
                toto_results["Toto-2.5B"] = {}
            toto_results["Toto-2.5B"][seg_name] = mae

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

seg_names = list(segments.keys())
x_pos = np.arange(len(seg_names))
width = 0.2

models = {
    "SARIMA": {"color": "#e74c3c", "marker": "o", "key": "sarima"},
    "MSTL+ARIMA": {"color": "#3498db", "marker": "s", "key": "tbats"},
    "Naive": {"color": "#2ecc71", "marker": "^", "key": "naive"},
    "Toto-2.5B": {"color": "#9b59b6", "marker": "D", "key": "Toto-2.5B"},
}

offsets = [-1.5, -0.5, 0.5, 1.5]
for idx, (model, style) in enumerate(models.items()):
    key = style["key"]
    if key not in r_results and key not in toto_results:
        continue
    vals = []
    for seg in seg_names:
        if key in r_results and seg in r_results[key]:
            vals.append(r_results[key][seg])
        elif key in toto_results and seg in toto_results[key]:
            vals.append(toto_results[key][seg])
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

print("Saved: horizon_comparison.png")

# Print table
print("\nMAE by horizon segment:")
print(f"{'Model':<15}", end="")
for seg in seg_names:
    print(f"{seg.replace(chr(10), ' '):>12}", end="")
print()
for model, style in models.items():
    key = style["key"]
    if key not in r_results and key not in toto_results:
        continue
    print(f"{model:<15}", end="")
    for seg in seg_names:
        v = None
        if key in r_results and seg in r_results[key]:
            v = r_results[key][seg]
        elif key in toto_results and seg in toto_results[key]:
            v = toto_results[key][seg]
        if v is not None:
            print(f"{v:>12.1f}", end="")
        else:
            print(f"{'N/A':>12}", end="")
    print()
