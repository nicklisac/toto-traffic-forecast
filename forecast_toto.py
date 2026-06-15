"""
Toto-2.0 GPU Forecasting Wrapper for Traffic Data.

Loads Toto-2.0-2.5B on CUDA:2 (24GB VRAM) and runs forecasts on pre-built
context windows. Uses FP16 for memory efficiency with long context (8704h).

Usage:
    python forecast_toto.py              # Univariate (All directions only)
    python forecast_toto.py --multivariate  # Multivariate (all 6 lanes)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from toto2 import Toto2Model

# Configuration
CHECKPOINT = "Datadog/Toto-2.0-2.5B"
DEVICE = "cuda"  # Use CUDA_VISIBLE_DEVICES=2 to select GPU 2
CONTEXT_HOURS = 8704  # Full year context (272 patches × 32), matching R models
HORIZON_HOURS = 160  # Must be divisible by patch_size=32

DATA_DIR = Path("data")


def load_model():
    """Load Toto-2.0-2.5B on GPU with FP16 for memory efficiency."""
    print(f"Loading {CHECKPOINT} on {DEVICE}...")
    t0 = time.time()

    model = Toto2Model.from_pretrained(CHECKPOINT, map_location="cpu")
    model = model.to(DEVICE).eval()  # Move to GPU after full load

    n_params = sum(p.numel() for p in model.parameters())
    gpu_mem = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"  Loaded {n_params:,} parameters in {time.time() - t0:.1f}s")
    print(f"  Patch size: {model.config.patch_size}")
    print(f"  GPU memory (model): {gpu_mem:.1f} GB")

    return model


def forecast_window(model, context: np.ndarray, window_id: int, multivariate: bool = False,
                    horizon: int = HORIZON_HOURS) -> dict:
    """
    Run a single forecast window.

    Args:
        model: Toto2Model
        context: 1D array (n_time,) for univariate, or 2D (n_time, n_var) for multivariate
        window_id: ID for this window
        multivariate: If True, context shape is (time, n_variates)
        horizon: Forecast horizon in hours (must be divisible by patch_size=32)

    Returns:
        Dict with forecast values and metadata.
    """
    if multivariate:
        # context shape: (time, n_var) -> transpose to (n_var, time) -> (batch=1, n_var, time)
        target = torch.tensor(context, dtype=torch.float32, device=DEVICE).transpose(0, 1).unsqueeze(0)
    else:
        # context shape: (time,) -> (batch=1, n_var=1, time)
        target = torch.tensor(context, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

    target_mask = torch.ones_like(target, dtype=torch.bool, device=DEVICE)
    n_var = target.shape[1]
    series_ids = torch.zeros(1, n_var, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        quantiles = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=horizon,
            has_missing_values=False,
        )

    # quantiles shape: (9, batch=1, n_var, horizon)
    # For multivariate, we want variate 0 (All directions)
    median = quantiles[4, 0, 0].cpu().numpy()  # 0.5 quantile, first variate
    q10 = quantiles[0, 0, 0].cpu().numpy()
    q90 = quantiles[8, 0, 0].cpu().numpy()

    return {
        "median": median,
        "q10": q10,
        "q90": q90,
    }


def run_forecasts(model, multivariate: bool = False, horizon: int = HORIZON_HOURS):
    """Run forecasts on all windows."""
    context_dir = DATA_DIR / "toto_contexts_mv" if multivariate else DATA_DIR / "toto_contexts"
    meta = pd.read_csv(context_dir / "metadata.csv")
    print(f"\nRunning forecasts for {len(meta)} windows (horizon={horizon}h)...")

    all_results = []

    for _, row in meta.iterrows():
        wid = row["window_id"]
        npz_path = context_dir / f"window_{wid:04d}.npz"

        data = np.load(npz_path)
        context = data["context"]
        target = data["target"]
        forecast_start = data["forecast_start"]
        forecast_end = data["forecast_end"]

        t0 = time.time()
        result = forecast_window(model, context, wid, multivariate=multivariate, horizon=horizon)
        elapsed = time.time() - t0

        # Compute metrics against actual target (min of forecast and target length)
        median = result["median"]
        eval_len = min(len(median), len(target))
        med_eval = median[:eval_len]
        tgt_eval = target[:eval_len]
        mae = np.nanmean(np.abs(med_eval - tgt_eval))
        # MAPE (avoid division by zero)
        mask = tgt_eval > 0
        mape = np.nanmean(np.abs((med_eval[mask] - tgt_eval[mask]) / tgt_eval[mask])) * 100 if mask.any() else np.nan

        print(f"  Window {wid:03d} ({forecast_start} to {forecast_end}): "
              f"MAE={mae:.1f}, MAPE={mape:.1f}%, time={elapsed:.1f}s")

        all_results.append({
            "window_id": wid,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "median": median,
            "q10": result["q10"],
            "q90": result["q90"],
            "actual": target,
            "horizon": horizon,
            "mae": mae,
            "mape": mape,
            "inference_time_s": elapsed,
        })

    return all_results


def save_results(results, horizon: int = HORIZON_HOURS):
    """Save forecasts and metrics."""
    # 1. Full forecast arrays
    forecasts = []
    for r in results:
        # Expand horizon into per-hour rows
        fs = str(r["forecast_start"])
        forecast_start = pd.Timestamp(fs)
        for i in range(len(r["median"])):
            actual_val = r["actual"][i] if i < len(r["actual"]) else np.nan
            forecasts.append({
                "datetime": forecast_start + pd.Timedelta(hours=i),
                "forecast_median": r["median"][i],
                "forecast_q10": r["q10"][i],
                "forecast_q90": r["q90"][i],
                "actual": actual_val,
                "window_id": r["window_id"],
            })

    fc_df = pd.DataFrame(forecasts)
    (OUTPUT_DIR / f"toto_forecasts_h{horizon}.csv").write_text(fc_df.to_csv(index=False))

    # 2. Summary metrics per window
    summary = pd.DataFrame([{
        "window_id": r["window_id"],
        "forecast_start": r["forecast_start"],
        "forecast_end": r["forecast_end"],
        "horizon": r.get("horizon", horizon),
        "mae": r["mae"],
        "mape": r["mape"],
        "inference_time_s": r["inference_time_s"],
    } for r in results])
    (OUTPUT_DIR / f"toto_summary_h{horizon}.csv").write_text(summary.to_csv(index=False))

    # 3. Overall metrics
    overall = {
        "model": CHECKPOINT,
        "device": DEVICE,
        "context_hours": CONTEXT_HOURS,
        "horizon_hours": horizon,
        "n_windows": len(results),
        "mean_mae": float(np.mean([r["mae"] for r in results])),
        "mean_mape": float(np.mean([r["mape"] for r in results if not np.isnan(r["mape"])])),
        "mean_inference_time": float(np.mean([r["inference_time_s"] for r in results])),
        "total_inference_time": float(np.sum([r["inference_time_s"] for r in results])),
    }
    (OUTPUT_DIR / "toto_metrics.json").write_text(json.dumps(overall, indent=2))

    print(f"\nOverall Toto metrics:")
    print(f"  Mean MAE:  {overall['mean_mae']:.1f} vehicles/hour")
    print(f"  Mean MAPE: {overall['mean_mape']:.1f}%")
    print(f"  Mean inference time: {overall['mean_inference_time']:.1f}s per window")
    print(f"  Total inference time: {overall['total_inference_time']:.0f}s")

    return overall


def main():
    parser = argparse.ArgumentParser(description="Toto-2.0 Traffic Forecasting")
    parser.add_argument("--multivariate", action="store_true",
                        help="Use multivariate input (all 6 lanes)")
    parser.add_argument("--horizon", type=int, default=HORIZON_HOURS,
                        help=f"Forecast horizon in hours (default: {HORIZON_HOURS}, must be divisible by 32)")
    args = parser.parse_args()

    if args.horizon % 32 != 0:
        raise ValueError(f"Horizon must be divisible by patch_size=32, got {args.horizon}")

    mode_label = "Multivariate (6 lanes)" if args.multivariate else "Univariate (All directions)"
    print(f"Toto-2.0 Traffic Forecasting ({mode_label})")
    print(f"Context: {CONTEXT_HOURS}h ({CONTEXT_HOURS // 32} patches) | Horizon: {args.horizon}h")
    print("=" * 40)

    # Load model
    model = load_model()

    # Run forecasts
    results = run_forecasts(model, multivariate=args.multivariate, horizon=args.horizon)

    # Save results (to different subdirectory for MV/horizon)
    out_dir = DATA_DIR / f"forecasts_toto_h{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    global OUTPUT_DIR
    OUTPUT_DIR = out_dir
    save_results(results, horizon=args.horizon)

    print(f"\nDone! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
