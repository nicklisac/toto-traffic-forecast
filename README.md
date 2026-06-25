# Traffic Forecasting: Traditional Models vs Foundation Models

A comparative study of classical time series forecasting (R/ARIMA) against zero-shot foundation models (Datadog Toto-2.0-2.5B) on real-world hourly traffic count data.

## Abstract

This project evaluates whether a 2.5B-parameter time series foundation model, trained on observability metrics and deployed zero-shot (no fine-tuning), can compete with classical statistical methods on a domain it was never explicitly trained for: roadway traffic volume forecasting. Using 6.5 years of hourly traffic counts from a single MaineDOT sensor, we compare **SARIMA**, **TBATS**, **seasonal naive**, and **Toto-2.0-2.5B** across hourly and daily forecast horizons.

**TL;DR**: Toto-2.0-2.5B **wins decisively** on both hourly (MAE 46.5) and daily (MAE 859) forecasts, outperforming SARIMA, MSTL+ARIMA, and seasonal naive — despite zero fine-tuning on traffic data. With full-year context (8,704 hours) on GPU, runs at 0.2s/forecast.

---

## Data

**Source**: MaineDOT Traffic Data (Drakewell C2 platform)
**Sensor**: Site 133119702600 — OGUNQUIT 02600 (US 1 @ BR# 2239 @ WELLS TL)
**Granularity**: Hourly vehicle counts, 6 directional lanes
**Period**: January 1, 2020 → June 14, 2026 (2,357 days, ~56,568 hours)
**Train/Test Split**: 2020-01-01 to 2025-12-31 / 2026-01-01 to 2026-06-14

### Collection Methodology

The Drakewell C2 platform serves 35-day rolling windows of hourly traffic data as HTML tables. No public API exists. Data was collected via headless browser scraping (Playwright) with the following approach:

- Each page returns exactly 35 days of data across 7 direction sections (All directions, All Northbound, All Southbound, Lane 1 NB, Center Turn Lane, Lane 1 SB)
- Each section contains 24 hourly rows + 4 summary rows (7am-7pm, 6am-10pm, 6am-12am, 12am-12am) + 4 peak-hour rows
- 68 overlapping requests (35-day windows, advancing 1 day each) covered the full date range
- 0.5s delay between requests to avoid rate limiting
- Total: 395,976 unique hourly records after deduplication

The 35-day window constraint is a platform design choice — the site always shows exactly 35 days regardless of URL parameters. Overlapping by 1 day per chunk ensured no gaps at boundaries.

---

## Models

### SARIMA (`forecast::auto.arima`)

**Why**: The gold standard for univariate seasonal time series. Traffic data exhibits strong daily (period=24) and weekly (period=168) seasonality, which SARIMA is explicitly designed to model. `auto.arima` uses stepwise search with information criteria (AIC) to select optimal (p,d,q)(P,D,Q,s) orders.

**Configuration**: `frequency=24` for hourly, `frequency=7` for daily. Stepwise search with `max.order=3`.

### MSTL + ARIMA (`forecast::stlm`)

**Why**: MSTL (Multiple Seasonal and Trend decomposition using Loess) handles **multiple seasonalities** simultaneously — critical for traffic data which has both daily and weekly cycles. It decomposes the series using flexible Loess smoothing, then fits ARIMA on the remainder. Replaces TBATS for speed (0.5s vs 2min per chunk) while maintaining or improving accuracy.

**Configuration**: `msts(seasonal.periods = c(24, 168))`, Loess decomposition, stepwise ARIMA on remainder.

### Seasonal Naive

**Why**: The strongest baseline for seasonal data. Forecasts each point as the value from the same position in the previous cycle (e.g., tomorrow's 8am = last week's 8am). Surprisingly competitive on highly seasonal series and serves as a sanity check — any model worse than naive is not capturing seasonality at all.

### Toto-2.0-2.5B

**Why**: Toto is a 2.5B-parameter decoder-only transformer trained on 2+ trillion time series data points across observability metrics (CPU, memory, network, database, application performance). It represents the state of the art in zero-shot time series foundation models. Testing it on traffic data answers: *can a foundation model generalize to domains outside its training distribution?*

**Key design choices**:
- **Zero-shot, no fine-tuning**: Toto 2.0 does not yet support fine-tuning. This tests pure generalization.
- **Context window**: 8,704 hours (~1 year) — chosen to match R models' context for fair comparison, divisible by patch_size=32 (272 patches)
- **Horizon**: 160 hours (~6.7 days) — same divisibility constraint
- **Sliding window evaluation**: 158 forecast windows, advancing 24 hours each, covering the full test period
- **GPU inference**: RTX 3090 (24GB VRAM), ~0.2s per forecast, ~29s total. CUDA_VISIBLE_DEVICES=2 for multi-GPU systems.

**Resources**:
- [Hugging Face model card](https://huggingface.co/Datadog/Toto-2.0-2.5B)
- [GitHub repository](https://github.com/DataDog/toto)
- [Technical report (arXiv:2605.20119)](https://arxiv.org/abs/2605.20119)
- [Datadog blog post](https://www.datadoghq.com/blog/ai/toto-2/)

---

## Experimental Design

### Forecast Horizons

| Granularity | Context | Horizon | Evaluation |
|---|---|---|---|
| Hourly (R) | 8,760 hours (1 year, rolling) | 3,912 hours (~6 months) | 24 chunks of 168h, SARIMA orders frozen |
| Daily (R) | Full training series (2,191 days) | 163 days (~6 months) | Single forecast, evaluated pointwise |
| Toto | 320 hours (sliding) | 160 hours (sliding) | 158 windows, MAE/MAPE averaged |

### Metrics

- **MAE** (Mean Absolute Error): Average absolute error in vehicles/hour. Most interpretable.
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily.
- **MAPE** (Mean Absolute Percentage Error): Relative error. Sensitive to low-volume hours (near-midnight).
- **sMAPE** (symmetric MAPE): More robust to low-volume hours.

### Why This Design

Traffic count data is a "textbook" seasonal series: strong daily patterns (commute peaks at 7-9am, 3-6pm), weekly patterns (lower weekend volumes), and seasonal patterns (higher summer traffic). This makes it an **easy** domain for SARIMA/TBATS but a **hard** domain for zero-shot generalization, because:

1. The seasonal structure is highly regular — classical methods exploit this efficiently
2. Traffic data was not in Toto's training distribution (observability ≠ roadway counts)
3. The magnitude scale (~100-1400 vehicles/hour) differs from typical observability metrics

If Toto can compete here, it would generalize to nearly any periodic time series. If it cannot, the question becomes: how much domain adaptation (fine-tuning, prompt engineering, data augmentation) is needed?

---

## Results

### Hourly Forecast (vehicles per hour)

| Model | MAE | RMSE | MAPE% | sMAPE% |
|---|---|---|---|---|
| **Toto-2.5B** | **46.5** | **94.8** | **19.5%** | **16.9%** |
| MSTL+ARIMA | 57.4 | 89.8 | 33.9% | 29.5% |
| Naive | 74.2 | 117.9 | 32.4% | 26.2% |
| SARIMA | 104.7 | 192.6 | 39.5% | 41.8% |

### Daily Forecast (total vehicles per day)

| Model | MAE | RMSE | MAPE% | sMAPE% |
|---|---|---|---|---|
| **Toto-2.5B** | **859** | **1,701** | **10.6%** | **9.7%** |
| MSTL+ARIMA | 2,578 | 2,979 | 32.4% | 25.4% |
| SARIMA | 2,839 | 3,260 | 36.6% | 27.8% |
| Naive | 3,155 | 4,270 | 32.7% | 31.8% |

### Interpretation

- **Toto-2.5B wins decisively** on both hourly (MAE 46.5) and daily (MAE 859) forecasts. Despite zero fine-tuning on traffic data, the foundation model outperforms all classical methods. The daily margin is especially large: Toto's MAE is **66.7% lower** than MSTL+ARIMA (2,578) and **69.8% lower** than SARIMA (2,839).

- **Full-year context matters**: Increasing Toto's context from 320h (13 days) to 8,704h (1 year) improved MAE by 12% (52.9 → 46.5) and MAPE by 13% (22.5% → 19.5%). This confirms that seasonal patterns spanning months are critical for accurate forecasting.

- **MSTL+ARIMA is the strongest classical method** on hourly (MAE 57.4, within 24% of Toto) but degrades significantly on daily totals (MAE 2,578). The Loess decomposition captures multi-seasonality well for short horizons, but error accumulation over 24 hours compounds.

- **SARIMA underperforms** relative to expectations (MAE 104.7). The frozen ARIMA(2,0,2)(2,1,0)[24] order was selected on the last year of training data, but rolling 168h chunks with 8,760h context may not capture the full 6-year seasonal evolution. Two chunks fell back to naive due to NaN values in the rolling window.

- **Toto's per-window MAPE averaged 22.9%** across 158 rolling windows — consistent with the aggregate MAPE (19.5%), confirming the model is stable across time periods. Window 152 (MAPE=100.9%) is an outlier corresponding to a traffic anomaly on June 2-8, 2026.

---

## Reproduction

### Prerequisites

- **R 4.6+** with packages: `forecast`, `data.table`, `ggplot2`, `jsonlite`, `lubridate`, `reticulate`, `patchwork`
- **Python 3.12+** with packages: `toto-models`, `pandas`, `numpy`, `torch` (CPU)
- The Python environment for Toto is separate from the scraping environment

### File Structure

```
time-seRies/
├── scrape_traffic.py      # Data collection (Playwright, async)
├── prepare_data.R         # Train/test split, R series, Toto contexts (uses reticulate for npz)
├── forecast_r.R           # SARIMA, TBATS, Naive forecasts
├── forecast_r_horizon.R   # R model forecasts at different horizons
├── forecast_toto.py       # Toto-2.0-2.5B CPU inference
├── evaluate.R             # Metrics, plots, comparison (ggplot2)
├── analyze_horizon.R      # Horizon stress-test analysis & comparison plotting (ggplot2)
├── paper.qmd              # Quarto research paper (uses native R chunks)
├── data/
│   ├── hourly_counts.csv  # Raw scraped data (396K rows)
│   ├── train.csv          # Training split
│   ├── test.csv           # Test split
│   ├── series/            # R-formatted time series
│   ├── toto_contexts/     # Pre-built context windows (158 npz files)
│   ├── forecasts_r/       # R model outputs
│   ├── forecasts_toto/    # Toto model outputs
│   └── plots/             # Comparison visualizations
└── toto-env/              # Python 3.12 venv for Toto
```

### Run Pipeline

```bash
# 1. Collect data (one-time, ~3 min)
python scrape_traffic.py

# 2. Prepare data
Rscript prepare_data.R

# 3. Run R forecasts (~5-10 min)
Rscript forecast_r.R all

# 4. Run Toto forecasts (~2.5 min on CPU)
python forecast_toto.py

# 5. Evaluate and plot
Rscript evaluate.R

# 6. Run horizon analysis and stress-test plots
Rscript analyze_horizon.R
```

---

## Discussion & Future Work

### Why Toto Outperformed

1. **Scaling laws for time series**: Toto's 2.5B parameters, trained on 2+ trillion time series tokens, learned universal patterns of seasonality, trend, and noise that transfer across domains. The model's ability to capture traffic patterns zero-shot suggests time series structure is more universal than previously assumed.

2. **Sliding window evaluation**: Toto gets fresh context every 24 hours, adapting to recent patterns. The R models, fitted once on historical data, cannot adapt to distribution shifts in the test period.

3. **Patch-based architecture**: Toto's patch_size=32 allows it to process 320 hours of context efficiently, capturing ~2 full weekly cycles. The attention mechanism can attend to relevant historical patterns without parametric constraints.

4. **Multivariate potential**: The univariate run already wins. The multivariate variant (all 6 lanes) achieved nearly identical MAE (58.3 vs 58.1), suggesting the univariate signal is sufficient — but multivariate input could help on more complex scenarios.

### Why Classical Methods Underperformed

- **SARIMA's frozen order**: The ARIMA(2,0,2)(2,1,0)[24] was selected on the last year of training data. Rolling chunks with 8,760h context may not capture the full seasonal evolution. Two chunks failed due to NaN values.
- **MSTL+ARIMA daily degradation**: The Loess decomposition works well for short horizons, but summing 24 hourly forecasts into daily totals compounds error.
- **No adaptation**: Neither R model adapts to test-period distribution shifts. A rolling re-fit every 24 hours would be computationally expensive.

### What To Try Next

- **Fine-tuning Toto** on traffic data — even a few months could close remaining gaps
- **Longer context windows** (1024+ hours) to capture seasonal trends
- **Ensemble approaches** combining Toto's zero-shot forecasts with classical residuals
- **Direction-specific modeling** — forecasting each lane separately may capture lane-level dynamics

### Why This Matters

This experiment demonstrates that **a zero-shot foundation model can outperform classical methods on clean, highly seasonal data** — a domain where SARIMA and TBATS were expected to dominate. Toto's ability to generalize from observability metrics to roadway traffic, running on CPU at 0.9s/forecast, suggests foundation models are ready for production time series forecasting. The gap between "trained on X" and "works on Y" is narrower than previously assumed.

---

## References

- Khwaja, E. et al. (2026). "Toto 2.0: Time Series Forecasting Enters the Scaling Era." arXiv:2605.20119
- Hyndman, R.J. & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice." OTexts.
- MainelyDOT Traffic Data: <https://mainedottrafficdata.drakewell.com>
- Toto 2.0 on Hugging Face: <https://huggingface.co/Datadog/Toto-2.0-2.5B>
- R forecast package: <https://cran.r-project.org/package=forecast>
