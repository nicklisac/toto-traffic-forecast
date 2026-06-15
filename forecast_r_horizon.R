#!/usr/bin/env Rscript
# Horizon stress-test for R models: forecast at different horizons from a single origin.
# Origin: 2026-01-08 (second week of January) to match Toto window 7.
# This ensures all models have seen Jan 1-2 holiday patterns in context.
# Usage: Rscript forecast_r_horizon.R <horizon_hours>
# e.g.:  Rscript forecast_r_horizon.R 168
#        Rscript forecast_r_horizon.R 336
#        Rscript forecast_r_horizon.R 672

suppressWarnings(suppressMessages({
  library(forecast)
  library(data.table)
}))

horizon <- as.integer(commandArgs(trailingOnly = TRUE)[1])
cat(sprintf("R Model Horizon Stress-Test: h=%d (origin: 2026-01-08)\n", horizon))

data_dir <- "data/series"
output_dir <- "data/forecasts_r"

# Load data
df <- fread(file.path(data_dir, "r_hourly_full_all_directions.csv"))
df$datetime <- as.POSIXct(df$datetime, tz = "UTC")

# Origin: 2026-01-08 to match Toto window 7
origin_dt <- as.POSIXct("2026-01-08 00:00:00", tz = "UTC")
context_end_dt <- origin_dt - 3600  # hour before forecast start

# All data up to origin (training + observed test data)
history_vals <- df$value[df$datetime <= context_end_dt]
# Test actuals starting from origin
test_vals <- df$value[df$datetime >= origin_dt]
test_dates <- df$datetime[df$datetime >= origin_dt]

# Context: last 8760h (1 year) ending at origin
context <- tail(history_vals, 8760)
cat(sprintf("Context: %d hours (ending %s)\n", length(context),
            format(context_end_dt, "%Y-%m-%d %H:%M")))

# Models
cat("\nFitting models...\n")
ts_single <- ts(context, frequency = 24)
ts_multi <- msts(context, seasonal.periods = c(24, 168))

# SARIMA with frozen order
best_order <- c(2, 0, 2, 2, 1, 0)  # ARIMA(2,0,2)(2,1,0)[24]
sarima_fc <- numeric()
tbats_fc <- numeric()
naive_fc <- numeric()

cat("  SARIMA... ")
fit_sarima <- Arima(ts_single, order = best_order[1:3],
                      seasonal = list(order = best_order[4:6], period = 24))
fc_sarima <- forecast(fit_sarima, h = horizon)
sarima_fc <- as.numeric(fc_sarima$mean)
cat("ok\n")

cat("  MSTL+ARIMA... ")
fit_stlm <- stlm(ts_multi, method = "arima", approximation = TRUE, stepwise = TRUE)
fc_stlm <- forecast(fit_stlm, h = horizon)
tbats_fc <- as.numeric(fc_stlm$mean)
cat("ok\n")

cat("  Naive... ")
n_ctx <- length(context)
naive_fc <- numeric(horizon)
last_cycle_start <- n_ctx - 24 + 1
for (i in seq_len(horizon)) {
  pos <- (i - 1) %% 24
  naive_fc[i] <- context[last_cycle_start + pos]
}
cat("ok\n")

# Save forecasts
dates <- origin_dt + (0:(horizon - 1)) * 3600
actuals <- test_vals[1:min(horizon, length(test_vals))]

out <- data.table(
  datetime = as.character(dates),
  sarima = sarima_fc,
  tbats = tbats_fc,
  naive = naive_fc,
  actual = c(actuals, rep(NA, max(0, horizon - length(actuals))))
)

fwrite(out, file.path(output_dir, paste0("horizon_stress_h", horizon, ".csv")))
cat(sprintf("\nSaved: horizon_stress_h%d.csv (%d hours)\n", horizon, horizon))
