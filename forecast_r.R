#!/usr/bin/env Rscript
# R Forecasting Script for Traffic Data (Rolling Evaluation)
# Models: SARIMA, TBATS (multi-seasonal), Naive (seasonal naive)
#
# Rolling evaluation: fit on training data, forecast 168h chunks,
# matching Toto's sliding-window setup for fair comparison.
#
# Usage: Rscript forecast_r.R [hourly|daily]

suppressWarnings(suppressMessages({
  library(forecast)
  library(data.table)
}))

# Configuration
data_dir <- "data/series"
output_dir <- "data/forecasts_r"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

direction_slug <- "all_directions"
train_end <- as.Date("2025-12-31")
test_start <- as.Date("2026-01-01")

# Rolling forecast chunk size (hours)
CHUNK_HOURS <- 168  # 1 week

##############################################################################
# Helper: Run all three models on a numeric vector, forecast h steps
##############################################################################
run_all_models <- function(values, name, h, best_order = NULL) {
  n <- length(values)
  cat(sprintf("  Modeling %s (n=%d, h=%d)...\n", name, n, h))

  results <- list()
  ts_single <- ts(values, frequency = 24)
  ts_multi <- if (n >= 168 * 2) msts(values, seasonal.periods = c(24, 168)) else ts_single

  # 1. SARIMA (single seasonality: 24h)
  tryCatch({
    cat("    SARIMA (freq=24)... ")
    if (!is.null(best_order)) {
      fit <- Arima(ts_single, order = best_order[1:3],
                   seasonal = list(order = best_order[4:6], period = 24))
    } else {
      fit <- auto.arima(ts_single, stepwise = TRUE, approximation = TRUE,
                        max.p = 3, max.d = 2, max.q = 3,
                        max.P = 2, max.D = 1, max.Q = 2, max.order = 3)
    }
    fc <- forecast(fit, h = h)
    results$sarima <- as.numeric(fc$mean)
    cat(sprintf("ok (AIC=%.0f)\n", AIC(fit)))
  }, error = function(e) {
    cat(sprintf("FAIL: %s — falling back to naive\n", e$message))
    # Fallback: use seasonal naive for this chunk
    last_cycle_start <- n - 24 + 1
    naive_fc <- numeric(h)
    for (i in seq_len(h)) {
      pos <- (i - 1) %% 24
      naive_fc[i] <- values[last_cycle_start + pos]
    }
    results$sarima <- naive_fc
  })

  # 2. MSTL + ARIMA (multi-seasonal: 24h + 168h) - Fast TBATS replacement
  tryCatch({
    cat("    MSTL+ARIMA (freq=24+168)... ")
    fit <- stlm(ts_multi, method = "arima",
                approximation = TRUE,
                stepwise = TRUE)
    fc <- forecast(fit, h = h)
    results$tbats <- as.numeric(fc$mean)
    cat("ok\n")
  }, error = function(e) {
    cat(sprintf("FAIL: %s\n", e$message))
  })

  # 3. Seasonal Naive: repeat last 24h cycle
  tryCatch({
    cat("    Naive (24h cycle)... ")
    naive_fc <- numeric(h)
    last_cycle_start <- n - 24 + 1
    for (i in seq_len(h)) {
      pos <- (i - 1) %% 24
      naive_fc[i] <- values[last_cycle_start + pos]
    }
    results$naive <- naive_fc
    cat("ok\n")
  }, error = function(e) {
    cat(sprintf("FAIL: %s\n", e$message))
  })

  return(results)
}

##############################################################################
# Rolling Hourly Forecast
# Fit on training data, forecast 168h chunks across the test period
##############################################################################
forecast_hourly_rolling <- function() {
  cat("\n=== Rolling Hourly Forecast (168h chunks) ===\n")

  df <- fread(file.path(data_dir, paste0("r_hourly_full_", direction_slug, ".csv")))
  df$datetime <- as.POSIXct(df$datetime, tz = "UTC")

  train_end_dt <- as.POSIXct(paste0(train_end, " 23:59:59"), tz = "UTC")
  test_start_dt <- as.POSIXct(paste0(test_start, " 00:00:00"), tz = "UTC")

  all_vals <- df$value
  all_dates <- df$datetime

  train_mask <- all_dates <= train_end_dt
  test_mask <- all_dates >= test_start_dt

  train_vals <- all_vals[train_mask]
  test_vals <- all_vals[test_mask]
  test_dates <- all_dates[test_mask]

  n_test <- length(test_vals)
  n_train <- length(train_vals)
  cat(sprintf("  Train: %d hours, Test: %d hours (%d chunks of %dh)\n",
              n_train, n_test, ceiling(n_test / CHUNK_HOURS), CHUNK_HOURS))

  # Storage for all forecasts
  all_fc <- list(sarima = numeric(n_test), tbats = numeric(n_test), naive = numeric(n_test))

  # Find optimal SARIMA order once on recent training data (saves stepwise search per chunk)
  cat("Finding optimal SARIMA order on training data...\n")
  initial_ts <- ts(tail(train_vals, 8760), frequency = 24)
  best_fit <- auto.arima(initial_ts, stepwise = TRUE, approximation = TRUE)
  best_order <- arimaorder(best_fit)
  cat(sprintf("  Best order: ARIMA(%d,%d,%d)(%d,%d,%d)[24]\n",
              best_order[1], best_order[2], best_order[3],
              best_order[4], best_order[5], best_order[6]))

  # Rolling: fit on training + previously seen test data, forecast next chunk
  for (chunk in seq(0, n_test - 1, by = CHUNK_HOURS)) {
    chunk_end <- min(chunk + CHUNK_HOURS, n_test)
    chunk_size <- chunk_end - chunk

    # Context: training data + test data up to this chunk
    context <- c(train_vals, test_vals[seq_len(chunk)])

    # Limit to last 1 year (8760h) - models only need recent patterns
    recent_context <- tail(context, 8760)

    cat(sprintf("\n  Chunk %d: hours %d-%d (context=%d, recent=%d)\n",
                chunk %/% CHUNK_HOURS, chunk + 1, chunk_end, length(context), length(recent_context)))

    results <- run_all_models(recent_context, "rolling", chunk_size, best_order = best_order)

    for (model_name in names(results)) {
      if (length(results[[model_name]]) >= chunk_size) {
        all_fc[[model_name]][(chunk + 1):chunk_end] <- results[[model_name]][seq_len(chunk_size)]
      }
    }
  }

  # Save results
  actual_df <- data.table(datetime = as.character(test_dates), actual = test_vals)
  fwrite(actual_df, file.path(output_dir, "actuals_hourly.csv"))

  for (model_name in names(all_fc)) {
    out <- data.table(
      datetime = as.character(test_dates),
      forecast = all_fc[[model_name]],
      model = model_name
    )
    fwrite(out, file.path(output_dir, paste0(model_name, "_hourly.csv")))
    cat(sprintf("  Saved %s_hourly (%d points)\n", model_name, n_test))
  }
}

##############################################################################
# Daily Forecast (single-shot, manageable horizon)
##############################################################################
forecast_daily <- function() {
  cat("\n=== Daily Forecast (weekly seasonality) ===\n")

  df <- fread(file.path(data_dir, paste0("r_daily_totals_", direction_slug, ".csv")))
  df$date <- as.Date(df$date)

  train_df <- df[date <= train_end]
  test_df <- df[date >= test_start]

  n <- nrow(test_df)
  cat(sprintf("  Train: %d days, Test: %d days\n", nrow(train_df), n))

  results <- run_all_models(train_df$total_count, "daily", n)

  # Save
  actual_df <- data.table(datetime = as.character(test_df$date), actual = test_df$total_count)
  fwrite(actual_df, file.path(output_dir, "actuals_daily.csv"))

  for (model_name in names(results)) {
    fc <- results[[model_name]]
    if (length(fc) == 0) next
    len <- min(length(fc), n)
    out <- data.table(
      datetime = as.character(test_df$date[seq_len(len)]),
      forecast = fc[seq_len(len)],
      model = model_name
    )
    fwrite(out, file.path(output_dir, paste0(model_name, "_daily.csv")))
    cat(sprintf("  Saved %s_daily (%d points)\n", model_name, len))
  }
}

##############################################################################
# Main
##############################################################################
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  mode <- if (length(args) == 0) "all" else args[1]

  cat("R Traffic Forecasting (Rolling Evaluation)\n")
  cat("==========================================\n")
  cat(sprintf("Direction: %s\n", direction_slug))

  if (mode %in% c("hourly", "all")) forecast_hourly_rolling()
  if (mode %in% c("daily", "all")) forecast_daily()

  cat("\nDone! Forecasts saved to", output_dir, "\n")
}

main()
