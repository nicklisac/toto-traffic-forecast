#!/usr/bin/env Rscript
# Evaluation framework: Compare R forecasts vs Toto-2.0 forecasts.
#
# Produces metrics tables and comparison plots.

suppressWarnings(suppressMessages({
  library(data.table)
  library(jsonlite)
  library(ggplot2)
  library(scales)
}))

DATA_DIR <- "data"
R_DIR <- file.path(DATA_DIR, "forecasts_r")
TOTO_DIR <- file.path(DATA_DIR, "forecasts_toto")
PLOTS_DIR <- file.path(DATA_DIR, "plots")
dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

compute_metrics <- function(actual, forecast, name = "") {
  # Keep only finite, non-NA values where both are present
  mask <- is.finite(actual) & is.finite(forecast)
  act <- actual[mask]
  fc <- forecast[mask]
  
  if (length(act) == 0) {
    return(list())
  }
  
  errors <- act - fc
  abs_errors <- abs(errors)
  
  mae <- mean(abs_errors)
  rmse <- sqrt(mean(errors^2))
  
  # Avoid division by zero
  actual_denom <- ifelse(act == 0, 1, act)
  mape <- mean(abs_errors / actual_denom) * 100
  
  sum_denom <- ifelse(act + fc == 0, 1, act + fc)
  smape <- mean(2 * abs_errors / sum_denom) * 100
  
  return(list(
    model = name,
    n = length(act),
    MAE = mae,
    RMSE = rmse,
    `MAPE%` = mape,
    `sMAPE%` = smape
  ))
}

compute_calibration <- function(actual, lower, upper, nominal = 0.80) {
  mask <- is.finite(actual) & is.finite(lower) & is.finite(upper)
  act <- actual[mask]
  lo <- lower[mask]
  up <- upper[mask]
  
  if (length(act) == 0) {
    return(list())
  }
  
  # Empirical coverage
  inside <- (act >= lo) & (act <= up)
  empirical_cov <- mean(inside) * 100
  
  # Empirical Coverage Error
  ece <- abs(empirical_cov - nominal * 100)
  
  # Winkler Score
  width <- up - lo
  width <- ifelse(width == 0, 1e-6, width)
  winkler <- width
  
  below_idx <- act < lo
  above_idx <- act > up
  
  if (any(below_idx)) {
    winkler[below_idx] <- winkler[below_idx] + 2 * (lo[below_idx] - act[below_idx])
  }
  if (any(above_idx)) {
    winkler[above_idx] <- winkler[above_idx] + 2 * (act[above_idx] - up[above_idx])
  }
  mean_winkler <- mean(winkler)
  
  return(list(
    `Coverage%` = round(empirical_cov, 1),
    `ECE%` = round(ece, 1),
    Winkler = round(mean_winkler, 1),
    Mean_Width = round(mean(width), 1)
  ))
}

load_r_forecasts <- function(granularity = "hourly") {
  forecasts <- list()
  for (model in c("sarima", "tbats", "naive")) {
    fpath <- file.path(R_DIR, paste0(model, "_", granularity, ".csv"))
    if (file.exists(fpath)) {
      df <- fread(fpath)
      forecasts[[model]] <- df
    }
  }
  return(forecasts)
}

load_toto_forecasts <- function() {
  fpath <- file.path(TOTO_DIR, "toto_forecasts.csv")
  if (!file.exists(fpath)) {
    return(NULL)
  }
  
  df <- fread(fpath)
  df[, datetime := as.POSIXct(datetime, tz = "UTC")]
  
  # Keep freshest forecast (highest window_id) for each datetime
  df <- df[order(datetime, -window_id)]
  df <- unique(df, by = "datetime")
  return(df)
}

load_actuals <- function(granularity = "hourly") {
  fpath <- file.path(R_DIR, paste0("actuals_", granularity, ".csv"))
  if (!file.exists(fpath)) {
    return(NULL)
  }
  df <- fread(fpath)
  return(df)
}

compare_hourly <- function() {
  cat("\n=== Hourly Forecast Comparison ===\n\n")
  
  r_fcs <- load_r_forecasts("hourly")
  toto_fc <- load_toto_forecasts()
  actuals <- load_actuals("hourly")
  
  if (is.null(actuals)) {
    cat("No actuals found, skipping hourly comparison\n")
    return(list())
  }
  
  actual_values <- actuals$actual
  metrics <- list()
  
  # R models
  for (model_name in names(r_fcs)) {
    fc_df <- r_fcs[[model_name]]
    fc_values <- fc_df$forecast[1:min(nrow(fc_df), length(actual_values))]
    m <- compute_metrics(actual_values[1:length(fc_values)], fc_values, toupper(model_name))
    metrics[[length(metrics) + 1]] <- m
    cat(sprintf("  %-8s: MAE=%8.1f, RMSE=%8.1f, MAPE=%6.1f%%, sMAPE=%6.1f%%\n",
                toupper(model_name), m$MAE, m$RMSE, m$`MAPE%`, m$`sMAPE%`))
  }
  
  # Toto
  if (!is.null(toto_fc)) {
    toto_values <- toto_fc$forecast_median[1:min(nrow(toto_fc), length(actual_values))]
    m <- compute_metrics(actual_values[1:length(toto_values)], toto_values, "TOTO-2.5B")
    
    # Calibration
    if ("forecast_q10" %in% names(toto_fc) && "forecast_q90" %in% names(toto_fc)) {
      cal <- compute_calibration(
        actual_values[1:length(toto_values)],
        toto_fc$forecast_q10[1:length(toto_values)],
        toto_fc$forecast_q90[1:length(toto_values)],
        nominal = 0.80
      )
      m <- c(m, cal)
    }
    
    metrics[[length(metrics) + 1]] <- m
    cal_str <- ""
    if ("Coverage%" %in% names(m)) {
      cal_str <- sprintf(", Coverage=%.1f%%, ECE=%.1f%%, Winkler=%.0f",
                         m$`Coverage%`, m$`ECE%`, m$Winkler)
    }
    cat(sprintf("  %-8s: MAE=%8.1f, RMSE=%8.1f, MAPE=%6.1f%%, sMAPE=%6.1f%%%s\n",
                "TOTO-2.5B", m$MAE, m$RMSE, m$`MAPE%`, m$`sMAPE%`, cal_str))
  }
  
  return(metrics)
}

compare_daily <- function() {
  cat("\n=== Daily Forecast Comparison ===\n\n")
  
  r_fcs <- load_r_forecasts("daily")
  actuals <- load_actuals("daily")
  
  if (is.null(actuals)) {
    cat("No daily actuals found, skipping\n")
    return(list())
  }
  
  actual_values <- actuals$actual
  metrics <- list()
  
  for (model_name in names(r_fcs)) {
    fc_df <- r_fcs[[model_name]]
    fc_values <- fc_df$forecast[1:min(nrow(fc_df), length(actual_values))]
    m <- compute_metrics(actual_values[1:length(fc_values)], fc_values, toupper(model_name))
    metrics[[length(metrics) + 1]] <- m
    cat(sprintf("  %-8s: MAE=%8.1f, RMSE=%8.1f, MAPE=%6.1f%%, sMAPE=%6.1f%%\n",
                toupper(model_name), m$MAE, m$RMSE, m$`MAPE%`, m$`sMAPE%`))
  }
  
  # Toto: aggregate hourly forecasts to daily totals
  toto_fc <- load_toto_forecasts()
  if (!is.null(toto_fc)) {
    # Extract date part
    toto_fc[, date := as.Date(datetime)]
    
    toto_daily <- toto_fc[, .(
      forecast_median = sum(forecast_median),
      forecast_q10 = sum(forecast_q10),
      forecast_q90 = sum(forecast_q90)
    ), by = date]
    
    # Align with actual daily dates
    actual_dates <- as.Date(actuals$datetime)
    
    # Merge to align dates
    daily_aligned <- merge(data.table(date = actual_dates, actual = actual_values),
                           toto_daily, by = "date", all.x = TRUE)
    
    # Fill NAs with 0 just in case
    daily_aligned[is.na(forecast_median), forecast_median := 0]
    daily_aligned[is.na(forecast_q10), forecast_q10 := 0]
    daily_aligned[is.na(forecast_q90), forecast_q90 := 0]
    
    m <- compute_metrics(daily_aligned$actual, daily_aligned$forecast_median, "TOTO-2.5B")
    
    # Calibration on daily intervals
    cal <- compute_calibration(
      daily_aligned$actual,
      daily_aligned$forecast_q10,
      daily_aligned$forecast_q90,
      nominal = 0.80
    )
    m <- c(m, cal)
    metrics[[length(metrics) + 1]] <- m
    
    cal_str <- sprintf(", Coverage=%.1f%%, ECE=%.1f%%, Winkler=%.0f",
                       m$`Coverage%`, m$`ECE%`, m$Winkler)
    
    cat(sprintf("  %-8s: MAE=%8.1f, RMSE=%8.1f, MAPE=%6.1f%%, sMAPE=%6.1f%%%s\n",
                "TOTO-2.5B", m$MAE, m$RMSE, m$`MAPE%`, m$`sMAPE%`, cal_str))
  }
  
  return(metrics)
}

plot_comparison <- function(hourly_metrics = NULL) {
  cat("\n=== Generating Plots ===\n\n")
  
  actuals <- load_actuals("hourly")
  r_fcs <- load_r_forecasts("hourly")
  toto_fc <- load_toto_forecasts()
  
  if (is.null(actuals)) {
    cat("  No data for plots\n")
    return()
  }
  
  # 1. Hourly: actual vs forecasts (first 168 hours = 1 week)
  n <- min(168, nrow(actuals))
  
  plot_dt <- data.table(
    datetime = as.POSIXct(actuals$datetime[1:n], tz = "UTC"),
    Actual = actuals$actual[1:n]
  )
  
  # Load R model forecasts for the first week
  for (model_name in names(r_fcs)) {
    fc_df <- r_fcs[[model_name]]
    plot_dt[, (toupper(model_name)) := fc_df$forecast[1:n]]
  }
  
  # Add Toto if present
  if (!is.null(toto_fc)) {
    plot_dt[, `TOTO-2.5B` := toto_fc$forecast_median[1:n]]
    plot_dt[, q10 := toto_fc$forecast_q10[1:n]]
    plot_dt[, q90 := toto_fc$forecast_q90[1:n]]
  }
  
  # Reshape for ggplot (except intervals)
  melt_cols <- intersect(names(plot_dt), c("Actual", "SARIMA", "TBATS", "NAIVE", "TOTO-2.5B"))
  plot_melt <- melt(plot_dt, id.vars = "datetime", measure.vars = melt_cols,
                    variable.name = "Model", value.name = "Forecast")
  
  # Map model names to colors
  colors_map <- c(
    "Actual" = "black",
    "SARIMA" = "#e74c3c",
    "TBATS" = "#3498db",
    "NAIVE" = "#2ecc71",
    "TOTO-2.5B" = "#9b59b6"
  )
  
  linestyles_map <- c(
    "Actual" = "solid",
    "SARIMA" = "solid",
    "TBATS" = "solid",
    "NAIVE" = "solid",
    "TOTO-2.5B" = "dashed"
  )
  
  p1 <- ggplot()
  
  # Shaded interval for Toto
  if ("q10" %in% names(plot_dt)) {
    p1 <- p1 + geom_ribbon(data = plot_dt, aes(x = datetime, ymin = q10, ymax = q90, fill = "TOTO 80% CI"), alpha = 0.15)
  }
  
  p1 <- p1 +
    geom_line(data = plot_melt, aes(x = datetime, y = Forecast, color = Model, linetype = Model), size = 0.8) +
    scale_color_manual(values = colors_map) +
    scale_linetype_manual(values = linestyles_map) +
    scale_fill_manual(values = c("TOTO 80% CI" = "#9b59b6")) +
    scale_x_datetime(labels = date_format("%b %d"), breaks = date_breaks("2 days")) +
    labs(
      title = "Hourly Traffic Forecast: First Week of Test Period",
      x = "Date",
      y = "Vehicle Count (per hour)",
      fill = NULL
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  ggsave(file.path(PLOTS_DIR, "hourly_comparison.png"), plot = p1, width = 11, height = 4.5, dpi = 150)
  cat("  Saved: hourly_comparison.png\n")
  
  # 2. Daily pattern: average traffic by hour of day (actual vs forecast)
  if (!is.null(toto_fc)) {
    toto_full <- fread(file.path(TOTO_DIR, "toto_forecasts.csv"))
    toto_full[, datetime := as.POSIXct(datetime, tz = "UTC")]
    toto_full[, hour_of_day := hour(datetime)]
    
    actual_full <- load_actuals("hourly")
    if (!is.null(actual_full)) {
      actual_full[, datetime := as.POSIXct(datetime, tz = "UTC")]
      actual_full[, hour_of_day := hour(datetime)]
      
      actual_by_hour <- actual_full[, .(Actual = mean(actual)), by = hour_of_day]
      toto_by_hour <- toto_full[, .(TOTO_2.5B = mean(forecast_median)), by = hour_of_day]
      
      pattern_dt <- merge(actual_by_hour, toto_by_hour, by = "hour_of_day")
      
      for (model_name in names(r_fcs)) {
        fc_df <- r_fcs[[model_name]]
        fc_df <- copy(fc_df)
        fc_df[, datetime := as.POSIXct(datetime, tz = "UTC")]
        fc_df[, hour_of_day := hour(datetime)]
        model_by_hour <- fc_df[, .(val = mean(forecast)), by = hour_of_day]
        setnames(model_by_hour, "val", toupper(model_name))
        pattern_dt <- merge(pattern_dt, model_by_hour, by = "hour_of_day")
      }
      
      pattern_melt <- melt(pattern_dt, id.vars = "hour_of_day", variable.name = "Model", value.name = "Avg_Count")
      
      colors_pattern <- c(
        "Actual" = "black",
        "SARIMA" = "#e74c3c",
        "TBATS" = "#3498db",
        "NAIVE" = "#2ecc71",
        "TOTO_2.5B" = "#9b59b6"
      )
      
      p2 <- ggplot(pattern_melt, aes(x = hour_of_day, y = Avg_Count, color = Model, linetype = Model, shape = Model)) +
        geom_line(size = 0.8) +
        geom_point(size = 2) +
        scale_color_manual(values = colors_pattern) +
        scale_x_continuous(breaks = seq(0, 23, by = 2)) +
        labs(
          title = "Average Traffic Pattern by Hour of Day",
          x = "Hour of Day",
          y = "Avg Vehicle Count"
        ) +
        theme_minimal() +
        theme(
          legend.position = "bottom",
          plot.title = element_text(face = "bold", size = 12)
        )
      
      ggsave(file.path(PLOTS_DIR, "hourly_pattern.png"), plot = p2, width = 9, height = 4.5, dpi = 150)
      cat("  Saved: hourly_pattern.png\n")
    }
  }
  
  # 3. Metrics bar chart
  if (is.null(hourly_metrics)) {
    hourly_metrics <- compare_hourly()
  }
  if (length(hourly_metrics) > 0) {
    hm <- rbindlist(lapply(hourly_metrics, as.data.table), fill = TRUE)
    hm_melt <- melt(hm, id.vars = "model", measure.vars = c("MAE", "MAPE%"),
                    variable.name = "Metric", value.name = "Value")
    
    p3 <- ggplot(hm_melt, aes(x = model, y = Value, fill = model)) +
      geom_bar(stat = "identity", width = 0.6, alpha = 0.85) +
      geom_text(aes(label = sprintf("%.1f", Value)), vjust = -0.3, size = 3) +
      facet_wrap(~Metric, scales = "free_y") +
      scale_fill_manual(values = c(
        "SARIMA" = "#e74c3c",
        "TBATS" = "#3498db",
        "NAIVE" = "#2ecc71",
        "TOTO-2.5B" = "#9b59b6"
      )) +
      labs(
        title = "Hourly Forecast Error Comparison",
        x = "Model",
        y = "Error Value",
        fill = "Model"
      ) +
      theme_minimal() +
      theme(
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(face = "bold", size = 10)
      )
    
    ggsave(file.path(PLOTS_DIR, "metrics_comparison.png"), plot = p3, width = 10, height = 4.5, dpi = 150)
    cat("  Saved: metrics_comparison.png\n")
  }
}

save_summary <- function(hourly_metrics = NULL, daily_metrics = NULL) {
  cat("\n=== Saving Summary ===\n\n")
  
  summary <- list(
    experiment = "Traffic Forecasting: R vs Toto-2.0-2.5B",
    site = "133119702600 (OGUNQUIT 02600)",
    train_period = "2020-01-01 to 2025-12-31",
    test_period = "2026-01-01 to 2026-06-14",
    models = list(
      R_SARIMA = "auto.arima with frequency=24",
      R_MSTL_ARIMA = "stlm with msts(seasonal.periods=c(24,168))",
      R_Naive = "Last week same-hour",
      Toto_2.5B = "Datadog Toto-2.0-2.5B, GPU (RTX 3090), 8704h context, 160h horizon"
    )
  )
  
  # Load Toto metrics details if available
  toto_metrics_path <- file.path(TOTO_DIR, "toto_metrics.json")
  if (file.exists(toto_metrics_path)) {
    summary$toto_details <- fromJSON(readLines(toto_metrics_path, warn = FALSE))
  }
  
  # Add calibration metrics to summary
  if (!is.null(hourly_metrics)) {
    for (m in hourly_metrics) {
      if (m$model == "TOTO-2.5B" && "Coverage%" %in% names(m)) {
        summary$calibration_hourly <- list(
          nominal_coverage = 80,
          empirical_coverage = m$`Coverage%`,
          ece = m$`ECE%`,
          winkler_score = m$Winkler,
          mean_interval_width = m$Mean_Width
        )
        break
      }
    }
  }
  
  if (!is.null(daily_metrics)) {
    for (m in daily_metrics) {
      if (m$model == "TOTO-2.5B" && "Coverage%" %in% names(m)) {
        summary$calibration_daily <- list(
          nominal_coverage = 80,
          empirical_coverage = m$`Coverage%`,
          ece = m$`ECE%`,
          winkler_score = m$Winkler,
          mean_interval_width = m$Mean_Width
        )
        break
      }
    }
  }
  
  write_json(summary, file.path(DATA_DIR, "comparison_summary.json"), pretty = TRUE, auto_unbox = TRUE)
  cat("  Saved: comparison_summary.json\n")
}

main <- function() {
  cat("Traffic Forecast Evaluation (R Framework)\n")
  cat("=========================================\n")
  
  hourly_metrics <- compare_hourly()
  daily_metrics <- compare_daily()
  plot_comparison(hourly_metrics)
  save_summary(hourly_metrics, daily_metrics)
  
  cat("\nDone! Results in data/plots/ and data/comparison_summary.json\n")
}

main()
