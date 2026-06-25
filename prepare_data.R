#!/usr/bin/env Rscript
# Prepare traffic data for forecasting experiments.
#
# Takes raw data/hourly_counts.csv and produces:
# - data/train.csv  (2020-01-01 to 2025-12-31)
# - data/test.csv   (2026-01-01 to 2026-06-14)
# - data/series/    Per-direction, per-hour-of-day time series for R
# - data/toto_contexts/  Pre-built context windows for Toto (numpy npz files)
# - data/toto_contexts_mv/ Pre-built multivariate contexts

suppressWarnings(suppressMessages({
  library(data.table)
  library(lubridate)
  library(reticulate)
}))

# Initialize reticulate and import numpy
np <- import("numpy", convert = FALSE)

DATA_DIR <- "data"
SERIES_DIR <- file.path(DATA_DIR, "series")
dir.create(SERIES_DIR, recursive = TRUE, showWarnings = FALSE)

TRAIN_END <- as.Date("2025-12-31")
TEST_START <- as.Date("2026-01-01")

TOTO_CONTEXT_HOURS <- 8704
TOTO_HORIZON_HOURS <- 160

# Map "12:00 am" style hour strings to integer 0-23
hour_map <- c(
  "12:00 am" = 0, "01:00 am" = 1, "02:00 am" = 2, "03:00 am" = 3,
  "04:00 am" = 4, "05:00 am" = 5, "06:00 am" = 6, "07:00 am" = 7,
  "08:00 am" = 8, "09:00 am" = 9, "10:00 am" = 10, "11:00 am" = 11,
  "12:00 pm" = 12, "01:00 pm" = 13, "02:00 pm" = 14, "03:00 pm" = 15,
  "04:00 pm" = 16, "05:00 pm" = 17, "06:00 pm" = 18, "07:00 pm" = 19,
  "08:00 pm" = 20, "09:00 pm" = 21, "10:00 pm" = 22, "11:00 pm" = 23
)

load_and_clean <- function() {
  # Load raw CSV
  df <- fread(file.path(DATA_DIR, "hourly_counts.csv"))
  
  # Keep only pure hourly rows (have :00 in hour label)
  hourly <- df[grepl(":00", hour)]
  
  # Convert count to numeric, drop NAs
  hourly[, count := as.numeric(count)]
  hourly <- hourly[!is.na(count)]
  
  # Parse date
  hourly[, date := as.Date(date)]
  
  # Extract hour of day as integer (0-23)
  hourly[, hour_of_day := hour_map[hour]]
  hourly <- hourly[!is.na(hour_of_day)]
  
  # Create datetime index (force UTC timezone to match Python)
  hourly[, datetime := as.POSIXct(paste(date, sprintf("%02d:00:00", hour_of_day)), tz = "UTC")]
  
  return(hourly)
}

prepare_r_series <- function(df, target_direction = "All directions") {
  dir_data <- df[direction == target_direction][order(datetime)]
  slug <- tolower(gsub("/", "_", gsub(" ", "_", target_direction)))
  
  # 1. Full hourly series
  hourly_flat <- dir_data[, .(datetime = format(datetime, "%Y-%m-%d %H:%M:%S"), value = count)]
  fwrite(hourly_flat, file.path(SERIES_DIR, paste0("r_hourly_full_", slug, ".csv")))
  
  # 2. Daily totals
  daily <- dir_data[, .(
    total_count = sum(count),
    peak_count = max(count)
  ), by = .(date = format(date, "%Y-%m-%d"))]
  fwrite(daily, file.path(SERIES_DIR, paste0("r_daily_totals_", slug, ".csv")))
  
  # 3. Per-hour-of-day series
  for (h in 0:23) {
    h_data <- dir_data[hour_of_day == h, .(date = format(date, "%Y-%m-%d"), count)]
    setnames(h_data, "count", sprintf("hour_%02d", h))
    h_data <- h_data[order(date)]
    fwrite(h_data, file.path(SERIES_DIR, sprintf("r_hour_%02d_%s.csv", h, slug)))
  }
  
  # 4. Wide format: each column is an hour of the day, rows are dates
  wide <- dcast(dir_data, date ~ hour_of_day, value.var = "count", fill = 0)
  # Rename columns to h00, h01, etc.
  hour_cols <- setdiff(names(wide), "date")
  new_names <- sprintf("h%02d", as.integer(hour_cols))
  setnames(wide, hour_cols, new_names)
  wide[, date := format(date, "%Y-%m-%d")]
  fwrite(wide, file.path(SERIES_DIR, paste0("r_wide_hourly_", slug, ".csv")))
  
  return(list(hourly_flat = hourly_flat, daily = daily))
}

prepare_toto_contexts <- function(df, target_direction = "All directions") {
  dir_data <- df[direction == target_direction][order(datetime)]
  
  # Create complete hourly index (fill missing hours with 0)
  min_dt <- min(dir_data$datetime)
  max_dt <- max(dir_data$datetime)
  full_seq <- seq(min_dt, max_dt, by = "hour")
  
  full_dt <- data.table(datetime = full_seq)
  series_dt <- merge(full_dt, dir_data[, .(datetime, count)], by = "datetime", all.x = TRUE)
  series_dt[is.na(count), count := 0]
  
  series_values <- series_dt$count
  series_dates <- series_dt$datetime
  
  test_start_dt <- as.POSIXct(paste(TEST_START, "00:00:00"), tz = "UTC")
  
  # Find where test period starts
  test_idx <- which(series_dates >= test_start_dt)[1]
  if (is.na(test_idx)) {
    stop("Test start date not found in series dates.")
  }
  
  # Build rolling windows: slide through test set
  windows <- list()
  idx_seq <- seq(test_idx, length(series_values) - TOTO_HORIZON_HOURS + 1, by = 24)
  
  j <- 0
  toto_dir <- file.path(DATA_DIR, "toto_contexts")
  dir.create(toto_dir, recursive = TRUE, showWarnings = FALSE)
  
  metadata <- list()
  
  for (i in idx_seq) {
    context_start_idx <- i - TOTO_CONTEXT_HOURS
    if (context_start_idx < 1) next
    
    # 1-based indexing in R, context_start_idx to i-1
    context_vals <- series_values[context_start_idx:(i - 1)]
    target_vals <- series_values[i:(i + TOTO_HORIZON_HOURS - 1)]
    
    context_start_dt <- series_dates[context_start_idx]
    context_end_dt <- series_dates[i - 1]
    forecast_start_dt <- series_dates[i]
    forecast_end_dt <- series_dates[i + TOTO_HORIZON_HOURS - 1]
    
    # Save as numpy npz file using reticulate
    npz_path <- file.path(toto_dir, sprintf("window_%04d.npz", j))
    
    # Convert R vectors to numpy arrays explicitly via reticulate
    np_context <- r_to_py(context_vals)
    np_target <- r_to_py(target_vals)
    
    np$savez(
      npz_path,
      context = np_context,
      target = np_target,
      forecast_start = format(forecast_start_dt, "%Y-%m-%d %H:%M:%S"),
      forecast_end = format(forecast_end_dt, "%Y-%m-%d %H:%M:%S")
    )
    
    metadata[[length(metadata) + 1]] <- data.table(
      window_id = j,
      context_start = format(context_start_dt, "%Y-%m-%d %H:%M:%S"),
      context_end = format(context_end_dt, "%Y-%m-%d %H:%M:%S"),
      forecast_start = format(forecast_start_dt, "%Y-%m-%d %H:%M:%S"),
      forecast_end = format(forecast_end_dt, "%Y-%m-%d %H:%M:%S")
    )
    
    j <- j + 1
  }
  
  meta_df <- rbindlist(metadata)
  fwrite(meta_df, file.path(toto_dir, "metadata.csv"))
  cat(sprintf("Created %d Toto forecast windows\n", j))
}

prepare_toto_contexts_multivariate <- function(df, directions = NULL) {
  if (is.null(directions)) {
    directions <- c(
      "All directions", "All Northbound", "All Southbound",
      "Ln 1 NB", "Center Turn Lane", "Ln 1 SB"
    )
  }
  
  min_dt <- min(df$datetime)
  max_dt <- max(df$datetime)
  full_seq <- seq(min_dt, max_dt, by = "hour")
  
  # Stack into (n_hours, n_variates) array
  full_dt <- data.table(datetime = full_seq)
  
  stacked_series <- matrix(0, nrow = length(full_seq), ncol = length(directions))
  
  for (idx in seq_along(directions)) {
    d <- directions[idx]
    dir_data <- df[direction == d][order(datetime)]
    merged <- merge(full_dt, dir_data[, .(datetime, count)], by = "datetime", all.x = TRUE)
    merged[is.na(count), count := 0]
    stacked_series[, idx] <- merged$count
  }
  
  test_start_dt <- as.POSIXct(paste(TEST_START, "00:00:00"), tz = "UTC")
  test_idx <- which(full_seq >= test_start_dt)[1]
  if (is.na(test_idx)) {
    stop("Test start date not found.")
  }
  
  j <- 0
  toto_mv_dir <- file.path(DATA_DIR, "toto_contexts_mv")
  dir.create(toto_mv_dir, recursive = TRUE, showWarnings = FALSE)
  
  metadata <- list()
  idx_seq <- seq(test_idx, nrow(stacked_series) - TOTO_HORIZON_HOURS + 1, by = 24)
  
  for (i in idx_seq) {
    context_start_idx <- i - TOTO_CONTEXT_HOURS
    if (context_start_idx < 1) next
    
    context_vals <- stacked_series[context_start_idx:(i - 1), ] # (context_hours, n_var)
    target_vals <- stacked_series[i:(i + TOTO_HORIZON_HOURS - 1), 1] # Target is variate 1 ("All directions") only
    
    context_start_dt <- full_seq[context_start_idx]
    context_end_dt <- full_seq[i - 1]
    forecast_start_dt <- full_seq[i]
    forecast_end_dt <- full_seq[i + TOTO_HORIZON_HOURS - 1]
    
    npz_path <- file.path(toto_mv_dir, sprintf("window_%04d.npz", j))
    
    np_context <- r_to_py(context_vals)
    np_target <- r_to_py(target_vals)
    
    np$savez(
      npz_path,
      context = np_context,
      target = np_target,
      forecast_start = format(forecast_start_dt, "%Y-%m-%d %H:%M:%S"),
      forecast_end = format(forecast_end_dt, "%Y-%m-%d %H:%M:%S")
    )
    
    metadata[[length(metadata) + 1]] <- data.table(
      window_id = j,
      context_start = format(context_start_dt, "%Y-%m-%d %H:%M:%S"),
      context_end = format(context_end_dt, "%Y-%m-%d %H:%M:%S"),
      forecast_start = format(forecast_start_dt, "%Y-%m-%d %H:%M:%S"),
      forecast_end = format(forecast_end_dt, "%Y-%m-%d %H:%M:%S")
    )
    
    j <- j + 1
  }
  
  meta_df <- rbindlist(metadata)
  fwrite(meta_df, file.path(toto_mv_dir, "metadata.csv"))
  cat(sprintf("Created %d multivariate Toto windows (%d variates)\n", j, length(directions)))
}

main <- function() {
  cat("Loading and cleaning data...\n")
  df <- load_and_clean()
  
  cat(sprintf("  %d hourly records, %d directions\n", nrow(df), uniqueN(df$direction)))
  cat(sprintf("  Date range: %s to %s\n", min(df$date), max(df$date)))
  
  cat("\nSplitting train/test...\n")
  train <- df[date <= TRAIN_END]
  test <- df[date >= TEST_START]
  
  fwrite(train, file.path(DATA_DIR, "train.csv"))
  fwrite(test, file.path(DATA_DIR, "test.csv"))
  cat(sprintf("  Train: %d records (%s to %s)\n", nrow(train), min(train$date), max(train$date)))
  cat(sprintf("  Test:  %d records (%s to %s)\n", nrow(test), min(test$date), max(test$date)))
  
  cat("\nPreparing R series...\n")
  # Combine train+test for continuous series (R script does its own split)
  all_data <- rbind(train, test)
  prepare_r_series(all_data, "All directions")
  for (d in c("All Northbound", "All Southbound")) {
    dir_data <- all_data[direction == d]
    prepare_r_series(dir_data, d)
  }
  cat("  R series saved to data/series/\n")
  
  cat("\nPreparing Toto contexts (univariate)...\n")
  prepare_toto_contexts(df, "All directions")
  cat("  Toto contexts saved to data/toto_contexts/\n")
  
  cat("\nPreparing Toto contexts (multivariate, 6 lanes)...\n")
  prepare_toto_contexts_multivariate(df)
  cat("  Toto MV contexts saved to data/toto_contexts_mv/\n")
  
  cat("\nDone!\n")
}

main()
