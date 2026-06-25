#!/usr/bin/env Rscript
# Horizon stress-test analysis: Measure how MAE grows with forecast horizon depth.
#
# Two analyses:
# 1. CPM Stress-Test (Toto-only): How error grows across all rolling windows at different horizons.
# 2. Single-Origin Comparison (R + Toto): All models fitted once on 8,760h context.
#    Uses Toto window_id == 7 (2026-01-08 origin) to match R model origin exactly.

suppressWarnings(suppressMessages({
  library(data.table)
  library(ggplot2)
}))

DATA_DIR <- "data"
PLOTS_DIR <- file.path(DATA_DIR, "plots")
dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

# Load actual test data for lookup beyond forecast window
actuals_df <- fread(file.path(DATA_DIR, "forecasts_r", "actuals_hourly.csv"))
actuals_df[, datetime := as.POSIXct(datetime, tz = "UTC")]
setkey(actuals_df, datetime)

# Segment definitions (hour ranges, 0-indexed in forecast step but 1-based in indices)
# We will define the start and end as 1-based indices in R (1-168, 169-336, etc.)
SEGMENTS <- list(
  "0-167h\n(1 week)" = c(1, 168),
  "168-335h\n(2 weeks)" = c(169, 336),
  "336-503h\n(3 weeks)" = c(337, 504),
  "504-671h\n(4 weeks)" = c(505, 672)
)

# ---------------------------------------------------------------------------
# 1. CPM Stress-Test (Toto-only, all windows)
# ---------------------------------------------------------------------------

load_toto_forecasts <- function(horizon) {
  if (horizon == 160) {
    fpath <- file.path(DATA_DIR, "forecasts_toto", "toto_forecasts.csv")
  } else {
    fpath <- file.path(DATA_DIR, paste0("forecasts_toto_h", horizon), paste0("toto_forecasts_h", horizon, ".csv"))
  }
  if (!file.exists(fpath)) {
    return(NULL)
  }
  df <- fread(fpath)
  df[, datetime := as.POSIXct(datetime, tz = "UTC")]
  return(df)
}

compute_toto_segment_mae <- function(df, horizon) {
  # Storage for errors per segment
  segment_errors <- list()
  for (name in names(SEGMENTS)) {
    segment_errors[[name]] <- numeric()
  }
  
  # Group by window_id
  windows <- unique(df$window_id)
  
  for (wid in windows) {
    w <- df[window_id == wid][order(datetime)]
    
    for (seg_name in names(SEGMENTS)) {
      start <- SEGMENTS[[seg_name]][1]
      end <- SEGMENTS[[seg_name]][2]
      
      if (start > horizon) {
        next
      }
      
      seg_end <- min(end, horizon, nrow(w))
      errors <- numeric()
      
      for (i in start:seg_end) {
        fc <- w$forecast_median[i]
        act <- w$actual[i]
        
        # Look up actual from test data if NA
        if (is.na(act)) {
          dt <- w$datetime[i]
          lookup_val <- actuals_df[.(dt), actual]
          if (length(lookup_val) > 0) {
            act <- lookup_val
          }
        }
        
        if (!is.na(act) && !is.na(fc)) {
          errors <- c(errors, abs(act - fc))
        }
      }
      
      if (length(errors) > 0) {
        segment_errors[[seg_name]] <- c(segment_errors[[seg_name]], mean(errors))
      }
    }
  }
  
  results <- list()
  for (name in names(SEGMENTS)) {
    vals <- segment_errors[[name]]
    results[[name]] <- if (length(vals) > 0) mean(vals) else NA
  }
  return(results)
}

plot_cpm_stress_test <- function() {
  results <- list()
  
  for (horizon in c(160, 320, 672)) {
    df <- load_toto_forecasts(horizon)
    if (!is.null(df)) {
      results[[as.character(horizon)]] <- compute_toto_segment_mae(df, horizon)
    }
  }
  
  # Prepare data for plotting
  plot_data_list <- list()
  for (horizon_str in names(results)) {
    h_res <- results[[horizon_str]]
    for (seg_name in names(h_res)) {
      val <- h_res[[seg_name]]
      if (!is.na(val)) {
        plot_data_list[[length(plot_data_list) + 1]] <- data.table(
          Horizon = paste0("Toto h=", horizon_str, "h"),
          Segment = seg_name,
          MAE = val
        )
      }
    }
  }
  
  if (length(plot_data_list) == 0) {
    cat("No Toto forecasts found for stress test plotting.\n")
    return(results)
  }
  
  plot_df <- rbindlist(plot_data_list)
  # Fix factor levels for ordered segment plotting
  plot_df[, Segment := factor(Segment, levels = names(SEGMENTS))]
  
  colors_map <- c(
    "Toto h=160h" = "#2563eb",
    "Toto h=320h" = "#7c3aed",
    "Toto h=672h" = "#dc2626"
  )
  
  p1 <- ggplot(plot_df, aes(x = Segment, y = MAE, fill = Horizon)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.0f", MAE)), 
              position = position_dodge(width = 0.8), vjust = -0.3, size = 3) +
    scale_fill_manual(values = colors_map) +
    labs(
      title = "Horizon Stress-Test: CPM Error Growth by Forecast Depth\n(averaged across all rolling windows)",
      x = "Forecast Horizon Depth",
      y = "MAE (vehicles/hour)",
      fill = "Total Horizon"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      legend.position = "bottom"
    )
  
  ggsave(file.path(PLOTS_DIR, "horizon_stress_test.png"), plot = p1, width = 9, height = 5.5, dpi = 150)
  cat("  Saved: horizon_stress_test.png\n")
  
  return(results)
}

# ---------------------------------------------------------------------------
# 2. Single-Origin Comparison (R + Toto, same origin)
# ---------------------------------------------------------------------------

load_r_forecasts <- function(horizon) {
  fpath <- file.path(DATA_DIR, "forecasts_r", paste0("horizon_stress_h", horizon, ".csv"))
  if (!file.exists(fpath)) {
    return(NULL)
  }
  df <- fread(fpath)
  df[, datetime := as.POSIXct(datetime, tz = "UTC")]
  return(df)
}

compute_r_segment_mae <- function(df, model_col, horizon) {
  segment_errors <- list()
  
  for (seg_name in names(SEGMENTS)) {
    start <- SEGMENTS[[seg_name]][1]
    end <- SEGMENTS[[seg_name]][2]
    
    if (start > horizon) {
      next
    }
    
    seg_end <- min(end, horizon, nrow(df))
    
    # 1-based indexing in R
    fc <- df[[model_col]][start:seg_end]
    actual <- df$actual[start:seg_end]
    
    mask <- !is.na(actual) & !is.na(fc)
    if (any(mask)) {
      segment_errors[[seg_name]] <- mean(abs(actual[mask] - fc[mask]))
    }
  }
  
  return(segment_errors)
}

compute_toto_single_origin_mae <- function(horizon, target_window_id = 7) {
  df <- load_toto_forecasts(horizon)
  if (is.null(df)) {
    return(list())
  }
  
  # Filter to single origin
  w <- df[window_id == target_window_id][order(datetime)]
  if (nrow(w) == 0) {
    return(list())
  }
  
  segment_errors <- list()
  
  for (seg_name in names(SEGMENTS)) {
    start <- SEGMENTS[[seg_name]][1]
    end <- SEGMENTS[[seg_name]][2]
    
    if (start > horizon) {
      next
    }
    
    seg_end <- min(end, horizon, nrow(w))
    errors <- numeric()
    
    for (i in start:seg_end) {
      fc <- w$forecast_median[i]
      act <- w$actual[i]
      
      # Look up actual from test data if NA
      if (is.na(act)) {
        dt <- w$datetime[i]
        lookup_val <- actuals_df[.(dt), actual]
        if (length(lookup_val) > 0) {
          act <- lookup_val
        }
      }
      
      if (!is.na(act) && !is.na(fc)) {
        errors <- c(errors, abs(act - fc))
      }
    }
    
    if (length(errors) > 0) {
      segment_errors[[seg_name]] <- mean(errors)
    }
  }
  
  return(segment_errors)
}

compute_all_single_origin <- function() {
  # Map R horizons to Toto horizons (R uses 168/336/672, Toto uses 160/320/672)
  horizon_pairs <- list(
    list(r = 168, toto = 160),
    list(r = 336, toto = 320),
    list(r = 672, toto = 672)
  )
  
  results <- list(
    SARIMA = list(),
    `MSTL+ARIMA` = list(),
    Naive = list(),
    `Toto-2.5B` = list()
  )
  
  r_model_cols <- c(
    SARIMA = "sarima",
    `MSTL+ARIMA` = "tbats",
    Naive = "naive"
  )
  
  for (pair in horizon_pairs) {
    r_horizon <- pair$r
    toto_horizon <- pair$toto
    
    # R models
    r_df <- load_r_forecasts(r_horizon)
    if (!is.null(r_df)) {
      for (model_name in names(r_model_cols)) {
        col <- r_model_cols[model_name]
        seg_mae <- compute_r_segment_mae(r_df, col, r_horizon)
        for (seg in names(seg_mae)) {
          results[[model_name]][[seg]] <- seg_mae[[seg]]
        }
      }
    }
    
    # Toto — single origin (window_id == 7, starts 2026-01-08)
    toto_seg_mae <- compute_toto_single_origin_mae(toto_horizon, target_window_id = 7)
    for (seg in names(toto_seg_mae)) {
      results$`Toto-2.5B`[[seg]] <- toto_seg_mae[[seg]]
    }
  }
  
  return(results)
}

plot_single_origin_comparison <- function(results) {
  # Reshape results to data.table
  plot_list <- list()
  for (model_name in names(results)) {
    m_res <- results[[model_name]]
    for (seg_name in names(m_res)) {
      val <- m_res[[seg_name]]
      if (!is.na(val)) {
        plot_list[[length(plot_list) + 1]] <- data.table(
          Model = model_name,
          Segment = seg_name,
          MAE = val
        )
      }
    }
  }
  
  if (length(plot_list) == 0) {
    cat("No single origin comparison data found.\n")
    return()
  }
  
  plot_df <- rbindlist(plot_list)
  plot_df[, Segment := factor(Segment, levels = names(SEGMENTS))]
  plot_df[, Model := factor(Model, levels = c("SARIMA", "MSTL+ARIMA", "Naive", "Toto-2.5B"))]
  
  colors_map <- c(
    "SARIMA" = "#e74c3c",
    "MSTL+ARIMA" = "#3498db",
    "Naive" = "#2ecc71",
    "Toto-2.5B" = "#9b59b6"
  )
  
  shapes_map <- c(
    "SARIMA" = 16,     # circle
    "MSTL+ARIMA" = 15, # square
    "Naive" = 17,      # triangle
    "Toto-2.5B" = 18   # diamond
  )
  
  # For annotation placement, we can use geom_text
  p2 <- ggplot(plot_df, aes(x = Segment, y = MAE, color = Model, group = Model, shape = Model)) +
    geom_line(size = 1.0) +
    geom_point(size = 3.5) +
    geom_text(aes(label = sprintf("%.0f", MAE)), vjust = -0.7, size = 3, fontface = "bold", show.legend = FALSE) +
    scale_color_manual(values = colors_map) +
    scale_shape_manual(values = shapes_map) +
    labs(
      title = "Error Growth by Horizon Depth:\nAutoregressive (SARIMA, MSTL+ARIMA, Naive) vs CPM (Toto-2.5B)\nAll models fitted once on 8,760h context, no re-scoring",
      x = "Forecast Horizon Depth",
      y = "MAE (vehicles/hour)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      legend.position = "bottom"
    )
  
  ggsave(file.path(PLOTS_DIR, "horizon_comparison.png"), plot = p2, width = 9, height = 5.5, dpi = 150)
  cat("  Saved: horizon_comparison.png\n")
}

print_comparison_table <- function(results) {
  seg_display <- c(
    "0-167h\n(1 week)" = "1wk",
    "168-335h\n(2 weeks)" = "2wk",
    "336-503h\n(3 weeks)" = "3wk",
    "504-671h\n(4 weeks)" = "4wk"
  )
  
  cat("\nMAE by horizon segment (single origin, 2026-01-08):\n")
  cat(sprintf("%-15s", "Model"))
  for (seg in names(seg_display)) {
    cat(sprintf("%10s", seg_display[seg]))
  }
  cat("\n")
  
  for (model in c("Toto-2.5B", "SARIMA", "MSTL+ARIMA", "Naive")) {
    cat(sprintf("%-15s", model))
    for (seg in names(seg_display)) {
      val <- results[[model]][[seg]]
      if (!is.null(val) && !is.na(val)) {
        cat(sprintf("%10.1f", val))
      } else {
        cat(sprintf("%10s", "N/A"))
      }
    }
    cat("\n")
  }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function() {
  cat("============================================================\n")
  cat("1. CPM Stress-Test (Toto-only, all rolling windows)\n")
  cat("============================================================\n")
  cpm_results <- plot_cpm_stress_test()
  
  for (horizon_str in names(cpm_results)) {
    cat(sprintf("\n  Horizon=%sh:\n", horizon_str))
    h_res <- cpm_results[[horizon_str]]
    for (seg in names(h_res)) {
      val <- h_res[[seg]]
      seg_display <- gsub("\n", " ", seg)
      if (!is.na(val)) {
        cat(sprintf("    %-20s: MAE=%.1f\n", seg_display, val))
      } else {
        cat(sprintf("    %-20s: N/A\n", seg_display))
      }
    }
  }
  
  cat("\n============================================================\n")
  cat("2. Single-Origin Comparison (R + Toto, 2026-01-08 origin)\n")
  cat("============================================================\n")
  comparison_results <- compute_all_single_origin()
  plot_single_origin_comparison(comparison_results)
  print_comparison_table(comparison_results)
}

main()
