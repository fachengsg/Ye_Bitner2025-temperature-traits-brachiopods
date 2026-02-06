# ======================================================================
# Machine learning (Part 1) - Data preparation
# Ye & Bitner (2025)
#
# Input:
# - appendix_1_SFT_datamatrix_submit.csv  (response: SFT)
# - appendix_2_SST_datamatrix_submit.csv  (response: SST)
#
# Data structure:
# - Column 1: temperature variable (SFT or SST) = response (y)
# - Columns 2–9: eight ecological variables = predictors (X)
#
# ======================================================================

install.packages("caret")

suppressPackageStartupMessages({
  library(caret)   # createDataPartition
})

# -----------------------------
# 1) File paths (simple)
# -----------------------------
file_sft <- "appendix_1_SFT_datamatrix_submit.csv"
file_sst <- "appendix_2_SST_datamatrix_submit.csv"

# -----------------------------
# 2) Helper: load + clean + split
# -----------------------------
load_clean_split <- function(file, seed = 14, p_train = 0.80, prefix = "SFT") {
  
  if (!file.exists(file)) {
    stop("Input file not found: ", file,
         "\nTip: place the CSV in the same folder as this script, or update 'file_*' paths.")
  }
  
  df <- read.csv(file, header = TRUE, check.names = FALSE)
  
  # Clean column names:
  # - trim whitespace
  # - make.names() to avoid spaces, '/' etc. that break formulas/ggpubr parsing
  names(df) <- trimws(names(df))
  names(df) <- make.names(names(df))
  
  # Basic structure checks
  if (ncol(df) < 9) stop(prefix, ": expected at least 9 columns (1 response + 8 predictors). Found: ", ncol(df))
  
  y_name <- names(df)[1]           # response = first column (SFT or SST)
  x_names <- names(df)[2:9]        # predictors = columns 2–9
  
  # Force numeric for response and predictors (robust to character columns)
  df[[y_name]] <- suppressWarnings(as.numeric(df[[y_name]]))
  for (v in x_names) df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
  
  # Drop rows with missing values in response or predictors
  df2 <- df[, c(y_name, x_names)]
  df2 <- df2[complete.cases(df2), ]
  
  # Split train/validation
  set.seed(seed)
  idx <- caret::createDataPartition(df2[[y_name]], p = p_train, list = FALSE)
  train <- df2[idx, ]
  valid <- df2[-idx, ]
  
  # Simple summary for the console
  message(prefix, " data loaded from: ", file)
  message("  Response: ", y_name, " | Predictors: ", paste(x_names, collapse = ", "))
  message("  Rows after NA removal: ", nrow(df2))
  message("  Train/Valid split: ", nrow(train), " / ", nrow(valid))
  
  list(train = train, valid = valid, y = y_name, x = x_names)
}

# -----------------------------
# 3) Run for SFT and SST
# -----------------------------
sft <- load_clean_split(file_sft, seed = 14, p_train = 0.80, prefix = "SFT")
sst <- load_clean_split(file_sst, seed = 14, p_train = 0.80, prefix = "SST")

# Backward-compatible object names (for the ML scripts)
ML391_train_data <- sst$train
ML391_valid_data <- sst$valid

ML486_train_data <- sft$train
ML486_valid_data <- sft$valid

# (optional) full cleaned data, useful for SHAP background etc.
ML486_all_data <- rbind(ML486_train_data, ML486_valid_data)
ML391_all_data <- rbind(ML391_train_data, ML391_valid_data)

