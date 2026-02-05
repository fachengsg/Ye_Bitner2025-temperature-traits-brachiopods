# ======================================================================
# Normality checks for trait variables (Ye & Bitner, 2025)
#
# This script:
#   1) reads the trait data matrix from a CSV file,
#   2) computes Shapiro–Wilk normality test, skewness, and kurtosis
#      for a set of numeric variables,
#   3) writes a tidy results table to /output/tables/.
#
# Notes:
# - Shapiro–Wilk is typically recommended for 3 <= n <= 5000.
#   If n is outside this range, the test is skipped and reported as NA.
# - Missing values (NA) are removed before computing statistics.
# ======================================================================

# -----------------------------
# 0) Packages
# -----------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(moments)  # skewness(), kurtosis()
  library(readr)    # read_csv(), write_csv()
  library(here)     # robust relative paths
})

# ==========================================================
# Data description
# - Column 1: SFT = Sea Floor Temperature (the predictor of interest).
# - Columns 2–9: eight ecological response variables.
#   We examine how SFT relates to (or influences) these eight traits/metrics.
# ==========================================================

Normality_ML_dat <- read.csv("appendix_1_SFT_datamatrix_submit.csv", header = TRUE)

# make sure column names are clean (remove leading/trailing spaces)
names(Normality_ML_dat) <- trimws(names(Normality_ML_dat))

# Optional quick check
str(Normality_ML_dat)
head(Normality_ML_dat)

# Columns 2–9 are the eight response variables
vars <- names(Normality_ML_dat)[2:9]

normality_summary <- do.call(
  rbind,
  lapply(vars, function(v) {
    x <- Normality_ML_dat[[v]]
    x <- x[is.finite(x)]  # remove NA/NaN/Inf
    n <- length(x)
    
    sh <- if (n >= 3 && n <= 5000) shapiro.test(x) else NULL
    
    data.frame(
      variable  = v,
      n         = n,
      shapiro_W = if (!is.null(sh)) unname(sh$statistic) else NA,
      shapiro_p = if (!is.null(sh)) sh$p.value else NA,
      skewness  = if (n >= 3) skewness(x) else NA,
      kurtosis  = if (n >= 4) kurtosis(x) else NA,
      note      = if (is.null(sh)) "Shapiro skipped (n < 3 or n > 5000)" else "OK",
      row.names = NULL
    )
  })
)

# ----------------------------------------------------------
# Export results
# The table written below corresponds to Table 1 in Ye & Bitner (2025),
# summarising Shapiro–Wilk normality tests, skewness and kurtosis for the
# eight ecological response variables used in the analysis.
# ----------------------------------------------------------

# Save as CSV (same folder as the script, or change path)
write.csv(normality_summary, "normality_summary.csv", row.names = FALSE)
