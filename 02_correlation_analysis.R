# ======================================================================
# Correlation analysis (Spearman) for Ye & Bitner (2025)
#
# Data structure:
# - Column 1: SFT or SST (predictor)
# - Columns 2–9: eight ecological response variables
#
# This script:
# 1) reads the SFT and SST data matrices (CSV),
# 2) computes Spearman correlation (rho, p-value) for each response variable,
# 3) generates scatterplots with a fitted line and 95% CI,
# 4) saves a figure panel and a summary table.
# ======================================================================

# -----------------------------
# Packages
# -----------------------------
suppressPackageStartupMessages({
  library(ggplot2)
  library(ggpubr)
})

# -----------------------------
# Input files (simple relative paths)
# Option 1: place CSV files in the same folder as this script and keep names as below
# Option 2: put them under data/ and use file.path("data", "xxx.csv")
# -----------------------------
file_sft <- "appendix_1_SFT_datamatrix_submit.csv"
file_sst <- "appendix_2_SST_datamatrix_submit.csv"

# Output folders
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables",  recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# Load data
# -----------------------------
dat_sft <- read.csv(file_sft, header = TRUE, check.names = FALSE)
dat_sst <- read.csv(file_sst, header = TRUE, check.names = FALSE)

# Clean column names (remove leading/trailing spaces)
names(dat_sft) <- trimws(names(dat_sft))
names(dat_sst) <- trimws(names(dat_sst))

# Make column names safe for ggpubr/ggplot parsing
names(dat_sft) <- make.names(names(dat_sft))
names(dat_sst) <- make.names(names(dat_sst))

# Define which columns are used:
# predictor = 1st column; responses = columns 2–9
predictor_sft <- names(dat_sft)[1]
predictor_sst <- names(dat_sst)[1]
responses_sft <- names(dat_sft)[2:9]
responses_sst <- names(dat_sst)[2:9]

# Safety checks
stopifnot(length(responses_sft) == 8, length(responses_sst) == 8)

# -----------------------------
# Helper functions
# -----------------------------
p_to_stars <- function(p) {
  if (is.na(p)) "" else if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else "ns"
}

cor_one <- function(df, x, y) {
  xx <- suppressWarnings(as.numeric(df[[x]]))
  yy <- suppressWarnings(as.numeric(df[[y]]))
  
  ct <- suppressWarnings(cor.test(xx, yy, method = "spearman", use = "complete.obs"))
  
  data.frame(
    predictor = x,
    response  = y,
    n         = sum(complete.cases(xx, yy)),
    rho       = unname(ct$estimate),
    p_value   = ct$p.value,
    stars     = p_to_stars(ct$p.value),
    row.names = NULL
  )
}

plot_one <- function(df, x, y, panel_label, xlab, ylab) {
  # Compute correlation for subtitle
  s <- cor_one(df, x, y)
  subtitle_txt <- paste0("Spearman rho = ", sprintf("%.2f", s$rho), " (p = ", signif(s$p_value, 3), ") ", s$stars)
  
  ggscatter(
    df, x = x, y = y,
    add = "reg.line", conf.int = TRUE,
    cor.coef = FALSE, cor.method = "spearman"
  ) +
    labs(title = panel_label, subtitle = subtitle_txt, x = xlab, y = ylab) +
    theme_pubr(base_size = 12)
}

run_correlation_panel <- function(df, predictor, responses, xlab, panel_prefix) {
  # correlation table
  tab <- do.call(rbind, lapply(responses, function(y) cor_one(df, predictor, y)))
  
  # plots
  labels <- letters[1:length(responses)]
  plots <- lapply(seq_along(responses), function(i) {
    plot_one(
      df = df,
      x  = predictor,
      y  = responses[i],
      panel_label = labels[i],
      xlab = xlab,
      ylab = responses[i]
    )
  })
  
  list(table = tab, plots = plots, prefix = panel_prefix)
}

# -----------------------------
# Run SFT and SST analyses
# -----------------------------

res_sft <- run_correlation_panel(
  df = dat_sft,
  predictor = predictor_sft,
  responses = responses_sft,
  xlab = "Sea Floor Temperature (SFT)",
  panel_prefix = "SFT"
)

res_sst <- run_correlation_panel(
  df = dat_sst,
  predictor = predictor_sst,
  responses = responses_sst,
  xlab = "Sea Surface Temperature (SST)",
  panel_prefix = "SST"
)

# -----------------------------
# Combine panels and save figures
# -----------------------------
fig_sft <- ggarrange(plotlist = res_sft$plots, ncol = 3, nrow = 3, align = "v")
fig_sst <- ggarrange(plotlist = res_sst$plots, ncol = 3, nrow = 3, align = "v")

ggsave("output/figures/correlation_SFT_panel.png", fig_sft, width = 10, height = 10, dpi = 600)
ggsave("output/figures/correlation_SST_panel.png", fig_sst, width = 10, height = 10, dpi = 600)

# -----------------------------
# Save summary table
# -----------------------------
cor_summary <- rbind(
  transform(res_sft$table, temperature_variable = "SFT"),
  transform(res_sst$table, temperature_variable = "SST")
)

write.csv(cor_summary, "output/tables/correlation_summary_spearman.csv", row.names = FALSE)

# ----------------------------------------------------------
# Notes for reproducibility
# The correlation panels produced by this script correspond to:
# - Fig. 3 in Ye & Bitner (2025): correlations using SFT (Sea Floor Temperature)
# - Fig. 4 in Ye & Bitner (2025): correlations using SST (Sea Surface Temperature)
# ----------------------------------------------------------