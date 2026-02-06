# ======================================================================
# Machine learning (SST) - Train + validate + interpret
# Ye & Bitner (2025)
#
# Goal:
# - Fit multiple ML models to predict SST from 8 ecological predictors.
# - Validate performance on a 20% hold-out set.
# - Summarise residual distributions (DALEX model_performance).
# - Estimate variable importance (DALEX model_parts).
# - Compute SHAP-style global importance (kernelshap + shapviz).
#
# Inputs (created in 03_ml_01_data_prep.R):
# - ML391_train_data: training set (80%)
# - ML391_valid_data: validation set (20%)
#

# ======================================================================

install.packages("ggplot2")
install.packages("ggpubr")
install.packages("dplyr")
install.packages("DALEX")
install.packages("kernelshap")
install.packages("shapviz")
install.packages(c("randomForest", "gbm", "kernlab", "nnet"))
install.packages("catboost")

suppressPackageStartupMessages({
  library(caret)
  library(ggplot2)
  library(ggpubr)
  library(dplyr)
  library(DALEX)
  library(kernelshap)
  library(shapviz)
  library(randomForest)
  library(gbm)
  library(kernlab)
  library(nnet)
  library(catboost)
})

# ============================================================
# 10-fold CV training + hold-out validation (SST)
# ============================================================

# -----------------------------
# Cross-validation controls
# -----------------------------
# 10-fold cross validation for caret models
control <- trainControl(method = "cv", number = 10)

fit_control <- caret::trainControl(
  method = "cv",
  number = 10,
  search = "random",
  classProbs = TRUE
)


# -----------------------------
# CatBoost tuning grid
# -----------------------------
grid <- expand.grid(
  depth = c(4, 6, 8),
  learning_rate = 0.1,
  l2_leaf_reg = 0.1,
  rsm = 0.95,
  border_count = 64,
  iterations = 10
)


# Train models (SST as response)
# -----------------------------
# rf: Random Forest | gbm: Gradient Boosting | svm: SVM radial | nn: Neural net | cat: CatBoost

set.seed(14)
fit.rfMLconSST  <- train(SST ~ ., data = ML391_train_data, method = "rf",        trControl = control)

set.seed(14)
fit.gbmMLconSST <- train(SST ~ ., data = ML391_train_data, method = "gbm",       trControl = control)

set.seed(14)
fit.svmMLconSST <- train(SST ~ ., data = ML391_train_data, method = "svmRadial", trControl = control)

set.seed(14)
fit.nnMLconSST  <- train(
  SST ~ ., data = ML391_train_data, method = "nnet",
  linout = TRUE,
  preProcess = c("center", "scale"),
  maxit = 500,
  tuneGrid = expand.grid(size = 2, decay = 0),
  trControl = control,
  trace = FALSE
)

# CatBoost model (kept identical to the original analysis call)
# IMPORTANT: RMSE is an error metric (smaller is better).
# We keep maximize=TRUE unchanged to match the published workflow exactly.
set.seed(14)
fit.catMLconSST <- caret::train(
  x = ML391_train_data[, -1],
  y = ML391_train_data[,  1],
  method = catboost.caret,
  metric = "RMSE",
  maximize = TRUE,     # kept for exact reproducibility of the published results
  preProc = NULL,
  tuneGrid = grid,
  tuneLength = 30,
  trControl = fit_control
)

# -----------------------------
# DALEX explainers on hold-out validation set
# -----------------------------
# Use predictors only (exclude SST), and provide y separately.
x_valid <- ML391_valid_data[, setdiff(names(ML391_valid_data), "SST"), drop = FALSE]
y_valid <- ML391_valid_data$SST

# Predict function for caret models: always returns a numeric vector
predict_caret_numeric <- function(model, newdata) {
  as.numeric(stats::predict(model, newdata = newdata))
}

set.seed(14)
expla_SST_rf <- DALEX::explain(
  model = fit.rfMLconSST, label = "RF",
  data  = x_valid, y = y_valid,
  predict_function = predict_caret_numeric,
  verbose = FALSE
)

set.seed(14)
expla_SST_gbm <- DALEX::explain(
  model = fit.gbmMLconSST, label = "GBM",
  data  = x_valid, y = y_valid,
  predict_function = predict_caret_numeric,
  verbose = FALSE
)

set.seed(14)
expla_SST_cat <- DALEX::explain(
  model = fit.catMLconSST, label = "CAT",
  data  = x_valid, y = y_valid,
  predict_function = predict_caret_numeric,
  verbose = FALSE
)

set.seed(14)
expla_SST_svm <- DALEX::explain(
  model = fit.svmMLconSST, label = "SVM",
  data  = x_valid, y = y_valid,
  predict_function = predict_caret_numeric,
  verbose = FALSE
)

set.seed(14)
expla_SST_nn <- DALEX::explain(
  model = fit.nnMLconSST, label = "NN",
  data  = x_valid, y = y_valid,
  predict_function = predict_caret_numeric,
  verbose = FALSE
)

# -----------------------------
# Compute performance / residual summaries
# -----------------------------
mp_SST_rf  <- model_performance(expla_SST_rf)
mp_SST_gbm <- model_performance(expla_SST_gbm)
mp_SST_cat <- model_performance(expla_SST_cat)
mp_SST_svm <- model_performance(expla_SST_svm)
mp_SST_nn  <- model_performance(expla_SST_nn)


# ============================================================
# Residual diagnostics on the validation set (DALEX)
# - reverse cumulative distribution of absolute residuals
# - boxplot of absolute residuals (lower = better)
# ============================================================

# Boxplot of absolute residuals
ML_SST_plot <- plot(mp_SST_rf, mp_SST_gbm, mp_SST_svm, mp_SST_nn, mp_SST_cat, geom = "boxplot")

ML_SST_plot2 <- ggpar(
  ML_SST_plot,
  font.title    = c(20, "black"),
  font.subtitle = c(15, "black"),
  legend.title  = "model",
  palette       = c("blue", "red", "green", "purple", "orange")
) + font("xy.text", size = 12, color = "black")

# Reverse cumulative distribution (default plot for model_performance)
ML_SST_Merg_plot <- plot(mp_SST_rf, mp_SST_gbm, mp_SST_svm, mp_SST_nn, mp_SST_cat) +
  theme_gray(base_size = 12)

ML_SST_Merg_plot2 <- ggpar(
  ML_SST_Merg_plot,
  font.title    = c(20, "black"),
  font.subtitle = c(15, "black"),
  ylab          = "absolute residual value",
  font.y        = c(15, "black"),
  legend.title  = "model",
  xlab          = "",
  palette       = c("blue", "red", "green", "purple", "orange")
) + font("xy.text", size = 12, color = "black")

gridExtra::grid.arrange(ML_SST_Merg_plot2, ML_SST_plot2, nrow = 1)


# ============================================================
# Permutation-based variable importance (DALEX)
# - importance values reflect increase in loss after permuting a variable
# - higher value = more important for prediction
# ============================================================

SST_impor_rf  <- model_parts(explainer = expla_SST_rf,  B = 100, N = NULL)
SST_impor_gbm <- model_parts(explainer = expla_SST_gbm, B = 100, N = NULL)
SST_impor_cat <- model_parts(explainer = expla_SST_cat, B = 100, N = NULL)
SST_impor_svm <- model_parts(explainer = expla_SST_svm, B = 100, N = NULL)
SST_impor_nn  <- model_parts(explainer = expla_SST_nn,  B = 100, N = NULL)

# NOTE: "SST" is the response variable; predictors are shown in the importance plots.
p_SST_rf  <- plot(SST_impor_rf  %>% dplyr::filter(variable != "SST")) + ylim(4.5, 11.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SST_gbm <- plot(SST_impor_gbm %>% dplyr::filter(variable != "SST")) + ylim(4.5, 11.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SST_cat <- plot(SST_impor_cat %>% dplyr::filter(variable != "SST")) + ylim(4.5, 11.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SST_svm <- plot(SST_impor_svm %>% dplyr::filter(variable != "SST")) + ylim(4.5, 11.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SST_nn  <- plot(SST_impor_nn  %>% dplyr::filter(variable != "SST")) + ylim(4.5, 11.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

# -----------------------------
# SHAP (kernelshap + shapviz)
# -----------------------------
# IMPORTANT: SHAP is computed on predictors (X) only, not including the response (SFT).

ML391_all <- rbind(ML391_train_data, ML391_valid_data)

xvars_391 <- colnames(ML391_train_data[-1])
X391_all  <- ML391_all[, xvars_391, drop = FALSE]

# --- Compute SHAP values for each model ---
# We pass predict as the prediction function 
SST_rf_SHAP  <- kernelshap(fit.rfMLconSST,  X391_all, predict, bg_X = X391_all, feature_names = xvars)
SST_gbm_SHAP <- kernelshap(fit.gbmMLconSST, X391_all, predict, bg_X = X391_all, feature_names = xvars)
SST_cat_SHAP <- kernelshap(fit.catMLconSST, X391_all, predict, bg_X = X391_all, feature_names = xvars)
SST_svm_SHAP <- kernelshap(fit.svmMLconSST, X391_all, predict, bg_X = X391_all, feature_names = xvars)
SST_nn_SHAP  <- kernelshap(fit.nnMLconSST,  X391_all, predict, bg_X = X391_all, feature_names = xvars)

# --- Convert to shapviz objects for plotting ---
SST_rf_SHAP_sv  <- shapviz(SST_rf_SHAP)
SST_gbm_SHAP_sv <- shapviz(SST_gbm_SHAP)
SST_cat_SHAP_sv <- shapviz(SST_cat_SHAP)
SST_svm_SHAP_sv <- shapviz(SST_svm_SHAP)
SST_nn_SHAP_sv  <- shapviz(SST_nn_SHAP)

# --- Plot global SHAP importance  ---
plot_SST_rf_SHAP  <- sv_importance(SST_rf_SHAP_sv)  + ggtitle("RF")  + xlim(0, 4) + font("title", size = 18, color = "black")
plot_SST_gbm_SHAP <- sv_importance(SST_gbm_SHAP_sv) + ggtitle("GBM") + xlim(0, 4) + font("title", size = 18, color = "black")
plot_SST_cat_SHAP <- sv_importance(SST_cat_SHAP_sv) + ggtitle("CAT") + xlim(0, 4) + font("title", size = 18, color = "black")
plot_SST_svm_SHAP <- sv_importance(SST_svm_SHAP_sv) + ggtitle("SVM") + xlim(0, 4) + font("title", size = 18, color = "black")
plot_SST_nn_SHAP  <- sv_importance(SST_nn_SHAP_sv)  + ggtitle("NN")  + xlim(0, 4) + font("title", size = 18, color = "black")




# ============================================================
# SFT models + validation + plots
# Notes:
# - Same random seed (14) and 10-fold CV as the original analysis.
# - RMSE is the main metric; lower RMSE = better predictive performance.
# ============================================================


# -----------------------------
# Train models (SFT)
# -----------------------------
# rf: Random Forest | gbm: Gradient Boosting | svm: Support Vector Machine
# nn: Neural Network | cat: CatBoost
set.seed(14)
fit.rfMLconSFT <- train(SFT ~ ., data = ML486_train_data, method = "rf", trControl = control)

set.seed(14)
fit.gbmMLconSFT <- train(SFT ~ ., data = ML486_train_data, method = "gbm", trControl = control)

set.seed(14)
fit.catMLconSFT <- caret::train(
  x = ML486_train_data[, -1],
  y = ML486_train_data[,  1],
  method = catboost.caret,
  metric = "RMSE",
  maximize = TRUE,   # kept identical to the original script (even though RMSE should be minimized)
  preProc = NULL,
  tuneGrid = grid,
  tuneLength = 30,
  trControl = fit_control
)

set.seed(14)
fit.svmMLconSFT <- train(SFT ~ ., data = ML486_train_data, method = "svmRadial", trControl = control)

set.seed(14)
fit.nnMLconSFT <- train(
  SFT ~ ., data = ML486_train_data,
  method = "nnet",
  linout = TRUE,
  preProcess = c("center", "scale"),
  maxit = 500,
  tuneGrid = expand.grid(size = 2, decay = 0),
  trControl = control
)

# -----------------------------
# Validation explainers (DALEX)
# -----------------------------
# IMPORTANT: DALEX::explain expects X (predictors) and y (response).
x_valid_SFT <- ML486_valid_data[, setdiff(names(ML486_valid_data), "SFT"), drop = FALSE]

set.seed(14)
expla_SFT_rf <- DALEX::explain(fit.rfMLconSFT, label = "RF",
                               data = x_valid_SFT, y = ML486_valid_data$SFT)

set.seed(14)
expla_SFT_gbm <- DALEX::explain(fit.gbmMLconSFT, label = "GBM",
                                data = x_valid_SFT, y = ML486_valid_data$SFT)

set.seed(14)
expla_SFT_cat <- DALEX::explain(fit.catMLconSFT, label = "CAT",
                                data = x_valid_SFT, y = ML486_valid_data$SFT)

set.seed(14)
expla_SFT_svm <- DALEX::explain(fit.svmMLconSFT, label = "SVM",
                                data = x_valid_SFT, y = ML486_valid_data$SFT)

set.seed(14)
expla_SFT_nn <- DALEX::explain(fit.nnMLconSFT, label = "NN",
                               data = x_valid_SFT, y = ML486_valid_data$SFT)

# compute predictions & residuals on the validation set
mp_SFT_rf  <- model_performance(expla_SFT_rf)
mp_SFT_gbm <- model_performance(expla_SFT_gbm)
mp_SFT_cat <- model_performance(expla_SFT_cat)
mp_SFT_svm <- model_performance(expla_SFT_svm)
mp_SFT_nn  <- model_performance(expla_SFT_nn)

# -----------------------------
# Residual plots (DALEX)
# -----------------------------
# - Boxplot of absolute residuals (lower = better)
# - Reverse cumulative distribution of absolute residuals
ML_SFT_plot <- plot(mp_SFT_rf, mp_SFT_gbm, mp_SFT_svm, mp_SFT_nn, mp_SFT_cat, geom = "boxplot")

ML_SFT_plot2 <- ggpar(
  ML_SFT_plot,
  font.title    = c(20, "black"),
  font.subtitle = c(15, "black"),
  legend.title  = "model",
  palette       = c("red", "green", "orange", "blue", "purple")
) + font("xy.text", size = 12, color = "black")

ML_SFT_Merg_plot <- plot(mp_SFT_rf, mp_SFT_gbm, mp_SFT_svm, mp_SFT_nn, mp_SFT_cat) +
  theme_gray(base_size = 12)

ML_SFT_Merg_plot2 <- ggpar(
  ML_SFT_Merg_plot,
  font.title    = c(20, "black"),
  font.subtitle = c(15, "black"),
  ylab          = "absolute residual value",
  font.y        = c(15, "black"),
  legend.title  = "model",
  xlab          = "",
  palette       = c("red", "green", "orange", "blue", "purple")
) + font("xy.text", size = 12, color = "black")

gridExtra::grid.arrange(ML_SFT_Merg_plot2, ML_SFT_plot2, nrow = 1)

# -----------------------------
# Variable importance (DALEX::model_parts)
# -----------------------------
# Permutation importance: higher value = larger increase in loss after permuting that variable.
SFT_impor_rf  <- model_parts(explainer = expla_SFT_rf,  B = 100, N = NULL)
SFT_impor_gbm <- model_parts(explainer = expla_SFT_gbm, B = 100, N = NULL)
SFT_impor_cat <- model_parts(explainer = expla_SFT_cat, B = 100, N = NULL)
SFT_impor_svm <- model_parts(explainer = expla_SFT_svm, B = 100, N = NULL)
SFT_impor_nn  <- model_parts(explainer = expla_SFT_nn,  B = 100, N = NULL)


p_SFT_rf  <- plot(SFT_impor_rf  %>% dplyr::filter(variable != "SFT")) + ylim(4, 7.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SFT_gbm <- plot(SFT_impor_gbm %>% dplyr::filter(variable != "SFT")) + ylim(4, 7.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SFT_cat <- plot(SFT_impor_cat %>% dplyr::filter(variable != "SFT")) + ylim(4, 7.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SFT_svm <- plot(SFT_impor_svm %>% dplyr::filter(variable != "SFT")) + ylim(4, 7.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

p_SFT_nn  <- plot(SFT_impor_nn  %>% dplyr::filter(variable != "SFT")) + ylim(4, 7.5) +
  font("xy.text", size = 12, color = "black") + font("title", size = 18, color = "black") +
  font("subtitle", size = 12, color = "black") + font("xylab", size = 12, color = "black")

# -----------------------------
# SHAP (kernelshap + shapviz)
# -----------------------------
# IMPORTANT: SHAP is computed on predictors (X) only, not including the response (SFT).

ML486_all <- rbind(ML486_train_data, ML486_valid_data)

xvars_486 <- colnames(ML486_train_data[-1])
X486_all  <- ML486_all[, xvars_486, drop = FALSE]

SFT_rf_SHAP  <- kernelshap(fit.rfMLconSFT,  X486_all, predict, bg_X = X486_all, feature_names = xvars_486)
SFT_gbm_SHAP <- kernelshap(fit.gbmMLconSFT, X486_all, predict, bg_X = X486_all, feature_names = xvars_486)
SFT_cat_SHAP <- kernelshap(fit.catMLconSFT, X486_all, predict, bg_X = X486_all, feature_names = xvars_486)
SFT_svm_SHAP <- kernelshap(fit.svmMLconSFT, X486_all, predict, bg_X = X486_all, feature_names = xvars_486)
SFT_nn_SHAP  <- kernelshap(fit.nnMLconSFT,  X486_all, predict, bg_X = X486_all, feature_names = xvars_486)

SFT_rf_SHAP_sv  <- shapviz(SFT_rf_SHAP)
SFT_gbm_SHAP_sv <- shapviz(SFT_gbm_SHAP)
SFT_cat_SHAP_sv <- shapviz(SFT_cat_SHAP)
SFT_svm_SHAP_sv <- shapviz(SFT_svm_SHAP)
SFT_nn_SHAP_sv  <- shapviz(SFT_nn_SHAP)

# SHAP importance summary (mean |SHAP|): higher = stronger influence on SFT prediction
plot_SFT_rf_SHAP  <- sv_importance(SFT_rf_SHAP_sv)  + ggtitle("RF")  + xlim(0, 2) + font("title", size = 18, color = "black")
plot_SFT_gbm_SHAP <- sv_importance(SFT_gbm_SHAP_sv) + ggtitle("GBM") + xlim(0, 2) + font("title", size = 18, color = "black")
plot_SFT_cat_SHAP <- sv_importance(SFT_cat_SHAP_sv) + ggtitle("CAT") + xlim(0, 2) + font("title", size = 18, color = "black")
plot_SFT_svm_SHAP <- sv_importance(SFT_svm_SHAP_sv) + ggtitle("SVM") + xlim(0, 2) + font("title", size = 18, color = "black")
plot_SFT_nn_SHAP  <- sv_importance(SFT_nn_SHAP_sv)  + ggtitle("NN")  + xlim(0, 2) + font("title", size = 18, color = "black")


# ============================================================
# Save objects for plotting
# ============================================================
dir.create("output/rdata", recursive = TRUE, showWarnings = FALSE)

save(
  # models
  fit.rfMLconSST, fit.gbmMLconSST, fit.catMLconSST, fit.svmMLconSST, fit.nnMLconSST,
  fit.rfMLconSFT, fit.gbmMLconSFT, fit.catMLconSFT, fit.svmMLconSFT, fit.nnMLconSFT,
  
  # validation sets 
  ML391_valid_data, ML486_valid_data,
  
  # DALEX performance objects 
  mp_SST_rf, mp_SST_gbm, mp_SST_cat, mp_SST_svm, mp_SST_nn,
  mp_SFT_rf, mp_SFT_gbm, mp_SFT_cat, mp_SFT_svm, mp_SFT_nn,
  
  # variable importance plots 
  p_SFT_rf, p_SST_rf, p_SFT_gbm, p_SST_gbm, p_SFT_cat, p_SST_cat,
  p_SFT_svm, p_SST_svm, p_SFT_nn, p_SST_nn,
  
  # SHAP plots 
  plot_SFT_rf_SHAP, plot_SST_rf_SHAP, plot_SFT_gbm_SHAP, plot_SST_gbm_SHAP,
  plot_SFT_cat_SHAP, plot_SST_cat_SHAP, plot_SFT_svm_SHAP, plot_SST_svm_SHAP,
  plot_SFT_nn_SHAP,  plot_SST_nn_SHAP,
  
  file = "output/rdata/ml_models_and_plots.rdata"
)



# ============================================================
# Build "predicted vs actual" data frames
# ============================================================

# --- SST (ML391) ---
SST_rf_predict_data  <- data.frame(
  SST_rf_predict = predict(fit.rfMLconSST,  newdata = ML391_valid_data),
  SST_rf_actual  = ML391_valid_data$SST
)

SST_gbm_predict_data <- data.frame(
  SST_gbm_predict = predict(fit.gbmMLconSST, newdata = ML391_valid_data),
  SST_gbm_actual  = ML391_valid_data$SST
)

SST_cat_predict_data <- data.frame(
  SST_cat_predict = predict(fit.catMLconSST, newdata = ML391_valid_data),
  SST_cat_actual  = ML391_valid_data$SST
)

SST_svm_predict_data <- data.frame(
  SST_svm_predict = predict(fit.svmMLconSST, newdata = ML391_valid_data),
  SST_svm_actual  = ML391_valid_data$SST
)

SST_nn_predict_data  <- data.frame(
  SST_nn_predict = predict(fit.nnMLconSST,  newdata = ML391_valid_data),
  SST_nn_actual  = ML391_valid_data$SST
)

# --- SFT (ML486) ---
SFT_rf_predict_data  <- data.frame(
  SFT_rf_predict = predict(fit.rfMLconSFT,  newdata = ML486_valid_data),
  SFT_rf_actual  = ML486_valid_data$SFT
)

SFT_gbm_predict_data <- data.frame(
  SFT_gbm_predict = predict(fit.gbmMLconSFT, newdata = ML486_valid_data),
  SFT_gbm_actual  = ML486_valid_data$SFT
)

SFT_cat_predict_data <- data.frame(
  SFT_cat_predict = predict(fit.catMLconSFT, newdata = ML486_valid_data),
  SFT_cat_actual  = ML486_valid_data$SFT
)

SFT_svm_predict_data <- data.frame(
  SFT_svm_predict = predict(fit.svmMLconSFT, newdata = ML486_valid_data),
  SFT_svm_actual  = ML486_valid_data$SFT
)

SFT_nn_predict_data  <- data.frame(
  SFT_nn_predict = predict(fit.nnMLconSFT,  newdata = ML486_valid_data),
  SFT_nn_actual  = ML486_valid_data$SFT
)

# ============================================================
# Predicted vs Observed (hold-out validation set)
# Notes:
# - Spearman's R is a rank-based correlation (robust to non-linearity).
# - The regression line is only a visual guide.
# ============================================================

plot_SST_rf_PO <- ggscatter(SST_rf_predict_data, x = "SST_rf_predict", y = "SST_rf_actual",
                            add = "reg.line", conf.int = TRUE,
                            cor.coef = FALSE, cor.method = "spearman",
                            title = "RF",
                            subtitle = "Spearman's correlation coefficient R = 0.88**",
                            xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SST_gbm_PO <- ggscatter(SST_gbm_predict_data, x = "SST_gbm_predict", y = "SST_gbm_actual",
                             add = "reg.line", conf.int = TRUE,
                             cor.coef = FALSE, cor.method = "spearman",
                             title = "GBM",
                             subtitle = "Spearman's correlation coefficient R = 0.87**",
                             xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SST_cat_PO <- ggscatter(SST_cat_predict_data, x = "SST_cat_predict", y = "SST_cat_actual",
                             add = "reg.line", conf.int = TRUE,
                             cor.coef = FALSE, cor.method = "spearman",
                             title = "CAT",
                             subtitle = "Spearman's correlation coefficient R = 0.80**",
                             xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SST_svm_PO <- ggscatter(SST_svm_predict_data, x = "SST_svm_predict", y = "SST_svm_actual",
                             add = "reg.line", conf.int = TRUE,
                             cor.coef = FALSE, cor.method = "spearman",
                             title = "SVM",
                             subtitle = "Spearman's correlation coefficient R = 0.90**",
                             xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SST_nn_PO <- ggscatter(SST_nn_predict_data, x = "SST_nn_predict", y = "SST_nn_actual",
                            add = "reg.line", conf.int = TRUE,
                            cor.coef = FALSE, cor.method = "spearman",
                            title = "NN",
                            subtitle = "Spearman's correlation coefficient R = 0.81**",
                            xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SFT_rf_PO <- ggscatter(SFT_rf_predict_data, x = "SFT_rf_predict", y = "SFT_rf_actual",
                            add = "reg.line", conf.int = TRUE,
                            cor.coef = FALSE, cor.method = "spearman",
                            title = "RF",
                            subtitle = "Spearman's correlation coefficient R = 0.65**",
                            xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SFT_gbm_PO <- ggscatter(SFT_gbm_predict_data, x = "SFT_gbm_predict", y = "SFT_gbm_actual",
                             add = "reg.line", conf.int = TRUE,
                             cor.coef = FALSE, cor.method = "spearman",
                             title = "GBM",
                             subtitle = "Spearman's correlation coefficient R = 0.70**",
                             xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SFT_cat_PO <- ggscatter(SFT_cat_predict_data, x = "SFT_cat_predict", y = "SFT_cat_actual",
                             add = "reg.line", conf.int = TRUE,
                             cor.coef = FALSE, cor.method = "spearman",
                             title = "CAT",
                             subtitle = "Spearman's correlation coefficient R = 0.59**",
                             xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SFT_svm_PO <- ggscatter(SFT_svm_predict_data, x = "SFT_svm_predict", y = "SFT_svm_actual",
                             add = "reg.line", conf.int = TRUE,
                             cor.coef = FALSE, cor.method = "spearman",
                             title = "SVM",
                             subtitle = "Spearman's correlation coefficient R = 0.69**",
                             xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

plot_SFT_nn_PO <- ggscatter(SFT_nn_predict_data, x = "SFT_nn_predict", y = "SFT_nn_actual",
                            add = "reg.line", conf.int = TRUE,
                            cor.coef = FALSE, cor.method = "spearman",
                            title = "NN",
                            subtitle = "Spearman's correlation coefficient R = 0.52**",
                            xlab = "predicted temperature", ylab = "actual temperature") +
  font("xlab", size = 12) +
  font("ylab", size = 12) +
  font("xy.text", size = 12)

# Merge plotting (SFT left, SST right)
gridExtra::grid.arrange(
  plot_SFT_rf_PO,  plot_SST_rf_PO,
  plot_SFT_gbm_PO, plot_SST_gbm_PO,
  plot_SFT_cat_PO, plot_SST_cat_PO,
  plot_SFT_svm_PO, plot_SST_svm_PO,
  plot_SFT_nn_PO,  plot_SST_nn_PO,
  nrow = 5, ncol = 2,
  top = textGrob("SFT dataset                                                                                                SST dataset",
                 gp = gpar(fontsize = 18))
)


# ======================================================================
# End of 03_ml_02_train_models.R
#
# Notes on interpretation (Ye & Bitner, 2025)
# - RMSE is the primary error metric used for tuning and comparison; smaller
#   RMSE indicates better predictive performance.
# - Residual diagnostics (reverse cumulative distribution + boxplot of absolute
#   residuals) and feature importance (DALEX / SHAP) generated here are used to
#   build Ye & Bitner (2025) Figures 5–8.









