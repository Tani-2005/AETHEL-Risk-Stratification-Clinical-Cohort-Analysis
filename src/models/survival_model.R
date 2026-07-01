# =============================================================================
# AETHEL: Advanced Biostatistical Inference Engine
# Stage: Survival Modelling (R)
# =============================================================================
# UPDATED: Now reads from data/processed/train.csv (training split only).
# Covariates updated: avg_no2_exposure removed (VIF=15.25);
#                     pollution_index added (combined z-scored burden).
# Model algorithms (Cox PH, RSF) are UNCHANGED.
# =============================================================================

cat("Initialising AETHEL Biostatistical Engine...\n")

library(survival)
library(randomForestSRC)

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("The 'yaml' R package is required. Install with: install.packages('yaml')")
}

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
`%||%` <- function(x, y) if (!is.null(x)) x else y

config     <- yaml::read_yaml("configs/default.yaml")
r_seed     <- config$seeds$r %||% 123
rsf_ntree  <- config$model_params$rsf_ntree %||% 500
covariates <- config$features$survival_covariates

# Updated: model trains on TRAIN SPLIT only (not full cohort)
train_path <- config$data_paths$train_cohort %||% "data/processed/train.csv"
val_path   <- config$data_paths$val_cohort   %||% "data/processed/val.csv"

metrics_dir <- config$output_paths$metrics %||% "outputs/metrics"
dir.create(metrics_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("  r_seed     : %d\n", r_seed))
cat(sprintf("  rsf_ntree  : %d\n", rsf_ntree))
cat(sprintf("  covariates : %s\n", paste(covariates, collapse = ", ")))
cat(sprintf("  train_path : %s\n", train_path))

# ---------------------------------------------------------------------------
# 1. Load TRAINING split only
# ---------------------------------------------------------------------------
train_data <- tryCatch(
  read.csv(train_path),
  error = function(e) {
    stop(paste(
      "Could not load training split from:", train_path,
      "\nEnsure run_full_preprocessing() has been executed first.",
      "\nRun: python scripts/run_pipeline.py (or python -m src.feature_engineering.preprocess_features)"
    ))
  }
)
cat(sprintf("Loaded training cohort: %d patients.\n", nrow(train_data)))

# Verify required covariates are present
missing_covs <- covariates[!covariates %in% names(train_data)]
if (length(missing_covs) > 0) {
  stop(paste(
    "Missing covariates in training data:", paste(missing_covs, collapse = ", "),
    "\nCheck that feature engineering and preprocessing completed successfully."
  ))
}

# Build survival object and formulas from config
surv_obj    <- Surv(time = train_data$months_observed, event = train_data$event_occurred)
cox_formula <- as.formula(paste("surv_obj ~", paste(covariates, collapse = " + ")))
rsf_formula <- as.formula(paste(
  "Surv(months_observed, event_occurred) ~",
  paste(covariates, collapse = " + ")
))

# ---------------------------------------------------------------------------
# MODEL 1: Classical Inference — Cox Proportional Hazards
# ---------------------------------------------------------------------------
cat("\n=== 1. COX PROPORTIONAL HAZARDS (Classical) ===\n")

cox_model <- coxph(cox_formula, data = train_data)
cox_coefs <- as.data.frame(summary(cox_model)$coefficients)
print(cox_coefs)

cox_output_path <- file.path(metrics_dir, "cox_coefficients.csv")
write.csv(cox_coefs, cox_output_path, row.names = TRUE)
cat(sprintf("\nCox coefficients saved to %s\n", cox_output_path))

# Proportional hazards assumption
ph_test <- cox.zph(cox_model)
cat("\nSchoenfeld Residuals Test (Proportionality Check):\n")
print(ph_test)

# ---------------------------------------------------------------------------
# MODEL 2: Random Survival Forest
# ---------------------------------------------------------------------------
cat("\n=== 2. RANDOM SURVIVAL FOREST (Machine Learning) ===\n")
cat(sprintf("Training non-linear survival ensemble (%d trees)...\n", rsf_ntree))

rsf_model <- rfsrc(
  rsf_formula,
  data       = train_data,
  ntree      = rsf_ntree,
  importance = TRUE,
  seed       = r_seed
)
cat(sprintf("Out-of-Bag (OOB) Error Rate: %.4f\n\n", rsf_model$err.rate[rsf_ntree]))

# ---------------------------------------------------------------------------
# MODULE 3: Explainability — Variable Importance
# ---------------------------------------------------------------------------
cat("=== 3. EXPLAINABILITY (Variable Importance) ===\n")

vimp_raw    <- vimp(rsf_model)$importance
sorted_vimp <- sort(vimp_raw, decreasing = TRUE)
print(sorted_vimp)

feature_labels <- c(
  "age"                = "Age",
  "is_smoker"          = "Smoking",
  "genomic_risk_score" = "Polygenic Score",
  "townsend_index"     = "Deprivation",
  "avg_pm25_exposure"  = "PM2.5 Exposure",
  "pollution_index"    = "Pollution Index",
  "lifestyle_risk"     = "Lifestyle Risk",
  "high_genomic_risk"  = "High Genomic Risk"
)

vimp_df <- data.frame(
  Feature    = names(sorted_vimp),
  Importance = as.numeric(sorted_vimp)
)
vimp_df$Feature <- ifelse(
  vimp_df$Feature %in% names(feature_labels),
  feature_labels[vimp_df$Feature],
  vimp_df$Feature
)

vimp_output_path <- file.path(metrics_dir, "vimp.csv")
write.csv(vimp_df, vimp_output_path, row.names = FALSE)
cat(sprintf("\nVIMP saved to %s\n", vimp_output_path))

cat("\nBiostatistical inference pipeline complete.\n")
cat("NOTE: Model trained on TRAIN SPLIT only. Evaluate on val/test splits.\n")
