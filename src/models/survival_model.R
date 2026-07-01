# =============================================================================
# AETHEL: Advanced Biostatistical Inference Engine
# Stage: Survival Modelling (R)
# =============================================================================
# Reads: data/processed/analytical_cohort.csv (via configs/default.yaml)
# Writes:
#   outputs/metrics/cox_coefficients.csv
#   outputs/metrics/vimp.csv
# =============================================================================

cat("Initialising AETHEL Biostatistical Engine...\n")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
library(survival)
library(randomForestSRC)

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("The 'yaml' R package is required. Install with: install.packages('yaml')")
}

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
`%||%` <- function(x, y) if (!is.null(x)) x else y

config      <- yaml::read_yaml("configs/default.yaml")
r_seed      <- config$seeds$r %||% 123
cohort_path <- config$data_paths$processed_cohort %||% "data/processed/analytical_cohort.csv"
rsf_ntree   <- config$model_params$rsf_ntree %||% 500
covariates  <- config$features$survival_covariates

# Output paths
metrics_dir  <- config$output_paths$metrics %||% "outputs/metrics"
dir.create(metrics_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("  r_seed     : %d\n", r_seed))
cat(sprintf("  rsf_ntree  : %d\n", rsf_ntree))
cat(sprintf("  covariates : %s\n", paste(covariates, collapse = ", ")))

# ---------------------------------------------------------------------------
# 1. Load the Analytical Dataset
# ---------------------------------------------------------------------------
cohort_data <- tryCatch(
  read.csv(cohort_path),
  error = function(e) {
    stop(paste(
      "Could not load analytical cohort from:", cohort_path,
      "\nEnsure the full pipeline has been run first."
    ))
  }
)
cat(sprintf("Loaded analytical cohort: %d patients.\n\n", nrow(cohort_data)))

# Build the survival object using column names from config
surv_obj <- Surv(time = cohort_data$months_observed, event = cohort_data$event_occurred)

# Build formula dynamically from config's covariate list
cox_formula <- as.formula(paste("surv_obj ~", paste(covariates, collapse = " + ")))
rsf_formula <- as.formula(paste(
  "Surv(months_observed, event_occurred) ~",
  paste(covariates, collapse = " + ")
))

# ---------------------------------------------------------------------------
# MODEL 1: Classical Inference — Cox Proportional Hazards
# ---------------------------------------------------------------------------
cat("=== 1. COX PROPORTIONAL HAZARDS (Classical) ===\n")

cox_model <- coxph(cox_formula, data = cohort_data)

cox_coefs <- as.data.frame(summary(cox_model)$coefficients)
print(cox_coefs)

# Save coefficients to outputs/metrics/
cox_output_path <- file.path(metrics_dir, "cox_coefficients.csv")
write.csv(cox_coefs, cox_output_path, row.names = TRUE)
cat(sprintf("\nCox coefficients saved to %s\n\n", cox_output_path))

# Proportional hazards assumption check
ph_test <- cox.zph(cox_model)
cat("Schoenfeld Residuals Test (Proportionality Check):\n")
print(ph_test)
cat("\n")

# ---------------------------------------------------------------------------
# MODEL 2: Biostatistical Machine Learning — Random Survival Forest
# ---------------------------------------------------------------------------
cat("=== 2. RANDOM SURVIVAL FOREST (Machine Learning) ===\n")
cat(sprintf("Training non-linear survival ensemble (%d trees)...\n", rsf_ntree))

rsf_model <- rfsrc(
  rsf_formula,
  data       = cohort_data,
  ntree      = rsf_ntree,
  importance = TRUE,
  seed       = r_seed
)

cat(sprintf("Out-of-Bag (OOB) Error Rate: %.4f\n\n", rsf_model$err.rate[rsf_ntree]))

# ---------------------------------------------------------------------------
# MODULE 3: Explainability — Variable Importance (VIMP)
# ---------------------------------------------------------------------------
cat("=== 3. EXPLAINABILITY (Variable Importance) ===\n")
cat("Ranking covariates by predictive power (VIMP):\n")

vimp_raw    <- vimp(rsf_model)$importance
sorted_vimp <- sort(vimp_raw, decreasing = TRUE)
print(sorted_vimp)

# Save VIMP for the dashboard to consume
vimp_df <- data.frame(
  Feature    = names(sorted_vimp),
  Importance = as.numeric(sorted_vimp)
)

# Use human-readable feature labels
feature_labels <- c(
  "age"                = "Age",
  "is_smoker"          = "Smoking",
  "genomic_risk_score" = "Polygenic Score",
  "townsend_index"     = "Deprivation",
  "avg_pm25_exposure"  = "PM2.5 Exposure",
  "avg_no2_exposure"   = "NO2 Exposure"
)
vimp_df$Feature <- ifelse(
  vimp_df$Feature %in% names(feature_labels),
  feature_labels[vimp_df$Feature],
  vimp_df$Feature
)

vimp_output_path <- file.path(metrics_dir, "vimp.csv")
write.csv(vimp_df, vimp_output_path, row.names = FALSE)
cat(sprintf("\nVIMP saved to %s (loaded by Streamlit dashboard)\n", vimp_output_path))

cat("\nBiostatistical inference pipeline complete.\n")
