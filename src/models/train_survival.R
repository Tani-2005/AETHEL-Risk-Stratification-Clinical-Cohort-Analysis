# =============================================================================
# AETHEL: Modular Survival Model Trainer
# =============================================================================
# Fits Cox PH and Random Survival Forest models on arbitrary split datasets.
# Outputs predicted risk scores for each split to predictions CSVs.
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("survival", quietly = TRUE)) {
  install.packages("survival")
}
if (!requireNamespace("randomForestSRC", quietly = TRUE)) {
  install.packages("randomForestSRC")
}
if (!requireNamespace("yaml", quietly = TRUE)) {
  install.packages("yaml")
}

library(survival)
library(randomForestSRC)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript train_survival.R <train_csv> <val_csv> <test_csv_or_none> <output_dir>")
}

train_path <- args[1]
val_path   <- args[2]
test_path  <- args[3]
output_dir <- args[4]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Read training data
train_df <- read.csv(train_path)
val_df   <- read.csv(val_path)

test_df <- NULL
if (test_path != "none" && file.exists(test_path)) {
  test_df <- read.csv(test_path)
}

# Check if survival columns exist
has_survival <- "months_observed" %in% names(train_df) && "event_occurred" %in% names(train_df)

if (!has_survival) {
  cat("Survival columns ('months_observed', 'event_occurred') not found in training data. Skipping survival model training.\n")
  # Write empty prediction files to signal no survival model prediction
  write.csv(data.frame(cox_risk=numeric(0), rsf_risk=numeric(0)), file.path(output_dir, "train_survival_preds.csv"), row.names = FALSE)
  write.csv(data.frame(cox_risk=numeric(0), rsf_risk=numeric(0)), file.path(output_dir, "val_survival_preds.csv"), row.names = FALSE)
  if (!is.null(test_df)) {
    write.csv(data.frame(cox_risk=numeric(0), rsf_risk=numeric(0)), file.path(output_dir, "test_survival_preds.csv"), row.names = FALSE)
  }
  quit(status = 0)
}

exclude_cols <- c("patient_id", "months_observed", "event_occurred", "h_outcome_binary", "_source_dataset", "X_source_dataset")
covariates   <- setdiff(names(train_df), exclude_cols)

# Filter down to covariates that are not completely NA and have variance
covariates <- keep <- sapply(covariates, function(col) {
  x <- train_df[[col]]
  !all(is.na(x)) && length(unique(x[!is.na(x)])) > 1
})
covariates <- names(covariates)[covariates]

cat(sprintf("Fitting survival models with %d covariates: %s\n", length(covariates), paste(covariates, collapse = ", ")))

# Load config to get hyperparameters
config <- NULL
if (file.exists("configs/default.yaml")) {
  config <- yaml::read_yaml("configs/default.yaml")
}
r_seed <- if (!is.null(config$seeds$r)) config$seeds$r else 123
rsf_ntree <- if (!is.null(config$model_params$rsf_ntree)) config$model_params$rsf_ntree else 500

# Formulas
surv_obj    <- Surv(time = train_df$months_observed, event = train_df$event_occurred)
cox_formula <- as.formula(paste("surv_obj ~", paste(covariates, collapse = " + ")))
rsf_formula <- as.formula(paste("Surv(months_observed, event_occurred) ~", paste(covariates, collapse = " + ")))

# Fit Cox PH
cox_model <- tryCatch({
  coxph(cox_formula, data = train_df)
}, error = function(e) {
  cat("Error fitting Cox PH model: ", e$message, "\n")
  NULL
})

# Fit RSF
rsf_model <- tryCatch({
  rfsrc(rsf_formula, data = train_df, ntree = rsf_ntree, seed = r_seed, importance = FALSE)
}, error = function(e) {
  cat("Error fitting RSF model: ", e$message, "\n")
  NULL
})

# Predict helper
get_predictions <- function(df) {
  res <- data.frame(
    cox_risk = rep(NA_real_, nrow(df)),
    rsf_risk = rep(NA_real_, nrow(df))
  )
  
  if (!is.null(cox_model)) {
    cox_pred <- tryCatch({
      predict(cox_model, newdata = df, type = "risk")
    }, error = function(e) {
      NULL
    })
    if (!is.null(cox_pred) && length(cox_pred) == nrow(df)) {
      res$cox_risk <- cox_pred
    }
  }
  
  if (!is.null(rsf_model)) {
    rsf_pred <- tryCatch({
      predict(rsf_model, newdata = df)$predicted
    }, error = function(e) {
      NULL
    })
    if (!is.null(rsf_pred) && length(rsf_pred) == nrow(df)) {
      res$rsf_risk <- rsf_pred
    }
  }
  
  return(res)
}

# Generate and save predictions
train_preds <- get_predictions(train_df)
val_preds   <- get_predictions(val_df)

write.csv(train_preds, file.path(output_dir, "train_survival_preds.csv"), row.names = FALSE)
write.csv(val_preds, file.path(output_dir, "val_survival_preds.csv"), row.names = FALSE)

if (!is.null(test_df)) {
  test_preds <- get_predictions(test_df)
  write.csv(test_preds, file.path(output_dir, "test_survival_preds.csv"), row.names = FALSE)
}

cat("Survival model fitting and prediction generation complete.\n")
