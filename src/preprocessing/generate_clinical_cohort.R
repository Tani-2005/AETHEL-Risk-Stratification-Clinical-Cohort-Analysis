# =============================================================================
# AETHEL: Clinical Cohort Generation
# Stage: Preprocessing (R)
# =============================================================================
# Reads study parameters from configs/default.yaml.
# Writes: data/raw/synthetic_clinical_cohort.csv
# =============================================================================

cat("Initialising AETHEL Clinical Cohort Generator...\n")

# Load config from YAML
if (!requireNamespace("yaml", quietly = TRUE)) {
  stop(paste(
    "The 'yaml' R package is required to read configs/default.yaml.",
    "Install with: install.packages('yaml')"
  ))
}

config <- yaml::read_yaml("configs/default.yaml")

# Extract parameters from config (with safe defaults)
n_subjects     <- config$study_parameters$n_subjects %||% 1000
r_seed         <- config$seeds$r %||% 123
output_path    <- config$data_paths$raw_clinical %||% "data/raw/synthetic_clinical_cohort.csv"

# R does not have a native %||% operator — define it
`%||%` <- function(x, y) if (!is.null(x)) x else y

# Re-extract now that %||% is defined
n_subjects     <- config$study_parameters$n_subjects %||% 1000
r_seed         <- config$seeds$r %||% 123
output_path    <- config$data_paths$raw_clinical %||% "data/raw/synthetic_clinical_cohort.csv"

cat(sprintf("  n_subjects : %d\n", n_subjects))
cat(sprintf("  r_seed     : %d\n", r_seed))
cat(sprintf("  output     : %s\n", output_path))

# ---------------------------------------------------------------------------
# Deterministic seed — value sourced from configs/default.yaml → seeds.r
# ---------------------------------------------------------------------------
set.seed(r_seed)

# ---------------------------------------------------------------------------
# 1. Baseline Demographics & Clinical Factors
# ---------------------------------------------------------------------------
patient_id <- paste0("PT_", sprintf("%04d", 1:n_subjects))
age        <- round(runif(n_subjects, 40, 85))
bmi        <- rnorm(n_subjects, mean = 26, sd = 4)
is_smoker  <- rbinom(n_subjects, 1, prob = 0.25)

# ---------------------------------------------------------------------------
# 2. Precision Medicine: Polygenic Risk Score (Standardised)
# ---------------------------------------------------------------------------
genomic_risk_score <- rnorm(n_subjects, mean = 0, sd = 1)

# ---------------------------------------------------------------------------
# 3. Population Health: Townsend Deprivation Index
# ---------------------------------------------------------------------------
townsend_index <- rnorm(n_subjects, mean = 0, sd = 2)

# ---------------------------------------------------------------------------
# 4. Survival Dynamics: Time to Respiratory Event (months, max 60)
# Hazard increases with age, smoking, genetic risk, and deprivation.
# ---------------------------------------------------------------------------
hazard_rate    <- 0.001 * exp(0.02 * age + 0.8 * is_smoker + 0.3 * genomic_risk_score + 0.1 * townsend_index)
time_to_event  <- rexp(n_subjects, rate = hazard_rate)

# Right-censoring at 60 months (5-year study window)
event_occurred  <- ifelse(time_to_event <= 60, 1, 0)
months_observed <- pmin(time_to_event, 60)

# ---------------------------------------------------------------------------
# Compile and save
# ---------------------------------------------------------------------------
clinical_cohort <- data.frame(
  patient_id, age, bmi, is_smoker, genomic_risk_score,
  townsend_index, months_observed, event_occurred
)

# Ensure output directory exists
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
write.csv(clinical_cohort, output_path, row.names = FALSE)

cat(sprintf("Success! Clinical cohort (%d subjects) saved to %s\n", n_subjects, output_path))
