# AETHEL: Clinical Cohort Generation
cat("Initializing AETHEL Clinical Cohort Generator...\n")

set.seed(123) # Reproducibility constraint
n_subjects <- 1000

# 1. Baseline Demographics & Clinical Factors
patient_id <- paste0("PT_", sprintf("%04d", 1:n_subjects))
age <- round(runif(n_subjects, 40, 85))
bmi <- rnorm(n_subjects, mean = 26, sd = 4)
is_smoker <- rbinom(n_subjects, 1, prob = 0.25)

# 2. Precision Medicine: Polygenic Risk Score (Standardized)
genomic_risk_score <- rnorm(n_subjects, mean = 0, sd = 1)

# 3. Population Health: Townsend Deprivation Index (UK-specific socio-economic metric)
townsend_index <- rnorm(n_subjects, mean = 0, sd = 2)

# 4. Survival Dynamics: Time to Respiratory Event (in months, max 60)
# Hazard increases with age, smoking, genetic risk, and deprivation
hazard_rate <- 0.001 * exp(0.02*age + 0.8*is_smoker + 0.3*genomic_risk_score + 0.1*townsend_index)
time_to_event <- rexp(n_subjects, rate = hazard_rate)

# Right-Censoring: If the event happens after our 5-year (60 month) study, they are "censored"
event_occurred <- ifelse(time_to_event <= 60, 1, 0)
months_observed <- pmin(time_to_event, 60)

# Compile the master clinical frame
clinical_cohort <- data.frame(
  patient_id, age, bmi, is_smoker, genomic_risk_score, 
  townsend_index, months_observed, event_occurred
)

# Save to the raw data vault
write.csv(clinical_cohort, "data/raw/synthetic_clinical_cohort.csv", row.names = FALSE)
cat("Success! Clinical cohort saved to data/raw/synthetic_clinical_cohort.csv\n")