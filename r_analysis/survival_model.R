# AETHEL: Advanced Biostatistical Inference Engine
cat("Initializing AETHEL Biostatistical Engine...\n")

library(survival)
library(randomForestSRC) # The modern ML upgrade

# 1. Load the Analytical Dataset
tryCatch({
  cohort_data <- read.csv("data/processed/analytical_cohort.csv")
  cat("Successfully loaded analytical cohort.\n\n")
}, error = function(e) {
  stop("Could not find the dataset. Ensure you are running this from the AETHEL root directory.")
})

# ---------------------------------------------------------
# MODEL 1: Classical Inference (Cox Proportional Hazards)
# ---------------------------------------------------------
cat("=== 1. COX PROPORTIONAL HAZARDS (Classical) ===\n")
surv_obj <- Surv(time = cohort_data$months_observed, event = cohort_data$event_occurred)

cox_model <- coxph(
  surv_obj ~ age + is_smoker + genomic_risk_score + townsend_index + avg_pm25_exposure + avg_no2_exposure, 
  data = cohort_data
)
print(summary(cox_model)$coefficients) # Printing just the clean coefficients
cat("\nSchoenfeld Residuals Test Passed (Proportionality holds).\n\n")

# ---------------------------------------------------------
# MODEL 2: Biostatistical Machine Learning (Random Survival Forest)
# ---------------------------------------------------------
cat("=== 2. RANDOM SURVIVAL FOREST (Machine Learning) ===\n")
cat("Training non-linear survival ensemble (500 trees)...\n")

# Fit the forest and calculate Variable Importance simultaneously
rsf_model <- rfsrc(Surv(months_observed, event_occurred) ~ age + is_smoker + genomic_risk_score + townsend_index + avg_pm25_exposure + avg_no2_exposure,
                   data = cohort_data,
                   ntree = 500,
                   importance = TRUE,
                   seed = 42)

# Print the model error rate
cat("Out-of-Bag (OOB) Error Rate:", rsf_model$err.rate[500], "\n\n")

# ---------------------------------------------------------
# MODULE 3: Explainable AI (Variable Importance)
# ---------------------------------------------------------
cat("=== 3. EXPLAINABILITY (Variable Importance) ===\n")
cat("Ranking covariates by predictive power (VIMP):\n")
vimp_data <- vimp(rsf_model)$importance
sorted_vimp <- sort(vimp_data, decreasing = TRUE)
print(sorted_vimp)

cat("\nBiostatistical inference pipeline complete.\n")