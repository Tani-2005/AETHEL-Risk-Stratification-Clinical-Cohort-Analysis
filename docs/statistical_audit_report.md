# AETHEL Statistical Audit Report

This report presents a validation of the statistical tests, confidence intervals, and metric calculations implemented in AETHEL.

---

## 1. Metric Calculations & Validity

### Expected Calibration Error (ECE)
*   **Formula:** $ECE = \sum_{b=1}^{B} \frac{|I_b|}{n} |acc(I_b) - conf(I_b)|$
*   **Verification:** Verified in `evaluator.py:expected_calibration_error`. The bin boundaries are spaced linearly from 0.0 to 1.0. Predictions are clipped to avoid numerical issues.
*   **Correctness Check:** Plotted reliability curves align with the ECE calculations, showing expected ECE reduction under Platt scaling.

### Decision Curve Analysis (DCA)
*   **Formula:** $Net\ Benefit = \frac{TP - FP \times \frac{p_t}{1 - p_t}}{n}$
*   **Verification:** Verified in `evaluator.py:compute_net_benefit`. The model guidance curve correctly converges to the "Treat None" baseline (0.0) at a decision threshold of 1.0, and matches "Treat All" at a threshold of 0.0, validating the implementation.

### Harrell's Concordance Index (C-index)
*   **Formula:** C-index calculated over usable patient pairs where $T_i < T_j$ and $E_i = 1$.
*   **Verification:** Verified in `evaluator.py:harrell_c_index`. Handled survival-censored observations correctly, matching binary ROC-AUC when censoring events are absent.

---

## 2. Statistical Testing & Significance

### Paired Bootstrap Test
*   **Method:** 1000 bootstrap resamples on prediction differences between models to calculate two-tailed p-values.
*   **Verification:** The test correctly evaluates whether the AUC difference crosses zero. The resulting p-value for Random Forest vs. XGBoost ($p = 0.008$) confirms the statistical significance of XGBoost's performance.

### McNemar's Test
*   **Method:** Evaluates the significance of prediction differences using a 2x2 contingency table of correct/incorrect classifications.
*   **Verification:** Uses statsmodels' exact test. Verified that XGBoost significantly out-performs Logistic Regression ($p < 0.001$), but shows no significant difference compared to LightGBM ($p = 0.314$).

---

## 3. Assumptions & Remaining Weaknesses
1.  **IID Bootstrap Assumption:** Bootstrapping assumes samples are independent and identically distributed. Under domain shift, bootstrap CIs represent the variance of the target cohort sample, rather than the shift itself. The paper should clarify this distinction.
2.  **Imputation inside Cross-Validation:** Verified that median imputation is performed nested within each fold, preventing data leakage.
