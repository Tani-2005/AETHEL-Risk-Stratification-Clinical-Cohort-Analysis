# AETHEL Scientific Claims-to-Evidence Matrix

This document maps the primary scientific claims of the AETHEL framework to their empirical evidence, supporting figures, tables, statistical tests, literature citations, discussion points, and limitations.

---

## Claim 01: Post-Hoc Calibration Decoupling Recoverability
*   **Scientific Claim:** Post-hoc calibration using Platt scaling or Isotonic regression can correct calibration drift ($ECE$ decay) induced by demographic or domain transfer without degrading model discrimination ($AUC$).
*   **Empirical Evidence:** In transferring an XGBoost model from the Synthetic source cohort to the Framingham target cohort, baseline uncalibrated $ECE$ rises from 0.028 to 0.085. Applying Platt scaling reduces $ECE$ to 0.018, recovering 78.8% of the calibration loss. Meanwhile, the ROC-AUC remains stable at 0.743 (target transfer baseline).
*   **Supporting Figures:** Figure 2 (Calibration Reliability Diagram & scaling curves in the Publication Figure Gallery).
*   **Supporting Tables:** Table 4 (Calibration Metrics) & Table 5 (Cross-Cohort Validation Table).
*   **Supporting Statistical Test:** Expected Calibration Error (ECE) partition bin difference, testing Platt-scaled predictions against uncalibrated distributions.
*   **Supporting Literature:** Guo et al. (2017) *On Calibration of Modern Neural Networks* (ICML).
*   **Discussion Points:** 
    *   Show how decoupling calibration from model training is highly effective for clinical safety under domain shift.
    *   Highlight why tree ensemble predictions tend to push probability outputs towards margins, making post-hoc Platt sigmoid repairs mathematically suitable.
*   **Limitations:** Sigmoid Platt scaling assumes the log-odds of predictions are linear, which may fail in cases of extreme, multi-modal domain shift.

---

## Claim 02: Cross-Cohort Explainability Consensus
*   **Scientific Claim:** Local and global explainability attributions (SHAP values) exhibit significant rank correlation drift when models are evaluated on target external cohorts, which necessitates consensus-based audit metrics.
*   **Empirical Evidence:** The Feature Agreement Score (FAS) across three importance methods (SHAP, Permutation, and Native) drops from 0.924 on the source validation cohort to 0.812 when evaluated on the Framingham cohort, indicating a 12.1% drift in explanation consensus under transfer shift.
*   **Supporting Figures:** Figure 4 (Cross-Cohort SHAP explanation consensus rank correlation heatmap).
*   **Supporting Tables:** Table 6 (Feature Importance Consensus) & Table 7 (Explanation Stability Table).
*   **Supporting Statistical Test:** Spearman rank correlation coefficient ($\rho$) and Jaccard similarity metrics (Top-k overlap).
*   **Supporting Literature:** Lundberg & Lee (2017) *A Unified Approach to Interpreting Model Predictions* (NeurIPS).
*   **Discussion Points:**
    *   Argue that validating model decisions based on feature importances from a single cohort is insufficient for multi-center deployment.
    *   Explain how the Feature Agreement Score (FAS) acts as a warning indicator for clinical feature attribution drift.
*   **Limitations:** Correlation-based consensus metrics (Spearman $\rho$) do not capture non-linear, interactive attribution changes between features.

---

## Claim 03: Vulnerability Boundary Stress Degradation
*   **Scientific Claim:** Controlled stress testing via Gaussian noise injection and feature ablation reveals non-linear model degradation thresholds, establishing safety boundaries for clinical deployment.
*   **Empirical Evidence:** The XGBoost model maintains performance up to a continuous noise level of $\sigma = 0.20$ (ROC-AUC remains above 0.82), but decays rapidly beyond $\sigma = 0.35$ (ROC-AUC drops below 0.70). Missing data (MCAR) tolerance shows a failure threshold at 40% missingness.
*   **Supporting Figures:** Figure 3 (2D Robustness Sensitivity Heatmap & missingness curves).
*   **Supporting Tables:** Table 8 (Robustness Metrics Table) & Table 9 (Failure Mode Analysis).
*   **Supporting Statistical Test:** Bootstrap prediction variance sweeps.
*   **Supporting Literature:** Subbaswamy et al. (2019) *Preventing Database Shift Failures in Clinical Machine Learning* (NeurIPS ML4H).
*   **Discussion Points:**
    *   Explain why clinical models must be stress-tested against data corruptions (e.g., faulty EHR sensor logs, mismatched units) before deployment.
    *   Propose the Robustness Index as a standardized metric for medical software validation.
*   **Limitations:** Synthetic noise injection (Gaussian, random flips) is a surrogate and may not fully replicate structured clinical missingness patterns.

---

## Claim 04: Clinical Net Benefit & Decision Curve Trade-offs
*   **Scientific Claim:** Standard statistical metrics (ROC-AUC) do not guarantee positive clinical utility; models must be audited using Decision Curve Analysis (DCA) to confirm net benefit across realistic treatment thresholds.
*   **Empirical Evidence:** XGBoost maintains positive Net Benefit compared to "Treat All" and "Treat None" baselines across decision thresholds from $0.05$ to $0.65$. At a standard clinical threshold of $0.20$, the model provides a Net Benefit of $0.18$, translating to 180 true positives detected per 1,000 patients without unnecessary treatment harms.
*   **Supporting Figures:** Figure 1 (Decision Curve Net Benefit & Bedside Utility zoom modal).
*   **Supporting Tables:** Table 3 (Performance Metrics) & Table 10 (Clinical Utility Summary Table).
*   **Supporting Statistical Test:** Decision Curve Analysis (DCA) Net Benefit integrals.
*   **Supporting Literature:** Steyerberg et al. (2010) *Assessing the Performance of Prediction Models: A Framework for Traditional and Novel Measures* (Epidemiology).
*   **Discussion Points:**
    *   Highlight why medical regulators should prioritize DCA curves over simple classification metrics.
    *   Demonstrate how the 100-cell consequence waffle chart bridges the gap between biostatistics and bedside decision-making.
*   **Limitations:** Net Benefit assumes the cost-benefit ratio of treatment harms is perfectly represented by the decision threshold, which varies across patient preferences.
