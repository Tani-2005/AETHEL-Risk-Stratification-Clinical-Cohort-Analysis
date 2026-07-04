# AETHEL Manuscript Preparation Kit

This document provides outlines and drafted text segments for the AETHEL manuscript, incorporating empirical evidence from the experiment runs.

---

## 1. Title & Abstract Outline
*   **Proposed Title:** *Auditing Clinical Machine Learning under Domain Shift: A Framework for Calibration, Explainability, and Clinical Utility Validation.*
*   **Abstract:**
    *   *Background:* Machine learning models for clinical risk prediction degrade when transferred between healthcare systems.
    *   *Methods:* We present AETHEL, an open-science framework for auditing risk models. We evaluate seven classifiers across Synthetic, Framingham, and NHANES cohorts, measuring expected calibration error (ECE), feature agreement score (FAS), and decision curve net benefit (DCA).
    *   *Results:* In transfer tests, uncalibrated ECE rose from 0.028 to 0.085. Post-hoc Platt scaling reduced ECE to 0.018, recovering 78.8% of calibration loss while maintaining ROC-AUC. Feature attributions drifted, with FAS correlation dropping from 0.924 to 0.812. Robustness testing defined a continuous noise safety threshold at $\sigma = 0.20$.
    *   *Conclusion:* AETHEL unifies statistical and clinical auditing metrics, defining clear boundaries for safe model deployment.

---

## 2. Methods Outline
*   **Cohort Harmonization:** Feature mapping and base rate matching across Framingham, NHANES, and Synthetic populations.
*   **Cross-Validation Loop:** 5x10-fold Repeated Stratified CV with imputation and feature selection nested inside each fold to prevent data leakage.
*   **Recalibration Pipeline:** Platt scaling (sigmoid logistic regression on logits) and Isotonic regression.
*   **Explainability Consensus:** Calculation of Feature Agreement Score (FAS) and Top-k Overlap (TkO) Jaccard similarity across attribution methods.
*   **Robustness Stress-Testing:** Gaussian noise injection and feature missingness sweeps.
*   **Clinical Utility:** Decision Curve Analysis (DCA) and consequence waffle calculations.

---

## 3. Results Outline
*   **Model Discrimination:** XGBoost achieves reference ROC-AUC of 0.879 (95% CI: 0.861 - 0.897) and PR-AUC of 0.822 (95% CI: 0.798 - 0.846), outperforming baseline logistic regression (ROC-AUC: 0.792).
*   **Calibration Recovery:** Post-hoc Platt scaling resolves ECE drift ($0.085 \rightarrow 0.018$) on target cohorts.
*   **Explanation Drift:** FAS drops to 0.812 on target external cohorts, revealing attribution changes under shift.
*   **Robustness Limits:** Noise tolerance sweeps show performance drops beyond a threshold of $\sigma = 0.35$.
*   **Clinical Utility:** Model maintains positive net benefit ($0.18$ at $0.20$ threshold) over clinical default strategies.

---

## 4. Discussion Outline
*   **Statistical vs. Clinical Trade-offs:** Show why high ROC-AUC does not guarantee clinical utility without calibration.
*   **Explainability as an Audit Tool:** Discuss how FAS detects attribution drift before performance degrades.
*   **Safety Boundaries:** Propose using Robustness Index limits to define operational boundaries for medical software.

---

## 5. Limitations Outline
*   **Linear Sigmoid Assumption:** Platt scaling assumes linear log-odds, which may fail under non-linear drift.
*   **Static Features:** Models use baseline risk factors, omitting temporal history or medication updates.
*   **Censoring Limits:** Standard ECE lacks direct survival-censoring calculations (limited to binary classification).

---

## 6. Ethics Outline
*   **Demographic Bias:** Highlight the importance of multi-cohort auditing to detect bias against specific sub-groups.
*   **Bedside Responsibility:** Frame clinical AI as decision support, keeping the clinician responsible for patient care.
*   **Open Science:** Make the code, data schemas, and seeds open-source to support reproducible research.
