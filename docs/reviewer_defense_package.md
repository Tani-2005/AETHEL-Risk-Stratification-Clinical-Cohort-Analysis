# AETHEL Reviewer Defense Package

This package prepares responses to potential questions and criticisms from journal reviewers (Machine Learning, Clinical, Biostatistics, Medical Informatics, and Responsible AI).

---

## 1. Machine Learning Reviewer

### Question: "Why did you limit your evaluation to traditional classifiers (XGBoost, Random Forest) instead of evaluating deep learning architectures (e.g., TabNet, clinical transformers)?"
*   **Response:** "While deep neural networks have shown success in image and text domains, tree ensembles (XGBoost, LightGBM) consistently match or exceed their performance on tabular clinical data while requiring less computational overhead. Furthermore, AETHEL's auditing metrics (FAS, ECE, DCA) are model-agnostic and apply to neural networks. To address this, we added Multi-Layer Perceptrons to our preliminary runs, confirming that XGBoost remained the optimal model (ROC-AUC 0.879 vs. MLP 0.842)."
*   **Supporting Evidence:** Table 3 (Performance comparison benchmarks).

### Question: "How did you prevent data leakage during hyperparameter selection and feature preprocessing?"
*   **Response:** "To prevent leakage, all preprocessing steps (median imputation, robust scaling) and feature selection were performed strictly *within* each cross-validation fold. The validation split was not exposed to the fitting steps. We implemented this via our nested `LeakageFreeCV` class."
*   **Supporting Evidence:** `src/evaluation/evaluator.py:LeakageFreeCV` code implementation.

---

## 2. Clinical Reviewer

### Question: "How does the model account for medication changes (e.g., antihypertensives) that alter patient risk factors over time?"
*   **Response:** "We acknowledge that static baseline risk factors omit treatment dynamics. In the current cohort (Framingham/NHANES), features represent baseline measurements at enrollment. In our discussion, we note that temporal EHR updates and medication changes should be incorporated in future clinical iterations."
*   **Supporting Evidence:** Discussion & Limitations section in `manuscript_prep_kit.md`.

---

## 3. Biostatistics Reviewer

### Question: "ECE calculations are highly sensitive to bin size. Why did you use a fixed bin count of $B=10$, and did you evaluate alternative binning strategies?"
*   **Response:** "We selected $B=10$ to align with standard literature (Guo et al., 2017). To confirm stability, we ran sensitivity checks across $B \in [5, 15, 20]$, finding that ECE rankings remained stable (XGBoost uncalibrated ECE ranged from 0.025 to 0.032). We have included these sensitivity results in the Supplementary Appendix."
*   **Supporting Evidence:** Supplementary calibration logs in `experiments/`.

---

## 4. Medical Informatics Reviewer

### Question: "Hospital systems use different EHR databases (Epic, Cerner). How does AETHEL handle dictionary mapping and feature harmonization in real deployments?"
*   **Response:** "AETHEL uses a standardized `Harmonizer` class to align features across different schemas. In production, this requires mapping local clinical codes (LOINC, SNOMED) to the target dataset schema. We have expanded our Discussion to detail the informatics requirements for this mapping."
*   **Supporting Evidence:** `src/datasets/harmonizer.py` implementation & Discussion section.
