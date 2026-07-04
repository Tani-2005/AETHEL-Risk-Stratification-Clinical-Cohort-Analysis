# AETHEL Literature Integration Matrix

This document maps key papers in clinical machine learning, calibration, explainability, and clinical decision support to the AETHEL framework, detailing how AETHEL differs and where each paper should be cited.

---

## 1. Literature Mapping

| Citation | Category | Key Finding | How AETHEL Differs | Manuscript Section & Citation |
| :--- | :--- | :--- | :--- | :--- |
| **Guo et al. (2017)** *ICML* | Calibration | Platt scaling and Temperature scaling correct neural net calibration. | AETHEL evaluates post-hoc calibration specifically under cross-cohort domain transfer shifts, whereas Guo assumes IID data. | Cite in Section 3.2 (Calibration Under Shift). `\cite{guo2017calibration}` |
| **Steyerberg et al. (2010)** *Epidemiology* | Clinical Utility | Renders Net Benefit curves and establishes DCA. | AETHEL automates DCA calculation and integrates a bedside 100-cell consequence waffle projection directly into the model audit. | Cite in Section 4.3 (Clinical Utility Analysis). `\cite{steyerberg2010assessing}` |
| **Lundberg & Lee (2017)** *NeurIPS* | Explainable AI | Formulates SHAP values as a unified feature attribution. | AETHEL implements rank correlation check metrics (FAS, Jaccard overlap) to audit explainability drift when models transfer between clinical sites. | Cite in Section 3.4 (Feature Attribution Stability). `\cite{lundberg2017unified}` |
| **Subbaswamy et al. (2019)** *NeurIPS ML4H* | Trustworthy AI / Drift | Identifies domain shifts and proposes dataset intervention bounds. | AETHEL implements an active, stress-testing audit console (Gaussian noise, missingness sweeps) to empirically define safety boundaries. | Cite in Section 4.1 (Robustness Audit Boundaries). `\cite{subbaswamy2019preventing}` |

---

## 2. Literature Gaps Filled by AETHEL

1.  **Calibration-Explainability Decoupling:** Existing literature treats calibration (Guo et al.) and local explainability (Lundberg & Lee) as disjoint problems. AETHEL establishes a unified auditing framework that evaluates how post-hoc calibration repairs interact with local feature attribution consensus.
2.  **Translating Theory to Bedside Utility:** While biostatisticians emphasize DCA Net Benefit (Steyerberg et al.), clinicians find it difficult to translate probability threshold curves into treatment decisions. AETHEL solves this by projecting abstract net benefit values into concrete counts of true and false positives per 1,000 patients.
3.  **Empirical Safety Boundaries:** Theoretical work on domain shift (Subbaswamy et al.) often relies on unobserved causal graphs. AETHEL provides an empirical approach, stress-testing models under controlled noise and missingness to define operational thresholds.
