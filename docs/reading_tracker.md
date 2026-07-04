# Literature Reading Tracker & Paper Review Template

This document tracks literature read, analyzed, and cited for the AETHEL clinical AI auditing framework.

---

## Master Bibliography Index

Refer to the companion [bibliography_template.bib](file:///c:/Antigravity%20Projects/AETHEL/docs/bibliography/bibliography_template.bib) for LaTeX/BibTeX references.

| Citation | Category | Status | Core Contribution to AETHEL |
| :--- | :--- | :--- | :--- |
| **[Steyerberg2010]** Steyerberg et al. (2010) | Clinical Decision Support | Read | Established Net Benefit and DCA foundations |
| **[Guo2017]** Guo et al. (2017) | Calibration | Read | ECE calculations and Platt scaling for deep/tree nets |
| **[Lundberg2017]** Lundberg & Lee (2017) | Explainable AI | Read | Classic SHAP value formulation |
| **[Subbaswamy2019]** Subbaswamy et al. (2019) | Trustworthy AI / Drift | Read | Domain shift bounds and algorithmic safety |

---

## Paper Review Template

Use this template to document every paper reviewed for AETHEL. Save individual reviews inside the `docs/notes/` directory or append them below.

### [Template: Title of the Paper]

*   **Citation:** `[AuthorYear]` Full citation (e.g. Authors, Journal, Year).
*   **Category:** [Literature Review / IEEE / Medical / Explainable AI / Trustworthy AI / Calibration / Robustness / Clinical Decision Support / Supplementary]
*   **Research Question:** What is the primary question or hypothesis addressed by the authors?
*   **Dataset:** What validation/clinical datasets were used? (Sample size, features, cohort details).
*   **Methods:** What classifiers, post-hoc methods, or auditing techniques were implemented?
*   **Strengths:** What are the key technical or clinical merits of this work?
*   **Weaknesses:** What are the limitations, blind spots, or flaws in this paper?
*   **Relevance to AETHEL:** How does this relate to domain shift, calibration, explainability, or clinical utility in AETHEL?
*   **Ideas Inspired:** What concepts did this paper trigger for AETHEL's code or manuscript?
*   **Possible Citations:** Text or BibTeX citations to include in the manuscript.

---

## Active Review Logs

### Review 1: On Calibration of Modern Neural Networks (Guo et al., 2017)
*   **Citation:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *International Conference on Machine Learning (ICML)*.
*   **Category:** Calibration
*   **Research Question:** Why do modern neural networks produce uncalibrated probability predictions, and how can post-hoc scaling fix this?
*   **Dataset:** ImageNet, CIFAR-100, SVHN, various NLP datasets.
*   **Methods:** Expected Calibration Error (ECE), Temperature Scaling, Platt Scaling, Isotonic Regression.
*   **Strengths:** Popularized ECE as a simple metric and proved that temperature scaling is highly effective for neural net calibration.
*   **Weaknesses:** Relies heavily on the assumption that validation and test sets are drawn from the same underlying distribution (no domain shift).
*   **Relevance to AETHEL:** AETHEL extends post-hoc calibration audits specifically to scenarios with *domain shift* (e.g., source to target transfer).
*   **Ideas Inspired:** Implementing adaptive/standard binning for ECE and comparing Platt scaling vs. Isotonic regression post-hoc.
*   **Possible Citations:** `\cite{guo2017calibration}`

### Review 2: Decision Curve Analysis (Steyerberg et al., 2010)
*   **Citation:** Steyerberg, E. W., Vickers, A. J., Cook, N. R., et al. (2010). Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*.
*   **Category:** Clinical Decision Support
*   **Research Question:** How can researchers calculate the clinical usefulness of a risk prediction model rather than just its statistical accuracy?
*   **Dataset:** Cardiovascular and cancer screening registry datasets.
*   **Methods:** Decision Curve Analysis (DCA), Net Benefit calculation.
*   **Strengths:** Integrates patient and clinician preferences directly into the evaluation by parameterizing the cost-benefit ratio as a decision threshold.
*   **Weaknesses:** Calculation assumes the decision threshold represents a perfect proxy for the clinical utility trade-off.
*   **Relevance to AETHEL:** Renders Net Benefit curves in the Scientific Evidence and Clinical Utility tabs.
*   **Ideas Inspired:** 100-cell consequence waffle charts translating mathematical Net Benefit into concrete counts of true and false positives per 1,000 patients.
*   **Possible Citations:** `\cite{steyerberg2010assessing}`
