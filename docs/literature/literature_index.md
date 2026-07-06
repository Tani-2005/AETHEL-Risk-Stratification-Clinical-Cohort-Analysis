# Master Literature Index

This index catalogues all 43 research papers reviewed for the AETHEL clinical machine learning auditing framework, organized by their scientific categories. Physical PDF paths are preserved in single directories, with cross-references mapped here.

## Table of Contents
- [01 Foundational Methods](#01_foundational_methods)
- [02 Survival Analysis](#02_survival_analysis)
- [03 Clinical Risk Prediction](#03_clinical_risk_prediction)
- [04 Explainable AI](#04_explainable_ai)
- [05 Calibration](#05_calibration)
- [06 External Validation](#06_external_validation)
- [07 Decision Curve Analysis](#07_decision_curve_analysis)
- [08 Trustworthy AI](#08_trustworthy_ai)
- [09 Fairness](#09_fairness)
- [10 Reporting Guidelines](#10_reporting_guidelines)
- [11 Background Reviews](#11_background_reviews)

---

## 01 Foundational Methods

### Developing clinical prediction models: a step by step guide

*   **PDF Path:** `docs/literature/01_Foundational_Methods/Developing clinical prediction models a step by step guide.pdf`
*   **Authors:** Orestis Efthimiou, et al.
*   **Year:** 2023
*   **Journal:** BMJ
*   **One-Paragraph Summary:** This paper provides a comprehensive, clinical-grade guide to developing predictive algorithms, detailing sample size considerations, variable selection, handling of missing data, and validating predictive capabilities before translation.
*   **Main Methodology:** Methodological tutorial and guideline.
*   **Main Contribution:** A practical step-by-step consensus-based roadmap for clinical predictive modeling.
*   **Strengths:** Clear, accessible translation of machine learning and biostatistics principles for clinical researchers.
*   **Weaknesses:** Lacks mathematical depth for complex post-hoc calibration and advanced XAI consensus metrics.
*   **How AETHEL Relates to It:** AETHEL implements this guide's recommendation for systematic development and external validation, extending it with automated ECE/DCA audits.
*   **Manuscript Citation Section:** Methods / Discussion

### Evaluation of clinical prediction models (part 1): from development to external validation

*   **PDF Path:** `docs/literature/01_Foundational_Methods/Evaluation of clinical prediction models (part 1).pdf`
*   **Authors:** Gary S. Collins, Paula Dhiman, Jie Ma, et al.
*   **Year:** 2024
*   **Journal:** BMJ
*   **One-Paragraph Summary:** The first article in a BMJ series outlining rigorous evaluation strategies for clinical prediction models. It defines internal, internal-external, and external validation, stressing the importance of transportability across demographics.
*   **Main Methodology:** Statistical validation guideline and framework description.
*   **Main Contribution:** Standardizes terminology and validation protocols for clinical risk models.
*   **Strengths:** Authored by top medical statisticians; emphasizes performance heterogeneity and transportability.
*   **Weaknesses:** Qualitative overview without automated code implementations or specific mathematical error bounds.
*   **How AETHEL Relates to It:** AETHEL executes the external validation and transportability audits advocated by Collins et al., assessing ECE and FAS drops under transfer shift.
*   **Manuscript Citation Section:** Introduction / Methods

### Youden Index and the optimal threshold for markers with mass at zero

*   **PDF Path:** `docs/literature/01_Foundational_Methods/Youden Index and the optimal threshold for markers with mass.pdf`
*   **Authors:** Enrique F. Schisterman, David Faraggi, Benjamin Reiser, Jessica Hu
*   **Year:** 2007
*   **Journal:** Statistics in Medicine
*   **One-Paragraph Summary:** Investigates the estimation of the Youden Index and the optimal decision threshold for biomarkers that have a positive mass at zero (e.g., coronary calcium score). It proposes a flexible mixture modeling approach to handle this zero inflation.
*   **Main Methodology:** Mixture model modeling, parametric and empirical ROC estimation.
*   **Main Contribution:** A mathematical formulation for Youden Index thresholds in zero-inflated biomarker data.
*   **Strengths:** Provides mathematical solutions for a common clinical data issue (zero inflation).
*   **Weaknesses:** Assumes specific parametric distributions which may fail for complex multi-modal clinical features.
*   **How AETHEL Relates to It:** AETHEL integrates Youden Index calculation in its evaluator module to compute optimal decision thresholds for skewed baseline clinical markers.
*   **Manuscript Citation Section:** Methods / Results


---

## 02 Survival Analysis

### Explainable AI for survival analysis: a median-SHAP approach

*   **PDF Path:** `docs/literature/02_Survival_Analysis/Explainable AI for survival analysis a median-SHAP approach.pdf`
*   **Authors:** Lucile Ter-Minassian, Sahra Ghalebikesabi, Karla Diaz-Ordaz, Chris Holmes
*   **Year:** 2024
*   **Journal:** arXiv preprint arXiv:2402.00072
*   **One-Paragraph Summary:** Exposes how standard mean-based SHAP anchors fail in survival models due to right-censoring, proposing 'median-SHAP' to construct local attributions around median survival time predictions.
*   **Main Methodology:** Local feature attribution, survival time estimation, Shapley value adjustments.
*   **Main Contribution:** Formulates median-SHAP for explainable survival analysis under right-censoring.
*   **Strengths:** Identifies a critical flaw in applying standard SHAP to survival models and provides a mathematically sound fix.
*   **Weaknesses:** Highly computationally intensive compared to standard linear approximations.
*   **How AETHEL Relates to It:** AETHEL uses survival-censoring evaluations and builds consensus metrics (FAS) comparing survival SHAP to standard attributions.
*   **Manuscript Citation Section:** Methods / Discussion

### Random survival forests

*   **PDF Path:** `docs/literature/02_Survival_Analysis/Random survival forests.pdf`
*   **Authors:** Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, Michael S. Lauer
*   **Year:** 2008
*   **Journal:** The Annals of Applied Statistics
*   **One-Paragraph Summary:** Introduces Random Survival Forests (RSF), an ensemble tree method for right-censored survival data. It formulates custom splitting rules based on log-rank statistics and defines ensemble mortality as a predictor.
*   **Main Methodology:** Ensemble learning, survival trees, log-rank splitting.
*   **Main Contribution:** Extension of Random Forests to right-censored time-to-event outcomes.
*   **Strengths:** Non-parametric method that handles complex interactions and non-linear hazards.
*   **Weaknesses:** Highly opaque black-box model requiring post-hoc explainability methods to understand feature importance.
*   **How AETHEL Relates to It:** AETHEL includes RSF as a benchmark model in its survival registry and training pipeline.
*   **Manuscript Citation Section:** Methods / Results

### Regression models and life-tables

*   **PDF Path:** `docs/literature/02_Survival_Analysis/Regression Models and Life-Tables.pdf`
*   **Authors:** David R. Cox
*   **Year:** 1972
*   **Journal:** Journal of the Royal Statistical Society: Series B
*   **One-Paragraph Summary:** The seminal paper presenting the Cox Proportional Hazards model. It enables regression modeling of censored survival data by leaving the baseline hazard function unspecified and maximizing a partial likelihood.
*   **Main Methodology:** Proportional hazards, partial likelihood estimation.
*   **Main Contribution:** The foundational Cox Proportional Hazards model.
*   **Strengths:** Mathematically elegant, semi-parametric, and highly interpretable model.
*   **Weaknesses:** Relies on the strong assumption of proportional hazards, which is frequently violated in real clinical cohorts.
*   **How AETHEL Relates to It:** AETHEL implements the Cox PH model as the primary baseline clinical model to compare against machine learning models (XGBoost/RF).
*   **Manuscript Citation Section:** Methods / Results


---

## 03 Clinical Risk Prediction

### General Cardiovascular Risk Profile for Use in Primary Care: The Framingham Heart Study

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/d-agostino-et-al-2008-general-cardiovascular-risk-profile-for-use-in-primary-care.pdf`
*   **Authors:** Ralph B. D'Agostino Sr., Ramachandran S. Vasan, Michael J. Pencina, et al.
*   **Year:** 2008
*   **Journal:** Circulation
*   **One-Paragraph Summary:** Derives a single multivariable risk function predicting 10-year risk of general cardiovascular disease. It uses Cox proportional hazards on Framingham participants, incorporating age, lipids, blood pressure, smoking, and diabetes.
*   **Main Methodology:** Cox regression, risk score derivation, Harrell's C-index validation.
*   **Main Contribution:** The widely used Framingham general CVD risk prediction algorithm.
*   **Strengths:** Highly practical, validated on thousands of patients, and clinically accepted.
*   **Weaknesses:** Derived from a predominantly white cohort, leading to poor generalization in diverse populations.
*   **How AETHEL Relates to It:** AETHEL utilizes the Framingham cohort dataset as an external target validation cohort to test models trained on synthetic or demographic sources.
*   **Manuscript Citation Section:** Introduction / Methods / Results

### 2013 ACC/AHA guideline on the assessment of cardiovascular risk

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/goff-et-al-2013-2013-acc-aha-guideline-on-the-assessment-of-cardiovascular-risk.pdf`
*   **Authors:** David C. Goff Jr., Donald M. Lloyd-Jones, et al.
*   **Year:** 2014
*   **Journal:** Circulation / Journal of the American College of Cardiology
*   **One-Paragraph Summary:** Establishes clinical guidelines for assessing cardiovascular risk, introducing the Pooled Cohort Equations (PCE) to estimate 10-year risk of ASCVD in black and white men and women.
*   **Main Methodology:** Clinical consensus guideline, pooled cohort regression.
*   **Main Contribution:** PCE equations for ASCVD risk estimation in primary prevention.
*   **Strengths:** First guideline to provide separate equations for race and gender to improve equity.
*   **Weaknesses:** Overestimates risk in modern cohorts, leading to over-treatment.
*   **How AETHEL Relates to It:** AETHEL benchmarks its machine learning models against the PCE variables and uses the standard treatment decision thresholds (e.g., 7.5% and 20%) in its DCA audits.
*   **Manuscript Citation Section:** Introduction / Discussion

### Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease: prospective cohort study

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/Development and validation of QRISK3 risk prediDevelopment and validation algorithms to estimate future risk of cardiovascular disease prospective cohort study.pdf`
*   **Authors:** Julia Hippisley-Cox, Carol Coupland, Peter Brindle
*   **Year:** 2017
*   **Journal:** BMJ
*   **One-Paragraph Summary:** Develops and validates the QRISK3 prediction algorithm to estimate 10-year risk of cardiovascular disease in the UK. It adds several new risk factors, such as chronic kidney disease, migraines, and mental health conditions.
*   **Main Methodology:** Prospective cohort study, Cox regression, huge database validation.
*   **Main Contribution:** QRISK3 risk prediction algorithm incorporating clinical comorbidities.
*   **Strengths:** Extremely large sample size (7.89 million patients) and comprehensive feature mapping.
*   **Weaknesses:** Uses traditional Cox regression models that cannot capture complex non-linear feature interactions.
*   **How AETHEL Relates to It:** AETHEL's multi-cohort registry implements feature harmonization similar to QRISK3 variables (e.g., incorporating renal and metabolic markers).
*   **Manuscript Citation Section:** Introduction / Discussion

### Explainable AI for Clinical Risk Prediction: A Survey of Concepts, Methods, and Modalities

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/EXPLAINABLE AI FOR CLINICAL RISK PREDICTION.pdf`
*   **Authors:** Munib Mesinovic, Peter Watkinson, Tingting Zhu
*   **Year:** 2023
*   **Journal:** University of Oxford Tech Report
*   **One-Paragraph Summary:** Surveys explainability methods used in clinical risk prediction models. Discusses post-hoc local and global XAI techniques (SHAP, LIME, Integrated Gradients) and maps them to clinical safety, bias detection, and physician trust.
*   **Main Methodology:** Systematic scoping review and taxonomy formulation.
*   **Main Contribution:** Comprehensive mapping of XAI methods to clinical risk prediction paradigms.
*   **Strengths:** Strong clinical grounding; connects mathematical metrics to clinical workflows.
*   **Weaknesses:** Lacks empirical benchmarks or quantitative comparisons of explanation drift.
*   **How AETHEL Relates to It:** AETHEL directly implements the recommendation of this survey by creating quantitative metrics (FAS, TkO) to validate explanation stability.
*   **Manuscript Citation Section:** Introduction / Methods / Discussion

### Explainable AI for malnutrition risk prediction from m-Health and clinical data

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/Explainable AI for Malnutrition Risk Prediction from m-Health and Clinical Data.pdf`
*   **Authors:** Flavio Di Martino, Franca Delmastro, Cristina Dolciotti
*   **Year:** 2023
*   **Journal:** Smart Health
*   **One-Paragraph Summary:** Presents a machine learning pipeline for malnutrition risk prediction using clinical and mobile health data. It integrates SHAP value explanations to help clinicians understand the decision logic of XGBoost models.
*   **Main Methodology:** XGBoost, Random Forest, SHAP explainability.
*   **Main Contribution:** ML pipeline for malnutrition prediction with post-hoc explainability.
*   **Strengths:** Integrates real-time m-Health telemetry data with static EHR records.
*   **Weaknesses:** Small cohort size; does not evaluate how explanation importances change under external cohort transfer.
*   **How AETHEL Relates to It:** AETHEL expands on this approach by analyzing explainability drift (FAS) specifically during demographic transfer.
*   **Manuscript Citation Section:** Methods / Discussion

### Explainable artificial intelligence in breast cancer detection and risk prediction: A systematic scoping review

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/Explainable artificial intelligence in breast cancer detection and risk prediction A systematic scoping review.pdf`
*   **Authors:** Amirehsan Ghasemi, Soheil Hashtarkhani, David L. Schwartz, Arash Shaban-Nejad
*   **Year:** 2024
*   **Journal:** Cancer Innovation
*   **One-Paragraph Summary:** Conducts a systematic scoping review of XAI applications in breast cancer prediction and detection. Highlights that while SHAP and LIME are popular, there is a lack of rigorous evaluation regarding their consistency and clinical validation.
*   **Main Methodology:** PRISMA systematic scoping review.
*   **Main Contribution:** A comprehensive taxonomy and gap analysis of XAI in oncology.
*   **Strengths:** Identifies the lack of consistency and validation in clinical XAI as a critical field limitation.
*   **Weaknesses:** Does not propose a new methodology or quantitative metrics to solve the identified gaps.
*   **How AETHEL Relates to It:** AETHEL directly solves the key gap identified by Ghasemi et al. by introducing the Feature Agreement Score (FAS) to measure explanation consistency.
*   **Manuscript Citation Section:** Introduction / Discussion

### Improvement of a prediction model for heart failure survival through explainable artificial intelligence

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/Improvement of a prediction model for heart failure survival through explainable artificial intelligence.pdf`
*   **Authors:** Pedro A. Moreno-Sánchez
*   **Year:** 2023
*   **Journal:** Scientific Reports
*   **One-Paragraph Summary:** Applies SHAP value attributions to a clinical prediction model for heart failure survival. The study demonstrates that analyzing explanation curves can help developers refine model inputs and eliminate redundant features.
*   **Main Methodology:** Random Forest, XGBoost, SHAP pruning.
*   **Main Contribution:** SHAP-based feature selection method for heart failure risk prediction.
*   **Strengths:** Provides empirical evidence that XAI can actively improve model training.
*   **Weaknesses:** Relies entirely on single-cohort internal validation; calibration drift is not examined.
*   **How AETHEL Relates to It:** AETHEL extends this by showing that post-hoc calibration does not degrade discrimination but improves calibration reliability under shift.
*   **Manuscript Citation Section:** Results / Discussion

### Risk prediction and interpretation for fall events using explainable AI and large language models

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/Risk Prediction and Interpretation for Fall Events.pdf`
*   **Authors:** Jake Luo, Masoud Khani, Jazzmyne Adams, et al.
*   **Year:** 2023
*   **Journal:** Journal of Biomedical Informatics
*   **One-Paragraph Summary:** Develops a risk prediction framework for fall events in clinics. It combines tree-based ML classifiers with SHAP for numerical explainability and maps these outputs to Large Language Models (LLMs) to generate clinical text reports.
*   **Main Methodology:** LightGBM, SHAP, GPT-based LLM summary generation.
*   **Main Contribution:** LLM-integrated clinical explanation reports for tree-based risk predictors.
*   **Strengths:** Combines feature attributions with generative text to improve clinician communication.
*   **Weaknesses:** LLM outputs are subjective and prone to hallucination; no clinical validation of text reports was conducted.
*   **How AETHEL Relates to It:** AETHEL focuses on rigorous quantitative auditing (FAS, ECE, DCA) rather than qualitative LLM texts to guarantee clinical safety.
*   **Manuscript Citation Section:** Introduction / Discussion

### SHAP-based explainable AI model for early detection of cardiovascular disease

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/SHAP_Based_Explainable_AI_Model_for_Earl.pdf`
*   **Authors:** Md. Abu Rayhan Imran, Tawhid Hasan, Nahida Akter, et al.
*   **Year:** 2025
*   **Journal:** Informatics in Medicine Unlocked
*   **One-Paragraph Summary:** Constructs an early detection system for cardiovascular disease using random forests and XGBoost. It applies SHAP value attributions to identify the relative importance of age, blood pressure, and cholesterol.
*   **Main Methodology:** Ensemble learning, SHAP values, risk classification.
*   **Main Contribution:** Cardiovascular risk prediction pipeline with local and global SHAP explanations.
*   **Strengths:** Demonstrates high classification metrics (ROC-AUC) on validation splits.
*   **Weaknesses:** Does not address calibration drift or clinical net benefit (DCA) of the predictions.
*   **How AETHEL Relates to It:** AETHEL acts as a direct critique of this paper, demonstrating that high validation AUC does not guarantee safety under external transportability shift.
*   **Manuscript Citation Section:** Introduction / Discussion

### Towards clinical prediction with transparency: An Explainable AI approach to survival modelling in residential aged care

*   **PDF Path:** `docs/literature/03_Clinical_Risk_Prediction/TOWARDS CLINICAL PREDICTION WITH TRANSPARENCY.pdf`
*   **Authors:** Teo Susnjak, Elise Griffin
*   **Year:** 2023
*   **Journal:** Scientific Reports
*   **One-Paragraph Summary:** Develops interpretable survival models for residential aged care using XGBoost and SHAP. It demonstrates that Tree SHAP can explain non-linear risks and survival times in geriatric clinical cohorts.
*   **Main Methodology:** XGBoost, Tree SHAP, survival time regression.
*   **Main Contribution:** Explainable survival modeling framework for elderly cohorts.
*   **Strengths:** Large cohort size (11,944 residents) and clinically rich features.
*   **Weaknesses:** Fails to audit model robustness under input sensor noise or missingness.
*   **How AETHEL Relates to It:** AETHEL incorporates robust noise sweeps to address the lack of safety boundaries in frameworks like Susnjak's.
*   **Manuscript Citation Section:** Methods / Discussion


---

## 04 Explainable AI

### Assessing the Communication Gap Between AI Models and Healthcare Professionals: Explainability, Utility and Trust in AI-driven Clinical Decision-making

*   **PDF Path:** `docs/literature/04_Explainable_AI/ASSESSING THE COMMUNICATION GAP BETWEEN AI MODELS.pdf`
*   **Authors:** Oskar Wysocki, Jessica K. Davies, Markel Vigo, et al.
*   **Year:** 2023
*   **Journal:** arXiv preprint arXiv:2308.11111
*   **One-Paragraph Summary:** Evaluates explainable machine learning models in clinical workflows. The study reveals that standard explanations can trigger automation bias and over-reliance, failing to expose model limitations to clinicians.
*   **Main Methodology:** Human-in-the-loop qualitative clinical evaluation.
*   **Main Contribution:** An empirical audit of XAI's impact on automation bias in clinical decision support.
*   **Strengths:** Highlights a critical gap between mathematical explanation metrics and actual clinical trust.
*   **Weaknesses:** Small clinician sample size; qualitative results are hard to scale.
*   **How AETHEL Relates to It:** AETHEL provides the Feature Agreement Score (FAS) to mathematically flag explanation instability, mitigating automation bias.
*   **Manuscript Citation Section:** Introduction / Discussion

### Evaluation of Popular XAI Applied to Clinical Prediction Models: Can They be Trusted?

*   **PDF Path:** `docs/literature/04_Explainable_AI/Evaluation of Popular XAI Applied to Clinical.pdf`
*   **Authors:** Aida Brankovic, David Cook, Jessica Rahman, et al.
*   **Year:** 2023
*   **Journal:** CSIRO Australian e-Health Research Centre Tech Report
*   **One-Paragraph Summary:** Evaluates SHAP and LIME on clinical prediction models, analyzing concordance at patient and cohort levels. Exposes significant inconsistency in feature importances, raising concerns about clinical trust.
*   **Main Methodology:** Local explanation comparison, rank correlation, consistency indexing.
*   **Main Contribution:** Benchmarking XAI methods on clinical datasets to demonstrate inconsistency.
*   **Strengths:** Evaluates stability of explanations across different initializations.
*   **Weaknesses:** Does not provide a unified framework to correct or calibrate these inconsistencies.
*   **How AETHEL Relates to It:** AETHEL directly adopts the Jaccard Top-k overlap (TkO) metric to quantify and audit these exact XAI inconsistencies.
*   **Manuscript Citation Section:** Methods / Results / Discussion

### “Why Should I Trust You?”: Explaining the Predictions of Any Classifier

*   **PDF Path:** `docs/literature/04_Explainable_AI/Explaining the Predictions of Any Classifier.pdf`
*   **Authors:** Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
*   **Year:** 2016
*   **Journal:** ACM SIGKDD
*   **One-Paragraph Summary:** Presents LIME (Local Interpretable Model-agnostic Explanations), a framework that explains predictions by fitting interpretable surrogate models locally around individual instances.
*   **Main Methodology:** Local surrogate modeling, perturbation analysis.
*   **Main Contribution:** The LIME explainability algorithm, popularizing local model-agnostic explanations.
*   **Strengths:** Applies to any classifier; highly intuitive local explanations.
*   **Weaknesses:** Explanations are highly unstable and sensitive to the local perturbation kernel.
*   **How AETHEL Relates to It:** AETHEL compares LIME and SHAP attributions under demographic shift to show that LIME is less stable than SHAP.
*   **Manuscript Citation Section:** Methods / Discussion

### It is not “accuracy vs. explainability” – we need both for trustworthy AI systems

*   **PDF Path:** `docs/literature/04_Explainable_AI/It is not “accuracy vs. explainability” – we need both for.pdf`
*   **Authors:** Dragutin Petkovic
*   **Year:** 2023
*   **Journal:** San Francisco State University Tech Report
*   **One-Paragraph Summary:** Argues against the common trade-off between accuracy and explainability, demonstrating that both are necessary to establish clinical and regulatory trust in AI systems.
*   **Main Methodology:** Conceptual analysis and regulatory literature review.
*   **Main Contribution:** Framing accuracy and explainability as complementary pillars of trustworthy AI.
*   **Strengths:** Strong policy and ethical arguments; aligned with FDA draft guidelines.
*   **Weaknesses:** Lacks empirical software implementations or quantitative code tests.
*   **How AETHEL Relates to It:** AETHEL proves this thesis empirically by demonstrating that post-hoc calibration preserves model discrimination (AUC) while recovering calibration and tracking FAS.
*   **Manuscript Citation Section:** Introduction / Discussion

### Towards A Rigorous Science of Interpretable Machine Learning

*   **PDF Path:** `docs/literature/04_Explainable_AI/Towards A Rigorous Science of Interpretable Machine Learning.pdf`
*   **Authors:** Finale Doshi-Velez, Been Kim
*   **Year:** 2017
*   **Journal:** arXiv preprint arXiv:1702.08608
*   **One-Paragraph Summary:** Establishes a taxonomy for evaluating interpretability, categorizing methods into application-grounded (human experiments), human-grounded (simple tasks), and functionally-grounded (proxy metrics).
*   **Main Methodology:** Evaluation taxonomy and scientific philosophy.
*   **Main Contribution:** A structured taxonomy to make the science of interpretability rigorous.
*   **Strengths:** Standardizes evaluation levels for explainable machine learning.
*   **Weaknesses:** Does not propose specific quantitative algorithms for post-hoc validation.
*   **How AETHEL Relates to It:** AETHEL operates in the functionally-grounded tier, using FAS and TkO as proxies for clinical explanation stability.
*   **Manuscript Citation Section:** Methods / Discussion

### A Review on Explainable Artificial Intelligence (XAI) in Healthcare

*   **PDF Path:** `docs/literature/04_Explainable_AI/A Review on Explainable Artificial Intelligence.pdf`
*   **Authors:** IEEE PRISMA survey authors
*   **Year:** 2023
*   **Journal:** IEEE Transactions on Artificial Intelligence
*   **One-Paragraph Summary:** Conducts a systematic review following PRISMA guidelines to analyze clinical XAI methods. Explains the why, how, and when of XAI, focusing on deriving trustworthy AI for clinical decision support.
*   **Main Methodology:** PRISMA systematic review.
*   **Main Contribution:** Structured mapping of healthcare XAI trends, limitations, and future directions.
*   **Strengths:** Comprehensive survey covering a decade of research (2012-2022).
*   **Weaknesses:** High-level overview; does not present original algorithms or data experiments.
*   **How AETHEL Relates to It:** AETHEL implements several post-hoc XAI approaches evaluated in this review, proving their real-world clinical utility.
*   **Manuscript Citation Section:** Introduction / Discussion

### Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI

*   **PDF Path:** `docs/literature/04_Explainable_AI/Explainable Artificial Intelligence (XAI) Concepts, Taxonomies, Responsible AI.pdf`
*   **Authors:** Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez, Javier Del Ser, et al.
*   **Year:** 2020
*   **Journal:** Information Fusion
*   **One-Paragraph Summary:** Presents a comprehensive taxonomy of Explainable AI. It defines the differences between transparent models (inherently interpretable) and post-hoc explainability, mapping them to responsible AI guidelines.
*   **Main Methodology:** XAI taxonomy and responsible AI review.
*   **Main Contribution:** Unified XAI definitions and a roadmap for responsible AI deployment.
*   **Strengths:** Highly cited, extensive, and detailed review of all machine learning domains.
*   **Weaknesses:** General review; lacks specific adaptations for clinical survival or biostatistical metrics.
*   **How AETHEL Relates to It:** AETHEL uses this taxonomy to classify post-hoc explanation techniques applied to ensemble classifiers.
*   **Manuscript Citation Section:** Methods / Discussion

### A unified approach to interpreting model predictions

*   **PDF Path:** `docs/literature/04_Explainable_AI/NIPS-2017-a-unified-approach-to-interpreting-model-predictions-Paper.pdf`
*   **Authors:** Scott M. Lundberg, Su-In Lee
*   **Year:** 2017
*   **Journal:** Advances in Neural Information Processing Systems
*   **One-Paragraph Summary:** Presents SHAP (SHapley Additive exPlanations), proving that Shapley values from game theory satisfy three essential properties: local accuracy, missingness, and consistency, defining a class of additive importances.
*   **Main Methodology:** Game theory, Shapley values, additive feature attribution.
*   **Main Contribution:** Formulation of SHAP and Tree SHAP for tree ensembles.
*   **Strengths:** Mathematically rigorous and unified; provides exact feature attributions.
*   **Weaknesses:** Computationally expensive; model-agnostic versions rely on perturbation approximations.
*   **How AETHEL Relates to It:** AETHEL implements SHAP as its core attribution method and audits SHAP rank stability using FAS.
*   **Manuscript Citation Section:** Methods / Results

### Peeking inside the black-box: A survey on Explainable Artificial Intelligence (XAI)

*   **PDF Path:** `docs/literature/04_Explainable_AI/Peeking_Inside_the_Black-Box_A_Survey_on_Explainable_Artificial_Intelligence_XAI.pdf`
*   **Authors:** Amina Adadi, Mohammed Berrada
*   **Year:** 2018
*   **Journal:** IEEE Access
*   **One-Paragraph Summary:** A widely cited survey outlining the motivations for explainable AI (explain to justify, explain to control, explain to improve, explain to discover) and comparing global vs. local explainability paradigms.
*   **Main Methodology:** Qualitative literature review and taxonomy.
*   **Main Contribution:** The four motivations of XAI and a survey of early post-hoc methods.
*   **Strengths:** Clear organization of motivations, highlighting why stakeholders require explanations.
*   **Weaknesses:** Lacks discussion of newer transformer models and has limited clinical applicability.
*   **How AETHEL Relates to It:** AETHEL addresses the 'explain to control' and 'explain to improve' motivations by implementing attribution auditing under domain shift.
*   **Manuscript Citation Section:** Introduction / Discussion


---

## 05 Calibration

### Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers

*   **PDF Path:** `docs/literature/05_Calibration/Obtaining calibrated probability estimates.pdf`
*   **Authors:** Bianca Zadrozny, Charles Elkan
*   **Year:** 2001
*   **Journal:** International Conference on Machine Learning (ICML)
*   **One-Paragraph Summary:** Investigates post-hoc calibration methods (Platt scaling, isotonic regression) for decision trees and naive Bayes. Shows that calibration is critical for cost-sensitive clinical decision-making.
*   **Main Methodology:** Platt scaling, isotonic regression, binning, Brier score.
*   **Main Contribution:** Popularized isotonic regression and Platt scaling for non-neural network classifiers.
*   **Strengths:** Identifies why specific classifiers produce uncalibrated scores and details simple repairs.
*   **Weaknesses:** Evaluations are restricted to IID datasets; does not evaluate calibration under covariate shift.
*   **How AETHEL Relates to It:** AETHEL directly implements Zadrozny's calibration methods (Platt scaling and Isotonic regression) in its calibration module.
*   **Manuscript Citation Section:** Methods / Results

### Predicting Good Probabilities With Supervised Learning

*   **PDF Path:** `docs/literature/05_Calibration/Predicting Good Probabilities With Supervised Learning.pdf`
*   **Authors:** Alexandru Niculescu-Mizil, Rich Caruana
*   **Year:** 2005
*   **Journal:** International Conference on Machine Learning (ICML)
*   **One-Paragraph Summary:** Analyses calibration across different supervised learning algorithms. Shows that maximum margin methods (SVMs, boosted trees) push predictions away from margins, creating a characteristic sigmoid distortion.
*   **Main Methodology:** Expected Calibration Error, Platt scaling, Isotonic regression, reliability curves.
*   **Main Contribution:** Discovered the characteristic calibration distortions of ensemble and margin classifiers.
*   **Strengths:** Comprehensive empirical study; explains the mathematical reasons behind uncalibrated ensembles.
*   **Weaknesses:** Does not address calibration drift in non-IID transfer scenarios.
*   **How AETHEL Relates to It:** AETHEL builds on these findings, applying Platt scaling to XGBoost models under transfer shift to correct margin distortions.
*   **Manuscript Citation Section:** Methods / Results


---

## 06 External Validation

### External validation of a Cox prognostic model: principles and methods

*   **PDF Path:** `docs/literature/06_External_Validation/External validation of a Cox prognostic model.pdf`
*   **Authors:** Patrick Royston, Douglas G. Altman
*   **Year:** 2013
*   **Journal:** BMC Medical Research Methodology
*   **One-Paragraph Summary:** Establishes statistical methods for externally validating Cox prognostic models. Details how to validate discrimination and calibration when baseline hazards are unspecified, proposing baseline survival approximations.
*   **Main Methodology:** Cox validation, baseline hazard approximation, prognostic index analysis.
*   **Main Contribution:** Methodology for validating Cox model calibration in independent datasets.
*   **Strengths:** Highly practical biostatistical guide addressing a major survival model validation issue.
*   **Weaknesses:** Restricted to traditional Cox proportional hazard equations; does not cover machine learning models.
*   **How AETHEL Relates to It:** AETHEL implements Harrell's C-index from scratch to evaluate survival baseline models on external target cohorts.
*   **Manuscript Citation Section:** Methods / Discussion

### Sample size considerations for the external validation of a multivariable prognostic model: a resampling study

*   **PDF Path:** `docs/literature/06_External_Validation/Sample size considerations for the.pdf`
*   **Authors:** Gary S. Collins, Emmanuel O. Ogundimu, Douglas G. Altman
*   **Year:** 2016
*   **Journal:** Statistics in Medicine
*   **One-Paragraph Summary:** Uses large clinical datasets and resampling to evaluate the sample size required for external validation. Recommends a minimum of 100 events (ideally 200+) to estimate performance metrics reliably.
*   **Main Methodology:** Bootstrap resampling, simulation study, sample size estimation.
*   **Main Contribution:** Empirical rules of thumb (minimum 100 events) for external validation study sizes.
*   **Strengths:** Provides concrete guidelines for clinical study design.
*   **Weaknesses:** Recommendations are based on traditional models and may not generalize to high-dimensional machine learning architectures.
*   **How AETHEL Relates to It:** AETHEL verifies that its external target cohorts (Framingham/NHANES) contain enough event counts (events > 200) to meet these validation requirements.
*   **Manuscript Citation Section:** Methods / Limitations

### Minimum sample size for external validation of a clinical prediction model with a binary outcome

*   **PDF Path:** `docs/literature/06_External_Validation/Statistics in Medicine - 2021 - Riley - Minimum sample size for external validation of a clinical prediction model with a.pdf`
*   **Authors:** Richard D. Riley, Thomas P. A. Debray, Gary S. Collins, et al.
*   **Year:** 2021
*   **Journal:** Statistics in Medicine
*   **One-Paragraph Summary:** Proposes a mathematical framework to calculate the minimum sample size required for validating a binary clinical model. Focuses on estimating calibration slope, calibration large, and AUC with sufficient precision.
*   **Main Methodology:** Mathematical formulas for sample size calculation based on target precision.
*   **Main Contribution:** Closed-form formulas for validation sample sizes, replacing empirical rules of thumb.
*   **Strengths:** Rigorous statistical formulation, moving away from arbitrary event counts.
*   **Weaknesses:** Requires prior estimates of target population outcome rates, which are often unknown.
*   **How AETHEL Relates to It:** AETHEL uses these formulas to audit and report target sample size adequacy in its reports.
*   **Manuscript Citation Section:** Methods / Results


---

## 07 Decision Curve Analysis

### Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance

*   **PDF Path:** `docs/literature/07_Decision_Curve_Analysis/Evaluation of performance measures in predictive artificial.pdf`
*   **Authors:** Ben Van Calster, Gary S. Collins, Andrew J. Vickers, et al. (STRATOS initiative)
*   **Year:** 2024
*   **Journal:** The Lancet Digital Health
*   **One-Paragraph Summary:** Provides guidance on performance measures for medical decision support. Discusses 32 measures across five domains, concluding that clinical utility (Decision Curve Analysis) is crucial for clinical translation.
*   **Main Methodology:** STRATOS consensus review, evaluation taxonomy.
*   **Main Contribution:** A unified guideline for validating predictive AI, prioritizing decision-analytical measures over ROC-AUC.
*   **Strengths:** Authored by leading clinical evaluators; provides a clear distinction between statistical and decision utility.
*   **Weaknesses:** Lacks details on how to present these curves to clinicians at the bedside.
*   **How AETHEL Relates to It:** AETHEL implements STRATOS recommendations by prioritizing DCA Net Benefit and translating it into consequence waffle charts.
*   **Manuscript Citation Section:** Methods / Results / Discussion


---

## 08 Trustworthy AI

### A roadmap to fair and trustworthy prediction model validation in healthcare

*   **PDF Path:** `docs/literature/08_Trustworthy_AI/A roadmap to fair and trustworthy prediction model validation in healthcare.pdf`
*   **Authors:** Yilin Ning, Victor Volovici, Marcus Eng Hock Ong, Benjamin Alan Goldstein, Nan Liu
*   **Year:** 2024
*   **Journal:** Duke-NUS CQM Tech Report
*   **One-Paragraph Summary:** Proposes a roadmap for clinical validation focusing on demographic fairness and trustworthiness. Highlights why validation should check performance across subgroups to avoid demographic harms.
*   **Main Methodology:** Validation framework and roadmap guidelines.
*   **Main Contribution:** Guidelines for auditing fairness and transportability in clinical predictive modeling.
*   **Strengths:** Emphasizes equity and safety under domain transfer.
*   **Weaknesses:** Lacks an open-source software implementation or code repository.
*   **How AETHEL Relates to It:** AETHEL represents the software platform implementing this validation roadmap.
*   **Manuscript Citation Section:** Introduction / Discussion

### FUTURE-AI: International consensus guideline for trustworthy and deployable artificial intelligence in healthcare

*   **PDF Path:** `docs/literature/08_Trustworthy_AI/FUTURE-AI International consensus guideline for trustworthy and deployable artificial.pdf`
*   **Authors:** Karim Lekadir, et al.
*   **Year:** 2022
*   **Journal:** Nature Medicine
*   **One-Paragraph Summary:** Presents international consensus guidelines for trustworthy clinical AI. It defines the FUTURE-AI framework, based on six pillars: Fairness, Universality, Traceability, Usability, Robustness, and Explainability.
*   **Main Methodology:** Consensus guideline development and Delphi survey.
*   **Main Contribution:** The FUTURE-AI guidelines for trustworthy medical AI.
*   **Strengths:** Broad international consensus; covers ethics, usability, and technical validation.
*   **Weaknesses:** High-level guidelines with few concrete validation tools or mathematical metrics.
*   **How AETHEL Relates to It:** AETHEL implements and automates the FUTURE-AI pillars, providing quantitative metrics for Fairness, Robustness, and Explainability.
*   **Manuscript Citation Section:** Introduction / Methods / Discussion


---

## 09 Fairness

### A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle

*   **PDF Path:** `docs/literature/09_Fairness/A Framework for Understanding Sources of Harm throughout.pdf`
*   **Authors:** Harini Suresh, John Guttag
*   **Year:** 2021
*   **Journal:** Proceedings of EAAMO '21
*   **One-Paragraph Summary:** Establishes a qualitative taxonomy of machine learning harms, defining historical, representation, measurement, aggregation, learning, evaluation, and deployment biases.
*   **Main Methodology:** Societal and algorithmic bias framework.
*   **Main Contribution:** Taxonomy of seven potential sources of downstream harm in the ML lifecycle.
*   **Strengths:** Bridges technical bias with real-world social harms; highly cited.
*   **Weaknesses:** Lacks quantitative formulas or automated auditing code.
*   **How AETHEL Relates to It:** AETHEL automates the evaluation of Suresh's 'evaluation and deployment harms' by stress-testing models on target cohorts.
*   **Manuscript Citation Section:** Introduction / Discussion

### Enabling Fairness in Healthcare Through Machine Learning

*   **PDF Path:** `docs/literature/09_Fairness/Enabling Fairness in Healthcare Through Machine Learning.pdf`
*   **Authors:** Thomas Grote, Stefan Berens
*   **Year:** 2022
*   **Journal:** Ethics and Information Technology
*   **One-Paragraph Summary:** Analyses algorithmic fairness in healthcare. It argues that ML models should be viewed as collaborative tools rather than autonomous decision-makers, and outlines how diversity in training data affects clinical equity.
*   **Main Methodology:** Conceptual ethics and clinical ML analysis.
*   **Main Contribution:** A collaborative human-AI framework for mitigating healthcare disparities.
*   **Strengths:** Deep ethical analysis of clinical decision-making and data bias.
*   **Weaknesses:** Does not propose automated fairness metrics or statistical audits.
*   **How AETHEL Relates to It:** AETHEL implements fairness audits across demographic subgroups (age, gender) to verify equity as advocated by Grote & Berens.
*   **Manuscript Citation Section:** Introduction / Discussion


---

## 10 Reporting Guidelines

### RoB 2: a revised tool for assessing risk of bias in randomised trials

*   **PDF Path:** `docs/literature/10_Reporting_Guidelines/a revised tool for assessing risk of bias.pdf`
*   **Authors:** Jonathan A. C. Sterne, et al.
*   **Year:** 2019
*   **Journal:** BMJ
*   **One-Paragraph Summary:** Details RoB 2, the updated Cochrane risk of bias tool for clinical trials. It focuses on bias in randomization, deviations, missing data, outcome measurement, and selective reporting.
*   **Main Methodology:** Consensus reporting checklist.
*   **Main Contribution:** The RoB 2 risk of bias tool for clinical trials.
*   **Strengths:** Systematic, structured, and widely accepted standard for trial auditing.
*   **Weaknesses:** Manual checklist; not designed for machine learning prediction models.
*   **How AETHEL Relates to It:** AETHEL translates these manual clinical bias audits into automated quantitative audits for predictive ML models.
*   **Manuscript Citation Section:** Methods / Discussion

### Protocol for development of a reporting guideline (TRIPOD-AI) and risk of bias tool (PROBAST-AI) for diagnostic and prognostic prediction model studies based on artificial intelligence

*   **PDF Path:** `docs/literature/10_Reporting_Guidelines/Protocol for development of a reporting.pdf`
*   **Authors:** Gary S. Collins, Paula Dhiman, Constanza L. Andaur Navarro, et al.
*   **Year:** 2021
*   **Journal:** BMJ Open
*   **One-Paragraph Summary:** Presents the protocol and development roadmap for TRIPOD-AI and PROBAST-AI guidelines. It details how the original statements are being expanded to address machine learning and artificial intelligence challenges.
*   **Main Methodology:** Delphi consensus survey protocol.
*   **Main Contribution:** Roadmap for establishing standard reporting and bias assessment guidelines for clinical AI.
*   **Strengths:** Addresses reporting deficiencies in the clinical AI literature.
*   **Weaknesses:** Protocol only; does not provide the final checklist or evaluation code.
*   **How AETHEL Relates to It:** AETHEL ensures compliance with these guidelines, using nested cross-validation and documenting hyperparameter choices.
*   **Manuscript Citation Section:** Methods / Discussion

### Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI

*   **PDF Path:** `docs/literature/10_Reporting_Guidelines/Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence DECIDE-AI.pdf`
*   **Authors:** DECIDE-AI Collaborators
*   **Year:** 2022
*   **Journal:** Nature Medicine
*   **One-Paragraph Summary:** Establishes reporting guidelines for the early-stage clinical evaluation of AI decision support systems (DECIDE-AI). It details 10 core domains and 27 items covering usability, safety, and clinical integration.
*   **Main Methodology:** Consensus reporting guideline, Delphi consensus study.
*   **Main Contribution:** The DECIDE-AI reporting statement for clinical decision support systems.
*   **Strengths:** Focuses on human-AI interaction, safety, and clinical workflows.
*   **Weaknesses:** Qualitative reporting guideline; does not define mathematical software audits.
*   **How AETHEL Relates to It:** AETHEL provides the technical evidence required to satisfy DECIDE-AI reporting requirements (usability and calibration).
*   **Manuscript Citation Section:** Methods / Discussion

### TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods

*   **PDF Path:** `docs/literature/10_Reporting_Guidelines/TRIPOD+AI statement updated guidance for reporting.pdf`
*   **Authors:** Gary S. Collins, Karel G. M. Moons, Paula Dhiman, et al.
*   **Year:** 2024
*   **Journal:** BMJ
*   **One-Paragraph Summary:** The official TRIPOD+AI reporting statement. It updates the TRIPOD statement to cover clinical prediction models that use machine learning, including data preprocessing, validation, and explainability.
*   **Main Methodology:** Consensus reporting guideline and checklist.
*   **Main Contribution:** The TRIPOD+AI reporting checklist for clinical machine learning models.
*   **Strengths:** Unified checklist covering regression and ML; widely accepted by journals.
*   **Weaknesses:** Checking compliance is still manual; no automated validation suite is provided.
*   **How AETHEL Relates to It:** AETHEL serves as an automated compliance tool, generating the technical verification data needed for TRIPOD+AI reporting.
*   **Manuscript Citation Section:** Methods / Discussion / Limitations

### Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement

*   **PDF Path:** `docs/literature/10_Reporting_Guidelines/TRIPOD.pdf`
*   **Authors:** Gary S. Collins, Johannes B. Reitsma, Douglas G. Altman, Karel G. M. Moons
*   **Year:** 2015
*   **Journal:** BMJ / Annals of Internal Medicine
*   **One-Paragraph Summary:** The seminal TRIPOD statement. It defines a 22-item checklist for reporting prediction model studies, aiming to improve the transparency and completeness of clinical model publications.
*   **Main Methodology:** Consensus reporting guideline and checklist.
*   **Main Contribution:** The TRIPOD checklist, the gold standard for clinical prediction model reporting.
*   **Strengths:** Has significantly improved the quality of clinical predictive model literature.
*   **Weaknesses:** Designed for traditional statistical models; lacks guidelines for complex ML and XAI.
*   **How AETHEL Relates to It:** AETHEL complies with TRIPOD guidelines and extends them to support machine learning and explainability.
*   **Manuscript Citation Section:** Methods / Discussion


---

## 11 Background Reviews

### Missing data in randomized controlled trials testing palliative interventions pose a significant risk of bias and loss of power: a systematic review and meta-analyses

*   **PDF Path:** `docs/literature/11_Background_Reviews/Missing-data-in-randomized-controlled-trials-testi.pdf`
*   **Authors:** Jamilla A. Hussain, Ian R. White, Dean Langan, et al.
*   **Year:** 2016
*   **Journal:** Journal of Clinical Epidemiology
*   **One-Paragraph Summary:** Investigates the impact of missing data in clinical trials. The study demonstrates that missing data is common, poorly reported, and poses a significant risk of bias and loss of statistical power.
*   **Main Methodology:** Systematic review and random-effects meta-analyses.
*   **Main Contribution:** Quantifies the risk of bias and power loss caused by missing data in clinical trials.
*   **Strengths:** Rigorous meta-analysis proving the clinical impact of missing data.
*   **Weaknesses:** Focuses on clinical trials; does not cover missing data in machine learning models.
*   **How AETHEL Relates to It:** AETHEL addresses this risk by simulating missing data in its robustness sweeps, mapping failure thresholds under clinical data corruption.
*   **Manuscript Citation Section:** Introduction / Discussion

### Pursuing minimally disruptive medicine: disruption from illness and health care-related demands is correlated with patient capacity

*   **PDF Path:** `docs/literature/11_Background_Reviews/Pursuing-minimally-disruptive-medicine--disruption.pdf`
*   **Authors:** Kasey R. Boehmer, Nathan D. Shippee, Timothy J. Beebe, Victor M. Montori
*   **Year:** 2016
*   **Journal:** Journal of Clinical Epidemiology
*   **One-Paragraph Summary:** Explores the concept of minimally disruptive medicine, investigating how illness and treatment burdens disrupt patients' lives. Shows that disruption is correlated with patients' capacity to manage demands.
*   **Main Methodology:** Patient survey study and multivariate regression analysis.
*   **Main Contribution:** Evidence supporting minimally disruptive medicine to reduce patient treatment burden.
*   **Strengths:** Focuses on patient capacity and treatment burden, improving clinical care design.
*   **Weaknesses:** Does not address computer science or clinical decision support systems directly.
*   **How AETHEL Relates to It:** AETHEL supports this philosophy by validating clinical decision utility, preventing unnecessary treatment harms and patient burden.
*   **Manuscript Citation Section:** Introduction / Discussion

### The potential for artificial intelligence in healthcare

*   **PDF Path:** `docs/literature/11_Background_Reviews/The potential for artificial intelligence in healthcare.pdf`
*   **Authors:** Thomas Davenport, Ravi Kalakota
*   **Year:** 2019
*   **Journal:** Future Healthcare Journal
*   **One-Paragraph Summary:** Reviews the potential applications of artificial intelligence in healthcare. It discusses diagnosis, treatment recommendation, patient engagement, and administrative tasks, outlining key ethical and implementation challenges.
*   **Main Methodology:** Qualitative review and outlook.
*   **Main Contribution:** A broad survey of AI technologies and their implementation in healthcare.
*   **Strengths:** Accessible overview of clinical AI technologies.
*   **Weaknesses:** Lacks technical depth, mathematical details, and validation metrics.
*   **How AETHEL Relates to It:** AETHEL addresses the deployment challenges identified by Davenport by providing a robust validation framework.
*   **Manuscript Citation Section:** Introduction


---

