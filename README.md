# AETHEL: Pan-European Environmental Survival Framework

![Status](https://img.shields.io/badge/Status-Active_Research-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![R](https://img.shields.io/badge/R-4.2+-blue)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## Abstract
AETHEL is a reproducible, cross-language biostatistical framework designed to model the longitudinal impact of spatio-temporal environmental factors ($PM_{2.5}$, $NO_2$) on polygenic disease susceptibility. Scaled across a metadata registry of 100 major European healthcare hubs, this engine integrates high-dimensional synthetic clinical cohorts with localized meteorological telemetry to execute advanced survival analysis (Cox Proportional Hazards & Random Survival Forests).

---

## System Architecture
This pipeline utilizes a decoupled architecture, leveraging Python for high-throughput data engineering and R for rigorous statistical inference.

* **`/python_engine`**: Handles spatio-temporal data generation, geographic registry management, and feature alignment. Calculates patient-specific cumulative environmental exposures.
* **`/r_analysis`**: Executes the biostatistical modeling. Fits Cox PH models, validates proportional hazards assumptions via Schoenfeld residuals, and trains Random Survival Forests for explainable variable importance (VIMP).
* **`dashboard.py`**: An interactive Streamlit UI for dynamic geospatial risk mapping and individualized patient telemetry profiling.

---

## Reproducing the Pipeline

**1. Install Dependencies**
```bash
pip install pandas numpy streamlit plotly
# In R console: install.packages(c("survival", "randomForestSRC"))
```
**2. Generate the Pan-European Data Vault**

```
python python_engine/build_eu_registry.py
python python_engine/generate_env_data.py
Rscript r_analysis/generate_clinical_cohort.R

```
**3. Engineer Analytical Features**

```
python python_engine/preprocess_features.py
```

**4. Execute Biostatistical Inference**

```
Rscript r_analysis/survival_model.R
```

**5. Launch the Clinical Dashboard**
```
streamlit run dashboard.py
```
---
# Statistical Methodology

The AETHEL pipeline models the respiratory hazard function $\lambda(t)$ by integrating clinical demographics, polygenic baseline risk, and longitudinal environmental telemetry. It advances beyond standard statistical inference by deploying Random Survival Forests to capture non-linear health interactions and generate high-fidelity, explainable patient risk profiles.

---
