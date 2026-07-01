# AETHEL: Pan-European Environmental Survival Framework

![Status](https://img.shields.io/badge/Status-Active_Research-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![R](https://img.shields.io/badge/R-4.2+-blue)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

## Abstract

AETHEL is a reproducible, cross-language biostatistical framework modelling the longitudinal impact of spatio-temporal environmental exposures (PM₂.₅, NO₂) on polygenic disease susceptibility. Scaled across a metadata registry of 100 major European healthcare hubs, it integrates high-dimensional synthetic clinical cohorts with localised meteorological telemetry to execute advanced survival analysis (Cox Proportional Hazards & Random Survival Forests).

---

## Repository Structure

```
AETHEL/
├── configs/                    # Centralised configuration (YAML)
│   └── default.yaml            # Single source of truth for all parameters
├── data/
│   ├── raw/                    # Generated raw data (registry, env, clinical)
│   ├── processed/              # Engineered analytical cohort
│   ├── synthetic/              # Placeholder for future synthetic datasets
│   ├── nhanes/                 # Placeholder for NHANES external validation
│   └── framingham/             # Framingham dataset (future)
├── src/
│   ├── preprocessing/          # Data generation and ingestion
│   ├── feature_engineering/    # Analytical feature construction
│   ├── models/                 # Statistical / ML model scripts (R)
│   ├── evaluation/             # Model evaluation metrics (future)
│   ├── explainability/         # SHAP, VIMP, interpretability (future)
│   ├── visualization/          # Streamlit dashboard
│   ├── calibration/            # Calibration analysis (future)
│   ├── robustness/             # Robustness / sensitivity experiments (future)
│   └── utils/                  # Shared infrastructure (paths, logging, config, seeds)
├── experiments/                # Experiment configuration variants
├── notebooks/                  # Jupyter / Quarto exploratory notebooks
├── reports/                    # Quarto technical report
├── tests/                      # Automated test suite (pytest)
├── docs/                       # Extended documentation
├── outputs/                    # All generated outputs (auto-created)
│   ├── models/                 # Serialised model artefacts
│   ├── figures/                # Plots and charts
│   ├── metrics/                # CSV metric summaries (vimp.csv, cox_coefficients.csv)
│   ├── predictions/            # Survival predictions
│   ├── shap/                   # SHAP value outputs
│   ├── calibration/            # Calibration outputs
│   └── logs/                   # Structured run logs
└── scripts/
    └── run_pipeline.py         # Single-command pipeline entry point
```

---

## Quickstart

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate aethel
```

### Option B — pip

```bash
pip install -r requirements.txt
```

### R Dependencies

```r
install.packages(c("survival", "randomForestSRC", "yaml"))
```

---

## Running the Pipeline

### Full Pipeline (single command)

```bash
python scripts/run_pipeline.py
```

### Python-only (skip R stages)

```bash
python scripts/run_pipeline.py --skip-r
```

### Custom Config (experiment variant)

```bash
python scripts/run_pipeline.py --config configs/my_experiment.yaml
```

### Launch the Clinical Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

---

## Manual Stage Execution

If you prefer to run stages individually:

```bash
# Stage 1 — Build city registry
python -m src.preprocessing.build_eu_registry

# Stage 2 — Generate environmental time-series
python -m src.preprocessing.generate_env_data

# Stage 3 — Generate clinical cohort (R)
Rscript src/preprocessing/generate_clinical_cohort.R

# Stage 4 — Engineer analytical features
python -m src.feature_engineering.preprocess_features

# Stage 5 — Fit survival models (R)
Rscript src/models/survival_model.R
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests verify registry shape (100 cities), environmental data shape (6,000 rows), column integrity, config loading, and path resolution. R is not required to run the test suite.

---

## Configuration

All pipeline parameters live in `configs/default.yaml`.  Key sections:

| Section | Description |
|---|---|
| `seeds.python` | NumPy/random seed (default: 42) |
| `seeds.r` | R `set.seed()` value (default: 123) |
| `study_parameters` | Cohort size, observation window, city count |
| `pipeline_controls` | Toggle individual stages on/off |
| `model_params` | RSF tree count, importance flag |
| `features.survival_covariates` | Authoritative covariate list for all models |
| `output_paths` | Where all outputs are written |

---

## Outputs

After a full pipeline run, the following files are automatically generated:

| Path | Contents |
|---|---|
| `outputs/metrics/cox_coefficients.csv` | Cox PH model coefficients and p-values |
| `outputs/metrics/vimp.csv` | RSF Variable Importance scores (loaded by dashboard) |
| `outputs/logs/aethel.log` | Full structured run log |

---

## System Architecture

```
configs/default.yaml
        │
        ├─► src/utils/ (paths, logging, seed, config_loader, constants)
        │           │
        │   ┌───────┴───────────────────────────────────────┐
        │   │ Python Pipeline                               │
        │   │ preprocessing/ → feature_engineering/         │
        │   └───────────────────────────────────────────────┘
        │               │
        │               ▼ data/processed/analytical_cohort.csv
        │   ┌───────────────────────────────────────────────┐
        │   │ R Pipeline                                    │
        │   │ preprocessing/ (clinical) → models/ (survival)│
        │   └───────────────────────────────────────────────┘
        │               │
        │               ▼ outputs/metrics/{vimp.csv, cox_coefficients.csv}
        │   ┌───────────────────────────────────────────────┐
        │   │ visualization/dashboard.py (Streamlit)        │
        │   └───────────────────────────────────────────────┘
```

**Python** handles spatio-temporal data engineering, geographic registry management, and feature alignment. **R** executes rigorous biostatistical inference (Cox PH + RSF). The **Streamlit dashboard** provides interactive geospatial risk mapping, loading real model outputs from `outputs/metrics/`.

---

## Statistical Methodology

The AETHEL pipeline models the respiratory hazard function λ(t) by integrating clinical demographics, polygenic baseline risk, and longitudinal environmental telemetry:

$$h(t, X) = h_0(t) \exp\left(\sum_{i=1}^{p} \beta_i X_i\right)$$

Where $h_0(t)$ is the baseline hazard and $X_i$ represents clinical, genetic, and environmental covariates. Random Survival Forests capture non-linear interactions and generate explainable variable importance (VIMP).

---

## Future Extensions

The modular `src/` architecture is designed to accommodate the following additions without disrupting existing code:

| Extension | Target Module |
|---|---|
| NHANES external validation | `src/preprocessing/` + `data/nhanes/` |
| Framingham cohort | `src/preprocessing/` + `data/framingham/` |
| Calibration (Platt, isotonic) | `src/calibration/` |
| SHAP values | `src/explainability/` |
| Fairness analysis | `src/robustness/` |
| Robustness / bootstrap CI | `src/robustness/` |
| Additional models | `src/models/` |
| Automated reporting | `reports/` + Quarto |

---

## Visuals

<img width="1906" height="1079" alt="AETHEL Dashboard Map" src="https://github.com/user-attachments/assets/eea46092-b365-4a11-bd75-708f416cdcb6" />
<img width="1901" height="883" alt="AETHEL Risk Distribution" src="https://github.com/user-attachments/assets/ba42334d-9b8d-4784-8c50-0ed3fbc9248e" />

---

## Citation

```bibtex
@software{aethel2026,
  title   = {AETHEL: Pan-European Environmental Survival Framework},
  year    = {2026},
  url     = {https://github.com/Tani-2005/AETHEL-Risk-Stratification-Clinical-Cohort-Analysis}
}
```
