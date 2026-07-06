# AETHEL Architecture Reference

AETHEL is structured as a decoupled, reproducible framework for auditing clinical machine learning models. 

## Data & Process Flow

```
                                  [DATA SOURCE LAYER]
                         ┌─────────────────────────────────┐
                         │   Framingham Heart Study (FHS)  │
                         │   NHANES Biochemical Survey     │
                         │   AETHEL Synthetic Cohort       │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                             [HARMONIZATION & ALIGNMENT]
                               (src/datasets/registry.py)
                                          │
                                          ▼
                               [EXPERIMENT PIPELINES]
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
           [INTERNAL VALIDATION]                     [CROSS-COHORT TRANSFER]
       (scripts/run_evaluation.py)                (scripts/run_generalization.py)
       - LeakageFreeCV (5x10-fold)                - Framingham ──► NHANES (Surrogate)
       - Train/Val/Test splits                    - Synthetic ──► Framingham
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                              [AUDITING & METRICS LAYER]
        ┌─────────────────────────────────┴─────────────────────────────────┐
        │  1. CALIBRATION (src/calibration/): Platt Scaling, Isotonic       │
        │  2. EXPLAINABILITY (src/explainability/): FAS, Jaccard TkO, SHAP  │
        │  3. ROBUSTNESS (src/robustness/): Noise sweeps, Missingness       │
        │  4. CLINICAL UTILITY (src/evaluation/): DCA, Bedside Waffles      │
        └─────────────────────────────────┬─────────────────────────────────┘
                                          │
                                          ▼
                             [REPORTING & USER INTERFACE]
       ┌──────────────────────────────────┴──────────────────────────────────┐
       │  - AETHEL Studio (research-workbench/): Next.js interactive UI      │
       │  - Streamlit Dashboard (src/visualization/dashboard.py): Geospatial │
       │  - LaTeX primary tables & high-res vector figures (outputs/)        │
       └─────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### `src/utils/`
The foundational infrastructure layer. Every other module depends on this — **never import from domain modules in utils**.

| File | Responsibility |
|---|---|
| `constants.py` | Column names, feature lists, categories, and labels. |
| `paths.py` | Resolves absolute file paths relative to project root. |
| `config_loader.py` | Loads and typings `configs/default.yaml` parameters. |
| `logging_setup.py` | Configures unified console and file logger. |

---

### `src/datasets/`
Data loading, harmonization, and registration of the clinical cohorts.

| File / Directory | Responsibility |
|---|---|
| `registry.py` | Dataset registry matching dataset strings to loaders. |
| `base_loader.py` | Abstract base loader class enforcing common APIs. |
| `synthetic_loader.py`| Loads and harmonizes AETHEL Synthetic Cohort. |
| `framingham_loader.py`| Loads and harmonizes Framingham Heart Study data. |
| `nhanes_loader.py` | Loads and harmonizes NHANES Biochemical Survey. |

---

### `src/evaluation/`
Core statistical, clinical utility, and model evaluation engines.

| File | Responsibility |
|---|---|
| `evaluator.py` | Computes traditional stats (AUC, Brier), bootstrapping CIs, McNemar tests, Decision Curve Analysis (DCA), and error categories. |
| `cohort_comparison.py`| Compares statistics across different study sub-cohorts. |

---

### `src/calibration/`
Post-hoc recalibration modeling for recovering probability confidence.

| File | Responsibility |
|---|---|
| `recalibration.py` | Platt Scaling (Sigmoid fit on logits) and Isotonic Regression models. |

---

### `src/explainability/`
Quantifying feature attributions and explainability consensus drift.

| File | Responsibility |
|---|---|
| `shap_analysis.py` | Tree and Linear SHAP value explainers. |
| `consensus_analysis.py`| Computes Feature Agreement Score (FAS) and Jaccard Top-k Overlap (TkO) rank correlations. |
| `clinical_interpretation.py`| Generates clinical guidance maps from model attributions. |
| `permutation_analysis.py`| Calculates model-agnostic permutation importances. |

---

### `src/robustness/`
Stress-testing classifiers under missing data and environmental/sensor noise.

| File | Responsibility |
|---|---|
| `noise_analysis.py` | Sweeps Gaussian continuous noise and binary bit-flips. |
| `missing_data_analysis.py`| Audits model degradation under simulated missingness (MCAR). |
| `repeated_runs.py` | Runs repeated executions to compute variance/confidence bands. |

---

### `src/domain_shift/`
Measuring multivariate covariate shifts.

| File | Responsibility |
|---|---|
| `shift_detector.py` | Calculates Wasserstein distance and Kullback-Leibler divergence. |

---

### `src/generalization/`
Quantifying performance and attribution gaps across cohort transfer.

| File | Responsibility |
|---|---|
| `generalization_gap.py`| Measures ROC-AUC drop and ECE increase. |
| `explanation_drift.py`| Tracks SHAP rank and agreement drift under transfer. |

---

### `src/trustworthiness/`
Consolidating multi-dimensional audits into clinical profiles.

| File | Responsibility |
|---|---|
| `trustworthiness_evaluator.py`| Ranks metrics into a unified AETHEL clinical trust grade. |
| `clinical_consistency.py`| Checks predictions against established physiological guidelines. |
| `publication_tables.py`| Generates publication-ready Markdown/LaTeX tables. |

---

### `src/external_validation/`
Orchestrating cross-cohort experiments and failure modes.

| File | Responsibility |
|---|---|
| `validation_runner.py`| Aligns features and outcomes across transfer tasks. |
| `failure_analysis.py` | Groups target failures into clinical clusters. |
| `uncertainty_transfer.py`| Measures prediction confidence stability. |

---

## Configuration Architecture

`configs/default.yaml` acts as the single source of truth for pipeline parameters.
*   Python scripts access config parameters via `src.utils.config_loader.load_config()`.
*   R scripts read the config parameters via `yaml::read_yaml("configs/default.yaml")`.

**Rule**: Parameters must never be hardcoded in any script if they exist in the YAML configuration.

---

## Reproducibility Contract

*   **Global Seeds:** Set via `configs/default.yaml` and set globally at execution start for Python (`numpy`, `random`) and R.
*   **Archiving:** Each execution of `scripts/run_pipeline.py` or the evaluation orchestrators outputs frozen config snapshots alongside reports.
*   **Environments:** Dependency versions are pinned within `requirements.txt` and `environment.yml`.
