# AETHEL Architecture Reference

## Data Flow

```
data/raw/                               data/raw/
eu_registry.csv  ──────────────────►  regional_environmental_history.csv
(build_eu_registry.py)                 (generate_env_data.py)
         │                                        │
         │                                        │
         └──────────────────────────────────────► │
                                                   ▼
                               data/raw/synthetic_clinical_cohort.csv
                               (generate_clinical_cohort.R — R stage)
                                                   │
                                                   ▼
                               data/processed/analytical_cohort.csv
                               (preprocess_features.py)
                                                   │
                                                   ▼
                               outputs/metrics/cox_coefficients.csv
                               outputs/metrics/vimp.csv
                               (survival_model.R — R stage)
                                                   │
                                                   ▼
                               src/visualization/dashboard.py
                               (Streamlit — reads analytical_cohort + vimp.csv)
```

---

## Module Responsibilities

### `src/utils/`

The foundational infrastructure layer. Every other module depends on this — **never import from domain modules in utils**.

| File | Responsibility |
|---|---|
| `constants.py` | All column names, feature lists, categorical labels |
| `paths.py` | All file paths resolved via `pathlib.Path` from project root |
| `config_loader.py` | Loads `configs/default.yaml` into typed `AETHELConfig` dataclass |
| `logging_setup.py` | Configures console + file logger; `get_logger(__name__)` pattern |
| `seed.py` | `set_global_seed(seed)` — sets numpy and random; R seed in config |

### `src/preprocessing/`

Data generation and ingestion — **no modelling logic**.

| File | Responsibility |
|---|---|
| `build_eu_registry.py` | Generates 100-city EU metadata registry (lat/lon, baseline pollution) |
| `generate_env_data.py` | Generates 60-month monthly PM2.5/NO2 time-series per city |
| `generate_clinical_cohort.R` | Generates synthetic clinical cohort (demographics, survival outcomes) |

### `src/feature_engineering/`

Constructs analytical features from raw data — **no raw data generation**.

| File | Responsibility |
|---|---|
| `preprocess_features.py` | Joins clinical + environmental data; computes patient-level exposures |

### `src/models/`

Statistical and ML model scripts — **reads only from `data/processed/`**.

| File | Responsibility |
|---|---|
| `survival_model.R` | Cox PH + RSF survival models; saves coefficients + VIMP to `outputs/metrics/` |

### `src/visualization/`

User-facing interface — **reads only from `data/processed/` and `outputs/`**.

| File | Responsibility |
|---|---|
| `dashboard.py` | Streamlit interactive dashboard with geospatial map and VIMP chart |

### `src/evaluation/` *(placeholder)*

Future: concordance index, Brier score, time-dependent AUC.

### `src/explainability/` *(placeholder)*

Future: SHAP value computation, partial dependence plots, ICE curves.

### `src/calibration/` *(placeholder)*

Future: Platt scaling, isotonic regression, reliability diagrams, ECE.

### `src/robustness/` *(placeholder)*

Future: bootstrap CIs, sensitivity analysis, fairness subgroup analysis, trustworthiness evaluation.

---

## Configuration Architecture

`configs/default.yaml` is the single source of truth.

Python scripts read it via `src/utils/config_loader.load_config()`.
R scripts read it via `yaml::read_yaml("configs/default.yaml")`.

**Rule**: Never hardcode a parameter that is in the config file.

### Config Sections

```yaml
project:          # Metadata (name, version)
pipeline_controls: # Toggle stages on/off
seeds:            # Python (42) and R (123) seeds
study_parameters: # n_subjects, observation_years, total_cities
data_paths:       # Input file paths (relative to project root)
output_paths:     # Output directory paths
model_params:     # RSF ntree, importance flag
features:         # survival_covariates list (single source of truth)
default_dashboard_regions: # UI defaults
```

---

## Reproducibility Contract

| Layer | Mechanism |
|---|---|
| Python RNG | `src/utils/seed.set_global_seed(cfg.seeds.python)` at stage start |
| R RNG | `set.seed(config$seeds$r)` at script start |
| Per-city R seed | `np.random.seed(abs(hash(city_id)) % 10^8)` — deterministic, not global |
| Config archiving | Archive `configs/default.yaml` alongside any publication |
| Requirements | Pinned versions in `requirements.txt` and `environment.yml` |

---

## Adding a New Dataset (e.g. NHANES)

1. Add raw data to `data/nhanes/`
2. Create `src/preprocessing/load_nhanes.py` following the `build_eu_registry.py` pattern
3. Add NHANES paths to `src/utils/paths.py` under `DataPaths`
4. Add a new pipeline control toggle in `configs/default.yaml`
5. Wire the new stage into `scripts/run_pipeline.py`
6. Add smoke tests in `tests/`

---

## Adding a New Model

1. Create `src/models/my_model.R` (or `.py`)
2. Read covariates from `config$features$survival_covariates`
3. Write outputs to `outputs/metrics/` and `outputs/models/`
4. Add a toggle in `configs/default.yaml → pipeline_controls`
5. Wire into `scripts/run_pipeline.py`

---

## Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Python files | `snake_case.py` | `preprocess_features.py` |
| R files | `snake_case.R` | `survival_model.R` |
| Config keys | `snake_case` | `rsf_ntree` |
| Column names | `snake_case` | `avg_pm25_exposure` |
| Class names | `PascalCase` | `AETHELConfig` |
| Constants | `UPPER_SNAKE` | `ANALYTICAL_COHORT` |
