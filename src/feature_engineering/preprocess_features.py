"""
preprocess_features.py
=======================
Stage 4 (Python) — Full Preprocessing & Feature Engineering Orchestrator.

Execution order (enforced — NO preprocessing sees test data before split):

  Load analytical_cohort.csv
        │
        ▼
  DataValidator.validate()          ← flags issues, logs report
        │
        ▼
  add_deterministic_features()      ← bmi_category, age_group, lifestyle_risk,
        │                              high_genomic_risk (safe pre-split)
        ▼
  CohortSplitter.split()            ← 70/15/15 stratified on event_occurred
        │
        ▼
  PollutionIndexComputer.fit(train) ← fit on TRAIN ONLY
  PollutionIndexComputer.transform  ← apply to all splits
        │
        ▼
  build_pipeline() → fit_pipeline() ← fit scaler on TRAIN ONLY
        │
        ▼
  FeatureSelector.analyse(train)    ← VIF, correlation, MI (train only)
        │
        ▼
  Save train/val/test + scaler + reports

Usage
-----
    python -m src.feature_engineering.preprocess_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.constants import Columns
from src.utils.logging_setup import configure_logging, get_logger
from src.utils.paths import DataDirs, DataPaths, OutputDirs, ensure_output_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def build_analytical_dataset() -> pd.DataFrame:
    """
    Stage 4a: Join clinical cohort with environmental telemetry.

    Unchanged from original — assigns patients to cities, computes
    mean PM2.5 and NO2 exposures over observation windows.
    """
    from src.utils.config_loader import load_config as _lc

    cfg = _lc()

    # Use isolated RNG for jitter to avoid contaminating global state
    rng = np.random.default_rng(cfg.seeds.python)

    required = [DataPaths.RAW_CLINICAL_COHORT, DataPaths.RAW_ENV_HISTORY, DataPaths.RAW_REGISTRY]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Ensure all earlier pipeline stages have been run."
            )

    clinical_df = pd.read_csv(DataPaths.RAW_CLINICAL_COHORT)
    env_df = pd.read_csv(DataPaths.RAW_ENV_HISTORY)
    registry = pd.read_csv(DataPaths.RAW_REGISTRY)

    logger.info("Loaded %d patients, %d env records, %d cities.",
                len(clinical_df), len(env_df), len(registry))

    # Set global seed for city assignment (reproducible)
    set_global_seed(cfg.seeds.python)
    clinical_df[Columns.CITY_ID] = np.random.choice(
        registry[Columns.CITY_ID], size=len(clinical_df)
    )

    pm25_exposures, no2_exposures, lats, lons, city_names = [], [], [], [], []

    for _, patient in clinical_df.iterrows():
        months = max(1, int(patient[Columns.MONTHS_OBSERVED]))
        city_id = patient[Columns.CITY_ID]

        city_meta = registry[registry[Columns.CITY_ID] == city_id].iloc[0]
        city_env = env_df[env_df[Columns.LOCATION] == city_id].head(months)

        pm25_exposures.append(float(city_env[Columns.PM25].mean()))
        no2_exposures.append(float(city_env[Columns.NO2].mean()))

        # Use isolated RNG for jitter — does NOT affect global seed state
        lats.append(float(city_meta[Columns.LAT]) + rng.normal(0, 0.02))
        lons.append(float(city_meta[Columns.LON]) + rng.normal(0, 0.02))
        city_names.append(str(city_meta[Columns.CITY]))

    clinical_df[Columns.AVG_PM25_EXPOSURE] = pm25_exposures
    clinical_df[Columns.AVG_NO2_EXPOSURE] = no2_exposures
    clinical_df[Columns.LAT] = lats
    clinical_df[Columns.LON] = lons
    clinical_df[Columns.CITY] = city_names

    DataDirs.PROCESSED.mkdir(parents=True, exist_ok=True)
    clinical_df.to_csv(DataPaths.ANALYTICAL_COHORT, index=False)

    logger.info(
        "Analytical dataset built (%d patients). Written to %s.",
        len(clinical_df), DataPaths.ANALYTICAL_COHORT,
    )
    return clinical_df


def run_full_preprocessing(df: pd.DataFrame | None = None) -> None:
    """
    Stage 4b: Full clinical ML preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame | None
        The analytical cohort. If None, reads from data/processed/analytical_cohort.csv.
    """
    from src.preprocessing.data_validator import DataValidator
    from src.preprocessing.cohort_splitter import CohortSplitter
    from src.feature_engineering.engineer_features import (
        add_deterministic_features,
        PollutionIndexComputer,
    )
    from src.feature_engineering.feature_selector import FeatureSelector
    from src.feature_engineering.sklearn_pipeline import (
        build_pipeline,
        fit_pipeline,
        transform_split,
    )

    cfg = load_config()
    ensure_output_dirs()

    # ---- Load ----
    if df is None:
        if not DataPaths.ANALYTICAL_COHORT.exists():
            raise FileNotFoundError(
                f"Analytical cohort not found at {DataPaths.ANALYTICAL_COHORT}. "
                "Run build_analytical_dataset() first."
            )
        df = pd.read_csv(DataPaths.ANALYTICAL_COHORT)
    logger.info("Loaded cohort: %d patients × %d columns.", len(df), len(df.columns))

    # ---- Validate (pre-split) ----
    if cfg.pipeline.run_validation:
        logger.info("--- Stage: Data Validation ---")
        validator = DataValidator(cfg)
        validator.validate(df)

    # ---- Deterministic feature engineering (pre-split) ----
    if cfg.pipeline.run_feature_engineering_derived:
        logger.info("--- Stage: Deterministic Feature Engineering ---")
        df = add_deterministic_features(df, cfg)

    # ---- Split (stratified on event_occurred) ----
    if cfg.pipeline.run_splitting:
        logger.info("--- Stage: Train/Val/Test Split ---")
        splitter = CohortSplitter(cfg)
        train_df, val_df, test_df = splitter.split(df)
    else:
        logger.warning("Splitting skipped. Loading existing splits from disk.")
        train_df = pd.read_csv(DataPaths.TRAIN)
        val_df = pd.read_csv(DataPaths.VAL)
        test_df = pd.read_csv(DataPaths.TEST)

    # ---- Pollution index (fit on TRAIN ONLY) ----
    if cfg.pipeline.run_preprocessing_pipeline and cfg.feature_eng.add_pollution_index:
        logger.info("--- Stage: Pollution Index (fit on train only) ---")
        pic = PollutionIndexComputer(cfg)
        train_df = pic.fit_transform(train_df)
        val_df = pic.transform(val_df)
        test_df = pic.transform(test_df)

        # Persist updated splits (with pollution_index added)
        train_df.to_csv(DataPaths.TRAIN, index=False)
        val_df.to_csv(DataPaths.VAL, index=False)
        test_df.to_csv(DataPaths.TEST, index=False)
        logger.info("Updated splits (with pollution_index) saved.")

    # ---- Sklearn pipeline (fit on TRAIN ONLY) ----
    if cfg.pipeline.run_preprocessing_pipeline:
        logger.info("--- Stage: Sklearn Preprocessing Pipeline ---")
        available_cols = list(train_df.columns)
        pipeline = build_pipeline(cfg, available_cols)
        fitted_pipeline = fit_pipeline(pipeline, train_df)

        # Transform all splits (for future Python-based models)
        transform_split(fitted_pipeline, train_df, "train")
        transform_split(fitted_pipeline, val_df, "val")
        transform_split(fitted_pipeline, test_df, "test")
        logger.info("Scaled splits ready for Python models (scaler at outputs/models/).")

    # ---- Feature selection (train only) ----
    if cfg.pipeline.run_feature_selection:
        logger.info("--- Stage: Feature Selection Analysis ---")
        selector = FeatureSelector(cfg)
        selector.analyse(train_df)

    logger.info(
        "Full preprocessing complete. Splits: train=%d, val=%d, test=%d patients.",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info("All reports saved to outputs/reports/")


if __name__ == "__main__":
    configure_logging()
    df = build_analytical_dataset()
    run_full_preprocessing(df)
