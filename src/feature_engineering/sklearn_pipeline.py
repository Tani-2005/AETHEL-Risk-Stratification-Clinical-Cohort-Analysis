"""
sklearn_pipeline.py
===================
Stage: Sklearn Preprocessing Pipeline

Builds a modular sklearn ColumnTransformer + Pipeline for the AETHEL cohort.
The pipeline is fitted ONLY on training data and serialised to disk.

Pipeline architecture
---------------------
┌─────────────────────────────────────────────┐
│ ColumnTransformer                            │
│  ├── continuous → RobustScaler              │
│  ├── binary     → passthrough               │
│  └── categorical → OneHotEncoder(drop=first)│
└─────────────────────────────────────────────┘

Why RobustScaler (not StandardScaler)?
  The audit found 1 patient with BMI=14.76 (below clinical threshold of 15).
  RobustScaler uses median and IQR rather than mean/std, making it robust
  to such outliers.  This is the preferred scaler for clinical data.

Why binary features are NOT scaled?
  is_smoker and high_genomic_risk are binary (0/1) indicators.
  Scaling them to [-1, 1] or similar would break their interpretability
  and potentially harm models that rely on the 0/1 contrast.

Why OneHotEncoder with drop='first'?
  Avoids the dummy variable trap (perfect multicollinearity) when
  categorical features with k levels produce k-1 columns.

Usage
-----
    from src.feature_engineering.sklearn_pipeline import build_pipeline, fit_pipeline

    pipeline = build_pipeline(cfg)
    fitted_pipeline = fit_pipeline(pipeline, train_df)
    X_train_scaled = fitted_pipeline.transform(train_df)
"""

from __future__ import annotations

import pandas as pd
import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.utils.config_loader import AETHELConfig
from src.utils.constants import Columns, Features
from src.utils.logging_setup import get_logger
from src.utils.paths import OutputDirs, OutputPaths

logger = get_logger(__name__)

_SCALER_MAP = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


def build_pipeline(cfg: AETHELConfig, available_columns: list[str]) -> Pipeline:
    """
    Construct (but do NOT fit) the preprocessing pipeline.

    Parameters
    ----------
    cfg : AETHELConfig
        Config providing scaler_type choice.
    available_columns : list[str]
        Columns actually present in the dataset (varies when
        optional features are toggled off).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted ColumnTransformer pipeline.
    """
    scaler_cls = _SCALER_MAP.get(cfg.preprocessing.scaler_type, RobustScaler)
    logger.info("Using scaler: %s", scaler_cls.__name__)

    cont_cols = [c for c in Features.CONTINUOUS_FEATURES if c in available_columns]
    bin_cols = [c for c in Features.BINARY_FEATURES if c in available_columns]
    cat_cols = [c for c in Features.CATEGORICAL_FEATURES if c in available_columns]

    logger.info("Continuous features (%d): %s", len(cont_cols), cont_cols)
    logger.info("Binary features    (%d): %s", len(bin_cols), bin_cols)
    logger.info("Categorical features(%d): %s", len(cat_cols), cat_cols)

    transformers = []
    if cont_cols:
        transformers.append(("continuous", scaler_cls(), cont_cols))
    if bin_cols:
        transformers.append(("binary", "passthrough", bin_cols))
    if cat_cols:
        transformers.append((
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # exclude identifiers, geospatial, outcomes
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def fit_pipeline(
    pipeline: Pipeline, train_df: pd.DataFrame
) -> Pipeline:
    """
    Fit the pipeline on TRAINING DATA ONLY and serialise to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn pipeline from build_pipeline().
    train_df : pd.DataFrame
        Training fold — no val/test data should be included.

    Returns
    -------
    Pipeline
        The fitted pipeline.
    """
    logger.info("Fitting preprocessing pipeline on training data (%d rows)...", len(train_df))
    pipeline.fit(train_df)

    OutputDirs.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OutputPaths.SCALER_JOBLIB)
    logger.info("Fitted pipeline saved to %s", OutputPaths.SCALER_JOBLIB)

    return pipeline


def transform_split(
    pipeline: Pipeline,
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Apply the fitted pipeline to a data split and return a DataFrame.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    df : pd.DataFrame
        Any split (train, val, or test).
    split_name : str
        Label for logging (e.g. 'train', 'val', 'test').

    Returns
    -------
    pd.DataFrame
        Transformed features with column names reconstructed.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    X_transformed = pipeline.transform(df)

    # Reconstruct column names
    feature_names = preprocessor.get_feature_names_out()
    # Clean sklearn prefixes (e.g. "continuous__age" → "age")
    clean_names = [n.split("__")[-1] for n in feature_names]

    result = pd.DataFrame(X_transformed, columns=clean_names, index=df.index)
    logger.info(
        "Transformed %s split: %d rows × %d features",
        split_name, len(result), len(result.columns),
    )
    return result
