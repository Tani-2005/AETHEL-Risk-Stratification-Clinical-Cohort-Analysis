"""
validation_runner.py
===================
Orchestrates training and cross-cohort testing across Synthetic, Framingham,
and NHANES. Automatically applies a clinical surrogate outcome to NHANES.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from src.datasets.registry import DatasetRegistry
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Features present in all three cohorts (intersection feature space)
BIOCHEMICAL_FEATURES = ["h_sys_bp", "h_dia_bp", "h_total_cholesterol"]

# Features present in Synthetic & Framingham
COMMON_INTERSECTION = ["h_age", "h_bmi", "h_is_smoker"]

def get_surrogate_outcome(df: pd.DataFrame) -> pd.Series:
    """Computes a clinical surrogate outcome: SBP >= 140 OR DBP >= 90 OR TC >= 240."""
    sbp = df.get("h_sys_bp", pd.Series(np.nan, index=df.index))
    dbp = df.get("h_dia_bp", pd.Series(np.nan, index=df.index))
    tc = df.get("h_total_cholesterol", pd.Series(np.nan, index=df.index))
    
    # Deterministic risk condition
    condition = (sbp >= 140) | (dbp >= 90) | (tc >= 240)
    return condition.astype(float)

def load_and_align_cohorts() -> dict[str, pd.DataFrame]:
    """Loads all cohorts, aligns features, and assigns surrogate outcomes where needed."""
    logger.info("ValidationRunner: loading and aligning cohorts...")
    registry = DatasetRegistry()
    
    # Load raw/harmonized
    synth_ds = registry.load("synthetic")
    fram_ds = registry.load("framingham")
    nhanes_ds = registry.load("nhanes")
    
    synth_df = synth_ds.df_harmonized.copy()
    fram_df = fram_ds.df_harmonized.copy()
    nhanes_df = nhanes_ds.df_harmonized.copy()
    
    # Assign surrogate outcome to NHANES
    nhanes_df["h_outcome_binary"] = get_surrogate_outcome(nhanes_df)
    
    # Also pre-calculate surrogate outcomes on Synthetic & Framingham for biochemical experiments
    synth_df["h_surrogate_outcome"] = get_surrogate_outcome(synth_df)
    fram_df["h_surrogate_outcome"] = get_surrogate_outcome(fram_df)
    
    return {
        "synthetic": synth_df,
        "framingham": fram_df,
        "nhanes": nhanes_df,
    }

def preprocess_cross_cohort(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    outcome_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, SimpleImputer, RobustScaler]:
    """Preprocesses train and test cohorts, imputing missingness and scaling features."""
    X_train_raw = df_train[features]
    y_train = df_train[outcome_col]
    X_test_raw = df_test[features]
    y_test = df_test[outcome_col]
    
    # Impute
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_train_raw)
    X_te_imp = imp.transform(X_test_raw)
    
    # Scale
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_te_scaled = scaler.transform(X_te_imp)
    
    X_tr = pd.DataFrame(X_tr_scaled, columns=features, index=df_train.index)
    X_te = pd.DataFrame(X_te_scaled, columns=features, index=df_test.index)
    
    return X_tr, y_train, X_te, y_test, imp, scaler
