"""
feature_ablation.py
===================
Implements hierarchical (group-level) and individual feature ablation.
Measures performance degradation, calibration shift, prediction shift, and
importance shift after feature removal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.evaluation.evaluator import calculate_metrics
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Feature grouping mapping
FEATURE_GROUP_MAP = {
    # Harmonized features
    "h_age": "Demographics",
    "h_sex_male": "Demographics",
    "h_bmi": "Clinical",
    "h_sys_bp": "Clinical",
    "h_dia_bp": "Clinical",
    "h_total_cholesterol": "Clinical",
    "h_ldl": "Clinical",
    "h_triglycerides": "Clinical",
    "h_glucose": "Clinical",
    "h_is_smoker": "Lifestyle",
    
    # Raw/Specific feature names
    "age": "Demographics",
    "sex_male": "Demographics",
    "bmi": "Clinical",
    "sys_bp": "Clinical",
    "dia_bp": "Clinical",
    "total_cholesterol": "Clinical",
    "glucose": "Clinical",
    "is_smoker": "Lifestyle",
    "townsend_index": "Environmental",
    "avg_pm25_exposure": "Environmental",
    "avg_no2_exposure": "Environmental",
    "pollution_index": "Environmental",
    "genomic_risk_score": "Genetics",
    "high_genomic_risk": "Genetics",
    "lifestyle_risk": "Lifestyle",
}

def get_feature_group(feature_name: str) -> str:
    """Returns the group name for a given feature."""
    return FEATURE_GROUP_MAP.get(feature_name.lower(), "Clinical")

def run_hierarchical_ablation(
    model_class: type,
    model_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    baseline_metrics: dict[str, float],
    baseline_probs: np.ndarray,
) -> dict:
    """
    Performs hierarchical ablation by removing one feature group at a time,
    retraining, and measuring degradation.
    """
    logger.info("FeatureAblation: starting hierarchical ablation...")
    
    # Group the active features
    grouped_features = {}
    for f in features:
        grp = get_feature_group(f)
        grouped_features.setdefault(grp, []).append(f)
        
    results = {}
    
    for grp, grp_features in grouped_features.items():
        logger.debug("  Ablating group: %s (%s)", grp, grp_features)
        
        # Remaining features
        rem_features = [f for f in features if f not in grp_features]
        if not rem_features:
            logger.warning("    Skipping group %s: no features left if removed.", grp)
            continue
            
        # Retrain model
        model = model_class(**model_kwargs)
        model.fit(X_train[rem_features], y_train)
        
        # Predict
        probs = model.predict_proba(X_val[rem_features])[:, 1]
        metrics = calculate_metrics(y_val, probs)
        
        # Performance Drop
        auc_drop = baseline_metrics.get("roc_auc", 0.5) - metrics.get("roc_auc", 0.5)
        brier_shift = metrics.get("brier", 0.0) - baseline_metrics.get("brier", 0.0)
        
        # Prediction Shift (MSE of probs)
        pred_shift = float(np.mean((baseline_probs - probs) ** 2))
        
        results[grp] = {
            "ablated_features": grp_features,
            "remaining_features": rem_features,
            "metrics": metrics,
            "auc_drop": float(auc_drop),
            "brier_shift": float(brier_shift),
            "prediction_shift": pred_shift,
        }
        
    return results

def run_individual_ablation(
    model_class: type,
    model_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    baseline_metrics: dict[str, float],
    baseline_probs: np.ndarray,
) -> dict:
    """
    Performs individual ablation by removing one feature at a time,
    retraining, and measuring degradation.
    """
    logger.info("FeatureAblation: starting individual ablation...")
    
    results = {}
    
    for f in features:
        logger.debug("  Ablating feature: %s", f)
        
        # Remaining features
        rem_features = [feat for feat in features if feat != f]
        if not rem_features:
            continue
            
        # Retrain model
        model = model_class(**model_kwargs)
        model.fit(X_train[rem_features], y_train)
        
        # Predict
        probs = model.predict_proba(X_val[rem_features])[:, 1]
        metrics = calculate_metrics(y_val, probs)
        
        # Performance Drop
        auc_drop = baseline_metrics.get("roc_auc", 0.5) - metrics.get("roc_auc", 0.5)
        brier_shift = metrics.get("brier", 0.0) - baseline_metrics.get("brier", 0.0)
        
        # Prediction Shift (MSE of probs)
        pred_shift = float(np.mean((baseline_probs - probs) ** 2))
        
        results[f] = {
            "metrics": metrics,
            "auc_drop": float(auc_drop),
            "brier_shift": float(brier_shift),
            "prediction_shift": pred_shift,
        }
        
    return results
