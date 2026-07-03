"""
missing_data_analysis.py
========================
Simulates missing data conditions (5%, 10%, 20%, 30% missing values) on test
inputs under MCAR (Missing Completely at Random), evaluates model degradation,
and determines failure thresholds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from src.evaluation.evaluator import calculate_metrics
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def run_missing_data_robustness(
    model: Any,
    X_val_raw: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    imputer: Any,
    scaler: Any,
    missing_rates: list[float] = [0.0, 0.05, 0.1, 0.2, 0.3],
    random_state: int = 42,
) -> dict:
    """
    Simulates MCAR missingness on validation set features, imputes using the
    provided imputer, scales, and evaluates degradation.
    """
    logger.info("MissingDataAnalysis: running missing data robustness sweep...")
    
    rng = np.random.default_rng(random_state)
    results = {}
    
    # 1. Baseline (0% missingness)
    X_base_imputed = pd.DataFrame(imputer.transform(X_val_raw[features]), columns=features)
    X_base_scaled = pd.DataFrame(scaler.transform(X_base_imputed), columns=features)
    probs_base = model.predict_proba(X_base_scaled)[:, 1]
    metrics_base = calculate_metrics(y_val, probs_base)
    
    # Compute baseline SHAP for explanation stability comparison
    try:
        bg_size = min(100, len(X_base_scaled))
        X_bg = X_base_scaled.iloc[:bg_size]
        model_name = model.__class__.__name__.lower()
        if "logistic" in model_name:
            shap_explainer = shap.LinearExplainer(model, X_bg)
            shap_base = shap_explainer.shap_values(X_base_scaled)
        elif "forest" in model_name or "tree" in model_name or "xgb" in model_name or "lgb" in model_name:
            shap_explainer = shap.TreeExplainer(model, X_bg)
            shap_base = shap_explainer.shap_values(X_base_scaled, check_additivity=False)
        else:
            shap_explainer = shap.Explainer(model, X_bg)
            shap_base = shap_explainer(X_base_scaled).values
            
        if isinstance(shap_base, list):
            shap_base = shap_base[1] if len(shap_base) > 1 else shap_base[0]
        elif isinstance(shap_base, np.ndarray) and len(shap_base.shape) == 3:
            shap_base = shap_base[:, :, 1]
    except Exception:
        shap_base = None

    results[0.0] = {
        "metrics": metrics_base,
        "explanation_stability": 1.0,
        "prediction_shift_mse": 0.0,
    }
    
    failure_threshold = None
    auc_drop_threshold = 0.03 # 3% drop is standard clinical failure threshold
    
    for rate in missing_rates:
        if rate == 0.0:
            continue
            
        # Introduce MCAR missingness
        X_miss = X_val_raw[features].copy()
        mask = rng.random(size=X_miss.shape) < rate
        X_miss[mask] = np.nan
        
        # Impute and Scale
        X_miss_imputed = pd.DataFrame(imputer.transform(X_miss), columns=features)
        X_miss_scaled = pd.DataFrame(scaler.transform(X_miss_imputed), columns=features)
        
        # Predict
        probs = model.predict_proba(X_miss_scaled)[:, 1]
        metrics = calculate_metrics(y_val, probs)
        
        # Performance degradation
        auc_drop = metrics_base.get("roc_auc", 0.5) - metrics.get("roc_auc", 0.5)
        
        # Explanation Stability
        exp_stability = 1.0
        if shap_base is not None:
            try:
                # Compute perturbed SHAP
                if "logistic" in model_name:
                    shap_curr = shap_explainer.shap_values(X_miss_scaled)
                elif "forest" in model_name or "tree" in model_name or "xgb" in model_name or "lgb" in model_name:
                    shap_curr = shap_explainer.shap_values(X_miss_scaled, check_additivity=False)
                else:
                    shap_curr = shap_explainer(X_miss_scaled).values
                    
                if isinstance(shap_curr, list):
                    shap_curr = shap_curr[1] if len(shap_curr) > 1 else shap_curr[0]
                elif isinstance(shap_curr, np.ndarray) and len(shap_curr.shape) == 3:
                    shap_curr = shap_curr[:, :, 1]
                
                # Pairwise spearman per patient
                rhos = []
                for idx in range(len(shap_base)):
                    r, _ = spearmanr(shap_base[idx], shap_curr[idx])
                    rhos.append(r if not np.isnan(r) else 1.0)
                exp_stability = float(np.mean(rhos))
            except Exception:
                exp_stability = np.nan
                
        results[rate] = {
            "metrics": metrics,
            "explanation_stability": exp_stability,
            "prediction_shift_mse": float(np.mean((probs_base - probs) ** 2)),
        }
        
        # Log failure threshold if AUC drops significantly
        if auc_drop > auc_drop_threshold and failure_threshold is None:
            failure_threshold = rate
            logger.warning("  Failure threshold reached at %.0f%% missing data (AUC drop = %.4f)", rate * 100, auc_drop)

    return {
        "sweep_results": results,
        "failure_threshold": failure_threshold,
    }
