"""
repeated_runs.py
================
Manages repeated experiments over multiple random seeds to quantify
performance variance, prediction stability, and explanation stability.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import shap
from src.utils.logging_setup import get_logger
from src.evaluation.evaluator import calculate_metrics

logger = get_logger(__name__)

def run_repeated_experiments(
    model_class: type,
    model_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    seeds: list[int],
) -> dict:
    """
    Runs a model across multiple seeds, logging metrics, feature importances,
    predictions, probabilities, and SHAP values for every run.
    """
    logger.info("RepeatedRuns: running %d seeds...", len(seeds))
    
    runs_data = []
    all_metrics = []
    all_importances = []
    all_shap_values = []
    all_probs = []
    all_preds = []

    for idx, seed in enumerate(seeds):
        logger.debug("  Seed [%d/%d]: %d", idx + 1, len(seeds), seed)
        
        # Instantiate model with seed
        kwargs = model_kwargs.copy()
        if "random_state" in kwargs or hasattr(model_class, "random_state"):
            kwargs["random_state"] = seed
        
        # Train model
        model = model_class(**kwargs)
        model.fit(X_train[features], y_train)
        
        # Predict
        probs = model.predict_proba(X_val[features])[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        all_probs.append(probs)
        all_preds.append(preds)
        
        # Metrics
        metrics = calculate_metrics(y_val, probs)
        all_metrics.append(metrics)
        
        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(len(features))
        all_importances.append(importances)
        
        # SHAP Values (using a background subsample of 100 to speed up)
        try:
            bg_size = min(100, len(X_train))
            X_bg = X_train[features].iloc[:bg_size]
            
            # Limit X_val size for SHAP speed
            val_explain_size = min(500, len(X_val))
            # Determine indices deterministically using seed for consistency
            X_val_sub = X_val[features].sample(n=val_explain_size, random_state=seed)
            
            # Defensive check for explainer
            model_name = model_class.__name__.lower()
            if "logistic" in model_name:
                explainer = shap.LinearExplainer(model, X_bg)
                shap_vals = explainer.shap_values(X_val_sub)
            elif "forest" in model_name or "tree" in model_name or "xgb" in model_name or "lgb" in model_name:
                explainer = shap.TreeExplainer(model, X_bg)
                # Avoid additivity error on Windows
                shap_vals = explainer.shap_values(X_val_sub, check_additivity=False)
            else:
                explainer = shap.Explainer(model, X_bg)
                shap_vals = explainer(X_val_sub).values
            
            # Handle multi-class output shapes from SHAP
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]
                
            all_shap_values.append(shap_vals)
        except Exception as e:
            logger.warning("    SHAP calculation failed for seed %d: %s", seed, e)
            # Fallback to zero SHAP values
            all_shap_values.append(np.zeros((val_explain_size if 'val_explain_size' in locals() else len(X_val), len(features))))

    # Compute summary statistics for metrics
    metric_keys = list(all_metrics[0].keys())
    metric_stats = {}
    
    for key in metric_keys:
        vals = np.array([m[key] for m in all_metrics if not np.isnan(m[key])])
        if len(vals) == 0:
            metric_stats[key] = {
                "mean": np.nan, "median": np.nan, "var": np.nan, "std": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan
            }
            continue
            
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        var = float(np.var(vals))
        std = float(np.std(vals))
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        
        metric_stats[key] = {
            "mean": mean,
            "median": median,
            "var": var,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return {
        "metric_stats": metric_stats,
        "raw_metrics": all_metrics,
        "importances": np.array(all_importances),
        "shap_values": np.array(all_shap_values),
        "predictions": np.array(all_preds),
        "probabilities": np.array(all_probs),
        "seeds": seeds,
    }
