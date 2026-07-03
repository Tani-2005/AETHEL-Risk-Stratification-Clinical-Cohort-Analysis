"""
shift_detector.py
==================
Quantifies covariate shift, prior shift, concept shift, and feature-wise drift
between clinical cohorts using Population Stability Index (PSI) and Wasserstein distance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from src.robustness.distribution_shift import calculate_psi
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def quantify_domain_shift(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    features: list[str],
    outcome_col: str,
) -> dict:
    """
    Measures multiple dimensions of domain shift:
    1. Covariate Shift (PSI and Wasserstein of features)
    2. Prior Shift (difference in outcome base rates)
    3. Population Shift (mean feature PSI)
    """
    logger.info("ShiftDetector: quantifying domain shift dimensions...")
    
    # 1. Feature Drift / Covariate Shift
    feature_shifts = {}
    psi_values = []
    
    for f in features:
        if f in df_src.columns and f in df_tgt.columns:
            src_vals = df_src[f].dropna().values
            tgt_vals = df_tgt[f].dropna().values
            
            if len(src_vals) > 0 and len(tgt_vals) > 0:
                w_dist = wasserstein_distance(src_vals, tgt_vals)
                psi_val = calculate_psi(src_vals, tgt_vals)
                
                # Grade shift severity
                if psi_val < 0.10:
                    severity = "Negligible"
                elif psi_val < 0.25:
                    severity = "Moderate"
                else:
                    severity = "Significant"
                    
                feature_shifts[f] = {
                    "wasserstein_distance": float(w_dist),
                    "psi": float(psi_val),
                    "severity": severity,
                }
                psi_values.append(psi_val)
            else:
                feature_shifts[f] = {
                    "wasserstein_distance": np.nan,
                    "psi": np.nan,
                    "severity": "Unknown",
                }
                
    # Rank features by drift severity
    drift_ranking = sorted(
        [f for f in feature_shifts if not np.isnan(feature_shifts[f]["psi"])],
        key=lambda x: feature_shifts[x]["psi"],
        reverse=True
    )
    
    # 2. Prior Shift P(Y)
    y_src = df_src[outcome_col].dropna().values if outcome_col in df_src.columns else []
    y_tgt = df_tgt[outcome_col].dropna().values if outcome_col in df_tgt.columns else []
    
    prior_shift_val = np.nan
    if len(y_src) > 0 and len(y_tgt) > 0:
        prior_shift_val = float(np.abs(np.mean(y_src) - np.mean(y_tgt)))
        
    # 3. Population Shift (Average PSI)
    avg_psi = float(np.mean(psi_values)) if psi_values else np.nan
    
    return {
        "covariate_shift": feature_shifts,
        "prior_shift": prior_shift_val,
        "population_shift_avg_psi": avg_psi,
        "most_drifted_features": drift_ranking,
    }
