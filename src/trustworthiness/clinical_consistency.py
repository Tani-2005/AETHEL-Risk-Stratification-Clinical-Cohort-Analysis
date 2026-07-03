"""
clinical_consistency.py
========================
Evaluates whether model predictions maintain clinically consistent relationships
with key risk factors (Age, Smoking, BP, BMI, Cholesterol) across cohorts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Expected direction of association with cardiovascular risk
# (positive correlation with risk probability)
CLINICAL_EXPECTATIONS = {
    "h_age": "positive",
    "h_sys_bp": "positive",
    "h_dia_bp": "positive",
    "h_total_cholesterol": "positive",
    "h_bmi": "positive",
    "h_is_smoker": "positive",
}

def evaluate_clinical_consistency(
    df: pd.DataFrame,
    probs: np.ndarray,
    features: list[str],
) -> dict:
    """
    Evaluates correlation of features with predicted probabilities,
    verifying if the direction matches established clinical literature.
    """
    logger.info("ClinicalConsistency: auditing predictor directions...")
    
    consistency_report = {}
    
    for f in features:
        if f in df.columns:
            vals = df[f].values
            # Spearman correlation between feature values and predicted probability
            corr, p_val = spearmanr(vals, probs)
            if np.isnan(corr):
                corr = 0.0
                
            expected = CLINICAL_EXPECTATIONS.get(f, "unknown")
            
            # Determine actual direction
            if corr > 0.05:
                actual = "positive"
            elif corr < -0.05:
                actual = "negative"
            else:
                actual = "neutral"
                
            # Verify consistency
            is_consistent = False
            if expected == "positive" and actual == "positive":
                is_consistent = True
            elif expected == "negative" and actual == "negative":
                is_consistent = True
            elif expected == "neutral":
                is_consistent = True
                
            consistency_report[f] = {
                "correlation": float(corr),
                "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                "expected_direction": expected,
                "actual_direction": actual,
                "is_consistent": is_consistent,
            }
            
    # Compute overall consistency rate
    total_audited = len([f for f in consistency_report if CLINICAL_EXPECTATIONS.get(f) is not None])
    consistent_count = len([f for f in consistency_report if consistency_report[f]["is_consistent"] and CLINICAL_EXPECTATIONS.get(f) is not None])
    
    consistency_rate = float(consistent_count / total_audited) if total_audited > 0 else 1.0
    
    logger.info(
        "Clinical Consistency: %d/%d features consistent (%.2f%%)",
        consistent_count,
        total_audited,
        consistency_rate * 100,
    )
    
    return {
        "feature_consistency": consistency_report,
        "consistency_rate": consistency_rate,
        "n_consistent": consistent_count,
        "n_audited": total_audited,
    }
