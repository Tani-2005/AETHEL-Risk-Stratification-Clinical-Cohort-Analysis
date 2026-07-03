"""
generalization_gap.py
======================
Measures internal performance vs external performance, performance drop,
calibration drop, and prediction distribution drift.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def measure_generalization_gap(
    metrics_src: dict,
    metrics_tgt: dict,
    y_prob_src: np.ndarray,
    y_prob_tgt: np.ndarray,
) -> dict:
    """
    Computes performance and calibration drops, and quantifies prediction drift.
    """
    logger.info("GeneralizationGap: measuring internal vs external drops...")
    
    # 1. Performance drop (AUC, PR-AUC, F1)
    auc_src = metrics_src.get("roc_auc", 0.0)
    auc_tgt = metrics_tgt.get("roc_auc", 0.0)
    auc_drop = auc_src - auc_tgt
    
    pr_src = metrics_src.get("pr_auc", 0.0)
    pr_tgt = metrics_tgt.get("pr_auc", 0.0)
    pr_drop = pr_src - pr_tgt
    
    f1_src = metrics_src.get("f1_score", 0.0)
    f1_tgt = metrics_tgt.get("f1_score", 0.0)
    f1_drop = f1_src - f1_tgt
    
    # 2. Calibration drop (ECE)
    ece_src = metrics_src.get("ece", 0.0)
    ece_tgt = metrics_tgt.get("ece", 0.0)
    ece_drop = ece_tgt - ece_src  # ECE increase is a calibration quality drop
    
    # 3. Prediction Drift / Distribution change
    # Compare predicted probability distributions between cohorts using Wasserstein & KS test
    y_prob_src = np.asarray(y_prob_src)
    y_prob_tgt = np.asarray(y_prob_tgt)
    
    w_dist = wasserstein_distance(y_prob_src, y_prob_tgt)
    ks_stat, ks_p = ks_2samp(y_prob_src, y_prob_tgt)
    
    results = {
        "internal_performance": {
            "roc_auc": float(auc_src),
            "pr_auc": float(pr_src),
            "f1_score": float(f1_src),
            "ece": float(ece_src),
        },
        "external_performance": {
            "roc_auc": float(auc_tgt),
            "pr_auc": float(pr_tgt),
            "f1_score": float(f1_tgt),
            "ece": float(ece_tgt),
        },
        "generalization_gap": {
            "roc_auc_drop": float(auc_drop),
            "pr_auc_drop": float(pr_drop),
            "f1_score_drop": float(f1_drop),
            "ece_increase": float(ece_drop),
        },
        "prediction_drift": {
            "wasserstein_distance": float(w_dist),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
        }
    }
    
    logger.info(
        "Generalization Gap: ROC-AUC drop = %.4f, ECE increase = %.4f",
        auc_drop,
        ece_drop,
    )
    return results
