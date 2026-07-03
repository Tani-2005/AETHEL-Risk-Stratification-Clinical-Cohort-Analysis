"""
uncertainty_transfer.py
=======================
Evaluates prediction confidence and entropy across different cohorts.
Identifies patients with high uncertainty under distribution shift.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Computes binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    return -probs * np.log2(probs) - (1.0 - probs) * np.log2(1.0 - probs)

def evaluate_uncertainty_transfer(
    y_prob_src: np.ndarray,
    y_prob_tgt: np.ndarray,
) -> dict:
    """
    Compares confidence and entropy distributions between source and target cohorts.
    Flags high uncertainty cases in the target cohort.
    """
    logger.info("UncertaintyTransfer: comparing uncertainty profiles across cohorts...")
    
    y_prob_src = np.asarray(y_prob_src)
    y_prob_tgt = np.asarray(y_prob_tgt)
    
    # 1. Source stats
    conf_src = 2.0 * np.abs(y_prob_src - 0.5)
    ent_src = compute_entropy(y_prob_src)
    
    # 2. Target stats
    conf_tgt = 2.0 * np.abs(y_prob_tgt - 0.5)
    ent_tgt = compute_entropy(y_prob_tgt)
    
    # Thresholds for high uncertainty
    # Entropy > 0.8 represents severe classification uncertainty
    high_ent_src = ent_src > 0.8
    high_ent_tgt = ent_tgt > 0.8
    
    # Borderline predictions (prob between 0.4 and 0.6)
    border_src = (y_prob_src >= 0.4) & (y_prob_src <= 0.6)
    border_tgt = (y_prob_tgt >= 0.4) & (y_prob_tgt <= 0.6)
    
    results = {
        "source": {
            "mean_confidence": float(np.mean(conf_src)),
            "std_confidence": float(np.std(conf_src)),
            "mean_entropy": float(np.mean(ent_src)),
            "std_entropy": float(np.std(ent_src)),
            "pct_high_uncertainty": float(np.mean(high_ent_src) * 100),
            "pct_borderline": float(np.mean(border_src) * 100),
        },
        "target": {
            "mean_confidence": float(np.mean(conf_tgt)),
            "std_confidence": float(np.std(conf_tgt)),
            "mean_entropy": float(np.mean(ent_tgt)),
            "std_entropy": float(np.std(ent_tgt)),
            "pct_high_uncertainty": float(np.mean(high_ent_tgt) * 100),
            "pct_borderline": float(np.mean(border_tgt) * 100),
        },
        "high_uncertainty_indices_target": np.where(high_ent_tgt)[0].tolist(),
        "borderline_indices_target": np.where(border_tgt)[0].tolist(),
    }
    
    logger.info(
        "Uncertainty Transfer: Source high uncertainty = %.2f%%, Target high uncertainty = %.2f%%",
        results["source"]["pct_high_uncertainty"],
        results["target"]["pct_high_uncertainty"],
    )
    return results
