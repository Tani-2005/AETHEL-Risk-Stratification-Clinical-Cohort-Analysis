"""
calibration_transfer.py
=======================
Evaluates calibration transfer across cohorts. Computes Brier scores,
expected calibration error (ECE), calibration slope/intercept drift,
and reliability comparison.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss
from src.evaluation.evaluator import (
    compute_calibration_intercept_slope,
    expected_calibration_error,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def evaluate_calibration_transfer(
    y_true_src: np.ndarray,
    y_prob_src: np.ndarray,
    y_true_tgt: np.ndarray,
    y_prob_tgt: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compares calibration metrics between source (internal) and target (external) cohorts.
    Returns differences representing calibration drift.
    """
    logger.info("CalibrationTransfer: auditing calibration transfer...")
    
    # 1. Source Calibration
    ece_src, mce_src = expected_calibration_error(y_true_src, y_prob_src, n_bins)
    intercept_src, slope_src = compute_calibration_intercept_slope(y_true_src, y_prob_src)
    brier_src = brier_score_loss(y_true_src, y_prob_src)
    
    # 2. Target Calibration
    ece_tgt, mce_tgt = expected_calibration_error(y_true_tgt, y_prob_tgt, n_bins)
    intercept_tgt, slope_tgt = compute_calibration_intercept_slope(y_true_tgt, y_prob_tgt)
    brier_tgt = brier_score_loss(y_true_tgt, y_prob_tgt)
    
    # 3. Calibration Drift / Differences
    # Ideal calibration has slope = 1, intercept = 0.
    # Drift measures how far the target calibration moves from the source.
    slope_drift = slope_tgt - slope_src if not np.isnan(slope_src) and not np.isnan(slope_tgt) else np.nan
    intercept_drift = intercept_tgt - intercept_src if not np.isnan(intercept_src) and not np.isnan(intercept_tgt) else np.nan
    ece_drift = ece_tgt - ece_src
    brier_drift = brier_tgt - brier_src
    
    # Construct calibration curves (binned accuracy vs confidence) for plotting
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve_src = []
    curve_tgt = []
    
    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i+1]
        
        # Source
        mask_src = (y_prob_src >= b_low) & (y_prob_src < b_high)
        if i == n_bins - 1:
            mask_src = mask_src | (y_prob_src == b_high)
        conf_src = np.mean(y_prob_src[mask_src]) if np.sum(mask_src) > 0 else np.nan
        acc_src = np.mean(y_true_src[mask_src]) if np.sum(mask_src) > 0 else np.nan
        curve_src.append({"bin": i, "confidence": conf_src, "accuracy": acc_src, "count": int(np.sum(mask_src))})
        
        # Target
        mask_tgt = (y_prob_tgt >= b_low) & (y_prob_tgt < b_high)
        if i == n_bins - 1:
            mask_tgt = mask_tgt | (y_prob_tgt == b_high)
        conf_tgt = np.mean(y_prob_tgt[mask_tgt]) if np.sum(mask_tgt) > 0 else np.nan
        acc_tgt = np.mean(y_true_tgt[mask_tgt]) if np.sum(mask_tgt) > 0 else np.nan
        curve_tgt.append({"bin": i, "confidence": conf_tgt, "accuracy": acc_tgt, "count": int(np.sum(mask_tgt))})
        
    return {
        "source": {
            "ece": float(ece_src),
            "mce": float(mce_src),
            "slope": float(slope_src),
            "intercept": float(intercept_src),
            "brier": float(brier_src),
            "curve": curve_src,
        },
        "target": {
            "ece": float(ece_tgt),
            "mce": float(mce_tgt),
            "slope": float(slope_tgt),
            "intercept": float(intercept_tgt),
            "brier": float(brier_tgt),
            "curve": curve_tgt,
        },
        "drift": {
            "ece_drift": float(ece_drift),
            "brier_drift": float(brier_drift),
            "slope_drift": float(slope_drift),
            "intercept_drift": float(intercept_drift),
        }
    }
