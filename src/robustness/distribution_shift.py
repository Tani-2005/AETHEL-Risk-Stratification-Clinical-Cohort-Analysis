"""
distribution_shift.py
=====================
Quantifies covariate shift, population shift, and cross-cohort drift between
Synthetic, NHANES, and Framingham cohorts using Wasserstein distance and
Population Stability Index (PSI).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from src.evaluation.evaluator import calculate_metrics
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between expected (source)
    and actual (target) feature distributions.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin edges based on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    
    # Adjust boundaries to avoid duplicates and ensure inclusion
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequencies
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions with Laplace smoothing to avoid division by zero
    expected_props = (expected_counts + 0.5) / (len(expected) + 0.5 * len(expected_counts))
    actual_props = (actual_counts + 0.5) / (len(actual) + 0.5 * len(actual_counts))

    # Calculate PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return float(psi)

def analyze_covariate_shift(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    features: list[str],
) -> dict:
    """
    Computes feature-wise Wasserstein Distance and PSI between source (train)
    and target (validation/cross-cohort) datasets.
    """
    logger.info("DistributionShift: analyzing covariate shift...")
    
    shift_results = {}
    for col in features:
        if col in df_source.columns and col in df_target.columns:
            src_vals = df_source[col].values
            tgt_vals = df_target[col].values
            
            w_dist = wasserstein_distance(src_vals, tgt_vals)
            psi_val = calculate_psi(src_vals, tgt_vals)
            
            # Interpret PSI stability
            if psi_val < 0.1:
                interpretation = "Stable — Little to no distribution shift"
            elif psi_val < 0.25:
                interpretation = "Moderate Shift — Model adjustments may be needed"
            else:
                interpretation = "Significant Shift — Action required (re-calibration/retraining)"
                
            shift_results[col] = {
                "wasserstein_distance": float(w_dist),
                "psi": float(psi_val),
                "interpretation": interpretation,
            }
            logger.debug("  Feature '%s': PSI = %.4f (%s)", col, psi_val, interpretation)
            
    return shift_results

def evaluate_cross_cohort_drift(
    model: Any,
    X_source: pd.DataFrame,
    y_source: pd.Series,
    X_target: pd.DataFrame,
    y_target: pd.Series | None,
    features: list[str],
) -> dict:
    """
    Evaluates how model performance, calibration, and prediction distributions
    drift when evaluated on a target cohort compared to the source cohort.
    """
    logger.info("DistributionShift: evaluating cross-cohort drift...")
    
    # 1. Prediction on source
    probs_src = model.predict_proba(X_source[features])[:, 1]
    metrics_src = calculate_metrics(y_source, probs_src)
    
    # 2. Prediction on target
    probs_tgt = model.predict_proba(X_target[features])[:, 1]
    
    # Target outcome could be missing (e.g. NHANES)
    if y_target is not None and len(np.unique(y_target)) >= 2:
        metrics_tgt = calculate_metrics(y_target, probs_tgt)
        perf_deg = metrics_src.get("roc_auc", 0.5) - metrics_tgt.get("roc_auc", 0.5)
        cal_deg = metrics_tgt.get("brier", 0.0) - metrics_src.get("brier", 0.0)
    else:
        metrics_tgt = None
        perf_deg = np.nan
        cal_deg = np.nan
        
    # Probability distribution shift
    prob_w_dist = wasserstein_distance(probs_src, probs_tgt)
    prob_psi = calculate_psi(probs_src, probs_tgt)
    
    return {
        "metrics_source": metrics_src,
        "metrics_target": metrics_tgt,
        "performance_degradation_auc": float(perf_deg),
        "calibration_degradation_brier": float(cal_deg),
        "probability_shift_wasserstein": float(prob_w_dist),
        "probability_shift_psi": float(prob_psi),
    }
