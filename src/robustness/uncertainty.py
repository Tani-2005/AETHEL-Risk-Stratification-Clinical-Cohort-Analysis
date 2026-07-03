"""
uncertainty.py
==============
Estimates prediction uncertainty (confidence, entropy, probability variance
across repeated runs) and performs automated failure analysis (identifying
hard cases, borderline predictions, and repeated misclassifications).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Computes binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p) for predicted probabilities."""
    probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    return -probs * np.log2(probs) - (1.0 - probs) * np.log2(1.0 - probs)

def estimate_uncertainty(
    probs_list: list[np.ndarray] | np.ndarray,
    y_true: pd.Series | np.ndarray | None = None,
) -> dict:
    """
    Estimates prediction uncertainty using predicted probabilities across repeated runs
    and flags high-uncertainty or hard/borderline cases.
    """
    logger.info("Uncertainty: estimating prediction uncertainty and failure analysis...")
    
    probs = np.asarray(probs_list)  # Shape: (n_runs, n_samples)
    n_runs, n_samples = probs.shape
    
    # Average predicted probability per sample
    mean_probs = np.mean(probs, axis=0)
    # Variance across repeated runs (epistemic uncertainty)
    run_var = np.var(probs, axis=0)
    # Confidence: 2 * |mean_prob - 0.5|
    confidence = 2.0 * np.abs(mean_probs - 0.5)
    # Entropy of the mean probability
    entropy = compute_entropy(mean_probs)
    
    # 95% CI bounds across runs for each sample
    ci_lower = np.percentile(probs, 2.5, axis=0)
    ci_upper = np.percentile(probs, 97.5, axis=0)
    
    # Flags
    uncertain_mask = (confidence < 0.2) | (entropy > 0.8) | (run_var > 0.05)
    borderline_mask = (mean_probs >= 0.4) & (mean_probs <= 0.6)
    
    summary_df = pd.DataFrame({
        "sample_idx": np.arange(n_samples),
        "mean_prob": mean_probs,
        "prob_var": run_var,
        "confidence": confidence,
        "entropy": entropy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "is_uncertain": uncertain_mask,
        "is_borderline": borderline_mask,
    })
    
    # Failure Analysis (requires ground truth outcome)
    failure_report = {}
    if y_true is not None:
        y_true = np.asarray(y_true)
        # Binary predictions per run
        preds = (probs >= 0.5).astype(int)
        
        # Misclassification rate per sample across runs
        misclassified_mask = preds != y_true[np.newaxis, :]  # Shape: (n_runs, n_samples)
        misclass_rate = np.mean(misclassified_mask, axis=0)
        
        summary_df["true_label"] = y_true
        summary_df["misclassification_rate"] = misclass_rate
        
        # Hard cases: misclassified in > 80% of runs
        hard_cases = np.where(misclass_rate >= 0.8)[0]
        # Repeated misclassifications: misclassified in 100% of runs
        persistent_failures = np.where(misclass_rate == 1.0)[0]
        # Low confidence errors: misclassified with low confidence
        low_conf_errors = np.where((misclass_rate > 0.5) & (confidence < 0.3))[0]
        # High confidence errors: misclassified with high confidence (mean prob far on wrong side)
        high_conf_errors = np.where(
            (misclass_rate > 0.5) & 
            (((y_true == 1) & (mean_probs < 0.3)) | ((y_true == 0) & (mean_probs > 0.7)))
        )[0]
        
        failure_report = {
            "n_hard_cases": len(hard_cases),
            "n_persistent_failures": len(persistent_failures),
            "n_low_confidence_errors": len(low_conf_errors),
            "n_high_confidence_errors": len(high_conf_errors),
            "hard_cases_indices": hard_cases.tolist(),
            "persistent_failures_indices": persistent_failures.tolist(),
            "high_confidence_errors_indices": high_conf_errors.tolist(),
        }
        
    return {
        "uncertainty_summary": summary_df,
        "failure_report": failure_report,
    }
