"""
bootstrap.py
============
Optimized bootstrap sampling to estimate 95% confidence intervals and store
full distributions for performance, calibration, and clinical utility metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.evaluation.evaluator import calculate_metrics
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def run_bootstrap_analysis(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    time_col: Optional[pd.Series | np.ndarray] = None,
    event_col: Optional[pd.Series | np.ndarray] = None,
    n_iterations: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Performs bootstrap resampling on predictions to calculate distributions and
    95% Confidence Intervals for key metrics.
    """
    logger.info("Bootstrap: running %d iterations...", n_iterations)
    
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    t_col = np.asarray(time_col) if time_col is not None else None
    e_col = np.asarray(event_col) if event_col is not None else None
    
    n_samples = len(y_true)
    rng = np.random.default_rng(random_state)
    
    bootstrap_results = []
    
    for i in range(n_iterations):
        # Resample indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        # Ensure we have at least one sample of each class to avoid AUC errors
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        boot_y_true = y_true[indices]
        boot_y_prob = y_prob[indices]
        boot_t = t_col[indices] if t_col is not None else None
        boot_e = e_col[indices] if e_col is not None else None
        
        # Compute metrics
        metrics = calculate_metrics(boot_y_true, boot_y_prob, time_col=boot_t, event_col=boot_e)
        bootstrap_results.append(metrics)
        
    if not bootstrap_results:
        raise ValueError("Bootstrap failed to generate valid splits.")
        
    # Convert list of dicts to dict of lists
    metric_keys = list(bootstrap_results[0].keys())
    distributions = {key: [run[key] for run in bootstrap_results] for key in metric_keys}
    
    # Calculate stats and CIs
    bootstrap_stats = {}
    for key in metric_keys:
        vals = np.array(distributions[key])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            bootstrap_stats[key] = {
                "mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan
            }
            continue
            
        bootstrap_stats[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }
        
    return {
        "bootstrap_stats": bootstrap_stats,
        "distributions": distributions,
    }
