"""
noise_analysis.py
=================
Evaluates model robustness against controlled noise injection, including
Gaussian noise on continuous features and random bit-flips on binary features.
Generates degradation curves.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.evaluation.evaluator import calculate_metrics
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def inject_gaussian_noise(
    df: pd.DataFrame,
    features: list[str],
    noise_level: float,
    continuous_features: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Adds zero-mean Gaussian noise to continuous features, scaled by standard deviation."""
    df_noisy = df.copy()
    rng = np.random.default_rng(random_state)
    
    for col in continuous_features:
        if col in features and col in df.columns:
            std = df[col].std()
            if np.isnan(std) or std == 0:
                std = 1.0
            noise = rng.normal(0, std * noise_level, size=len(df))
            df_noisy[col] = df_noisy[col] + noise
            
    return df_noisy

def inject_binary_flips(
    df: pd.DataFrame,
    features: list[str],
    flip_prob: float,
    binary_features: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomly flips binary features (0 -> 1, 1 -> 0) with a given probability."""
    df_noisy = df.copy()
    rng = np.random.default_rng(random_state)
    
    for col in binary_features:
        if col in features and col in df.columns:
            # Generate mask of indices to flip
            flip_mask = rng.random(size=len(df)) < flip_prob
            # Flip values (assumes 0/1 representation)
            df_noisy.loc[flip_mask, col] = 1 - df_noisy.loc[flip_mask, col]
            
    return df_noisy

def run_noise_robustness(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    continuous_features: list[str],
    binary_features: list[str],
    noise_levels: list[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
    random_state: int = 42,
) -> dict:
    """
    Sweeps noise levels, perturbing the validation set, and measuring metric
    degradation curves.
    """
    logger.info("NoiseAnalysis: running noise robustness sweep...")
    
    # Identify active continuous/binary features in the current experiment features list
    active_cont = [f for f in continuous_features if f in features]
    active_bin = [f for f in binary_features if f in features]
    
    results = {}
    
    for lvl in noise_levels:
        # 1. Apply Gaussian noise to continuous
        X_perturbed = inject_gaussian_noise(X_val, features, lvl, active_cont, random_state)
        # 2. Apply Bit flips to binary
        X_perturbed = inject_binary_flips(X_perturbed, features, lvl, active_bin, random_state)
        
        # Predict on perturbed data
        probs = model.predict_proba(X_perturbed[features])[:, 1]
        metrics = calculate_metrics(y_val, probs)
        
        results[lvl] = {
            "metrics": metrics,
            "predictions_mse_with_original": float(np.mean((model.predict_proba(X_val[features])[:, 1] - probs) ** 2)),
        }
        logger.debug("  Noise level %.2f: ROC-AUC = %.4f", lvl, metrics.get("roc_auc", np.nan))
        
    return results
