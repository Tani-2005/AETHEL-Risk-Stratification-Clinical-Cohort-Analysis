"""
explanation_drift.py
=====================
Compares SHAP rankings, permutation importance, rank correlation, and top-k agreement
across independent cohorts to identify stable and cohort-specific risk factors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def calculate_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    features: list[str],
    random_state: int = 42,
) -> dict[str, float]:
    """Calculates permutation feature importance for a given model and dataset."""
    y = np.asarray(y)
    try:
        res = permutation_importance(
            model, X[features], y, n_repeats=3, random_state=random_state, n_jobs=-1
        )
        importances = {f: float(res.importances_mean[i]) for i, f in enumerate(features)}
    except Exception as e:
        logger.warning("  Permutation importance calculation failed: %s. Using dummy.", e)
        importances = {f: 0.0 for f in features}
    return importances

def compare_explanation_drift(
    shap_imp_src: dict[str, float],
    shap_imp_tgt: dict[str, float],
    perm_imp_src: dict[str, float],
    perm_imp_tgt: dict[str, float],
    features: list[str],
    top_k: int = 3,
) -> dict:
    """
    Compares explanation models between source and target cohorts.
    Computes Spearman rank correlation and top-k agreement for SHAP and Permutation importance.
    """
    logger.info("ExplanationDrift: auditing explanation rankings...")
    
    # 1. SHAP Rank Correlation
    src_shap_ranks = [shap_imp_src.get(f, 0.0) for f in features]
    tgt_shap_ranks = [shap_imp_tgt.get(f, 0.0) for f in features]
    
    shap_corr, shap_p = spearmanr(src_shap_ranks, tgt_shap_ranks)
    if np.isnan(shap_corr):
        shap_corr = 0.0
        
    # 2. Permutation Importance Rank Correlation
    src_perm_ranks = [perm_imp_src.get(f, 0.0) for f in features]
    tgt_perm_ranks = [perm_imp_tgt.get(f, 0.0) for f in features]
    
    perm_corr, perm_p = spearmanr(src_perm_ranks, tgt_perm_ranks)
    if np.isnan(perm_corr):
        perm_corr = 0.0
        
    # 3. Top-k Agreement
    # Sort features by importance to get top-k sets
    top_k_shap_src = sorted(features, key=lambda f: shap_imp_src.get(f, 0.0), reverse=True)[:top_k]
    top_k_shap_tgt = sorted(features, key=lambda f: shap_imp_tgt.get(f, 0.0), reverse=True)[:top_k]
    
    top_k_perm_src = sorted(features, key=lambda f: perm_imp_src.get(f, 0.0), reverse=True)[:top_k]
    top_k_perm_tgt = sorted(features, key=lambda f: perm_imp_tgt.get(f, 0.0), reverse=True)[:top_k]
    
    shap_agreement = len(set(top_k_shap_src) & set(top_k_shap_tgt)) / top_k
    perm_agreement = len(set(top_k_perm_src) & set(top_k_perm_tgt)) / top_k
    
    # 4. Identify Stable vs Cohort-Specific Risk Factors (Consensus)
    # Stable: present in top-k in both source and target (SHAP or Permutation)
    stable_factors = list(set(top_k_shap_src) & set(top_k_shap_tgt))
    
    # Cohort-specific: top-k in source but not in target, or vice versa
    src_only_factors = list(set(top_k_shap_src) - set(top_k_shap_tgt))
    tgt_only_factors = list(set(top_k_shap_tgt) - set(top_k_shap_src))
    
    return {
        "shap_rank_correlation": float(shap_corr),
        "shap_p_value": float(shap_p) if not np.isnan(shap_p) else 1.0,
        "perm_rank_correlation": float(perm_corr),
        "perm_p_value": float(perm_p) if not np.isnan(perm_p) else 1.0,
        "shap_top_k_agreement": float(shap_agreement),
        "perm_top_k_agreement": float(perm_agreement),
        "stable_risk_factors": stable_factors,
        "source_cohort_specific_factors": src_only_factors,
        "target_cohort_specific_factors": tgt_only_factors,
        "top_k_features": {
            "source_shap": top_k_shap_src,
            "target_shap": top_k_shap_tgt,
            "source_perm": top_k_perm_src,
            "target_perm": top_k_perm_tgt,
        }
    }
