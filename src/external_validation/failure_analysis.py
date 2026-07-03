"""
failure_analysis.py
===================
Analyzes target cohort prediction failures, categorizes failure types
(high/low confidence, consistently misclassified), and clusters failure modes
using feature values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def run_failure_analysis(
    X_tgt: pd.DataFrame,
    y_tgt: pd.Series | np.ndarray,
    probs_tgt: np.ndarray,
    features: list[str],
    n_clusters: int = 3,
) -> dict:
    """
    Identifies failure categories and clusters the feature vectors of failed predictions
    to find common clinical sub-profiles where the model fails.
    """
    logger.info("FailureAnalysis: auditing failures on target cohort...")
    
    y_tgt = np.asarray(y_tgt)
    preds_tgt = (probs_tgt >= 0.5).astype(int)
    
    # Identify indices of correct and failed predictions
    correct_indices = np.where(preds_tgt == y_tgt)[0]
    failed_indices = np.where(preds_tgt != y_tgt)[0]
    
    n_failures = len(failed_indices)
    if n_failures == 0:
        logger.info("  Zero failures detected on target cohort.")
        return {
            "n_failures": 0,
            "failure_types": {},
            "cluster_centroids": [],
        }
        
    probs_failed = probs_tgt[failed_indices]
    y_failed = y_tgt[failed_indices]
    
    # 1. Categorize failures
    # High confidence failures: prob far from true label
    high_conf_failed = []
    # Low confidence failures: prob close to 0.5
    low_conf_failed = []
    
    for idx, f_idx in enumerate(failed_indices):
        p = probs_failed[idx]
        y = y_failed[idx]
        
        # High confidence failure
        if (y == 1 and p < 0.3) or (y == 0 and p > 0.7):
            high_conf_failed.append(int(f_idx))
        # Low confidence failure
        elif 0.4 <= p <= 0.6:
            low_conf_failed.append(int(f_idx))
            
    # 2. Cluster failure modes
    X_failed = X_tgt[features].iloc[failed_indices].copy()
    
    cluster_labels = np.zeros(len(X_failed), dtype=int)
    centroids_df = pd.DataFrame()
    
    if len(X_failed) >= n_clusters:
        try:
            scaler = StandardScaler()
            X_failed_scaled = scaler.fit_transform(X_failed)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_failed_scaled)
            
            # Compute raw mean feature values for each cluster
            X_failed["failure_cluster"] = cluster_labels
            centroids_df = X_failed.groupby("failure_cluster")[features].mean()
        except Exception as e:
            logger.warning("  KMeans clustering of failures failed: %s", e)
            
    # Count of failures in each cluster
    cluster_counts = pd.Series(cluster_labels).value_counts().to_dict()
    cluster_stats = []
    for c_id in sorted(cluster_counts.keys()):
        stats = {
            "cluster_id": int(c_id),
            "n_patients": int(cluster_counts[c_id]),
        }
        if not centroids_df.empty and c_id in centroids_df.index:
            stats["features_mean"] = centroids_df.loc[c_id].to_dict()
        cluster_stats.append(stats)

    return {
        "n_failures": n_failures,
        "failure_rate": float(n_failures / len(X_tgt)),
        "failed_indices": failed_indices.tolist(),
        "cluster_labels": cluster_labels.tolist(),
        "failure_types": {
            "n_high_confidence_failures": len(high_conf_failed),
            "n_low_confidence_failures": len(low_conf_failed),
            "high_confidence_failures_indices": high_conf_failed,
            "low_confidence_failures_indices": low_conf_failed,
        },
        "clusters": cluster_stats,
    }
