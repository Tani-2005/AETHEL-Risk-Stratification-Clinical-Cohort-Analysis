"""
stability_metrics.py
====================
Computes prediction, probability, feature ranking, and explanation (SHAP)
stability across repeated runs of machine learning models.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

def compute_prediction_stability(predictions_list: list[np.ndarray] | np.ndarray) -> dict[str, float]:
    """
    Computes pairwise Jaccard similarity (on positive predictions) and raw agreement
    rates across repeated runs.
    """
    preds = np.asarray(predictions_list)  # Shape: (n_runs, n_samples)
    n_runs, n_samples = preds.shape
    if n_runs < 2:
        return {"jaccard_stability": 1.0, "prediction_agreement": 1.0}

    triu_indices = np.triu_indices(n_runs, k=1)
    
    # Raw agreement rate: mean of (preds[i] == preds[j])
    agreement_matrix = np.mean(preds[:, None, :] == preds[None, :, :], axis=2)
    agreement_scores = agreement_matrix[triu_indices]

    # Jaccard similarity: intersection / union of positive predictions
    pos_preds = (preds == 1)  # bool array shape (n_runs, n_samples)
    intersection = np.sum(pos_preds[:, None, :] & pos_preds[None, :, :], axis=2) # (n_runs, n_runs)
    union = np.sum(pos_preds[:, None, :] | pos_preds[None, :, :], axis=2) # (n_runs, n_runs)
    
    # Avoid divide by zero
    jaccard_matrix = np.where(union > 0, intersection / union, 1.0)
    jaccard_scores = jaccard_matrix[triu_indices]

    return {
        "jaccard_stability": float(np.mean(jaccard_scores)),
        "prediction_agreement": float(np.mean(agreement_scores)),
    }

def compute_probability_stability(probabilities_list: list[np.ndarray] | np.ndarray) -> dict[str, float]:
    """
    Computes pairwise Pearson correlation and Mean Squared Error (MSE)
    of predicted probabilities across repeated runs.
    """
    probs = np.asarray(probabilities_list)  # Shape: (n_runs, n_samples)
    n_runs, n_samples = probs.shape
    if n_runs < 2:
        return {"probability_correlation": 1.0, "probability_mse": 0.0}

    triu_indices = np.triu_indices(n_runs, k=1)

    # Pearson correlation
    corr_matrix = np.corrcoef(probs)  # Shape: (n_runs, n_runs)
    corrs = corr_matrix[triu_indices]
    corrs = np.nan_to_num(corrs, nan=1.0)

    # MSE
    diff = probs[:, None, :] - probs[None, :, :]
    mse_matrix = np.mean(diff ** 2, axis=2)
    mses = mse_matrix[triu_indices]

    return {
        "probability_correlation": float(np.mean(corrs)),
        "probability_mse": float(np.mean(mses)),
    }

def compute_ranking_stability(importances_list: list[np.ndarray] | np.ndarray) -> dict[str, float]:
    """
    Computes pairwise Spearman rank correlation of feature importances across runs.
    """
    imps = np.asarray(importances_list)  # Shape: (n_runs, n_features)
    n_runs, n_features = imps.shape
    if n_runs < 2 or n_features < 2:
        return {"ranking_stability_rho": 1.0}

    ranks = rankdata(imps, axis=1)
    # Check for cases where all ranks are constant (e.g. constant importances)
    if np.all(np.std(ranks, axis=1) == 0):
        return {"ranking_stability_rho": 1.0}
        
    corr_matrix = np.corrcoef(ranks)
    triu_indices = np.triu_indices(n_runs, k=1)
    rhos = corr_matrix[triu_indices]
    rhos = np.nan_to_num(rhos, nan=1.0)

    return {
        "ranking_stability_rho": float(np.mean(rhos)),
    }

def compute_explanation_stability(shap_values_list: list[np.ndarray] | np.ndarray) -> dict[str, float]:
    """
    Computes pairwise Spearman rank correlation of SHAP values across runs.
    shap_values_list shape: (n_runs, n_samples, n_features)
    """
    shaps = np.asarray(shap_values_list)
    n_runs, n_samples, n_features = shaps.shape
    if n_runs < 2 or n_features < 2:
        return {"explanation_stability_rho": 1.0}

    # Measure average correlation per sample, then average across samples
    sample_rhos = []
    for s in range(n_samples):
        run_vectors = shaps[:, s, :]  # Shape: (n_runs, n_features)
        ranks = rankdata(run_vectors, axis=1)  # Shape: (n_runs, n_features)
        
        if np.all(np.std(ranks, axis=1) == 0):
            sample_rhos.append(1.0)
            continue
            
        corr_matrix = np.corrcoef(ranks)  # Shape: (n_runs, n_runs)
        triu_indices = np.triu_indices(n_runs, k=1)
        rhos = corr_matrix[triu_indices]
        rhos = np.nan_to_num(rhos, nan=1.0)
        sample_rhos.append(np.mean(rhos))

    return {
        "explanation_stability_rho": float(np.mean(sample_rhos)),
    }

