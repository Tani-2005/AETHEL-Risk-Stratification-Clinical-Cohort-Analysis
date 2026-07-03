"""
stability_metrics.py
====================
Computes prediction, probability, feature ranking, and explanation (SHAP)
stability across repeated runs of machine learning models.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, pearsonr

def compute_prediction_stability(predictions_list: list[np.ndarray] | np.ndarray) -> dict[str, float]:
    """
    Computes pairwise Jaccard similarity (on positive predictions) and raw agreement
    rates across repeated runs.
    """
    preds = np.asarray(predictions_list)  # Shape: (n_runs, n_samples)
    n_runs, n_samples = preds.shape
    if n_runs < 2:
        return {"jaccard_stability": 1.0, "prediction_agreement": 1.0}

    jaccard_scores = []
    agreement_scores = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            p1 = preds[i]
            p2 = preds[j]
            
            # Raw agreement rate
            agreement_scores.append(np.mean(p1 == p2))
            
            # Jaccard similarity of positive predictions
            union_pos = np.sum((p1 == 1) | (p2 == 1))
            if union_pos > 0:
                jaccard = np.sum((p1 == 1) & (p2 == 1)) / union_pos
                jaccard_scores.append(jaccard)
            else:
                jaccard_scores.append(1.0)

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

    corrs = []
    mses = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            pr1 = probs[i]
            pr2 = probs[j]
            
            # Pearson correlation
            if np.std(pr1) > 0 and np.std(pr2) > 0:
                r, _ = pearsonr(pr1, pr2)
                corrs.append(r if not np.isnan(r) else 1.0)
            else:
                corrs.append(1.0)
                
            # MSE
            mses.append(np.mean((pr1 - pr2) ** 2))

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

    rhos = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            rho, _ = spearmanr(imps[i], imps[j])
            rhos.append(rho if not np.isnan(rho) else 1.0)

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
        rhos = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                rho, _ = spearmanr(run_vectors[i], run_vectors[j])
                rhos.append(rho if not np.isnan(rho) else 1.0)
        sample_rhos.append(np.mean(rhos))

    return {
        "explanation_stability_rho": float(np.mean(sample_rhos)),
    }
