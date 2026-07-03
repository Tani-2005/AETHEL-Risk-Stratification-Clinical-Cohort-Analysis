"""
robustness_reports.py
=====================
Aggregates individual robustness metrics (stability, noise degradation, missing
data, covariate shift) into a transparent, consolidated Robustness Score
and generates publication-quality summary tables.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def calculate_robustness_score(
    prediction_stability: float,
    probability_stability: float,
    explanation_stability: float,
    feature_stability: float,
    noise_retention: float,
    missing_data_retention: float,
    cross_cohort_retention: float = 1.0,
) -> dict:
    """
    Computes a transparent, aggregated Robustness Index as the unweighted mean
    of normalized measured quantities.
    """
    components = {
        "Prediction Stability (Jaccard)": max(0.0, min(1.0, prediction_stability)),
        "Probability Stability (1 - RMSE)": max(0.0, min(1.0, probability_stability)),
        "Explanation Stability (SHAP Rho)": max(0.0, min(1.0, explanation_stability)),
        "Feature Importance Stability (Rho)": max(0.0, min(1.0, feature_stability)),
        "Noise Robustness (AUC Retention at 20% noise)": max(0.0, min(1.0, noise_retention)),
        "Missing Data Robustness (AUC Retention at 20% missing)": max(0.0, min(1.0, missing_data_retention)),
    }
    
    # Optional cross-cohort retention
    if cross_cohort_retention is not None and not np.isnan(cross_cohort_retention):
        components["Cross-Cohort Stability (AUC Retention)"] = max(0.0, min(1.0, cross_cohort_retention))
        
    overall_score = float(np.mean(list(components.values())))
    
    # Qualitative grade
    if overall_score >= 0.90:
        grade = "High Robustness (Clinical-Grade)"
    elif overall_score >= 0.75:
        grade = "Moderate Robustness (Acceptable)"
    else:
        grade = "Low Robustness (Caution: Unstable)"
        
    return {
        "overall_score": overall_score,
        "components": components,
        "grade": grade,
    }

def generate_publication_table(
    all_models_results: dict[str, dict],
    output_path: Path,
) -> pd.DataFrame:
    """
    Constructs a publication-grade DataFrame summarizing the robustness properties
    of all evaluated models. Saves as CSV.
    """
    logger.info("RobustnessReports: building publication summary table...")
    
    rows = []
    for model_name, res in all_models_results.items():
        score_info = res.get("robustness_score", {})
        components = score_info.get("components", {})
        
        row = {
            "Model": model_name,
            "Overall Robustness Score": f"{score_info.get('overall_score', 0.0):.3f}",
            "Grade": score_info.get("grade", "N/A"),
            "Prediction Stability": f"{components.get('Prediction Stability (Jaccard)', 0.0):.3f}",
            "Probability Stability": f"{components.get('Probability Stability (1 - RMSE)', 0.0):.3f}",
            "Explanation Stability": f"{components.get('Explanation Stability (SHAP Rho)', 0.0):.3f}",
            "Feature Stability": f"{components.get('Feature Importance Stability (Rho)', 0.0):.3f}",
            "Noise Robustness": f"{components.get('Noise Robustness (AUC Retention at 20% noise)', 0.0):.3f}",
            "Missing Data Robustness": f"{components.get('Missing Data Robustness (AUC Retention at 20% missing)', 0.0):.3f}",
        }
        if "Cross-Cohort Stability (AUC Retention)" in components:
            row["Cross-Cohort Stability"] = f"{components.get('Cross-Cohort Stability (AUC Retention)', 0.0):.3f}"
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved publication robustness table to %s", output_path)
    return df
