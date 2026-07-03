"""
trustworthiness_evaluator.py
=============================
Assembles the independent dimensions of clinical AI trustworthiness:
1. Prediction Reliability
2. Calibration Reliability
3. Explanation Reliability
4. Robustness
5. Generalization
6. Clinical Consistency
7. Reproducibility
"""
from __future__ import annotations
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def build_trustworthiness_profile(
    prediction_metrics: dict,
    calibration_metrics: dict,
    explanation_metrics: dict,
    robustness_score: float,
    generalization_gap_metrics: dict,
    consistency_metrics: dict,
    reproducibility_std_auc: float,
) -> dict:
    """
    Combines the distinct dimensions of model trustworthiness into a structured profile.
    DO NOT collapse this into a single index; present as a multi-dimensional assessment.
    """
    logger.info("TrustworthinessEvaluator: building multi-dimensional profile...")
    
    # Assess grades for each dimension independently
    # 1. Prediction Reliability
    auc_val = prediction_metrics.get("roc_auc", 0.0)
    if auc_val >= 0.85:
        pred_grade = "Excellent (Clinical-Grade)"
    elif auc_val >= 0.70:
        pred_grade = "Acceptable"
    else:
        pred_grade = "Poor"
        
    # 2. Calibration Reliability
    ece_val = calibration_metrics.get("ece", 0.0)
    if ece_val <= 0.03:
        calib_grade = "High Calibration (Tight Alignment)"
    elif ece_val <= 0.08:
        calib_grade = "Moderate Calibration"
    else:
        calib_grade = "Poorly Calibrated (Significant Over/Underestimation)"
        
    # 3. Explanation Reliability
    shap_corr = explanation_metrics.get("shap_rank_correlation", 0.0)
    if shap_corr >= 0.80:
        explain_grade = "Highly Reliable (Consistent Features)"
    elif shap_corr >= 0.50:
        explain_grade = "Moderately Reliable"
    else:
        explain_grade = "Unreliable (Cohort-Specific Rankings)"
        
    # 4. Robustness
    if robustness_score >= 0.90:
        robust_grade = "High Robustness"
    elif robustness_score >= 0.75:
        robust_grade = "Moderate Robustness"
    else:
        robust_grade = "Vulnerable to Noise/Perturbations"
        
    # 5. Generalization
    auc_drop = generalization_gap_metrics.get("roc_auc_drop", 0.0)
    if auc_drop <= 0.05:
        gen_grade = "High Generalizability (Stable Performance)"
    elif auc_drop <= 0.15:
        gen_grade = "Moderate Generalization Gap"
    else:
        gen_grade = "Severe Generalization Drop (Overfitted to Source Cohort)"
        
    # 6. Clinical Consistency
    const_rate = consistency_metrics.get("consistency_rate", 1.0)
    if const_rate >= 0.90:
        clinical_grade = "Completely Consistent with Clinical Expectation"
    elif const_rate >= 0.70:
        clinical_grade = "Mostly Consistent"
    else:
        clinical_grade = "Clinical Contradictions Present"
        
    # 7. Reproducibility
    if reproducibility_std_auc <= 0.01:
        reprod_grade = "High Reproducibility (Stable across Seeds)"
    elif reproducibility_std_auc <= 0.03:
        reprod_grade = "Moderate Reproducibility"
    else:
        reprod_grade = "Low Reproducibility (Highly Seed-Dependent)"
        
    return {
        "prediction_reliability": {
            "roc_auc": float(auc_val),
            "pr_auc": float(prediction_metrics.get("pr_auc", 0.0)),
            "f1_score": float(prediction_metrics.get("f1_score", 0.0)),
            "grade": pred_grade,
        },
        "calibration_reliability": {
            "ece": float(ece_val),
            "mce": float(calibration_metrics.get("mce", 0.0)),
            "brier_score": float(calibration_metrics.get("brier", 0.0)),
            "slope": float(calibration_metrics.get("slope", 1.0)),
            "intercept": float(calibration_metrics.get("intercept", 0.0)),
            "grade": calib_grade,
        },
        "explanation_reliability": {
            "shap_rank_correlation": float(shap_corr),
            "perm_rank_correlation": float(explanation_metrics.get("perm_rank_correlation", 0.0)),
            "top_k_agreement": float(explanation_metrics.get("shap_top_k_agreement", 0.0)),
            "grade": explain_grade,
        },
        "robustness": {
            "score": float(robustness_score),
            "grade": robust_grade,
        },
        "generalization": {
            "roc_auc_drop": float(auc_drop),
            "ece_increase": float(generalization_gap_metrics.get("ece_increase", 0.0)),
            "grade": gen_grade,
        },
        "clinical_consistency": {
            "consistency_rate": float(const_rate),
            "n_consistent": int(consistency_metrics.get("n_consistent", 0)),
            "n_audited": int(consistency_metrics.get("n_audited", 0)),
            "grade": clinical_grade,
        },
        "reproducibility": {
            "std_auc": float(reproducibility_std_auc),
            "grade": reprod_grade,
        }
    }
