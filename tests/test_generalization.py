"""
test_generalization.py
======================
Unit tests for the generalization, domain shift, and trustworthiness components.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.external_validation import (
    load_and_align_cohorts,
    get_surrogate_outcome,
    preprocess_cross_cohort,
    evaluate_calibration_transfer,
    evaluate_uncertainty_transfer,
    run_failure_analysis,
    BIOCHEMICAL_FEATURES,
)
from src.domain_shift import quantify_domain_shift
from src.generalization import (
    measure_generalization_gap,
    compare_explanation_drift,
)
from src.trustworthiness import (
    evaluate_clinical_consistency,
    build_trustworthiness_profile,
)

@pytest.fixture(scope="module")
def sample_data():
    """Generates dummy source and target dataframes for validation testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Source (Framingham-like)
    df_src = pd.DataFrame({
        "h_sys_bp": np.random.normal(120, 15, n_samples),
        "h_dia_bp": np.random.normal(80, 10, n_samples),
        "h_total_cholesterol": np.random.normal(200, 30, n_samples),
        "h_age": np.random.normal(50, 10, n_samples),
        "h_bmi": np.random.normal(25, 4, n_samples),
        "h_is_smoker": np.random.binomial(1, 0.25, n_samples).astype(float),
        "h_outcome_binary": np.random.binomial(1, 0.20, n_samples).astype(float),
    })
    
    # Target (NHANES-like with shift)
    df_tgt = pd.DataFrame({
        "h_sys_bp": np.random.normal(130, 20, n_samples),
        "h_dia_bp": np.random.normal(85, 12, n_samples),
        "h_total_cholesterol": np.random.normal(215, 35, n_samples),
        "h_outcome_binary": np.random.binomial(1, 0.28, n_samples).astype(float),
    })
    
    return df_src, df_tgt

def test_surrogate_outcome_calculation(sample_data):
    df_src, _ = sample_data
    surr = get_surrogate_outcome(df_src)
    assert len(surr) == len(df_src)
    assert set(surr.unique()).issubset({0, 1})
    
    # Check simple cases manually
    manual_df = pd.DataFrame({
        "h_sys_bp": [120, 140, 120, 120],
        "h_dia_bp": [80, 80, 90, 80],
        "h_total_cholesterol": [200, 200, 200, 240],
    })
    res = get_surrogate_outcome(manual_df)
    assert list(res) == [0.0, 1.0, 1.0, 1.0]

def test_preprocess_cross_cohort(sample_data):
    df_src, df_tgt = sample_data
    X_tr, y_tr, X_te, y_te, imp, scaler = preprocess_cross_cohort(
        df_src, df_tgt, BIOCHEMICAL_FEATURES, "h_outcome_binary"
    )
    
    assert X_tr.shape == (200, 3)
    assert X_te.shape == (200, 3)
    assert len(y_tr) == 200
    assert len(y_te) == 200
    assert list(X_tr.columns) == BIOCHEMICAL_FEATURES

def test_calibration_transfer(sample_data):
    df_src, df_tgt = sample_data
    
    # Mock probabilities
    y_true_src = df_src["h_outcome_binary"].values
    y_prob_src = np.clip(y_true_src * 0.8 + np.random.normal(0, 0.1, len(df_src)), 0.01, 0.99)
    
    y_true_tgt = df_tgt["h_outcome_binary"].values
    y_prob_tgt = np.clip(y_true_tgt * 0.6 + np.random.normal(0, 0.2, len(df_tgt)), 0.01, 0.99)
    
    res = evaluate_calibration_transfer(y_true_src, y_prob_src, y_true_tgt, y_prob_tgt)
    
    assert "source" in res
    assert "target" in res
    assert "drift" in res
    assert 0.0 <= res["source"]["ece"] <= 1.0
    assert 0.0 <= res["target"]["ece"] <= 1.0
    assert "slope_drift" in res["drift"]

def test_uncertainty_transfer(sample_data):
    df_src, df_tgt = sample_data
    y_prob_src = np.random.uniform(0.1, 0.9, len(df_src))
    y_prob_tgt = np.random.uniform(0.1, 0.9, len(df_tgt))
    
    res = evaluate_uncertainty_transfer(y_prob_src, y_prob_tgt)
    
    assert "source" in res
    assert "target" in res
    assert "high_uncertainty_indices_target" in res
    assert 0.0 <= res["source"]["mean_entropy"] <= 1.0
    assert 0.0 <= res["target"]["mean_entropy"] <= 1.0

def test_failure_analysis(sample_data):
    df_src, df_tgt = sample_data
    features = BIOCHEMICAL_FEATURES
    
    X_tr, y_tr, X_te, y_te, imp, scaler = preprocess_cross_cohort(
        df_src, df_tgt, features, "h_outcome_binary"
    )
    
    model = LogisticRegression()
    model.fit(X_tr, y_tr)
    probs_tgt = model.predict_proba(X_te)[:, 1]
    
    res = run_failure_analysis(X_te, y_te, probs_tgt, features, n_clusters=2)
    
    assert "n_failures" in res
    assert "failure_types" in res
    assert "clusters" in res
    if res["n_failures"] >= 2:
        assert len(res["clusters"]) == 2

def test_quantify_domain_shift(sample_data):
    df_src, df_tgt = sample_data
    res = quantify_domain_shift(df_src, df_tgt, BIOCHEMICAL_FEATURES, "h_outcome_binary")
    
    assert "covariate_shift" in res
    assert "prior_shift" in res
    assert "population_shift_avg_psi" in res
    assert "h_sys_bp" in res["covariate_shift"]
    assert 0.0 <= res["prior_shift"] <= 1.0

def test_generalization_gap(sample_data):
    df_src, df_tgt = sample_data
    metrics_src = {"roc_auc": 0.82, "pr_auc": 0.78, "f1_score": 0.75, "ece": 0.02}
    metrics_tgt = {"roc_auc": 0.71, "pr_auc": 0.64, "f1_score": 0.61, "ece": 0.06}
    
    y_prob_src = np.random.uniform(0.1, 0.9, len(df_src))
    y_prob_tgt = np.random.uniform(0.1, 0.9, len(df_tgt))
    
    res = measure_generalization_gap(metrics_src, metrics_tgt, y_prob_src, y_prob_tgt)
    
    assert "generalization_gap" in res
    assert "prediction_drift" in res
    assert round(res["generalization_gap"]["roc_auc_drop"], 2) == 0.11
    assert round(res["generalization_gap"]["ece_increase"], 2) == 0.04

def test_explanation_drift():
    shap_src = {"h_sys_bp": 0.5, "h_dia_bp": 0.3, "h_total_cholesterol": 0.2}
    shap_tgt = {"h_sys_bp": 0.4, "h_dia_bp": 0.2, "h_total_cholesterol": 0.4}
    perm_src = {"h_sys_bp": 0.15, "h_dia_bp": 0.10, "h_total_cholesterol": 0.05}
    perm_tgt = {"h_sys_bp": 0.12, "h_dia_bp": 0.04, "h_total_cholesterol": 0.14}
    
    res = compare_explanation_drift(shap_src, shap_tgt, perm_src, perm_tgt, BIOCHEMICAL_FEATURES)
    
    assert "shap_rank_correlation" in res
    assert "shap_top_k_agreement" in res
    assert "stable_risk_factors" in res

def test_clinical_consistency(sample_data):
    df_src, _ = sample_data
    probs = np.random.uniform(0.1, 0.9, len(df_src))
    
    res = evaluate_clinical_consistency(df_src, probs, BIOCHEMICAL_FEATURES)
    
    assert "feature_consistency" in res
    assert "consistency_rate" in res
    assert 0.0 <= res["consistency_rate"] <= 1.0

def test_trustworthiness_profile():
    pred = {"roc_auc": 0.88, "pr_auc": 0.81, "f1_score": 0.79}
    cal = {"ece": 0.02, "mce": 0.05, "brier": 0.11, "slope": 0.95, "intercept": 0.01}
    exp = {"shap_rank_correlation": 0.85, "perm_rank_correlation": 0.80, "shap_top_k_agreement": 1.0}
    gap = {"roc_auc_drop": 0.03, "ece_increase": 0.01}
    const = {"consistency_rate": 1.0, "n_consistent": 3, "n_audited": 3}
    
    res = build_trustworthiness_profile(pred, cal, exp, 0.92, gap, const, 0.005)
    
    assert res["prediction_reliability"]["grade"] == "Excellent (Clinical-Grade)"
    assert res["calibration_reliability"]["grade"] == "High Calibration (Tight Alignment)"
    assert res["explanation_reliability"]["grade"] == "Highly Reliable (Consistent Features)"
