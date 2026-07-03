"""
test_robustness.py
==================
Unit tests for the AETHEL robustness and stability framework.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.robustness.stability_metrics import (
    compute_prediction_stability,
    compute_probability_stability,
    compute_ranking_stability,
    compute_explanation_stability,
)
from src.robustness.repeated_runs import run_repeated_experiments
from src.robustness.bootstrap import run_bootstrap_analysis
from src.robustness.feature_ablation import (
    run_hierarchical_ablation,
    run_individual_ablation,
    get_feature_group,
)
from src.robustness.noise_analysis import (
    inject_gaussian_noise,
    inject_binary_flips,
    run_noise_robustness,
)
from src.robustness.missing_data_analysis import run_missing_data_robustness
from src.robustness.distribution_shift import (
    calculate_psi,
    analyze_covariate_shift,
)
from src.robustness.uncertainty import estimate_uncertainty
from src.robustness.robustness_reports import calculate_robustness_score

@pytest.fixture
def mock_data():
    """Generates a small mock dataset for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "h_age": np.random.uniform(30, 80, size=n),
        "h_bmi": np.random.uniform(18, 35, size=n),
        "h_is_smoker": np.random.choice([0, 1], p=[0.7, 0.3], size=n),
        "h_outcome_binary": np.random.choice([0, 1], p=[0.8, 0.2], size=n),
    })
    return df

def test_stability_metrics():
    """Verifies that stability calculations run and return values in expected ranges."""
    np.random.seed(42)
    
    # 3 runs, 10 samples
    preds = np.array([
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    ])
    probs = np.array([
        [0.8, 0.1, 0.9, 0.7, 0.2, 0.1, 0.8, 0.2, 0.3, 0.9],
        [0.8, 0.1, 0.9, 0.4, 0.2, 0.1, 0.8, 0.2, 0.3, 0.9],
        [0.8, 0.1, 0.9, 0.7, 0.2, 0.1, 0.8, 0.2, 0.6, 0.9],
    ])
    
    pred_stab = compute_prediction_stability(preds)
    assert 0.0 <= pred_stab["jaccard_stability"] <= 1.0
    assert 0.0 <= pred_stab["prediction_agreement"] <= 1.0
    
    prob_stab = compute_probability_stability(probs)
    assert 0.0 <= prob_stab["probability_correlation"] <= 1.0
    assert prob_stab["probability_mse"] >= 0.0
    
    # Rankings (3 runs, 4 features)
    ranks = np.array([
        [0.5, 0.3, 0.2, 0.0],
        [0.5, 0.3, 0.2, 0.0],
        [0.5, 0.2, 0.3, 0.0],
    ])
    rank_stab = compute_ranking_stability(ranks)
    assert 0.0 <= rank_stab["ranking_stability_rho"] <= 1.0
    
    # SHAP (3 runs, 10 samples, 4 features)
    shap_vals = np.random.normal(size=(3, 10, 4))
    exp_stab = compute_explanation_stability(shap_vals)
    assert -1.0 <= exp_stab["explanation_stability_rho"] <= 1.0

def test_bootstrap_analysis():
    """Verifies bootstrap resampler calculates distributions and 95% CIs."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    y_prob = np.random.uniform(0, 1, size=100)
    
    res = run_bootstrap_analysis(y_true, y_prob, n_iterations=20, random_state=42)
    assert "bootstrap_stats" in res
    assert "distributions" in res
    assert "roc_auc" in res["bootstrap_stats"]
    assert "ci_lower" in res["bootstrap_stats"]["roc_auc"]
    assert len(res["distributions"]["roc_auc"]) == 20

def test_feature_ablation(mock_data):
    """Verifies hierarchical and individual feature ablation retrains and tracks drops."""
    X_train = mock_data.iloc[:70]
    y_train = X_train["h_outcome_binary"]
    X_val = mock_data.iloc[70:]
    y_val = X_val["h_outcome_binary"]
    features = ["h_age", "h_bmi", "h_is_smoker"]
    
    # Baseline
    model = LogisticRegression()
    model.fit(X_train[features], y_train)
    probs = model.predict_proba(X_val[features])[:, 1]
    from src.evaluation.evaluator import calculate_metrics
    metrics = calculate_metrics(y_val, probs)
    
    hier = run_hierarchical_ablation(
        LogisticRegression, {}, X_train, y_train, X_val, y_val, features, metrics, probs
    )
    assert len(hier) > 0
    assert "Demographics" in hier or "Clinical" in hier or "Lifestyle" in hier
    
    ind = run_individual_ablation(
        LogisticRegression, {}, X_train, y_train, X_val, y_val, features, metrics, probs
    )
    assert len(ind) == len(features)
    assert "h_age" in ind
    assert "auc_drop" in ind["h_age"]

def test_noise_injection(mock_data):
    """Verifies Gaussian noise and binary flips are correctly injected."""
    features = ["h_age", "h_bmi", "h_is_smoker"]
    
    # Gaussian noise injection
    df_noisy = inject_gaussian_noise(mock_data, features, 0.5, ["h_age", "h_bmi"])
    assert not df_noisy.equals(mock_data)
    assert df_noisy["h_is_smoker"].equals(mock_data["h_is_smoker"]) # untouched binary
    
    # Binary flip injection
    df_flipped = inject_binary_flips(mock_data, features, 0.5, ["h_is_smoker"])
    assert not df_flipped["h_is_smoker"].equals(mock_data["h_is_smoker"])
    assert df_flipped["h_age"].equals(mock_data["h_age"]) # untouched continuous

def test_missing_data_robustness(mock_data):
    """Verifies MCAR missingness injection and imputer/scaler integration."""
    X_train = mock_data.iloc[:70]
    features = ["h_age", "h_bmi", "h_is_smoker"]
    
    imp = SimpleImputer(strategy="median").fit(X_train[features])
    scaler = RobustScaler().fit(imp.transform(X_train[features]))
    
    model = LogisticRegression()
    X_tr_pp = scaler.transform(imp.transform(X_train[features]))
    model.fit(X_tr_pp, X_train["h_outcome_binary"])
    
    X_val = mock_data.iloc[70:]
    y_val = X_val["h_outcome_binary"]
    
    res = run_missing_data_robustness(model, X_val, y_val, features, imp, scaler, missing_rates=[0.0, 0.1, 0.2])
    assert 0.0 in res["sweep_results"]
    assert 0.1 in res["sweep_results"]
    assert "metrics" in res["sweep_results"][0.1]

def test_distribution_shift():
    """Verifies PSI and Wasserstein distance calculations."""
    np.random.seed(42)
    expected = np.random.normal(0, 1, size=100)
    actual = np.random.normal(0.5, 1.2, size=100)
    
    psi = calculate_psi(expected, actual, num_bins=10)
    assert psi >= 0.0
    
    df_src = pd.DataFrame({"f1": expected})
    df_tgt = pd.DataFrame({"f1": actual})
    
    shift = analyze_covariate_shift(df_src, df_tgt, ["f1"])
    assert "f1" in shift
    assert "psi" in shift["f1"]
    assert "wasserstein_distance" in shift["f1"]

def test_uncertainty_estimation():
    """Verifies entropy, run variance, and failure flags are correctly calculated."""
    np.random.seed(42)
    # 5 runs, 20 samples
    probs = np.random.uniform(0.1, 0.9, size=(5, 20))
    y_true = np.random.choice([0, 1], size=20)
    
    res = estimate_uncertainty(probs, y_true)
    assert "uncertainty_summary" in res
    assert "failure_report" in res
    assert "mean_prob" in res["uncertainty_summary"].columns
    assert "entropy" in res["uncertainty_summary"].columns
    assert "n_hard_cases" in res["failure_report"]

def test_robustness_scoring():
    """Verifies that the overall robustness index is computed properly."""
    res = calculate_robustness_score(
        prediction_stability=0.95,
        probability_stability=0.92,
        explanation_stability=0.88,
        feature_stability=0.90,
        noise_retention=0.85,
        missing_data_retention=0.80,
    )
    assert "overall_score" in res
    assert 0.80 <= res["overall_score"] <= 0.95
    assert "grade" in res
    assert res["grade"] == "High Robustness (Clinical-Grade)"

