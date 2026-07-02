"""
test_evaluator.py
=================
Unit tests for the publication-grade model evaluation framework.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.evaluator import (
    harrell_c_index, compute_calibration_intercept_slope,
    expected_calibration_error, calculate_metrics,
    BootstrapEvaluator, LeakageFreeCV
)


def test_harrell_c_index_perfect_concordance() -> None:
    # Perfect concordance: risk scores are perfectly ordered with events
    time_col = np.array([1.0, 2.0, 3.0, 4.0])
    event_col = np.array([1, 1, 1, 0])  # Censored at 4.0
    risk_score = np.array([4.0, 3.0, 2.0, 1.0])  # Earlier event -> higher risk
    
    # Pairs check:
    # (1, 2): usable (both events). t[1] < t[2] -> 1 should have higher risk. 4.0 > 3.0 (concordant)
    # (1, 3): usable (both events). t[1] < t[3] -> 1 should have higher risk. 4.0 > 2.0 (concordant)
    # (1, 4): usable. t[1] < t[4] (censored). 1 should have higher risk. 4.0 > 1.0 (concordant)
    # (2, 3): usable (both events). t[2] < t[3] -> 2 should have higher risk. 3.0 > 2.0 (concordant)
    # (2, 4): usable. t[2] < t[4] (censored). 2 should have higher risk. 3.0 > 1.0 (concordant)
    # (3, 4): usable. t[3] < t[4] (censored). 3 should have higher risk. 2.0 > 1.0 (concordant)
    # Total usable: 6. Concordant: 6. C-index = 1.0
    c_index = harrell_c_index(time_col, event_col, risk_score)
    assert abs(c_index - 1.0) < 1e-6


def test_harrell_c_index_binary_equivalent() -> None:
    # Without censoring, Harrell's C-index must equal ROC-AUC
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    
    # Map to time-to-event where event=1 has time=1.0, event=0 has time=2.0 (censored or not)
    # If event=1 occurs at 1.0 and event=0 is censored at 2.0:
    time_col = np.array([2.0, 2.0, 1.0, 1.0])
    event_col = np.array([0, 0, 1, 1])
    
    c_index = harrell_c_index(time_col, event_col, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    assert abs(c_index - auc) < 1e-6


def test_expected_calibration_error() -> None:
    # Perfect calibration: prob matches outcomes exactly in each bin
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    
    ece, mce = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert abs(ece - 0.0) < 1e-6
    assert abs(mce - 0.0) < 1e-6


def test_calibration_intercept_slope() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    
    intercept, slope = compute_calibration_intercept_slope(y_true, y_prob)
    # Should fit successfully
    assert not np.isnan(intercept)
    assert not np.isnan(slope)


def test_bootstrap_evaluator() -> None:
    y_true = np.array([0, 0, 1, 1] * 10)  # n = 40
    y_prob = np.array([0.2, 0.1, 0.8, 0.9] * 10)
    
    evaluator = BootstrapEvaluator(n_bootstrap=10, seed=42)
    cis, df_boot = evaluator.evaluate(y_true, y_prob)
    
    assert "roc_auc" in cis
    assert len(df_boot) == 10
    assert not df_boot["roc_auc"].isna().any()


def test_leakage_free_cv() -> None:
    # Create simple dataset
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=["f1", "f2", "f3", "f4"])
    # Introduce high correlation to trigger VIF/Correlation dropping inside fold
    X["f5"] = X["f1"] * 0.99
    
    y = pd.Series(rng.choice([0, 1], size=100))
    
    cv = LeakageFreeCV(n_splits=3, n_repeats=1, seed=42)
    res = cv.run(X, y, LogisticRegression, {"max_iter": 100})
    
    assert "roc_auc" in res["summary"]
    assert "mean" in res["summary"]["roc_auc"]
    assert len(res["raw_folds"]) == 3
