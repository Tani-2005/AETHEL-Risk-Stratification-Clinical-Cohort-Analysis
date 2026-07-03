"""
test_scientific_refinements.py
==============================
Unit tests for the new calibration scaling, DCA, and survival evaluation features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import calculate_metrics, run_dca
from src.calibration.recalibration import (
    fit_platt_scaling, predict_platt_scaling,
    fit_isotonic_regression, predict_isotonic_regression,
    compare_recalibration
)

def test_calculate_metrics_survival() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    time_col = np.array([2.0, 2.0, 1.0, 1.0])
    event_col = np.array([0, 0, 1, 1])
    
    metrics = calculate_metrics(y_true, y_prob, time_col=time_col, event_col=event_col, is_survival=True)
    
    # Classification metrics must be NaN
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["accuracy"])
    assert np.isnan(metrics["f1"])
    assert np.isnan(metrics["ece"])
    
    # C-index must be computed and equal to 1.0
    assert abs(metrics["c_index"] - 1.0) < 1e-6

def test_decision_curve_analysis() -> None:
    y_true = np.array([0, 0, 1, 1] * 10)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9] * 10)
    
    dca_df = run_dca(y_true, y_prob)
    
    assert isinstance(dca_df, pd.DataFrame)
    assert "threshold" in dca_df.columns
    assert "net_benefit_model" in dca_df.columns
    assert "net_benefit_treat_all" in dca_df.columns
    assert "net_benefit_treat_none" in dca_df.columns
    assert len(dca_df) == 99

def test_recalibration_comparison() -> None:
    # Set seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Generate some miscalibrated predictions: model systematically overestimates risk
    y_true_train = rng.choice([0, 1], size=100, p=[0.7, 0.3])
    y_prob_train = rng.uniform(0.4, 0.9, size=100)  # High predicted risks
    
    y_true_test = rng.choice([0, 1], size=100, p=[0.7, 0.3])
    y_prob_test = rng.uniform(0.4, 0.9, size=100)
    
    # Platt scaling calibration
    platt_model = fit_platt_scaling(y_true_train, y_prob_train)
    y_prob_platt = predict_platt_scaling(platt_model, y_prob_test)
    
    # Isotonic regression calibration
    isotonic_model = fit_isotonic_regression(y_true_train, y_prob_train)
    y_prob_iso = predict_isotonic_regression(isotonic_model, y_prob_test)
    
    # Compare ECE
    comp = compare_recalibration(y_true_train, y_prob_train, y_true_test, y_prob_test)
    
    assert "ece_uncalibrated" in comp
    assert "ece_platt_scaling" in comp
    assert "ece_isotonic_regression" in comp
    
    # Recalibrated ECEs should typically be lower than or equal to uncalibrated ECE
    assert comp["ece_platt_scaling"] <= comp["ece_uncalibrated"] or abs(comp["ece_platt_scaling"] - comp["ece_uncalibrated"]) < 0.05
    assert comp["ece_isotonic_regression"] <= comp["ece_uncalibrated"] or abs(comp["ece_isotonic_regression"] - comp["ece_uncalibrated"]) < 0.05
