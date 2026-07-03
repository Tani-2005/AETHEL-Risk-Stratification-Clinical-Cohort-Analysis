"""
recalibration.py
================
Post-hoc calibration methods (Platt Scaling and Isotonic Regression)
to correct calibration drift in clinical machine learning models under domain shift.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from src.evaluation.evaluator import expected_calibration_error

def fit_platt_scaling(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """
    Fits a Platt scaling (logistic calibration) model.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
    
    lr = LogisticRegression(C=1e9)  # Unregularized logistic regression
    lr.fit(logits, y_true)
    return lr

def predict_platt_scaling(model: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    """
    Applies a fitted Platt scaling model to new predictions.
    """
    y_prob = np.asarray(y_prob)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]

def fit_isotonic_regression(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """
    Fits an Isotonic regression calibration model.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    return ir

def predict_isotonic_regression(model: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    """
    Applies a fitted Isotonic regression model to new predictions.
    """
    y_prob = np.asarray(y_prob)
    return model.transform(y_prob)

def compare_recalibration(
    y_true_train: np.ndarray,
    y_prob_train: np.ndarray,
    y_true_test: np.ndarray,
    y_prob_test: np.ndarray,
) -> dict[str, float]:
    """
    Compares ECE before and after Platt and Isotonic recalibration.
    The calibration models are fit on (y_prob_train, y_true_train)
    and evaluated on (y_prob_test, y_true_test).
    """
    y_true_train = np.asarray(y_true_train)
    y_prob_train = np.asarray(y_prob_train)
    y_true_test = np.asarray(y_true_test)
    y_prob_test = np.asarray(y_prob_test)
    
    # Baseline
    ece_base, _ = expected_calibration_error(y_true_test, y_prob_test)
    
    # Platt
    platt_model = fit_platt_scaling(y_true_train, y_prob_train)
    y_prob_platt = predict_platt_scaling(platt_model, y_prob_test)
    ece_platt, _ = expected_calibration_error(y_true_test, y_prob_platt)
    
    # Isotonic
    isotonic_model = fit_isotonic_regression(y_true_train, y_prob_train)
    y_prob_iso = predict_isotonic_regression(isotonic_model, y_prob_test)
    ece_iso, _ = expected_calibration_error(y_true_test, y_prob_iso)
    
    return {
        "ece_uncalibrated": float(ece_base),
        "ece_platt_scaling": float(ece_platt),
        "ece_isotonic_regression": float(ece_iso)
    }
