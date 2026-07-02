"""
evaluator.py
============
Publication-grade evaluation framework for clinical machine learning models.
Implements primary, clinical, and calibration metrics, bootstrapping,
leakage-free cross-validation, statistical testing, and error analysis.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import wilcoxon
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Concordance / Harrell's C-Index (from scratch)
# ---------------------------------------------------------------------------

def harrell_c_index(time_col: np.ndarray, event_col: np.ndarray, risk_score: np.ndarray) -> float:
    """
    Compute Harrell's Concordance Index from scratch to handle survival censoring.
    If censoring is absent, this is mathematically equivalent to binary ROC-AUC.
    """
    time_col = np.asarray(time_col)
    event_col = np.asarray(event_col)
    risk_score = np.asarray(risk_score)
    
    n = len(time_col)
    if n <= 1:
        return 0.5
        
    i_idx, j_idx = np.triu_indices(n, k=1)
    t1, t2 = time_col[i_idx], time_col[j_idx]
    e1, e2 = event_col[i_idx], event_col[j_idx]
    r1, r2 = risk_score[i_idx], risk_score[j_idx]
    
    # Case A: t1 < t2 and e1 == 1
    mask_a = (t1 < t2) & (e1 == 1)
    # Case B: t2 < t1 and e2 == 1
    mask_b = (t2 < t1) & (e2 == 1)
    
    r_first = np.concatenate([r1[mask_a], r2[mask_b]])
    r_second = np.concatenate([r2[mask_a], r1[mask_b]])
    
    usable_pairs = len(r_first)
    if usable_pairs == 0:
        return 0.5
        
    concordant = np.sum(r_first > r_second)
    discordant = np.sum(r_first < r_second)
    tied_risk = np.sum(r_first == r_second)
    
    return (concordant + 0.5 * tied_risk) / usable_pairs


# ---------------------------------------------------------------------------
# Calibration Metrics
# ---------------------------------------------------------------------------

def compute_calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Calculates the Calibration Intercept and Slope via Logistic Calibration.
    logit(y) = alpha + beta * logit(y_prob)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Clip to avoid infinity in logit
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = logit(y_prob_clipped)
    
    try:
        clf = LogisticRegression(C=1e9, solver="lbfgs", tol=1e-5)
        clf.fit(logits.reshape(-1, 1), y_true)
        intercept = float(clf.intercept_[0])
        slope = float(clf.coef_[0][0])
    except Exception as e:
        logger.debug("Calibration fit failed: %s. Using fallback.", str(e))
        intercept, slope = np.nan, np.nan
    return intercept, slope


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    calculated over bin partitions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_diff = np.abs(bin_acc - bin_conf)
            
            ece += (bin_size / n_samples) * bin_diff
            mce = max(mce, bin_diff)
            
    return ece, mce


# ---------------------------------------------------------------------------
# Core Metrics Calculator
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: pd.Series | np.ndarray, 
                      y_prob: pd.Series | np.ndarray, 
                      time_col: Optional[pd.Series | np.ndarray] = None, 
                      event_col: Optional[pd.Series | np.ndarray] = None) -> dict[str, float]:
    """
    Compute primary, clinical, calibration, and risk metrics for a model.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Primary Metrics
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        total = int(tn + fp + fn + tp)
        acc = float(tp + tn) / total if total > 0 else 0.0
        prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = float(tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        
        sens = rec
        spec = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = prec
        npv = float(tn) / (tn + fn) if (tn + fn) > 0 else 0.0
    else:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        sens = rec
        spec = np.nan
        ppv = prec
        npv = np.nan
        
    # Calibration Metrics
    brier = brier_score_loss(y_true, y_prob)
    cal_intercept, cal_slope = compute_calibration_intercept_slope(y_true, y_prob)
    ece, mce = expected_calibration_error(y_true, y_prob)
    
    # Risk Metrics (C-index)
    if time_col is not None and event_col is not None:
        c_index = harrell_c_index(time_col, event_col, y_prob)
    else:
        c_index = auc  # Binary equivalent

    return {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "brier_score": brier,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "ece": ece,
        "mce": mce,
        "c_index": c_index
    }


# ---------------------------------------------------------------------------
# Bootstrapping Engine
# ---------------------------------------------------------------------------

class BootstrapEvaluator:
    """
    Computes 95% Confidence Intervals for clinical metrics using 1000 bootstrap runs.
    """
    def __init__(self, n_bootstrap: int = 1000, seed: int = 42) -> None:
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray, 
                 time_col: Optional[np.ndarray] = None, 
                 event_col: Optional[np.ndarray] = None) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        if time_col is not None:
            time_col = np.asarray(time_col)
        if event_col is not None:
            event_col = np.asarray(event_col)
            
        rng = np.random.default_rng(self.seed)
        n_samples = len(y_true)
        
        metrics_list = []
        for _ in range(self.n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_true = y_true[indices]
            boot_prob = y_prob[indices]
            
            boot_time = time_col[indices] if time_col is not None else None
            boot_event = event_col[indices] if event_col is not None else None
            
            # Avoid bootstrap failure if single class
            if len(np.unique(boot_true)) < 2:
                continue
                
            try:
                metrics = calculate_metrics(boot_true, boot_prob, boot_time, boot_event)
                metrics_list.append(metrics)
            except Exception:
                continue
                
        df_boot = pd.DataFrame(metrics_list)
        
        # Calculate CIs (2.5th and 97.5th percentiles)
        ci_results = {}
        for col in df_boot.columns:
            lower = np.percentile(df_boot[col].dropna(), 2.5)
            upper = np.percentile(df_boot[col].dropna(), 97.5)
            ci_results[col] = (lower, upper)
            
        return ci_results, df_boot


# ---------------------------------------------------------------------------
# Leakage-Free Cross Validation Loop
# ---------------------------------------------------------------------------

class LeakageFreeCV:
    """
    Executes a 5x10-fold Repeated Stratified Cross-Validation.
    Performs scaling and feature selection inside each fold.
    """
    def __init__(self, n_splits: int = 10, n_repeats: int = 5, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed

    def run(self, X: pd.DataFrame, y: pd.Series, model_class, model_kwargs: dict) -> dict[str, Any]:
        rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed)
        
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 1. Imputation
            imputer = SimpleImputer(strategy="median")
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
                X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
                
            # 2. Scaling
            scaler = RobustScaler()
            if len(numeric_cols) > 0:
                X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
                
            # 3. Correlation-based Feature Selection (Inside fold)
            corr_matrix = X_train[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
            
            X_train_sel = X_train.drop(columns=to_drop)
            X_val_sel = X_val.drop(columns=to_drop)
            
            # Train and predict
            clf = model_class(**model_kwargs)
            clf.fit(X_train_sel, y_train)
            
            y_prob = clf.predict_proba(X_val_sel)[:, 1]
            
            metrics = calculate_metrics(y_val, y_prob)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)
            
        df_cv = pd.DataFrame(fold_metrics)
        
        summary = {}
        for col in df_cv.columns:
            if col == "fold":
                continue
            summary[col] = {
                "mean": df_cv[col].mean(),
                "std": df_cv[col].std()
            }
        return {"summary": summary, "raw_folds": df_cv}


# ---------------------------------------------------------------------------
# Statistical Significance Testing
# ---------------------------------------------------------------------------

def run_paired_bootstrap_test(y_true: np.ndarray, y_prob1: np.ndarray, y_prob2: np.ndarray, 
                              n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """
    Performs a paired bootstrap test to compare the ROC-AUC of two models.
    Returns: (auc1_mean, auc2_mean, p_value)
    """
    y_true = np.asarray(y_true)
    y_prob1 = np.asarray(y_prob1)
    y_prob2 = np.asarray(y_prob2)
    
    n_samples = len(y_true)
    rng = np.random.default_rng(seed)
    
    diffs = []
    auc1s = []
    auc2s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc1 = roc_auc_score(y_true[idx], y_prob1[idx])
        auc2 = roc_auc_score(y_true[idx], y_prob2[idx])
        auc1s.append(auc1)
        auc2s.append(auc2)
        diffs.append(auc1 - auc2)
        
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # Calculate p-value: proportion of bootstrap differences that cross zero
    # (two-tailed check against zero-centered hypothesis)
    if mean_diff > 0:
        p_val = 2.0 * np.mean(diffs <= 0)
    else:
        p_val = 2.0 * np.mean(diffs >= 0)
        
    p_val = min(p_val, 1.0)
    return np.mean(auc1s), np.mean(auc2s), p_val


def run_mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> tuple[float, float]:
    """
    Executes McNemar's test for binary classification predictions.
    """
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    tb = np.zeros((2, 2), dtype=int)
    tb[0, 0] = np.sum(correct1 & correct2)
    tb[0, 1] = np.sum(correct1 & ~correct2)
    tb[1, 0] = np.sum(~correct1 & correct2)
    tb[1, 1] = np.sum(~correct1 & ~correct2)
    
    res = mcnemar(tb, exact=True)
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------

def perform_error_analysis(y_true: pd.Series | np.ndarray, 
                           y_prob: pd.Series | np.ndarray, 
                           df_features: pd.DataFrame) -> dict[str, Any]:
    """
    Identifies False Positives, False Negatives, and parses High vs Low confidence errors.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    df_err = df_features.copy()
    df_err["y_true"] = y_true
    df_err["y_prob"] = y_prob
    df_err["y_pred"] = y_pred
    df_err["error_type"] = "Correct"
    
    df_err.loc[(y_true == 0) & (y_pred == 1), "error_type"] = "False Positive"
    df_err.loc[(y_true == 1) & (y_pred == 0), "error_type"] = "False Negative"
    
    # High confidence errors: predicted probability far from truth
    df_err["confidence"] = np.where(y_pred == 1, y_prob, 1.0 - y_prob)
    df_err["is_high_conf_error"] = (df_err["error_type"] != "Correct") & (df_err["confidence"] >= 0.80)
    df_err["is_low_conf_error"] = (df_err["error_type"] != "Correct") & (df_err["confidence"] < 0.60)
    
    fp_summary = df_err[df_err["error_type"] == "False Positive"].mean(numeric_only=True).to_dict()
    fn_summary = df_err[df_err["error_type"] == "False Negative"].mean(numeric_only=True).to_dict()
    
    return {
        "n_false_positives": int(np.sum(df_err["error_type"] == "False Positive")),
        "n_false_negatives": int(np.sum(df_err["error_type"] == "False Negative")),
        "n_high_confidence_errors": int(np.sum(df_err["is_high_conf_error"])),
        "n_low_confidence_errors": int(np.sum(df_err["is_low_conf_error"])),
        "fp_feature_averages": fp_summary,
        "fn_feature_averages": fn_summary,
        "raw_errors": df_err[df_err["error_type"] != "Correct"]
    }
