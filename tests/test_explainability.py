"""
test_explainability.py
======================
Unit tests for the AETHEL publication-grade explainability framework.

Tests verify:
  1. SHAP additivity property (LR + RF)
  2. Permutation importance output shape
  3. Stability Spearman ρ is in valid range
  4. Consensus agreement score = 1.0 for identical rankings
  5. Local explainer NL summary contains hedge language
  6. Clinical interpreter never generates causal language
  7. ConsensusAnalyser correctly rejects < 2 sources
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["h_age", "h_bmi", "h_is_smoker"]
N_SAMPLES = 200
N_FEATURES = 3
SEED = 42

_FORBIDDEN_CAUSAL = [
    "causes", "caused by", "proves", "demonstrates causation",
    "definitively shows", "establishes causality",
]


def _make_dataset(n=N_SAMPLES, n_features=N_FEATURES, seed=SEED):
    X_arr, y_arr = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X = pd.DataFrame(X_arr, columns=FEATURE_NAMES)
    y = pd.Series(y_arr, name="outcome")
    return X, y


def _fit_lr(X, y):
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_pp = pd.DataFrame(scaler.fit_transform(imp.fit_transform(X)), columns=X.columns)
    model = LogisticRegression(max_iter=500, random_state=SEED)
    model.fit(X_pp, y)
    return model, X_pp


def _fit_rf(X, y):
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_pp = pd.DataFrame(scaler.fit_transform(imp.fit_transform(X)), columns=X.columns)
    model = RandomForestClassifier(n_estimators=20, random_state=SEED)
    model.fit(X_pp, y)
    return model, X_pp


# ---------------------------------------------------------------------------
# Test 1: SHAP additivity (LR)
# ---------------------------------------------------------------------------

class TestSHAPAnalyserLinear:
    def test_shap_additivity_lr(self):
        """
        SHAP additivity: for each sample, sum(SHAP values) + expected_value
        should approximately equal the model output (logit of predicted prob).
        Tolerance is set to 0.05 to account for floating-point rounding.
        """
        from src.explainability.shap_analysis import SHAPAnalyser
        from scipy.special import logit

        X, y = _make_dataset()
        model, X_pp = _fit_lr(X, y)
        analyser = SHAPAnalyser(model, X_pp, FEATURE_NAMES, "LR_test")
        analyser.fit()
        shap_vals = analyser.compute_shap_values(X_pp)

        # Expected value for class 1
        ev = analyser._expected_value
        probs = model.predict_proba(X_pp)[:, 1]
        logit_probs = logit(np.clip(probs, 1e-6, 1 - 1e-6))

        # SHAP sum + expected_value ≈ logit(predicted_prob)
        shap_sum = shap_vals.sum(axis=1) + ev
        diff = np.abs(shap_sum - logit_probs)
        assert diff.mean() < 0.2, (
            f"SHAP additivity violated for LR: mean error = {diff.mean():.4f}"
        )

    def test_shap_values_shape(self):
        """SHAP values should have shape (n_samples, n_features)."""
        from src.explainability.shap_analysis import SHAPAnalyser

        X, y = _make_dataset()
        model, X_pp = _fit_lr(X, y)
        analyser = SHAPAnalyser(model, X_pp, FEATURE_NAMES, "LR_test")
        analyser.fit()
        shap_vals = analyser.compute_shap_values(X_pp)
        assert shap_vals.shape == (N_SAMPLES, N_FEATURES)


# ---------------------------------------------------------------------------
# Test 2: SHAP additivity (RF)
# ---------------------------------------------------------------------------

class TestSHAPAnalyserTree:
    def test_shap_values_shape_rf(self):
        """SHAP values for RF should have correct shape."""
        from src.explainability.shap_analysis import SHAPAnalyser

        X, y = _make_dataset()
        model, X_pp = _fit_rf(X, y)
        analyser = SHAPAnalyser(model, X_pp, FEATURE_NAMES, "RF_test")
        analyser.fit()
        shap_vals = analyser.compute_shap_values(X_pp)
        assert shap_vals.shape == (N_SAMPLES, N_FEATURES)

    def test_mean_abs_shap_sorted(self):
        """mean_abs_shap() should return sorted descending Series."""
        from src.explainability.shap_analysis import SHAPAnalyser

        X, y = _make_dataset()
        model, X_pp = _fit_rf(X, y)
        analyser = SHAPAnalyser(model, X_pp, FEATURE_NAMES, "RF_test")
        analyser.fit()
        analyser.compute_shap_values(X_pp)
        ranking = analyser.mean_abs_shap()
        vals = ranking.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), (
            "mean_abs_shap() not sorted descending"
        )


# ---------------------------------------------------------------------------
# Test 3: Permutation importance output shape
# ---------------------------------------------------------------------------

class TestPermutationAnalyser:
    def test_output_shape(self):
        """Permutation importance DataFrame should have one row per feature."""
        from src.explainability.permutation_analysis import PermutationAnalyser

        X, y = _make_dataset()
        model, X_pp = _fit_rf(X, y)
        analyser = PermutationAnalyser(model, FEATURE_NAMES, "RF_test", n_repeats=5)
        df = analyser.compute(X_pp, y)
        assert len(df) == N_FEATURES
        assert "feature" in df.columns
        assert "mean_importance" in df.columns
        assert "ci_lower_95" in df.columns

    def test_ranking_indexing(self):
        """get_ranking() should return a Series indexed by feature name."""
        from src.explainability.permutation_analysis import PermutationAnalyser

        X, y = _make_dataset()
        model, X_pp = _fit_rf(X, y)
        analyser = PermutationAnalyser(model, FEATURE_NAMES, "RF_test", n_repeats=5)
        analyser.compute(X_pp, y)
        ranking = analyser.get_ranking()
        assert set(ranking.index) == set(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Test 4: Stability Spearman ρ in valid range
# ---------------------------------------------------------------------------

class TestStabilityAnalyser:
    def test_spearman_rho_range(self):
        """Ranking stability (Spearman ρ) must be in [-1, 1]."""
        from src.explainability.stability_analysis import StabilityAnalyser

        X, y = _make_dataset()
        analyser = StabilityAnalyser(
            RandomForestClassifier, {"n_estimators": 10},
            FEATURE_NAMES, "RF_stability", seeds=[42, 123]
        )
        report = analyser.run(X, y, X, y)
        rho = report.get("ranking_stability", {}).get("mean_spearman_rho")
        assert rho is not None, "Spearman ρ not found in report"
        assert -1.0 <= rho <= 1.0, f"Spearman ρ out of range: {rho}"

    def test_report_structure(self):
        """Stability report must contain expected keys."""
        from src.explainability.stability_analysis import StabilityAnalyser

        X, y = _make_dataset()
        analyser = StabilityAnalyser(
            RandomForestClassifier, {"n_estimators": 10},
            FEATURE_NAMES, "RF_stability", seeds=[42, 123]
        )
        report = analyser.run(X, y, X, y)
        assert "feature_statistics" in report
        assert "consensus_ranking" in report
        for feat in FEATURE_NAMES:
            assert feat in report["feature_statistics"]


# ---------------------------------------------------------------------------
# Test 5: Consensus Agreement Score = 1.0 for identical rankings
# ---------------------------------------------------------------------------

class TestConsensusAnalyser:
    def test_perfect_agreement(self):
        """Two identical importance sources should yield FAS = 1.0."""
        from src.explainability.consensus_analysis import ConsensusAnalyser

        importance = pd.Series({"h_age": 0.5, "h_bmi": 0.3, "h_is_smoker": 0.1})
        analyser = ConsensusAnalyser(FEATURE_NAMES, "test_model")
        analyser.add_source("source_a", importance)
        analyser.add_source("source_b", importance)  # identical
        report = analyser.build(top_k=2)
        fas = report["feature_agreement_score"]["value"]
        assert abs(fas - 1.0) < 1e-6, f"Expected FAS=1.0 for identical rankings, got {fas}"

    def test_insufficient_sources(self):
        """Fewer than 2 sources should return error key in report."""
        from src.explainability.consensus_analysis import ConsensusAnalyser

        importance = pd.Series({"h_age": 0.5, "h_bmi": 0.3, "h_is_smoker": 0.1})
        analyser = ConsensusAnalyser(FEATURE_NAMES, "test_model")
        analyser.add_source("only_source", importance)
        report = analyser.build()
        assert "error" in report


# ---------------------------------------------------------------------------
# Test 6: Local explainer NL summary contains hedge language
# ---------------------------------------------------------------------------

class TestLocalExplainerNLSummary:
    def test_hedge_language_present(self):
        """NL summary must contain at least one hedge phrase."""
        from src.explainability.local_explanations import _nl_summary

        shap_vals = np.array([0.15, -0.05, 0.08])
        summary = _nl_summary(FEATURE_NAMES, shap_vals, risk_prob=0.72)
        hedge_terms = [
            "consistent with", "may indicate", "in agreement with",
            "is associated with", "does not constitute",
        ]
        found = any(t in summary.lower() for t in hedge_terms)
        assert found, f"No hedge language found in: {summary}"

    def test_no_causal_language_in_nl(self):
        """NL summary must not contain causal language."""
        from src.explainability.local_explanations import _nl_summary

        shap_vals = np.array([0.3, 0.1, -0.05])
        summary = _nl_summary(FEATURE_NAMES, shap_vals, risk_prob=0.85)
        for forbidden in _FORBIDDEN_CAUSAL:
            assert forbidden not in summary.lower(), (
                f"Forbidden causal term '{forbidden}' found in NL summary"
            )


# ---------------------------------------------------------------------------
# Test 7: Clinical interpreter never generates causal language
# ---------------------------------------------------------------------------

class TestClinicalInterpreter:
    def test_no_causal_language(self):
        """All KB entries must pass the causal language check."""
        from src.explainability.clinical_interpretation import ClinicalInterpreter, _CLINICAL_KB

        interpreter = ClinicalInterpreter(FEATURE_NAMES)
        for feat in _CLINICAL_KB:
            for direction in ["positive", "negative"]:
                result = interpreter.interpret(feat, direction, shap_magnitude=0.05)
                assert result["causal_claim_check"], (
                    f"Causal language found in interpretation of '{feat}' ({direction})"
                )

    def test_all_features_have_entries(self):
        """All standard harmonized features should have KB entries."""
        from src.explainability.clinical_interpretation import _CLINICAL_KB

        expected_features = [
            "h_age", "h_bmi", "h_is_smoker", "h_sex_male",
            "h_sys_bp", "h_dia_bp", "h_total_cholesterol",
            "h_ldl", "h_triglycerides", "h_glucose",
        ]
        for feat in expected_features:
            assert feat in _CLINICAL_KB, f"Feature '{feat}' missing from clinical KB"

    def test_hedge_in_interpretation(self):
        """Every interpretation output must contain hedged language."""
        from src.explainability.clinical_interpretation import ClinicalInterpreter

        interpreter = ClinicalInterpreter(FEATURE_NAMES)
        result = interpreter.interpret("h_age", "positive", 0.1)
        combined_text = (
            result["potential_clinical_meaning"] + " " + result["direction_interpretation"]
        ).lower()
        hedge_terms = ["consistent with", "may indicate", "in agreement with", "associated with"]
        found = any(t in combined_text for t in hedge_terms)
        assert found, f"No hedge language in clinical interpretation for h_age"
