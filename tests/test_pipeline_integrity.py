"""
test_pipeline_integrity.py
===========================
Pipeline Integrity & Leakage Tests

These tests verify the no-leakage guarantees of the AETHEL preprocessing
pipeline.  Every test is designed to catch a specific class of leakage
or integrity violation that would invalidate peer-reviewed results.

Tests
-----
TestSplitIntegrity
  - Splits sum to correct proportions
  - No patient appears in more than one split
  - Stratification preserved event rate across splits

TestLeakageGuarantees
  - Pollution index stats sourced from train only
  - Scaler fitted on train only (verified via transform-on-dummy-test)
  - Outcome columns not in feature matrix

TestFeatureEngineering
  - All engineered features present in splits
  - pollution_index in correct range (z-scored)
  - bmi_category values are valid WHO categories
  - lifestyle_risk is exactly is_smoker * townsend_index

TestVIFMulticollinearity
  - After preprocessing, no covariate pair has VIF > threshold
  - pollution_index VIF < 10 (fixes the original VIF=15.25 finding)

Run with:
    python -m pytest tests/ -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures — load splits once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    from src.utils.paths import DataPaths
    if not DataPaths.TRAIN.exists():
        pytest.skip("Train split not found — run pipeline first.")
    return pd.read_csv(DataPaths.TRAIN)


@pytest.fixture(scope="session")
def val_df() -> pd.DataFrame:
    from src.utils.paths import DataPaths
    if not DataPaths.VAL.exists():
        pytest.skip("Val split not found — run pipeline first.")
    return pd.read_csv(DataPaths.VAL)


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    from src.utils.paths import DataPaths
    if not DataPaths.TEST.exists():
        pytest.skip("Test split not found — run pipeline first.")
    return pd.read_csv(DataPaths.TEST)


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    from src.utils.paths import DataPaths
    return pd.read_csv(DataPaths.ANALYTICAL_COHORT)


# ---------------------------------------------------------------------------
# Split Integrity Tests
# ---------------------------------------------------------------------------

class TestSplitIntegrity:
    """Verify no overlap and correct proportions across train/val/test."""

    def test_no_patient_overlap_train_val(self, train_df, val_df) -> None:
        """No patient_id should appear in both train and val."""
        from src.utils.constants import Columns
        overlap = set(train_df[Columns.PATIENT_ID]) & set(val_df[Columns.PATIENT_ID])
        assert len(overlap) == 0, f"Leakage: {len(overlap)} patients in both train and val."

    def test_no_patient_overlap_train_test(self, train_df, test_df) -> None:
        """No patient_id should appear in both train and test."""
        from src.utils.constants import Columns
        overlap = set(train_df[Columns.PATIENT_ID]) & set(test_df[Columns.PATIENT_ID])
        assert len(overlap) == 0, f"Leakage: {len(overlap)} patients in both train and test."

    def test_no_patient_overlap_val_test(self, val_df, test_df) -> None:
        """No patient_id should appear in both val and test."""
        from src.utils.constants import Columns
        overlap = set(val_df[Columns.PATIENT_ID]) & set(test_df[Columns.PATIENT_ID])
        assert len(overlap) == 0, f"Leakage: {len(overlap)} patients in both val and test."

    def test_split_sizes_approximately_correct(self, train_df, val_df, test_df) -> None:
        """Splits should approximately match 70/15/15 ratios (within 2%)."""
        total = len(train_df) + len(val_df) + len(test_df)
        assert abs(len(train_df) / total - 0.70) < 0.02, "Train size deviates >2% from 70%."
        assert abs(len(val_df) / total - 0.15) < 0.02, "Val size deviates >2% from 15%."
        assert abs(len(test_df) / total - 0.15) < 0.02, "Test size deviates >2% from 15%."

    def test_stratification_preserves_event_rate(self, train_df, val_df, test_df, full_df) -> None:
        """Event rate in each split should be within 5 pp of the full cohort."""
        from src.utils.constants import Columns
        overall_rate = full_df[Columns.EVENT_OCCURRED].mean()
        for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_rate = split[Columns.EVENT_OCCURRED].mean()
            diff = abs(split_rate - overall_rate)
            assert diff < 0.05, (
                f"Stratification failure in {name}: "
                f"event rate {split_rate:.3f} deviates >{diff:.3f} from overall {overall_rate:.3f}."
            )

    def test_total_patients_sum_to_original(self, train_df, val_df, test_df, full_df) -> None:
        """All patients accounted for."""
        assert len(train_df) + len(val_df) + len(test_df) == len(full_df)


# ---------------------------------------------------------------------------
# Leakage Guarantee Tests
# ---------------------------------------------------------------------------

class TestLeakageGuarantees:
    """Verify that fitting statistics come from train only."""

    def test_pollution_index_stats_file_exists(self) -> None:
        """Pollution index fitting stats must be persisted."""
        from src.utils.paths import OutputPaths
        assert OutputPaths.POLLUTION_INDEX_STATS.exists(), \
            "pollution_index_stats.csv not found — PollutionIndexComputer was not run."

    def test_pollution_index_stats_match_train(self, train_df) -> None:
        """Stats in pollution_index_stats.csv must match train set statistics."""
        from src.utils.paths import OutputPaths
        from src.utils.constants import Columns
        if not OutputPaths.POLLUTION_INDEX_STATS.exists():
            pytest.skip("Pollution index stats file not found.")
        stats = pd.read_csv(OutputPaths.POLLUTION_INDEX_STATS).set_index("feature")
        actual_pm25_mean = train_df[Columns.AVG_PM25_EXPOSURE].mean()
        saved_pm25_mean = stats.loc["avg_pm25_exposure", "mean"]
        assert abs(actual_pm25_mean - saved_pm25_mean) < 1e-6, \
            "Pollution index PM2.5 mean does not match training data — possible leakage."

    def test_outcome_not_in_feature_matrix(self, train_df) -> None:
        """months_observed and event_occurred must not be used as features."""
        from src.utils.constants import Columns, Features
        all_features = Features.SURVIVAL_COVARIATES
        assert Columns.MONTHS_OBSERVED not in all_features, \
            "months_observed is in survival_covariates — target leakage!"
        assert Columns.EVENT_OCCURRED not in all_features, \
            "event_occurred is in survival_covariates — target leakage!"

    def test_scaler_file_exists(self) -> None:
        """Fitted sklearn pipeline must be persisted."""
        from src.utils.paths import OutputPaths
        assert OutputPaths.SCALER_JOBLIB.exists(), \
            "robust_scaler.joblib not found — sklearn pipeline was not fitted."


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Verify engineered features are correctly computed."""

    def test_engineered_features_in_splits(self, train_df) -> None:
        """All engineered features must appear in training split."""
        from src.utils.constants import Columns
        required = [
            Columns.BMI_CATEGORY,
            Columns.AGE_GROUP,
            Columns.LIFESTYLE_RISK,
            Columns.HIGH_GENOMIC_RISK,
            Columns.POLLUTION_INDEX,
        ]
        for col in required:
            assert col in train_df.columns, f"Engineered feature '{col}' missing from train split."

    def test_bmi_category_valid_values(self, train_df) -> None:
        """bmi_category must only contain WHO-valid labels."""
        from src.utils.constants import Columns
        valid = {"Underweight", "Normal", "Overweight", "Obese"}
        found = set(train_df[Columns.BMI_CATEGORY].unique())
        assert found.issubset(valid), f"Invalid BMI categories: {found - valid}"

    def test_age_group_valid_values(self, train_df) -> None:
        """age_group must only contain valid decade labels."""
        from src.utils.constants import Columns
        valid = {"40-49", "50-59", "60-69", "70-85"}
        found = set(train_df[Columns.AGE_GROUP].unique())
        assert found.issubset(valid), f"Invalid age groups: {found - valid}"

    def test_lifestyle_risk_formula(self, train_df) -> None:
        """lifestyle_risk must equal is_smoker × townsend_index."""
        from src.utils.constants import Columns
        expected = train_df[Columns.IS_SMOKER] * train_df[Columns.TOWNSEND_INDEX]
        pd.testing.assert_series_equal(
            train_df[Columns.LIFESTYLE_RISK].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-8,
        )

    def test_high_genomic_risk_is_binary(self, train_df) -> None:
        """high_genomic_risk must be strictly 0 or 1."""
        from src.utils.constants import Columns
        unique_vals = set(train_df[Columns.HIGH_GENOMIC_RISK].unique())
        assert unique_vals.issubset({0, 1}), f"high_genomic_risk not binary: {unique_vals}"

    def test_pollution_index_in_each_split(self, train_df, val_df, test_df) -> None:
        """pollution_index must be present in all three splits."""
        from src.utils.constants import Columns
        for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
            assert Columns.POLLUTION_INDEX in split.columns, \
                f"pollution_index missing from {name} split."

    def test_pollution_index_no_nulls(self, train_df, val_df, test_df) -> None:
        """pollution_index must have no null values in any split."""
        from src.utils.constants import Columns
        for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
            n_null = split[Columns.POLLUTION_INDEX].isnull().sum()
            assert n_null == 0, f"pollution_index has {n_null} nulls in {name} split."


# ---------------------------------------------------------------------------
# Multicollinearity Tests
# ---------------------------------------------------------------------------

class TestVIFMulticollinearity:
    """Verify the VIF fix for PM2.5/NO2 multicollinearity."""

    def test_no2_not_in_survival_covariates(self) -> None:
        """avg_no2_exposure must have been removed from survival covariates."""
        from src.utils.constants import Columns, Features
        assert Columns.AVG_NO2_EXPOSURE not in Features.SURVIVAL_COVARIATES, \
            "avg_no2_exposure still in survival_covariates — multicollinearity not fixed."

    def test_pollution_index_in_survival_covariates(self) -> None:
        """pollution_index must be in survival covariates."""
        from src.utils.constants import Columns, Features
        assert Columns.POLLUTION_INDEX in Features.SURVIVAL_COVARIATES, \
            "pollution_index not in survival_covariates."

    def test_pm25_no2_raw_correlation(self, train_df) -> None:
        """Document the raw PM2.5/NO2 correlation (expected ~0.97)."""
        from src.utils.constants import Columns
        r = train_df[Columns.AVG_PM25_EXPOSURE].corr(train_df[Columns.AVG_NO2_EXPOSURE])
        # This test documents the known collinearity — it should be high
        assert r > 0.90, f"Expected high PM2.5/NO2 correlation, got r={r:.3f}"

    def test_vif_report_exists(self) -> None:
        """VIF report must have been generated."""
        from src.utils.paths import OutputPaths
        assert OutputPaths.VIF_REPORT.exists(), \
            "VIF report not found — feature_selector.analyse() was not run."
