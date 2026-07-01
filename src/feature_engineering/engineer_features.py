"""
engineer_features.py
====================
Stage: Clinically Justified Feature Engineering

Adds scientifically motivated derived features to the analytical cohort.
Every feature here has a documented epidemiological or clinical justification.
Features that would be medically meaningless are explicitly rejected.

Feature inventory
-----------------

DETERMINISTIC (safe to apply before train/test split):
  bmi_category      WHO BMI classification. Standard in all epidemiological
                    studies. Captures non-linear BMI-mortality relationship.
  age_group         Decade-based age strata. Standard epidemiological grouping;
                    respects the non-linear effect of age on hazard.
  lifestyle_risk    is_smoker × townsend_index interaction term.
                    Mackenbach et al. (2008) established synergistic
                    smoking–deprivation effects on mortality in Europe.
  high_genomic_risk Binary flag: genomic_risk_score > 1 SD above mean.
                    Precision medicine convention for high-PRS individuals.
                    Used in UK Biobank analyses and NICE guidance.

FIT-BASED (must be applied after train/test split, fit on train only):
  pollution_index   (z_PM2.5 + z_NO2) / 2, where z-scores are computed
                    from the TRAINING set. Mirrors EEA combined burden
                    approach (Beelen et al., 2014). Addresses the
                    PM2.5/NO2 multicollinearity (VIF=15.25 measured
                    in audit). Replaces avg_no2_exposure in covariates.

Rejected features (with rationale):
  age × genomic_risk_score  — No established biological mechanism for
                              this interaction in respiratory disease.
  city-level features       — Already captured by exposure variables.
  PM2.5 × NO2 product       — Severely collinear; adds no information.

Usage
-----
    from src.feature_engineering.engineer_features import (
        add_deterministic_features,
        PollutionIndexComputer,
    )

    # Before split:
    df = add_deterministic_features(df, cfg)

    # After split (fit on train only):
    pic = PollutionIndexComputer(cfg).fit(train_df)
    train_df = pic.transform(train_df)
    val_df   = pic.transform(val_df)
    test_df  = pic.transform(test_df)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib

from src.utils.config_loader import AETHELConfig
from src.utils.constants import Columns
from src.utils.logging_setup import get_logger
from src.utils.paths import OutputDirs, OutputPaths

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BMI category thresholds (WHO classification)
# ---------------------------------------------------------------------------

def _bmi_category(bmi: float) -> str:
    """WHO BMI classification."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


# ---------------------------------------------------------------------------
# Age group bins (epidemiological decade stratification)
# ---------------------------------------------------------------------------

def _age_group(age: float) -> str:
    """Decade-based age grouping (40–85 range)."""
    if age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    else:
        return "70-85"


# ---------------------------------------------------------------------------
# Deterministic feature engineering (safe pre-split)
# ---------------------------------------------------------------------------

def add_deterministic_features(df: pd.DataFrame, cfg: AETHELConfig) -> pd.DataFrame:
    """
    Add all deterministic engineered features to the dataset.

    These features are computed row-wise without any fitted statistics,
    so they are safe to apply to the full dataset before the train/test
    split.  No leakage risk.

    Features added (controlled by feature_engineering config toggles):
      - ``bmi_category``     WHO BMI class
      - ``age_group``        Decade age stratum
      - ``lifestyle_risk``   is_smoker × townsend_index interaction
      - ``high_genomic_risk`` PRS high-risk binary flag

    Parameters
    ----------
    df : pd.DataFrame
        Analytical cohort (pre-split).
    cfg : AETHELConfig
        Configuration controlling which features to add.

    Returns
    -------
    pd.DataFrame
        Updated cohort with new feature columns appended.
    """
    feng = cfg.feature_eng
    df = df.copy()
    added: list[str] = []

    if feng.add_bmi_category and Columns.BMI in df.columns:
        df[Columns.BMI_CATEGORY] = df[Columns.BMI].apply(_bmi_category)
        added.append(Columns.BMI_CATEGORY)
        logger.debug(
            "bmi_category distribution:\n%s",
            df[Columns.BMI_CATEGORY].value_counts().to_dict(),
        )

    if feng.add_age_group and Columns.AGE in df.columns:
        df[Columns.AGE_GROUP] = df[Columns.AGE].apply(_age_group)
        added.append(Columns.AGE_GROUP)
        logger.debug(
            "age_group distribution:\n%s",
            df[Columns.AGE_GROUP].value_counts().to_dict(),
        )

    if feng.add_lifestyle_risk and all(c in df.columns for c in [Columns.IS_SMOKER, Columns.TOWNSEND_INDEX]):
        # Smoking × deprivation interaction (Mackenbach et al., 2008)
        df[Columns.LIFESTYLE_RISK] = df[Columns.IS_SMOKER] * df[Columns.TOWNSEND_INDEX]
        added.append(Columns.LIFESTYLE_RISK)
        logger.debug(
            "lifestyle_risk: mean=%.3f std=%.3f",
            df[Columns.LIFESTYLE_RISK].mean(),
            df[Columns.LIFESTYLE_RISK].std(),
        )

    if feng.add_high_genomic_risk and Columns.GENOMIC_RISK_SCORE in df.columns:
        threshold = feng.genomic_risk_threshold
        df[Columns.HIGH_GENOMIC_RISK] = (
            df[Columns.GENOMIC_RISK_SCORE] > threshold
        ).astype(int)
        n_high = int(df[Columns.HIGH_GENOMIC_RISK].sum())
        added.append(Columns.HIGH_GENOMIC_RISK)
        logger.debug(
            "high_genomic_risk: %d patients (%.1f%%) above threshold %.1f",
            n_high, n_high / len(df) * 100, threshold,
        )

    if added:
        logger.info("Deterministic features added: %s", added)
    else:
        logger.warning("No deterministic features were added (check config toggles).")

    return df


# ---------------------------------------------------------------------------
# Fit-based feature engineering (must be applied post-split)
# ---------------------------------------------------------------------------

class PollutionIndexComputer:
    """
    Computes the combined pollution burden index.

    **What:** Standardised composite of PM2.5 and NO2:
        pollution_index = (z_PM2.5 + z_NO2) / 2

    **Why:** PM2.5 and NO2 have r=0.967 and VIF=15.25 in this dataset
    (by construction: NO2 = PM2.5 × 1.4 + noise).  Including both in
    Cox PH inflates standard errors and makes hazard ratios uninterpretable.
    A combined index preserves the environmental burden signal without
    multicollinearity.  This approach mirrors EEA APHEIS methodology
    (Beelen et al., 2014; WHO Air Quality Guidelines, 2021).

    **Leakage prevention:** The mean and std used for z-scoring are
    computed ONLY from the training set.  The fitted statistics are then
    applied identically to val and test sets.  The stats are persisted
    to ``outputs/models/pollution_index_stats.csv`` for full auditability.

    Parameters
    ----------
    cfg : AETHELConfig
        Configuration (provides the add_pollution_index toggle).
    """

    def __init__(self, cfg: AETHELConfig) -> None:
        self._enabled: bool = cfg.feature_eng.add_pollution_index
        self.pm25_mean_: float | None = None
        self.pm25_std_: float | None = None
        self.no2_mean_: float | None = None
        self.no2_std_: float | None = None
        self._is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "PollutionIndexComputer":
        """
        Fit the standardisation statistics on the training set ONLY.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training fold of the cohort.

        Returns
        -------
        PollutionIndexComputer
            self (for method chaining).
        """
        if not self._enabled:
            return self

        self.pm25_mean_ = float(train_df[Columns.AVG_PM25_EXPOSURE].mean())
        self.pm25_std_ = float(train_df[Columns.AVG_PM25_EXPOSURE].std(ddof=1))
        self.no2_mean_ = float(train_df[Columns.AVG_NO2_EXPOSURE].mean())
        self.no2_std_ = float(train_df[Columns.AVG_NO2_EXPOSURE].std(ddof=1))
        self._is_fitted = True

        logger.info(
            "PollutionIndexComputer fitted on train: "
            "PM2.5 mean=%.3f std=%.3f | NO2 mean=%.3f std=%.3f",
            self.pm25_mean_, self.pm25_std_, self.no2_mean_, self.no2_std_,
        )
        self._save_stats()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append pollution_index using training statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Any split (train, val, or test).

        Returns
        -------
        pd.DataFrame
            Copy with ``pollution_index`` column appended.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._enabled:
            return df

        if not self._is_fitted:
            raise RuntimeError("PollutionIndexComputer must be fit() before transform().")

        df = df.copy()
        z_pm25 = (df[Columns.AVG_PM25_EXPOSURE] - self.pm25_mean_) / self.pm25_std_
        z_no2 = (df[Columns.AVG_NO2_EXPOSURE] - self.no2_mean_) / self.no2_std_
        df[Columns.POLLUTION_INDEX] = (z_pm25 + z_no2) / 2.0
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on train, then transform train."""
        return self.fit(train_df).transform(train_df)

    def _save_stats(self) -> None:
        """Persist fitting statistics for reproducibility auditing."""
        OutputDirs.MODELS.mkdir(parents=True, exist_ok=True)
        stats = pd.DataFrame([{
            "feature": "avg_pm25_exposure",
            "mean": self.pm25_mean_,
            "std": self.pm25_std_,
        }, {
            "feature": "avg_no2_exposure",
            "mean": self.no2_mean_,
            "std": self.no2_std_,
        }])
        stats.to_csv(OutputPaths.POLLUTION_INDEX_STATS, index=False)
        logger.info("Pollution index stats saved to %s", OutputPaths.POLLUTION_INDEX_STATS)
