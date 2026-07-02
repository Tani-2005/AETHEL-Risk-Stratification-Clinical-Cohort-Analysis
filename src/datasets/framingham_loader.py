"""
framingham_loader.py
====================
Dataset Loader: Framingham Heart Study

Loads ``data/framingham/framingham.csv`` and maps it into the harmonized
feature space.  Handles missing values (glucose 9.2%, education 2.5%, etc.)
by documenting them — imputation occurs downstream in ExperimentRunner.

Variable dictionary
-------------------
Raw variable       Harmonized       Notes
male               h_sex_male       0=Female, 1=Male
age                h_age            Years
BMI                h_bmi            kg/m²
currentSmoker      h_is_smoker      0/1
sysBP              h_sys_bp         mmHg
diaBP              h_dia_bp         mmHg
totChol            h_total_cholesterol  mg/dL
glucose            h_glucose        mg/dL (9.2% missing — MAR-likely)
TenYearCHD         h_outcome_binary 10-year coronary heart disease risk

Dataset-specific (not in common space):
  cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
  diabetes, heartRate, education

Missing from common space:
  h_ldl, h_triglycerides
  (Framingham did not measure LDL or triglycerides in this release)

Missing value mechanism assessment
-----------------------------------
glucose (9.2%): Likely MAR — fasting glucose was not collected uniformly
  across all examination cycles. Impute with median (train-only).
education (2.5%): Likely MCAR — administrative gaps. Impute with mode.
totChol (1.2%): Likely MCAR — occasional missed draws. Impute with median.
BPMeds (1.2%): Likely MCAR. Impute with mode (0 = not on BP meds).
BMI (0.4%): Likely MCAR. Impute with median.
heartRate (0.02%): MCAR. Impute with median.
cigsPerDay (0.7%): Non-smokers recorded as 0; actual missing likely MCAR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.datasets.base_loader import (
    BaseDatasetLoader,
    DatasetMetadata,
    FeatureSchema,
)
from src.utils.constants import DatasetNames, HarmonizedColumns
from src.utils.logging_setup import get_logger
from src.utils.paths import DataPaths

logger = get_logger(__name__)

# Raw → Harmonized column mapping
_FRAMINGHAM_MAPPING: dict[str, str] = {
    "male": HarmonizedColumns.SEX_MALE,
    "age": HarmonizedColumns.AGE,
    "BMI": HarmonizedColumns.BMI,
    "currentSmoker": HarmonizedColumns.IS_SMOKER,
    "sysBP": HarmonizedColumns.SYS_BP,
    "diaBP": HarmonizedColumns.DIA_BP,
    "totChol": HarmonizedColumns.TOTAL_CHOLESTEROL,
    "glucose": HarmonizedColumns.GLUCOSE,
    "TenYearCHD": HarmonizedColumns.OUTCOME_BINARY,
}

_DATASET_SPECIFIC: list[str] = [
    "cigsPerDay", "BPMeds", "prevalentStroke",
    "prevalentHyp", "diabetes", "heartRate", "education",
]

_MISSING_FROM_COMMON: list[str] = [
    HarmonizedColumns.LDL,
    HarmonizedColumns.TRIGLYCERIDES,
]

_MISSING_RATES: dict[str, float] = {
    "glucose": 0.092,
    "education": 0.025,
    "totChol": 0.012,
    "BPMeds": 0.012,
    "BMI": 0.004,
    "cigsPerDay": 0.007,
    "heartRate": 0.0002,
}


class FraminghamLoader(BaseDatasetLoader):
    """
    Loader for the Framingham Heart Study dataset.

    Parameters
    ----------
    path : Path | None
        Override path for testing.
    """

    @property
    def name(self) -> str:
        return DatasetNames.FRAMINGHAM

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or DataPaths.FRAMINGHAM_RAW

    def load_raw(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Framingham data not found at {self._path}."
            )
        df = pd.read_csv(self._path)
        logger.info("FraminghamLoader: loaded %d participants from %s", len(df), self._path)

        # Log missing value summary
        missing = df.isnull().sum()
        missing_nonzero = missing[missing > 0]
        if len(missing_nonzero) > 0:
            logger.info(
                "Framingham missing values:\n%s",
                missing_nonzero.to_string(),
            )
        return df

    def harmonize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        # Map raw columns to h_* names (keep originals too)
        for raw_col, h_col in _FRAMINGHAM_MAPPING.items():
            if raw_col in df.columns:
                df[h_col] = df[raw_col]
        # Ensure all h_* common-space columns exist (NaN if absent)
        df = self._ensure_h_columns(
            df, HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY]
        )
        logger.debug("FraminghamLoader: harmonized to %d columns.", len(df.columns))
        return df

    def get_schema(self, df_harmonized: pd.DataFrame) -> FeatureSchema:
        all_h = HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY]
        common_available = [
            c for c in all_h
            if c in df_harmonized.columns and df_harmonized[c].notna().any()
        ]
        ds_specific = [c for c in _DATASET_SPECIFIC if c in df_harmonized.columns]
        return FeatureSchema(
            common_available=common_available,
            missing_from_common=_MISSING_FROM_COMMON,
            dataset_specific=ds_specific,
            all_harmonized_cols=[c for c in df_harmonized.columns if c.startswith("h_")],
        )

    def get_metadata(self, df_harmonized: pd.DataFrame) -> DatasetMetadata:
        h_out = HarmonizedColumns.OUTCOME_BINARY
        event_rate = (
            float(df_harmonized[h_out].mean())
            if h_out in df_harmonized.columns and df_harmonized[h_out].notna().any()
            else None
        )
        return DatasetMetadata(
            name="Framingham Heart Study",
            dataset_id=DatasetNames.FRAMINGHAM,
            n=len(df_harmonized),
            outcome_col=h_out,
            outcome_type="binary",
            supervised=True,
            event_rate=event_rate,
            source_path=self._path,
            population="US community (Framingham, MA)",
            domain="Cardiovascular",
            description="US longitudinal cardiovascular cohort — 10-year CHD binary risk",
            limitations=[
                "glucose missing in 9.2% of participants (MAR-likely).",
                "No LDL or triglycerides measured in this release.",
                "Older cohort (1948-origin study) — secular trends apply.",
                "10-year CHD risk endpoint differs from AETHEL respiratory outcome.",
            ],
            harmonized_outcome_note=(
                "TenYearCHD = 10-year coronary heart disease risk (binary). "
                "Cross-cohort comparison with AETHEL Synthetic (respiratory) "
                "requires explicit acknowledgement of different clinical endpoints. "
                "Structural binary equivalence is assumed for framework purposes."
            ),
        )
