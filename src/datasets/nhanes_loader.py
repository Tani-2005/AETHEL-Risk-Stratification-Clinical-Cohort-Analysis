"""
nhanes_loader.py
================
Dataset Loader: NHANES Biochemical Survey (combined_data.csv)

Loads the NHANES-derived biochemical dataset for domain shift analysis.

⚠️  IMPORTANT: This dataset has NO outcome variable.
    It cannot be used for supervised training or testing.
    Use cases: covariate distribution comparison, domain shift analysis,
    feature availability mapping.

Variable dictionary (NHANES column codes)
-----------------------------------------
Raw column    Harmonized         Description
st            h_sys_bp           Systolic blood pressure (mmHg)
dt            h_dia_bp           Diastolic blood pressure (mmHg)
tc            h_total_cholesterol Total cholesterol (mg/dL)
lbdldl        h_ldl              LDL cholesterol (mg/dL)
lbxtr         h_triglycerides    Triglycerides (mg/dL)
lbxin         —                  Serum insulin (pmol/L) [dataset-specific]
lbdinsi       —                  Fasting insulin (pmol/L) [dataset-specific]
luxsmed       —                  NHANES-specific marker
luxcapm       —                  NHANES-specific marker
tcs           —                  Cholesterol standardised [dataset-specific]
Column1       —                  NHANES participant sequence number (SEQN)

Missing from common space (NO DATA — absent from source):
  h_age, h_bmi, h_is_smoker, h_sex_male, h_glucose, h_outcome_binary

Documented limitations
-----------------------
- No demographic variables (age, sex, BMI, smoking).
- No outcome variable — strictly unsupervised use only.
- Some columns (luxsmed, luxcapm) have non-standard NHANES codes;
  their exact meaning cannot be confirmed from this file alone.
- Zero values in BP and cholesterol (st=0, tc=0) likely indicate
  missing/not-measured rather than true physiological values.
  These are flagged but NOT imputed at the loader level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
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

# Raw → Harmonized mapping (biochemical columns only)
_NHANES_MAPPING: dict[str, str] = {
    "st": HarmonizedColumns.SYS_BP,
    "dt": HarmonizedColumns.DIA_BP,
    "tc": HarmonizedColumns.TOTAL_CHOLESTEROL,
    "lbdldl": HarmonizedColumns.LDL,
    "lbxtr": HarmonizedColumns.TRIGLYCERIDES,
}

_DATASET_SPECIFIC: list[str] = [
    "Column1",   # SEQN participant ID
    "lbxin",     # serum insulin
    "lbdinsi",   # fasting insulin
    "luxsmed",   # unknown NHANES marker
    "luxcapm",   # unknown NHANES marker
    "tcs",       # cholesterol standardised
]

# Features available in NHANES vs common space
_COMMON_AVAILABLE: list[str] = [
    HarmonizedColumns.SYS_BP,
    HarmonizedColumns.DIA_BP,
    HarmonizedColumns.TOTAL_CHOLESTEROL,
    HarmonizedColumns.LDL,
    HarmonizedColumns.TRIGLYCERIDES,
]

_MISSING_FROM_COMMON: list[str] = [
    HarmonizedColumns.AGE,
    HarmonizedColumns.BMI,
    HarmonizedColumns.IS_SMOKER,
    HarmonizedColumns.SEX_MALE,
    HarmonizedColumns.GLUCOSE,
    HarmonizedColumns.OUTCOME_BINARY,
]


class NHANESLoader(BaseDatasetLoader):
    """
    Loader for the NHANES Biochemical Survey (domain shift analysis only).

    This loader flags zero-values in physiological measurements as NaN,
    since zero systolic BP or zero total cholesterol is physiologically
    impossible and almost certainly indicates missing data codes.

    Parameters
    ----------
    path : Path | None
        Override path for testing.
    flag_zero_as_nan : bool
        If True (default), replace zero values in BP and cholesterol
        columns with NaN, as these likely represent missing data codes.
    """

    @property
    def name(self) -> str:
        return DatasetNames.NHANES

    def __init__(
        self,
        path: Optional[Path] = None,
        flag_zero_as_nan: bool = True,
    ) -> None:
        self._path = path or DataPaths.NHANES_RAW
        self._flag_zero_as_nan = flag_zero_as_nan

    def load_raw(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(f"NHANES data not found at {self._path}.")
        df = pd.read_csv(self._path)
        logger.info("NHANESLoader: loaded %d records from %s", len(df), self._path)

        if self._flag_zero_as_nan:
            # Physiologically impossible zero values → NaN
            physio_cols = ["st", "dt", "tc", "lbdldl", "lbxtr"]
            for col in physio_cols:
                if col in df.columns:
                    n_zeros = (df[col] == 0).sum()
                    if n_zeros > 0:
                        logger.warning(
                            "NHANES: %d zero values in '%s' replaced with NaN "
                            "(likely missing data codes).", n_zeros, col
                        )
                        df[col] = df[col].replace(0, np.nan)
        return df

    def harmonize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        for raw_col, h_col in _NHANES_MAPPING.items():
            if raw_col in df.columns:
                df[h_col] = df[raw_col]
        # Ensure all common h_* columns exist (NaN for absent ones)
        df = self._ensure_h_columns(
            df, HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY]
        )
        logger.debug("NHANESLoader: harmonized to %d columns.", len(df.columns))
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
        return DatasetMetadata(
            name="NHANES Biochemical Survey",
            dataset_id=DatasetNames.NHANES,
            n=len(df_harmonized),
            outcome_col=None,
            outcome_type="none",
            supervised=False,
            event_rate=None,
            source_path=self._path,
            population="US general population",
            domain="Cardiovascular/Metabolic",
            description="NHANES-derived biochemical markers dataset (no outcome variable)",
            limitations=[
                "No outcome variable — cannot be used for supervised experiments.",
                "No age, BMI, smoking, or sex variables present.",
                "Zero values in BP/cholesterol replaced with NaN (missing data codes).",
                "luxsmed and luxcapm column meanings cannot be confirmed.",
                "Use for domain shift / covariate distribution analysis only.",
            ],
            harmonized_outcome_note="No outcome available. Supervised experiments not supported.",
        )
