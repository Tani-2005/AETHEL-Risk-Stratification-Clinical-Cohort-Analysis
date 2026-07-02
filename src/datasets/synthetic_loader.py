"""
synthetic_loader.py
===================
Dataset Loader: AETHEL Synthetic Cohort

Loads ``data/processed/analytical_cohort.csv`` and maps it into the
common harmonized feature space.

Harmonization mapping
---------------------
Raw column          → Harmonized column
age                 → h_age
bmi                 → h_bmi
is_smoker           → h_is_smoker
event_occurred      → h_outcome_binary

Dataset-specific features (retained, not harmonized):
  genomic_risk_score, townsend_index, avg_pm25_exposure,
  avg_no2_exposure, months_observed, pollution_index,
  lifestyle_risk, high_genomic_risk, city, city_id, lat, lon

Missing from common space (set to NaN):
  h_sex_male, h_sys_bp, h_dia_bp, h_total_cholesterol,
  h_ldl, h_triglycerides, h_glucose
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.datasets.base_loader import (
    BaseDatasetLoader,
    CohortDataset,
    DatasetMetadata,
    FeatureSchema,
)
from src.utils.constants import DatasetNames, HarmonizedColumns
from src.utils.logging_setup import get_logger
from src.utils.paths import DataPaths

logger = get_logger(__name__)

# Raw → Harmonized column mapping
_SYNTHETIC_MAPPING: dict[str, str] = {
    "age": HarmonizedColumns.AGE,
    "bmi": HarmonizedColumns.BMI,
    "is_smoker": HarmonizedColumns.IS_SMOKER,
    "event_occurred": HarmonizedColumns.OUTCOME_BINARY,
}

_DATASET_SPECIFIC: list[str] = [
    "patient_id", "genomic_risk_score", "townsend_index",
    "avg_pm25_exposure", "avg_no2_exposure", "months_observed",
    "pollution_index", "lifestyle_risk", "high_genomic_risk",
    "city", "city_id", "lat", "lon",
    "bmi_category", "age_group",
]

_MISSING_FROM_COMMON: list[str] = [
    HarmonizedColumns.SEX_MALE,
    HarmonizedColumns.SYS_BP,
    HarmonizedColumns.DIA_BP,
    HarmonizedColumns.TOTAL_CHOLESTEROL,
    HarmonizedColumns.LDL,
    HarmonizedColumns.TRIGLYCERIDES,
    HarmonizedColumns.GLUCOSE,
]


class SyntheticLoader(BaseDatasetLoader):
    """
    Loader for the AETHEL Synthetic Pan-European Respiratory Cohort.

    Parameters
    ----------
    path : Path | None
        Override the default cohort path. Useful for testing.
    """

    @property
    def name(self) -> str:
        return DatasetNames.SYNTHETIC

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or DataPaths.ANALYTICAL_COHORT

    def load_raw(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Synthetic cohort not found at {self._path}.\n"
                "Run the full pipeline first: python scripts/run_pipeline.py"
            )
        df = pd.read_csv(self._path)
        logger.info("SyntheticLoader: loaded %d patients from %s", len(df), self._path)
        return df

    def harmonize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        # Apply column mapping (add h_* columns alongside originals)
        for raw_col, h_col in _SYNTHETIC_MAPPING.items():
            if raw_col in df.columns:
                df[h_col] = df[raw_col]
        # Fill missing common features with NaN
        df = self._ensure_h_columns(df, HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY])
        logger.debug("SyntheticLoader: harmonized %d columns.", len(df.columns))
        return df

    def get_schema(self, df_harmonized: pd.DataFrame) -> FeatureSchema:
        common_available = [
            c for c in HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY]
            if c in df_harmonized.columns and df_harmonized[c].notna().any()
        ]
        ds_specific = [
            c for c in _DATASET_SPECIFIC if c in df_harmonized.columns
        ]
        return FeatureSchema(
            common_available=common_available,
            missing_from_common=_MISSING_FROM_COMMON,
            dataset_specific=ds_specific,
            all_harmonized_cols=[c for c in df_harmonized.columns if c.startswith("h_")],
        )

    def get_metadata(self, df_harmonized: pd.DataFrame) -> DatasetMetadata:
        n = len(df_harmonized)
        h_out = HarmonizedColumns.OUTCOME_BINARY
        event_rate = (
            float(df_harmonized[h_out].mean())
            if h_out in df_harmonized.columns and df_harmonized[h_out].notna().any()
            else None
        )
        return DatasetMetadata(
            name="AETHEL Synthetic Cohort",
            dataset_id=DatasetNames.SYNTHETIC,
            n=n,
            outcome_col=h_out,
            outcome_type="binary",
            supervised=True,
            event_rate=event_rate,
            source_path=self._path,
            population="European (simulated)",
            domain="Respiratory",
            description="Pan-European synthetic respiratory disease cohort",
            limitations=[
                "Synthetic data — does not represent real patients.",
                "No sex variable available.",
                "No cardiovascular markers (cholesterol, BP, glucose).",
                "Genomic risk score and Townsend deprivation index are synthetic.",
            ],
            harmonized_outcome_note=(
                "event_occurred = respiratory disease event. "
                "Structurally binary but clinically distinct from TenYearCHD."
            ),
        )
