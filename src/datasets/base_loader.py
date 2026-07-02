"""
base_loader.py
==============
Abstract base class and shared dataclasses for all AETHEL dataset loaders.

Every dataset loader (SyntheticLoader, FraminghamLoader, NHANESLoader)
extends BaseDatasetLoader and returns a standardised CohortDataset object.
This guarantees a uniform interface regardless of the underlying data source.

Design principles
-----------------
- Raw data is NEVER modified. Harmonized views are written separately.
- Each loader is responsible for its own column mapping and imputation contract.
- Imputation fitting always happens downstream (ExperimentRunner), never here.
- The returned CohortDataset carries both raw and harmonized DataFrames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetMetadata:
    """
    Descriptive metadata about a clinical dataset.

    Attributes
    ----------
    name            : Human-readable dataset name.
    dataset_id      : Short identifier (synthetic / framingham / nhanes).
    n               : Number of observations in the loaded dataset.
    outcome_col     : Name of the harmonized outcome column (h_outcome_binary),
                      or None if no outcome is available.
    outcome_type    : 'binary' | 'survival' | 'none'
    supervised      : True if outcome labels are available for model training.
    event_rate      : Fraction of positive outcome events (None if no outcome).
    source_path     : Absolute path to the raw source file.
    population      : Population description (e.g. "US community").
    domain          : Clinical domain (e.g. "Cardiovascular").
    description     : One-line dataset description.
    limitations     : List of documented limitations and caveats.
    harmonized_outcome_note : Warning text if cross-cohort outcome comparison
                              requires caveats (different endpoints, etc.).
    """
    name: str
    dataset_id: str
    n: int
    outcome_col: Optional[str]
    outcome_type: str
    supervised: bool
    event_rate: Optional[float]
    source_path: Path
    population: str = ""
    domain: str = ""
    description: str = ""
    limitations: list[str] = field(default_factory=list)
    harmonized_outcome_note: str = ""


@dataclass
class FeatureSchema:
    """
    Documents which features are available in this dataset relative to
    the common harmonized feature space.

    Attributes
    ----------
    common_available    : h_* columns present in THIS dataset.
    missing_from_common : h_* columns absent in this dataset (will be NaN).
    dataset_specific    : Raw columns unique to this dataset (not harmonized).
    all_harmonized_cols : All h_* columns in the returned harmonized DataFrame.
    """
    common_available: list[str]
    missing_from_common: list[str]
    dataset_specific: list[str]
    all_harmonized_cols: list[str]


@dataclass
class CohortDataset:
    """
    Standardised container returned by every dataset loader.

    Fields
    ------
    name            : Dataset identifier string.
    df_harmonized   : DataFrame with all h_* columns + dataset-specific columns.
                      Missing h_* columns are present but filled with NaN.
    df_raw          : Original unmodified DataFrame (read-only reference).
    feature_schema  : Feature availability documentation.
    metadata        : Dataset-level metadata and statistics.
    """
    name: str
    df_harmonized: pd.DataFrame
    df_raw: pd.DataFrame
    feature_schema: FeatureSchema
    metadata: DatasetMetadata

    def __repr__(self) -> str:
        return (
            f"CohortDataset({self.name!r}, n={self.metadata.n}, "
            f"outcome={self.metadata.outcome_col!r}, "
            f"common_features={len(self.feature_schema.common_available)})"
        )


# ---------------------------------------------------------------------------
# Abstract loader
# ---------------------------------------------------------------------------

class BaseDatasetLoader(ABC):
    """
    Abstract base class for all AETHEL dataset loaders.

    Concrete loaders must implement:
    - ``load_raw()``   : Read the source CSV and return a raw DataFrame.
    - ``harmonize()``  : Map raw columns to h_* harmonized names, fill
                         missing h_* columns with NaN, return harmonized DF.
    - ``get_schema()`` : Build FeatureSchema describing feature availability.
    - ``get_metadata()``: Build DatasetMetadata from the harmonized DataFrame.

    The concrete ``load()`` method orchestrates all steps and returns a
    standardised CohortDataset — callers never need to call the sub-methods
    directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short dataset identifier (matches DatasetNames constant)."""
        ...

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Read source file and return unmodified DataFrame."""
        ...

    @abstractmethod
    def harmonize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw columns to harmonized h_* names.

        Rules
        -----
        - Retain all original columns (do not drop anything).
        - Add h_* columns via rename/copy — never in-place on raw data.
        - Set missing h_* columns to NaN explicitly (so schema is consistent).
        - Do NOT impute here. Return harmonized DataFrame with NaNs intact.
        """
        ...

    @abstractmethod
    def get_schema(self, df_harmonized: pd.DataFrame) -> FeatureSchema:
        """Build FeatureSchema from the harmonized DataFrame."""
        ...

    @abstractmethod
    def get_metadata(self, df_harmonized: pd.DataFrame) -> DatasetMetadata:
        """Build DatasetMetadata from the harmonized DataFrame."""
        ...

    def load(self) -> CohortDataset:
        """
        Full load pipeline: raw → harmonize → schema → metadata → CohortDataset.

        Returns
        -------
        CohortDataset
            Standardised dataset object ready for use by ExperimentRunner.
        """
        df_raw = self.load_raw()
        df_harmonized = self.harmonize(df_raw.copy())
        schema = self.get_schema(df_harmonized)
        metadata = self.get_metadata(df_harmonized)
        return CohortDataset(
            name=self.name,
            df_harmonized=df_harmonized,
            df_raw=df_raw,
            feature_schema=schema,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_h_columns(df: pd.DataFrame, all_h_cols: list[str]) -> pd.DataFrame:
        """Add any missing h_* columns as NaN to keep schema uniform."""
        for col in all_h_cols:
            if col not in df.columns:
                df[col] = float("nan")
        return df
