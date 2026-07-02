"""
registry.py
===========
Dataset Registry — config-driven loader factory.

The registry maps dataset names to loader classes.  All dataset references
throughout the pipeline should go through the registry rather than
instantiating loaders directly.  This makes it trivial to add a new dataset:
register one new entry and provide a loader class.

Extending the framework
-----------------------
To add a new cohort (e.g. UK Biobank):

1. Create ``src/datasets/ukbiobank_loader.py`` extending BaseDatasetLoader.
2. Register it below: ``"ukbiobank": UKBiobankLoader``
3. Add an entry to ``configs/default.yaml → datasets:``
4. No other code changes required.

Usage
-----
    from src.datasets.registry import DatasetRegistry

    registry = DatasetRegistry.from_config()
    dataset = registry.load("framingham")
    print(dataset.metadata)
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import pandas as pd

from src.datasets.base_loader import BaseDatasetLoader, CohortDataset
from src.datasets.synthetic_loader import SyntheticLoader
from src.datasets.framingham_loader import FraminghamLoader
from src.datasets.nhanes_loader import NHANESLoader
from src.utils.config_loader import load_config
from src.utils.constants import DatasetNames
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Loader class map — add new datasets here
# ---------------------------------------------------------------------------
_LOADER_MAP: dict[str, Type[BaseDatasetLoader]] = {
    DatasetNames.SYNTHETIC: SyntheticLoader,
    DatasetNames.FRAMINGHAM: FraminghamLoader,
    DatasetNames.NHANES: NHANESLoader,
}


class DatasetRegistry:
    """
    Config-driven registry for all AETHEL dataset loaders.

    Parameters
    ----------
    enabled_datasets : list[str]
        Dataset IDs to include (sourced from config). If None, load all.
    dataset_paths : dict[str, Path]
        Optional path overrides per dataset (useful for testing).
    """

    def __init__(
        self,
        enabled_datasets: list[str] | None = None,
        dataset_paths: dict[str, Path] | None = None,
    ) -> None:
        self._enabled = enabled_datasets or list(_LOADER_MAP.keys())
        self._paths = dataset_paths or {}

    @classmethod
    def from_config(cls) -> "DatasetRegistry":
        """Instantiate the registry from ``configs/default.yaml``."""
        cfg = load_config()
        raw_datasets = getattr(cfg, "datasets", {}) or {}
        enabled = [
            name for name, meta in raw_datasets.items()
            if isinstance(meta, dict) and meta.get("enabled", True)
        ]
        if not enabled:
            logger.warning("No datasets enabled in config — defaulting to all registered.")
            enabled = list(_LOADER_MAP.keys())
        logger.info("DatasetRegistry: enabled datasets = %s", enabled)
        return cls(enabled_datasets=enabled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, name: str) -> CohortDataset:
        """
        Load and harmonize a single dataset by name.

        Parameters
        ----------
        name : str
            Dataset identifier (e.g. 'synthetic', 'framingham', 'nhanes').

        Returns
        -------
        CohortDataset
            Fully loaded and harmonized dataset.

        Raises
        ------
        KeyError
            If the dataset name is not registered.
        ValueError
            If the dataset is not enabled in the current config.
        """
        if name not in _LOADER_MAP:
            raise KeyError(
                f"Dataset '{name}' not in registry. "
                f"Available: {list(_LOADER_MAP.keys())}"
            )
        if name not in self._enabled:
            raise ValueError(
                f"Dataset '{name}' is disabled in config. "
                f"Enabled: {self._enabled}"
            )

        loader_cls = _LOADER_MAP[name]
        path_override = self._paths.get(name)
        loader = loader_cls(path=path_override) if path_override else loader_cls()

        logger.info("Loading dataset: %s", name)
        dataset = loader.load()
        logger.info(
            "Loaded %s: n=%d, outcome=%s, common_features=%d",
            name, dataset.metadata.n,
            dataset.metadata.outcome_col,
            len(dataset.feature_schema.common_available),
        )
        return dataset

    def load_all(self) -> dict[str, CohortDataset]:
        """Load all enabled datasets and return a name → CohortDataset dict."""
        results = {}
        for name in self._enabled:
            try:
                results[name] = self.load(name)
            except Exception as exc:
                logger.error("Failed to load dataset '%s': %s", name, exc)
        return results

    def available(self) -> list[str]:
        """Return the list of enabled dataset names."""
        return list(self._enabled)

    def audit(self) -> pd.DataFrame:
        """
        Load all datasets and generate a summary audit table.

        Returns
        -------
        pd.DataFrame
            One row per dataset with key statistics.
        """
        datasets = self.load_all()
        rows = []
        for name, ds in datasets.items():
            m = ds.metadata
            s = ds.feature_schema
            rows.append({
                "dataset": name,
                "n": m.n,
                "outcome": m.outcome_col or "None",
                "outcome_type": m.outcome_type,
                "supervised": m.supervised,
                "event_rate_pct": round(m.event_rate * 100, 1) if m.event_rate else None,
                "common_features_available": len(s.common_available),
                "common_features_missing": len(s.missing_from_common),
                "dataset_specific_features": len(s.dataset_specific),
                "population": m.population,
                "domain": m.domain,
            })
        return pd.DataFrame(rows)
