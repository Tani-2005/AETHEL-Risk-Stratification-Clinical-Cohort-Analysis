"""
experiment_config.py
====================
Experiment configuration dataclass and YAML loader.

An ExperimentConfig defines a complete, reproducible experiment including:
- Which datasets are used for training, validation, and testing
- Which feature set is used (intersection / union / custom list)
- The harmonized outcome column
- The random seed for reproducibility

All 7 experiment modes are supported through configuration files
in ``configs/experiments/``. No code changes are needed to add new
experiment variants.

Usage
-----
    from src.experiments.experiment_config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/experiments/exp_mode4_synthetic_to_framingham.yaml")
    print(cfg.train_datasets, cfg.val_datasets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """
    Complete specification for a multi-cohort experiment.

    Attributes
    ----------
    name            : Unique experiment identifier.
    mode            : Integer mode (1–7) matching ExperimentModes constants.
    description     : Human-readable description of the experiment.
    train_datasets  : Dataset names used for model training.
    val_datasets    : Dataset names used for validation (hyperparameter tuning).
    test_datasets   : Dataset names used for final evaluation.
    feature_set     : Feature selection strategy:
                      'intersection' — only features available in all train+val datasets
                      'union'        — all features (NaN where absent)
                      'custom'       — use custom_features list
    custom_features : Explicit list of h_* features (only used when feature_set='custom').
    outcome         : Harmonized outcome column (default: h_outcome_binary).
    seed            : Random seed for all split operations.
    train_split     : Fraction of train_datasets used for training (remainder → val within-dataset).
    notes           : Documented assumptions and caveats.
    supervised_only : If True, raise error if any dataset lacks outcome labels.
    domain_shift_only: If True, experiment is covariate comparison only (no labels required).
    """
    name: str
    mode: int
    description: str = ""
    train_datasets: list[str] = field(default_factory=list)
    val_datasets: list[str] = field(default_factory=list)
    test_datasets: list[str] = field(default_factory=list)
    feature_set: str = "intersection"      # intersection | union | custom
    custom_features: list[str] = field(default_factory=list)
    outcome: str = "h_outcome_binary"
    seed: int = 42
    train_split: float = 0.70
    notes: list[str] = field(default_factory=list)
    supervised_only: bool = True
    domain_shift_only: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls(
            name=raw["name"],
            mode=raw["mode"],
            description=raw.get("description", ""),
            train_datasets=raw.get("train_datasets", []),
            val_datasets=raw.get("val_datasets", []),
            test_datasets=raw.get("test_datasets", []),
            feature_set=raw.get("feature_set", "intersection"),
            custom_features=raw.get("custom_features", []),
            outcome=raw.get("outcome", "h_outcome_binary"),
            seed=raw.get("seed", 42),
            train_split=raw.get("train_split", 0.70),
            notes=raw.get("notes", []),
            supervised_only=raw.get("supervised_only", True),
            domain_shift_only=raw.get("domain_shift_only", False),
        )

    def all_datasets(self) -> list[str]:
        """All unique dataset names referenced in this experiment."""
        seen: set[str] = set()
        result: list[str] = []
        for name in self.train_datasets + self.val_datasets + self.test_datasets:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result
