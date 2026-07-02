"""
experiment_runner.py — ExperimentRunner for AETHEL multi-cohort framework.
Supports all 7 experiment modes through configuration-driven data preparation.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.datasets.base_loader import CohortDataset
from src.datasets.registry import DatasetRegistry
from src.experiments.experiment_config import ExperimentConfig
from src.utils.constants import HarmonizedColumns
from src.utils.logging_setup import get_logger
from src.utils.paths import ExperimentDirs

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    experiment_name: str
    mode: int
    feature_set: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: Optional[pd.DataFrame]
    outcome_col: str
    dataset_sources: dict[str, list[str]]
    output_dir: Path
    notes: list[str] = field(default_factory=list)


class ExperimentRunner:
    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self._registry = registry or DatasetRegistry.from_config()

    def run(self, exp_cfg: ExperimentConfig) -> ExperimentResult:
        logger.info("ExperimentRunner: starting '%s' (mode %d)", exp_cfg.name, exp_cfg.mode)
        if exp_cfg.mode == 3:
            raise ValueError("Mode 3 (NHANES-only) is not supported: no outcome labels.")

        # Load datasets
        datasets: dict[str, CohortDataset] = {}
        for name in exp_cfg.all_datasets():
            ds = self._registry.load(name)
            if exp_cfg.supervised_only and not exp_cfg.domain_shift_only:
                if not ds.metadata.supervised and name in (exp_cfg.train_datasets + exp_cfg.test_datasets):
                    raise ValueError(f"Dataset '{name}' has no labels but is used in supervised experiment.")
            datasets[name] = ds

        # Resolve features
        feature_cols = self._resolve_feature_set(exp_cfg, datasets)
        logger.info("Feature set (%d): %s", len(feature_cols), feature_cols)

        # Build training data
        train_df = self._merge_datasets([datasets[n] for n in exp_cfg.train_datasets], feature_cols, exp_cfg.outcome)
        imputer = self._fit_imputer(train_df, feature_cols)
        train_df = self._apply_imputer(train_df, feature_cols, imputer)

        if exp_cfg.mode in (1, 2):
            train_df, val_df, test_df = self._split_within(train_df, exp_cfg)
        else:
            val_df = self._build_eval_split(exp_cfg.val_datasets, datasets, feature_cols, exp_cfg.outcome, imputer)
            test_df = self._build_eval_split(exp_cfg.test_datasets, datasets, feature_cols, exp_cfg.outcome, imputer) if exp_cfg.test_datasets else None

        out_dir = ExperimentDirs.BASE / exp_cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save(train_df, val_df, test_df, out_dir, exp_cfg, feature_cols)

        logger.info("'%s' done: train=%d, val=%d, test=%s", exp_cfg.name, len(train_df), len(val_df),
                    len(test_df) if test_df is not None else "N/A")
        return ExperimentResult(
            experiment_name=exp_cfg.name, mode=exp_cfg.mode, feature_set=feature_cols,
            train_df=train_df, val_df=val_df, test_df=test_df, outcome_col=exp_cfg.outcome,
            dataset_sources={"train": exp_cfg.train_datasets, "val": exp_cfg.val_datasets, "test": exp_cfg.test_datasets},
            output_dir=out_dir, notes=exp_cfg.notes,
        )

    def _resolve_feature_set(self, exp_cfg: ExperimentConfig, datasets: dict[str, CohortDataset]) -> list[str]:
        if exp_cfg.feature_set == "custom":
            return exp_cfg.custom_features
        supervised = [n for n in exp_cfg.all_datasets() if n in datasets and datasets[n].metadata.supervised]
        if exp_cfg.feature_set == "intersection":
            sets = [set(datasets[n].feature_schema.common_available) for n in supervised]
            if not sets:
                return HarmonizedColumns.SUPERVISED_INTERSECTION
            result = sorted(set.intersection(*sets) - {HarmonizedColumns.OUTCOME_BINARY})
            return result or HarmonizedColumns.SUPERVISED_INTERSECTION
        elif exp_cfg.feature_set == "union":
            union: set[str] = set()
            for n in supervised:
                union.update(datasets[n].feature_schema.common_available)
            union.discard(HarmonizedColumns.OUTCOME_BINARY)
            return sorted(union)
        raise ValueError(f"Unknown feature_set: '{exp_cfg.feature_set}'")

    def _merge_datasets(self, datasets: list[CohortDataset], feature_cols: list[str], outcome_col: str) -> pd.DataFrame:
        frames = []
        for ds in datasets:
            extra_cols = [c for c in ["months_observed", "event_occurred"] if c in ds.df_harmonized.columns]
            cols = [c for c in feature_cols + [outcome_col] + extra_cols if c in ds.df_harmonized.columns]
            sub = ds.df_harmonized[cols].copy()
            sub["_source_dataset"] = ds.name
            frames.append(sub)
        return pd.concat(frames, ignore_index=True)

    def _fit_imputer(self, train_df: pd.DataFrame, feature_cols: list[str]) -> SimpleImputer:
        numeric_cols = [c for c in feature_cols if c in train_df.columns and train_df[c].dtype != object]
        imp = SimpleImputer(strategy="median")
        if numeric_cols:
            imp.fit(train_df[numeric_cols])
        return imp

    def _apply_imputer(self, df: pd.DataFrame, feature_cols: list[str], imputer: SimpleImputer) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = [c for c in feature_cols if c in df.columns and df[c].dtype != object]
        if numeric_cols:
            df[numeric_cols] = imputer.transform(df[numeric_cols])
        return df

    def _split_within(self, df: pd.DataFrame, exp_cfg: ExperimentConfig):
        stratify = df[exp_cfg.outcome] if exp_cfg.outcome in df.columns else None
        train_val, test = train_test_split(df, test_size=0.15, stratify=stratify, random_state=exp_cfg.seed)
        strat_tv = train_val[exp_cfg.outcome] if exp_cfg.outcome in train_val.columns else None
        train, val = train_test_split(train_val, test_size=0.15 / 0.85, stratify=strat_tv, random_state=exp_cfg.seed)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def _build_eval_split(self, names: list[str], datasets: dict, feature_cols: list[str], outcome_col: str, imputer: SimpleImputer) -> Optional[pd.DataFrame]:
        available = [n for n in names if n in datasets]
        if not available:
            return None
        df = self._merge_datasets([datasets[n] for n in available], feature_cols, outcome_col)
        return self._apply_imputer(df, feature_cols, imputer)

    def _save(self, train_df, val_df, test_df, out_dir: Path, exp_cfg: ExperimentConfig, feature_cols: list[str]):
        train_df.to_csv(out_dir / "train.csv", index=False)
        val_df.to_csv(out_dir / "val.csv", index=False)
        if test_df is not None:
            test_df.to_csv(out_dir / "test.csv", index=False)
        summary = {
            "experiment": exp_cfg.name, "mode": exp_cfg.mode,
            "feature_set_strategy": exp_cfg.feature_set, "features_used": feature_cols,
            "n_features": len(feature_cols),
            "train_datasets": exp_cfg.train_datasets, "val_datasets": exp_cfg.val_datasets,
            "n_train": len(train_df), "n_val": len(val_df),
            "n_test": len(test_df) if test_df is not None else None,
            "notes": exp_cfg.notes,
        }
        with (out_dir / "experiment_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Saved experiment outputs to %s", out_dir)
