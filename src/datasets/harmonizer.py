"""
harmonizer.py
=============
Harmonization Layer — generates all mapping tables and reports.

This module produces the data artefacts that document every harmonization
decision: mapping tables (raw→harmonized), feature availability matrix,
dataset audit, and the harmonization report.

It also saves the harmonized datasets to ``data/harmonized/``.

Reports generated
-----------------
- outputs/reports/dataset_audit.csv
- outputs/reports/feature_availability_matrix.csv
- outputs/reports/harmonization_report.csv
- data/harmonized/synthetic_harmonized.csv
- data/harmonized/framingham_harmonized.csv
- data/harmonized/nhanes_harmonized.csv
- data/mapping_tables/synthetic_mapping.csv
- data/mapping_tables/framingham_mapping.csv
- data/mapping_tables/nhanes_mapping.csv
- data/feature_metadata/common_feature_space.json

Usage
-----
    from src.datasets.harmonizer import HarmonizationLayer
    from src.datasets.registry import DatasetRegistry

    registry = DatasetRegistry.from_config()
    datasets = registry.load_all()
    layer = HarmonizationLayer()
    layer.run(datasets)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.datasets.base_loader import CohortDataset
from src.utils.constants import DatasetNames, HarmonizedColumns
from src.utils.logging_setup import get_logger
from src.utils.paths import DataDirs, DataPaths, OutputDirs, OutputPaths

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mapping tables (documented per-dataset)
# ---------------------------------------------------------------------------

_SYNTHETIC_MAPPING_TABLE = [
    {"dataset": "synthetic", "raw_column": "age", "harmonized_column": "h_age", "unit": "years", "recoding": "none", "notes": "Direct rename"},
    {"dataset": "synthetic", "raw_column": "bmi", "harmonized_column": "h_bmi", "unit": "kg/m²", "recoding": "none", "notes": "Direct rename"},
    {"dataset": "synthetic", "raw_column": "is_smoker", "harmonized_column": "h_is_smoker", "unit": "binary 0/1", "recoding": "none", "notes": "Direct rename"},
    {"dataset": "synthetic", "raw_column": "event_occurred", "harmonized_column": "h_outcome_binary", "unit": "binary 0/1", "recoding": "none", "notes": "Respiratory event. Clinically distinct from TenYearCHD."},
]

_FRAMINGHAM_MAPPING_TABLE = [
    {"dataset": "framingham", "raw_column": "age", "harmonized_column": "h_age", "unit": "years", "recoding": "none", "notes": "Direct rename"},
    {"dataset": "framingham", "raw_column": "BMI", "harmonized_column": "h_bmi", "unit": "kg/m²", "recoding": "none", "notes": "Direct rename (uppercase BMI)"},
    {"dataset": "framingham", "raw_column": "currentSmoker", "harmonized_column": "h_is_smoker", "unit": "binary 0/1", "recoding": "none", "notes": "Direct rename. Note: cigsPerDay retained as dataset-specific."},
    {"dataset": "framingham", "raw_column": "male", "harmonized_column": "h_sex_male", "unit": "binary 0/1", "recoding": "none", "notes": "0=Female, 1=Male"},
    {"dataset": "framingham", "raw_column": "sysBP", "harmonized_column": "h_sys_bp", "unit": "mmHg", "recoding": "none", "notes": "Direct rename"},
    {"dataset": "framingham", "raw_column": "diaBP", "harmonized_column": "h_dia_bp", "unit": "mmHg", "recoding": "none", "notes": "Direct rename"},
    {"dataset": "framingham", "raw_column": "totChol", "harmonized_column": "h_total_cholesterol", "unit": "mg/dL", "recoding": "none", "notes": "Direct rename. 1.2% missing (MCAR)."},
    {"dataset": "framingham", "raw_column": "glucose", "harmonized_column": "h_glucose", "unit": "mg/dL", "recoding": "none", "notes": "9.2% missing (MAR-likely). Imputed downstream with train median."},
    {"dataset": "framingham", "raw_column": "TenYearCHD", "harmonized_column": "h_outcome_binary", "unit": "binary 0/1", "recoding": "none", "notes": "10-year CHD risk. Structurally binary but endpoint differs from Synthetic."},
]

_NHANES_MAPPING_TABLE = [
    {"dataset": "nhanes", "raw_column": "st", "harmonized_column": "h_sys_bp", "unit": "mmHg", "recoding": "zero→NaN", "notes": "NHANES BPXSY equivalent. Zero values replaced with NaN."},
    {"dataset": "nhanes", "raw_column": "dt", "harmonized_column": "h_dia_bp", "unit": "mmHg", "recoding": "zero→NaN", "notes": "NHANES BPXDI equivalent. Zero values replaced with NaN."},
    {"dataset": "nhanes", "raw_column": "tc", "harmonized_column": "h_total_cholesterol", "unit": "mg/dL", "recoding": "zero→NaN", "notes": "Total cholesterol. Zero values replaced with NaN."},
    {"dataset": "nhanes", "raw_column": "lbdldl", "harmonized_column": "h_ldl", "unit": "mg/dL", "recoding": "zero→NaN", "notes": "LDL cholesterol (Friedewald equation in NHANES)."},
    {"dataset": "nhanes", "raw_column": "lbxtr", "harmonized_column": "h_triglycerides", "unit": "mg/dL", "recoding": "zero→NaN", "notes": "Serum triglycerides."},
]


class HarmonizationLayer:
    """
    Applies harmonization, saves outputs, and generates all reports.
    """

    def run(self, datasets: dict[str, CohortDataset]) -> None:
        """
        Run full harmonization pipeline:
        1. Save harmonized CSVs to data/harmonized/
        2. Write mapping tables to data/mapping_tables/
        3. Write feature metadata to data/feature_metadata/
        4. Generate dataset audit report
        5. Generate feature availability matrix
        6. Generate harmonization report

        Parameters
        ----------
        datasets : dict[str, CohortDataset]
            Name → CohortDataset from DatasetRegistry.load_all()
        """
        ensure_dirs()
        logger.info("HarmonizationLayer: running for %d datasets.", len(datasets))

        self._save_harmonized_csvs(datasets)
        self._write_mapping_tables()
        self._write_feature_metadata(datasets)
        self._write_dataset_audit(datasets)
        self._write_feature_availability_matrix(datasets)
        self._write_harmonization_report(datasets)

        logger.info("Harmonization complete. All artefacts saved.")

    # ------------------------------------------------------------------
    # Harmonized CSVs
    # ------------------------------------------------------------------

    def _save_harmonized_csvs(self, datasets: dict[str, CohortDataset]) -> None:
        path_map = {
            DatasetNames.SYNTHETIC: DataPaths.SYNTHETIC_HARMONIZED,
            DatasetNames.FRAMINGHAM: DataPaths.FRAMINGHAM_HARMONIZED,
            DatasetNames.NHANES: DataPaths.NHANES_HARMONIZED,
        }
        for name, ds in datasets.items():
            out_path = path_map.get(name)
            if out_path:
                # Save only h_* columns + dataset-specific columns (not raw originals that duplicate h_*)
                h_cols = [c for c in ds.df_harmonized.columns if c.startswith("h_")]
                ds_cols = [c for c in ds.feature_schema.dataset_specific if c in ds.df_harmonized.columns]
                save_cols = h_cols + ds_cols
                ds.df_harmonized[save_cols].to_csv(out_path, index=False)
                logger.info("Saved harmonized %s -> %s (%d rows, %d cols)", name, out_path, len(ds.df_harmonized), len(save_cols))

    # ------------------------------------------------------------------
    # Mapping tables
    # ------------------------------------------------------------------

    def _write_mapping_tables(self) -> None:
        all_mappings = (
            _SYNTHETIC_MAPPING_TABLE
            + _FRAMINGHAM_MAPPING_TABLE
            + _NHANES_MAPPING_TABLE
        )
        full_df = pd.DataFrame(all_mappings)
        # Combined
        full_df.to_csv(DataDirs.MAPPING_TABLES / "all_mappings.csv", index=False)
        # Per-dataset
        for name, rows in [
            (DatasetNames.SYNTHETIC, _SYNTHETIC_MAPPING_TABLE),
            (DatasetNames.FRAMINGHAM, _FRAMINGHAM_MAPPING_TABLE),
            (DatasetNames.NHANES, _NHANES_MAPPING_TABLE),
        ]:
            pd.DataFrame(rows).to_csv(DataDirs.MAPPING_TABLES / f"{name}_mapping.csv", index=False)
        logger.info("Mapping tables written to %s", DataDirs.MAPPING_TABLES)

    # ------------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------------

    def _write_feature_metadata(self, datasets: dict[str, CohortDataset]) -> None:
        common_space = {
            "description": "Common harmonized feature space for AETHEL multi-cohort framework",
            "version": "3.0.0",
            "h_prefix_convention": "All harmonized features use the 'h_' prefix",
            "features": {
                "h_age": {"type": "continuous", "unit": "years", "source": ["synthetic", "framingham"]},
                "h_bmi": {"type": "continuous", "unit": "kg/m²", "source": ["synthetic", "framingham"]},
                "h_is_smoker": {"type": "binary", "unit": "0/1", "source": ["synthetic", "framingham"]},
                "h_sex_male": {"type": "binary", "unit": "0=Female,1=Male", "source": ["framingham"]},
                "h_sys_bp": {"type": "continuous", "unit": "mmHg", "source": ["framingham", "nhanes"]},
                "h_dia_bp": {"type": "continuous", "unit": "mmHg", "source": ["framingham", "nhanes"]},
                "h_total_cholesterol": {"type": "continuous", "unit": "mg/dL", "source": ["framingham", "nhanes"]},
                "h_ldl": {"type": "continuous", "unit": "mg/dL", "source": ["nhanes"]},
                "h_triglycerides": {"type": "continuous", "unit": "mg/dL", "source": ["nhanes"]},
                "h_glucose": {"type": "continuous", "unit": "mg/dL", "source": ["framingham"]},
                "h_outcome_binary": {"type": "binary", "unit": "0/1", "source": ["synthetic", "framingham"], "note": "Different clinical endpoints per dataset"},
            },
            "supervised_intersection": HarmonizedColumns.SUPERVISED_INTERSECTION,
            "cross_cohort_note": "For cross-cohort supervised experiments (Synthetic→Framingham), only supervised_intersection features are guaranteed available in both datasets.",
        }
        out = DataDirs.FEATURE_METADATA / "common_feature_space.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(common_space, fh, indent=2)

        # Per-dataset schemas
        for name, ds in datasets.items():
            schema_dict = {
                "dataset": name,
                "common_available": ds.feature_schema.common_available,
                "missing_from_common": ds.feature_schema.missing_from_common,
                "dataset_specific": ds.feature_schema.dataset_specific,
            }
            out = DataDirs.FEATURE_METADATA / f"{name}_schema.json"
            with out.open("w", encoding="utf-8") as fh:
                json.dump(schema_dict, fh, indent=2)

        logger.info("Feature metadata written to %s", DataDirs.FEATURE_METADATA)

    # ------------------------------------------------------------------
    # Dataset audit report
    # ------------------------------------------------------------------

    def _write_dataset_audit(self, datasets: dict[str, CohortDataset]) -> None:
        rows = []
        for name, ds in datasets.items():
            m = ds.metadata
            s = ds.feature_schema
            # Missing rates in raw data
            miss_pct = ds.df_raw.isnull().mean().mean() * 100
            rows.append({
                "dataset": name,
                "n": m.n,
                "outcome": m.outcome_col or "None",
                "outcome_type": m.outcome_type,
                "supervised": m.supervised,
                "event_rate_pct": round(m.event_rate * 100, 1) if m.event_rate else "N/A",
                "common_features_available": len(s.common_available),
                "common_features_missing": len(s.missing_from_common),
                "dataset_specific_features": len(s.dataset_specific),
                "overall_missing_pct": round(miss_pct, 2),
                "population": m.population,
                "domain": m.domain,
                "description": m.description,
            })
        df = pd.DataFrame(rows)
        df.to_csv(OutputPaths.DATASET_AUDIT, index=False)
        logger.info("Dataset audit saved to %s", OutputPaths.DATASET_AUDIT)

    # ------------------------------------------------------------------
    # Feature availability matrix
    # ------------------------------------------------------------------

    def _write_feature_availability_matrix(self, datasets: dict[str, CohortDataset]) -> None:
        all_h = HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY]
        rows = []
        for h_feat in all_h:
            row = {"harmonized_feature": h_feat}
            for name, ds in datasets.items():
                row[name] = "available" if h_feat in ds.feature_schema.common_available else "absent"
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(OutputPaths.FEATURE_AVAILABILITY_MATRIX, index=False)
        logger.info("Feature availability matrix saved to %s", OutputPaths.FEATURE_AVAILABILITY_MATRIX)

    # ------------------------------------------------------------------
    # Harmonization report
    # ------------------------------------------------------------------

    def _write_harmonization_report(self, datasets: dict[str, CohortDataset]) -> None:
        all_rows = (
            _SYNTHETIC_MAPPING_TABLE
            + _FRAMINGHAM_MAPPING_TABLE
            + _NHANES_MAPPING_TABLE
        )
        pd.DataFrame(all_rows).to_csv(OutputPaths.HARMONIZATION_REPORT, index=False)
        logger.info("Harmonization report saved to %s", OutputPaths.HARMONIZATION_REPORT)


def ensure_dirs() -> None:
    for d in (
        DataDirs.HARMONIZED, DataDirs.FEATURE_METADATA,
        DataDirs.MAPPING_TABLES, OutputDirs.REPORTS,
    ):
        d.mkdir(parents=True, exist_ok=True)
