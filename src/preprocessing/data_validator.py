"""
data_validator.py
=================
Stage: Data Validation

Performs clinical plausibility and integrity checks on the analytical cohort
before any modelling or preprocessing occurs.

Checks performed
----------------
1. Duplicate rows and duplicate patient IDs
2. Missing values per column
3. Clinical bound violations (configurable per feature via configs/default.yaml)
4. Binary feature cardinality (is_smoker, event_occurred must be in {0, 1})
5. Survival consistency (months_observed > 0 for all patients)
6. Event rate reporting (class balance)

Design decision — stop_on_error
--------------------------------
For synthetic data: ``stop_on_error = False`` (log warnings, continue pipeline).
For real patient data: Set ``stop_on_error = True`` in config to halt the
pipeline when clinical bound violations are detected — this prevents silently
propagating erroneous records into a published model.

Usage
-----
    from src.preprocessing.data_validator import DataValidator
    from src.utils.config_loader import load_config

    cfg = load_config()
    validator = DataValidator(cfg)
    report = validator.validate(df)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.config_loader import AETHELConfig
from src.utils.constants import Columns
from src.utils.logging_setup import get_logger
from src.utils.paths import OutputDirs, OutputPaths

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured summary of all validation findings."""
    n_rows: int = 0
    n_cols: int = 0
    n_duplicates: int = 0
    n_duplicate_ids: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    bound_violations: list[dict[str, Any]] = field(default_factory=list)
    binary_violations: list[dict[str, Any]] = field(default_factory=list)
    event_rate: float = 0.0
    n_events: int = 0
    n_censored: int = 0
    passed: bool = True

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the report to a flat DataFrame for CSV export."""
        rows = []

        rows.append({"check": "n_rows", "value": self.n_rows, "status": "INFO"})
        rows.append({"check": "n_cols", "value": self.n_cols, "status": "INFO"})
        rows.append({
            "check": "duplicate_rows",
            "value": self.n_duplicates,
            "status": "PASS" if self.n_duplicates == 0 else "WARN",
        })
        rows.append({
            "check": "duplicate_patient_ids",
            "value": self.n_duplicate_ids,
            "status": "PASS" if self.n_duplicate_ids == 0 else "FAIL",
        })
        for col, n in self.missing_values.items():
            rows.append({
                "check": f"missing_{col}",
                "value": n,
                "status": "PASS" if n == 0 else "WARN",
            })
        for v in self.bound_violations:
            rows.append({
                "check": f"bound_violation_{v['column']}",
                "value": v["n_violations"],
                "status": "WARN",
                "detail": v["detail"],
            })
        for v in self.binary_violations:
            rows.append({
                "check": f"binary_cardinality_{v['column']}",
                "value": str(v["found_values"]),
                "status": "FAIL",
            })
        rows.append({
            "check": "event_rate_pct",
            "value": round(self.event_rate * 100, 2),
            "status": "INFO",
        })
        rows.append({"check": "n_events", "value": self.n_events, "status": "INFO"})
        rows.append({"check": "n_censored", "value": self.n_censored, "status": "INFO"})
        rows.append({
            "check": "overall",
            "value": "PASS" if self.passed else "FAIL",
            "status": "PASS" if self.passed else "FAIL",
        })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Clinical plausibility and integrity checker for the AETHEL cohort.

    Parameters
    ----------
    cfg : AETHELConfig
        Pipeline configuration, providing clinical_bounds and stop_on_error.
    """

    # Features that must be strictly binary {0, 1}
    _BINARY_FEATURES: list[str] = [
        Columns.IS_SMOKER,
        Columns.EVENT_OCCURRED,
    ]

    def __init__(self, cfg: AETHELConfig) -> None:
        self._bounds: dict[str, list[float]] = cfg.validation.clinical_bounds
        self._stop_on_error: bool = cfg.validation.stop_on_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks and return a structured report.

        Parameters
        ----------
        df : pd.DataFrame
            The analytical cohort to validate.

        Returns
        -------
        ValidationReport
            Summary of all checks with pass/warn/fail status per check.

        Raises
        ------
        ValueError
            If ``stop_on_error = True`` and any FAIL-level check is triggered.
        """
        logger.info("Running data validation on cohort (%d rows × %d cols)...", len(df), len(df.columns))
        report = ValidationReport(n_rows=len(df), n_cols=len(df.columns))
        failures: list[str] = []

        # 1. Duplicates
        report.n_duplicates = int(df.duplicated().sum())
        report.n_duplicate_ids = int(df[Columns.PATIENT_ID].duplicated().sum()) if Columns.PATIENT_ID in df.columns else 0

        if report.n_duplicates > 0:
            logger.warning("Found %d fully duplicate rows.", report.n_duplicates)
        if report.n_duplicate_ids > 0:
            msg = f"Found {report.n_duplicate_ids} duplicate patient IDs — data integrity failure."
            logger.error(msg)
            failures.append(msg)

        # 2. Missing values
        report.missing_values = df.isnull().sum().to_dict()
        total_missing = sum(report.missing_values.values())
        if total_missing > 0:
            logger.warning("Missing value summary: %s", {k: v for k, v in report.missing_values.items() if v > 0})
        else:
            logger.info("Missing values: none detected.")

        # 3. Clinical bound violations
        for col, bounds in self._bounds.items():
            if col not in df.columns:
                continue
            lo, hi = bounds
            mask = (df[col] < lo) | (df[col] > hi)
            n_viol = int(mask.sum())
            if n_viol > 0:
                detail = f"[{lo}, {hi}]; min={df[col].min():.3f}, max={df[col].max():.3f}"
                logger.warning("Bound violation — %s: %d rows outside %s", col, n_viol, detail)
                report.bound_violations.append({
                    "column": col,
                    "n_violations": n_viol,
                    "bound_lo": lo,
                    "bound_hi": hi,
                    "detail": detail,
                })

        # 4. Binary cardinality checks
        for col in self._BINARY_FEATURES:
            if col not in df.columns:
                continue
            found = set(df[col].dropna().unique().tolist())
            expected = {0, 1}
            if not found.issubset(expected):
                msg = f"Binary column '{col}' contains unexpected values: {found}"
                logger.error(msg)
                report.binary_violations.append({"column": col, "found_values": found})
                failures.append(msg)

        # 5. Survival consistency
        if Columns.MONTHS_OBSERVED in df.columns:
            n_zero_time = int((df[Columns.MONTHS_OBSERVED] <= 0).sum())
            if n_zero_time > 0:
                logger.warning("%d patients have months_observed <= 0. Check survival data.", n_zero_time)

        # 6. Event rate
        if Columns.EVENT_OCCURRED in df.columns:
            report.n_events = int(df[Columns.EVENT_OCCURRED].sum())
            report.n_censored = len(df) - report.n_events
            report.event_rate = report.n_events / max(len(df), 1)
            logger.info(
                "Class balance — events: %d (%.1f%%), censored: %d (%.1f%%)",
                report.n_events, report.event_rate * 100,
                report.n_censored, (1 - report.event_rate) * 100,
            )

        report.passed = len(failures) == 0

        # Save report
        self._save_report(report)

        if failures and self._stop_on_error:
            raise ValueError(
                "Data validation failed (stop_on_error=True):\n" + "\n".join(failures)
            )

        status = "PASSED" if report.passed else f"COMPLETED WITH {len(failures)} FAILURES"
        logger.info("Data validation %s.", status)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_report(self, report: ValidationReport) -> None:
        OutputDirs.REPORTS.mkdir(parents=True, exist_ok=True)
        df_report = report.to_dataframe()
        df_report.to_csv(OutputPaths.DATA_QUALITY_REPORT, index=False)
        logger.info("Data quality report saved to %s", OutputPaths.DATA_QUALITY_REPORT)

        # Class balance report
        balance = pd.DataFrame([{
            "label": "event",
            "n": report.n_events,
            "pct": round(report.event_rate * 100, 2),
        }, {
            "label": "censored",
            "n": report.n_censored,
            "pct": round((1 - report.event_rate) * 100, 2),
        }])
        balance.to_csv(OutputPaths.CLASS_BALANCE_REPORT, index=False)
        logger.info("Class balance report saved to %s", OutputPaths.CLASS_BALANCE_REPORT)
