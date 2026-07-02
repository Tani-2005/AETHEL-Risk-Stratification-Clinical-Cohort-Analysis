"""
cohort_comparison.py
====================
Cohort comparison, domain shift analysis, and cross-cohort summary reports.
"""
from __future__ import annotations
import textwrap
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from src.datasets.base_loader import CohortDataset
from src.utils.constants import HarmonizedColumns
from src.utils.logging_setup import get_logger
from src.utils.paths import OutputDirs, OutputPaths

logger = get_logger(__name__)


class CohortComparison:
    """
    Compares distributions across cohorts and generates domain shift reports.
    """

    def run(self, datasets: dict[str, CohortDataset]) -> None:
        OutputDirs.REPORTS.mkdir(parents=True, exist_ok=True)
        names = list(datasets.keys())
        comparison_rows: list[dict] = []
        domain_shift_rows: list[dict] = []

        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                ds1, ds2 = datasets[n1], datasets[n2]
                common = list(
                    set(ds1.feature_schema.common_available)
                    & set(ds2.feature_schema.common_available)
                    - {HarmonizedColumns.OUTCOME_BINARY}
                )
                for feat in common:
                    row = {"dataset_1": n1, "dataset_2": n2, "feature": feat}
                    s1 = ds1.df_harmonized[feat].dropna()
                    s2 = ds2.df_harmonized[feat].dropna()
                    if feat in HarmonizedColumns.CONTINUOUS and len(s1) > 0 and len(s2) > 0:
                        stat, pval = ks_2samp(s1, s2)
                        row.update({"test": "KS", "statistic": round(stat, 4),
                                    "p_value": round(pval, 4),
                                    "significant_drift": bool(pval < 0.05),
                                    "n1_mean": round(s1.mean(), 3), "n2_mean": round(s2.mean(), 3),
                                    "n1_std": round(s1.std(), 3), "n2_std": round(s2.std(), 3)})
                        domain_shift_rows.append({"pair": f"{n1}->{n2}", "feature": feat,
                                                  "ks_statistic": round(stat, 4), "p_value": round(pval, 4),
                                                  "drift_flag": "HIGH" if stat > 0.3 else ("MODERATE" if stat > 0.1 else "LOW")})
                    elif feat in HarmonizedColumns.BINARY and len(s1) > 0 and len(s2) > 0:
                        r1, r2 = s1.mean(), s2.mean()
                        row.update({"test": "rate_diff", "statistic": round(abs(r1 - r2), 4),
                                    "p_value": None, "significant_drift": abs(r1 - r2) > 0.10,
                                    "n1_mean": round(r1, 4), "n2_mean": round(r2, 4),
                                    "n1_std": None, "n2_std": None})
                    comparison_rows.append(row)

        # Dataset-level summary
        summary_rows = []
        for name, ds in datasets.items():
            m = ds.metadata
            hdf = ds.df_harmonized
            row = {"dataset": name, "n": m.n, "supervised": m.supervised,
                   "event_rate_pct": round(m.event_rate * 100, 1) if m.event_rate else "N/A",
                   "common_features": len(ds.feature_schema.common_available)}
            for feat in HarmonizedColumns.ALL_FEATURES:
                if feat in hdf.columns and hdf[feat].notna().any():
                    row[f"{feat}_mean"] = round(hdf[feat].mean(), 3)
                    row[f"{feat}_miss_pct"] = round(hdf[feat].isnull().mean() * 100, 1)
            summary_rows.append(row)

        pd.DataFrame(comparison_rows).to_csv(OutputPaths.COHORT_COMPARISON_REPORT, index=False)
        pd.DataFrame(domain_shift_rows).to_csv(OutputPaths.DOMAIN_SHIFT_REPORT, index=False)
        pd.DataFrame(summary_rows).to_csv(OutputPaths.CROSS_COHORT_SUMMARY, index=False)
        logger.info("Cohort comparison reports saved to %s", OutputDirs.REPORTS)

        # Log high-drift features
        high_drift = [r for r in domain_shift_rows if r.get("drift_flag") == "HIGH"]
        if high_drift:
            logger.warning("HIGH domain shift features: %s", [(r["pair"], r["feature"]) for r in high_drift])

