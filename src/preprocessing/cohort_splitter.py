"""
cohort_splitter.py
==================
Stage: Stratified Train / Val / Test Split

Splits the analytical cohort into non-overlapping train, validation, and
test sets using stratified random sampling on the survival outcome.

Design decisions
----------------
**Stratification on event_occurred:**
  The event rate is 23.4%.  Without stratification, random splits could
  produce sets with significantly different event rates, invalidating
  any comparison of model performance across splits.  Stratification
  ensures each split mirrors the original event distribution.

**Why 70 / 15 / 15:**
  With n=1,000, a 70/15/15 split gives 700 train, 150 val, 150 test.
  150 events × 23.4% ≈ 35 events per val/test split — sufficient for
  survival analysis (rule of thumb: ≥10 events per covariate; we have
  8 covariates → need ≥80 events in train, which ≈ 700 × 0.234 = 164).

**No data leakage:**
  The split is performed BEFORE any preprocessing fitting.
  Preprocessing transformers (scaler, pollution index stats) are
  subsequently fit ONLY on the training fold.

**Reproducibility:**
  Split uses the global Python seed from config.  Re-running with the
  same seed always produces the same split.

Usage
-----
    from src.preprocessing.cohort_splitter import CohortSplitter
    train, val, test = CohortSplitter(cfg).split(df)
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import AETHELConfig
from src.utils.constants import Columns
from src.utils.logging_setup import get_logger
from src.utils.paths import DataDirs, DataPaths, OutputDirs, OutputPaths

logger = get_logger(__name__)


class CohortSplitter:
    """
    Stratified train / val / test splitter for the AETHEL cohort.

    Parameters
    ----------
    cfg : AETHELConfig
        Configuration providing split ratios, stratification column, and seed.
    """

    def __init__(self, cfg: AETHELConfig) -> None:
        self._train_size: float = cfg.preprocessing.train_size
        self._val_size: float = cfg.preprocessing.val_size
        self._test_size: float = cfg.preprocessing.test_size
        self._stratify_col: str = cfg.preprocessing.stratify_on
        self._seed: int = cfg.seeds.python

        total = self._train_size + self._val_size + self._test_size
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"train + val + test must sum to 1.0, got {total:.3f}. "
                "Check preprocessing.train_size/val_size/test_size in config."
            )

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the cohort into train, val, and test sets.

        The split is stratified on ``event_occurred`` to preserve class
        balance across all three folds.

        Parameters
        ----------
        df : pd.DataFrame
            Full analytical cohort (all patients).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, val_df, test_df) with no patient overlap.

        Side-effects
        ------------
        Writes:
            - ``data/processed/train.csv``
            - ``data/processed/val.csv``
            - ``data/processed/test.csv``
            - ``outputs/reports/split_summary.csv``
        """
        stratify = df[self._stratify_col] if self._stratify_col in df.columns else None

        logger.info(
            "Splitting cohort (%d patients) -> train %.0f%% / val %.0f%% / test %.0f%%",
            len(df),
            self._train_size * 100,
            self._val_size * 100,
            self._test_size * 100,
        )

        # Step 1: carve out test set
        val_ratio_of_remaining = self._val_size / (self._train_size + self._val_size)

        train_val, test = train_test_split(
            df,
            test_size=self._test_size,
            stratify=stratify,
            random_state=self._seed,
        )

        # Step 2: split remainder into train / val
        stratify_tv = train_val[self._stratify_col] if self._stratify_col in train_val.columns else None

        train, val = train_test_split(
            train_val,
            test_size=val_ratio_of_remaining,
            stratify=stratify_tv,
            random_state=self._seed,
        )

        self._log_split_stats(train, val, test)
        self._save_splits(train, val, test)
        self._save_split_summary(train, val, test)

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_split_stats(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        for name, split in [("train", train), ("val", val), ("test", test)]:
            n = len(split)
            if Columns.EVENT_OCCURRED in split.columns:
                n_events = int(split[Columns.EVENT_OCCURRED].sum())
                rate = n_events / max(n, 1) * 100
                logger.info(
                    "  %-6s: %4d patients | %3d events | event rate %.1f%%",
                    name, n, n_events, rate,
                )
            else:
                logger.info("  %-6s: %4d patients", name, n)

    def _save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        DataDirs.PROCESSED.mkdir(parents=True, exist_ok=True)
        train.to_csv(DataPaths.TRAIN, index=False)
        val.to_csv(DataPaths.VAL, index=False)
        test.to_csv(DataPaths.TEST, index=False)
        logger.info(
            "Splits saved -> train: %s | val: %s | test: %s",
            DataPaths.TRAIN, DataPaths.VAL, DataPaths.TEST,
        )

    def _save_split_summary(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        OutputDirs.REPORTS.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, split in [("train", train), ("val", val), ("test", test)]:
            n = len(split)
            n_ev = int(split[Columns.EVENT_OCCURRED].sum()) if Columns.EVENT_OCCURRED in split.columns else None
            rows.append({
                "split": name,
                "n_patients": n,
                "n_events": n_ev,
                "event_rate_pct": round(n_ev / max(n, 1) * 100, 2) if n_ev is not None else None,
                "pct_of_total": round(n / (len(train) + len(val) + len(test)) * 100, 1),
            })
        pd.DataFrame(rows).to_csv(OutputPaths.SPLIT_SUMMARY, index=False)
        logger.info("Split summary saved to %s", OutputPaths.SPLIT_SUMMARY)
