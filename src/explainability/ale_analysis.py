"""
ale_analysis.py
===============
Accumulated Local Effects (ALE) plots.

ALE is a bias-corrected alternative to PDP for correlated features.
Unlike PDP (which marginalises over the entire feature distribution),
ALE conditions on local neighbourhoods, making it unbiased even when
features are correlated.

Uses the PyALE library: https://github.com/DanaJomar/PyALE

Reference:
    Apley & Zhu (2020). Visualizing the effects of predictor variables in black
    box supervised learning models. JRSS-B 82(4): 1059-1086.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _try_import_pyale():
    try:
        from PyALE import ale as pyale_fn
        return pyale_fn
    except ImportError:
        logger.error("PyALE not installed. Run: pip install PyALE")
        return None


class ALEAnalyser:
    """
    Computes and visualises ALE plots using PyALE.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    feature_names : list[str]
    model_name : str
    grid_size : int
        Number of quantile-based bins for continuous features. Default 50.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        model_name: str,
        grid_size: int = 50,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.grid_size = grid_size
        self._ale_results: dict[str, pd.DataFrame | None] = {}

    def compute(self, X: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
        """
        Compute ALE for all features.

        Returns
        -------
        dict[feature_name → ALE DataFrame (from PyALE) or None on failure]
        """
        pyale_fn = _try_import_pyale()
        if pyale_fn is None:
            return {}

        logger.info("ALEAnalyser: computing ALE for %s (%d features)", self.model_name, len(self.feature_names))

        for feat in self.feature_names:
            if feat not in X.columns:
                logger.warning("ALEAnalyser: feature '%s' not in X. Skipping.", feat)
                self._ale_results[feat] = None
                continue
            try:
                is_binary = set(X[feat].dropna().unique()).issubset({0, 1})
                feature_type = "discrete" if is_binary else "continuous"
                grid = min(self.grid_size, X[feat].nunique())

                ale_df = pyale_fn(
                    X=X,
                    model=self.model,
                    feature=[feat],
                    feature_type=feature_type,
                    grid_size=grid,
                    plot=False,
                    include_CI=True,
                    C=0.95,
                    predictors=self.feature_names,
                )
                self._ale_results[feat] = ale_df
                logger.debug("ALEAnalyser: completed ALE for '%s'", feat)
            except Exception as e:
                logger.warning("ALEAnalyser: failed for '%s': %s", feat, e)
                self._ale_results[feat] = None

        return self._ale_results

    def plot_ale(self, feature: str, out_path: Path) -> None:
        """
        Plot ALE curve for a single feature with 95% CI shading.
        """
        if feature not in self._ale_results or self._ale_results[feature] is None:
            logger.warning("No ALE results for '%s'. Skipping plot.", feature)
            return

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
        df = self._ale_results[feature]

        # PyALE DataFrame columns: x, eff (ALE values), optional lowerCI_eff, upperCI_eff
        fig, ax = plt.subplots(figsize=(7, 5))

        x_col = df.columns[0]   # PyALE puts feature values in first column
        eff_col = [c for c in df.columns if "eff" in c.lower() and "ci" not in c.lower()]
        eff_col = eff_col[0] if eff_col else df.columns[1]

        x_vals = df[x_col].values
        eff_vals = df[eff_col].values

        ax.plot(x_vals, eff_vals, color="steelblue", linewidth=2.5, label="ALE", zorder=5)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

        # CI shading if available
        lo_cols = [c for c in df.columns if "lower" in c.lower() or "lo" in c.lower()]
        hi_cols = [c for c in df.columns if "upper" in c.lower() or "hi" in c.lower()]
        if lo_cols and hi_cols:
            ax.fill_between(x_vals, df[lo_cols[0]].values, df[hi_cols[0]].values,
                            alpha=0.2, color="steelblue", label="95% CI")

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel("ALE (centred effect on predicted probability)", fontsize=10)
        ax.set_title(f"ALE: {feature} — {self.model_name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved ALE plot for '%s': %s", feature, out_path)

    def plot_all_features(self, out_dir: Path) -> None:
        """Save one ALE plot per feature and one combined grid figure."""
        out_dir.mkdir(parents=True, exist_ok=True)
        valid = {k: v for k, v in self._ale_results.items() if v is not None}
        if not valid:
            logger.warning("ALEAnalyser: no valid ALE results. Skipping grid plot.")
            return

        # Individual plots
        for feat in valid:
            safe = feat.replace(" ", "_").lower()
            self.plot_ale(feat, out_dir / f"ale_{safe}.png")

        # Combined grid
        n = len(valid)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

        for ax_idx, (feat, df) in enumerate(valid.items()):
            row, col = divmod(ax_idx, ncols)
            ax = axes[row][col]
            x_col = df.columns[0]
            eff_col = [c for c in df.columns if "eff" in c.lower() and "ci" not in c.lower()]
            eff_col = eff_col[0] if eff_col else df.columns[1]
            ax.plot(df[x_col].values, df[eff_col].values, color="steelblue", linewidth=2)
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel("ALE", fontsize=9)
            ax.set_title(feat, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for ax_idx in range(n, nrows * ncols):
            row, col = divmod(ax_idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Accumulated Local Effects Grid — {self.model_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(out_dir / "ale_grid.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved ALE grid: %s", out_dir / "ale_grid.png")
