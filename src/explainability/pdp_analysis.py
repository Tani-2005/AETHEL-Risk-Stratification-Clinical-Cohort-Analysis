"""
pdp_analysis.py
===============
Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) curves.

PDP shows the marginal effect of one or two features on the predicted outcome,
averaging over all other features. ICE curves show the effect for each individual
sample, revealing heterogeneity that PDP averages away.

Both are computed via sklearn.inspection.partial_dependence to ensure
consistency with the training preprocessing pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class PDPAnalyser:
    """
    Computes and visualises Partial Dependence Plots and ICE curves.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    feature_names : list[str]
    model_name : str
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        model_name: str,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self._pdp_results: dict[str, dict] = {}

    def compute(self, X: pd.DataFrame, percentiles: tuple[float, float] = (0.05, 0.95)) -> dict:
        """
        Compute PDP + ICE for all features.

        Returns dict: feature → {grid_values, pdp_values, ice_values}
        """
        logger.info("PDPAnalyser: computing PDP/ICE for %s (%d features)", self.model_name, len(self.feature_names))
        for i, feat in enumerate(self.feature_names):
            try:
                result = partial_dependence(
                    self.model, X,
                    features=[i],
                    kind="both",                 # both average PDP and ICE
                    percentiles=percentiles,
                    grid_resolution=50,
                )
                self._pdp_results[feat] = {
                    "grid_values": result["grid_values"][0],     # 1-D array of x-axis values
                    "pdp_values": result["average"][0],          # mean prediction at each grid point
                    "ice_values": result["individual"][0],       # (n_samples, n_grid) ICE lines
                }
            except Exception as e:
                logger.warning("PDPAnalyser: failed for feature '%s': %s", feat, e)
        return self._pdp_results

    def plot_pdp_ice(self, feature: str, out_path: Path, n_ice_lines: int = 50) -> None:
        """
        Plot PDP with ICE curves for a single feature.

        ICE lines are plotted in light grey; the mean PDP line is plotted bold red.
        """
        if feature not in self._pdp_results:
            logger.warning("No PDP results for feature '%s'. Call compute() first.", feature)
            return

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
        res = self._pdp_results[feature]
        grid = res["grid_values"]
        pdp = res["pdp_values"]
        ice = res["ice_values"]

        fig, ax = plt.subplots(figsize=(7, 5))

        # ICE lines (subsample for readability)
        n_show = min(n_ice_lines, ice.shape[0])
        rng = np.random.default_rng(42)
        indices = rng.choice(ice.shape[0], size=n_show, replace=False)
        for idx in indices:
            ax.plot(grid, ice[idx], color="lightsteelblue", alpha=0.3, linewidth=0.7)

        # Average PDP line
        ax.plot(grid, pdp, color="crimson", linewidth=2.5, label="PDP (mean)", zorder=5)
        ax.fill_between(grid,
                        pdp - np.std(ice, axis=0),
                        pdp + np.std(ice, axis=0),
                        color="crimson", alpha=0.1, label="±1 SD")

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel("Partial Dependence\n(predicted probability)", fontsize=11)
        ax.set_title(f"PDP + ICE: {feature} — {self.model_name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved PDP/ICE plot for '%s': %s", feature, out_path)

    def plot_all_features(self, out_dir: Path) -> None:
        """Save one PDP+ICE plot per feature and one combined grid figure."""
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(self._pdp_results)
        if n == 0:
            return

        # Individual plots
        for feat in self._pdp_results:
            safe = feat.replace(" ", "_").lower()
            self.plot_pdp_ice(feat, out_dir / f"pdp_ice_{safe}.png")

        # Combined grid
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        for ax_idx, (feat, res) in enumerate(self._pdp_results.items()):
            row, col = divmod(ax_idx, ncols)
            ax = axes[row][col]
            grid = res["grid_values"]
            pdp = res["pdp_values"]
            ice = res["ice_values"]
            n_show = min(30, ice.shape[0])
            rng = np.random.default_rng(42)
            for idx in rng.choice(ice.shape[0], size=n_show, replace=False):
                ax.plot(grid, ice[idx], color="lightsteelblue", alpha=0.25, linewidth=0.5)
            ax.plot(grid, pdp, color="crimson", linewidth=2, label="PDP")
            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel("Pred. prob.", fontsize=9)
            ax.set_title(feat, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide unused axes
        for ax_idx in range(n, nrows * ncols):
            row, col = divmod(ax_idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"PDP + ICE Grid — {self.model_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(out_dir / "pdp_ice_grid.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved PDP/ICE grid: %s", out_dir / "pdp_ice_grid.png")
