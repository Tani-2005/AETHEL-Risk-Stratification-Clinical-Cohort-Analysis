"""
permutation_analysis.py
=======================
Multi-repeat permutation importance for model-agnostic feature attribution.

Permutation importance is computed by randomly shuffling each feature and
measuring the resulting drop in model performance (ROC-AUC). Multiple repeats
produce mean ± std estimates, enabling uncertainty quantification.

This is complementary to SHAP: permutation importance is purely model-agnostic
and does not assume feature independence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class PermutationAnalyser:
    """
    Computes and visualises permutation importance for a fitted classifier.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    feature_names : list[str]
    model_name : str
    n_repeats : int
        Number of permutation repeats per feature. Higher = more stable.
        Default 30 is recommended for publication-quality estimates.
    seed : int
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        model_name: str,
        n_repeats: int = 30,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.n_repeats = n_repeats
        self.seed = seed

        self._results: dict[str, Any] = {}
        self._df: pd.DataFrame | None = None

    def compute(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compute permutation importance.

        Returns
        -------
        pd.DataFrame with columns:
            feature, mean_importance, std_importance, ci_lower_95, ci_upper_95
        Sorted descending by mean_importance.
        """
        logger.info(
            "PermutationAnalyser: computing importance for %s (%d repeats, %d samples)",
            self.model_name, self.n_repeats, len(X),
        )
        result = permutation_importance(
            self.model, X, y,
            n_repeats=self.n_repeats,
            random_state=self.seed,
            scoring="roc_auc",
            n_jobs=1,
        )
        self._results = result

        # Build summary DataFrame
        rows = []
        for i, feat in enumerate(self.feature_names):
            vals = result.importances[i]  # shape (n_repeats,)
            rows.append({
                "feature": feat,
                "mean_importance": float(np.mean(vals)),
                "std_importance": float(np.std(vals)),
                "ci_lower_95": float(np.percentile(vals, 2.5)),
                "ci_upper_95": float(np.percentile(vals, 97.5)),
                "median_importance": float(np.median(vals)),
            })

        self._df = (
            pd.DataFrame(rows)
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("PermutationAnalyser: done. Top feature: %s", self._df.iloc[0]["feature"])
        return self._df

    def get_ranking(self) -> pd.Series:
        """Return feature ranking as a Series indexed by feature name, sorted descending."""
        if self._df is None:
            raise RuntimeError("Call compute() first.")
        return self._df.set_index("feature")["mean_importance"].sort_values(ascending=False)

    def plot_bar(self, out_path: Path) -> None:
        """
        Publication-ready bar chart with 95% CI error bars.
        """
        if self._df is None:
            raise RuntimeError("Call compute() first.")

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
        df = self._df.copy()
        n = len(df)

        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.55 + 1)))
        y_pos = np.arange(n)
        colors = plt.cm.coolwarm_r(np.linspace(0.1, 0.9, n))  # type: ignore[attr-defined]

        # Error bars: asymmetric 95% CI
        xerr_lo = df["mean_importance"] - df["ci_lower_95"]
        xerr_hi = df["ci_upper_95"] - df["mean_importance"]

        ax.barh(
            y_pos, df["mean_importance"][::-1].values,
            xerr=[xerr_lo[::-1].values, xerr_hi[::-1].values],
            color=colors[::-1], edgecolor="white", capsize=4, linewidth=0.8,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature"][::-1].values)
        ax.set_xlabel("Permutation Importance\n(mean drop in ROC-AUC ± 95% CI)", fontsize=11)
        ax.set_title(f"Permutation Importance — {self.model_name}", fontsize=13)
        ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved permutation importance plot: %s", out_path)

    def plot_violin(self, out_path: Path) -> None:
        """
        Violin plot showing full distribution of permutation importance across repeats.
        """
        if self._results is None:
            raise RuntimeError("Call compute() first.")

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        n = len(self.feature_names)
        fig, ax = plt.subplots(figsize=(max(6, n * 1.1), 5))

        data = [self._results.importances[i] for i in range(n)]
        # Sort by mean descending
        order = np.argsort([np.mean(d) for d in data])[::-1]
        data_sorted = [data[i] for i in order]
        labels_sorted = [self.feature_names[i] for i in order]

        parts = ax.violinplot(data_sorted, vert=True, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(labels_sorted, rotation=30, ha="right")
        ax.set_ylabel("Drop in ROC-AUC", fontsize=11)
        ax.set_title(
            f"Permutation Importance Distribution ({self.n_repeats} repeats) — {self.model_name}",
            fontsize=12,
        )
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved permutation violin plot: %s", out_path)
