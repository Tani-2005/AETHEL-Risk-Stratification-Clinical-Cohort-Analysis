"""
interaction_analysis.py
=======================
SHAP interaction value analysis for tree-based models.

SHAP interaction values decompose each prediction into main effects
and pairwise interaction effects. Only supported for tree-based models
(RandomForest, XGBoost, LightGBM, DecisionTree) via TreeExplainer.

The module auto-detects model type and gracefully skips analysis for
linear or kernel-based models with a clear log message.

Reference:
    Lundberg et al. (2020). From local explanations to global understanding
    with explainable AI for trees. Nature Machine Intelligence 2: 56-67.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_TREE_MODEL_TYPES = frozenset({
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
})


def is_tree_model(model: Any) -> bool:
    return type(model).__name__ in _TREE_MODEL_TYPES


class InteractionAnalyser:
    """
    Computes SHAP interaction values and visualises feature interactions.

    Only runs for tree-based models. Calling compute() on a non-tree model
    logs a warning and returns an empty result.

    Parameters
    ----------
    model : fitted tree-based classifier
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

        self._interaction_matrix: Optional[np.ndarray] = None   # (p, p) mean |interaction|
        self._top_interactions: list[dict] = []
        self._supported = is_tree_model(model)

    def is_supported(self) -> bool:
        return self._supported

    def compute(self, X: pd.DataFrame, max_samples: int = 200) -> np.ndarray | None:
        """
        Compute SHAP interaction values.

        Parameters
        ----------
        X : pd.DataFrame — data to explain (subsample for efficiency)
        max_samples : int — maximum rows to use (interaction values are O(n*p²))

        Returns
        -------
        np.ndarray, shape (p, p) of mean |SHAP interaction values|
        OR None if not supported.
        """
        if not self._supported:
            logger.info(
                "InteractionAnalyser: %s is not a tree model — skipping interaction analysis.",
                type(self.model).__name__,
            )
            return None

        n_use = min(max_samples, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n_use, replace=False)
        X_sub = X.iloc[idx].reset_index(drop=True)

        logger.info(
            "InteractionAnalyser: computing SHAP interactions for %s (%d samples, %d features)",
            self.model_name, n_use, len(self.feature_names),
        )
        try:
            explainer = shap.TreeExplainer(self.model)
            # shap_interaction_values: (n, p, p) — raw interactions
            raw_interactions = explainer.shap_interaction_values(X_sub)

            # For RF: list[class_0_array, class_1_array]; each (n, p, p)
            # For XGB/LGBM: ndarray (n, p, p)
            if isinstance(raw_interactions, list):
                interactions = np.asarray(raw_interactions[1])
            else:
                interactions = np.asarray(raw_interactions)
                if interactions.ndim == 4:
                    interactions = interactions[:, :, :, 1]

            # Mean absolute interaction: (p, p)
            self._interaction_matrix = np.abs(interactions).mean(axis=0)

            # Top interactions (off-diagonal pairs)
            p = len(self.feature_names)
            pairs = []
            for i in range(p):
                for j in range(i + 1, p):
                    strength = (self._interaction_matrix[i, j] + self._interaction_matrix[j, i]) / 2.0
                    pairs.append({
                        "feature_a": self.feature_names[i],
                        "feature_b": self.feature_names[j],
                        "interaction_strength": float(strength),
                    })
            self._top_interactions = sorted(pairs, key=lambda x: x["interaction_strength"], reverse=True)

            logger.info(
                "InteractionAnalyser: top interaction: %s × %s (strength=%.4f)",
                self._top_interactions[0]["feature_a"] if self._top_interactions else "n/a",
                self._top_interactions[0]["feature_b"] if self._top_interactions else "n/a",
                self._top_interactions[0]["interaction_strength"] if self._top_interactions else 0,
            )
            return self._interaction_matrix

        except Exception as e:
            logger.error("InteractionAnalyser: failed — %s", e)
            return None

    def get_top_interactions(self, k: int = 10) -> list[dict]:
        """Return top-k interaction pairs sorted by strength."""
        return self._top_interactions[:k]

    def plot_interaction_heatmap(self, out_path: Path) -> None:
        """
        Publication-ready symmetric heatmap of mean |SHAP interaction values|.
        """
        if self._interaction_matrix is None:
            logger.warning("No interaction matrix. Call compute() first.")
            return

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        p = len(self.feature_names)
        fig, ax = plt.subplots(figsize=(max(5, p), max(4, p - 1)))

        mat = self._interaction_matrix.copy()
        # Symmetrise for cleaner display
        sym_mat = (mat + mat.T) / 2.0

        im = ax.imshow(sym_mat, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="mean |SHAP interaction value|", shrink=0.8)

        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(self.feature_names, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(self.feature_names, fontsize=9)

        # Annotate cells
        for i in range(p):
            for j in range(p):
                val = sym_mat[i, j]
                text_color = "white" if val > sym_mat.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, color=text_color)

        ax.set_title(f"SHAP Interaction Heatmap — {self.model_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved interaction heatmap: %s", out_path)

    def plot_interaction_dependence(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature_a: str,
        feature_b: str,
        out_path: Path,
    ) -> None:
        """
        SHAP dependence plot for feature_a coloured by feature_b (interaction context).
        """
        if feature_a not in self.feature_names or feature_b not in self.feature_names:
            return

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
        idx_a = self.feature_names.index(feature_a)
        idx_b = self.feature_names.index(feature_b)

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            X[feature_a], shap_values[:, idx_a],
            c=X[feature_b], cmap="RdYlBu_r", alpha=0.7, s=20, edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(feature_b, fontsize=10)
        ax.set_xlabel(feature_a, fontsize=11)
        ax.set_ylabel(f"SHAP value for {feature_a}", fontsize=11)
        ax.set_title(f"Interaction: {feature_a} × {feature_b} — {self.model_name}", fontsize=12)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved interaction dependence plot (%s × %s): %s", feature_a, feature_b, out_path)

    def plot_top_interaction_dependences(
        self, X: pd.DataFrame, shap_values: np.ndarray, out_dir: Path, k: int = 3
    ) -> None:
        """Save dependence plots for top-k interaction pairs."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for pair in self._top_interactions[:k]:
            fa, fb = pair["feature_a"], pair["feature_b"]
            safe = f"{fa}_{fb}".replace(" ", "_").lower()
            self.plot_interaction_dependence(
                X, shap_values, fa, fb, out_dir / f"interaction_{safe}.png"
            )
