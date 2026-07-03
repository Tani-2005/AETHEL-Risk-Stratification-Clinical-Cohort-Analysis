"""
consensus_analysis.py
=====================
Cross-method consensus and explanation agreement scoring.

Compares feature importance rankings from multiple sources:
  - SHAP mean absolute values
  - Permutation importance
  - Model-native importances (LR coefficients, RF/XGB/LGBM feature_importances_)

Metrics (all formulae documented below):
  1. Feature Agreement Score (FAS):
     Mean pairwise Spearman rank correlation between all importance sources.
     Formula: FAS = mean(Spearman_ρ(rank_i, rank_j)) for all i≠j pairs
     Range: [-1, 1]. Value ≥ 0.8 indicates strong cross-method agreement.

  2. Top-k Overlap (TkO):
     Jaccard similarity of top-k feature sets across source pairs.
     Formula: TkO(k) = mean(|top_k_i ∩ top_k_j| / |top_k_i ∪ top_k_j|)
     Range: [0, 1]. Value ≥ 0.6 indicates consistent top-k identification.

  3. Consensus Index (CI):
     Weighted mean rank across all sources (equal weights unless specified).
     Formula: CI_feature = mean(rank_source(feature)) across sources
     Lower CI = higher consensus importance.

  4. Explanation Coverage (EC):
     Fraction of total features appearing in the top-k of at least one source.
     Formula: EC(k) = |union(top_k over all sources)| / n_features
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _extract_native_importance(model: Any, feature_names: list[str]) -> Optional[pd.Series]:
    """
    Extract model-native feature importance as a pd.Series.
    Returns None if the model type doesn't expose importance natively.
    """
    mtype = type(model).__name__

    if mtype == "LogisticRegression":
        coefs = np.abs(model.coef_[0])
        return pd.Series(coefs, index=feature_names)

    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)

    return None


class ConsensusAnalyser:
    """
    Measures cross-method explanation agreement and builds consensus rankings.

    Parameters
    ----------
    feature_names : list[str]
    model_name : str
    """

    def __init__(self, feature_names: list[str], model_name: str) -> None:
        self.feature_names = feature_names
        self.model_name = model_name
        self._sources: dict[str, pd.Series] = {}
        self._report: dict = {}

    def add_source(self, name: str, importance: pd.Series) -> None:
        """Register an importance source. importance is indexed by feature_name."""
        aligned = importance.reindex(self.feature_names).fillna(0.0)
        self._sources[name] = aligned / (aligned.abs().max() + 1e-10)  # normalise to [0,1]
        logger.debug("ConsensusAnalyser: added source '%s'", name)

    def add_model_native(self, model: Any) -> None:
        """Automatically extract and add model-native importance."""
        native = _extract_native_importance(model, self.feature_names)
        if native is not None:
            self.add_source(f"native_{type(model).__name__}", native)

    def build(self, top_k: int = 3) -> dict:
        """
        Compute all consensus metrics.

        Parameters
        ----------
        top_k : int — used for top-k overlap and explanation coverage

        Returns
        -------
        dict with all metrics (see module docstring for formulae)
        """
        sources = self._sources
        n_sources = len(sources)
        names = list(sources.keys())

        if n_sources < 2:
            logger.warning("ConsensusAnalyser: fewer than 2 sources — cannot compute consensus.")
            self._report = {"error": "insufficient sources", "n_sources": n_sources}
            return self._report

        # Build rank matrix: rows = features, cols = sources
        rank_df = pd.DataFrame({
            name: sources[name].rank(ascending=False)
            for name in names
        }, index=self.feature_names)

        # --- 1. Feature Agreement Score (FAS) ---
        rho_pairs = {}
        rho_values = []
        for i in range(n_sources):
            for j in range(i + 1, n_sources):
                rho, p_val = spearmanr(rank_df.iloc[:, i], rank_df.iloc[:, j])
                pair_key = f"{names[i]} vs {names[j]}"
                rho_pairs[pair_key] = {"spearman_rho": float(rho), "p_value": float(p_val)}
                rho_values.append(float(rho))

        fas = float(np.mean(rho_values))
        fas_grade = (
            "Excellent" if fas >= 0.9 else
            "Good" if fas >= 0.8 else
            "Acceptable" if fas >= 0.7 else
            "Weak — methods disagree significantly"
        )

        # --- 2. Top-k Overlap (TkO) ---
        top_k_sets = {
            name: set(sources[name].nlargest(top_k).index)
            for name in names
        }
        jaccard_values = []
        for i in range(n_sources):
            for j in range(i + 1, n_sources):
                a, b = top_k_sets[names[i]], top_k_sets[names[j]]
                jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
                jaccard_values.append(jaccard)
        mean_jaccard = float(np.mean(jaccard_values)) if jaccard_values else float("nan")

        # --- 3. Consensus Index (CI) ---
        consensus_index = rank_df.mean(axis=1).sort_values()
        consensus_ranking = consensus_index.to_dict()

        # --- 4. Explanation Coverage (EC) ---
        union_top_k = set()
        for s in top_k_sets.values():
            union_top_k |= s
        ec = len(union_top_k) / len(self.feature_names)

        # --- Disagreements (features with high rank variance) ---
        rank_std = rank_df.std(axis=1).sort_values(ascending=False)
        disagreements = rank_std.head(3).to_dict()

        self._report = {
            "model": self.model_name,
            "n_sources": n_sources,
            "sources": names,
            "top_k_used": top_k,
            "feature_agreement_score": {
                "value": fas,
                "grade": fas_grade,
                "pairwise_correlations": rho_pairs,
                "formula": "mean(Spearman_rho(rank_i, rank_j)) for all i!=j pairs",
            },
            "top_k_overlap": {
                "value": mean_jaccard,
                "top_k_sets": {k: list(v) for k, v in top_k_sets.items()},
                "formula": "mean(|top_k_i ∩ top_k_j| / |top_k_i ∪ top_k_j|)",
            },
            "consensus_index": {
                "ranking": consensus_ranking,
                "formula": "mean(rank_source(feature)) across all sources",
                "interpretation": "Lower value = higher consensus importance",
            },
            "explanation_coverage": {
                "value": ec,
                "union_top_k_features": list(union_top_k),
                "formula": "|union(top_k over all sources)| / n_features",
            },
            "disagreements": {
                "high_variance_features": disagreements,
                "note": "Features with high rank variance across sources",
            },
        }

        logger.info(
            "ConsensusAnalyser: FAS = %.3f (%s), Top-%d overlap = %.3f",
            fas, fas_grade, top_k, mean_jaccard,
        )
        return self._report

    def plot_rank_correlation_heatmap(self, out_path: Path) -> None:
        """Heatmap of pairwise Spearman ρ between all importance sources."""
        if not self._sources:
            return
        names = list(self._sources.keys())
        n = len(names)
        rank_df = pd.DataFrame(
            {name: self._sources[name].rank(ascending=False) for name in names},
            index=self.feature_names,
        )
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                rho, _ = spearmanr(rank_df.iloc[:, i], rank_df.iloc[:, j])
                corr_matrix[i, j] = rho

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_names = [nm.replace("native_", "").replace("Classifier", "") for nm in names]
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)
        for i in range(n):
            for j in range(n):
                col = "white" if abs(corr_matrix[i, j]) > 0.7 else "black"
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=col)
        ax.set_title(
            f"Explanation Consensus: Rank Correlation — {self.model_name}\n"
            f"(Feature Agreement Score = {self._report.get('feature_agreement_score', {}).get('value', float('nan')):.3f})",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved consensus rank correlation heatmap: %s", out_path)

    def plot_consensus_bar(self, out_path: Path) -> None:
        """Bar chart of Consensus Index per feature (lower = more important)."""
        if not self._report:
            return
        ci = self._report.get("consensus_index", {}).get("ranking", {})
        if not ci:
            return
        df = pd.Series(ci).sort_values()  # lower rank = more important
        n = len(df)

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.55 + 1)))
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n))  # type: ignore[attr-defined]
        ax.barh(range(n), df.values, color=colors, edgecolor="white")
        ax.set_yticks(range(n))
        ax.set_yticklabels(df.index)
        ax.set_xlabel("Consensus Index (mean rank across methods — lower = more important)", fontsize=10)
        ax.set_title(f"Feature Consensus Ranking — {self.model_name}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved consensus bar chart: %s", out_path)

    def save_report(self, out_path: Path) -> None:
        with out_path.open("w") as fh:
            json.dump(self._report, fh, indent=2)
        logger.info("Saved consensus report: %s", out_path)
