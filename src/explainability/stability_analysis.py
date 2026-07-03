"""
stability_analysis.py
=====================
Explanation stability analysis across random seeds.

Trains the same model architecture multiple times with different random seeds
and measures how consistently feature importance rankings remain.

Metrics:
  - Ranking Stability: mean pairwise Spearman rho across seed pairs
  - Feature Importance Variance: std of mean|SHAP| per feature across seeds
  - 95% CI: percentile bootstrap over seed runs
  - Stability Grade: Excellent (>=0.9) / Good (>=0.8) / Acceptable (>=0.7) / Concerning
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_DEFAULT_SEEDS = [42, 123, 456, 789, 1000]


class StabilityAnalyser:
    """
    Measures explanation stability by training a model under multiple random seeds
    and comparing SHAP-based feature rankings across runs.

    Parameters
    ----------
    model_class : sklearn-compatible classifier class
    model_kwargs : dict — passed to model_class(); random_state is overridden per seed
    feature_names : list[str]
    model_name : str
    seeds : list[int]
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict,
        feature_names: list[str],
        model_name: str,
        seeds: list[int] | None = None,
    ) -> None:
        self.model_class = model_class
        self.model_kwargs = dict(model_kwargs)
        self.feature_names = feature_names
        self.model_name = model_name
        self.seeds = seeds or _DEFAULT_SEEDS

        self._seed_shap_rankings: dict[int, pd.Series] = {}
        self._seed_shap_means: dict[int, np.ndarray] = {}
        self._stability_report: dict = {}

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series | None = None,
    ) -> dict:
        """
        Train model under each seed, compute SHAP values, record rankings.

        Returns the full stability report dict.
        """
        logger.info(
            "StabilityAnalyser: running %d seeds for '%s'",
            len(self.seeds), self.model_name,
        )

        # Align columns
        X_train = X_train[self.feature_names] if all(f in X_train.columns for f in self.feature_names) else X_train
        X_eval = X_eval[self.feature_names] if all(f in X_eval.columns for f in self.feature_names) else X_eval

        for seed in self.seeds:
            try:
                kwargs = dict(self.model_kwargs)
                # Inject random_state if the model supports it
                try:
                    dummy = self.model_class(**kwargs)
                    if "random_state" in dummy.get_params():
                        kwargs["random_state"] = seed
                except Exception:
                    pass

                model = self.model_class(**kwargs)

                # Preprocess inside each seed (fit-on-train-only for leakage-free estimates)
                imp = SimpleImputer(strategy="median")
                scaler = RobustScaler()
                X_tr_pp = pd.DataFrame(
                    scaler.fit_transform(imp.fit_transform(X_train)),
                    columns=self.feature_names,
                )
                X_ev_pp = pd.DataFrame(
                    scaler.transform(imp.transform(X_eval)),
                    columns=self.feature_names,
                )

                model.fit(X_tr_pp, y_train)

                # SHAP values via SHAPAnalyser
                from src.explainability.shap_analysis import SHAPAnalyser
                bg_size = min(100, len(X_tr_pp))
                rng = np.random.default_rng(seed)
                bg_idx = rng.choice(len(X_tr_pp), size=bg_size, replace=False)
                X_bg = X_tr_pp.iloc[bg_idx].reset_index(drop=True)

                analyser = SHAPAnalyser(model, X_bg, self.feature_names, self.model_name)
                analyser.fit()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shap_vals = analyser.compute_shap_values(X_ev_pp)

                mean_abs = np.abs(shap_vals).mean(axis=0)  # (p,)
                ranking = pd.Series(mean_abs, index=self.feature_names).sort_values(ascending=False)

                self._seed_shap_rankings[seed] = ranking
                self._seed_shap_means[seed] = mean_abs
                logger.debug("Seed %d: top feature = %s", seed, ranking.index[0])

            except Exception as e:
                logger.warning("StabilityAnalyser: seed %d failed — %s", seed, e)

        self._stability_report = self._compute_metrics()
        return self._stability_report

    def _compute_metrics(self) -> dict:
        seeds = sorted(self._seed_shap_rankings.keys())
        n_seeds = len(seeds)

        if n_seeds < 2:
            logger.warning("StabilityAnalyser: < 2 successful seeds; cannot compute stability.")
            return {"n_seeds": n_seeds, "error": "insufficient seeds"}

        # Mean importance matrix: shape (n_seeds, n_features)
        mean_matrix = np.stack([self._seed_shap_means[s] for s in seeds], axis=0)

        # Pairwise Spearman rank correlations
        rho_values = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rho, _ = spearmanr(mean_matrix[i], mean_matrix[j])
                rho_values.append(float(rho))

        mean_rho = float(np.mean(rho_values))
        std_rho = float(np.std(rho_values))

        # Per-feature statistics
        feature_stats = {}
        for fi, feat in enumerate(self.feature_names):
            vals = mean_matrix[:, fi]
            feature_stats[feat] = {
                "mean_importance": float(np.mean(vals)),
                "std_importance": float(np.std(vals)),
                "ci_lower_95": float(np.percentile(vals, 2.5)),
                "ci_upper_95": float(np.percentile(vals, 97.5)),
                "coefficient_of_variation": float(np.std(vals) / (np.mean(vals) + 1e-10)),
            }

        # Consensus ranking (median across seeds)
        median_importance = np.median(mean_matrix, axis=0)
        consensus_ranking = (
            pd.Series(median_importance, index=self.feature_names)
            .sort_values(ascending=False)
            .to_dict()
        )

        # Grade
        if mean_rho >= 0.9:
            grade = "Excellent"
        elif mean_rho >= 0.8:
            grade = "Good"
        elif mean_rho >= 0.7:
            grade = "Acceptable"
        else:
            grade = "Concerning — explanations may be unreliable"

        report = {
            "model": self.model_name,
            "n_seeds": n_seeds,
            "seeds": seeds,
            "ranking_stability": {
                "mean_spearman_rho": mean_rho,
                "std_spearman_rho": std_rho,
                "min_spearman_rho": float(min(rho_values)),
                "pairwise_rho_values": rho_values,
                "stability_grade": grade,
                "interpretation": (
                    f"Mean Spearman rho = {mean_rho:.3f} ({grade}). "
                    f"A value >= 0.9 indicates excellent explanation reproducibility. "
                    f"{'Explanations are reliable.' if mean_rho >= 0.8 else 'Interpret with caution.'}"
                ),
            },
            "feature_statistics": feature_stats,
            "consensus_ranking": consensus_ranking,
        }

        logger.info(
            "StabilityAnalyser: Spearman rho = %.3f +/- %.3f (%s)",
            mean_rho, std_rho, grade,
        )
        return report

    def plot_stability_violin(self, out_path: Path) -> None:
        """Violin plot of mean|SHAP| distribution per feature across seeds."""
        if not self._seed_shap_means:
            return

        seeds = sorted(self._seed_shap_means.keys())
        mean_matrix = np.stack([self._seed_shap_means[s] for s in seeds], axis=0)
        p = len(self.feature_names)

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        order = np.argsort(mean_matrix.mean(axis=0))[::-1]
        data = [mean_matrix[:, i] for i in order]
        labels = [self.feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=(max(6, p * 1.2), 5))
        parts = ax.violinplot(data, vert=True, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#3498db")
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, p + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("mean |SHAP value|", fontsize=11)
        rho = self._stability_report.get("ranking_stability", {}).get("mean_spearman_rho", float("nan"))
        ax.set_title(
            f"Explanation Stability across {len(seeds)} Seeds — {self.model_name}\n"
            f"(Spearman rho = {rho:.3f})",
            fontsize=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved stability violin plot: %s", out_path)

    def plot_ranking_heatmap(self, out_path: Path) -> None:
        """Heatmap: feature rank per seed. Stable = consistent colours across columns."""
        if not self._seed_shap_rankings:
            return

        seeds = sorted(self._seed_shap_rankings.keys())
        rank_df = pd.DataFrame({
            f"seed_{s}": self._seed_shap_rankings[s].rank(ascending=False).astype(int)
            for s in seeds
        })

        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        fig, ax = plt.subplots(figsize=(max(5, len(seeds) + 1), max(3, len(self.feature_names))))
        im = ax.imshow(rank_df.values, cmap="YlOrRd_r", aspect="auto",
                       vmin=1, vmax=len(self.feature_names))
        plt.colorbar(im, ax=ax, shrink=0.8, label="Feature Rank (1 = most important)")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"Seed {s}" for s in seeds], rotation=30, ha="right")
        ax.set_yticks(range(len(rank_df)))
        ax.set_yticklabels(rank_df.index)
        for i in range(len(rank_df)):
            for j in range(len(seeds)):
                ax.text(j, i, str(rank_df.iloc[i, j]), ha="center", va="center", fontsize=9)
        ax.set_title(f"Feature Ranking Heatmap across Seeds — {self.model_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved ranking heatmap: %s", out_path)

    def save_report(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(self._stability_report, fh, indent=2)
        logger.info("Saved stability report: %s", out_path)
