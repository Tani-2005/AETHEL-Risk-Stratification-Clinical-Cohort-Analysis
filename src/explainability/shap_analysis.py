"""
shap_analysis.py
================
Global SHAP analysis for publication-quality explainability.

Supports:
  - TreeExplainer  : RandomForest, XGBoost, LightGBM, DecisionTree
  - LinearExplainer: LogisticRegression
  - KernelExplainer: fallback for any sklearn-compatible model

Figures produced (300 DPI, publication-ready):
  - SHAP summary plot (dot/beeswarm)
  - SHAP bar plot (mean |SHAP|)
  - SHAP beeswarm plot
  - SHAP dependence plots (one per feature)
"""
from __future__ import annotations

import warnings
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

# Model class names that support TreeExplainer
_TREE_MODEL_TYPES = frozenset({
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
})

_LINEAR_MODEL_TYPES = frozenset({
    "LogisticRegression",
    "LinearSVC",
    "Ridge",
    "RidgeClassifier",
})


def _model_type_name(model: Any) -> str:
    return type(model).__name__


def _extract_shap_values_class1(raw: Any) -> np.ndarray:
    """
    Normalise heterogeneous SHAP return types to a single (n_samples, n_features)
    array representing class-1 SHAP values for binary classification.
    """
    if isinstance(raw, list):
        # Old API: [class_0_array, class_1_array]
        arr = np.asarray(raw[1])
    elif isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            arr = raw[:, :, 1]   # (n, p, 2) → (n, p)
        else:
            arr = raw             # already (n, p)
    else:
        # shap.Explanation object
        vals = raw.values
        if vals.ndim == 3:
            arr = vals[:, :, 1]
        else:
            arr = vals
    return arr


def _extract_expected_value_class1(expected: Any) -> float:
    """Extract scalar expected value for class 1."""
    if isinstance(expected, (list, np.ndarray)):
        ev = float(np.asarray(expected).ravel()[-1])
    else:
        ev = float(expected)
    return ev


class SHAPAnalyser:
    """
    Computes and visualises global SHAP values for a fitted binary classifier.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_background : pd.DataFrame
        Background dataset for KernelExplainer (subsample of training data).
        Also used as masker for LinearExplainer.
    feature_names : list[str]
    model_name : str
        Human-readable name for plot titles.
    """

    def __init__(
        self,
        model: Any,
        X_background: pd.DataFrame,
        feature_names: list[str],
        model_name: str,
    ) -> None:
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.model_name = model_name

        self._explainer: Optional[Any] = None
        self._shap_values: Optional[np.ndarray] = None   # (n, p) class-1
        self._expected_value: Optional[float] = None
        self._shap_explanation: Optional[shap.Explanation] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "SHAPAnalyser":
        """Build the appropriate SHAP explainer based on model type."""
        mtype = _model_type_name(self.model)
        logger.info("SHAPAnalyser: fitting %s explainer for %s", mtype, self.model_name)

        if mtype in _TREE_MODEL_TYPES:
            try:
                self._explainer = shap.TreeExplainer(
                    self.model,
                    data=self.X_background,
                    model_output="raw",
                    feature_perturbation="interventional",
                )
                logger.debug("SHAPAnalyser: using TreeExplainer")
            except Exception as e:
                logger.warning("TreeExplainer failed (%s); falling back to KernelExplainer", e)
                self._explainer = self._make_kernel_explainer()

        elif mtype in _LINEAR_MODEL_TYPES:
            try:
                masker = shap.maskers.Independent(self.X_background)
                self._explainer = shap.LinearExplainer(self.model, masker=masker)
                logger.debug("SHAPAnalyser: using LinearExplainer")
            except Exception as e:
                logger.warning("LinearExplainer failed (%s); falling back to KernelExplainer", e)
                self._explainer = self._make_kernel_explainer()
        else:
            self._explainer = self._make_kernel_explainer()

        return self

    def _make_kernel_explainer(self) -> shap.KernelExplainer:
        bg = shap.sample(self.X_background, min(100, len(self.X_background)))
        logger.debug("SHAPAnalyser: using KernelExplainer with %d background samples", len(bg))
        return shap.KernelExplainer(
            lambda x: self.model.predict_proba(x)[:, 1],
            bg,
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for X.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Class-1 SHAP values.
        """
        if self._explainer is None:
            self.fit()

        logger.info("SHAPAnalyser: computing SHAP values for %d samples", len(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mtype = _model_type_name(self.model)
            if mtype in _TREE_MODEL_TYPES:
                raw = self._explainer.shap_values(X, check_additivity=False)
            else:
                raw = self._explainer.shap_values(X)

        self._shap_values = _extract_shap_values_class1(raw)

        # Expected value
        ev = getattr(self._explainer, "expected_value", 0.0)
        self._expected_value = _extract_expected_value_class1(ev)

        # Build Explanation object for new-API plots
        self._shap_explanation = shap.Explanation(
            values=self._shap_values,
            base_values=np.full(len(X), self._expected_value),
            data=X.values,
            feature_names=self.feature_names,
        )

        logger.info(
            "SHAPAnalyser: done. mean|SHAP|= %s",
            dict(zip(self.feature_names, np.abs(self._shap_values).mean(axis=0).round(4))),
        )
        return self._shap_values

    def mean_abs_shap(self) -> pd.Series:
        """Return mean absolute SHAP value per feature, sorted descending."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        vals = np.abs(self._shap_values).mean(axis=0)
        return pd.Series(vals, index=self.feature_names).sort_values(ascending=False)

    def get_explanation(self) -> shap.Explanation:
        """Return the shap.Explanation object (needed for waterfall/decision plots)."""
        if self._shap_explanation is None:
            raise RuntimeError("Call compute_shap_values() first.")
        return self._shap_explanation

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _setup_figure_style(self) -> None:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        })

    def plot_summary(self, X: pd.DataFrame, out_path: Path) -> None:
        """SHAP dot/beeswarm summary plot (feature × SHAP value coloured by feature value)."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        self._setup_figure_style()
        plt.figure(figsize=(9, max(4, len(self.feature_names) * 0.55 + 1)))
        shap.summary_plot(
            self._shap_values, X,
            feature_names=self.feature_names,
            plot_type="dot",
            show=False,
            max_display=len(self.feature_names),
            plot_size=None,
        )
        plt.title(f"SHAP Summary — {self.model_name}", fontsize=13, pad=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP summary plot: %s", out_path)

    def plot_bar(self, out_path: Path) -> None:
        """SHAP bar plot (mean |SHAP| per feature)."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        self._setup_figure_style()
        ranking = self.mean_abs_shap()
        fig, ax = plt.subplots(figsize=(8, max(3, len(ranking) * 0.5 + 1)))
        colors = plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(ranking)))  # type: ignore[attr-defined]
        bars = ax.barh(ranking.index[::-1], ranking.values[::-1], color=colors)
        ax.set_xlabel("mean(|SHAP value|)", fontsize=11)
        ax.set_title(f"Feature Importance (SHAP) — {self.model_name}", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Annotate bar values
        for bar, val in zip(bars, ranking.values[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP bar plot: %s", out_path)

    def plot_beeswarm(self, X: pd.DataFrame, out_path: Path) -> None:
        """SHAP beeswarm plot using the new shap.plots API."""
        if self._shap_explanation is None:
            raise RuntimeError("Call compute_shap_values() first.")
        self._setup_figure_style()
        plt.figure(figsize=(9, max(4, len(self.feature_names) * 0.6 + 1)))
        shap.plots.beeswarm(self._shap_explanation, show=False,
                            max_display=len(self.feature_names))
        plt.title(f"SHAP Beeswarm — {self.model_name}", fontsize=13, pad=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP beeswarm plot: %s", out_path)

    def plot_dependence(self, X: pd.DataFrame, feature: str, out_path: Path) -> None:
        """SHAP dependence plot for a single feature."""
        if self._shap_values is None or feature not in self.feature_names:
            return
        self._setup_figure_style()
        feat_idx = self.feature_names.index(feature)
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(
            feat_idx, self._shap_values, X,
            feature_names=self.feature_names,
            ax=ax, show=False,
        )
        ax.set_title(f"SHAP Dependence: {feature} — {self.model_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP dependence plot for '%s': %s", feature, out_path)

    def plot_all_dependences(self, X: pd.DataFrame, out_dir: Path) -> None:
        """Save one dependence plot per feature."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for feat in self.feature_names:
            safe = feat.replace(" ", "_").lower()
            self.plot_dependence(X, feat, out_dir / f"shap_dependence_{safe}.png")
