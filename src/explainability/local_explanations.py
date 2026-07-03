"""
local_explanations.py
=====================
Patient-level (local) explanations with natural language summaries.

For each selected patient, generates:
  1. Predicted risk probability
  2. SHAP waterfall plot
  3. SHAP decision plot
  4. Ranked feature contributions (positive risk drivers + protective factors)
  5. Natural language summary (template-based, hedged language — never causal)

Natural language format:
    "This patient's [elevated/reduced] predicted risk (p = {prob:.2f}) is
     primarily driven by {top_positive_drivers}, while {protective_factors}
     partially offsets the overall risk."

Patients selected for explanation:
  - True Positive (highest confidence correct positive)
  - True Negative (highest confidence correct negative)
  - False Positive (highest confidence error)
  - False Negative (highest confidence missed)
  - Uncertain (predicted probability closest to 0.5)
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
import shap

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Hedge language — never causal
_HEDGE_VERBS = ["is consistent with", "may indicate", "in agreement with", "is associated with"]


def _feature_display_name(feat: str) -> str:
    """Map harmonized column names to readable clinical labels."""
    _MAP = {
        "h_age": "advanced age",
        "h_bmi": "body mass index (BMI)",
        "h_is_smoker": "smoking history",
        "h_sex_male": "male sex",
        "h_sys_bp": "elevated systolic blood pressure",
        "h_dia_bp": "elevated diastolic blood pressure",
        "h_total_cholesterol": "total cholesterol level",
        "h_ldl": "LDL cholesterol",
        "h_triglycerides": "triglyceride level",
        "h_glucose": "blood glucose level",
    }
    return _MAP.get(feat, feat.replace("h_", "").replace("_", " "))


def _nl_summary(
    feature_names: list[str],
    shap_vals: np.ndarray,
    risk_prob: float,
    threshold: float = 0.5,
) -> str:
    """
    Generate a hedged, natural-language explanation for a single patient.

    Parameters
    ----------
    feature_names : list[str]
    shap_vals : np.ndarray, shape (n_features,) — SHAP values for class 1
    risk_prob : float — predicted probability for class 1
    threshold : float — decision threshold

    Returns
    -------
    str — natural language summary
    """
    contributions = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    positive_drivers = [_feature_display_name(f) for f, s in contributions if s > 0.001]
    protective_factors = [_feature_display_name(f) for f, s in contributions if s < -0.001]

    risk_label = "elevated" if risk_prob >= threshold else "reduced"

    # Build sentence components
    if positive_drivers:
        top_pos = positive_drivers[0]
        other_pos = positive_drivers[1:3]
        if other_pos:
            pos_str = f"{top_pos}, {', '.join(other_pos)}"
        else:
            pos_str = top_pos
    else:
        pos_str = "no single dominant factor"

    if protective_factors:
        prot_str = protective_factors[0]
        if len(protective_factors) > 1:
            prot_str += f" and {protective_factors[1]}"
        prot_clause = f", while {prot_str} may partially offset the overall risk"
    else:
        prot_clause = ""

    summary = (
        f"This patient's {risk_label} predicted risk (probability = {risk_prob:.3f}) "
        f"is primarily associated with {pos_str}{prot_clause}. "
        f"These findings are consistent with established clinical risk patterns and "
        f"may indicate elevated clinical attention is warranted. "
        f"Note: this explanation describes the model's prediction and does not constitute "
        f"clinical advice or a causal claim."
    )
    return summary


class LocalExplainer:
    """
    Generates local (patient-level) explanations for a fitted classifier.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    shap_analyser : SHAPAnalyser (already fitted and with computed shap values)
    feature_names : list[str]
    model_name : str
    """

    def __init__(
        self,
        model: Any,
        shap_analyser: Any,   # SHAPAnalyser — avoid circular import
        feature_names: list[str],
        model_name: str,
    ) -> None:
        self.model = model
        self.shap_analyser = shap_analyser
        self.feature_names = feature_names
        self.model_name = model_name

    def _select_representative_patients(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Optional[int]]:
        """
        Select 5 representative patient indices:
        - true_positive: highest-confidence TP
        - true_negative: highest-confidence TN
        - false_positive: highest-confidence FP
        - false_negative: highest-confidence FN (missed case)
        - uncertain: closest to decision boundary (prob ≈ 0.5)
        """
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        def _argmax_if_any(mask: np.ndarray, scores: np.ndarray) -> Optional[int]:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return None
            return int(indices[np.argmax(scores[indices])])

        tp_mask = (y == 1) & (y_pred == 1)
        tn_mask = (y == 0) & (y_pred == 0)
        fp_mask = (y == 0) & (y_pred == 1)
        fn_mask = (y == 1) & (y_pred == 0)

        # Confidence = distance from 0.5
        confidence = np.abs(y_prob - 0.5)
        boundary_dist = 1 - confidence  # high = close to boundary

        return {
            "true_positive": _argmax_if_any(tp_mask, confidence),
            "true_negative": _argmax_if_any(tn_mask, confidence),
            "false_positive": _argmax_if_any(fp_mask, confidence),
            "false_negative": _argmax_if_any(fn_mask, confidence),
            "uncertain": _argmax_if_any(np.ones(len(y), dtype=bool), boundary_dist),
        }

    def explain_patient(
        self,
        patient_idx: int,
        X: pd.DataFrame,
        y_prob: np.ndarray,
        shap_values: np.ndarray,
        out_dir: Path,
        patient_label: str = "patient",
    ) -> dict:
        """
        Generate full explanation for a single patient.

        Returns structured dict with prediction, feature contributions, NL summary.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        row = X.iloc[patient_idx]
        prob = float(y_prob[patient_idx])
        shap_row = shap_values[patient_idx]

        # Feature contributions table
        contributions = pd.DataFrame({
            "feature": self.feature_names,
            "feature_value": row[self.feature_names].values,
            "shap_value": shap_row,
            "direction": ["risk_increasing" if s > 0 else "protective" if s < 0 else "neutral"
                         for s in shap_row],
        }).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)

        nl = _nl_summary(self.feature_names, shap_row, prob)

        # Waterfall plot
        safe_label = patient_label.replace(" ", "_")
        explanation = self.shap_analyser.get_explanation()
        self._plot_waterfall(explanation, patient_idx, prob, out_dir / f"waterfall_{safe_label}.png", patient_label)

        # Contributions bar
        self._plot_contributions_bar(contributions, prob, out_dir / f"contributions_{safe_label}.png", patient_label)

        result = {
            "patient_idx": patient_idx,
            "patient_label": patient_label,
            "predicted_probability": prob,
            "predicted_class": int(prob >= 0.5),
            "risk_level": "High" if prob >= 0.7 else "Moderate" if prob >= 0.4 else "Low",
            "top_risk_drivers": contributions[contributions["shap_value"] > 0]["feature"].tolist()[:3],
            "top_protective_factors": contributions[contributions["shap_value"] < 0]["feature"].tolist()[:3],
            "feature_contributions": contributions.to_dict(orient="records"),
            "natural_language_summary": nl,
        }
        return result

    def _plot_waterfall(
        self,
        explanation: shap.Explanation,
        patient_idx: int,
        prob: float,
        out_path: Path,
        patient_label: str,
    ) -> None:
        """SHAP waterfall plot for a single patient."""
        try:
            plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
            single_exp = explanation[patient_idx]
            shap.plots.waterfall(single_exp, show=False, max_display=len(self.feature_names))
            plt.title(
                f"SHAP Waterfall — {self.model_name}\n{patient_label} (p={prob:.3f})",
                fontsize=11, pad=8
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close("all")
            logger.info("Saved waterfall plot: %s", out_path)
        except Exception as e:
            logger.warning("Waterfall plot failed for %s: %s", patient_label, e)
            plt.close("all")

    def _plot_contributions_bar(
        self,
        contributions: pd.DataFrame,
        prob: float,
        out_path: Path,
        patient_label: str,
    ) -> None:
        """Horizontal bar chart of SHAP values coloured by direction."""
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
        df = contributions.copy()
        n = len(df)
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df["shap_value"]]

        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.5 + 1)))
        y_pos = np.arange(n)
        ax.barh(y_pos, df["shap_value"].values[::-1], color=colors[::-1],
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['feature']} = {r['feature_value']:.2f}" if isinstance(r["feature_value"], float)
                            else f"{r['feature']} = {r['feature_value']}"
                            for r in df.to_dict("records")][::-1],
                           fontsize=9)
        ax.set_xlabel("SHAP value (impact on predicted probability)", fontsize=10)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_title(
            f"Feature Contributions — {self.model_name}\n{patient_label} (p={prob:.3f})",
            fontsize=11
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="Risk-increasing"),
            Patch(facecolor="#2ecc71", label="Protective"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")

    def batch_explain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        shap_values: np.ndarray,
        out_dir: Path,
    ) -> list[dict]:
        """
        Explain 5 representative patients (TP, TN, FP, FN, uncertain).

        Saves all plots and returns a list of explanation dicts.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        patient_indices = self._select_representative_patients(X, y, y_prob, y_pred)
        explanations = []
        for label, idx in patient_indices.items():
            if idx is None:
                logger.info("LocalExplainer: no %s patient found — skipping.", label)
                continue
            logger.info("LocalExplainer: explaining %s (idx=%d)", label, idx)
            exp = self.explain_patient(idx, X, y_prob, shap_values, out_dir, label)
            explanations.append(exp)

        # Save combined report
        report_path = out_dir / "local_explanations.json"
        with report_path.open("w") as fh:
            json.dump(explanations, fh, indent=2)
        logger.info("Saved local explanations report: %s", report_path)
        return explanations
