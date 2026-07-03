"""
visualizations.py
==================
Generates publication-quality visualizations for generalization, calibration transfer,
explanation drift, domain shift, and failure clustering.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Style parameters for publication
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})
sns.set_theme(style="whitegrid")

def plot_performance_drop(
    internal_auc: float,
    external_auc: float,
    train_cohort: str,
    test_cohort: str,
    output_path: Path,
) -> None:
    """Plots internal vs external performance showing performance degradation."""
    fig, ax = plt.subplots(figsize=(6, 5))
    categories = [f"Internal ({train_cohort})", f"External ({test_cohort})"]
    values = [internal_auc, external_auc]
    
    # Custom colors
    colors = ["#4A90E2", "#E24A8D"]
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")
        
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Generalization Performance: {train_cohort} → {test_cohort}")
    
    # Draw drop line/arrow
    drop = internal_auc - external_auc
    ax.annotate(
        f"Generalization Gap: -{drop:.3f}",
        xy=(1, external_auc),
        xytext=(0.5, (internal_auc + external_auc)/2),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        ha="center",
        fontweight="bold",
        color="#D0021B"
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug("Saved performance drop plot to %s", output_path)

def plot_calibration_transfer_curves(
    curve_src: list[dict],
    curve_tgt: list[dict],
    src_cohort: str,
    tgt_cohort: str,
    output_path: Path,
) -> None:
    """Plots reliability curves for source and target calibration side-by-side."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot ideal line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Extract source curve data
    conf_src = [c["confidence"] for c in curve_src if not np.isnan(c["confidence"])]
    acc_src = [c["accuracy"] for c in curve_src if not np.isnan(c["accuracy"])]
    if conf_src:
        ax.plot(conf_src, acc_src, marker="o", color="#4A90E2", label=f"Source ({src_cohort})", linewidth=1.5)
        
    # Extract target curve data
    conf_tgt = [c["confidence"] for c in curve_tgt if not np.isnan(c["confidence"])]
    acc_tgt = [c["accuracy"] for c in curve_tgt if not np.isnan(c["accuracy"])]
    if conf_tgt:
        ax.plot(conf_tgt, acc_tgt, marker="s", color="#E24A8D", label=f"Target ({tgt_cohort})", linewidth=1.5)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Outcome Fraction")
    ax.set_title("Calibration Transfer Assessment")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug("Saved calibration transfer plot to %s", output_path)

def plot_explanation_drift_heatmap(
    shap_src: dict[str, float],
    shap_tgt: dict[str, float],
    features: list[str],
    src_cohort: str,
    tgt_cohort: str,
    output_path: Path,
) -> None:
    """Plots a heatmap comparing SHAP importance ranking across cohorts."""
    # Normalize importances
    sum_src = sum(shap_src.values()) if sum(shap_src.values()) > 0 else 1.0
    sum_tgt = sum(shap_tgt.values()) if sum(shap_tgt.values()) > 0 else 1.0
    
    norm_src = {f: shap_src.get(f, 0.0)/sum_src for f in features}
    norm_tgt = {f: shap_tgt.get(f, 0.0)/sum_tgt for f in features}
    
    df_plot = pd.DataFrame({
        src_cohort: [norm_src[f] for f in features],
        tgt_cohort: [norm_tgt[f] for f in features]
    }, index=features)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_plot, annot=True, cmap="Blues", fmt=".3f", cbar_kws={"label": "Normalized SHAP Importance"}, ax=ax)
    ax.set_title("SHAP Explanation Drift")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug("Saved explanation drift heatmap to %s", output_path)

def plot_domain_shift_kde(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    feature: str,
    src_label: str,
    tgt_label: str,
    output_path: Path,
) -> None:
    """Plots Kernel Density Estimate (KDE) comparing distribution of a feature between cohorts."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    vals_src = df_src[feature].dropna()
    vals_tgt = df_tgt[feature].dropna()
    
    if len(vals_src) > 0:
        sns.kdeplot(vals_src, fill=True, color="#4A90E2", label=src_label, ax=ax, alpha=0.4, linewidth=1.5)
    if len(vals_tgt) > 0:
        sns.kdeplot(vals_tgt, fill=True, color="#E24A8D", label=tgt_label, ax=ax, alpha=0.4, linewidth=1.5)
        
    ax.set_xlabel(feature.replace("h_", "").upper())
    ax.set_ylabel("Density")
    ax.set_title(f"Covariate Shift: {feature.replace('h_', '').upper()}")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug("Saved domain shift KDE plot for %s to %s", feature, output_path)

def plot_trustworthiness_radar(
    profile: dict,
    output_path: Path,
) -> None:
    """Plots a 7-axis radar chart showing multi-dimensional trustworthiness profile."""
    # Axes: Prediction, Calibration, Explanation, Robustness, Generalization, Consistency, Reproducibility
    categories = [
        "Prediction\nReliability", "Calibration\nReliability", "Explanation\nReliability",
        "Robustness", "Generalization", "Clinical\nConsistency", "Reproducibility"
    ]
    
    # Scale values to 0-1 range
    # Prediction: ROC-AUC
    pred_val = profile["prediction_reliability"]["roc_auc"]
    # Calibration: 1 - ECE (max ECE clipped to 0.3)
    cal_val = max(0.0, 1.0 - (profile["calibration_reliability"]["ece"] / 0.3))
    # Explanation: SHAP correlation
    exp_val = max(0.0, profile["explanation_reliability"]["shap_rank_correlation"])
    # Robustness: Robustness score (0-1)
    rob_val = profile["robustness"]["score"]
    # Generalization: 1 - ROC-AUC drop (max drop clipped to 0.3)
    gen_val = max(0.0, 1.0 - (profile["generalization"]["roc_auc_drop"] / 0.3))
    # Consistency: consistency rate (0-1)
    con_val = profile["clinical_consistency"]["consistency_rate"]
    # Reproducibility: 1 - STD AUC (max std clipped to 0.05)
    rep_val = max(0.0, 1.0 - (profile["reproducibility"]["std_auc"] / 0.05))
    
    values = [pred_val, cal_val, exp_val, rob_val, gen_val, con_val, rep_val]
    # Complete the circular loop
    values += values[:1]
    
    # Calculate angles
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axis per category and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=9)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.0)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color="#5D6D7E")
    ax.fill(angles, values, '#5D6D7E', alpha=0.35)
    
    ax.set_title("Clinical AI Trustworthiness Assessment", y=1.08, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug("Saved trustworthiness radar chart to %s", output_path)

def plot_failure_clusters(
    X_failed: pd.DataFrame,
    features: list[str],
    cluster_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plots failures projected onto 2D PCA space, colored by KMeans cluster ID."""
    if len(X_failed) < 3:
        # Too few failed cases to cluster visually
        return
        
    try:
        # Standardize and project to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_failed[features])
        
        df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
        df_pca["Cluster"] = [f"Cluster {c}" for c in cluster_labels]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df_pca, x="PCA1", y="PCA2", hue="Cluster", style="Cluster",
            palette="Set2", s=80, edgecolor="black", alpha=0.8, ax=ax
        )
        
        ax.set_xlabel(f"PCA Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PCA Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("Failure Mode Clustering (Target Cohort)")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.debug("Saved failure clusters plot to %s", output_path)
    except Exception as e:
        logger.warning("  Failed to generate failure cluster visualization: %s", e)
