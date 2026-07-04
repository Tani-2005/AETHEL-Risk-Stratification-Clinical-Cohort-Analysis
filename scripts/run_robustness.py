"""
run_robustness.py
=================
Publication-grade Robustness and Sensitivity Framework Orchestrator for AETHEL.

Runs repeated experiments, bootstrap confidence intervals, feature ablation,
noise/missingness sweeps, distribution shift, uncertainty estimation, and
failure analysis across all models and experiment modes.

Usage
-----
    # Fast development mode:
    python -m scripts.run_robustness --mode dev

    # Full paper mode:
    python -m scripts.run_robustness --mode paper
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config_loader import load_config
from src.utils.logging_setup import configure_logging, get_logger
from src.utils.paths import ensure_output_dirs, PROJECT_ROOT, RobustnessDirs
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner
from src.evaluation.evaluator import calculate_metrics

# Robustness modules
from src.robustness.stability_metrics import (
    compute_prediction_stability,
    compute_probability_stability,
    compute_ranking_stability,
    compute_explanation_stability,
)
from src.robustness.repeated_runs import run_repeated_experiments
from src.robustness.bootstrap import run_bootstrap_analysis
from src.robustness.feature_ablation import (
    run_hierarchical_ablation,
    run_individual_ablation,
)
from src.robustness.noise_analysis import run_noise_robustness
from src.robustness.missing_data_analysis import run_missing_data_robustness
from src.robustness.distribution_shift import (
    analyze_covariate_shift,
    evaluate_cross_cohort_drift,
)
from src.robustness.uncertainty import estimate_uncertainty
from src.robustness.robustness_reports import (
    calculate_robustness_score,
    generate_publication_table,
)

logger = get_logger(__name__)

# Style parameters for publication-grade plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "legend.fontsize": 10,
})

ALL_CLASSIFIERS = {
    "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    "Decision Tree":       (DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
    "Random Forest":       (RandomForestClassifier, {"n_estimators": 100, "n_jobs": -1, "random_state": 42}),
    "XGBoost":             (XGBClassifier, {"eval_metric": "logloss", "random_state": 42}),
    "LightGBM":            (LGBMClassifier, {"verbose": -1, "random_state": 42}),
}

EXPERIMENT_MODES = {
    "dev": ["Logistic Regression", "Random Forest", "XGBoost"],
    "paper": list(ALL_CLASSIFIERS.keys()),
}

def preprocess_splits(X_train, X_val, features):
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_tr = pd.DataFrame(
        scaler.fit_transform(imp.fit_transform(X_train[features])),
        columns=features,
    )
    X_val_pp = pd.DataFrame(
        scaler.transform(imp.transform(X_val[features])),
        columns=features,
    )
    return X_tr, X_val_pp, imp, scaler

def plot_bootstrap_distribution(distributions: dict, out_path: Path, model_name: str) -> None:
    """Plots histogram/density of bootstrap ROC-AUC and C-index."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC-AUC
    if "roc_auc" in distributions:
        sns.histplot(distributions["roc_auc"], kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title(f"Bootstrap ROC-AUC Distribution\n(Mean = {np.mean(distributions['roc_auc']):.3f})")
        axes[0].set_xlabel("ROC-AUC")
        axes[0].set_ylabel("Count")
        
    # Brier Score
    if "brier" in distributions:
        sns.histplot(distributions["brier"], kde=True, ax=axes[1], color="salmon")
        axes[1].set_title(f"Bootstrap Brier Score Distribution\n(Mean = {np.mean(distributions['brier']):.4f})")
        axes[1].set_xlabel("Brier Score")
        axes[1].set_ylabel("Count")
        
    fig.suptitle(f"Bootstrap Robustness — {model_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_ablation_performance_drop(
    hier_results: dict, ind_results: dict, out_path: Path, model_name: str
) -> None:
    """Plots performance drop (ROC-AUC drop) under group-level and feature ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Hierarchical ablation
    grps = list(hier_results.keys())
    grp_drops = [res["auc_drop"] for res in hier_results.values()]
    if grps:
        sns.barplot(x=grp_drops, y=grps, ax=axes[0], palette="Blues_r")
        axes[0].set_title("Hierarchical (Group) Ablation")
        axes[0].set_xlabel("ROC-AUC Drop")
        axes[0].axvline(0, color="grey", linestyle="--")
        
    # 2. Individual ablation
    feats = list(ind_results.keys())
    feat_drops = [res["auc_drop"] for res in ind_results.values()]
    if feats:
        # Sort by drop
        sorted_idx = np.argsort(feat_drops)[::-1]
        sorted_feats = [feats[i] for i in sorted_idx]
        sorted_drops = [feat_drops[i] for i in sorted_idx]
        
        sns.barplot(x=sorted_drops, y=sorted_feats, ax=axes[1], palette="Oranges_r")
        axes[1].set_title("Individual Feature Ablation")
        axes[1].set_xlabel("ROC-AUC Drop")
        axes[1].axvline(0, color="grey", linestyle="--")
        
    fig.suptitle(f"Feature Ablation Impact — {model_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_degradation_curves(
    noise_results: dict, miss_results: dict, out_path: Path, model_name: str
) -> None:
    """Plots degradation curves for noise and missingness."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Noise Curve
    noise_lvls = list(noise_results.keys())
    noise_aucs = [res["metrics"].get("roc_auc", 0.5) for res in noise_results.values()]
    axes[0].plot(noise_lvls, noise_aucs, marker="o", color="crimson", linewidth=2)
    axes[0].set_title("Noise Robustness")
    axes[0].set_xlabel("Noise Level / Perturbation Prob")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim(0.4, 1.05)
    
    # 2. Missingness Curve
    miss_lvls = list(miss_results.keys())
    miss_aucs = [res["metrics"].get("roc_auc", 0.5) for res in miss_results.values()]
    axes[1].plot(miss_lvls, miss_aucs, marker="s", color="darkgreen", linewidth=2)
    axes[1].set_title("Missing Data Robustness")
    axes[1].set_xlabel("Missing Data Rate (MCAR)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.4, 1.05)
    
    fig.suptitle(f"Input Perturbation Degradation Curves — {model_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_uncertainty_distributions(
    uncertainty_df: pd.DataFrame, out_path: Path, model_name: str
) -> None:
    """Plots probability variance and entropy distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Run Variance
    sns.histplot(uncertainty_df["prob_var"], kde=True, ax=axes[0], color="purple")
    axes[0].set_title("Epistemic Uncertainty\n(Probability Variance across runs)")
    axes[0].set_xlabel("Variance")
    axes[0].set_ylabel("Count")
    
    # Entropy
    sns.histplot(uncertainty_df["entropy"], kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Prediction Entropy\n(Information Uncertainty)")
    axes[1].set_xlabel("Entropy")
    axes[1].set_ylabel("Count")
    
    fig.suptitle(f"Prediction Uncertainty Profile — {model_name}", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def run_experiment_robustness(
    exp_name: str,
    exp_cfg: ExperimentConfig,
    runner: ExperimentRunner,
    run_mode: str,
) -> dict:
    """Runs full robustness framework for a single experiment cohort."""
    logger.info("=" * 70)
    logger.info("Robustness Audit: starting %s under %s mode", exp_name, run_mode)
    logger.info("=" * 70)
    
    # Run data preparation
    res_data = runner.run(exp_cfg)
    X_train = res_data.train_df
    y_train = X_train[res_data.outcome_col]
    X_val = res_data.val_df
    y_val = X_val[res_data.outcome_col]
    features = res_data.feature_set
    
    # Extract continuous & binary feature lists from defaults
    from src.utils.constants import Features
    continuous_features = Features.CONTINUOUS_FEATURES
    binary_features = Features.BINARY_FEATURES
    
    # Determine repeated seeds and bootstrap iterations
    if run_mode == "dev":
        seeds = [42, 100, 200, 300, 400]
        n_boot = 100
        classifiers_list = EXPERIMENT_MODES["dev"]
    else:
        seeds = [42 + i for i in range(100)]
        n_boot = 1000
        classifiers_list = EXPERIMENT_MODES["paper"]
        
    exp_dir = RobustnessDirs.experiment_dir(exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "reports").mkdir(parents=True, exist_ok=True)
    
    all_models_results = {}
    
    for model_name in classifiers_list:
        logger.info("-" * 50)
        logger.info("Robustness Model: %s", model_name)
        logger.info("-" * 50)
        
        model_class, model_kwargs = ALL_CLASSIFIERS[model_name]
        model_dir = RobustnessDirs.subdir(exp_name, model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess splits (impute and scale)
        X_tr, X_val_pp, imp, scaler = preprocess_splits(X_train, X_val, features)
        
        # Fit baseline model
        base_model = model_class(**model_kwargs)
        base_model.fit(X_tr, y_train)
        probs_base = base_model.predict_proba(X_val_pp)[:, 1]
        metrics_base = calculate_metrics(y_val, probs_base)
        
        # 1. Repeated Runs
        rep_results = run_repeated_experiments(
            model_class, model_kwargs, X_tr, y_train, X_val_pp, y_val, features, seeds
        )
        
        # 2. Stability Metrics
        pred_stab = compute_prediction_stability(rep_results["predictions"])
        prob_stab = compute_probability_stability(rep_results["probabilities"])
        rank_stab = compute_ranking_stability(rep_results["importances"])
        exp_stab = compute_explanation_stability(rep_results["shap_values"])
        
        # 3. Bootstrap
        boot_res = run_bootstrap_analysis(
            y_val, probs_base,
            time_col=X_val["months_observed"] if "months_observed" in X_val.columns else None,
            event_col=X_val["event_occurred"] if "event_occurred" in X_val.columns else None,
            n_iterations=n_boot
        )
        plot_bootstrap_distribution(boot_res["distributions"], model_dir / "bootstrap_distributions.png", model_name)
        
        # 4. Feature Ablation
        hier_ablation = run_hierarchical_ablation(
            model_class, model_kwargs, X_tr, y_train, X_val_pp, y_val, features, metrics_base, probs_base
        )
        ind_ablation = run_individual_ablation(
            model_class, model_kwargs, X_tr, y_train, X_val_pp, y_val, features, metrics_base, probs_base
        )
        plot_ablation_performance_drop(hier_ablation, ind_ablation, model_dir / "ablation_impact.png", model_name)
        
        # 5. Noise Robustness
        noise_res = run_noise_robustness(
            base_model, X_val_pp, y_val, features, continuous_features, binary_features
        )
        
        # 6. Missingness Robustness
        # Note: missingness requires original raw X_val to simulate MCAR imputation
        miss_res = run_missing_data_robustness(
            base_model, X_val, y_val, features, imp, scaler
        )
        plot_degradation_curves(noise_res, miss_res["sweep_results"], model_dir / "degradation_curves.png", model_name)
        
        # 7. Distribution Shift (Within-Cohort vs Cross-Cohort if Mode is 4, 5, or 6)
        covariate_shift = analyze_covariate_shift(X_train, X_val, features)
        
        cross_cohort_retention = 1.0
        cross_cohort_res = None
        
        # If we have a domain shift config, calculate performance retention
        if exp_cfg.mode in (4, 5, 6):
            cross_cohort_res = evaluate_cross_cohort_drift(
                base_model, X_train, y_train, X_val, y_val, features
            )
            cross_cohort_retention = cross_cohort_res["metrics_target"].get("roc_auc", 0.5) / metrics_base.get("roc_auc", 0.5)
            
        # 8. Uncertainty & Failure Analysis
        unc_res = estimate_uncertainty(rep_results["probabilities"], y_val)
        plot_uncertainty_distributions(unc_res["uncertainty_summary"], model_dir / "uncertainty_profiles.png", model_name)
        
        # Save detailed uncertainty table
        unc_res["uncertainty_summary"].to_csv(model_dir / "predictions_uncertainty_table.csv", index=False)
        
        # 9. Compute Overall Robustness Score
        prob_stability_metric = 1.0 - np.sqrt(prob_stab["probability_mse"])
        noise_retention = noise_res[0.2]["metrics"].get("roc_auc", 0.5) / metrics_base.get("roc_auc", 0.5)
        missing_retention = miss_res["sweep_results"][0.2]["metrics"].get("roc_auc", 0.5) / metrics_base.get("roc_auc", 0.5)
        
        rob_score = calculate_robustness_score(
            prediction_stability=pred_stab["jaccard_stability"],
            probability_stability=prob_stability_metric,
            explanation_stability=exp_stab["explanation_stability_rho"],
            feature_stability=rank_stab["ranking_stability_rho"],
            noise_retention=noise_retention,
            missing_data_retention=missing_retention,
            cross_cohort_retention=cross_cohort_retention,
        )
        
        # Save individual model report
        model_report = {
            "model_name": model_name,
            "baseline_metrics": metrics_base,
            "repeated_run_stats": rep_results["metric_stats"],
            "prediction_stability": pred_stab,
            "probability_stability": prob_stab,
            "ranking_stability": rank_stab,
            "explanation_stability": exp_stab,
            "bootstrap_stats": boot_res["bootstrap_stats"],
            "hierarchical_ablation": hier_ablation,
            "individual_ablation": ind_ablation,
            "noise_robustness": noise_res,
            "missing_data_robustness": miss_res["sweep_results"],
            "failure_threshold_missingness": miss_res["failure_threshold"],
            "covariate_shift": covariate_shift,
            "cross_cohort_drift": cross_cohort_res,
            "failure_analysis": unc_res["failure_report"],
            "robustness_score": rob_score,
        }
        
        with (model_dir / "model_robustness_report.json").open("w") as fh:
            json.dump(model_report, fh, indent=2)
            
        all_models_results[model_name] = model_report
        logger.info("Robustness Score for %s: %.3f (%s)", model_name, rob_score["overall_score"], rob_score["grade"])
        
    # Generate publication tables for all models
    summary_df = generate_publication_table(all_models_results, exp_dir / "reports" / "robustness_publication_table.csv")
    
    # Save overall experiment report
    with (exp_dir / "reports" / "experiment_robustness_report.json").open("w") as fh:
        json.dump(all_models_results, fh, indent=2)
        
    logger.info("Saved all experiment reports to %s", exp_dir / "reports")
    return all_models_results

def main() -> None:
    parser = argparse.ArgumentParser(description="AETHEL Robustness Framework Orchestrator")
    parser.add_argument(
        "--mode", type=str, choices=["dev", "paper"], default="dev",
        help="Execution mode (dev runs fast sweeps/few seeds, paper runs full sweeps/100 seeds)"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Specify a single experiment config to run (e.g. exp_mode1_synthetic)"
    )
    args = parser.parse_args()
    
    configure_logging()
    ensure_output_dirs()
    
    # Find experiments
    exp_base = PROJECT_ROOT / "configs" / "experiments"
    if args.experiment:
        configs = [exp_base / f"{args.experiment}.yaml"]
    else:
        configs = sorted(exp_base.glob("*.yaml"))
        
    # Filter out empty gitkeeps
    configs = [c for c in configs if c.name != ".gitkeep"]
    
    runner = ExperimentRunner()
    
    t_start = time.time()
    logger.info("Robustness Orchestrator: starting analysis across %d configs", len(configs))
    
    for cfg_path in configs:
        if not cfg_path.exists():
            logger.error("Config path does not exist: %s", cfg_path)
            continue
            
        exp_cfg = ExperimentConfig.from_yaml(cfg_path)
        # Skip unsupported mode 3 (NHANES unsupervised)
        if exp_cfg.mode == 3:
            logger.info("Skipping Mode 3 (NHANES) - unsupervised cohort")
            continue
            
        if exp_cfg.domain_shift_only:
            logger.info("Skipping domain-shift-only experiment: %s", exp_cfg.name)
            continue
            
        run_experiment_robustness(cfg_path.stem, exp_cfg, runner, args.mode)
        
    duration = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Robustness Orchestrator complete in %.1f seconds", duration)
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
