"""
run_evaluation.py
=================
Publication-grade Model Evaluation Orchestrator for AETHEL.
Runs complete statistical, clinical, and calibration evaluation across all
experiments (Modes 1, 2, 4, 7), producing curves, tables, and CIs.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config_loader import load_config
from src.utils.logging_setup import configure_logging, get_logger
from src.utils.paths import ensure_output_dirs, PROJECT_ROOT
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner
from src.evaluation.evaluator import (
    calculate_metrics, BootstrapEvaluator, LeakageFreeCV,
    run_paired_bootstrap_test, run_mcnemar_test, perform_error_analysis
)

logger = get_logger(__name__)

# Output directories for evaluation reports
EVAL_OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of classifier models to train and evaluate
CLASSIFIER_SUITE = {
    "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    "Decision Tree": (DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
    "Random Forest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
    "XGBoost": (XGBClassifier, {"eval_metric": "logloss", "random_state": 42}),
    "LightGBM": (LGBMClassifier, {"verbose": -1, "random_state": 42})
}


def run_survival_r_script(train_path: Path, val_path: Path, test_path: Path, out_dir: Path) -> bool:
    """
    Invoke train_survival.R script via subprocess to fit Cox/RSF models
    and generate risk score predictions.
    """
    import shutil
    import glob
    
    r_exe = shutil.which("Rscript")
    if not r_exe:
        r_paths = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True)
        r_exe = r_paths[0] if r_paths else "Rscript"

    r_script = PROJECT_ROOT / "src" / "models" / "train_survival.R"
    t_path_str = str(test_path) if test_path.exists() else "none"
    
    logger.info("Running R survival model fitting for split data...")
    cmd = [r_exe, str(r_script), str(train_path), str(val_path), t_path_str, str(out_dir)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("R survival modeling failed: %s", e.stderr)
        return False


def load_survival_predictions(out_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    """
    Load Cox PH and RSF predicted risk scores from R output CSV.
    """
    pred_path = out_dir / f"{split_name}_survival_preds.csv"
    if not pred_path.exists():
        return {}
    df = pd.read_csv(pred_path)
    if len(df) == 0:
        return {}
    return {
        "Cox PH": df["cox_risk"].values,
        "Random Survival Forest": df["rsf_risk"].values
    }


def evaluate_experiment(exp_cfg: ExperimentConfig, runner: ExperimentRunner) -> None:
    logger.info("=" * 70)
    logger.info("Evaluating Experiment: %s (Mode %d)", exp_cfg.name, exp_cfg.mode)
    logger.info("=" * 70)
    
    # 1. Load split datasets
    result = runner.run(exp_cfg)
    train_df = result.train_df
    val_df = result.val_df
    test_df = result.test_df
    features = result.feature_set
    outcome_col = result.outcome_col
    
    exp_dir = EVAL_OUT_DIR / exp_cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the split paths for the R script
    train_split_path = exp_dir / "train_split.csv"
    val_split_path = exp_dir / "val_split.csv"
    test_split_path = exp_dir / "test_split.csv"
    
    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)
    if test_df is not None:
        test_df.to_csv(test_split_path, index=False)
        
    # 2. Fit and Predict Classifiers
    # Extract feature matrices and outcome targets
    X_train = train_df[features].copy()
    y_train = train_df[outcome_col]
    X_val = val_df[features].copy()
    y_val = val_df[outcome_col]
    
    # Store predictions: model_name -> dict(y_prob, y_pred)
    val_predictions = {}
    test_predictions = {}
    
    # Impute and scale classifiers (pre-fitted on training data)
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=features)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
    
    X_val_imp = pd.DataFrame(imp.transform(X_val), columns=features)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=features)
    
    X_test_scaled = None
    y_test = None
    if test_df is not None:
        X_test = test_df[features].copy()
        y_test = test_df[outcome_col]
        X_test_imp = pd.DataFrame(imp.transform(X_test), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)
        
    for name, (model_class, kwargs) in CLASSIFIER_SUITE.items():
        logger.info("Training classifier: %s", name)
        clf = model_class(**kwargs)
        clf.fit(X_train_scaled, y_train)
        
        val_prob = clf.predict_proba(X_val_scaled)[:, 1]
        val_predictions[name] = {"prob": val_prob, "pred": (val_prob >= 0.5).astype(int)}
        
        if X_test_scaled is not None:
            test_prob = clf.predict_proba(X_test_scaled)[:, 1]
            test_predictions[name] = {"prob": test_prob, "pred": (test_prob >= 0.5).astype(int)}
            
    # 3. Fit and Load Survival Models
    r_success = run_survival_r_script(train_split_path, val_split_path, test_split_path, exp_dir)
    if r_success:
        val_surv = load_survival_predictions(exp_dir, "val")
        test_surv = load_survival_predictions(exp_dir, "test") if test_df is not None else {}
        
        for name, prob in val_surv.items():
            if len(prob) > 0 and not np.isnan(prob).any():
                p_min, p_max = prob.min(), prob.max()
                prob_scaled = (prob - p_min) / (p_max - p_min) if p_max > p_min else prob
                val_predictions[name] = {"prob": prob_scaled, "pred": (prob_scaled >= 0.5).astype(int)}
                
        for name, prob in test_surv.items():
            if len(prob) > 0 and not np.isnan(prob).any():
                p_min, p_max = prob.min(), prob.max()
                prob_scaled = (prob - p_min) / (p_max - p_min) if p_max > p_min else prob
                test_predictions[name] = {"prob": prob_scaled, "pred": (prob_scaled >= 0.5).astype(int)}
                
    # Clean up split CSVs
    for p in [train_split_path, val_split_path, test_split_path]:
        if p.exists():
            p.unlink()

    # 4. Repeated Stratified K-Fold CV (inside training fold, Python classifiers only)
    logger.info("Running Repeated Stratified K-Fold CV (5x10-folds)...")
    cv_runner = LeakageFreeCV(n_splits=10, n_repeats=5, seed=exp_cfg.seed)
    cv_summaries = {}
    for name, (model_class, kwargs) in CLASSIFIER_SUITE.items():
        cv_res = cv_runner.run(X_train, y_train, model_class, kwargs)
        cv_summaries[name] = cv_res["summary"]
        cv_res["raw_folds"].to_csv(exp_dir / f"cv_folds_{name.replace(' ', '_').lower()}.csv", index=False)
        
    with (exp_dir / "cv_summary.json").open("w") as fh:
        json.dump(cv_summaries, fh, indent=2)

    # 5. Compute Metrics on Validation and Test Split (with Bootstrapping CIs)
    # Evaluate on Validation set first
    logger.info("Computing metrics and bootstrap CIs on Validation split...")
    boot_eval = BootstrapEvaluator(n_bootstrap=1000, seed=exp_cfg.seed)
    
    time_c = val_df["months_observed"].values if "months_observed" in val_df.columns else None
    event_c = val_df["event_occurred"].values if "event_occurred" in val_df.columns else None
    
    val_results = []
    for model_name, preds in val_predictions.items():
        probs = preds["prob"]
        is_surv = "cox" in model_name.lower() or "survival" in model_name.lower() or "rsf" in model_name.lower()
        
        # Core metrics
        metrics = calculate_metrics(y_val, probs, time_col=time_c, event_col=event_c, is_survival=is_surv)
        
        # Bootstrap CIs
        cis, df_boot = boot_eval.evaluate(y_val, probs, time_col=time_c, event_col=event_c, is_survival=is_surv)
        df_boot.to_csv(exp_dir / f"bootstrap_dist_val_{model_name.replace(' ', '_').lower()}.csv", index=False)
        
        row = {"model": model_name}
        for metric, val in metrics.items():
            ci = cis.get(metric, (np.nan, np.nan))
            row[metric] = val
            row[f"{metric}_ci_lower"] = ci[0]
            row[f"{metric}_ci_upper"] = ci[1]
            
        val_results.append(row)
        
        # Recalibration & DCA
        if not is_surv:
            # 1. Decision Curve Analysis (DCA)
            from src.evaluation.evaluator import run_dca
            val_dca = run_dca(y_val, probs)
            val_dca.to_csv(exp_dir / f"dca_val_{model_name.replace(' ', '_').lower()}.csv", index=False)
            
            if test_predictions and model_name in test_predictions:
                test_probs = test_predictions[model_name]["prob"]
                test_dca = run_dca(y_test, test_probs)
                test_dca.to_csv(exp_dir / f"dca_test_{model_name.replace(' ', '_').lower()}.csv", index=False)
                
                # 2. Recalibration comparison (Platt vs Isotonic)
                from src.calibration.recalibration import compare_recalibration
                cal_comparison = compare_recalibration(y_val, probs, y_test, test_probs)
                with open(exp_dir / f"calibration_recalibration_{model_name.replace(' ', '_').lower()}.json", "w") as fh:
                    json.dump(cal_comparison, fh, indent=2)
        
    df_val_results = pd.DataFrame(val_results)
    df_val_results.to_csv(exp_dir / "metrics_val.csv", index=False)
    
    # If test split exists, evaluate on Test set
    df_test_results = None
    if test_predictions:
        logger.info("Computing metrics and bootstrap CIs on Test split...")
        test_results = []
        
        time_c_test = test_df["months_observed"].values if (test_df is not None and "months_observed" in test_df.columns) else None
        event_c_test = test_df["event_occurred"].values if (test_df is not None and "event_occurred" in test_df.columns) else None
        
        for model_name, preds in test_predictions.items():
            probs = preds["prob"]
            is_surv = "cox" in model_name.lower() or "survival" in model_name.lower() or "rsf" in model_name.lower()
            
            metrics = calculate_metrics(y_test, probs, time_col=time_c_test, event_col=event_c_test, is_survival=is_surv)
            cis, df_boot = boot_eval.evaluate(y_test, probs, time_col=time_c_test, event_col=event_c_test, is_survival=is_surv)
            df_boot.to_csv(exp_dir / f"bootstrap_dist_test_{model_name.replace(' ', '_').lower()}.csv", index=False)
            
            row = {"model": model_name}
            for metric, val in metrics.items():
                ci = cis.get(metric, (np.nan, np.nan))
                row[metric] = val
                row[f"{metric}_ci_lower"] = ci[0]
                row[f"{metric}_ci_upper"] = ci[1]
                
            test_results.append(row)
        df_test_results = pd.DataFrame(test_results)
        df_test_results.to_csv(exp_dir / "metrics_test.csv", index=False)
        
    # 6. Pairwise Statistical Significance Testing
    # Compare all pairs on validation set
    logger.info("Performing pairwise statistical testing...")
    stat_comparisons = []
    models_to_test = list(val_predictions.keys())
    for i, m1 in enumerate(models_to_test):
        for m2 in models_to_test[i+1:]:
            probs1 = val_predictions[m1]["prob"]
            probs2 = val_predictions[m2]["prob"]
            preds1 = val_predictions[m1]["pred"]
            preds2 = val_predictions[m2]["pred"]
            
            # Paired bootstrap for AUC difference
            auc1, auc2, p_auc = run_paired_bootstrap_test(y_val, probs1, probs2, seed=exp_cfg.seed)
            
            # McNemar test on binary classifications
            mcnemar_stat, p_mcnemar = run_mcnemar_test(y_val, preds1, preds2)
            
            stat_comparisons.append({
                "model_1": m1, "model_2": m2,
                "auc_1": auc1, "auc_2": auc2, "auc_difference": auc1 - auc2,
                "p_value_paired_bootstrap": p_auc,
                "mcnemar_statistic": mcnemar_stat, "p_value_mcnemar": p_mcnemar
            })
    pd.DataFrame(stat_comparisons).to_csv(exp_dir / "pairwise_statistical_tests.csv", index=False)

    # 7. Error Analysis
    logger.info("Performing error analysis...")
    for model_name, preds in val_predictions.items():
        err_res = perform_error_analysis(y_val, preds["prob"], val_df[features])
        err_res["raw_errors"].to_csv(exp_dir / f"errors_val_{model_name.replace(' ', '_').lower()}.csv", index=False)
        
        # Save error summary
        summary = {
            "n_false_positives": err_res["n_false_positives"],
            "n_false_negatives": err_res["n_false_negatives"],
            "n_high_confidence_errors": err_res["n_high_confidence_errors"],
            "n_low_confidence_errors": err_res["n_low_confidence_errors"],
        }
        with (exp_dir / f"error_summary_{model_name.replace(' ', '_').lower()}.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

    # 8. Plot Curves (ROC, PR, Calibration, Reliability, Confusion Matrix)
    logger.info("Plotting evaluation curves...")
    sns.set_theme(style="whitegrid")
    
    # A. ROC Curve
    plt.figure(figsize=(8, 6))
    for model_name, preds in val_predictions.items():
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_val, preds["prob"])
        auc_val = roc_auc_score(y_val, preds["prob"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curves - Validation Split ({exp_cfg.name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(exp_dir / "roc_curves_val.png", dpi=300)
    plt.close()
    
    # B. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for model_name, preds in val_predictions.items():
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_val, preds["prob"])
        ap_val = average_precision_score(y_val, preds["prob"])
        plt.plot(recall, precision, label=f"{model_name} (AP = {ap_val:.3f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title(f"Precision-Recall Curves - Validation Split ({exp_cfg.name})")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(exp_dir / "pr_curves_val.png", dpi=300)
    plt.close()
    
    # C. Calibration Curve (Reliability Diagram)
    plt.figure(figsize=(8, 6))
    for model_name, preds in val_predictions.items():
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_val, preds["prob"], n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Rate")
    plt.title(f"Calibration Curves - Validation Split ({exp_cfg.name})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(exp_dir / "calibration_curves_val.png", dpi=300)
    plt.close()

    # D. Confusion Matrices
    for model_name, preds in val_predictions.items():
        cm = confusion_matrix(y_val, preds["pred"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Event", "Event"], yticklabels=["No Event", "Event"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()
        plt.savefig(exp_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png", dpi=150)
        plt.close()

    # 9. Save Metadata JSON for tracking
    tracking_meta = {
        "experiment_name": exp_cfg.name,
        "mode": exp_cfg.mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": exp_cfg.seed,
        "features": features,
        "outcome": outcome_col,
        "datasets": exp_cfg.all_datasets()
    }
    with (exp_dir / "metadata_tracking.json").open("w") as fh:
        json.dump(tracking_meta, fh, indent=2)

    logger.info("Finished evaluation for %s.", exp_cfg.name)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Model Evaluation Orchestrator")
    parser.add_argument("--experiment", type=str, default=None, help="Name of the experiment to run")
    args = parser.parse_args()

    configure_logging()
    ensure_output_dirs()
    
    logger.info("Initializing multi-cohort experiment runner...")
    runner = ExperimentRunner()
    
    # Load all experiment configurations
    exp_configs_dir = PROJECT_ROOT / "configs" / "experiments"
    
    if args.experiment:
        # Check if experiment file exists
        exp_file = exp_configs_dir / f"{args.experiment}.yaml"
        if not exp_file.exists():
            # Fallback to direct path or checking file name
            exp_file = Path(args.experiment)
        if exp_file.exists():
            yaml_files = [exp_file]
        else:
            logger.error("Experiment config file not found: %s", args.experiment)
            return
    else:
        yaml_files = sorted(exp_configs_dir.glob("*.yaml"))
    
    if not yaml_files:
        logger.error("No YAML experiment configurations found in %s", exp_configs_dir)
        return
        
    for yf in yaml_files:
        # Skip domain shift experiments unless explicitly requested
        if not args.experiment and "domain_shift" in yf.name.lower():
            logger.info("Skipping domain-shift-only experiment: %s", yf.name)
            continue

        # NHANES-only experiments lack target outcomes, skip evaluation unless explicitly requested
        if not args.experiment and "nhanes" in yf.name.lower() and not ("synthetic" in yf.name.lower() or "framingham" in yf.name.lower()):
            logger.info("Skipping NHANES-only experiment (unsupervised domain-shift): %s", yf.name)
            continue

        try:
            exp_cfg = ExperimentConfig.from_yaml(yf)
            if exp_cfg.domain_shift_only:
                logger.info("Skipping domain-shift-only experiment: %s", yf.name)
                continue
            evaluate_experiment(exp_cfg, runner)
        except Exception as e:
            logger.exception("Failed to run evaluation for %s: %s", yf.name, str(e))
            
    logger.info("=" * 70)
    logger.info("Evaluation complete. Reports saved to outputs/evaluation/.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
