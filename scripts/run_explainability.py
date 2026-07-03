"""
run_explainability.py
=====================
Publication-grade Explainability Framework Orchestrator for AETHEL.

Runs all 9 explainability modules across experiment configurations and
saves structured reports + publication-quality figures to
outputs/explainability/{experiment_name}/{model_name}/

Usage
-----
    # Fast development mode (LR + RF + XGB):
    python -m scripts.run_explainability --mode dev

    # Full paper mode (all 5 models):
    python -m scripts.run_explainability --mode paper

    # Single experiment:
    python -m scripts.run_explainability --mode dev --experiment exp_mode1_synthetic

Configurable Modes
------------------
  dev   : ["Logistic Regression", "Random Forest", "XGBoost"]  (~5-7 min)
  paper : all 5 models                                          (~15-20 min)

Output structure:
    outputs/explainability/
    └── {exp_name}/
        └── {model_name}/
            ├── shap/              SHAP summary, bar, beeswarm, dependences
            ├── permutation/       Importance bar + violin
            ├── pdp/               PDP + ICE per feature + grid
            ├── ale/               ALE per feature + grid
            ├── interactions/      Heatmap + dependence plots
            ├── local/             Waterfall + contributions for 5 patients
            ├── stability/         Violin + ranking heatmap + stability.json
            ├── consensus/         Rank correlation heatmap + consensus bar
            └── reports/
                ├── global_explainability_report.json
                ├── local_explanation_report.json
                ├── feature_interaction_report.json
                ├── stability_report.json
                ├── consensus_report.json
                └── clinical_interpretation_report.json
    └── cross_cohort_report.json   (written once after all experiments)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config_loader import load_config
from src.utils.logging_setup import configure_logging, get_logger
from src.utils.paths import ensure_output_dirs, PROJECT_ROOT, ExplainabilityDirs
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner

from src.explainability.shap_analysis import SHAPAnalyser
from src.explainability.permutation_analysis import PermutationAnalyser
from src.explainability.pdp_analysis import PDPAnalyser
from src.explainability.ale_analysis import ALEAnalyser
from src.explainability.interaction_analysis import InteractionAnalyser
from src.explainability.local_explanations import LocalExplainer
from src.explainability.stability_analysis import StabilityAnalyser
from src.explainability.consensus_analysis import ConsensusAnalyser
from src.explainability.clinical_interpretation import ClinicalInterpreter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configurable Model Suites
# ---------------------------------------------------------------------------

ALL_CLASSIFIERS = {
    "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    "Decision Tree":       (DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
    "Random Forest":       (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
    "XGBoost":             (XGBClassifier, {"eval_metric": "logloss", "random_state": 42}),
    "LightGBM":            (LGBMClassifier, {"verbose": -1, "random_state": 42}),
}

EXPERIMENT_MODES = {
    "dev": ["Logistic Regression", "Random Forest", "XGBoost"],
    "paper": list(ALL_CLASSIFIERS.keys()),
}

# Stability analysis uses fewer seeds in dev mode for speed
STABILITY_SEEDS = {
    "dev": [42, 123, 456],
    "paper": [42, 123, 456, 789, 1000],
}

# ---------------------------------------------------------------------------
# Preprocessing helper (fit-on-train-only, consistent with evaluation pipeline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-model explainability pipeline
# ---------------------------------------------------------------------------

def run_model_explainability(
    model_name: str,
    model_class: type,
    model_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    exp_name: str,
    run_mode: str,
) -> dict:
    """
    Run full explainability pipeline for a single (model, experiment) pair.
    Returns dict summarising outputs for cross-cohort reporting.
    """
    t0 = time.time()
    safe_model = model_name.replace(" ", "_").lower()
    base_dir = ExplainabilityDirs.BASE / exp_name / safe_model
    base_dir.mkdir(parents=True, exist_ok=True)

    # Sub-directories
    dirs = {
        "shap": base_dir / "shap",
        "permutation": base_dir / "permutation",
        "pdp": base_dir / "pdp",
        "ale": base_dir / "ale",
        "interactions": base_dir / "interactions",
        "local": base_dir / "local",
        "stability": base_dir / "stability",
        "consensus": base_dir / "consensus",
        "reports": base_dir / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Explainability: %s | %s [%s mode]", exp_name, model_name, run_mode)
    logger.info("=" * 60)

    # --- Preprocess ---
    X_tr, X_val_pp, _, _ = preprocess_splits(X_train, X_val, features)

    # --- Fit model ---
    logger.info("  Fitting model: %s", model_name)
    model = model_class(**model_kwargs)
    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_val_pp)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ----------------------------------------------------------------
    # 1. SHAP Analysis
    # ----------------------------------------------------------------
    logger.info("  [1/8] SHAP analysis...")
    bg_size = min(200, len(X_tr))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_tr), size=bg_size, replace=False)
    X_background = X_tr.iloc[bg_idx].reset_index(drop=True)

    shap_analyser = SHAPAnalyser(model, X_background, features, model_name)
    shap_analyser.fit()
    shap_values = shap_analyser.compute_shap_values(X_val_pp)

    shap_analyser.plot_summary(X_val_pp, dirs["shap"] / "shap_summary.png")
    shap_analyser.plot_bar(dirs["shap"] / "shap_bar.png")
    shap_analyser.plot_beeswarm(X_val_pp, dirs["shap"] / "shap_beeswarm.png")
    shap_analyser.plot_all_dependences(X_val_pp, dirs["shap"])

    mean_abs_shap = shap_analyser.mean_abs_shap()

    # ----------------------------------------------------------------
    # 2. Permutation Importance
    # ----------------------------------------------------------------
    logger.info("  [2/8] Permutation importance...")
    perm_analyser = PermutationAnalyser(model, features, model_name, n_repeats=20)
    perm_df = perm_analyser.compute(X_val_pp, y_val)
    perm_df.to_csv(dirs["permutation"] / "permutation_importance.csv", index=False)
    perm_analyser.plot_bar(dirs["permutation"] / "permutation_bar.png")
    perm_analyser.plot_violin(dirs["permutation"] / "permutation_violin.png")

    # ----------------------------------------------------------------
    # 3. PDP + ICE
    # ----------------------------------------------------------------
    logger.info("  [3/8] PDP + ICE...")
    pdp_analyser = PDPAnalyser(model, features, model_name)
    pdp_analyser.compute(X_val_pp)
    pdp_analyser.plot_all_features(dirs["pdp"])

    # ----------------------------------------------------------------
    # 4. ALE
    # ----------------------------------------------------------------
    logger.info("  [4/8] ALE...")
    ale_analyser = ALEAnalyser(model, features, model_name)
    ale_analyser.compute(X_val_pp)
    ale_analyser.plot_all_features(dirs["ale"])

    # ----------------------------------------------------------------
    # 5. Interactions (tree models only)
    # ----------------------------------------------------------------
    logger.info("  [5/8] Interaction analysis...")
    interaction_analyser = InteractionAnalyser(model, features, model_name)
    interaction_matrix = interaction_analyser.compute(X_val_pp, max_samples=150)
    interaction_report = []
    if interaction_matrix is not None:
        interaction_analyser.plot_interaction_heatmap(dirs["interactions"] / "interaction_heatmap.png")
        interaction_analyser.plot_top_interaction_dependences(X_val_pp, shap_values, dirs["interactions"])
        interaction_report = interaction_analyser.get_top_interactions(k=10)
    with (dirs["reports"] / "feature_interaction_report.json").open("w") as fh:
        json.dump({
            "model": model_name,
            "supported": interaction_analyser.is_supported(),
            "top_interactions": interaction_report,
        }, fh, indent=2)

    # ----------------------------------------------------------------
    # 6. Local Explanations (5 representative patients)
    # ----------------------------------------------------------------
    logger.info("  [6/8] Local explanations (5 patients)...")
    local_explainer = LocalExplainer(model, shap_analyser, features, model_name)
    local_reports = local_explainer.batch_explain(
        X_val_pp, y_val, y_prob, y_pred, shap_values, dirs["local"]
    )

    # ----------------------------------------------------------------
    # 7. Stability Analysis
    # ----------------------------------------------------------------
    logger.info("  [7/8] Stability analysis (%s mode)...", run_mode)
    seeds = STABILITY_SEEDS[run_mode]
    stability_analyser = StabilityAnalyser(model_class, model_kwargs, features, model_name, seeds=seeds)
    stability_report = stability_analyser.run(X_train, y_train, X_val, y_val)
    stability_analyser.plot_stability_violin(dirs["stability"] / "stability_violin.png")
    stability_analyser.plot_ranking_heatmap(dirs["stability"] / "ranking_heatmap.png")
    stability_analyser.save_report(dirs["reports"] / "stability_report.json")

    # ----------------------------------------------------------------
    # 8. Consensus Analysis
    # ----------------------------------------------------------------
    logger.info("  [8/8] Consensus analysis...")
    consensus_analyser = ConsensusAnalyser(features, model_name)
    consensus_analyser.add_source("SHAP", mean_abs_shap)
    consensus_analyser.add_source("Permutation", perm_analyser.get_ranking())
    consensus_analyser.add_model_native(model)

    consensus_report = consensus_analyser.build(top_k=min(3, len(features)))
    consensus_analyser.plot_rank_correlation_heatmap(dirs["consensus"] / "rank_correlation_heatmap.png")
    consensus_analyser.plot_consensus_bar(dirs["consensus"] / "consensus_bar.png")
    consensus_analyser.save_report(dirs["reports"] / "consensus_report.json")

    # ----------------------------------------------------------------
    # Clinical Interpretation Report
    # ----------------------------------------------------------------
    interpreter = ClinicalInterpreter(features)
    interpreter.generate_feature_report(
        mean_abs_shap.to_dict(),
        dirs["reports"] / "clinical_interpretation_report.json",
    )

    # ----------------------------------------------------------------
    # Global Explainability Report (aggregates all metrics)
    # ----------------------------------------------------------------
    global_report = {
        "experiment": exp_name,
        "model": model_name,
        "run_mode": run_mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
        "features": features,
        "n_val_samples": len(X_val_pp),
        "shap_mean_abs_importance": mean_abs_shap.to_dict(),
        "permutation_importance": perm_df.set_index("feature")["mean_importance"].to_dict(),
        "explanation_quality_metrics": {
            "feature_agreement_score": consensus_report.get("feature_agreement_score", {}).get("value"),
            "top_k_overlap": consensus_report.get("top_k_overlap", {}).get("value"),
            "explanation_coverage": consensus_report.get("explanation_coverage", {}).get("value"),
            "ranking_stability_spearman_rho": stability_report.get("ranking_stability", {}).get("mean_spearman_rho"),
            "stability_grade": stability_report.get("ranking_stability", {}).get("stability_grade"),
            "consensus_index_fas_grade": consensus_report.get("feature_agreement_score", {}).get("grade"),
        },
        "interaction_analysis_supported": interaction_analyser.is_supported(),
        "top_interactions": interaction_report[:5],
    }
    with (dirs["reports"] / "global_explainability_report.json").open("w") as fh:
        json.dump(global_report, fh, indent=2)

    logger.info(
        "  Done: %s | %s in %.1fs | FAS=%.3f | Stability rho=%.3f",
        exp_name, model_name,
        time.time() - t0,
        consensus_report.get("feature_agreement_score", {}).get("value", float("nan")),
        stability_report.get("ranking_stability", {}).get("mean_spearman_rho", float("nan")),
    )

    return {
        "exp": exp_name,
        "model": model_name,
        "shap_ranking": mean_abs_shap.to_dict(),
        "consensus_fas": consensus_report.get("feature_agreement_score", {}).get("value"),
        "stability_rho": stability_report.get("ranking_stability", {}).get("mean_spearman_rho"),
    }


# ---------------------------------------------------------------------------
# Experiment-level orchestrator
# ---------------------------------------------------------------------------

def run_experiment_explainability(
    exp_cfg: ExperimentConfig,
    runner: ExperimentRunner,
    model_suite: list[str],
    run_mode: str,
) -> list[dict]:
    """Run explainability for all models in the suite for one experiment."""
    logger.info("Loading data for experiment: %s", exp_cfg.name)
    result = runner.run(exp_cfg)
    features = result.feature_set
    outcome = result.outcome_col
    train_df = result.train_df
    val_df = result.val_df

    if val_df is None or len(val_df) == 0:
        logger.warning("No validation split for %s — skipping.", exp_cfg.name)
        return []

    X_train = train_df[features].copy()
    y_train = train_df[outcome]
    X_val = val_df[features].copy()
    y_val = val_df[outcome]

    summaries = []
    for model_name in model_suite:
        model_class, model_kwargs = ALL_CLASSIFIERS[model_name]
        try:
            summary = run_model_explainability(
                model_name=model_name,
                model_class=model_class,
                model_kwargs=model_kwargs,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                features=features,
                exp_name=exp_cfg.name,
                run_mode=run_mode,
            )
            summaries.append(summary)
        except Exception as e:
            logger.exception(
                "Explainability failed for %s / %s: %s", exp_cfg.name, model_name, e
            )
    return summaries


# ---------------------------------------------------------------------------
# Cross-cohort comparison
# ---------------------------------------------------------------------------

def generate_cross_cohort_report(all_summaries: list[dict]) -> None:
    """Generate cross-cohort explainability comparison from all experiment summaries."""
    if not all_summaries:
        return

    # Group by model, collect SHAP rankings per experiment (cohort)
    model_to_cohort_rankings: dict[str, dict[str, dict]] = {}
    for s in all_summaries:
        model = s["model"]
        exp = s["exp"]
        if model not in model_to_cohort_rankings:
            model_to_cohort_rankings[model] = {}
        model_to_cohort_rankings[model][exp] = s["shap_ranking"]

    cross_cohort_report = {}
    for model_name, cohort_rankings in model_to_cohort_rankings.items():
        if len(cohort_rankings) < 2:
            continue
        interpreter = ClinicalInterpreter(list(list(cohort_rankings.values())[0].keys()))
        out_path = (
            ExplainabilityDirs.BASE /
            f"cross_cohort_{model_name.replace(' ', '_').lower()}.json"
        )
        report = interpreter.generate_cross_cohort_comparison(cohort_rankings, out_path)
        cross_cohort_report[model_name] = report

    with (ExplainabilityDirs.BASE / "cross_cohort_report.json").open("w") as fh:
        json.dump(cross_cohort_report, fh, indent=2)
    logger.info("Saved cross-cohort report: %s", ExplainabilityDirs.BASE / "cross_cohort_report.json")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AETHEL Explainability Framework Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "paper"],
        default="dev",
        help=(
            "dev  = LR + RF + XGB (~5-7 min)\n"
            "paper = all 5 models  (~15-20 min)"
        ),
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run only a specific experiment by name (e.g. exp_mode1_synthetic). "
             "Defaults to all eligible experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_output_dirs()

    model_suite = EXPERIMENT_MODES[args.mode]
    logger.info("=" * 70)
    logger.info("AETHEL Explainability Framework | mode=%s | models=%s", args.mode, model_suite)
    logger.info("=" * 70)

    runner = ExperimentRunner()
    exp_configs_dir = PROJECT_ROOT / "configs" / "experiments"
    yaml_files = sorted(exp_configs_dir.glob("*.yaml"))

    if not yaml_files:
        logger.error("No experiment YAML files found in %s", exp_configs_dir)
        return

    all_summaries = []
    for yf in yaml_files:
        # Skip domain-shift-only and NHANES-only (no outcome labels)
        if "domain_shift" in yf.name.lower():
            logger.info("Skipping domain-shift-only experiment: %s", yf.name)
            continue
        if "nhanes" in yf.name.lower() and not (
            "synthetic" in yf.name.lower() or "framingham" in yf.name.lower()
        ):
            logger.info("Skipping NHANES-only experiment: %s", yf.name)
            continue

        # Filter by --experiment flag if specified
        if args.experiment and yf.stem != args.experiment:
            continue

        try:
            exp_cfg = ExperimentConfig.from_yaml(yf)
            summaries = run_experiment_explainability(
                exp_cfg, runner, model_suite, args.mode
            )
            all_summaries.extend(summaries)
        except Exception as e:
            logger.exception("Failed experiment %s: %s", yf.name, e)

    # Cross-cohort comparison
    logger.info("Generating cross-cohort explainability comparison...")
    generate_cross_cohort_report(all_summaries)

    logger.info("=" * 70)
    logger.info("All explainability analyses complete.")
    logger.info("Reports saved to: %s", ExplainabilityDirs.BASE)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
