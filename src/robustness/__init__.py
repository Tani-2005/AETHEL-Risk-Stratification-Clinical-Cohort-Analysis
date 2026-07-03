"""
AETHEL robustness package — robustness and sensitivity experiments.
"""
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
