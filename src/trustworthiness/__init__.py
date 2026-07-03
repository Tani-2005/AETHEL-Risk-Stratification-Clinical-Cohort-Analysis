"""
AETHEL clinical AI trustworthiness assessment package.
"""
from src.trustworthiness.clinical_consistency import evaluate_clinical_consistency
from src.trustworthiness.trustworthiness_evaluator import build_trustworthiness_profile
from src.trustworthiness.publication_tables import (
    build_table_1_characteristics,
    build_table_2_internal_performance,
    build_table_3_external_validation,
    build_table_4_calibration_comparison,
    build_table_5_explanation_stability,
    build_table_6_generalization_analysis,
    build_table_7_failure_analysis,
    build_table_8_trustworthiness_summary,
)
from src.trustworthiness.visualizations import (
    plot_performance_drop,
    plot_calibration_transfer_curves,
    plot_explanation_drift_heatmap,
    plot_domain_shift_kde,
    plot_trustworthiness_radar,
    plot_failure_clusters,
)
