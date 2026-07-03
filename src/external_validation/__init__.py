"""
AETHEL external validation package.
"""
from src.external_validation.validation_runner import (
    load_and_align_cohorts,
    get_surrogate_outcome,
    preprocess_cross_cohort,
    BIOCHEMICAL_FEATURES,
    COMMON_INTERSECTION,
)
from src.external_validation.calibration_transfer import evaluate_calibration_transfer
from src.external_validation.uncertainty_transfer import evaluate_uncertainty_transfer
from src.external_validation.failure_analysis import run_failure_analysis
