"""
config_loader.py
================
Centralised YAML configuration loader for AETHEL.

All pipeline scripts should obtain their parameters by calling
``load_config()`` rather than hardcoding values inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import ConfigPaths


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    name: str = "AETHEL"
    version: str = "3.0.0"
    description: str = ""


@dataclass
class PipelineControls:
    generate_environment: bool = True
    generate_clinical: bool = True
    run_feature_engineering: bool = True
    run_validation: bool = True
    run_feature_engineering_derived: bool = True
    run_splitting: bool = True
    run_preprocessing_pipeline: bool = True
    run_feature_selection: bool = True
    execute_survival_model: bool = True


@dataclass
class SeedConfig:
    python: int = 42
    r: int = 123


@dataclass
class StudyConfig:
    n_subjects: int = 1000
    observation_years: int = 5
    total_cities: int = 90


@dataclass
class ModelParams:
    rsf_ntree: int = 500
    rsf_importance: bool = True


@dataclass
class FeaturesConfig:
    survival_covariates: list[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    """Train/val/test split and scaler configuration."""
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    stratify_on: str = "event_occurred"
    scaler_type: str = "robust"   # robust | standard | minmax


@dataclass
class ValidationConfig:
    """Clinical plausibility bounds per feature."""
    stop_on_error: bool = False
    clinical_bounds: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class FeatureEngineeringConfig:
    """Toggles and parameters for derived feature construction."""
    add_bmi_category: bool = True
    add_age_group: bool = True
    add_pollution_index: bool = True
    add_lifestyle_risk: bool = True
    add_high_genomic_risk: bool = True
    genomic_risk_threshold: float = 1.0


@dataclass
class FeatureSelectionConfig:
    """Thresholds for feature selection (applied on training fold only)."""
    vif_threshold: float = 10.0
    correlation_threshold: float = 0.90
    mi_top_k: int = 10


@dataclass
class AETHELConfig:
    """
    Fully-typed representation of ``configs/default.yaml``.

    Attributes
    ----------
    project         : Project metadata.
    pipeline        : Stage toggle flags.
    seeds           : Deterministic RNG seeds for Python and R.
    study           : Cohort size, observation window, city count.
    model_params    : RSF hyperparameters.
    features        : Covariate lists for survival models.
    preprocessing   : Train/val/test split ratios and scaler choice.
    validation      : Clinical bound checks per column.
    feature_eng     : Derived feature construction toggles.
    feature_sel     : Feature selection thresholds.
    dashboard_regions: Default highlighted regions in the Streamlit UI.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    pipeline: PipelineControls = field(default_factory=PipelineControls)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    model_params: ModelParams = field(default_factory=ModelParams)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    feature_eng: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    feature_sel: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    dashboard_regions: list[str] = field(default_factory=list)
    datasets: dict[str, Any] = field(default_factory=dict)
    harmonization: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> AETHELConfig:
    """Load and parse ``configs/default.yaml`` into a typed ``AETHELConfig``."""
    path = config_path or ConfigPaths.DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Ensure you are running scripts from the AETHEL project root."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return _parse(raw)


def _parse(raw: dict[str, Any]) -> AETHELConfig:
    """Map raw YAML dict to ``AETHELConfig`` dataclass hierarchy."""
    p = raw.get("project", {})
    pipeline_raw = raw.get("pipeline_controls", {})
    seed_raw = raw.get("seeds", {})
    study_raw = raw.get("study_parameters", {})
    model_raw = raw.get("model_params", {})
    feat_raw = raw.get("features", {})
    pre_raw = raw.get("preprocessing", {})
    val_raw = raw.get("validation", {})
    feng_raw = raw.get("feature_engineering", {})
    fsel_raw = raw.get("feature_selection", {})
    datasets_raw = raw.get("datasets", {})
    harmonization_raw = raw.get("harmonization", {})

    return AETHELConfig(
        project=ProjectConfig(
            name=p.get("name", "AETHEL"),
            version=p.get("version", "3.0.0"),
            description=p.get("description", ""),
        ),
        pipeline=PipelineControls(
            generate_environment=pipeline_raw.get("generate_environment", True),
            generate_clinical=pipeline_raw.get("generate_clinical", True),
            run_feature_engineering=pipeline_raw.get("run_feature_engineering", True),
            run_validation=pipeline_raw.get("run_validation", True),
            run_feature_engineering_derived=pipeline_raw.get("run_feature_engineering_derived", True),
            run_splitting=pipeline_raw.get("run_splitting", True),
            run_preprocessing_pipeline=pipeline_raw.get("run_preprocessing_pipeline", True),
            run_feature_selection=pipeline_raw.get("run_feature_selection", True),
            execute_survival_model=pipeline_raw.get("execute_survival_model", True),
        ),
        seeds=SeedConfig(
            python=seed_raw.get("python", 42),
            r=seed_raw.get("r", 123),
        ),
        study=StudyConfig(
            n_subjects=study_raw.get("n_subjects", 1000),
            observation_years=study_raw.get("observation_years", 5),
            total_cities=study_raw.get("total_cities", 90),
        ),
        model_params=ModelParams(
            rsf_ntree=model_raw.get("rsf_ntree", 500),
            rsf_importance=model_raw.get("rsf_importance", True),
        ),
        features=FeaturesConfig(
            survival_covariates=feat_raw.get("survival_covariates", []),
        ),
        preprocessing=PreprocessingConfig(
            train_size=pre_raw.get("train_size", 0.70),
            val_size=pre_raw.get("val_size", 0.15),
            test_size=pre_raw.get("test_size", 0.15),
            stratify_on=pre_raw.get("stratify_on", "event_occurred"),
            scaler_type=pre_raw.get("scaler_type", "robust"),
        ),
        validation=ValidationConfig(
            stop_on_error=val_raw.get("stop_on_error", False),
            clinical_bounds=val_raw.get("clinical_bounds", {}),
        ),
        feature_eng=FeatureEngineeringConfig(
            add_bmi_category=feng_raw.get("add_bmi_category", True),
            add_age_group=feng_raw.get("add_age_group", True),
            add_pollution_index=feng_raw.get("add_pollution_index", True),
            add_lifestyle_risk=feng_raw.get("add_lifestyle_risk", True),
            add_high_genomic_risk=feng_raw.get("add_high_genomic_risk", True),
            genomic_risk_threshold=feng_raw.get("genomic_risk_threshold", 1.0),
        ),
        feature_sel=FeatureSelectionConfig(
            vif_threshold=fsel_raw.get("vif_threshold", 10.0),
            correlation_threshold=fsel_raw.get("correlation_threshold", 0.90),
            mi_top_k=fsel_raw.get("mi_top_k", 10),
        ),
        dashboard_regions=raw.get("default_dashboard_regions", []),
        datasets=datasets_raw,
        harmonization=harmonization_raw,
    )

