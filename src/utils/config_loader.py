"""
config_loader.py
================
Centralised YAML configuration loader for AETHEL.

All pipeline scripts should obtain their parameters by calling
``load_config()`` rather than hardcoding values inline.  This ensures:

- One config change propagates to the entire pipeline
- Experiments can be reproduced by archiving a single YAML file
- Future hyperparameter sweeps can override individual fields

Usage
-----
    from src.utils.config_loader import load_config

    cfg = load_config()
    print(cfg.study.n_subjects)     # 1000
    print(cfg.seeds.python)         # 42
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
    version: str = "2.0.0"
    description: str = ""


@dataclass
class PipelineControls:
    generate_environment: bool = True
    generate_clinical: bool = True
    run_feature_engineering: bool = True
    execute_survival_model: bool = True


@dataclass
class SeedConfig:
    python: int = 42
    r: int = 123


@dataclass
class StudyConfig:
    n_subjects: int = 1000
    observation_years: int = 5
    total_cities: int = 100


@dataclass
class ModelParams:
    rsf_ntree: int = 500
    rsf_importance: bool = True


@dataclass
class FeaturesConfig:
    survival_covariates: list[str] = field(default_factory=list)


@dataclass
class AETHELConfig:
    """
    Fully-typed representation of ``configs/default.yaml``.

    Attributes
    ----------
    project       : Project metadata (name, version).
    pipeline      : Toggle individual pipeline stages on/off.
    seeds         : Deterministic random seeds for Python and R.
    study         : Study-level parameters (cohort size, follow-up).
    model_params  : Model hyperparameters (RSF tree count, etc.).
    features      : Covariate lists for models and evaluation.
    dashboard_regions : Default highlighted regions in the Streamlit UI.
    """

    project: ProjectConfig = field(default_factory=ProjectConfig)
    pipeline: PipelineControls = field(default_factory=PipelineControls)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    model_params: ModelParams = field(default_factory=ModelParams)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    dashboard_regions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> AETHELConfig:
    """
    Load and parse ``configs/default.yaml`` into a typed ``AETHELConfig``.

    Parameters
    ----------
    config_path:
        Override the config file location.  Defaults to
        ``configs/default.yaml`` resolved relative to the project root.

    Returns
    -------
    AETHELConfig
        Fully populated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config YAML file does not exist at the resolved path.
    """
    path = config_path or ConfigPaths.DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Ensure you are running scripts from the AETHEL project root "
            "or that configs/default.yaml has not been deleted."
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

    return AETHELConfig(
        project=ProjectConfig(
            name=p.get("name", "AETHEL"),
            version=p.get("version", "2.0.0"),
            description=p.get("description", ""),
        ),
        pipeline=PipelineControls(
            generate_environment=pipeline_raw.get("generate_environment", True),
            generate_clinical=pipeline_raw.get("generate_clinical", True),
            run_feature_engineering=pipeline_raw.get("run_feature_engineering", True),
            execute_survival_model=pipeline_raw.get("execute_survival_model", True),
        ),
        seeds=SeedConfig(
            python=seed_raw.get("python", 42),
            r=seed_raw.get("r", 123),
        ),
        study=StudyConfig(
            n_subjects=study_raw.get("n_subjects", 1000),
            observation_years=study_raw.get("observation_years", 5),
            total_cities=study_raw.get("total_cities", 100),
        ),
        model_params=ModelParams(
            rsf_ntree=model_raw.get("rsf_ntree", 500),
            rsf_importance=model_raw.get("rsf_importance", True),
        ),
        features=FeaturesConfig(
            survival_covariates=feat_raw.get("survival_covariates", []),
        ),
        dashboard_regions=raw.get("default_dashboard_regions", []),
    )
