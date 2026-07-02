"""
run_pipeline.py
===============
AETHEL Pipeline Orchestrator — Single entry point for the full analysis pipeline.

Replaces the five-step manual execution sequence described in the original README
with a single configurable command.  Each stage is toggled via
``configs/default.yaml → pipeline_controls``.

Pipeline stages
---------------
1. build_eu_registry      — Generate 100-city EU metadata registry
2. generate_env_data      — Generate 60-month environmental time-series
3. generate_clinical      — Generate synthetic clinical cohort (R)
4. preprocess_features    — Engineer patient-level analytical cohort
5. execute_survival_model — Fit Cox PH + Random Survival Forest (R)

Usage
-----
    # Run the full pipeline (default config):
    python scripts/run_pipeline.py

    # Run with an alternative config (e.g. a NHANES experiment):
    python scripts/run_pipeline.py --config configs/nhanes_experiment.yaml

    # Run Python-only stages (skip R stages):
    python scripts/run_pipeline.py --skip-r

Notes
-----
- R scripts are invoked via subprocess using ``Rscript``.
  Ensure R is installed and available on PATH.
- All outputs are written to ``outputs/`` automatically.
- Run logs are written to ``outputs/logs/aethel.log``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, AETHELConfig
from src.utils.logging_setup import configure_logging, get_logger
from src.utils.paths import ensure_output_dirs, ConfigPaths

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# R stage runner
# ---------------------------------------------------------------------------

def run_r_script(script_path: Path, cfg: AETHELConfig) -> bool:  # noqa: ARG001
    """
    Execute an R script via Rscript subprocess.

    Parameters
    ----------
    script_path:
        Absolute path to the ``.R`` file.
    cfg:
        Pipeline config (reserved for future argument passing).

    Returns
    -------
    bool
        True if the R script exited with code 0, False otherwise.
    """
    logger.info("Running R script: %s", script_path.relative_to(PROJECT_ROOT))
    result = subprocess.run(
        ["Rscript", str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # let R output stream to console
        text=True,
    )
    if result.returncode != 0:
        logger.error("R script failed with exit code %d: %s", result.returncode, script_path)
        return False
    return True


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_build_registry(cfg: AETHELConfig) -> None:
    """Stage 1: Build Pan-European city registry."""
    from src.preprocessing.build_eu_registry import build_registry
    build_registry()


def stage_generate_env_data(cfg: AETHELConfig) -> None:
    """Stage 2: Generate environmental time-series."""
    from src.preprocessing.generate_env_data import generate_pan_european_data
    generate_pan_european_data()


def stage_generate_clinical(cfg: AETHELConfig, skip_r: bool) -> None:
    """Stage 3: Generate synthetic clinical cohort (R)."""
    if skip_r:
        logger.warning("Skipping R stage: generate_clinical_cohort.R (--skip-r flag set)")
        return
    r_script = PROJECT_ROOT / "src" / "preprocessing" / "generate_clinical_cohort.R"
    success = run_r_script(r_script, cfg)
    if not success:
        raise RuntimeError("Clinical cohort generation (R) failed. See logs above.")


def stage_preprocess_features(cfg: AETHELConfig) -> None:
    """Stage 4a: Join clinical + environmental data → analytical_cohort.csv."""
    from src.feature_engineering.preprocess_features import build_analytical_dataset
    build_analytical_dataset()


def stage_full_preprocessing(cfg: AETHELConfig) -> None:
    """Stage 4b: Validate → engineer → split → scale → select features."""
    from src.feature_engineering.preprocess_features import run_full_preprocessing
    run_full_preprocessing()


def stage_survival_model(cfg: AETHELConfig, skip_r: bool) -> None:
    """Stage 5: Fit Cox PH + RSF survival models (R)."""
    if skip_r:
        logger.warning("Skipping R stage: survival_model.R (--skip-r flag set)")
        return
    r_script = PROJECT_ROOT / "src" / "models" / "survival_model.R"
    success = run_r_script(r_script, cfg)
    if not success:
        raise RuntimeError("Survival modelling (R) failed. See logs above.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(config_path: Path | None = None, skip_r: bool = False) -> None:
    """
    Execute the full AETHEL pipeline sequentially.

    Parameters
    ----------
    config_path:
        Path to a YAML config file.  Defaults to ``configs/default.yaml``.
    skip_r:
        If True, R-dependent stages are skipped (useful when R is unavailable).
    """
    cfg = load_config(config_path)

    logger.info("=" * 60)
    logger.info("AETHEL Pipeline — %s v%s", cfg.project.name, cfg.project.version)
    logger.info("=" * 60)
    logger.info("Python seed : %d", cfg.seeds.python)
    logger.info("R seed      : %d", cfg.seeds.r)
    logger.info("Subjects    : %d", cfg.study.n_subjects)
    logger.info("Cities      : %d", cfg.study.total_cities)
    logger.info("-" * 60)

    pipeline_start = time.time()
    stages_run = 0

    if cfg.pipeline.generate_environment:
        logger.info("[Stage 1/6] Building EU city registry...")
        stage_build_registry(cfg)
        logger.info("[Stage 2/6] Generating environmental time-series...")
        stage_generate_env_data(cfg)
        stages_run += 2

    if cfg.pipeline.generate_clinical:
        logger.info("[Stage 3/6] Generating clinical cohort (R)...")
        stage_generate_clinical(cfg, skip_r)
        stages_run += 1

    if cfg.pipeline.run_feature_engineering:
        logger.info("[Stage 4a/6] Building analytical cohort (join clinical + env)...")
        stage_preprocess_features(cfg)
        stages_run += 1

    # Stages 4b: validate, engineer, split, scale, select — Python only
    if any([
        cfg.pipeline.run_validation,
        cfg.pipeline.run_feature_engineering_derived,
        cfg.pipeline.run_splitting,
        cfg.pipeline.run_preprocessing_pipeline,
        cfg.pipeline.run_feature_selection,
    ]):
        logger.info("[Stage 4b/6] Full preprocessing pipeline (validate->engineer->split->scale->select)...")
        stage_full_preprocessing(cfg)
        stages_run += 1

    if cfg.pipeline.execute_survival_model:
        logger.info("[Stage 5/6] Executing survival models on TRAIN split (R)...")
        stage_survival_model(cfg, skip_r)
        stages_run += 1

    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("Pipeline complete. %d stages executed in %.1fs.", stages_run, elapsed)
    logger.info("Outputs written to: outputs/")
    logger.info("Launch dashboard : streamlit run src/visualization/dashboard.py")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="AETHEL full pipeline orchestrator",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip R-dependent stages (generate_clinical, survival_model)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    configure_logging()
    ensure_output_dirs()
    run_pipeline(config_path=args.config, skip_r=args.skip_r)
