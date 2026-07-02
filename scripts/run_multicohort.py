"""
run_multicohort.py
==================
Orchestrator for the AETHEL multi-cohort framework.

Usage:
    python scripts/run_multicohort.py           # full run
    python scripts/run_multicohort.py --audit   # dataset audit only
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_setup import configure_logging, get_logger
from src.utils.paths import ensure_output_dirs

logger = get_logger(__name__)


def run_audit():
    from src.datasets.registry import DatasetRegistry
    registry = DatasetRegistry.from_config()
    audit_df = registry.audit()
    print("\n=== AETHEL DATASET AUDIT ===")
    print(audit_df.to_string(index=False))
    return audit_df


def run_harmonization():
    from src.datasets.registry import DatasetRegistry
    from src.datasets.harmonizer import HarmonizationLayer
    registry = DatasetRegistry.from_config()
    datasets = registry.load_all()
    layer = HarmonizationLayer()
    layer.run(datasets)
    return datasets


def run_cohort_comparison(datasets):
    from src.evaluation.cohort_comparison import CohortComparison
    comparison = CohortComparison()
    comparison.run(datasets)


def main():
    configure_logging()
    ensure_output_dirs()

    parser = argparse.ArgumentParser(description="AETHEL Multi-Cohort Framework")
    parser.add_argument("--audit", action="store_true", help="Dataset audit only")
    args = parser.parse_args()

    if args.audit:
        run_audit()
        return

    logger.info("=== AETHEL Multi-Cohort Framework ===")
    logger.info("Step 1: Dataset audit")
    run_audit()

    logger.info("Step 2: Harmonization + report generation")
    datasets = run_harmonization()

    logger.info("Step 3: Cohort comparison + domain shift analysis")
    run_cohort_comparison(datasets)

    logger.info("Step 4: Experiment runner (Modes 1, 2, 4, 7)")
    from src.experiments.experiment_config import ExperimentConfig
    from src.experiments.experiment_runner import ExperimentRunner
    runner = ExperimentRunner()
    config_dir = PROJECT_ROOT / "configs" / "experiments"
    for yaml_file in sorted(config_dir.glob("exp_mode*.yaml")):
        cfg = ExperimentConfig.from_yaml(yaml_file)
        if cfg.domain_shift_only:
            logger.info("Skipping domain-shift-only experiment: %s", cfg.name)
            continue
        try:
            result = runner.run(cfg)
            logger.info(
                "Experiment '%s' complete — train=%d, val=%d",
                result.experiment_name, len(result.train_df), len(result.val_df)
            )
        except Exception as exc:
            logger.error("Experiment '%s' failed: %s", cfg.name, exc)

    logger.info("Multi-cohort framework run complete. Check outputs/reports/ for all reports.")


if __name__ == "__main__":
    main()
