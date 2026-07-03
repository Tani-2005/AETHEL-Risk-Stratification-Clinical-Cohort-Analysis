"""
run_publication_report.py
=========================
Entry point command for generating all publication figures, tables, and reports
for a completed AETHEL experiment.

Usage:
------
    python scripts/run_publication_report.py --experiment exp_mode1_synthetic
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_setup import configure_logging, get_logger
from src.reporting.report_generator import run_reporting_pipeline

logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AETHEL Scientific Visualization & Reporting Automator"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp_mode1_synthetic",
        help="Name of the completed experiment (default: exp_mode1_synthetic)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory. Defaults to results/<experiment_name>_<timestamp>/"
    )
    args = parser.parse_args()
    
    configure_logging()
    
    # Establish canonical results folder structure
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        out_base = PROJECT_ROOT / "results" / f"{args.experiment}_{timestamp}"
        
    logger.info("=" * 70)
    logger.info("AETHEL Scientific Visualization & Reporting Framework")
    logger.info("=" * 70)
    logger.info("Experiment : %s", args.experiment)
    logger.info("Output Dir : %s", out_base)
    logger.info("-" * 70)
    
    try:
        results = run_reporting_pipeline(args.experiment, out_base)
        
        logger.info("=" * 70)
        logger.info("REPORT GENERATION SUCCESSFUL")
        logger.info("=" * 70)
        logger.info("PDF Report      : %s", results["reports"]["pdf"])
        logger.info("HTML Report     : %s", results["reports"]["html"])
        logger.info("Markdown Report : %s", results["reports"]["markdown"])
        logger.info("All tables saved in: %s", out_base / "tables")
        logger.info("All figures saved in: %s", out_base / "figures")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.exception("Failed to generate publication reports: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
