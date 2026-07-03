"""
cli_entry.py
============
CLI interface for the AETHEL Academic Orchestration & Reproducibility Framework.
Defines parameters for experiment runs, customized dataset/model overrides,
batch executions, parallel workers, and resuming runs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from src.orchestrator.orchestrator import ExperimentOrchestrator

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AETHEL Academic Experiment Orchestrator & Reproducibility Center"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp_mode1_synthetic",
        help="Name of the experiment configuration to run (e.g. exp_mode1_synthetic)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available experiment configurations in batch mode"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override/limit train dataset list (comma-separated, e.g. synthetic,framingham)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override/limit model name to run (e.g. xgboost)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last interrupted run for the specified experiment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "paper"],
        help="Experiment execution mode: dev (fast) or paper (full validation suite)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers for parallel batch execution"
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip R-dependent pipeline stages"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    orchestrator = ExperimentOrchestrator(project_root)
    
    # Compile overrides
    overrides = {}
    if args.dataset:
        overrides["train_datasets"] = [d.strip() for d in args.dataset.split(",")]
    if args.model:
        # If user specifies a single model override, we configure that in the run metadata
        overrides["models_limit"] = args.model
        
    try:
        if args.all:
            # Gather all configurations from configs/experiments/
            exp_dir = project_root / "configs" / "experiments"
            all_configs = [f.stem for f in exp_dir.glob("*.yaml") if f.is_file() and not f.name.startswith(".")]
            if not all_configs:
                print("Error: No experiment configurations found under configs/experiments/")
                sys.exit(1)
            print(f"Starting batch execution of {len(all_configs)} experiments: {all_configs}")
            orchestrator.run_batch(all_configs, overrides, args.mode, args.workers, args.skip_r)
        else:
            print(f"Starting execution of experiment: {args.experiment}")
            orchestrator.run_single(args.experiment, overrides, args.resume, args.mode, args.skip_r)
            
    except Exception as e:
        print(f"Execution aborted with error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
