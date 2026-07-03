"""
orchestrator.py
===============
Central experiment orchestrator. Unifies preprocessing, training, evaluation,
explainability, robustness, generalization, and reporting pipelines.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.config.config_manager import ConfigManager
from src.config.reproducibility import audit_environment
from src.config.validator import validate_pipeline, PipelineValidationError
from src.logging.structured_logger import configure_structured_logging, get_logger
from src.checkpointing.checkpoint_manager import CheckpointManager
from src.tracking.tracker import ExperimentTracker

logger = get_logger(__name__)

class ExperimentOrchestrator:
    """
    Orchestrates the lifecycle of AETHEL clinical machine learning experiments.
    """
    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root)
        self.config_manager = ConfigManager(self.project_root)
        self.experiments_base = self.project_root / "experiments"
        self.experiments_base.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        experiment_name: str,
        custom_overrides: dict[str, Any] | None = None,
        resume: bool = False,
        mode: str = "dev",
        skip_r: bool = False
    ) -> dict[str, Any]:
        """
        Runs a single end-to-end experiment pipeline:
        Preprocessing -> Evaluation -> Explainability -> Robustness -> Generalization -> Reporting.
        """
        # 1. Load config
        config = self.config_manager.get_config(experiment_name, custom_overrides)
        
        # Determine run ID and path
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        experiment_id = f"{experiment_name}_{timestamp}"
        exp_dir = self.experiments_base / experiment_id
        
        # If resume, check if there's an existing folder with the same name
        if resume:
            existing = sorted(self.experiments_base.glob(f"{experiment_name}_*"))
            if existing:
                exp_dir = existing[-1] # Resume the latest run folder
                experiment_id = exp_dir.name
                logger.info("Resuming experiment %s in directory: %s", experiment_name, exp_dir)
                
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Configure Logging
        configure_structured_logging(exp_dir / "logs")
        logger.info("=" * 80)
        logger.info("Starting AETHEL Experiment: %s (ID: %s)", experiment_name, experiment_id)
        logger.info("=" * 80)
        
        # 3. Pre-flight Validation
        try:
            logger.info("Running pre-flight pipeline validation...")
            validate_pipeline(self.project_root, config)
            logger.info("Pipeline validation passed.")
        except PipelineValidationError as e:
            logger.error("Pre-flight validation failed: %s", e)
            raise e
            
        # 4. Reproducibility Audit
        logger.info("Performing reproducibility audit...")
        audit = audit_environment(self.project_root, config)
        if audit["reproducibility_warnings"]:
            for warning in audit["reproducibility_warnings"]:
                logger.warning("Reproducibility warning: %s", warning)
        else:
            logger.info("Reproducibility audit passed without warnings.")
            
        # 5. Initialize Managers
        checkpoint_mgr = CheckpointManager(exp_dir)
        tracker = ExperimentTracker(exp_dir, experiment_name)
        
        # Save snapshot of config & environment audit
        checkpoint_mgr.save_metrics(audit, "../metadata/reproducibility_audit")
        self.config_manager.save_snapshot(exp_dir / "config" / "configuration_snapshot.yaml")
        
        # Log parameters to tracker
        tracker.log_parameter("experiment_id", experiment_id)
        tracker.log_parameter("mode", mode)
        tracker.log_parameter("python_seed", config.get("seeds", {}).get("python"))
        tracker.log_parameter("r_seed", config.get("seeds", {}).get("r"))
        
        # Clean outputs folder to prevent leakage from previous runs
        outputs_dir = self.project_root / "outputs"
        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Restore checkpoints if resume is requested
        if resume and checkpoint_mgr.has_checkpoint("preprocessing"):
            logger.info("Restoring intermediate split data from checkpoints...")
            (outputs_dir / "models").mkdir(parents=True, exist_ok=True)
            for split in ["train_split", "val_split", "test_split"]:
                cached_csv = exp_dir / "models" / f"{split}.csv"
                if cached_csv.exists():
                    shutil.copy(cached_csv, outputs_dir / "models" / f"{split}.csv")
                    
        # 6. Execute Stages
        try:
            # Stage A: Run data generation & feature preprocessing
            if not resume or not checkpoint_mgr.has_checkpoint("preprocessing"):
                logger.info("Executing Pipeline Stage A: Preprocessing and Data Generation...")
                cmd = [sys.executable, "-m", "scripts.run_pipeline"]
                if skip_r:
                    cmd.append("--skip-r")
                # Pass config path if it's not the default
                if experiment_name != "default":
                    cmd.extend(["--config", str(self.project_root / "configs" / "experiments" / f"{experiment_name}.yaml")])
                subprocess.run(cmd, cwd=str(self.project_root), check=True)
                
                # Checkpoint intermediate splits
                logger.info("Saving preprocessing checkpoints...")
                for split in ["train_split", "val_split", "test_split"]:
                    split_csv = outputs_dir / "evaluation" / experiment_name / f"{split}.csv"
                    if split_csv.exists():
                        shutil.copy(split_csv, exp_dir / "models" / f"{split}.csv")
            else:
                logger.info("Resuming: Preprocessing step skipped (checkpoints loaded).")
                
            # Stage B: Model Evaluation & Metrics
            if not resume or not checkpoint_mgr.has_checkpoint("metrics"):
                logger.info("Executing Pipeline Stage B: Model Evaluation and Calibration...")
                cmd = [sys.executable, "-m", "scripts.run_evaluation", "--experiment", experiment_name]
                subprocess.run(cmd, cwd=str(self.project_root), check=True)
            else:
                logger.info("Resuming: Model Evaluation step skipped.")
                
            # Stage C: Explainability Framework
            logger.info("Executing Pipeline Stage C: Explainability Audit...")
            cmd = [sys.executable, "-m", "scripts.run_explainability", "--mode", mode, "--experiment", experiment_name]
            subprocess.run(cmd, cwd=str(self.project_root), check=True)
            
            # Stage D: Robustness Sweeps
            logger.info("Executing Pipeline Stage D: Robustness & Epistemic Uncertainty Audits...")
            cmd = [sys.executable, "-m", "scripts.run_robustness", "--mode", mode, "--experiment", experiment_name]
            subprocess.run(cmd, cwd=str(self.project_root), check=True)
            
            # Stage E: Generalization transfer Checks
            logger.info("Executing Pipeline Stage E: Cross-Cohort Generalization Validation...")
            cmd = [sys.executable, "-m", "scripts.run_generalization", "--mode", mode]
            subprocess.run(cmd, cwd=str(self.project_root), check=True)
            
            # Stage F: Scientific Visualization & Reporting
            logger.info("Executing Pipeline Stage F: Publication Reports compilation...")
            # We output to the local experiment's report dir
            report_out = exp_dir / "reports"
            cmd = [sys.executable, "-m", "scripts.run_publication_report", "--experiment", experiment_name, "--output-dir", str(exp_dir)]
            subprocess.run(cmd, cwd=str(self.project_root), check=True)
            
            # 7. Harvest outputs from outputs/ to experiments/<run_id>/
            logger.info("Harvesting final output files into experiment structure...")
            self.harvest_artifacts(outputs_dir, exp_dir)
            
            manifest = tracker.finish(status="COMPLETED")
            logger.info("Experiment run complete. manifest.json generated at metadata/.")
            logger.info("=" * 80)
            return manifest
            
        except Exception as e:
            logger.exception("Experiment execution failed: %s", str(e))
            tracker.finish(status="FAILED")
            raise e

    def harvest_artifacts(self, outputs_dir: Path, exp_dir: Path) -> None:
        """
        Copies final artifacts from the temporary outputs/ directory to the structured
        subfolders of experiments/<run_id>/.
        """
        # Models
        if (outputs_dir / "models").exists():
            for f in (outputs_dir / "models").glob("*"):
                shutil.copy(f, exp_dir / "models" / f.name)
                
        # Metrics & Evaluations
        if (outputs_dir / "evaluation").exists():
            shutil.copytree(outputs_dir / "evaluation", exp_dir / "metrics" / "evaluation", dirs_exist_ok=True)
        if (outputs_dir / "metrics").exists():
            shutil.copytree(outputs_dir / "metrics", exp_dir / "metrics", dirs_exist_ok=True)
            
        # Figures
        if (outputs_dir / "figures").exists():
            shutil.copytree(outputs_dir / "figures", exp_dir / "figures", dirs_exist_ok=True)
        if (outputs_dir / "explainability").exists():
            shutil.copytree(outputs_dir / "explainability", exp_dir / "figures" / "explainability", dirs_exist_ok=True)
        if (outputs_dir / "robustness").exists():
            shutil.copytree(outputs_dir / "robustness", exp_dir / "figures" / "robustness", dirs_exist_ok=True)
            
        # Generalization outputs
        if (outputs_dir / "generalization").exists():
            shutil.copytree(outputs_dir / "generalization", exp_dir / "metrics" / "generalization", dirs_exist_ok=True)
            
        # Clean up outputs directory
        try:
            shutil.rmtree(outputs_dir)
            outputs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def run_batch(
        self,
        experiment_names: list[str],
        custom_overrides: dict[str, Any] | None = None,
        mode: str = "dev",
        num_workers: int = 1,
        skip_r: bool = False
    ) -> dict[str, dict[str, Any]]:
        """
        Runs multiple experiments in batch mode.
        If num_workers > 1, runs them in parallel using a ThreadPoolExecutor.
        """
        results = {}
        if num_workers > 1:
            logger.info("Executing batch of %d experiments with %d parallel workers...", len(experiment_names), num_workers)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self.run_single, name, custom_overrides, False, mode, skip_r): name
                    for name in experiment_names
                }
                for fut in futures:
                    name = futures[fut]
                    try:
                        results[name] = fut.result()
                    except Exception as e:
                        logger.error("Batch experiment '%s' failed: %s", name, e)
                        results[name] = {"status": "FAILED", "error": str(e)}
        else:
            logger.info("Executing batch of %d experiments sequentially...", len(experiment_names))
            for name in experiment_names:
                try:
                    results[name] = self.run_single(name, custom_overrides, False, mode, skip_r)
                except Exception as e:
                    logger.error("Batch experiment '%s' failed: %s", name, e)
                    results[name] = {"status": "FAILED", "error": str(e)}
                    
        return results
