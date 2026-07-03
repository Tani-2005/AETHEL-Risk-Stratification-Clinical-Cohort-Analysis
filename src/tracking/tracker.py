"""
tracker.py
==========
Lightweight experiment tracker. Records parameters, metrics, runtime, memory usage,
and environment metadata. Optionally logs to MLflow if available.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Optional psutil import
try:
    import psutil
except ImportError:
    psutil = None

# Optional mlflow import
try:
    import mlflow
except ImportError:
    mlflow = None


class ExperimentTracker:
    """
    Lightweight tracker to collect metrics, parameters, and environmental metadata.
    """
    def __init__(self, experiment_dir: Path, experiment_name: str):
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.metadata_dir = experiment_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.parameters: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        
        self.use_mlflow = (mlflow is not None)
        if self.use_mlflow:
            try:
                mlflow.set_experiment(experiment_name)
                mlflow.start_run(run_name=experiment_name)
            except Exception as e:
                print(f"Warning: MLflow start_run failed: {e}. Falling back to internal tracker.")
                self.use_mlflow = False

    def log_parameter(self, key: str, value: Any) -> None:
        """Logs a parameter."""
        self.parameters[key] = value
        if self.use_mlflow:
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_parameters(self, params: dict[str, Any]) -> None:
        """Logs multiple parameters."""
        self.parameters.update(params)
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception:
                pass

    def log_metric(self, key: str, value: float) -> None:
        """Logs a numeric metric."""
        self.metrics[key] = value
        if self.use_mlflow:
            try:
                mlflow.log_metric(key, value)
            except Exception:
                pass

    def get_resource_usage(self) -> dict[str, Any]:
        """Captures hardware and memory resource usage snapshots."""
        usage = {
            "ram_memory_mb": 0.0,
            "cpu_count": os.cpu_count() or 0
        }
        if psutil:
            try:
                process = psutil.Process(os.getpid())
                usage["ram_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return usage

    def finish(self, status: str = "COMPLETED") -> dict[str, Any]:
        """
        Ends tracking, compiles metadata, and outputs the final manifest.json.
        """
        elapsed_seconds = time.time() - self.start_time
        resource_usage = self.get_resource_usage()
        
        if self.use_mlflow:
            try:
                mlflow.end_run(status=status)
            except Exception:
                pass
                
        manifest = {
            "experiment_name": self.experiment_name,
            "status": status,
            "runtime_seconds": round(elapsed_seconds, 2),
            "resource_usage": resource_usage,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            }
        }
        
        manifest_path = self.metadata_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            
        return manifest
