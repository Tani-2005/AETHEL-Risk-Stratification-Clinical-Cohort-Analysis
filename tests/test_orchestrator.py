"""
test_orchestrator.py
====================
Unit tests for the AETHEL Academic Orchestration & Reproducibility framework.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config.config_manager import ConfigManager
from src.config.reproducibility import audit_environment, calculate_sha256
from src.config.validator import validate_pipeline, PipelineValidationError
from src.checkpointing.checkpoint_manager import CheckpointManager
from src.tracking.tracker import ExperimentTracker
from src.orchestrator.orchestrator import ExperimentOrchestrator

@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """Fixture to set up a temporary workspace with dummy configs and data."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    
    # Create default.yaml
    default_yaml = configs_dir / "default.yaml"
    default_yaml.write_text("""
project:
  name: "TestProject"
  version: "1.0.0"
seeds:
  python: 42
  r: 123
pipeline_controls:
  generate_environment: false
  generate_clinical: false
  run_feature_engineering: false
  run_validation: false
  run_feature_engineering_derived: false
  run_splitting: false
  run_preprocessing_pipeline: false
  run_feature_selection: false
  execute_survival_model: false
output_paths:
  models: "outputs/models"
  figures: "outputs/figures"
""", encoding="utf-8")
    
    # Create datasets subfolders and configs
    datasets_dir = configs_dir / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "synthetic.yaml").write_text("""
datasets:
  synthetic:
    name: "Synthetic"
    path: "data/analytical_cohort.csv"
    enabled: true
""", encoding="utf-8")
    
    # Create dummy dataset CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(data_dir / "analytical_cohort.csv", index=False)
    
    # Create experiment folder
    exp_dir = configs_dir / "experiments"
    exp_dir.mkdir()
    (exp_dir / "test_exp.yaml").write_text("""
name: "test_exp"
seeds:
  python: 99
""", encoding="utf-8")
    
    return tmp_path


def test_config_manager(temp_workspace):
    mgr = ConfigManager(temp_workspace)
    
    # Check default load
    cfg = mgr.get_config()
    assert cfg["project"]["name"] == "TestProject"
    assert cfg["seeds"]["python"] == 42
    assert "synthetic" in cfg["datasets"]
    
    # Check experiment override
    cfg_exp = mgr.get_config("test_exp")
    assert cfg_exp["seeds"]["python"] == 99


def test_pipeline_validation(temp_workspace):
    mgr = ConfigManager(temp_workspace)
    cfg = mgr.get_config("test_exp")
    
    # Valid config should pass
    validate_pipeline(temp_workspace, cfg)
    
    # Modify config to fail (invalid path)
    cfg["datasets"]["synthetic"]["path"] = "data/missing.csv"
    with pytest.raises(PipelineValidationError) as excinfo:
        validate_pipeline(temp_workspace, cfg)
    assert "missing.csv" in str(excinfo.value)


def test_reproducibility_audit(temp_workspace):
    mgr = ConfigManager(temp_workspace)
    cfg = mgr.get_config("test_exp")
    
    audit = audit_environment(temp_workspace, cfg)
    assert "environment" in audit
    assert "package_versions" in audit
    assert "configured_seeds" in audit
    assert audit["configured_seeds"]["python"] == 99


def test_checkpoint_manager(temp_workspace):
    exp_run_dir = temp_workspace / "experiments" / "test_run"
    mgr = CheckpointManager(exp_run_dir)
    
    # Model checkpointing
    clf = LogisticRegression(C=0.5)
    model_path = mgr.save_model(clf, "logistic_regression")
    assert model_path.exists()
    
    loaded_clf = mgr.load_model("logistic_regression")
    assert loaded_clf is not None
    assert loaded_clf.C == 0.5
    
    # Metric checkpointing
    metrics = {"accuracy": 0.95, "f1": 0.94}
    metrics_path = mgr.save_metrics(metrics, "validation_metrics")
    assert metrics_path.exists()
    
    loaded_metrics = mgr.load_metrics("validation_metrics")
    assert loaded_metrics["accuracy"] == 0.95


def test_tracker(temp_workspace):
    exp_run_dir = temp_workspace / "experiments" / "test_run"
    tracker = ExperimentTracker(exp_run_dir, "test_exp")
    tracker.log_parameter("learning_rate", 0.01)
    tracker.log_metric("auc", 0.89)
    
    manifest = tracker.finish()
    assert manifest["status"] == "COMPLETED"
    assert manifest["parameters"]["learning_rate"] == 0.01
    assert manifest["metrics"]["auc"] == 0.89
    assert (exp_run_dir / "metadata" / "manifest.json").exists()
