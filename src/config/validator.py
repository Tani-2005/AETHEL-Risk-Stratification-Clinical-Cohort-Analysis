"""
validator.py
============
Pre-flight validator for the AETHEL pipeline. Validates configuration keys, dataset presence,
directory writeability, model availability, and external dependencies (like R).
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any

class PipelineValidationError(Exception):
    """Exception raised when pre-flight pipeline validation fails."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Pipeline validation failed:\n" + "\n".join(f" - {err}" for err in errors))


def validate_pipeline(project_root: Path, config: dict[str, Any]) -> None:
    """
    Validates configuration settings, checks dependencies, verifies data paths,
    and returns list of errors or raises PipelineValidationError if errors found.
    """
    errors = []
    
    # 1. Check basic config structure
    required_sections = ["project", "seeds"]
    for sec in required_sections:
        if sec not in config:
            errors.append(f"Missing required configuration section: '{sec}'")
            
    # 2. Check dataset availability
    datasets = config.get("datasets", {})
    if not datasets:
        errors.append("No datasets configured in 'datasets' section.")
    else:
        for name, details in datasets.items():
            if not details.get("enabled", True):
                continue
            path_str = details.get("path")
            if not path_str:
                errors.append(f"Dataset '{name}' does not have a configured 'path'.")
                continue
            d_path = project_root / path_str
            if not d_path.exists():
                errors.append(f"Dataset file not found for '{name}' at: {d_path.relative_to(project_root) if project_root in d_path.parents else d_path}")
                
    # 3. Check model compatibility
    supported_models = {
        "Logistic Regression", "Decision Tree", "Random Forest",
        "XGBoost", "LightGBM", "Cox PH", "Random Survival Forest"
    }
    
    # If there is a models list or model params check
    models_cfg = config.get("models", {})
    for m_name in models_cfg:
        # Normalize key names or check supported list
        # We can map yaml model keys to human readable model names
        normalized = m_name.replace("_", " ").title()
        # Handle exceptions like "Xgboost" or "Lightgbm"
        if normalized == "Xgboost":
            normalized = "XGBoost"
        elif normalized == "Lightgbm":
            normalized = "LightGBM"
        elif normalized == "Cox Ph":
            normalized = "Cox PH"
            
        if normalized not in supported_models:
            errors.append(f"Model '{m_name}' (parsed as '{normalized}') is not supported. Supported: {supported_models}")
            
    # 4. Check folder structure writeability
    out_paths = config.get("output_paths", {})
    for key, path_str in out_paths.items():
        if not path_str:
            continue
        try:
            p = project_root / path_str
            p.mkdir(parents=True, exist_ok=True)
            # Try to write a temp file to ensure it's writeable
            temp_file = p / ".write_test"
            temp_file.touch()
            temp_file.unlink()
        except Exception as e:
            errors.append(f"Output path for '{key}' is not writeable: {path_str} (error: {e})")

    # 5. Check Rscript if pipeline controls execute survival model or generate clinical cohort
    pipeline_controls = config.get("pipeline_controls", {})
    needs_r = pipeline_controls.get("generate_clinical", False) or pipeline_controls.get("execute_survival_model", False)
    
    if needs_r:
        rscript_path = shutil.which("Rscript")
        if not rscript_path:
            # Fallback check on standard Windows paths
            import glob
            standard_paths = glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe")
            if not standard_paths:
                errors.append("R is required by the pipeline controls, but 'Rscript' was not found on PATH or in standard C:/Program Files/R/ folders.")
                
    if errors:
        raise PipelineValidationError(errors)
