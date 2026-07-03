"""
checkpoint_manager.py
=====================
Handles experiment checkpointing. Saves and loads trained models (pickle),
intermediate datasets (CSV), predictions, and evaluation metrics.
Allows resuming runs by skipping completed phases.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any
import pandas as pd

class CheckpointManager:
    """
    Manages caching and serialization of pipeline artifacts.
    """
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.models_dir = experiment_dir / "models"
        self.metrics_dir = experiment_dir / "metrics"
        self.config_dir = experiment_dir / "config"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, name: str) -> Path:
        """Saves a fitted model to disk as a pickle file."""
        model_path = self.models_dir / f"{name.replace(' ', '_').lower()}.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(model, fh)
        return model_path

    def load_model(self, name: str) -> Any | None:
        """Loads a model from disk if it exists."""
        model_path = self.models_dir / f"{name.replace(' ', '_').lower()}.pkl"
        if not model_path.exists():
            return None
        with model_path.open("rb") as fh:
            return pickle.load(fh)

    def save_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        """Saves intermediate DataFrame (like split data) to disk."""
        data_path = self.experiment_dir / "models" / f"{name}.csv"
        df.to_csv(data_path, index=False)
        return data_path

    def load_dataframe(self, name: str) -> pd.DataFrame | None:
        """Loads intermediate DataFrame if it exists."""
        data_path = self.experiment_dir / "models" / f"{name}.csv"
        if not data_path.exists():
            return None
        return pd.read_csv(data_path)

    def save_metrics(self, metrics: dict[str, Any], name: str) -> Path:
        """Saves calculated metrics to a JSON file."""
        import json
        metrics_path = self.metrics_dir / f"{name}.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        return metrics_path

    def load_metrics(self, name: str) -> dict[str, Any] | None:
        """Loads metrics JSON file if it exists."""
        import json
        metrics_path = self.metrics_dir / f"{name}.json"
        if not metrics_path.exists():
            return None
        with metrics_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def has_checkpoint(self, step_name: str) -> bool:
        """
        Checks if a specific pipeline step is already completed.
        Steps:
          - 'preprocessing': Checks if train_split.csv and val_split.csv exist.
          - 'classifiers': Checks if fitted models are saved.
          - 'metrics': Checks if final metrics.json is saved.
        """
        if step_name == "preprocessing":
            return (
                (self.models_dir / "train_split.csv").exists() and
                (self.models_dir / "val_split.csv").exists()
            )
        elif step_name == "metrics":
            return (self.metrics_dir / "overall_metrics.json").exists()
        return False
