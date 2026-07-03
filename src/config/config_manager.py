"""
config_manager.py
=================
Hierarchical configuration manager. Merges base settings, sub-configurations
(datasets, models, evaluation), and specific experiment parameter overrides.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import yaml

class ConfigManager:
    """
    Manages loading, merging, and snapshotting of hierarchical YAML configurations.
    """
    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / "configs"
        self.default_path = self.configs_dir / "default.yaml"
        self.merged_config: dict[str, Any] = {}

    @staticmethod
    def deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merges dict2 into dict1.
        """
        merged = copy.deepcopy(dict1)
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def load_subconfigs(self) -> dict[str, Any]:
        """
        Recursively loads all YAML files under datasets/, models/, evaluation/
        and returns a nested dictionary configuration structure.
        """
        sub_cfg: dict[str, Any] = {}
        sub_dirs = ["datasets", "models", "evaluation"]
        
        for s_dir in sub_dirs:
            target_path = self.configs_dir / s_dir
            if not target_path.exists():
                continue
            
            sub_cfg[s_dir] = {}
            for file_path in target_path.glob("**/*.yaml"):
                try:
                    with file_path.open("r", encoding="utf-8") as fh:
                        content = yaml.safe_load(fh) or {}
                    # If file has top-level key matching the category, merge it, otherwise nest it
                    # Example: datasets/synthetic.yaml might have top-level "datasets" key
                    if s_dir in content:
                        sub_cfg[s_dir] = self.deep_merge(sub_cfg[s_dir], content[s_dir])
                    else:
                        stem = file_path.stem
                        sub_cfg[s_dir][stem] = content
                except Exception as e:
                    # Log or print warning
                    print(f"Warning: Failed to load sub-config {file_path}: {e}")
                    
        return sub_cfg

    def load_experiment_config(self, experiment_name: str) -> dict[str, Any]:
        """
        Loads the specific experiment YAML configuration.
        """
        # Checks configs/experiments/<name>.yaml or configs/experiments/<name>
        candidates = [
            self.configs_dir / "experiments" / f"{experiment_name}.yaml",
            self.configs_dir / "experiments" / experiment_name,
            Path(experiment_name)
        ]
        
        for cand in candidates:
            if cand.exists() and cand.is_file():
                try:
                    with cand.open("r", encoding="utf-8") as fh:
                        return yaml.safe_load(fh) or {}
                except Exception as e:
                    print(f"Warning: Failed to load experiment config {cand}: {e}")
        return {}

    def get_config(
        self,
        experiment_name: str | None = None,
        custom_overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Loads and merges:
          1. configs/default.yaml
          2. Subconfigs (datasets/, models/, evaluation/)
          3. Experiment YAML overrides
          4. Custom dict overrides (e.g. from command line parameters)
        """
        # 1. Base default
        if self.default_path.exists():
            with self.default_path.open("r", encoding="utf-8") as fh:
                base = yaml.safe_load(fh) or {}
        else:
            base = {}
            
        # 2. Subconfigs
        subconfigs = self.load_subconfigs()
        merged = self.deep_merge(base, subconfigs)
        
        # 3. Experiment overrides
        if experiment_name:
            exp_overrides = self.load_experiment_config(experiment_name)
            merged = self.deep_merge(merged, exp_overrides)
            # Retain experiment name
            if "name" not in merged:
                merged["name"] = experiment_name
        
        # 4. Custom overrides
        if custom_overrides:
            merged = self.deep_merge(merged, custom_overrides)
            
        self.merged_config = merged
        return merged

    def save_snapshot(self, output_path: Path) -> None:
        """
        Saves the current merged configuration dictionary to a target path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.merged_config, fh, default_flow_style=False)
