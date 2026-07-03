"""
reproducibility.py
==================
Performs reproducibility audits by capturing library versions, hardware/OS metadata,
calculating dataset and config checksums, verifying random seeds, and tracking Git metadata.
"""
from __future__ import annotations
import hashlib
import importlib
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Any

def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA-256 checksum of a file."""
    if not file_path.exists() or not file_path.is_file():
        return "missing"
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_git_metadata(project_root: Path) -> dict[str, str]:
    """Retrieves Git commit hash and dirty status."""
    meta = {"commit": "unknown", "dirty": "unknown"}
    try:
        # Commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        meta["commit"] = commit
        
        # Dirty status
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        meta["dirty"] = "true" if status else "false"
    except Exception:
        pass
    return meta


def audit_environment(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """
    Performs the full reproducibility audit.
    """
    # 1. Platform details
    env_info = {
        "python_version": sys.version.split()[0],
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
    }
    
    # 2. Package versions
    packages = ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "reportlab", "tabulate", "yaml"]
    package_versions = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            package_versions[pkg] = version
        except ImportError:
            package_versions[pkg] = "not_installed"
            
    # 3. Checksums for datasets
    dataset_checksums = {}
    data_dir = project_root / "data"
    for csv_file in data_dir.glob("**/*.csv"):
        rel_path = csv_file.relative_to(project_root)
        dataset_checksums[str(rel_path)] = calculate_sha256(csv_file)
        
    # 4. Checksums for config files
    config_checksums = {}
    configs_dir = project_root / "configs"
    for yaml_file in configs_dir.glob("**/*.yaml"):
        rel_path = yaml_file.relative_to(project_root)
        config_checksums[str(rel_path)] = calculate_sha256(yaml_file)
        
    # 5. Git Metadata
    git_meta = get_git_metadata(project_root)
    
    # 6. Verify Seeds
    seeds = config.get("seeds", {})
    python_seed = seeds.get("python")
    r_seed = seeds.get("r")
    
    warnings = []
    if python_seed is None:
        warnings.append("Python random seed is not configured.")
    if r_seed is None:
        warnings.append("R random seed is not configured.")
    if git_meta["dirty"] == "true":
        warnings.append("Git working tree has uncommitted modifications; build is dirty.")
        
    audit_report = {
        "timestamp": platform.node(),
        "environment": env_info,
        "package_versions": package_versions,
        "git": git_meta,
        "dataset_checksums": dataset_checksums,
        "config_checksums": config_checksums,
        "configured_seeds": {
            "python": python_seed,
            "r": r_seed
        },
        "reproducibility_warnings": warnings,
        "reproducible": len(warnings) == 0
    }
    
    return audit_report
