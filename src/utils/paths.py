"""
paths.py
========
Canonical path registry for the AETHEL project.

All file I/O across Python modules must import paths from here rather than
constructing strings inline.  Using ``pathlib.Path`` throughout ensures:

- OS-agnostic separators (Windows / Linux / macOS)
- Paths are always resolved relative to the repository root
- A single place to update if the directory layout changes

Usage
-----
    from src.utils.paths import DataPaths, OutputPaths

    df = pd.read_csv(DataPaths.RAW_REGISTRY)
    fig.savefig(OutputPaths.FIGURES / "km_curve.png")
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — everything is relative to this
# ---------------------------------------------------------------------------

# __file__ is  src/utils/paths.py  →  .parent.parent.parent = project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------

class Dirs:
    """Top-level project directories."""

    CONFIGS: Path = PROJECT_ROOT / "configs"
    DATA: Path = PROJECT_ROOT / "data"
    SRC: Path = PROJECT_ROOT / "src"
    OUTPUTS: Path = PROJECT_ROOT / "outputs"
    SCRIPTS: Path = PROJECT_ROOT / "scripts"
    REPORTS: Path = PROJECT_ROOT / "reports"
    TESTS: Path = PROJECT_ROOT / "tests"
    DOCS: Path = PROJECT_ROOT / "docs"
    NOTEBOOKS: Path = PROJECT_ROOT / "notebooks"
    EXPERIMENTS: Path = PROJECT_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Data sub-directories
# ---------------------------------------------------------------------------

class DataDirs:
    """Data layer sub-directories."""

    RAW: Path = Dirs.DATA / "raw"
    PROCESSED: Path = Dirs.DATA / "processed"
    SYNTHETIC: Path = Dirs.DATA / "synthetic"
    NHANES: Path = Dirs.DATA / "nhanes"
    FRAMINGHAM: Path = Dirs.DATA / "framingham"


# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------

class DataPaths:
    """Canonical paths to specific data files."""

    # --- Raw inputs ---
    RAW_REGISTRY: Path = DataDirs.RAW / "eu_registry.csv"
    RAW_ENV_HISTORY: Path = DataDirs.RAW / "regional_environmental_history.csv"
    RAW_CLINICAL_COHORT: Path = DataDirs.RAW / "synthetic_clinical_cohort.csv"

    # --- Processed / analytical ---
    ANALYTICAL_COHORT: Path = DataDirs.PROCESSED / "analytical_cohort.csv"

    # --- Future dataset placeholders ---
    FRAMINGHAM_RAW: Path = DataDirs.FRAMINGHAM / "framingham.csv"
    NHANES_RAW: Path = DataDirs.NHANES / "nhanes.csv"


# ---------------------------------------------------------------------------
# Output sub-directories
# ---------------------------------------------------------------------------

class OutputDirs:
    """Output layer sub-directories — created at runtime if missing."""

    MODELS: Path = Dirs.OUTPUTS / "models"
    FIGURES: Path = Dirs.OUTPUTS / "figures"
    METRICS: Path = Dirs.OUTPUTS / "metrics"
    PREDICTIONS: Path = Dirs.OUTPUTS / "predictions"
    SHAP: Path = Dirs.OUTPUTS / "shap"
    CALIBRATION: Path = Dirs.OUTPUTS / "calibration"
    LOGS: Path = Dirs.OUTPUTS / "logs"


# ---------------------------------------------------------------------------
# Output file paths
# ---------------------------------------------------------------------------

class OutputPaths:
    """Canonical paths to specific output artefacts."""

    VIMP_CSV: Path = OutputDirs.METRICS / "vimp.csv"
    COX_COEFFICIENTS_CSV: Path = OutputDirs.METRICS / "cox_coefficients.csv"
    RISK_DISTRIBUTION_CSV: Path = OutputDirs.METRICS / "risk_distribution.csv"


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

class ConfigPaths:
    """Project configuration files."""

    DEFAULT_CONFIG: Path = Dirs.CONFIGS / "default.yaml"


# ---------------------------------------------------------------------------
# Convenience: ensure output directories exist
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """
    Create all output directories if they do not already exist.

    Call this once at the start of any script that writes outputs.
    """
    for attr_name in vars(OutputDirs):
        path = getattr(OutputDirs, attr_name)
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
