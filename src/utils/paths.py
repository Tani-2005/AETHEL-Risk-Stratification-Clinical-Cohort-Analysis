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

    # Multi-cohort framework additions
    HARMONIZED: Path = Dirs.DATA / "harmonized"
    FEATURE_METADATA: Path = Dirs.DATA / "feature_metadata"
    MAPPING_TABLES: Path = Dirs.DATA / "mapping_tables"


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

    # --- Train / val / test splits (unscaled, with engineered features) ---
    TRAIN: Path = DataDirs.PROCESSED / "train.csv"
    VAL: Path = DataDirs.PROCESSED / "val.csv"
    TEST: Path = DataDirs.PROCESSED / "test.csv"

    # --- Raw dataset files ---
    FRAMINGHAM_RAW: Path = DataDirs.FRAMINGHAM / "framingham.csv"
    NHANES_RAW: Path = DataDirs.FRAMINGHAM / "combined_data.csv"  # NHANES-derived

    # --- Harmonized datasets (never overwrite raw) ---
    SYNTHETIC_HARMONIZED: Path = DataDirs.HARMONIZED / "synthetic_harmonized.csv"
    FRAMINGHAM_HARMONIZED: Path = DataDirs.HARMONIZED / "framingham_harmonized.csv"
    NHANES_HARMONIZED: Path = DataDirs.HARMONIZED / "nhanes_harmonized.csv"


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
    REPORTS: Path = Dirs.OUTPUTS / "reports"
    EXPERIMENTS: Path = Dirs.OUTPUTS / "experiments"


class ExperimentDirs:
    """Experiment-specific output directories."""

    BASE: Path = Dirs.OUTPUTS / "experiments"
    CONFIGS: Path = Dirs.CONFIGS / "experiments"


# ---------------------------------------------------------------------------
# Output file paths
# ---------------------------------------------------------------------------

class OutputPaths:
    """Canonical paths to specific output artefacts."""

    # Model outputs
    VIMP_CSV: Path = OutputDirs.METRICS / "vimp.csv"
    COX_COEFFICIENTS_CSV: Path = OutputDirs.METRICS / "cox_coefficients.csv"
    RISK_DISTRIBUTION_CSV: Path = OutputDirs.METRICS / "risk_distribution.csv"

    # Preprocessing artefacts
    SCALER_JOBLIB: Path = OutputDirs.MODELS / "robust_scaler.joblib"
    POLLUTION_INDEX_STATS: Path = OutputDirs.MODELS / "pollution_index_stats.csv"

    # Data quality / preprocessing reports
    DATA_QUALITY_REPORT: Path = OutputDirs.REPORTS / "data_quality_report.csv"
    CLASS_BALANCE_REPORT: Path = OutputDirs.REPORTS / "class_balance_report.csv"
    CORRELATION_REPORT: Path = OutputDirs.REPORTS / "correlation_report.csv"
    VIF_REPORT: Path = OutputDirs.REPORTS / "vif_report.csv"
    FEATURE_SUMMARY: Path = OutputDirs.REPORTS / "feature_summary.csv"
    PREPROCESSING_SUMMARY: Path = OutputDirs.REPORTS / "preprocessing_summary.txt"
    SPLIT_SUMMARY: Path = OutputDirs.REPORTS / "split_summary.csv"

    # Multi-cohort framework reports
    DATASET_AUDIT: Path = OutputDirs.REPORTS / "dataset_audit.csv"
    FEATURE_AVAILABILITY_MATRIX: Path = OutputDirs.REPORTS / "feature_availability_matrix.csv"
    HARMONIZATION_REPORT: Path = OutputDirs.REPORTS / "harmonization_report.csv"
    COHORT_COMPARISON_REPORT: Path = OutputDirs.REPORTS / "cohort_comparison_report.csv"
    DOMAIN_SHIFT_REPORT: Path = OutputDirs.REPORTS / "domain_shift_report.csv"
    CROSS_COHORT_SUMMARY: Path = OutputDirs.REPORTS / "cross_cohort_summary.csv"


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

class ConfigPaths:
    """Project configuration files."""

    DEFAULT_CONFIG: Path = Dirs.CONFIGS / "default.yaml"
    EXPERIMENTS_DIR: Path = Dirs.CONFIGS / "experiments"


# ---------------------------------------------------------------------------
# Convenience: ensure all runtime directories exist
# ---------------------------------------------------------------------------

class ExplainabilityDirs:
    """Explainability framework output directories — one sub-dir per experiment."""

    BASE: Path = Dirs.OUTPUTS / "explainability"

    # Sub-directories created dynamically per experiment: BASE / {exp_name} / {subdir}
    SHAP: str = "shap"
    PERMUTATION: str = "permutation"
    PDP: str = "pdp"
    ALE: str = "ale"
    INTERACTIONS: str = "interactions"
    LOCAL: str = "local"
    STABILITY: str = "stability"
    CONSENSUS: str = "consensus"
    REPORTS: str = "reports"

    @classmethod
    def experiment_dir(cls, exp_name: str) -> Path:
        return cls.BASE / exp_name

    @classmethod
    def subdir(cls, exp_name: str, model_name: str, subdir: str) -> Path:
        safe_model = model_name.replace(" ", "_").lower()
        return cls.BASE / exp_name / safe_model / subdir


def ensure_output_dirs() -> None:
    """Create all output and data directories if they do not already exist."""
    for cls in (OutputDirs, ExperimentDirs):
        for attr_name in vars(cls):
            path = getattr(cls, attr_name)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
    # Data versioning directories
    for d in (DataDirs.HARMONIZED, DataDirs.FEATURE_METADATA, DataDirs.MAPPING_TABLES):
        d.mkdir(parents=True, exist_ok=True)
    # Explainability base
    ExplainabilityDirs.BASE.mkdir(parents=True, exist_ok=True)
