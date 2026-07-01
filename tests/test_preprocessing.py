"""
test_preprocessing.py
=====================
Smoke tests for the AETHEL preprocessing pipeline.

These tests verify that the core data generation functions produce outputs
of the correct shape and column structure — without testing the exact numeric
values (which are deterministic by seed and covered implicitly).

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short

Tests are designed to be fast (<5s total) and runnable without R installed
by testing only the Python preprocessing stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def registry_df() -> pd.DataFrame:
    """Build the EU registry once and reuse across all tests in this session."""
    from src.preprocessing.build_eu_registry import build_registry
    return build_registry()


@pytest.fixture(scope="session")
def env_df(registry_df: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG001
    """Generate environmental data (depends on registry)."""
    from src.preprocessing.generate_env_data import generate_pan_european_data
    return generate_pan_european_data()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestBuildEuRegistry:
    """Tests for src.preprocessing.build_eu_registry"""

    def test_registry_row_count(self, registry_df: pd.DataFrame) -> None:
        """Registry must contain exactly 90 cities."""
        assert len(registry_df) == 90, f"Expected 90 cities, got {len(registry_df)}"

    def test_registry_columns_present(self, registry_df: pd.DataFrame) -> None:
        """All expected columns must exist."""
        from src.utils.constants import Columns
        expected = [Columns.CITY_ID, Columns.CITY, Columns.COUNTRY,
                    Columns.LAT, Columns.LON, Columns.BASE_PM25, Columns.BASE_NO2]
        for col in expected:
            assert col in registry_df.columns, f"Missing column: {col}"

    def test_registry_no_nulls(self, registry_df: pd.DataFrame) -> None:
        """No null values in the registry."""
        assert registry_df.isnull().sum().sum() == 0

    def test_registry_pollution_positive(self, registry_df: pd.DataFrame) -> None:
        """Baseline pollution values must be positive."""
        from src.utils.constants import Columns
        assert (registry_df[Columns.BASE_PM25] > 0).all()
        assert (registry_df[Columns.BASE_NO2] > 0).all()

    def test_registry_file_written(self) -> None:
        """Output CSV must exist on disk."""
        from src.utils.paths import DataPaths
        assert DataPaths.RAW_REGISTRY.exists()


# ---------------------------------------------------------------------------
# Environmental data tests
# ---------------------------------------------------------------------------

class TestGenerateEnvData:
    """Tests for src.preprocessing.generate_env_data"""

    def test_env_data_row_count(self, env_df: pd.DataFrame) -> None:
        """90 cities × 60 months = 5,400 records."""
        expected_rows = 90 * 60
        assert len(env_df) == expected_rows, f"Expected {expected_rows}, got {len(env_df)}"

    def test_env_data_columns_present(self, env_df: pd.DataFrame) -> None:
        """Required columns must exist."""
        from src.utils.constants import Columns
        expected = [Columns.DATE, Columns.LOCATION, Columns.PM25, Columns.NO2]
        for col in expected:
            assert col in env_df.columns, f"Missing column: {col}"

    def test_env_data_no_negative_pollution(self, env_df: pd.DataFrame) -> None:
        """Pollution values are floored at 0 — no negatives allowed."""
        from src.utils.constants import Columns
        assert (env_df[Columns.PM25] >= 0).all()
        assert (env_df[Columns.NO2] >= 0).all()

    def test_env_data_file_written(self) -> None:
        """Output CSV must exist on disk."""
        from src.utils.paths import DataPaths
        assert DataPaths.RAW_ENV_HISTORY.exists()


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------

class TestConfigLoader:
    """Tests for src.utils.config_loader"""

    def test_config_loads_successfully(self) -> None:
        """Config must load without errors."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg is not None

    def test_config_seed_values(self) -> None:
        """Seeds must be positive integers."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert isinstance(cfg.seeds.python, int) and cfg.seeds.python > 0
        assert isinstance(cfg.seeds.r, int) and cfg.seeds.r > 0

    def test_config_study_params(self) -> None:
        """Study parameters must be positive."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg.study.n_subjects > 0
        assert cfg.study.total_cities > 0
        assert cfg.study.observation_years > 0

    def test_config_covariates_not_empty(self) -> None:
        """Survival covariate list must not be empty."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert len(cfg.features.survival_covariates) > 0


# ---------------------------------------------------------------------------
# Paths tests
# ---------------------------------------------------------------------------

class TestPaths:
    """Tests for src.utils.paths"""

    def test_project_root_exists(self) -> None:
        """Project root directory must exist."""
        from src.utils.paths import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_configs_dir_exists(self) -> None:
        """configs/ directory must exist."""
        from src.utils.paths import Dirs
        assert Dirs.CONFIGS.exists()

    def test_default_config_exists(self) -> None:
        """configs/default.yaml must exist."""
        from src.utils.paths import ConfigPaths
        assert ConfigPaths.DEFAULT_CONFIG.exists()
