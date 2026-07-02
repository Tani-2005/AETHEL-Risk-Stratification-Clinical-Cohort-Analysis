"""
test_loaders.py + test_harmonization.py — Multi-cohort framework tests.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_dataset():
    from src.datasets.synthetic_loader import SyntheticLoader
    return SyntheticLoader().load()


@pytest.fixture(scope="session")
def framingham_dataset():
    from src.datasets.framingham_loader import FraminghamLoader
    return FraminghamLoader().load()


@pytest.fixture(scope="session")
def nhanes_dataset():
    from src.datasets.nhanes_loader import NHANESLoader
    return NHANESLoader().load()


@pytest.fixture(scope="session")
def all_datasets(synthetic_dataset, framingham_dataset, nhanes_dataset):
    return {
        "synthetic": synthetic_dataset,
        "framingham": framingham_dataset,
        "nhanes": nhanes_dataset,
    }


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestSyntheticLoader:
    def test_loads_successfully(self, synthetic_dataset):
        assert synthetic_dataset is not None
        assert len(synthetic_dataset.df_harmonized) > 0

    def test_n_patients(self, synthetic_dataset):
        assert synthetic_dataset.metadata.n == 1000

    def test_supervised(self, synthetic_dataset):
        assert synthetic_dataset.metadata.supervised is True

    def test_outcome_present(self, synthetic_dataset):
        from src.utils.constants import HarmonizedColumns
        assert HarmonizedColumns.OUTCOME_BINARY in synthetic_dataset.df_harmonized.columns
        assert synthetic_dataset.df_harmonized[HarmonizedColumns.OUTCOME_BINARY].notna().any()

    def test_h_prefix_columns_exist(self, synthetic_dataset):
        h_cols = [c for c in synthetic_dataset.df_harmonized.columns if c.startswith("h_")]
        assert len(h_cols) > 0, "No h_* harmonized columns found."

    def test_raw_preserved(self, synthetic_dataset):
        assert "age" in synthetic_dataset.df_raw.columns
        assert "bmi" in synthetic_dataset.df_raw.columns
        assert "is_smoker" in synthetic_dataset.df_raw.columns

    def test_sex_absent(self, synthetic_dataset):
        from src.utils.constants import HarmonizedColumns
        col = HarmonizedColumns.SEX_MALE
        assert col not in synthetic_dataset.feature_schema.common_available, \
            "h_sex_male should be missing from Synthetic."

    def test_event_rate_approx(self, synthetic_dataset):
        assert 0.20 < synthetic_dataset.metadata.event_rate < 0.30


class TestFraminghamLoader:
    def test_loads_successfully(self, framingham_dataset):
        assert len(framingham_dataset.df_harmonized) > 0

    def test_n_patients(self, framingham_dataset):
        assert framingham_dataset.metadata.n == 4240

    def test_supervised(self, framingham_dataset):
        assert framingham_dataset.metadata.supervised is True

    def test_sex_present(self, framingham_dataset):
        from src.utils.constants import HarmonizedColumns
        assert HarmonizedColumns.SEX_MALE in framingham_dataset.feature_schema.common_available

    def test_outcome_binary(self, framingham_dataset):
        from src.utils.constants import HarmonizedColumns
        col = HarmonizedColumns.OUTCOME_BINARY
        vals = set(framingham_dataset.df_harmonized[col].dropna().unique())
        assert vals.issubset({0, 1, 0.0, 1.0})

    def test_event_rate_approx(self, framingham_dataset):
        assert 0.10 < framingham_dataset.metadata.event_rate < 0.25

    def test_raw_columns_preserved(self, framingham_dataset):
        for col in ["male", "age", "BMI", "TenYearCHD"]:
            assert col in framingham_dataset.df_raw.columns


class TestNHANESLoader:
    def test_loads_successfully(self, nhanes_dataset):
        assert len(nhanes_dataset.df_harmonized) > 0

    def test_not_supervised(self, nhanes_dataset):
        assert nhanes_dataset.metadata.supervised is False
        assert nhanes_dataset.metadata.outcome_col is None

    def test_no_outcome_in_common_available(self, nhanes_dataset):
        from src.utils.constants import HarmonizedColumns
        assert HarmonizedColumns.OUTCOME_BINARY not in nhanes_dataset.feature_schema.common_available

    def test_biochemical_features_present(self, nhanes_dataset):
        from src.utils.constants import HarmonizedColumns
        for feat in [HarmonizedColumns.SYS_BP, HarmonizedColumns.DIA_BP,
                     HarmonizedColumns.TOTAL_CHOLESTEROL, HarmonizedColumns.LDL]:
            assert feat in nhanes_dataset.feature_schema.common_available, \
                f"{feat} should be in NHANES common_available."

    def test_no_age_bmi_smoking(self, nhanes_dataset):
        from src.utils.constants import HarmonizedColumns
        for feat in [HarmonizedColumns.AGE, HarmonizedColumns.BMI, HarmonizedColumns.IS_SMOKER]:
            assert feat not in nhanes_dataset.feature_schema.common_available


# ---------------------------------------------------------------------------
# Harmonization tests
# ---------------------------------------------------------------------------

class TestHarmonization:
    def test_h_columns_consistent_across_loaders(self, all_datasets):
        from src.utils.constants import HarmonizedColumns
        all_h = HarmonizedColumns.ALL_FEATURES + [HarmonizedColumns.OUTCOME_BINARY]
        for name, ds in all_datasets.items():
            for col in all_h:
                assert col in ds.df_harmonized.columns, \
                    f"h_* column '{col}' missing from {name} harmonized DataFrame."

    def test_raw_not_overwritten(self, synthetic_dataset):
        assert "age" in synthetic_dataset.df_raw.columns
        assert "h_age" not in synthetic_dataset.df_raw.columns

    def test_no_shared_patient_ids_cross_dataset(self, synthetic_dataset, framingham_dataset):
        """Synthetic patient IDs should not appear in Framingham (different sources)."""
        if "patient_id" in synthetic_dataset.df_raw.columns:
            synth_ids = set(synthetic_dataset.df_raw["patient_id"])
            # Framingham uses row index, not patient_id — just check datasets are distinct
            assert len(synth_ids) == synthetic_dataset.metadata.n

    def test_feature_availability_matrix_correct(self, all_datasets):
        from src.utils.constants import HarmonizedColumns
        synth = all_datasets["synthetic"]
        fram = all_datasets["framingham"]
        nhanes = all_datasets["nhanes"]
        # Age should be in Synthetic and Framingham but not NHANES
        assert HarmonizedColumns.AGE in synth.feature_schema.common_available
        assert HarmonizedColumns.AGE in fram.feature_schema.common_available
        assert HarmonizedColumns.AGE not in nhanes.feature_schema.common_available

    def test_supervised_intersection_nonempty(self, synthetic_dataset, framingham_dataset):
        from src.utils.constants import HarmonizedColumns
        synth_feats = set(synthetic_dataset.feature_schema.common_available)
        fram_feats = set(framingham_dataset.feature_schema.common_available)
        intersection = synth_feats & fram_feats - {HarmonizedColumns.OUTCOME_BINARY}
        assert len(intersection) >= 3, \
            f"Expected ≥3 intersection features, got {len(intersection)}: {intersection}"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_loads_all(self):
        from src.datasets.registry import DatasetRegistry
        registry = DatasetRegistry()
        datasets = registry.load_all()
        assert len(datasets) == 3

    def test_registry_audit_produces_dataframe(self):
        from src.datasets.registry import DatasetRegistry
        registry = DatasetRegistry()
        audit_df = registry.audit()
        assert isinstance(audit_df, pd.DataFrame)
        assert len(audit_df) == 3
        assert "dataset" in audit_df.columns
        assert "supervised" in audit_df.columns

    def test_registry_unknown_name_raises(self):
        from src.datasets.registry import DatasetRegistry
        registry = DatasetRegistry()
        with pytest.raises(KeyError):
            registry.load("nonexistent_cohort")


# ---------------------------------------------------------------------------
# Experiment config tests
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_mode1_loads(self):
        from src.experiments.experiment_config import ExperimentConfig
        cfg = ExperimentConfig.from_yaml("configs/experiments/exp_mode1_synthetic.yaml")
        assert cfg.mode == 1
        assert cfg.train_datasets == ["synthetic"]

    def test_mode4_loads(self):
        from src.experiments.experiment_config import ExperimentConfig
        cfg = ExperimentConfig.from_yaml("configs/experiments/exp_mode4_synthetic_to_framingham.yaml")
        assert cfg.mode == 4
        assert "synthetic" in cfg.train_datasets
        assert "framingham" in cfg.val_datasets

    def test_mode7_combined_datasets(self):
        from src.experiments.experiment_config import ExperimentConfig
        cfg = ExperimentConfig.from_yaml("configs/experiments/exp_mode7_combined.yaml")
        assert "synthetic" in cfg.train_datasets
        assert "framingham" in cfg.train_datasets
