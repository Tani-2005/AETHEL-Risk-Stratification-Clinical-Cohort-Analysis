"""
feature_selector.py
===================
Identifies multicollinearity using Variance Inflation Factors (VIF).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils.config_loader import AETHELConfig
from src.utils.paths import OutputPaths
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """
    Analyzes feature multicollinearity and correlation.
    """
    def __init__(self, cfg: AETHELConfig) -> None:
        self._cfg = cfg
        self._vif_threshold = cfg.feature_sel.vif_threshold
        self._corr_threshold = cfg.feature_sel.correlation_threshold

    def analyse(self, df: pd.DataFrame) -> None:
        # Exclude IDs, targets, and non-numeric columns
        exclude = ["patient_id", "months_observed", "event_occurred", "h_outcome_binary", "_source_dataset"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = [c for c in numeric_cols if c not in exclude]
        
        # Check if we have features to analyze
        if not features:
            logger.warning("No numeric features found to analyze VIF.")
            return
            
        # Fit VIF using data without NaNs
        df_clean = df[features].dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = features
        
        if len(df_clean) > 5:
            # Add constant intercept column
            X_const = sm.add_constant(df_clean)
            vifs = []
            for i in range(len(features)):
                try:
                    # Index + 1 because column 0 is the constant intercept
                    v = variance_inflation_factor(X_const.values, i + 1)
                    vifs.append(round(v, 4))
                except Exception:
                    vifs.append(np.nan)
            vif_data["VIF"] = vifs
        else:
            vif_data["VIF"] = np.nan
            
        OutputPaths.VIF_REPORT.parent.mkdir(parents=True, exist_ok=True)
        vif_data.to_csv(OutputPaths.VIF_REPORT, index=False)
        logger.info("VIF report saved to %s", OutputPaths.VIF_REPORT)
