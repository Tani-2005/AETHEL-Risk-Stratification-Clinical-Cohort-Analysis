"""
publication_tables.py
======================
Generates the 8 publication-ready tables requested for the research paper.
Saves them as CSV files in the reports output directory.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

def build_table_1_characteristics(
    cohorts: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 1: Dataset Characteristics."""
    rows = []
    for name, df in cohorts.items():
        n = len(df)
        
        # Outcome Rate
        if "h_outcome_binary" in df.columns and df["h_outcome_binary"].notna().any():
            rate = f"{df['h_outcome_binary'].mean() * 100:.1f}%"
        else:
            rate = "N/A (Unsupervised)"
            
        # Age
        if "h_age" in df.columns and df["h_age"].notna().any():
            age = f"{df['h_age'].mean():.1f} ± {df['h_age'].std():.1f}"
        else:
            age = "N/A"
            
        # BMI
        if "h_bmi" in df.columns and df["h_bmi"].notna().any():
            bmi = f"{df['h_bmi'].mean():.1f} ± {df['h_bmi'].std():.1f}"
        else:
            bmi = "N/A"
            
        # Smoker
        if "h_is_smoker" in df.columns and df["h_is_smoker"].notna().any():
            smoker = f"{df['h_is_smoker'].mean() * 100:.1f}%"
        else:
            smoker = "N/A"
            
        # BP & Cholesterol
        sbp = f"{df['h_sys_bp'].mean():.1f} ± {df['h_sys_bp'].std():.1f}" if "h_sys_bp" in df.columns and df["h_sys_bp"].notna().any() else "N/A"
        dbp = f"{df['h_dia_bp'].mean():.1f} ± {df['h_dia_bp'].std():.1f}" if "h_dia_bp" in df.columns and df["h_dia_bp"].notna().any() else "N/A"
        tc = f"{df['h_total_cholesterol'].mean():.1f} ± {df['h_total_cholesterol'].std():.1f}" if "h_total_cholesterol" in df.columns and df["h_total_cholesterol"].notna().any() else "N/A"
        
        rows.append({
            "Cohort": name.capitalize(),
            "N": n,
            "Outcome Rate": rate,
            "Age (Mean ± SD)": age,
            "BMI (Mean ± SD)": bmi,
            "Smoker (%)": smoker,
            "SBP (Mean ± SD)": sbp,
            "DBP (Mean ± SD)": dbp,
            "Total Cholesterol (Mean ± SD)": tc,
        })
        
    df_t1 = pd.DataFrame(rows)
    df_t1.to_csv(output_dir / "table1_dataset_characteristics.csv", index=False)
    logger.info("Saved Table 1 (Characteristics) to %s", output_dir)
    return df_t1

def build_table_2_internal_performance(
    internal_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 2: Internal Performance."""
    df = pd.DataFrame(internal_results)
    df.to_csv(output_dir / "table2_internal_performance.csv", index=False)
    logger.info("Saved Table 2 (Internal Performance) to %s", output_dir)
    return df

def build_table_3_external_validation(
    external_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 3: External Validation."""
    df = pd.DataFrame(external_results)
    df.to_csv(output_dir / "table3_external_validation.csv", index=False)
    logger.info("Saved Table 3 (External Validation) to %s", output_dir)
    return df

def build_table_4_calibration_comparison(
    calib_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 4: Calibration Comparison."""
    df = pd.DataFrame(calib_results)
    df.to_csv(output_dir / "table4_calibration_comparison.csv", index=False)
    logger.info("Saved Table 4 (Calibration Comparison) to %s", output_dir)
    return df

def build_table_5_explanation_stability(
    explain_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 5: Explanation Stability."""
    df = pd.DataFrame(explain_results)
    df.to_csv(output_dir / "table5_explanation_stability.csv", index=False)
    logger.info("Saved Table 5 (Explanation Stability) to %s", output_dir)
    return df

def build_table_6_generalization_analysis(
    gen_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 6: Generalization Analysis."""
    df = pd.DataFrame(gen_results)
    df.to_csv(output_dir / "table6_generalization_analysis.csv", index=False)
    logger.info("Saved Table 6 (Generalization Analysis) to %s", output_dir)
    return df

def build_table_7_failure_analysis(
    failure_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 7: Failure Analysis."""
    df = pd.DataFrame(failure_results)
    df.to_csv(output_dir / "table7_failure_analysis.csv", index=False)
    logger.info("Saved Table 7 (Failure Analysis) to %s", output_dir)
    return df

def build_table_8_trustworthiness_summary(
    summary_results: list[dict],
    output_dir: Path,
) -> pd.DataFrame:
    """Table 8: Trustworthiness Summary."""
    df = pd.DataFrame(summary_results)
    df.to_csv(output_dir / "table8_trustworthiness_summary.csv", index=False)
    logger.info("Saved Table 8 (Trustworthiness Summary) to %s", output_dir)
    return df
