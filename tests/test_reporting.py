"""
test_reporting.py
=================
Unit tests for the AETHEL Scientific Visualization and Reporting Framework.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from src.reporting.publication_layout import apply_publication_theme, clean_plot
from src.reporting.export_utils import export_figure, export_table, build_pdf_report
from src.reporting.table_generator import (
    build_table_1_characteristics,
    build_table_2_hyperparameters,
    build_table_3_performance,
    build_table_4_calibration,
    build_table_5_cross_cohort,
    build_table_6_feature_importance,
    build_table_7_explanation_stability,
    build_table_8_robustness,
    build_table_9_failure_analysis,
    build_table_10_trustworthiness
)
from src.reporting.report_generator import run_reporting_pipeline

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for test outputs."""
    return tmp_path

def test_publication_theme():
    """Verify that applying the publication theme does not raise errors."""
    try:
        apply_publication_theme()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        clean_plot(ax)
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"Applying publication theme failed: {e}")

def test_export_figure(temp_output_dir):
    """Verify that figures are correctly exported in SVG, PDF, and PNG."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    
    paths = export_figure(fig, temp_output_dir, "test_fig")
    plt.close(fig)
    
    assert "png" in paths
    assert "svg" in paths
    assert "pdf" in paths
    
    assert paths["png"].exists()
    assert paths["svg"].exists()
    assert paths["pdf"].exists()

def test_export_table(temp_output_dir):
    """Verify that tables are correctly exported in CSV, Excel, LaTeX, and Markdown."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    paths = export_table(df, temp_output_dir, "test_table")
    
    assert "csv" in paths
    assert "xlsx" in paths
    assert "tex" in paths
    assert "md" in paths
    
    assert paths["csv"].exists()
    assert paths["xlsx"].exists()
    assert paths["tex"].exists()
    assert paths["md"].exists()

def test_table_generation():
    """Verify that all 10 tables are correctly built and have contents."""
    generators = [
        build_table_1_characteristics,
        build_table_2_hyperparameters,
        build_table_3_performance,
        build_table_4_calibration,
        build_table_5_cross_cohort,
        build_table_6_feature_importance,
        build_table_7_explanation_stability,
        build_table_8_robustness,
        build_table_9_failure_analysis,
        build_table_10_trustworthiness
    ]
    
    for gen in generators:
        df = gen()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

def test_pdf_report_builder(temp_output_dir):
    """Verify building a ReportLab PDF report works without crashing."""
    pdf_path = temp_output_dir / "test_report.pdf"
    metadata = {"Author": "AETHEL Framework", "Version": "1.0"}
    
    content = [
        {"type": "h1", "text": "Test Report"},
        {"type": "p", "text": "This is a test paragraph for the PDF builder."},
        {"type": "table", "data": pd.DataFrame({"A": [1, 2], "B": [3, 4]}), "col_widths": [100, 100]}
    ]
    
    build_pdf_report("Test Document", metadata, content, pdf_path)
    assert pdf_path.exists()

def test_run_reporting_pipeline(temp_output_dir):
    """Verify the entire reporting pipeline executes on a temporary directory."""
    # Run reporting pipeline
    res = run_reporting_pipeline("test_experiment", temp_output_dir)
    
    # Check that outputs exist
    assert (temp_output_dir / "reports" / "publication_report.pdf").exists()
    assert (temp_output_dir / "reports" / "publication_report.html").exists()
    assert (temp_output_dir / "reports" / "publication_report.md").exists()
    assert (temp_output_dir / "supplementary" / "report_metadata.json").exists()
    
    # Check figures and tables subdirectories contain files
    fig_files = list((temp_output_dir / "figures").glob("*.png"))
    tbl_files = list((temp_output_dir / "tables").glob("*.csv"))
    
    assert len(fig_files) > 0
    assert len(tbl_files) > 0
