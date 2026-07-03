"""
src/reporting package
=====================
Package for scientific visualization, table generation, and PDF/HTML report compilation.
"""
from __future__ import annotations

from src.reporting.publication_layout import apply_publication_theme, clean_plot, CLINICAL_COLORS
from src.reporting.export_utils import export_figure, export_table, build_pdf_report
from src.reporting.report_generator import run_reporting_pipeline
