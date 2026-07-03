"""
report_generator.py
===================
Orchestrates publication asset generation and compiles reports (HTML, MD, PDF).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from src.reporting.figure_generator import generate_all_plots
from src.reporting.table_generator import (
    generate_all_tables,
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
from src.reporting.export_utils import build_pdf_report

def build_markdown_report(
    experiment_name: str,
    metadata: dict,
    output_dir: Path
) -> Path:
    """Generates the main Markdown publication report."""
    md_path = output_dir / "reports" / "publication_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# AETHEL Scientific Evaluation & Trustworthiness Report",
        f"**Experiment Name:** {experiment_name}  ",
        f"**Timestamp:** {metadata.get('Timestamp', 'N/A')}  ",
        f"**Seed:** {metadata.get('Random Seed', 'N/A')}  ",
        "",
        "## Executive Summary",
        "This document presents the complete clinical validation, explainability stability, adversarial robustness, and cross-cohort generalization profile for the risk stratification models under the AETHEL framework.",
        "",
        "## Section 1: Pipeline & Workflow Diagrams",
        "The complete modeling and validation steps are programmatically tracked and detailed in the following diagrams:",
        "- Overall Pipeline: `../figures/workflow_overall_pipeline.png`",
        "- Data Processing: `../figures/workflow_data_processing.png`",
        "- Model Training: `../figures/workflow_model_training.png`",
        "- Evaluation: `../figures/workflow_evaluation.png`",
        "- Explainability: `../figures/workflow_explainability.png`",
        "- Cross-Cohort: `../figures/workflow_cross_cohort.png`",
        "- Trustworthiness: `../figures/workflow_trustworthiness_framework.png`",
        "",
        "## Section 2: Dataset Characteristics",
        "Detailed characteristics, missing data checks, and availability matrix across cohorts are detailed in Table 1.",
        "- Missing Data Heatmap: `../figures/dataset_missing_heatmap.png`",
        "- Class Distribution: `../figures/dataset_class_distribution.png`",
        "- Feature Distribution: `../figures/dataset_feature_distributions.png`",
        "- Correlation Matrix: `../figures/dataset_correlation_matrix.png`",
        "- Feature Availability Matrix: `../figures/dataset_feature_availability.png`",
        "- Cross-Cohort Distribution Comparison: `../figures/dataset_cross_cohort_kde.png`",
        "",
        "## Section 3: Model Performance & Calibration",
        "Evaluation metrics with 95% confidence intervals, McNemar p-values, and calibration expected errors are listed in Tables 3 and 4.",
        "- ROC Curves: `../figures/performance_roc_curves.png`",
        "- PR Curves: `../figures/performance_pr_curves.png`",
        "- Calibration Curves: `../figures/performance_calibration.png`",
        "- Confusion Matrices: `../figures/performance_confusion_matrices.png`",
        "- Metric Comparison: `../figures/performance_metric_comparison.png`",
        "- Model Ranking: `../figures/performance_model_ranking.png`",
        "- Bootstrap Distribution: `../figures/performance_bootstrap_distributions.png`",
        "",
        "## Section 4: Explainability Stability",
        "Global and local explanations using SHAP values, Permutation Importance, and 1D/2D Partial Dependence Plots are illustrated in the plots and detailed in Table 6.",
        "- SHAP Summary: `../figures/explainability_shap_summary.png`",
        "- SHAP Beeswarm: `../figures/explainability_shap_beeswarm.png`",
        "- Waterfall: `../figures/explainability_waterfall.png`",
        "- Decision Plot: `../figures/explainability_decision.png`",
        "- Force Plot: `../figures/explainability_force.png`",
        "- Permutation Importance: `../figures/explainability_permutation.png`",
        "- PDP: `../figures/explainability_pdp.png`",
        "- ALE: `../figures/explainability_ale.png`",
        "- Interaction Heatmap: `../figures/explainability_interactions.png`",
        "- Consensus Ranking: `../figures/explainability_consensus.png`",
        "",
        "## Section 5: Robustness Audits",
        "Stability heatmaps under seed variation, MCAR missingness tolerance thresholds, and Gaussian noise limits are detailed in Table 8.",
        "- Noise Robustness: `../figures/robustness_noise_decay.png`",
        "- Feature Ablation: `../figures/robustness_feature_ablation.png`",
        "- Missing Data Decay: `../figures/robustness_missing_decay.png`",
        "- Prediction Variance: `../figures/robustness_prediction_variance.png`",
        "- Uncertainty Profile: `../figures/robustness_uncertainty_profile.png`",
        "",
        "## Section 6: Cross-Cohort Generalization & Trustworthiness",
        "Domain shift (PSI and Wasserstein distance), calibration transfer, and explanation rank agreement Spearman correlations are detailed in Tables 5, 7, 9 and summarized in Table 10.",
        "- Cross-Cohort Matrix: `../figures/generalization_cross_cohort_matrix.png`",
        "- Performance Drop: `../figures/generalization_performance_drop.png`",
        "- Calibration Transfer: `../figures/generalization_calibration_transfer.png`",
        "- Domain Shift Heatmap: `../figures/generalization_domain_shift_heatmap.png`",
        "- Feature Drift: `../figures/generalization_feature_drift.png`",
        "- Explanation Drift: `../figures/generalization_explanation_drift.png`",
        "- Clinical Consistency: `../figures/generalization_clinical_consistency.png`",
        "",
        "## Section 7: Tables List",
        "- Table 1: Dataset Characteristics (`../tables/table1_dataset_characteristics.md`)",
        "- Table 2: Model Hyperparameters (`../tables/table2_model_hyperparameters.md`)",
        "- Table 3: Performance Metrics (`../tables/table3_performance_metrics.md`)",
        "- Table 4: Calibration Metrics (`../tables/table4_calibration_metrics.md`)",
        "- Table 5: Cross-Cohort Validation (`../tables/table5_cross_cohort_validation.md`)",
        "- Table 6: Feature Importance (`../tables/table6_feature_importance.md`)",
        "- Table 7: Explanation Stability (`../tables/table7_explanation_stability.md`)",
        "- Table 8: Robustness Metrics (`../tables/table8_robustness_metrics.md`)",
        "- Table 9: Failure Analysis (`../tables/table9_failure_analysis.md`)",
        "- Table 10: Trustworthiness Summary (`../tables/table10_trustworthiness_summary.md`)"
    ]
    
    with md_path.open("w") as fh:
        fh.write("\n".join(lines))
    return md_path


def build_html_report(
    experiment_name: str,
    metadata: dict,
    output_dir: Path
) -> Path:
    """Generates the main HTML publication report with inline styled CSS."""
    html_path = output_dir / "reports" / "publication_report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We will construct a beautiful HTML page with inline styling and responsive structures.
    # We will embed Tables as HTML tables.
    t1_html = build_table_1_characteristics().to_html(classes="table", index=False)
    t3_html = build_table_3_performance().to_html(classes="table", index=False)
    t4_html = build_table_4_calibration().to_html(classes="table", index=False)
    t10_html = build_table_10_trustworthiness().to_html(classes="table", index=False)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AETHEL Scientific Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #f8f9fa;
        }}
        h1, h2, h3 {{
            color: #1f77b4;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 8px;
        }}
        h1 {{
            font-size: 2.2em;
            color: #2c3e50;
        }}
        .meta-box {{
            background-color: #ecf0f1;
            border-left: 5px solid #1f77b4;
            padding: 15px;
            margin-bottom: 30px;
            font-size: 0.9em;
            border-radius: 4px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }}
        .card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .card h4 {{
            margin: 10px 0 5px 0;
            color: #2c3e50;
        }}
        .table-container {{
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            overflow-x: auto;
        }}
        table.table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
        }}
        table.table th, table.table td {{
            padding: 10px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        table.table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        table.table tr:nth-child(even) {{
            background-color: #fdfdfd;
        }}
    </style>
</head>
<body>
    <h1>AETHEL Scientific Evaluation & Trustworthiness Report</h1>
    <div class="meta-box">
        <strong>Experiment:</strong> {experiment_name}<br/>
        <strong>Timestamp:</strong> {metadata.get('Timestamp', 'N/A')}<br/>
        <strong>Random Seed:</strong> {metadata.get('Random Seed', 'N/A')}<br/>
        <strong>Features Mode:</strong> {metadata.get('Features Strategy', 'N/A')}
    </div>
    
    <h2>1. Executive Summary</h2>
    <p>This report compiles clinical validation, explainability stability, adversarial robustness, and cross-cohort generalization profile for the risk stratification models under the AETHEL framework.</p>

    <h2>2. Pipeline & Workflow Diagrams</h2>
    <div class="grid">
        <div class="card">
            <img src="../figures/workflow_overall_pipeline.png" alt="Overall Pipeline">
            <h4>Figure 1.1: Overall Pipeline Workflow</h4>
        </div>
        <div class="card">
            <img src="../figures/workflow_trustworthiness_framework.png" alt="Trustworthiness Framework">
            <h4>Figure 1.2: Trustworthiness Scoring Framework</h4>
        </div>
    </div>

    <h2>3. Dataset Characteristics</h2>
    <div class="table-container">
        <h3>Table 1: Dataset Characteristics</h3>
        {t1_html}
    </div>
    <div class="grid">
        <div class="card">
            <img src="../figures/dataset_missing_heatmap.png" alt="Missing Heatmap">
            <h4>Figure 2.1: Missingness Patterns</h4>
        </div>
        <div class="card">
            <img src="../figures/dataset_cross_cohort_kde.png" alt="Cross-Cohort SBP">
            <h4>Figure 2.2: SBP Cohort Comparison</h4>
        </div>
    </div>

    <h2>4. Model Performance & Calibration</h2>
    <div class="table-container">
        <h3>Table 3: Performance Metrics</h3>
        {t3_html}
    </div>
    <div class="table-container">
        <h3>Table 4: Calibration Metrics</h3>
        {t4_html}
    </div>
    <div class="grid">
        <div class="card">
            <img src="../figures/performance_roc_curves.png" alt="ROC">
            <h4>Figure 3.1: ROC Curves</h4>
        </div>
        <div class="card">
            <img src="../figures/performance_calibration.png" alt="Calibration">
            <h4>Figure 3.2: Calibration Curves</h4>
        </div>
    </div>

    <h2>5. Explainability Stability</h2>
    <div class="grid">
        <div class="card">
            <img src="../figures/explainability_shap_summary.png" alt="SHAP Summary">
            <h4>Figure 4.1: Global SHAP Summary</h4>
        </div>
        <div class="card">
            <img src="../figures/explainability_shap_beeswarm.png" alt="Beeswarm">
            <h4>Figure 4.2: SHAP Beeswarm</h4>
        </div>
    </div>

    <h2>6. Trustworthiness Summary</h2>
    <div class="table-container">
        <h3>Table 10: Trustworthiness Grades</h3>
        {t10_html}
    </div>
</body>
</html>
"""
    with html_path.open("w") as fh:
        fh.write(html_content)
    return html_path


def build_pdf_publication_report(
    experiment_name: str,
    metadata: dict,
    output_dir: Path
) -> Path:
    """Generates the main ReportLab PDF publication report."""
    pdf_path = output_dir / "reports" / "publication_report.pdf"
    
    # Define ReportLab elements
    content_list = [
        {"type": "h1", "text": "AETHEL Clinical AI Trustworthiness Report"},
        {"type": "p", "text": "This report compiles the multi-dimensional statistical, explainability, robustness, and generalization assessment for the AETHEL clinical risk stratification framework."},
        
        {"type": "h2", "text": "1. Overall System Architecture & Workflow"},
        {"type": "p", "text": "The complete execution flow of the AETHEL multi-cohort clinical AI validator is programmatically structured as shown in the pipeline workflow diagram below."},
        {"type": "image", "path": output_dir / "figures" / "workflow_overall_pipeline.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "2. Cohort Data Characteristics"},
        {"type": "p", "text": "We validate AETHEL models across three harmonized datasets: Synthetic, Framingham, and NHANES (Composite Cardiovascular Risk outcome)."},
        {"type": "table", "data": build_table_1_characteristics(), "col_widths": [140, 110, 100, 100]},
        {"type": "image", "path": output_dir / "figures" / "dataset_cross_cohort_kde.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "3. Model Performance & Statistical Evaluation"},
        {"type": "p", "text": "All performance estimates include 95% confidence intervals constructed via 1,000 bootstrap resamples. XGBoost and LightGBM are evaluated in development/paper suites."},
        {"type": "table", "data": build_table_3_performance(), "col_widths": [85, 100, 100, 50, 60, 60]},
        {"type": "image", "path": output_dir / "figures" / "performance_roc_curves.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "4. Model Calibration & Reliability"},
        {"type": "p", "text": "Expected Calibration Error (ECE) and Brier scores are computed to evaluate risk score reliability across the target predictions."},
        {"type": "table", "data": build_table_4_calibration(), "col_widths": [100, 80, 80, 80, 80, 80]},
        {"type": "image", "path": output_dir / "figures" / "performance_calibration.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "5. Explainability Stability & Consensus"},
        {"type": "p", "text": "Explainability is assessed using additive SHAP explanations and Permutation feature importances to guarantee transparency in risk factors."},
        {"type": "table", "data": build_table_6_feature_importance(), "col_widths": [100, 80, 100, 70, 100]},
        {"type": "image", "path": output_dir / "figures" / "explainability_shap_summary.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "6. Robustness Audits & MCAR Sensitivity"},
        {"type": "p", "text": "Robustness checks include MCAR missing data sweeping and Gaussian feature noise injections, measuring stability indices across seeds."},
        {"type": "table", "data": build_table_8_robustness(), "col_widths": [110, 90, 100, 100, 100]},
        {"type": "image", "path": output_dir / "figures" / "robustness_noise_decay.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "7. Cross-Cohort Generalization & Shift Detection"},
        {"type": "p", "text": "Generalization gaps and Wasserstein domain shifts measure how reliably model weights transfer from source cohorts to target patients."},
        {"type": "table", "data": build_table_5_cross_cohort(), "col_widths": [100, 100, 100, 70, 80, 50]},
        {"type": "image", "path": output_dir / "figures" / "generalization_cross_cohort_matrix.png"},
        {"type": "page_break"},
        
        {"type": "h2", "text": "8. Trustworthiness Grading Summary"},
        {"type": "p", "text": "The final grading index rates models across seven key areas of trustworthiness and clinical reliability."},
        {"type": "table", "data": build_table_10_trustworthiness(), "col_widths": [110, 50, 50, 50, 50, 50, 50, 50, 90]},
        {"type": "image", "path": output_dir / "figures" / "workflow_trustworthiness_framework.png"}
    ]
    
    build_pdf_report(
        title="AETHEL Publication-Grade Evaluation & Trustworthiness Report",
        metadata=metadata,
        content_list=content_list,
        output_path=pdf_path
    )
    return pdf_path


def run_reporting_pipeline(experiment_name: str, output_dir: Path) -> dict:
    """
    Orchestrates the entire scientific visualization, table construction,
    and HTML/MD/PDF report compilation for a completed experiment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Metadata
    metadata = {
        "Project": "AETHEL Clinical Cohort Analysis",
        "Experiment Name": experiment_name,
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "Random Seed": 42,
        "Features Strategy": "Intersection Feature Set",
        "Status": "COMPLETED",
        "Evaluator Mode": "Nature Digital Medicine Standards"
    }
    
    # Save metadata
    (output_dir / "supplementary").mkdir(parents=True, exist_ok=True)
    with (output_dir / "supplementary" / "report_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
        
    # 2. Run plot generation
    print("Generating publication figures...")
    plots_paths = generate_all_plots(output_dir)
    
    # 3. Run table generation
    print("Generating publication tables...")
    tables_paths = generate_all_tables(output_dir)
    
    # 4. Build reports
    print("Compiling publication-ready reports (HTML, MD, PDF)...")
    md_report = build_markdown_report(experiment_name, metadata, output_dir)
    html_report = build_html_report(experiment_name, metadata, output_dir)
    pdf_report = build_pdf_publication_report(experiment_name, metadata, output_dir)
    
    print(f"Reporting complete. Assets saved under: {output_dir}")
    return {
        "metadata": metadata,
        "figures": plots_paths,
        "tables": tables_paths,
        "reports": {
            "markdown": md_report,
            "html": html_report,
            "pdf": pdf_report
        }
    }
