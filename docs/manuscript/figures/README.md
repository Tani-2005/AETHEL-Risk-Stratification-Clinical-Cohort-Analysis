# Figures Directory

This directory stores high-resolution export plots, vectors, and diagram drafts for the AETHEL manuscript.

## Directory Structure

*   `primary/`: Core publication-grade figures referenced in the paper:
    *   `fig1_dca_net_benefit.png`: Clinical Net Benefit curve from Decision Curve Analysis.
    *   `fig2_calibration_curves.png`: Reliability diagrams comparing uncalibrated vs. Platt scaled vs. Isotonic models.
    *   `fig3_robustness_stress.png`: Robustness and performance decay profile under feature ablation/noise stress.
    *   `fig4_shap_consensus.png`: Attribution rank consensus heatmap across target cohorts.
*   `dataset/`: Baseline cohort profiling and characteristics plots.
*   `explainability/`: Local and global XAI explanations (SHAP summary, beeswarm, PDP/ICE grids, and ALE plots).
*   `generalization/`: Domain shift and model transferability reports between Framingham and NHANES cohorts.
*   `performance/`: Model validation and testing metrics (ROC curves, PR curves, confusion matrices).
*   `robustness/`: Stability metrics under feature perturbations, data ablation, and noise injection.
*   `workflow/`: Operational pipeline charts and AETHEL clinical auditing workflows.
