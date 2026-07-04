# Manuscript Directory

This directory stores the official publication-grade manuscript draft and its generated graphical and tabular assets.

## Structure

*   `report.qmd`: The master Quarto manuscript document (compiles the paper with inline R execution for statistics, generating PDF/HTML/Markdown outputs).
*   `figures/`: Organized manuscript figures:
    *   `primary/`: The 4 primary figures (Fig 1 to Fig 4) compiled for the manuscript.
    *   `dataset/`, `explainability/`, `generalization/`, `performance/`, `robustness/`, `workflow/`: Background sub-component figures.
*   `tables/`: Organized table outputs:
    *   `primary/`: The 10 core publication-grade academic tables (available in `.tex`, `.md`, `.csv`, and `.xlsx` formats).
    *   `raw_reports/`: Intermediate and raw pipeline reports (VIF, class balance, data quality splits, and model permutation importances).
