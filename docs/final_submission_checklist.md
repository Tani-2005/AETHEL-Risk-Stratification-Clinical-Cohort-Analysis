# AETHEL Publication Submission Checklist

This checklist guides the manuscript preparation, repository freezing, and final submission steps.

---

## 1. Manuscript Preparation
- [ ] Draft Abstract and Title.
- [ ] Populate Methods section with details of the harmonized cohorts and the `LeakageFreeCV` loop.
- [ ] Write Results section using metrics from Table 3 (Performance) and Table 4 (Calibration).
- [ ] Incorporate Figure 1 (DCA) and Figure 2 (Calibration curves) into the results text.
- [ ] Add Discussion points on calibration decoupling, explanation stability (FAS), and robustness boundaries.
- [ ] Address clinical limitations (static features, lack of longitudinal data) and ethical considerations.

---

## 2. Repository & Reproducibility
- [ ] Verify that all unit tests pass (`pytest` returns exit code 0).
- [ ] Confirm that `docs/reading_tracker.md` is updated with all literature citations.
- [ ] Ensure that `bibliography/bibliography_template.bib` contains all required BibTeX records.
- [ ] Freeze the repository state by creating a release tag (e.g., `v3.0.0-publication`).
- [ ] Include the Dockerfile or conda environment specs (`environment.yml`) in the repository root.

---

## 3. Final Submission Steps
- [ ] Format tables to match target journal styling guidelines (e.g., JAMIA/Nature formatting).
- [ ] Verify that high-resolution vector figures are exported to `docs/figures/`.
- [ ] Compile the supplementary appendix containing mathematical derivations of FAS and bootstrap intervals.
- [ ] Submit the manuscript and supplementary files to the target journal's editorial portal.
