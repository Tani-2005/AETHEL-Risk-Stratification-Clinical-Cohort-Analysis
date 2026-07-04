# =============================================================================
# AETHEL — Reproducible Research Makefile
# =============================================================================

.PHONY: help preprocess train evaluate explain robustness report paper clean test

help:
	@echo "AETHEL Pipeline Targets:"
	@echo "  make preprocess  - Run data generation and pre-processing pipeline"
	@echo "  make train       - Train Cox/RSF and classifier models"
	@echo "  make evaluate    - Perform statistical and calibration evaluations"
	@echo "  make explain     - Run explainability suite (SHAP, Permutation, etc.)"
	@echo "  make robustness  - Run robustness, missingness, and stability audits"
	@echo "  make report      - Generate publication-grade HTML and PDF reports"
	@echo "  make paper       - Execute the full paper-mode experiment suite"
	@echo "  make clean       - Remove intermediate logs, outputs, and cache files"
	@echo "  make test        - Run all automated unit tests"

preprocess:
	python scripts/run_pipeline.py --skip-r

train:
	python scripts/run_evaluation.py

evaluate:
	python scripts/run_evaluation.py

explain:
	python scripts/run_explainability.py --mode dev

robustness:
	python scripts/run_robustness.py --mode dev

report:
	python scripts/run_publication_report.py --experiment exp_mode1_synthetic

paper:
	python run.py --experiment exp_mode1_synthetic --mode paper

test:
	pytest tests/

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['outputs', '.pytest_cache', '__pycache__']]"
	python -c "import glob, os; [os.remove(f) for f in glob.glob('**/*.pyc', recursive=True)]"
