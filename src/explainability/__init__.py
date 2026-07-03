"""
AETHEL Explainability Package
==============================
Publication-grade explainability framework for clinical ML models.

Evaluates explanation reliability through:
  - SHAP global and local explanations
  - Permutation importance
  - Partial Dependence Plots (PDP) and ICE curves
  - Accumulated Local Effects (ALE)
  - SHAP interaction analysis
  - Explanation stability across random seeds
  - Cross-method consensus and agreement scoring
  - Clinically hedged interpretation generation

All modules are model-agnostic where possible.
Tree-specific features (SHAP interactions) detect model type automatically.
"""

from src.explainability.shap_analysis import SHAPAnalyser
from src.explainability.permutation_analysis import PermutationAnalyser
from src.explainability.pdp_analysis import PDPAnalyser
from src.explainability.ale_analysis import ALEAnalyser
from src.explainability.interaction_analysis import InteractionAnalyser
from src.explainability.local_explanations import LocalExplainer
from src.explainability.stability_analysis import StabilityAnalyser
from src.explainability.consensus_analysis import ConsensusAnalyser
from src.explainability.clinical_interpretation import ClinicalInterpreter

__all__ = [
    "SHAPAnalyser",
    "PermutationAnalyser",
    "PDPAnalyser",
    "ALEAnalyser",
    "InteractionAnalyser",
    "LocalExplainer",
    "StabilityAnalyser",
    "ConsensusAnalyser",
    "ClinicalInterpreter",
]
