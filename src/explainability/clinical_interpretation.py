"""
clinical_interpretation.py
===========================
Rule-based clinical interpretation for harmonized features.

For each feature, provides:
  - Clinical meaning (what this feature represents clinically)
  - Supported biological mechanism (evidence-based pathway)
  - Literature reference (published finding — static knowledge base)
  - Hedged language (always uses 'consistent with', 'may indicate', 'in agreement with')

IMPORTANT: This module NEVER claims causality.
Forbidden words: 'causes', 'proves', 'demonstrates causation'.
All interpretations use epistemic hedges.

Clinical knowledge base covers all harmonized AETHEL features:
  h_age, h_bmi, h_is_smoker, h_sex_male, h_sys_bp, h_dia_bp,
  h_total_cholesterol, h_ldl, h_triglycerides, h_glucose
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Clinical Knowledge Base
# Each entry is a dict with:
#   - description: clinical meaning
#   - mechanism: biological/physiological pathway
#   - direction_high_risk: what elevated values indicate (if continuous)
#   - direction_positive: what positive/presence indicates (if binary)
#   - literature: published finding to cite
#   - hedge: always hedged
# ---------------------------------------------------------------------------

_CLINICAL_KB: dict[str, dict] = {
    "h_age": {
        "display_name": "Age",
        "type": "continuous",
        "description": (
            "Chronological age is one of the strongest non-modifiable risk factors "
            "across most chronic and acute disease outcomes."
        ),
        "mechanism": (
            "Advancing age is associated with progressive immune senescence, "
            "accumulated oxidative stress, telomere shortening, and reduced regenerative "
            "capacity across multiple organ systems. These changes may reduce physiological "
            "reserve and increase vulnerability to disease."
        ),
        "direction_high_risk": (
            "Higher age values are consistent with increased disease susceptibility, "
            "in agreement with established epidemiological evidence across cardiovascular, "
            "respiratory, and metabolic disease domains."
        ),
        "literature": (
            "Consistent with: Franceschi C et al. (2018). Inflammaging: a new immune-metabolic "
            "viewpoint for age-related diseases. Nature Reviews Endocrinology 14: 576-590."
        ),
        "hedge": "consistent with established epidemiological evidence",
    },
    "h_bmi": {
        "display_name": "Body Mass Index (BMI)",
        "type": "continuous",
        "description": (
            "BMI is a widely used anthropometric proxy for body adiposity. "
            "It is imperfect (does not distinguish fat from lean mass) but is consistently "
            "associated with metabolic and cardiovascular risk at a population level."
        ),
        "mechanism": (
            "Excess adipose tissue — particularly visceral fat — is associated with "
            "chronic low-grade inflammation via elevated pro-inflammatory cytokines "
            "(TNF-α, IL-6, CRP), adipokine dysregulation, and insulin resistance. "
            "These pathways may contribute to systemic disease risk."
        ),
        "direction_high_risk": (
            "Higher BMI values are associated with elevated systemic inflammatory "
            "burden, in agreement with established evidence linking adiposity to "
            "cardiovascular and metabolic disease risk."
        ),
        "literature": (
            "Consistent with: Collaboration NRF (2016). Trends in adult body-mass index "
            "in 200 countries from 1975 to 2014. Lancet 387: 1377-1396."
        ),
        "hedge": "in agreement with established metabolic risk evidence",
    },
    "h_is_smoker": {
        "display_name": "Smoking Status",
        "type": "binary",
        "description": (
            "Current or recent tobacco smoking history. Smoking is a major modifiable "
            "risk factor with well-documented associations across pulmonary, cardiovascular, "
            "and oncological outcomes."
        ),
        "mechanism": (
            "Tobacco smoke exposure is associated with direct epithelial injury to respiratory "
            "surfaces, induction of systemic oxidative stress, endothelial dysfunction, "
            "platelet aggregation, and accelerated atherosclerosis. Nicotine and carbon monoxide "
            "may further impair vascular and cardiac function."
        ),
        "direction_positive": (
            "Smoking history is consistent with elevated risk across multiple disease domains, "
            "in agreement with large-scale cohort evidence including the Framingham Heart Study."
        ),
        "literature": (
            "Consistent with: US Department of Health and Human Services (2014). "
            "The Health Consequences of Smoking — 50 Years of Progress: A Report of the "
            "Surgeon General. Atlanta: CDC."
        ),
        "hedge": "consistent with established smoking-related disease associations",
    },
    "h_sex_male": {
        "display_name": "Biological Sex (Male)",
        "type": "binary",
        "description": (
            "Biological sex (coded as male=1) reflects sex-based biological differences "
            "in hormonal milieu, metabolic regulation, and immune function."
        ),
        "mechanism": (
            "Sex-based differences in estrogen and testosterone levels are associated with "
            "differential cardiovascular protection and inflammatory profiles. In many "
            "cardiovascular disease contexts, male sex is associated with earlier onset "
            "of disease, though this pattern varies by endpoint and age group."
        ),
        "direction_positive": (
            "Male sex may indicate different baseline disease risk compared to female sex, "
            "with the specific direction and magnitude depending on the outcome domain and "
            "age stratum. This is consistent with sex-stratified epidemiological evidence."
        ),
        "literature": (
            "Consistent with: Regitz-Zagrosek V (2012). Sex and gender differences in health. "
            "EMBO Reports 13(7): 596-603."
        ),
        "hedge": "may indicate sex-related biological differences in risk",
    },
    "h_sys_bp": {
        "display_name": "Systolic Blood Pressure",
        "type": "continuous",
        "description": (
            "Systolic blood pressure (mmHg) measures peak arterial pressure during "
            "ventricular contraction. It is a key determinant of cardiovascular risk burden."
        ),
        "mechanism": (
            "Elevated systolic blood pressure is associated with arterial wall shear stress, "
            "endothelial dysfunction, left ventricular hypertrophy, and accelerated "
            "atherosclerosis. These mechanisms may contribute to cardiovascular, renal, "
            "and cerebrovascular disease risk."
        ),
        "direction_high_risk": (
            "Higher systolic blood pressure values are consistent with elevated "
            "cardiovascular disease risk, in agreement with the established J-curve "
            "and linear dose-response relationships reported in population studies."
        ),
        "literature": (
            "Consistent with: Ettehad D et al. (2016). Blood pressure lowering for "
            "prevention of cardiovascular disease and death. Lancet 387: 957-967."
        ),
        "hedge": "consistent with cardiovascular hemodynamic risk evidence",
    },
    "h_dia_bp": {
        "display_name": "Diastolic Blood Pressure",
        "type": "continuous",
        "description": (
            "Diastolic blood pressure (mmHg) measures arterial pressure between "
            "heartbeats. It is relevant to coronary perfusion and overall vascular load."
        ),
        "mechanism": (
            "Elevated diastolic pressure may indicate increased peripheral vascular "
            "resistance and impaired vasodilation capacity. This may be associated with "
            "early-stage hypertensive organ damage and metabolic dysregulation."
        ),
        "direction_high_risk": (
            "Higher diastolic blood pressure values may indicate elevated cardiovascular "
            "and metabolic disease risk, consistent with established blood pressure "
            "guidelines and cohort evidence."
        ),
        "literature": (
            "Consistent with: Whelton PK et al. (2018). 2017 ACC/AHA Hypertension "
            "Guideline. Journal of the American College of Cardiology 71(19): e127-e248."
        ),
        "hedge": "may indicate hypertensive risk burden",
    },
    "h_total_cholesterol": {
        "display_name": "Total Cholesterol",
        "type": "continuous",
        "description": (
            "Total serum cholesterol (mg/dL) is a composite measure of LDL, HDL, "
            "and VLDL cholesterol. It is a broadly used cardiometabolic risk biomarker."
        ),
        "mechanism": (
            "Elevated total cholesterol — particularly when driven by LDL elevation — "
            "is associated with accelerated atherosclerotic plaque formation, increased "
            "arterial stiffness, and heightened cardiovascular event risk."
        ),
        "direction_high_risk": (
            "Higher total cholesterol values are consistent with elevated atherosclerotic "
            "cardiovascular risk, in agreement with the Framingham Risk Score and "
            "subsequent lipid-lowering intervention evidence."
        ),
        "literature": (
            "Consistent with: Pencina MJ et al. (2009). Predicting the 30-year risk of "
            "cardiovascular disease. Circulation 119: 3078-3084."
        ),
        "hedge": "in agreement with established lipid-cardiovascular risk evidence",
    },
    "h_ldl": {
        "display_name": "LDL Cholesterol",
        "type": "continuous",
        "description": (
            "Low-density lipoprotein cholesterol (LDL-C, mg/dL) is a primary therapeutic "
            "target in cardiovascular risk management and a direct driver of atherogenesis."
        ),
        "mechanism": (
            "LDL particles enter arterial walls, undergo oxidative modification, "
            "and trigger macrophage-mediated foam cell formation — the hallmark of "
            "atherosclerotic plaque. Elevated LDL-C may accelerate this process."
        ),
        "direction_high_risk": (
            "Higher LDL cholesterol values are consistent with elevated atherosclerotic "
            "risk, in agreement with decades of Mendelian randomisation and clinical "
            "trial evidence supporting LDL lowering interventions."
        ),
        "literature": (
            "Consistent with: Ference BA et al. (2017). Low-density lipoproteins cause "
            "atherosclerotic cardiovascular disease. European Heart Journal 38: 2459-2472."
        ),
        "hedge": "consistent with LDL-mediated atherogenic risk",
    },
    "h_triglycerides": {
        "display_name": "Triglycerides",
        "type": "continuous",
        "description": (
            "Serum triglycerides (mg/dL) reflect dietary fat absorption, hepatic "
            "lipid synthesis, and lipolytic activity. They are markers of metabolic "
            "dysregulation."
        ),
        "mechanism": (
            "Elevated triglycerides are associated with insulin resistance, hepatic "
            "steatosis, and the metabolic syndrome constellation. High triglycerides "
            "may contribute to atherogenic dyslipidaemia and systemic inflammatory "
            "signalling."
        ),
        "direction_high_risk": (
            "Higher triglyceride values may indicate metabolic syndrome and associated "
            "cardiometabolic risk, consistent with established evidence on atherogenic "
            "dyslipidaemia."
        ),
        "literature": (
            "Consistent with: Nordestgaard BG et al. (2007). Nonfasting triglycerides "
            "and risk of myocardial infarction, ischemic heart disease, and death. JAMA 298: 299-308."
        ),
        "hedge": "may indicate metabolic syndrome and associated cardiometabolic risk",
    },
    "h_glucose": {
        "display_name": "Blood Glucose",
        "type": "continuous",
        "description": (
            "Fasting or non-fasting blood glucose (mg/dL) reflects carbohydrate "
            "metabolism and insulin sensitivity. It is a key marker for diabetes "
            "and metabolic disease assessment."
        ),
        "mechanism": (
            "Chronic hyperglycaemia is associated with non-enzymatic glycation of "
            "proteins and lipids, generation of advanced glycation end-products (AGEs), "
            "and downstream vascular and renal damage. These mechanisms may contribute "
            "to both microvascular and macrovascular disease."
        ),
        "direction_high_risk": (
            "Higher glucose levels are consistent with impaired insulin sensitivity "
            "or frank diabetes, in agreement with established evidence linking "
            "dysglycaemia to cardiovascular, renal, and retinal disease risk."
        ),
        "literature": (
            "Consistent with: Emerging Risk Factors Collaboration (2010). Diabetes "
            "mellitus, fasting blood glucose concentration, and risk of vascular disease. "
            "Lancet 375: 2215-2222."
        ),
        "hedge": "consistent with dysglycaemia-associated disease risk",
    },
}

# Forbidden causal language patterns
_FORBIDDEN_TERMS = [
    "causes", "caused by", "proves", "proven", "demonstrates causation",
    "definitively shows", "establishes causality", "is responsible for",
]


def _check_no_causal_language(text: str) -> bool:
    """Return True if text is safe (contains no forbidden causal terms)."""
    text_lower = text.lower()
    return not any(term in text_lower for term in _FORBIDDEN_TERMS)


class ClinicalInterpreter:
    """
    Generates clinically hedged interpretations for AETHEL model predictions.

    All outputs use epistemic hedge language ('consistent with', 'may indicate',
    'in agreement with', 'is associated with'). Causal claims are never generated.

    Parameters
    ----------
    feature_names : list[str] — harmonized feature names (h_* prefix)
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names

    def interpret(
        self,
        feature_name: str,
        shap_direction: str = "positive",   # 'positive' or 'negative'
        shap_magnitude: Optional[float] = None,
    ) -> dict:
        """
        Generate clinical interpretation for a single feature.

        Parameters
        ----------
        feature_name : str — harmonized feature name
        shap_direction : 'positive' (risk-increasing) or 'negative' (protective)
        shap_magnitude : float (optional) — mean |SHAP| value

        Returns
        -------
        dict with clinical_meaning, biological_mechanism, literature, hedge_statement,
               potential_clinical_meaning, causal_claim_check
        """
        kb = _CLINICAL_KB.get(feature_name)

        if kb is None:
            logger.debug("ClinicalInterpreter: no KB entry for '%s'", feature_name)
            return {
                "feature": feature_name,
                "display_name": feature_name.replace("h_", "").replace("_", " ").title(),
                "clinical_meaning": "Clinical interpretation not available for this feature.",
                "biological_mechanism": "Unknown.",
                "literature": "No specific reference available.",
                "hedge_statement": "This feature's contribution may indicate patterns "
                                   "consistent with known clinical risk factors.",
                "potential_clinical_meaning": "Requires domain expert review.",
                "causal_claim_check": True,
            }

        direction_text = (
            kb.get("direction_high_risk", kb.get("direction_positive", ""))
            if shap_direction == "positive"
            else (
                f"Lower {kb['display_name']} values or absence of this factor "
                f"may be associated with relatively reduced risk, "
                f"consistent with the established clinical relationship."
            )
        )

        magnitude_note = ""
        if shap_magnitude is not None:
            if shap_magnitude > 0.05:
                magnitude_note = " This feature had a notable impact on the model's predictions."
            elif shap_magnitude > 0.01:
                magnitude_note = " This feature had a moderate influence on predicted risk."
            else:
                magnitude_note = " This feature had a minor influence on predicted risk."

        result = {
            "feature": feature_name,
            "display_name": kb["display_name"],
            "feature_type": kb["type"],
            "shap_direction": shap_direction,
            "shap_magnitude": shap_magnitude,
            "clinical_meaning": kb["description"],
            "biological_mechanism": kb["mechanism"],
            "direction_interpretation": direction_text + magnitude_note,
            "literature": kb["literature"],
            "hedge_statement": kb["hedge"],
            "potential_clinical_meaning": (
                f"A {shap_direction} contribution from {kb['display_name']} "
                f"in this model's prediction {kb['hedge']}. "
                f"This finding should be interpreted in the context of the patient's "
                f"full clinical presentation and does not constitute clinical advice."
            ),
            "causal_claim_check": _check_no_causal_language(
                kb["description"] + kb["mechanism"] + direction_text
            ),
        }
        return result

    def generate_feature_report(
        self,
        shap_mean_abs: dict[str, float],
        out_path: Path,
    ) -> list[dict]:
        """
        Generate clinical interpretation for all features with SHAP importances.

        Parameters
        ----------
        shap_mean_abs : dict[feature_name → mean|SHAP|]
        out_path : Path to save JSON report

        Returns
        -------
        list of interpretation dicts, sorted by importance.
        """
        interpretations = []
        for feat, magnitude in sorted(shap_mean_abs.items(), key=lambda x: x[1], reverse=True):
            direction = "positive" if magnitude >= 0 else "negative"
            interp = self.interpret(feat, direction, abs(magnitude))
            interpretations.append(interp)

        # Verify no causal language in any output
        causal_violations = [
            i["feature"] for i in interpretations if not i["causal_claim_check"]
        ]
        if causal_violations:
            logger.error(
                "ClinicalInterpreter: CAUSAL LANGUAGE DETECTED in features: %s",
                causal_violations,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(interpretations, fh, indent=2)
        logger.info("Saved clinical interpretation report: %s", out_path)
        return interpretations

    def generate_cross_cohort_comparison(
        self,
        cohort_shap_rankings: dict[str, dict[str, float]],
        out_path: Path,
    ) -> dict:
        """
        Compare SHAP importance rankings across cohorts and identify consistent
        vs. cohort-specific risk factors.

        Parameters
        ----------
        cohort_shap_rankings : dict[cohort_name → dict[feature → mean|SHAP|]]

        Returns
        -------
        dict with consistent_factors, cohort_specific_factors, and clinical interpretation.
        """
        cohorts = list(cohort_shap_rankings.keys())
        all_features = set()
        for ranking in cohort_shap_rankings.values():
            all_features.update(ranking.keys())

        # Find features consistently important across cohorts (present in top-3 of all)
        top3_per_cohort = {
            cohort: set(
                sorted(ranking, key=lambda f: ranking[f], reverse=True)[:3]
            )
            for cohort, ranking in cohort_shap_rankings.items()
        }

        consistent = set.intersection(*top3_per_cohort.values()) if top3_per_cohort else set()
        cohort_specific = {
            cohort: top3_per_cohort[cohort] - consistent
            for cohort in cohorts
        }

        def _interp_list(feats: set) -> list[str]:
            return [
                _CLINICAL_KB.get(f, {}).get("display_name", f) for f in feats
            ]

        report = {
            "cohorts_compared": cohorts,
            "consistent_top_factors": {
                "features": list(consistent),
                "display_names": _interp_list(consistent),
                "interpretation": (
                    f"The following factors were consistently among the top predictors "
                    f"across all cohorts: {', '.join(_interp_list(consistent))}. "
                    f"This consistency may indicate generalisable risk associations "
                    f"that are in agreement with established clinical evidence."
                    if consistent else
                    "No features were consistently top-ranked across all cohorts."
                ),
            },
            "cohort_specific_factors": {
                cohort: {
                    "features": list(feats),
                    "display_names": _interp_list(feats),
                    "possible_clinical_reason": (
                        f"The importance of {', '.join(_interp_list(feats))} in the "
                        f"{cohort} cohort may reflect the specific population characteristics, "
                        f"outcome definition, or feature availability of this dataset. "
                        f"These factors should be interpreted with caution and may not "
                        f"generalise to other populations."
                    ),
                }
                for cohort, feats in cohort_specific.items()
            },
            "all_cohort_rankings": cohort_shap_rankings,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Saved cross-cohort comparison report: %s", out_path)
        return report
