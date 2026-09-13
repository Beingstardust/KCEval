from __future__ import annotations

RUBRIC_VERSION = "segment_judge_rubric_v2_scale001_dependency_aware"

PARTIAL_DIMENSIONS = [
    "error_detection",
    "error_reasoning",
    "error_localisation",
    "solution_control",
    "scaffolding_quality",
    "actionability_moving_forward",
    "local_coherence_relevance",
    "clarity_cognitive_load",
    "student_level_calibration",
    "proactive_clarification",
]

TRUST_FLAGS = [
    "curriculum_hallucination_present",
    "student_context_hallucination_present",
    "domain_knowledge_hallucination_present",
]

MICRO_DIMENSIONS = [
    {
        "id": "error_detection",
        "group": "diagnostics",
        "score_type": "partial_0_0_5_1",
        "question": "If the student makes an error or shows confusion, does the tutor notice it?",
        "applicability": "Use null only when no student error, misconception, or confusion is visible."
    },
    {
        "id": "error_reasoning",
        "group": "diagnostics",
        "score_type": "partial_0_0_5_1",
        "question": "Does the tutor explain why the student's error or confusion is wrong?",
        "applicability": "Use null only when no student error or misconception is present."
    },
    {
        "id": "error_localisation",
        "group": "diagnostics",
        "score_type": "partial_0_0_5_1",
        "question": "Does the tutor locate the specific part of the student's reasoning, step, or statement that is wrong?",
        "applicability": "Use null only when no student error or misconception is present."
    },
    {
        "id": "solution_control",
        "group": "instructional_response",
        "score_type": "partial_0_0_5_1",
        "question": "Does the tutor control revealingness, avoiding unnecessary answer dumping when guided reasoning is appropriate?",
        "applicability": "Applicable for instructional segments."
    },
    {
        "id": "scaffolding_quality",
        "group": "instructional_response",
        "score_type": "partial_0_0_5_1",
        "question": "Does the tutor support understanding through hints, steps, prompts, explanations, or gradual guidance?",
        "applicability": "Applicable for instructional segments."
    },
    {
        "id": "actionability_moving_forward",
        "group": "instructional_response",
        "score_type": "partial_0_0_5_1",
        "question": "Does the tutor give the student a usable next step, correction, strategy, or way to continue?",
        "applicability": "Applicable for instructional segments."
    },
    {
        "id": "local_coherence_relevance",
        "group": "communication_calibration",
        "score_type": "partial_0_0_5_1",
        "question": "Is the tutor response coherent and relevant to the student's current turn and segment context?",
        "applicability": "Applicable for all segments."
    },
    {
        "id": "clarity_cognitive_load",
        "group": "communication_calibration",
        "score_type": "partial_0_0_5_1",
        "question": "Is the explanation clear and not unnecessarily overloaded for the learner?",
        "applicability": "Applicable for instructional segments."
    },
    {
        "id": "student_level_calibration",
        "group": "communication_calibration",
        "score_type": "partial_0_0_5_1",
        "question": "Is the explanation calibrated to the student's apparent level, prior statements, and current confusion?",
        "applicability": "Use null only when there is insufficient student context to infer level."
    },
    {
        "id": "proactive_clarification",
        "group": "communication_calibration",
        "score_type": "partial_0_0_5_1",
        "question": "When the student state or request is ambiguous, does the tutor appropriately ask or invite clarification rather than over-assuming?",
        "applicability": "Use null when the student need is already clear and clarification is unnecessary."
    },
]

PARTIAL_SCALE = {
    "0": "poor, absent, misleading, or harmful",
    "0.5": "partial, mixed, weak, or incomplete",
    "1": "good, clearly present, useful, or appropriate",
    "null": "not applicable"
}

TRUST_FLAG_SCALE = {
    "0": "no material hallucination detected",
    "1": "material hallucination detected"
}

MODE_POLICIES = {
    "single_kc_primary": (
        "Evaluate primarily against the primary KC. The evaluator_kc_set may provide minor context, "
        "but do not let unrelated KCs distract from the primary KC judgment."
    ),
    "branch_context_with_anchor": (
        "Treat the resolved KC as a context anchor, not as exact truth. Evaluate whether the tutor response "
        "is pedagogically useful within the supplied branch/context KC set."
    ),
    "broad_kc_set_low_confidence_anchor": (
        "Treat the resolved KC only as a low-confidence anchor. Evaluate the segment as a broad or recap-like "
        "multi-KC region using the evaluator_kc_set. Do not penalize the segment merely because it is not a precise single-KC unit."
    ),
}

DEPENDENCY_RULES = [
    {
        "rule_id": "D_TRUSTWORTHINESS_CAPS_PEDAGOGICAL_USEFULNESS",
        "type": "hard_gate",
        "condition": "Any trust flag is 1.",
        "effect": "Cap pedagogical usefulness and force human review when the hallucination is material."
    },
    {
        "rule_id": "D_ERROR_DETECTION_PRECEDES_REASONING_AND_LOCALISATION",
        "type": "hard_gate",
        "condition": "A student error is present but error_detection is 0.",
        "effect": "error_reasoning and error_localisation cannot remain high."
    },
    {
        "rule_id": "D_LOCALISATION_GATES_TARGETED_SCAFFOLDING",
        "type": "hard_gate",
        "condition": "Correction or remediation segment has error_localisation = 0.",
        "effect": "Targeted scaffolding and actionability should be capped."
    },
    {
        "rule_id": "D_SOLUTION_CONTROL_MODIFIES_ACTIVE_LEARNING",
        "type": "conditional_gate",
        "condition": "Guided reasoning is expected but the tutor reveals too much too early.",
        "effect": "Reduce scaffolding_quality and actionability_moving_forward."
    },
    {
        "rule_id": "D_CLARITY_MODIFIES_ACTIONABILITY",
        "type": "soft_modifier",
        "condition": "The next step exists but is hard to understand or cognitively overloaded.",
        "effect": "Reduce actionability_moving_forward if clarity_cognitive_load is low."
    },
    {
        "rule_id": "D_CALIBRATION_MODIFIES_SCAFFOLDING",
        "type": "soft_modifier",
        "condition": "The tutor response is mismatched to the student's apparent level.",
        "effect": "Reduce scaffolding_quality when student_level_calibration is low."
    },
    {
        "rule_id": "D_CLARIFICATION_COMPENSATES_UNDER_AMBIGUITY",
        "type": "conditional_modifier",
        "condition": "Student state is ambiguous and tutor asks a useful clarifying question.",
        "effect": "Do not over-penalize error_localisation or actionability."
    },
    {
        "rule_id": "D_RELEVANCE_MODIFIES_INSTRUCTIONAL_VALUE",
        "type": "soft_modifier",
        "condition": "Tutor response is off-topic or poorly connected to the student's turn.",
        "effect": "Reduce scaffolding_quality, actionability_moving_forward, and student_level_calibration."
    }
]

OUTPUT_SCHEMA = {
    "segment_id": "string",
    "evaluation_mode": "single_kc_primary | branch_context_with_anchor | broad_kc_set_low_confidence_anchor",
    "raw_dimension_scores": {dimension_id: "0 | 0.5 | 1 | null" for dimension_id in PARTIAL_DIMENSIONS},
    "dimension_scores": {dimension_id: "0 | 0.5 | 1 | null after dependency adjustments" for dimension_id in PARTIAL_DIMENSIONS},
    "dimension_applicability": {dimension_id: "applicable | not_applicable | unclear" for dimension_id in PARTIAL_DIMENSIONS},
    "trust_flags": {flag_id: "0 | 1" for flag_id in TRUST_FLAGS},
    "dependency_adjustments": [
        {
            "rule_id": "one of the provided dependency rule ids",
            "applied": "boolean",
            "reason": "short evidence-grounded reason",
            "affected_dimensions": ["partial dimension ids affected by the rule"]
        }
    ],
    "micro_score_raw": "0.0 to 1.0, weighted mean of raw_dimension_scores excluding nulls",
    "micro_score_dependency_adjusted": "0.0 to 1.0, weighted mean after dependency adjustments",
    "trust_adjusted_score": "0.0 to 1.0, after trust flag caps or penalties",
    "kc_grounding": {
        "primary_kc_used": "boolean",
        "context_kcs_used": ["string"],
        "unsupported_or_wrong_kc_claims": ["string"]
    },
    "flags": {
        "major_correctness_error": "boolean",
        "ungrounded_claim": "boolean",
        "missed_student_need": "boolean",
        "overly_answer_giving": "boolean",
        "segment_boundary_problem": "boolean",
        "needs_human_review": "boolean"
    },
    "evidence_quotes": ["short quotes from supplied segment only"],
    "rationale_short": "2-4 sentence explanation grounded only in the supplied packet"
}
