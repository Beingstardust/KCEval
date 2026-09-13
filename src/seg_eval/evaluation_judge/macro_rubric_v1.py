"""Full-dialogue macro rubric — 4 dimensions, defined by the thesis proposal, section 6.5.

Source: `Thesis_Proposal.pdf`, "6.5 Macro-level evaluation of full dialogues". The one-line
definitions below are the proposal's own wording, not invented here; the ``question`` and
``applicability`` fields operationalise each definition into judge-usable guidance in the same
shape as ``rubric_v2.MICRO_DIMENSIONS``, kept as literal to the source sentence as possible.

Proposal quote for context (not reproduced field-by-field elsewhere):
    "Macro evaluation targets dialogue properties that emerge only across multiple segments
    and cannot be reliably inferred from local scoring alone... The macro evaluation is run
    separately from individual segment scoring and contributes to the final tutor score to
    ensure the framework avoids treating dialogue-level quality as a simple aggregation of
    segment-level scores, even when dialogue structure and adaptation matter."

Scale: reuses the same 0 / 0.5 / 1 / null convention as the micro rubric (``rubric_v2.PARTIAL_SCALE``)
rather than defining a new one — the proposal does not specify a different scale for macro
dimensions, and introducing one would break the shared aggregation code in
``src/seg_eval/aggregation/deterministic_aggregation.py``.
"""

from __future__ import annotations

MACRO_RUBRIC_VERSION = "dialogue_macro_rubric_v1_from_proposal_section_6_5"

MACRO_DIMENSIONS: list[dict[str, str]] = [
    {
        "id": "adaptability",
        "proposal_definition": "How the tutor responds to the student state over time.",
        "question": (
            "Across the dialogue, does the tutor's approach change in response to signals of "
            "the student's evolving state -- demonstrated understanding, persistent confusion, "
            "a shift in pace, or a change in what the student is asking for? A tutor that "
            "notices and adjusts (e.g. slows down after repeated confusion, moves faster once "
            "the student demonstrates mastery, changes explanation style after one fails) "
            "scores well; a tutor that repeats the same approach regardless of how the student "
            "is doing does not."
        ),
        "applicability": (
            "Use null only when the dialogue is too short (effectively one exchange) to show "
            "whether the tutor adapts to anything -- adaptability requires at least a "
            "before/after to observe."
        ),
    },
    {
        "id": "consistency",
        "proposal_definition": (
            "Whether the tutor stays consistent and avoids contradictions within itself "
            "across exchanges."
        ),
        "question": (
            "Does the tutor avoid contradicting its own earlier statements, terminology, or "
            "guidance later in the same dialogue? This is about self-consistency across the "
            "tutor's own turns, not about factual correctness against the curriculum (that is "
            "covered by the micro-level trustworthiness dimensions) and not about the "
            "student's consistency."
        ),
        "applicability": (
            "Use null only when the dialogue is too short for a later statement to have a "
            "chance to contradict an earlier one."
        ),
    },
    {
        "id": "outcome_completion",
        "proposal_definition": "Whether the dialogue reaches a coherent resolution.",
        "question": (
            "By the end of the dialogue, has the student's initial question or difficulty "
            "reached some coherent resolution -- an answer, a corrected understanding, or an "
            "explicit, sensible stopping point -- rather than trailing off unresolved or ending "
            "mid-explanation?"
        ),
        "applicability": (
            "Always applicable if the dialogue has an end; do not mark not_applicable merely "
            "because the ending is a poor one -- a poor or absent resolution is itself the "
            "thing being scored (score low, don't exclude)."
        ),
    },
    {
        "id": "sequentiality",
        "proposal_definition": (
            "Whether the tutor follows a logical ordering and dependencies across steps."
        ),
        "question": (
            "Across the dialogue, does the tutor introduce concepts and steps in an order that "
            "respects their dependencies -- prerequisites before what depends on them, "
            "diagnosis before remediation, explanation before application -- rather than "
            "jumping around in a way that would confuse a student following along in order?"
        ),
        "applicability": (
            "Use null only when the dialogue's content has no meaningful ordering to assess "
            "(e.g. a single self-contained exchange with no multi-step structure)."
        ),
    },
]

MACRO_DIMENSION_IDS = [d["id"] for d in MACRO_DIMENSIONS]

# Deliberately NOT importing PARTIAL_SCALE here to avoid a fragile cross-module identity
# assumption; this is the same literal scale, kept as a local constant so this module stays
# readable on its own and a change to rubric_v2's scale text doesn't silently reword this one.
MACRO_SCALE = {
    "0": "poor, absent, misleading, or harmful",
    "0.5": "partial, mixed, weak, or incomplete",
    "1": "good, clearly present, useful, or appropriate",
    "null": "not applicable",
}

MACRO_OUTPUT_SCHEMA = {
    "dialogue_id": "string, the dialogue this macro judgment covers",
    "dimension_scores": {d["id"]: "0 | 0.5 | 1 | null" for d in MACRO_DIMENSIONS},
    "dimension_applicability": {
        d["id"]: "applicable | not_applicable | unclear" for d in MACRO_DIMENSIONS
    },
    "evidence_pointers": [
        {
            "dimension": "one of the 4 macro dimension ids",
            "exchange_id": "the exchange_id this evidence comes from",
            "quote": "short verbatim quote from that exchange",
        }
    ],
    "rationale_short": "20-1400 chars, no hidden chain-of-thought, one paragraph per dimension is fine",
}
