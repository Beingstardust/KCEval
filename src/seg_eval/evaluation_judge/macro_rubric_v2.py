"""Macro rubric v2 — the four v1 dimensions plus one: `factual_correctness`. AN EXTENSION.

WHY IT EXISTS
-------------
v1 asks four questions (adaptability, consistency, outcome_completion, sequentiality) and its
prompt explicitly instructs the judge: "Do not evaluate curriculum correctness or KC grounding
here; that is handled by the separate per-segment judge." That was a deliberate design decision
inherited from the proposal, and its consequence has been measured twice:

  - doc 24: the tutor variant carrying 15 planted factual errors scored a PERFECT 1/1/1/1 at macro
    level and its rationale praised a definition that had been deliberately corrupted. Raising the
    macro weight therefore makes the framework LESS sensitive to hallucination, not more.
  - doc 46: with the micro level capped by grounded factuality and the macro level blind to it, the
    two channels disagree about that arm, so the final ranking depends on the weighting rather than
    on the tutor.

v1 is unchanged and stays the default everywhere; every published number keeps reproducing from it.
v2 is built, run and reported alongside it so the choice is visible rather than assumed.

WHAT THE NEW DIMENSION IS, AND IS NOT
-------------------------------------
It asks whether what the tutor TAUGHT was true, across the dialogue as a whole. It is not
`consistency`, which asks only whether the tutor contradicts itself and is satisfied by a tutor who
is uniformly wrong. It is not the per-segment trust flags, which fire inside a single segment and
cannot see a claim that is only wrong in the light of what was said ten exchanges earlier.
"""
from __future__ import annotations

from .macro_rubric_v1 import MACRO_DIMENSIONS as _V1_DIMENSIONS
from .macro_rubric_v1 import MACRO_SCALE  # noqa: F401  (re-exported for callers)

MACRO_RUBRIC_VERSION = "macro_judge_rubric_v2_scale001_with_factual_correctness"

FACTUAL_CORRECTNESS = {
    "id": "factual_correctness",
    "proposal_definition": (
        "Whether what the tutor taught across the dialogue is factually correct. An extension to "
        "the proposal's four macro dimensions, not one of them."
    ),
    "question": (
        "Across the dialogue as a whole, is what the tutor TAUGHT factually correct — its "
        "definitions, procedures, worked results, and claims about the subject matter? Judge the "
        "correctness of the tutor's own statements about the subject. Do not judge teaching style, "
        "how much of the answer the tutor revealed, how well the tutor explained, or whether the "
        "student understood. A tutor that teaches well and states something false about the "
        "subject scores low here; a tutor that teaches poorly but says nothing false scores high."
    ),
    "applicability": (
        "Use null only when the tutor makes no substantive subject-matter claim anywhere in the "
        "dialogue. Any dialogue in which the tutor teaches something is applicable."
    ),
}

MACRO_DIMENSIONS = list(_V1_DIMENSIONS) + [FACTUAL_CORRECTNESS]
MACRO_DIMENSION_IDS = [d["id"] for d in MACRO_DIMENSIONS]

MACRO_OUTPUT_SCHEMA = {
    "dialogue_id": "string, the dialogue this macro judgment covers",
    "dimension_scores": {d["id"]: "0 | 0.5 | 1 | null" for d in MACRO_DIMENSIONS},
    "dimension_applicability": {
        d["id"]: "applicable | not_applicable | unclear" for d in MACRO_DIMENSIONS
    },
    "evidence_pointers": [
        {
            "dimension": "one of the %d macro dimension ids" % len(MACRO_DIMENSIONS),
            "exchange_id": "the exchange_id this evidence comes from",
            "quote": "short verbatim quote from that exchange",
        }
    ],
    "rationale_short": (
        "20-1400 chars, no hidden chain-of-thought, one paragraph per dimension is fine"
    ),
}
