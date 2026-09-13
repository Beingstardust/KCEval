"""KC-grounding ablation conditions (execution spec §29).

Three conditions, differing ONLY in how much curriculum grounding reaches the judge. Everything
else -- rubric, response contract, model, revision, decoding, segmentation, KC assignment,
focus_target layer -- is held fixed.

    A  ungrounded      No KC context at all. primary_kc=None, evaluator_kc_set=[]. The floor:
                       what does the judge score on segment text + rubric alone?
    B  baseline        Exactly what the deployed pipeline sends today: KC identity, definition,
                       matching cues, sibling contrast notes. This is retrieval scaffolding --
                       it says which KC this is, not what the tutor should have taught.
    C  curriculum      B + the reviewed library's `evaluation_support` block:
       grounded        what_tutor_should_explain, common_confusions_or_errors, red_flags,
                       acceptable_teaching_moves.

WHY C IS THE INTERESTING CONDITION
----------------------------------
`evaluation_support` is richly populated in the reviewed library (335 what_tutor_should_explain
entries, 197 common_confusions_or_errors, 184 red_flags, 229 acceptable_teaching_moves -- 945
total across 138-159 of 159 KCs) and reaches the judge NOWHERE today. The cause is structural,
not a decision: `evaluation_packets/builder.py:compact_profile()` reads the *runtime matching
profile* (data/processed/matching_profiles/.../frozen_matching_profiles.jsonl), which is a
stripped derivative built for retrieval and does not carry these fields. Only the reviewed
library (data/input/frozen_library_incoming/kc_library_reviewed_v2/.../frozen_reviewed_library.jsonl)
has them. Condition C joins that file back in, on kc_id <-> unit_id (verified 159/159 exact,
no orphans on either side).

ABSTAINED UNITS ARE PASSED THROUGH, NOT FILTERED
------------------------------------------------
21 of 159 KCs have scope_status="abstained". Their `what_tutor_should_explain` is empty and
their `red_flags` carries an explicit self-guard ("Do not evaluate tutor correctness from this
machine draft alone; no target-bound source evidence supported..."). That guard is real,
useful calibration signal -- it tells the judge not to treat that KC as curriculum truth -- so
it is passed through verbatim rather than stripped. Filtering it would hide from the judge
exactly the uncertainty the library reviewers took care to record.

KNOWN DATA-QUALITY CAVEAT, DELIBERATELY NOT CLEANED
---------------------------------------------------
A lexical-overlap scan of the 138 non-abstained KCs found 2 (~1.5%) whose evaluation_support
content appears to be about a different topic than the KC it is attached to (KC_CLF_DT_007
"Split Information", KC_EVAL_BASIC_008 "MAE for Ordinal Targets"); at least one more
(KC_CLU_HIER_002 "Agglomerative Clustering", carrying FP-growth content) escapes that scan
because it shares surface vocabulary. These are NOT filtered out. Hand-cleaning the input to
make a grounding condition look better would be tuning the experiment to its desired result --
the ablation's question is whether THIS library, as it actually exists, helps. The caveat is
recorded here and in the ablation report instead.
"""

from __future__ import annotations

from ..corpus_config import kc_library_path

import json
from pathlib import Path
from typing import Any

GROUNDING_CONDITIONS = ("A_ungrounded", "B_baseline", "C_curriculum_grounded")

DEFAULT_CONDITION = "B_baseline"

# The four evaluation_support fields, in a fixed order so prompt rendering is deterministic.
EVALUATION_SUPPORT_FIELDS = (
    "what_tutor_should_explain",
    "common_confusions_or_errors",
    "red_flags",
    "acceptable_teaching_moves",
)

# Domain-specific INPUT, not framework logic. Resolved through corpus_config so a
# different subject domain is a config change, never a source edit. See
# seg_eval.corpus_config for the override mechanism.
REVIEWED_LIBRARY_PATH = kc_library_path()


def load_reviewed_library_grounding(path: str | Path) -> dict[str, dict[str, Any]]:
    """kc_id -> {evaluation_support fields, scope_status}, read from the reviewed library.

    Returns only what condition C is allowed to add. Deliberately does NOT return
    kc_specific_criteria (empty on all 159 KCs -- the real draft content lives in
    reviewer_criteria at 0% expert-approved, so surfacing it as if it were curriculum truth
    would be dishonest) or review/provenance metadata (not curriculum content).
    """
    path = Path(path)
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            kc_id = row.get("kc_id")
            if not kc_id:
                continue
            support = row.get("evaluation_support") or {}
            entry: dict[str, Any] = {"scope_status": row.get("scope_status")}
            for field in EVALUATION_SUPPORT_FIELDS:
                values = support.get(field)
                if isinstance(values, list) and values:
                    entry[field] = list(values)
            out[str(kc_id)] = entry
    return out


def grounding_block_for_kc(kc_id: str, grounding_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The `curriculum_grounding` block condition C attaches to one KC, or None if there is
    nothing to attach (no entry, or an entry with scope_status but no actual content)."""
    entry = grounding_index.get(kc_id)
    if not entry:
        return None

    block: dict[str, Any] = {}
    for field in EVALUATION_SUPPORT_FIELDS:
        if entry.get(field):
            block[field] = entry[field]

    if not block:
        return None

    scope_status = entry.get("scope_status")
    if scope_status:
        block["scope_status"] = scope_status
    if scope_status == "abstained":
        # Make the abstention explicit rather than relying on the judge to infer it from a
        # red_flags string it may or may not read carefully.
        block["grounding_caution"] = (
            "This KC is marked abstained in the reviewed curriculum library: it has no "
            "source-grounded teaching content. Do not treat this block as curriculum truth."
        )
    return block


def condition_shows_kc_context(condition: str) -> bool:
    """Condition A sends no KC context at all; B and C both do."""
    _require_valid(condition)
    return condition != "A_ungrounded"


def condition_adds_curriculum_grounding(condition: str) -> bool:
    """Only condition C adds the reviewed library's evaluation_support block."""
    _require_valid(condition)
    return condition == "C_curriculum_grounded"


def _require_valid(condition: str) -> None:
    if condition not in GROUNDING_CONDITIONS:
        raise ValueError(
            f"unknown grounding condition {condition!r}; expected one of {GROUNDING_CONDITIONS}"
        )
