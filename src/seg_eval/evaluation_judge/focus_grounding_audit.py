"""Deterministic, post-hoc focus-grounding audit -- computed from data already available,
never generated or echoed by the judge.

WHY THIS EXISTS, AND WHY IT REPLACED AN EARLIER APPROACH
----------------------------------------------------------
An earlier version added a new required response field (`focus_grounding`) asking the judge to
echo `focus_target.target_type` and self-report whether it anchored its correctness check to
`focus_target.focus_span`. That measurably destabilized the response contract (0/52 -> 8/52
failures on an unrelated field, from schema-ordering sensitivity -- see
local_audits/evaluation_completion_20260824T093426Z/09_FOCUS_TARGET_IMPLEMENTATION.md) for a
field that didn't need to be model-generated at all:

- `target_type` is already known -- it's deterministic packet data, computed once by
  focus_target_v1.py before the judge ever runs. Asking the judge to echo it back adds nothing;
  it's simply read from the packet directly here.
- "Did the judge anchor its correctness check to the focus span" is better answered by checking
  whether the judge's OWN `evidence_quotes` actually overlap the focus span, than by asking the
  judge to self-report a boolean. A self-report can be wrong or unconsidered; an overlap check
  against the judge's own cited evidence is a real, verifiable signal computed from what the
  judge already produced under the unmodified response contract.

So this module takes a packet and an ALREADY-VALIDATED judge response (validated against the
plain, unmodified response_contract_v2 -- no new required field) and computes audit metadata
entirely outside the response JSON. It never blocks or fails validation; it can be computed, or
skipped, independent of whether the judge's response happens to satisfy the core contract.
"""

from __future__ import annotations

from typing import Any

from .response_contract_v2 import normalize_for_quote_match


def compute_focus_grounding_audit(
    packet: dict[str, Any], judge_response: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic audit record: what the packet's focus_target says, and whether the judge's
    own evidence_quotes actually draw from it.

    ``computed_anchored_to_focus_span`` is True iff at least one of the judge's evidence_quotes
    is a (normalized) substring of focus_span -- i.e. the judge cited something that was, in
    fact, inside the identified focus target. This is meaningful only when target_type is
    precise_focus_target; for broad/no_domain/insufficient targets, focus_span may be the whole
    tutor content or empty, so the flag is still computed (for completeness) but should be read
    as "quoted from within the broad target" rather than "anchored precisely".
    """
    focus_target = packet.get("focus_target") or {}
    target_type = focus_target.get("target_type")
    pattern = focus_target.get("pattern")
    focus_span = focus_target.get("focus_span") or ""
    focus_span_norm = normalize_for_quote_match(focus_span)

    quotes = judge_response.get("evidence_quotes") or []
    matched_quotes: list[str] = []
    if focus_span_norm:
        for q in quotes:
            if not isinstance(q, str):
                continue
            q_norm = normalize_for_quote_match(q)
            if q_norm and q_norm in focus_span_norm:
                matched_quotes.append(q)

    return {
        "target_type": target_type,
        "pattern": pattern,
        "focus_span_present": bool(focus_span_norm),
        "computed_anchored_to_focus_span": bool(matched_quotes),
        "matched_evidence_quotes": matched_quotes,
        "method": (
            "deterministic: at least one evidence_quotes entry is a substring of "
            "packet.focus_target.focus_span. Not model-reported."
        ),
    }
