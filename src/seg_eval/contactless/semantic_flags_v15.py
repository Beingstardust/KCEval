"""semantic_flags_v15.py

Phase 3 (v15): advisory low-confidence flag.

Design note -- why this is advisory-only, not a hard reject
-------------------------------------------------------------
The brief asked for a floor derived from the composite_score distribution of
confirmed-good vs confirmed-bad top-1 picks. That distribution was checked
across all 63 exchanges (see data/gold/v15_phase3_score_distribution_audit.csv)
and it does NOT separate cleanly:

  - Confirmed-WRONG top-1 picks include the highest composite scores in the
    entire dataset (54.3, 53.7, 53.3, 49.8, 42.1 -- ex_0010, ex_0052, ex_0042,
    ex_0038, ex_0039). These are not weak guesses; they are confident
    sibling-KC confusions where the wrong specific concept within the right
    branch outscored the right one (e.g. ex_0010 resolves NB Classification
    Phase instead of Prior Probability -- both are Naive Bayes KCs, both
    score highly, only one is what the exchange is actually about).
  - Confirmed-CORRECT top-1 picks go as low as composite_score=3.46
    (ex_0012, candidate_state=reject, identity_component=0.0) -- i.e. a
    genuinely correct answer with the weakest possible intrinsic evidence
    profile.

Because a composite_score floor would necessarily sit between these two
overlapping ranges, any threshold either (a) fails to catch the majority of
wrong picks (the confident sibling confusions, which this data shows are
10 of 17 wrong top-1 picks) or (b) flags a validated Phase-1 win (ex_0012)
as low-confidence.

Given that trade-off, this flag is scoped narrowly and kept advisory:
it targets only the cleanly-separable "weak evidence" cluster --
candidate_state in {reject, lexical_only} AND identity_component == 0.0 AND
support_component == 0.0 -- and it never changes resolved_kc_id, final_label,
or score. It only adds `low_confidence_no_strong_match: true` plus a reason
string, for human review and reporting. ex_0012 is expected to fire this flag
despite being correct; that is reported explicitly rather than hidden, per
the instruction to check the guard against gold in both directions.

The dominant wrong-pick failure mode (confident sibling-KC confusion) is
NOT addressed by this flag. It would need branch-aware disambiguation
(comparing top1 vs top2 within the same branch/sibling set), which is a
distinct mechanism this phase does not attempt to design or validate.
"""

from __future__ import annotations

from typing import Any


LOW_CONFIDENCE_FLAG = "low_confidence_no_strong_match"


def _candidate_by_unit_id(pool: dict, unit_id: str | None) -> dict | None:
    if not unit_id:
        return None
    for candidate in pool.get("candidates") or []:
        if candidate.get("unit_id") == unit_id:
            return candidate
    return None


def annotate_low_confidence_v15(resolved: list[dict[str, Any]], pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row, pool in zip(resolved, pools):
        resolved_id = row.get("resolved_kc_id")
        if not resolved_id:
            row[LOW_CONFIDENCE_FLAG] = False
            continue

        candidate = _candidate_by_unit_id(pool, resolved_id)
        if not candidate:
            row[LOW_CONFIDENCE_FLAG] = False
            continue

        state = candidate.get("candidate_state")
        identity = float(candidate.get("identity_component") or 0.0)
        support = float(candidate.get("support_component") or 0.0)

        weak = state in {"reject", "lexical_only"} and identity == 0.0 and support == 0.0
        row[LOW_CONFIDENCE_FLAG] = weak
        if weak:
            row.setdefault("resolution_reasons", [])
            row["resolution_reasons"].append(
                f"{LOW_CONFIDENCE_FLAG}: candidate_state={state} identity_component=0.0 support_component=0.0"
            )

    return resolved
