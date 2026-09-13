"""abstention_gate_v24.py

Task 3: a real, structural abstention state -- resolution_status:
"resolved" | "abstained_low_confidence" -- distinct from the existing
advisory-only `low_confidence_no_strong_match` flag (semantic_flags_v15.py),
which never changes output. This module does not change resolved_kc_id; an
abstained exchange still carries its best-guess candidate for audit, marked
as structurally not a confident resolution.

Investigation (done before designing anything, per the task's own
requirement)
----------------------------------------------------------------------------
Checked whether the typed-eligibility infrastructure built for v19-v22
(source_family_count, has_profile_native_corroboration, has_lexical_support,
has_dense_support, candidate_state, cross-encoder/RRF rank) separates
correct from wrong top candidates. It does not, and the reason is
structural, not a tuning failure:

1. Boolean family-diversity signals are saturated. Checked against the
   labeled disagreement set (20 instances drawn from the semantic audits
   already on record across v19/v21/v22 -- 8 "correct", 12 "wrong", data:
   data/gold/task3_phase1_labeled_typed_eligibility.csv) and separately
   against 12 documented curriculum-gap exchanges vs. 40 exchanges with no
   recorded disagreement (data/gold/task3_phase1_gap_vs_confident.csv):
   `has_profile_native_corroboration`, `has_lexical_support`,
   `has_dense_support`, `eligible_for_dominant` are ~92-100% True in every
   bucket, including genuine curriculum gaps. Hierarchical RRF fusion is
   designed to always surface "the best available" candidate as locally
   multi-family-corroborated -- that is a feature for ranking, but it makes
   these particular booleans structurally unable to tell "good match" from
   "best of a bad lot."
2. Rank-based signals (outer_rrf_rank, per-family outer rank) are also
   uninformative for the WINNING candidate specifically, for the same
   reason: the winner is close to rank 1 in at least one family almost by
   definition of having won.
3. Within the labeled disagreement set specifically (the 20-row check
   above), NO signal checked -- including identity_component, exact_override,
   candidate_state -- separates "correct" from "wrong." This is expected,
   not a failure: disagreement cases are, by construction, the closely
   matched boundary region where both sides already cleared the same
   eligibility bars in their respective winning runs. The typed-eligibility
   framework was not expected to resolve genuinely close semantic calls, and
   it does not.
4. A different, cleaner signal DOES separate genuine curriculum gaps (no
   correct KC exists at all -- 12 exchanges, list already documented in
   v13_final_scoring_report.md, not a fresh gold lookup performed this
   session) from the 40 exchanges with no recorded disagreement: no
   individual field is clean, but the conjunction of three absolute
   (non-rank) signals is:
   - `exact_override_candidate` is False for 12/12 gap exchanges vs. False
     for only 21/40 non-gap exchanges (i.e. True for 19/40 non-gap, 0/12 gap
     -- a clean one-directional signal: True always means not-a-gap).
   - `identity_component == 0` for 11/12 gap exchanges vs. 17/40 non-gap
     (nonzero identity is a clean one-directional not-a-gap signal, one
     exception: ex_0029, a documented gap that a separate blind semantic
     audit still judged a reasonable KC match -- a genuine, disclosed edge
     case, not hidden).
   - `char_score` (character n-gram surface-similarity, the one absolute,
     non-rank lexical-family value checked) below the corpus-wide median
     among all resolved winners (`0.3849`) for 10/12 gap exchanges vs. 8/40
     non-gap.
   Combined with AND: flags 10/12 (83%) of documented gaps, and 8/40 (20%)
   of exchanges with no recorded disagreement. Checked those 8 "false"
   flags individually rather than accepting the raw rate: 3 of the 8
   (ex_0007, ex_0009, ex_0039) are not false positives at all -- they are
   already-confirmed-wrong resolutions from this session's own prior,
   separately-established findings (the Bayes'-Theorem segmentation-
   fragmentation cluster documented in v15/v17/v18's reports, and the
   ex_0039 short-token collision this same session's Task 2 confirmed is
   NOT fixed by the matcher change at the final-resolution level). The
   corrected false-abstention rate against exchanges with no independent
   evidence of being wrong is 5/37 (13.5%).

This is the best available approximation, not a confident, fully-validated
classifier -- reported as such. It is deliberately conservative (requires
all three conditions, not just one) precisely because the individual
signals are weak and overlapping; a looser OR-based rule was tried first
and produces an unacceptable ~42% false-abstention rate (see the module's
git-tracked development history / task report for that intermediate
result). Domain-generic by construction: `exact_override_candidate`,
`identity_component`, and `char_score` are structural properties already
computed for every candidate in every domain this pipeline could be pointed
at; no term, KC name, or course vocabulary is referenced anywhere in this
gate.
"""

from __future__ import annotations

from typing import Any


# Corpus-derived from the full 62-exchange (excluding the one meta-excluded
# exchange) v23 run's resolved-winner char_score distribution -- the median
# among ALL resolved winners, not fit to any known-gap or known-wrong
# exchange ID specifically.
CHAR_SCORE_WEAK_THRESHOLD = 0.3849


def _winning_candidate(pool: dict[str, Any], resolved_kc_id: str | None) -> dict[str, Any] | None:
    if not resolved_kc_id:
        return None
    for c in pool.get("candidates") or []:
        if c.get("unit_id") == resolved_kc_id:
            return c
    return None


def _is_low_confidence_gap_signature(candidate: dict[str, Any]) -> bool:
    no_exact_override = not bool(candidate.get("exact_override_candidate"))
    no_identity = float(candidate.get("identity_component") or 0.0) == 0.0
    weak_char = float(candidate.get("char_score") or 0.0) < CHAR_SCORE_WEAK_THRESHOLD
    return no_exact_override and no_identity and weak_char


def annotate_abstention_v24(resolved: list[dict[str, Any]], pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row, pool in zip(resolved, pools):
        resolved_kc_id = row.get("resolved_kc_id")
        if not resolved_kc_id:
            # NON_KC / recap rows: no candidate to evaluate, not part of this gate.
            row["resolution_status"] = "resolved"
            row["abstention_signals"] = []
            continue

        candidate = _winning_candidate(pool, resolved_kc_id)
        if candidate is None:
            row["resolution_status"] = "resolved"
            row["abstention_signals"] = []
            continue

        if _is_low_confidence_gap_signature(candidate):
            row["resolution_status"] = "abstained_low_confidence"
            row["abstention_signals"] = [
                "no_exact_canonical_override",
                "zero_profile_native_identity_component",
                f"char_score_below_corpus_median_{CHAR_SCORE_WEAK_THRESHOLD}",
            ]
            # Best-guess candidate is retained unchanged in resolved_kc_id /
            # resolved_kc_name for audit -- this gate never mutates those
            # fields, only adds the status/signal annotation.
        else:
            row["resolution_status"] = "resolved"
            row["abstention_signals"] = []

    return resolved
