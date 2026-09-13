"""multilabel_kc_v17.py

v19 multi-label annotation for the RRF-based pipeline.

Why v16 needed a version bump
-----------------------------
v16 decided secondary KCs by comparing additive `composite_score` ratios to
the dominant candidate. Under v19, additive composite scores are retained for
audit only; the live ranking semantics are now:

- fused source-family rank (plus CE rerank when active)
- typed eligibility / corroboration gates

So v19 secondaries should follow the FINAL ranked order and the same typed
eligibility notion as the dominant answer, rather than a ratio over a score
whose cardinal interpretation was intentionally retired.
"""

from __future__ import annotations

from typing import Any


MAX_SECONDARIES = 3


def _candidate_by_unit_id(pool: dict[str, Any], unit_id: str | None) -> dict[str, Any] | None:
    if not unit_id:
        return None
    for candidate in pool.get("candidates") or []:
        if candidate.get("unit_id") == unit_id:
            return candidate
    return None


def _pool_rank(pool: dict[str, Any], unit_id: str) -> int | None:
    for i, candidate in enumerate(pool.get("candidates") or [], start=1):
        if candidate.get("unit_id") == unit_id:
            return i
    return None


def _entry_for_candidate(candidate: dict[str, Any], role: str, pool: dict[str, Any]) -> dict[str, Any]:
    return {
        "kc_id": candidate.get("unit_id"),
        "kc_name": candidate.get("canonical_name"),
        "role": role,
        "final_rank_score": round(float(candidate.get("final_rank_score") or 0.0), 6),
        "rank": _pool_rank(pool, candidate.get("unit_id") or ""),
        "candidate_state": candidate.get("candidate_state"),
        "source_family_count": int(candidate.get("source_family_count") or 0),
    }


def resolved_kc_ids_for_row(row: dict[str, Any], pool: dict[str, Any]) -> list[dict[str, Any]]:
    dominant_id = row.get("resolved_kc_id")
    if not dominant_id:
        return []

    dominant_cand = _candidate_by_unit_id(pool, dominant_id)
    if not dominant_cand:
        return []

    entries: list[dict[str, Any]] = [_entry_for_candidate(dominant_cand, "dominant", pool)]

    secondaries = 0
    for candidate in pool.get("candidates") or []:
        uid = candidate.get("unit_id")
        if not uid or uid == dominant_id:
            continue
        if not bool(candidate.get("eligible_for_dominant")):
            continue
        entries.append(_entry_for_candidate(candidate, "secondary", pool))
        secondaries += 1
        if secondaries >= MAX_SECONDARIES:
            break

    return entries


def annotate_resolved_kc_ids_v17(resolved: list[dict[str, Any]], pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row, pool in zip(resolved, pools):
        row["resolved_kc_ids"] = resolved_kc_ids_for_row(row, pool)
    return resolved
