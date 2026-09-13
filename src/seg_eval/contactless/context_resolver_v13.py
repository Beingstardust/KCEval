"""context_resolver_v13.py

v22 pass-2 resolver for typed-eligible, deterministic RRF-ranked pools.

Relative-strength continuation from v11 is retained, but all absolute
additive-score assumptions are removed. Candidate choice now prioritizes
typed-eligible rows instead of raw `candidates[0]`.

v13 refines the broad-anchor tutor-only noncanonical guard: do not demote the
top candidate when it is CE-top, typed-eligible, candidate-state `candidate`,
and corroborated by all three source families.
"""

from __future__ import annotations

from typing import Any


CONTINUATION_PROTECTION_RULE = "pass1_protected_from_continuation_relative_evidence_v19"


def _candidate_ids(pool: dict[str, Any], limit: int = 8) -> list[str]:
    return [c["unit_id"] for c in (pool.get("candidates") or [])[:limit]]


def _candidate_by_unit_id(pool: dict[str, Any], unit_id: str | None) -> dict | None:
    if not unit_id:
        return None
    for candidate in pool.get("candidates") or []:
        if candidate.get("unit_id") == unit_id:
            return candidate
    return None


def _branch(pool: dict[str, Any]) -> str:
    candidates = pool.get("candidates") or []
    return candidates[0].get("branch") if candidates else ""


def _best_candidate(pool: dict[str, Any]) -> dict[str, Any] | None:
    candidates = pool.get("candidates") or []
    for cand in candidates:
        if bool(cand.get("eligible_for_dominant")):
            return cand
    return candidates[0] if candidates else None


def _is_raw_ce_top(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    ce = candidate.get("cross_encoder_score")
    if ce is None:
        return False
    ce_scores = [
        float(c.get("cross_encoder_score"))
        for c in candidates
        if c.get("cross_encoder_score") is not None
    ]
    return bool(ce_scores) and float(ce) >= max(ce_scores)


def _has_all_source_families(candidate: dict[str, Any]) -> bool:
    flags = candidate.get("family_support_flags") or {}
    return (
        bool(flags.get("profile_native"))
        and bool(flags.get("lexical"))
        and bool(flags.get("dense"))
        and int(candidate.get("source_family_count") or 0) >= 3
    )


def _strong_tutor_only_broad_anchor(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    return bool(
        candidate.get("tutor_only_noncanonical_identity")
        and bool(candidate.get("eligible_for_broad"))
        and candidate.get("candidate_state") == "candidate"
        and _has_all_source_families(candidate)
        and _is_raw_ce_top(candidate, candidates)
    )


def _best_candidate_for_broad(pool: dict[str, Any]) -> dict | None:
    candidates = pool.get("candidates") or []
    if candidates and _strong_tutor_only_broad_anchor(candidates[0], candidates):
        return candidates[0]
    for cand in candidates:
        if cand.get("tutor_only_noncanonical_identity"):
            continue
        if bool(cand.get("eligible_for_broad")):
            return cand
    for cand in candidates:
        if cand.get("tutor_only_noncanonical_identity"):
            continue
        if bool(cand.get("has_profile_native_corroboration")):
            return cand
    return _best_candidate(pool)


def _nearest_anchor(resolved: list[dict[str, Any]], idx: int, direction: int, max_distance: int = 3) -> dict[str, Any] | None:
    j = idx + direction
    steps = 0
    while 0 <= j < len(resolved) and steps < max_distance:
        row = resolved[j]
        if row.get("final_label") in {"AUTO_HIGH_KC", "AUTO_MEDIUM_KC"} and row.get("resolved_kc_id"):
            return row
        if row.get("final_label") == "MISSING_KC_REVIEW":
            return None
        j += direction
        steps += 1
    return None


def _same_or_supported_by_anchor(anchor: dict[str, Any], pool: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    target = anchor.get("resolved_kc_id")
    if not target:
        return False, reasons

    if target in _candidate_ids(pool, limit=8):
        reasons.append("anchor_kc_in_candidate_pool")

    target_cand = _candidate_by_unit_id(pool, target)
    if target_cand:
        if bool(target_cand.get("eligible_for_dominant")):
            reasons.append("anchor_kc_typed_eligible_locally")
        if bool(target_cand.get("has_profile_native_corroboration")):
            reasons.append("anchor_kc_profile_corroborated_locally")
        if target_cand.get("branch") and target_cand.get("branch") == anchor.get("branch"):
            reasons.append("anchor_branch_match")

    best = _best_candidate(pool)
    if best and best.get("branch") and best.get("branch") == anchor.get("branch"):
        reasons.append("best_candidate_same_branch_as_anchor")

    return bool(reasons), reasons


def _ranking_score(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    return float(candidate.get("final_rank_score") or candidate.get("ce_reranked_composite") or candidate.get("composite_score") or 0.0)


def _relative_continuation_permitted(row: dict, pool: dict, anchor_kc_id: str | None) -> tuple[bool, list[str]]:
    provisional_id = row.get("provisional_kc_id")
    pass1_score = float(row.get("top1_score") or 0.0)
    pass1_candidate = _candidate_by_unit_id(pool, provisional_id)
    pass1_ce = pass1_candidate.get("cross_encoder_score") if pass1_candidate else None

    anchor_candidate = _candidate_by_unit_id(pool, anchor_kc_id)
    if not anchor_candidate:
        return False, [f"continuation_blocked_anchor_kc_absent_from_own_pool anchor_kc={anchor_kc_id}"]

    anchor_score = _ranking_score(anchor_candidate)
    anchor_ce = anchor_candidate.get("cross_encoder_score")

    if anchor_score <= pass1_score:
        return False, [
            f"continuation_blocked_anchor_score_not_above_pass1 "
            f"anchor_score={anchor_score:.6f} pass1_score={pass1_score:.6f}"
        ]

    if pass1_ce is not None and anchor_ce is not None and float(anchor_ce) < float(pass1_ce):
        return False, [
            f"continuation_blocked_anchor_ce_below_pass1 "
            f"anchor_ce={float(anchor_ce):.6f} pass1_ce={float(pass1_ce):.6f}"
        ]

    return True, [
        f"continuation_permitted_anchor_stronger_than_pass1 "
        f"anchor_score={anchor_score:.6f} pass1_score={pass1_score:.6f}"
    ]


def _protect_from_pass1(row: dict, reasons: list[str]) -> dict:
    row["final_label"] = "AUTO_LOW_KC"
    row["resolved_kc_id"] = row.get("provisional_kc_id")
    row["resolved_kc_name"] = row.get("provisional_kc_name")
    row["confidence_band"] = row.get("confidence_hint") or "low"
    row["resolution_action"] = CONTINUATION_PROTECTION_RULE
    row["rule_fired"] = CONTINUATION_PROTECTION_RULE
    row["resolution_reasons"] = list(row.get("decision_reasons") or []) + reasons
    return row


def _novelty_score(pool: dict[str, Any], left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    candidates = pool.get("candidates") or []
    typed_best = next((cand for cand in candidates if bool(cand.get("eligible_for_dominant"))), None)
    best = _best_candidate(pool)
    best_score = _ranking_score(best)
    branch_scores = pool.get("branch_scores") or {}
    top_branch_score = max(branch_scores.values()) if branch_scores else 0.0
    top_branch_share = float(pool.get("top_branch_share") or 0.0)

    neighbour_support = False
    if left:
        ok, _ = _same_or_supported_by_anchor(left, pool)
        neighbour_support = neighbour_support or ok
    if right:
        ok, _ = _same_or_supported_by_anchor(right, pool)
        neighbour_support = neighbour_support or ok

    score = 0.0
    reasons = []

    if not typed_best:
        score += 0.45
        reasons.append("no_typed_eligible_candidate")
    elif best:
        if not bool(best.get("has_profile_native_corroboration")) and int(best.get("source_family_count") or 0) < 2:
            score += 0.25
            reasons.append("best_candidate_weakly_supported")

    if top_branch_share < 0.25:
        score += 0.25
        reasons.append("weak_branch_support")
    if not neighbour_support:
        score += 0.20
        reasons.append("no_neighbour_rescue_support")
    if len(candidates) <= 2:
        score += 0.15
        reasons.append("shallow_candidate_pool")

    return {
        "novelty_score": round(min(1.0, score), 6),
        "novelty_reasons": reasons,
        "best_candidate_score": round(best_score, 6),
        "top_branch_score": round(top_branch_score, 6),
        "top_branch_share": round(top_branch_share, 6),
        "neighbour_support": neighbour_support,
    }


def _make_final_from_pass1_v12(row: dict, pool: dict) -> dict:
    label = row["pass1_label"]
    if label == "NON_KC":
        final = "NON_KC"
    elif label == "AUTO_HIGH_KC":
        final = "AUTO_HIGH_KC"
    elif label == "AUTO_MEDIUM_KC":
        final = "AUTO_MEDIUM_KC"
    else:
        final = "UNRESOLVED_AFTER_PASS1"

    return {
        **row,
        "final_label": final,
        "resolved_kc_id": row.get("provisional_kc_id") if final != "NON_KC" else None,
        "resolved_kc_name": row.get("provisional_kc_name") if final != "NON_KC" else None,
        "confidence_band": row.get("confidence_hint"),
        "resolution_action": "pass1_closed_set" if final.startswith("AUTO_") else "pending_pass2",
        "resolution_reasons": list(row.get("decision_reasons") or []),
        "review_required": False,
        "review_reasons": [],
        "branch": row.get("branch") or ((pool.get("candidates") or [{}])[0].get("branch") or ""),
        "kc_set": [c["unit_id"] for c in (pool.get("candidates") or [])[:6]],
    }


def resolve_pass2_v13(pass1_rows: list[dict], pools: list[dict]) -> tuple[list[dict], list[dict]]:
    resolved = [_make_final_from_pass1_v12(row, pool) for row, pool in zip(pass1_rows, pools)]

    for idx, row in enumerate(list(resolved)):
        if row["final_label"] != "UNRESOLVED_AFTER_PASS1":
            continue

        pool = pools[idx]
        best = _best_candidate(pool)
        left = _nearest_anchor(resolved, idx, -1)
        right = _nearest_anchor(resolved, idx, 1)
        broad = bool(pool.get("broad_multibranch_signal")) or bool(row.get("broad_multibranch_signal"))

        if not best:
            row.update(_novelty_score(pool, left, right))
            continue

        if broad and float(pool.get("top_branch_share") or 0.0) < 0.45:
            chosen = _best_candidate_for_broad(pool) or best
            row["final_label"] = "AUTO_LOW_KC"
            row["resolved_kc_id"] = chosen["unit_id"]
            row["resolved_kc_name"] = chosen["canonical_name"]
            row["confidence_band"] = "low"
            row["resolution_action"] = "pass2_broad_multibranch_closed_set"
            row["resolution_reasons"] = ["broad_multibranch_library_support", "typed_closed_set_low_confidence_assignment"]
            if chosen is not best:
                row["resolution_reasons"].append("broad_anchor_avoided_tutor_only_noncanonical_top_candidate")
            elif _strong_tutor_only_broad_anchor(chosen, pool.get("candidates") or []):
                row["resolution_reasons"].append("broad_anchor_retained_strong_tutor_only_noncanonical_top_candidate")
            row["branch"] = chosen.get("branch") or row.get("branch")
            continue

        if left and right and left.get("resolved_kc_id") == right.get("resolved_kc_id"):
            ok, reasons = _same_or_supported_by_anchor(left, pool)
            if ok:
                row["final_label"] = "AUTO_LOW_KC"
                row["resolved_kc_id"] = left["resolved_kc_id"]
                row["resolved_kc_name"] = left["resolved_kc_name"]
                row["confidence_band"] = "low"
                row["resolution_action"] = "pass2_same_neighbour_rescue"
                row["resolution_reasons"] = ["left_right_same_anchor"] + reasons
                row["branch"] = left.get("branch") or row.get("branch") or best.get("branch")
                continue

        anchor_scores = []
        for name, anchor in [("left", left), ("right", right)]:
            if not anchor:
                continue
            ok, reasons = _same_or_supported_by_anchor(anchor, pool)
            if ok:
                anchor_scores.append((name, anchor, reasons))

        if anchor_scores and not broad:
            name, anchor, reasons = anchor_scores[0]
            anchor_kc_id = anchor.get("resolved_kc_id")
            permitted, guard_reasons = _relative_continuation_permitted(row, pool, anchor_kc_id)

            if not permitted:
                _protect_from_pass1(row, guard_reasons)
                row["branch"] = row.get("branch") or best.get("branch")
                continue

            row["final_label"] = "AUTO_LOW_KC"
            row["resolved_kc_id"] = anchor["resolved_kc_id"]
            row["resolved_kc_name"] = anchor["resolved_kc_name"]
            row["confidence_band"] = "low"
            row["resolution_action"] = f"pass2_{name}_context_continuation"
            row["resolution_reasons"] = reasons + guard_reasons
            row["branch"] = anchor.get("branch") or best.get("branch") or row.get("branch")
            continue

        if best and (
            bool(best.get("eligible_for_dominant"))
            or bool(best.get("has_profile_native_corroboration"))
            or float(pool.get("top_branch_share") or 0.0) >= 0.30
        ):
            row["final_label"] = "AUTO_LOW_KC"
            row["resolved_kc_id"] = best["unit_id"]
            row["resolved_kc_name"] = best["canonical_name"]
            row["confidence_band"] = "low"
            row["resolution_action"] = "pass2_branch_guided_best_candidate"
            row["resolution_reasons"] = ["best_available_typed_library_candidate", "not_enough_evidence_for_missing_kc"]
            row["branch"] = best.get("branch") or row.get("branch")
            continue

        novelty = _novelty_score(pool, left, right)
        row.update(novelty)

        if novelty["novelty_score"] >= 0.70:
            row["final_label"] = "MISSING_KC_REVIEW"
            row["resolved_kc_id"] = None
            row["resolved_kc_name"] = None
            row["confidence_band"] = "missing_review"
            row["resolution_action"] = "novelty_gate_missing_kc_review"
            row["resolution_reasons"] = novelty["novelty_reasons"]
            row["review_required"] = True
            row["review_reasons"] = ["probable_missing_kc_after_pass2_failure"]
        else:
            row["final_label"] = "AUTO_LOW_KC"
            row["resolved_kc_id"] = best["unit_id"]
            row["resolved_kc_name"] = best["canonical_name"]
            row["confidence_band"] = "low"
            row["resolution_action"] = "pass2_low_confidence_closed_set_fallback"
            row["resolution_reasons"] = ["novelty_gate_did_not_fire", "typed_closed_set_assignment_preferred"]

    novelty_rows = []
    for idx, row in enumerate(resolved):
        left = _nearest_anchor(resolved, idx, -1)
        right = _nearest_anchor(resolved, idx, 1)
        novelty = _novelty_score(pools[idx], left, right)
        for key, value in novelty.items():
            row.setdefault(key, value)
        novelty_rows.append(
            {
                "dialogue_id": row.get("dialogue_id"),
                "exchange_id": row.get("exchange_id"),
                "final_label": row.get("final_label"),
                "resolved_kc_id": row.get("resolved_kc_id"),
                **novelty,
            }
        )

    return resolved, novelty_rows
