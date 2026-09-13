"""local_assigner_v13.py

v19 pass-1 assignment logic for the RRF-based candidate pool.

Key change from v12:
- pass 1 no longer trusts raw additive composite scores
- it chooses from typed-eligible candidates first
- confidence uses exact override, corroboration, family diversity, and rank
  separation rather than `signal_agreement_count`

The non-KC / meta / recap gates are intentionally retained unchanged from
v12 because they are orthogonal to the retrieval architecture change.
"""

from __future__ import annotations

import re
from typing import Any


LOW_CONTENT_FLAGS = {"empty_or_near_empty", "acknowledgement_only", "greeting_only", "low_content_short"}

META_FRAMING_CUES = [
    r"feel like a conversation",
    r"real human student",
    r"respond like an? llm tutor",
    r"like an? llm tutor",
    r"dialogue setup",
    r"uploaded exercise",
    r"teach me instead of",
    r"messy questions",
    r"refer back to earlier things",
    r"diagnose your confusion",
    r"the data mining content itself",
    r"this to feel like",
    r"i will respond like",
    r"i may ask messy",
]

RECAP_SYNTHESIS_CUES = [
    r"bring.{0,20}(it all|everything|all of it) together",
    r"putting it all together",
    r"(let'?s |to |)recap (everything|all|what we'?ve|all we'?ve)",
    r"to sum (it|everything) up",
    r"looking back at (everything|what we'?ve|all (that|the))",
    r"we'?ve (now )?covered (everything|all|a lot|quite a bit|all (the|these))",
    r"across all (the )?(exercises|topics|sections|concepts|examples|material)",
    r"connect(ing)?.{0,20}(all (the|these)|everything)",
    r"overall (summary|recap|picture|view) of",
    r"throughout (this|our) (session|conversation|discussion|dialogue)",
    r"general (strategy|approach|method|framework).{0,30}(all|any|every)",
    r"applies (to all|across|regardless)",
    r"\bfinal strategy\b",
    r"\bidentify the task type\b",
]


def _meta_framing_hit(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(cue, lowered) for cue in META_FRAMING_CUES)


def _recap_synthesis_hit(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(cue, lowered) for cue in RECAP_SYNTHESIS_CUES)


def _is_non_kc(exchange: Any) -> bool:
    flags = set(getattr(exchange, "noise_flags", []) or [])
    text = f"{getattr(exchange, 'student_text', '')} {getattr(exchange, 'tutor_text', '')}".strip()
    if "empty_or_near_empty" in flags:
        return True
    if flags & LOW_CONTENT_FLAGS and len(text.split()) <= 8 and "?" not in text:
        return True
    return False


def _is_non_kc_meta_content(exchange: Any, candidate_pool: dict[str, Any]) -> bool:
    text = f"{getattr(exchange, 'student_text', '')} {getattr(exchange, 'tutor_text', '')}".strip()
    if not text:
        return False
    return _meta_framing_hit(text)


def _is_recap_synthesis(exchange: Any, candidate_pool: dict[str, Any]) -> bool:
    text = f"{getattr(exchange, 'student_text', '')} {getattr(exchange, 'tutor_text', '')}".strip()
    if not text:
        return False
    if not _recap_synthesis_hit(text):
        return False
    return bool(candidate_pool.get("broad_multibranch_signal"))


def _non_kc_row(exchange: Any, candidate_pool: dict[str, Any], reason: str) -> dict[str, Any]:
    candidates = candidate_pool.get("candidates") or []
    top_score = float(candidates[0].get("final_rank_score") or 0.0) if candidates else 0.0
    return {
        "dialogue_id": getattr(exchange, "dialogue_id", None) or candidate_pool.get("dialogue_id"),
        "exchange_id": getattr(exchange, "exchange_id", None) or candidate_pool.get("exchange_id"),
        "exchange_index": getattr(exchange, "exchange_index", None) or candidate_pool.get("exchange_index"),
        "pass1_label": "NON_KC",
        "provisional_kc_id": None,
        "provisional_kc_name": None,
        "secondary_kc_ids": [],
        "top1_score": 0.0,
        "top2_score": 0.0,
        "margin": 0.0,
        "branch": "",
        "confidence_hint": "none",
        "needs_pass2": False,
        "needs_novelty_check": False,
        "decision_reasons": [reason],
        "diagnostic_suppressed_top_candidate_score": top_score,
    }


def _eligible_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in candidates if bool(c.get("eligible_for_dominant"))]


def _best_candidate_for_pass1(candidate_pool: dict[str, Any]) -> dict[str, Any] | None:
    candidates = candidate_pool.get("candidates") or []
    eligible = _eligible_candidates(candidates)
    return eligible[0] if eligible else (candidates[0] if candidates else None)


def _secondary_ids(candidates: list[dict[str, Any]], dominant_id: str | None, limit: int = 4) -> list[str]:
    out: list[str] = []
    for cand in _eligible_candidates(candidates):
        uid = cand.get("unit_id")
        if not uid or uid == dominant_id:
            continue
        out.append(uid)
        if len(out) >= max(0, limit - 1):
            break
    return out


def _margin(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return float(candidates[0].get("final_rank_score") or 0.0)
    return float(candidates[0].get("final_rank_score") or 0.0) - float(candidates[1].get("final_rank_score") or 0.0)


def _relative_margin_ok(margin: float, top_score: float) -> bool:
    if top_score <= 0.0:
        return False
    return (margin / top_score) >= 0.15


def assign_pass1_v13(exchange: Any, candidate_pool: dict[str, Any]) -> dict[str, Any]:
    candidates = candidate_pool.get("candidates") or []

    if _is_non_kc(exchange):
        return _non_kc_row(exchange, candidate_pool, "low_content_or_admin_exchange")

    if _is_non_kc_meta_content(exchange, candidate_pool):
        return _non_kc_row(exchange, candidate_pool, "meta_dialogue_framing_content_weak_curriculum_evidence")

    if _is_recap_synthesis(exchange, candidate_pool):
        return _non_kc_row(exchange, candidate_pool, "recap_synthesis_broad_multibranch_detected")

    if not candidates:
        return {
            "dialogue_id": candidate_pool["dialogue_id"],
            "exchange_id": candidate_pool["exchange_id"],
            "exchange_index": candidate_pool["exchange_index"],
            "pass1_label": "NEEDS_NOVELTY_CHECK",
            "provisional_kc_id": None,
            "provisional_kc_name": None,
            "secondary_kc_ids": [],
            "top1_score": 0.0,
            "top2_score": 0.0,
            "margin": 0.0,
            "branch": "",
            "confidence_hint": "none",
            "needs_pass2": True,
            "needs_novelty_check": True,
            "decision_reasons": ["empty_candidate_pool"],
        }

    eligible = _eligible_candidates(candidates)
    ranked = eligible if eligible else candidates
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    top_score = float(top.get("final_rank_score") or 0.0)
    top2_score = float(second.get("final_rank_score") or 0.0) if second else 0.0
    margin = _margin(ranked)

    exact_override = bool(candidate_pool.get("exact_override_active")) and bool(top.get("exact_override_candidate"))
    corroborated = bool(top.get("has_profile_native_corroboration"))
    nonlexical_support = bool(top.get("has_nonlexical_family_support"))
    family_count = int(top.get("source_family_count") or 0)
    top_branch_share = float(candidate_pool.get("top_branch_share") or 0.0)
    active_branch_count = int(candidate_pool.get("active_branch_count") or 0)
    broad = bool(candidate_pool.get("broad_multibranch_signal"))
    tutor_only_noncanonical = bool(top.get("tutor_only_noncanonical_identity"))
    state = str(top.get("candidate_state") or "")
    margin_ok_high = _relative_margin_ok(margin, top_score)

    reasons: list[str] = []

    if exact_override and corroborated and not (broad and top_branch_share < 0.35):
        label, hint = "AUTO_HIGH_KC", "high"
        reasons.append("exact_canonical_override_pre_fusion")
    elif (
        bool(top.get("eligible_for_dominant"))
        and corroborated
        and nonlexical_support
        and family_count >= 2
        and state == "candidate"
        and margin_ok_high
        and not tutor_only_noncanonical
        and not broad
    ):
        label, hint = "AUTO_HIGH_KC", "high"
        reasons.append("typed_multi_family_corroborated_top_candidate")
    elif (
        bool(top.get("eligible_for_dominant"))
        and (corroborated or (family_count >= 2 and nonlexical_support))
        and top_branch_share >= 0.32
        and not tutor_only_noncanonical
    ):
        label, hint = "AUTO_MEDIUM_KC", "medium"
        reasons.append("typed_branch_coherent_known_library_candidate")
    elif ranked:
        label, hint = "NEEDS_PASS2", "low"
        reasons.append("typed_candidate_needs_context_or_broad_resolution")
        if broad:
            reasons.append("broad_multibranch_support_needs_context")
        if tutor_only_noncanonical:
            reasons.append("tutor_only_noncanonical_identity_needs_context")
        if not eligible:
            reasons.append("no_typed_eligible_candidate_in_top_pool")
    else:
        label, hint = "NEEDS_NOVELTY_CHECK", "none"
        reasons.append("very_shallow_candidate_pool")

    return {
        "dialogue_id": candidate_pool["dialogue_id"],
        "exchange_id": candidate_pool["exchange_id"],
        "exchange_index": candidate_pool["exchange_index"],
        "pass1_label": label,
        "provisional_kc_id": top.get("unit_id"),
        "provisional_kc_name": top.get("canonical_name"),
        "secondary_kc_ids": _secondary_ids(candidates, top.get("unit_id")),
        "top1_score": round(top_score, 6),
        "top2_score": round(top2_score, 6),
        "margin": round(margin, 6),
        "branch": top.get("branch") or "",
        "leaf_topic": top.get("leaf_topic") or "",
        "confidence_hint": hint,
        "needs_pass2": label in {"NEEDS_PASS2", "NEEDS_NOVELTY_CHECK"},
        "needs_novelty_check": label == "NEEDS_NOVELTY_CHECK",
        "decision_reasons": reasons,
        "signal_agreement_count": int(top.get("signal_agreement_count") or 0),
        "source_family_count": family_count,
        "eligible_for_dominant": bool(top.get("eligible_for_dominant")),
        "has_profile_native_corroboration": corroborated,
        "broad_multibranch_signal": broad,
        "active_branch_count": active_branch_count,
        "top_branch_share": round(top_branch_share, 6),
        "tutor_only_noncanonical_identity": tutor_only_noncanonical,
    }

