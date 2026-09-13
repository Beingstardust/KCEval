"""pre_fusion_non_kc_gate_v1.py

Exchange-level NON_KC gate for v20.

Why this exists
---------------
The v19 follow-up regression analysis found that ex_0057 regressed because
the recap-synthesis detector changed semantics in `local_assigner_v13.py`:
v15's actual code path gated on the recap cue phrase alone, while v19 added
an additional `broad_multibranch_signal` requirement and therefore let the
exchange proceed into KC resolution.

This module moves the generic NON_KC gate family ahead of retrieval/fusion so
exchange-level admin/meta/recap turns can be short-circuited before scoring.
That is deliberately narrow: it does not touch candidate scoring, RRF, CE
reranking, or pass-2 logic.
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


def _exchange_text(exchange: Any) -> str:
    return f"{getattr(exchange, 'student_text', '')} {getattr(exchange, 'tutor_text', '')}".strip()


def _meta_framing_hit(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(cue, lowered) for cue in META_FRAMING_CUES)


def _recap_synthesis_hit(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(cue, lowered) for cue in RECAP_SYNTHESIS_CUES)


def classify_pre_fusion_non_kc(exchange: Any) -> dict[str, str] | None:
    """Return a gate record when the exchange should bypass retrieval.

    The recap branch intentionally mirrors v15's *actual* behavior:
    cue-phrase match alone is sufficient. The v15 docstring described an
    additional broad-signal guard, but the code path that produced the frozen
    v15 outputs did not require it, and ex_0057's correct NON_KC exclusion
    depends on that implementation fact.
    """
    text = _exchange_text(exchange)
    flags = set(getattr(exchange, "noise_flags", []) or [])

    if "empty_or_near_empty" in flags:
        return {"reason": "low_content_or_admin_exchange", "gate_type": "low_content"}

    if flags & LOW_CONTENT_FLAGS and len(text.split()) <= 8 and "?" not in text:
        return {"reason": "low_content_or_admin_exchange", "gate_type": "low_content"}

    if text and _meta_framing_hit(text):
        return {
            "reason": "meta_dialogue_framing_content_weak_curriculum_evidence",
            "gate_type": "meta_framing",
        }

    if text and _recap_synthesis_hit(text):
        return {
            "reason": "recap_synthesis_broad_multibranch_detected",
            "gate_type": "recap_synthesis",
        }

    return None


def make_pre_fusion_non_kc_pool(exchange: Any, gate: dict[str, str]) -> dict[str, Any]:
    return {
        "dialogue_id": getattr(exchange, "dialogue_id"),
        "exchange_id": getattr(exchange, "exchange_id"),
        "exchange_index": getattr(exchange, "exchange_index"),
        "student_text_raw": getattr(exchange, "student_text", "") or "",
        "tutor_text_raw": getattr(exchange, "tutor_text", "") or "",
        "exchange_text_raw": getattr(exchange, "exchange_text", "") or "",
        "candidate_count": 0,
        "full_candidate_count": 0,
        "top_candidate_ids": [],
        "cross_encoder_active": False,
        "dense_retrieval_active": False,
        "exact_override_active": False,
        "rrf_k": None,
        "family_support_rank_cutoff": None,
        "profile_family_views_used": [],
        "branch_scores": {},
        "top_branch_share": 0.0,
        "active_branch_count": 0,
        "broad_multibranch_signal": False,
        "hybrid_bm25_top_ids": [],
        "hybrid_char_top_ids": [],
        "dense_top_ids": [],
        "pre_fusion_non_kc_gate_active": True,
        "pre_fusion_non_kc_gate_type": gate["gate_type"],
        "pre_fusion_non_kc_gate_reason": gate["reason"],
        "candidates": [],
    }


def make_pre_fusion_non_kc_pass1(exchange: Any, pool: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "dialogue_id": getattr(exchange, "dialogue_id", None) or pool.get("dialogue_id"),
        "exchange_id": getattr(exchange, "exchange_id", None) or pool.get("exchange_id"),
        "exchange_index": getattr(exchange, "exchange_index", None) or pool.get("exchange_index"),
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
        "diagnostic_suppressed_top_candidate_score": 0.0,
        "pre_fusion_non_kc_gate_active": True,
        "pre_fusion_non_kc_gate_reason": reason,
    }
