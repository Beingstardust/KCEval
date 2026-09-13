"""Shared constants and the immutable-row-hash function used by both the human-validation
workbook builder and its validator/finalizer.

The hash function living in exactly one place, imported by both sides, is what makes tamper
detection actually work: the builder writes a hash computed here, the validator recomputes the
same hash from the same fields and compares. If these two ever drifted apart (e.g. a copy-pasted
variant in each script), every row would silently fail tamper detection.
"""

from __future__ import annotations

import hashlib
import json

# Human's own independently-defined focus-target categories (Pass 1). Deliberately NOT required
# to share exact string spelling with focus_target_v1.TARGET_TYPES -- see
# FOCUS_HUMAN_TO_MACHINE_TARGET_TYPE_MAP below for the explicit alignment used only when
# comparing human vs. machine, never used to rename or alter the machine's own categories.
FOCUS_MATERIAL_CLAIM_OPTIONS = ["yes", "no", "uncertain"]

FOCUS_HUMAN_TARGET_TYPE_OPTIONS = [
    "precise_focus_target",
    "broad_tutor_claim_target",
    "no_material_domain_claim",
    "ambiguous_or_multi_claim_target",
]

# Explicit alignment for agreement computation only. The machine's own enum
# (seg_eval.evaluation_packets.focus_target_v1.TARGET_TYPES) is never renamed to match this --
# this map translates FROM the machine's spelling TO the human sheet's spelling, one direction,
# used only inside the analyzer.
FOCUS_MACHINE_TO_HUMAN_TARGET_TYPE = {
    "precise_focus_target": "precise_focus_target",
    "broad_tutor_claim_target": "broad_tutor_claim_target",
    "no_domain_claim": "no_material_domain_claim",
    "insufficient_or_ambiguous_target": "ambiguous_or_multi_claim_target",
}

FOCUS_PASS2_APPROPRIATE_OPTIONS = ["yes", "partial", "no", "not_applicable"]
FOCUS_PASS2_BROAD_FALLBACK_OPTIONS = ["yes", "no", "not_applicable"]
FOCUS_PASS2_MISSED_CLAIM_OPTIONS = ["yes", "no"]

PARTIAL_SCORE_OPTIONS = ["0", "0.5", "1", "N/A"]
TRUST_SCORE_OPTIONS = ["0", "1"]
MACRO_SCORE_OPTIONS = ["0", "0.5", "1"]


def compute_row_hash(fields: dict) -> str:
    """Deterministic hash over IMMUTABLE context fields only -- never a human-editable field.
    Same function, same field set, used to write the hash and later to verify it; callers on
    both sides must pass exactly the same fields dict shape for a given row type."""
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def focus_pass1_row_hash_fields(row: dict) -> dict:
    return {
        "review_item_id": row["review_item_id"],
        "dialogue_id": row["dialogue_id"],
        "family": row["family"],
        "evaluation_mode": row.get("evaluation_mode"),
        "segment_text": row.get("segment_text", ""),
    }


def judge_validation_row_hash_fields(row: dict) -> dict:
    return {
        "review_item_id": row["review_item_id"],
        "dialogue_id": row["dialogue_id"],
        "family": row["family"],
        "evaluation_mode": row.get("evaluation_mode"),
        "segment_text": row.get("segment_text", ""),
    }


def macro_validation_row_hash_fields(row: dict, dialogue_text: str = "") -> dict:
    return {
        "review_item_id": row["review_item_id"],
        "dialogue_id": row["dialogue_id"],
        "exchange_count": row.get("exchange_count"),
        "dialogue_text": dialogue_text,
    }


def focus_pass2_row_hash_fields(row: dict, pass1_answers: dict) -> dict:
    """Includes the frozen Pass-1 answers, so Pass-2 editing cannot silently alter them --
    the validator recomputing this hash will fail if any Pass-1 value changed."""
    return {
        "review_item_id": row["review_item_id"],
        "dialogue_id": row["dialogue_id"],
        "family": row["family"],
        "pass1_human_material_domain_claim": pass1_answers.get("human_material_domain_claim"),
        "pass1_human_target_type": pass1_answers.get("human_target_type"),
        "pass1_human_preferred_focus_text_or_turn": pass1_answers.get("human_preferred_focus_text_or_turn"),
        "pass1_annotator_initials": pass1_answers.get("annotator_initials"),
    }
