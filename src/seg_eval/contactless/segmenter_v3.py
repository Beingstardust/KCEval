"""segmenter_v3.py

Fix 13: adds a conservative low/medium-confidence stabilisation rule to
the v2 contactless segmenter. The rule only attaches a low/medium row to
an already-established segment when the segment's dominant KC is present
in the row's own KC candidate set.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)


def _same_segment(current: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str]:
    label = row.get("final_label")
    kc = row.get("resolved_kc_id")

    if label == "NON_KC":
        return True, "non_kc_absorbed"

    if label == "MISSING_KC_REVIEW":
        return False, "missing_kc_review_boundary"

    if current.get("dominant_kc_id") and kc and current.get("dominant_kc_id") == kc:
        return True, "same_resolved_kc"

    dominant = current.get("dominant_kc_id")
    exchange_kc_set = set(row.get("kc_set") or [])

    # Fix 13 TODO: if pass2 rows later expose candidate-level scores, require
    # direct candidate support for the dominant KC here. The current row schema
    # exposes kc_set but not the scored candidate details, so kc_set is the
    # available support signal.
    if (
        row.get("confidence_band") in {"low", "medium"}
        and dominant
        and len(current.get("member_exchange_ids") or []) >= 2
        and dominant in exchange_kc_set
    ):
        return True, "medium_low_confidence_established_segment_kc_stabilisation"

    # Fix 7a retained from v2: low-confidence rows can attach when the current
    # segment's dominant KC is present in the row's own KC candidate set.
    if row.get("confidence_band") == "low":
        if dominant and dominant in exchange_kc_set:
            return True, "low_confidence_dominant_kc_in_exchange_candidate_set"

    return False, "boundary_required"


def _start_segment(exchange: Any, row: dict[str, Any], idx: int) -> dict[str, Any]:
    ex = _obj_to_dict(exchange)
    label = row.get("final_label")

    return {
        "segment_id": f"{row['dialogue_id']}::seg_{idx:04d}",
        "dialogue_id": row["dialogue_id"],
        "exchange_start": row["exchange_id"],
        "exchange_end": row["exchange_id"],
        "turn_start": ex["turn_start"],
        "turn_end": ex["turn_end"],
        "member_exchange_ids": [row["exchange_id"]],
        "member_turn_ends": [ex["turn_end"]],
        "dominant_kc_id": row.get("resolved_kc_id"),
        "dominant_kc_name": row.get("resolved_kc_name"),
        "dominant_branch": row.get("branch") or "",
        "kc_set": list(row.get("kc_set") or []),
        "confidence_bands": [row.get("confidence_band")],
        "final_labels": [label],
        "resolution_actions": [row.get("resolution_action")],
        "contains_auto_low": row.get("confidence_band") == "low",
        "review_required": label == "MISSING_KC_REVIEW",
        "review_reasons": list(row.get("review_reasons") or []),
        "boundary_reasons": {"start": ["new_contactless_segment"]},
    }


def _append(seg: dict[str, Any], exchange: Any, row: dict[str, Any], reason: str) -> None:
    ex = _obj_to_dict(exchange)
    seg["exchange_end"] = row["exchange_id"]
    seg["turn_end"] = ex["turn_end"]
    seg["member_exchange_ids"].append(row["exchange_id"])
    seg["member_turn_ends"].append(ex["turn_end"])
    seg["confidence_bands"].append(row.get("confidence_band"))
    seg["final_labels"].append(row.get("final_label"))
    seg["resolution_actions"].append(row.get("resolution_action"))
    seg["contains_auto_low"] = seg["contains_auto_low"] or row.get("confidence_band") == "low"
    seg["review_required"] = seg["review_required"] or row.get("final_label") == "MISSING_KC_REVIEW"
    seg["review_reasons"] = sorted(set(seg["review_reasons"] + list(row.get("review_reasons") or [])))
    for uid in row.get("kc_set") or []:
        if uid not in seg["kc_set"]:
            seg["kc_set"].append(uid)
    seg["boundary_reasons"].setdefault("merge", []).append(reason)


def build_contactless_segments(
    exchanges: list[Any],
    assignments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for exchange, row in zip(exchanges, assignments):
        if current is None:
            current = _start_segment(exchange, row, len(segments))
            continue

        ok, reason = _same_segment(current, row)
        if ok:
            _append(current, exchange, row, reason)
        else:
            segments.append(current)
            current = _start_segment(exchange, row, len(segments))

    if current is not None:
        segments.append(current)

    for seg in segments:
        if seg.get("review_required"):
            review_queue.append(
                {
                    "review_id": f"missing_kc_review_{len(review_queue):04d}",
                    "segment_id": seg["segment_id"],
                    "member_exchange_ids": seg["member_exchange_ids"],
                    "review_reasons": seg["review_reasons"],
                    "dominant_branch": seg.get("dominant_branch"),
                    "kc_set": seg.get("kc_set"),
                }
            )

    return segments, review_queue
