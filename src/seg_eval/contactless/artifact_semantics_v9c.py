from __future__ import annotations

from copy import deepcopy
from typing import Any


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def assignment_scope(row: dict[str, Any]) -> str:
    label = row.get("final_label")
    band = row.get("confidence_band")
    action = row.get("resolution_action")

    if label == "MISSING_KC_REVIEW":
        return "missing_kc"
    if label == "NON_KC":
        return "non_kc"

    if (
        bool(row.get("broad_multibranch_signal"))
        and band == "low"
        and action == "pass2_broad_multibranch_closed_set"
    ):
        return "broad_kc_set"

    if band == "low" or str(action or "").startswith("pass2_"):
        return "branch_context"

    return "single_kc"


def dominant_kc_role(scope: str) -> str:
    if scope == "single_kc":
        return "primary"
    if scope == "branch_context":
        return "context_anchor"
    if scope == "broad_kc_set":
        return "low_confidence_anchor"
    return "none"


def evaluator_kc_set(row: dict[str, Any], scope: str) -> list[str]:
    kc_set = list(row.get("kc_set") or [])
    resolved = row.get("resolved_kc_id")

    if scope in {"missing_kc", "non_kc"}:
        return kc_set

    if scope == "single_kc":
        return _unique([resolved] + list(row.get("secondary_kc_ids") or []))

    if kc_set:
        return _unique(kc_set)

    return _unique([resolved])


def annotate_assignment(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    scope = assignment_scope(out)

    out["assignment_scope"] = scope
    out["dominant_kc_role"] = dominant_kc_role(scope)
    out["evaluator_kc_set"] = evaluator_kc_set(out, scope)
    out["evaluator_notes"] = []

    if scope == "broad_kc_set":
        out["evaluator_notes"].append(
            "Broad multi-branch exchange: resolved KC is a low-confidence anchor, not a precise single-KC truth."
        )
    elif scope == "branch_context":
        out["evaluator_notes"].append(
            "Context-resolved or low-confidence exchange: use KC set and neighbouring segment context."
        )

    return out


def segment_scope(member_scopes: list[str]) -> str:
    if any(s == "missing_kc" for s in member_scopes):
        return "missing_kc"
    if any(s == "broad_kc_set" for s in member_scopes):
        return "broad_kc_set"
    if any(s == "branch_context" for s in member_scopes):
        return "branch_context"
    if all(s == "non_kc" for s in member_scopes):
        return "non_kc"
    return "single_kc"


def annotate_segments(assignments: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_exchange = {row["exchange_id"]: row for row in assignments}
    out_segments: list[dict[str, Any]] = []

    for seg in segments:
        out = dict(seg)
        member_rows = [by_exchange[eid] for eid in out.get("member_exchange_ids", []) if eid in by_exchange]
        scopes = [row.get("assignment_scope", "single_kc") for row in member_rows]
        roles = [row.get("dominant_kc_role", "primary") for row in member_rows]

        scope = segment_scope(scopes)
        evaluator_set: list[str] = []
        for row in member_rows:
            evaluator_set.extend(row.get("evaluator_kc_set") or [])

        out["segment_scope"] = scope
        out["dominant_kc_role"] = dominant_kc_role(scope)
        out["contains_broad_kc_set"] = any(s == "broad_kc_set" for s in scopes)
        out["contains_branch_context"] = any(s == "branch_context" for s in scopes)
        out["assignment_scopes"] = scopes
        out["dominant_kc_roles"] = roles
        out["evaluator_kc_set"] = _unique(evaluator_set or list(out.get("kc_set") or []))
        out["evaluator_notes"] = []

        if out["contains_broad_kc_set"]:
            out["evaluator_notes"].append(
                "At least one exchange is broad multi-branch. Treat dominant KC as an anchor and evaluate against evaluator_kc_set."
            )
        elif out["contains_branch_context"]:
            out["evaluator_notes"].append(
                "At least one exchange is context-resolved or low-confidence. Evaluate with the confidence bands and KC set visible."
            )

        out_segments.append(out)

    return out_segments


def annotate_contactless_artifacts(
    assignments: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated_assignments = [annotate_assignment(row) for row in assignments]
    annotated_segments = annotate_segments(annotated_assignments, segments)
    return annotated_assignments, annotated_segments
