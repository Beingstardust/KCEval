"""Segment evaluation packet builder.

Consumes accepted contactless segmentation artifacts and produces judge-ready
segment packets. This module does not change segmentation or assignment.

The builder is intentionally domain-generic. It reads semantic fields from the
segmentation artifacts using generic aliases because different segmentation
versions may expose the same concept under different names, for example
``resolved_kc_id`` versus ``dominant_kc_id``.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .focus_target_v1 import compute_focus_target


KC_PROFILE_FIELDS = [
    "canonical_terms",
    "matching_cues",
    "likely_dialogue_surface_forms",
    "formula_or_symbol_forms",
    "matched_surface_terms_from_evidence",
]

PRIMARY_ID_ALIASES = [
    "primary_kc_id",
    "dominant_kc_id",
    "resolved_dominant_kc_id",
    "resolved_kc_id",
    "provisional_kc_id",
]

PRIMARY_NAME_ALIASES = [
    "primary_kc_name",
    "dominant_kc_name",
    "resolved_dominant_kc_name",
    "resolved_kc_name",
    "provisional_kc_name",
]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def first_nonempty(row: dict[str, Any], names: list[str]) -> tuple[Any, str | None]:
    for name in names:
        value = row.get(name)
        if value not in (None, "", []):
            return value, name
    return None, None


def unique_nonempty(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value in (None, "", []):
            continue
        text = str(value)
        if text not in out:
            out.append(text)
    return out


def compact_profile(profile: dict[str, Any] | None, unit_id: str) -> dict[str, Any]:
    if not profile:
        return {
            "unit_id": unit_id,
            "missing_profile": True,
        }

    positive = profile.get("positive_profile") or {}
    negative = profile.get("negative_profile") or {}

    compact_positive = {k: positive.get(k, []) for k in KC_PROFILE_FIELDS if positive.get(k)}

    return {
        "unit_id": profile.get("unit_id"),
        "unit_type": profile.get("unit_type"),
        "canonical_name": profile.get("canonical_name"),
        "topic_path": profile.get("topic_path") or [],
        "source_hierarchy_path": profile.get("source_hierarchy_path") or [],
        "definition_or_summary": profile.get("definition_or_summary"),
        "source_packet_sha256": profile.get("source_packet_sha256"),
        "positive_profile": compact_positive,
        "negative_profile": {
            "sibling_kc_names": negative.get("sibling_kc_names", []),
            "sibling_contrast_notes": negative.get("sibling_contrast_notes", []),
        },
    }


def packet_mode(segment: dict[str, Any]) -> str:
    scope = segment.get("segment_scope")
    role = segment.get("dominant_kc_role")

    if scope == "single_kc" and role == "primary":
        return "single_kc_primary"
    if scope == "branch_context":
        return "branch_context_with_anchor"
    if scope == "broad_kc_set" or role == "low_confidence_anchor":
        return "broad_kc_set_low_confidence_anchor"
    return "mixed_or_low_confidence"


def evaluator_guidance(mode: str) -> list[str]:
    base = [
        "Evaluate the tutor response only within this segment, using the segment text and KC grounding below.",
        "Do not treat a low-confidence anchor as exact truth; use the evaluator_kc_set and segment_scope fields.",
        "If kc_specific_criteria are absent, judge against the KC definition/summary and the general tutor-evaluation rubric.",
    ]

    if mode == "single_kc_primary":
        base.append("Primary evaluation target: the dominant KC is the main grounding concept for this segment.")
    elif mode == "branch_context_with_anchor":
        base.append("Context evaluation target: the dominant KC is an anchor chosen from neighbouring/contextual evidence; inspect the KC set before judging.")
    elif mode == "broad_kc_set_low_confidence_anchor":
        base.append("Broad-set evaluation target: this segment spans multiple plausible KCs; judge coverage/coherence against the KC set rather than only the anchor.")
    else:
        base.append("Mixed/low-confidence target: preserve uncertainty in the judgment and avoid over-penalising anchor mismatch.")

    return base


def exchange_packet(
    exchange: dict[str, Any],
    assignment: dict[str, Any] | None,
) -> dict[str, Any]:
    assignment = assignment or {}
    return {
        "exchange_id": exchange.get("exchange_id"),
        "exchange_index": exchange.get("exchange_index"),
        "turn_start": exchange.get("turn_start"),
        "turn_end": exchange.get("turn_end"),
        "student_text": exchange.get("student_text"),
        "tutor_text": exchange.get("tutor_text"),
        "assignment": {
            "final_label": assignment.get("final_label"),
            "confidence_band": assignment.get("confidence_band"),
            "resolved_kc_id": assignment.get("resolved_kc_id"),
            "resolved_kc_name": assignment.get("resolved_kc_name"),
            "assignment_scope": assignment.get("assignment_scope"),
            "dominant_kc_role": assignment.get("dominant_kc_role"),
            "resolution_action": assignment.get("resolution_action"),
            "evaluator_kc_set": assignment.get("evaluator_kc_set") or assignment.get("kc_set", []),
            "context_dependency": assignment.get("context_dependency"),
            "context_dependency_anchor_handling": assignment.get("context_dependency_anchor_handling"),
            "novelty_score": assignment.get("novelty_score"),
            "review_required": assignment.get("review_required", False),
        },
    }


def build_segment_text(member_exchanges: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ex in member_exchanges:
        ex_id = ex.get("exchange_id")
        parts.append(f"[{ex_id}] Student: {ex.get('student_text') or ''}")
        parts.append(f"[{ex_id}] Tutor: {ex.get('tutor_text') or ''}")
    return "\n".join(parts)


def collect_primary(segment: dict[str, Any], evaluator_kc_set: list[str]) -> tuple[str | None, str | None, str | None, str | None]:
    primary_id, primary_id_source = first_nonempty(segment, PRIMARY_ID_ALIASES)
    primary_name, primary_name_source = first_nonempty(segment, PRIMARY_NAME_ALIASES)

    if not primary_id and segment.get("segment_scope") == "single_kc" and len(evaluator_kc_set) == 1:
        primary_id = evaluator_kc_set[0]
        primary_id_source = "single_kc_evaluator_kc_set_fallback"

    if primary_id:
        primary_id = str(primary_id)

    if primary_name:
        primary_name = str(primary_name)

    return primary_id, primary_name, primary_id_source, primary_name_source


def collect_evaluator_kc_set(segment: dict[str, Any]) -> list[str]:
    return unique_nonempty(list(segment.get("evaluator_kc_set") or segment.get("kc_set") or []))


def build_packet(
    segment: dict[str, Any],
    exchanges_by_id: dict[str, dict[str, Any]],
    assignments_by_id: dict[str, dict[str, Any]],
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    member_ids = segment.get("member_exchange_ids") or []
    member_exchanges = [exchanges_by_id[eid] for eid in member_ids if eid in exchanges_by_id]
    member_packets = [exchange_packet(ex, assignments_by_id.get(ex.get("exchange_id"))) for ex in member_exchanges]

    evaluator_kc_set = collect_evaluator_kc_set(segment)
    primary_kc_id, primary_kc_name, primary_id_source, primary_name_source = collect_primary(segment, evaluator_kc_set)

    if primary_kc_id and primary_kc_id not in evaluator_kc_set:
        evaluator_kc_set.insert(0, primary_kc_id)

    mode = packet_mode(segment)
    kc_context = [compact_profile(profiles_by_id.get(uid), uid) for uid in evaluator_kc_set]

    return {
        "packet_schema": "segment_evaluation_packet.v1b",
        "packet_id": f"packet::{segment.get('segment_id')}",
        "segment_id": segment.get("segment_id"),
        "dialogue_id": segment.get("dialogue_id"),
        "turn_start": segment.get("turn_start"),
        "turn_end": segment.get("turn_end"),
        "exchange_start": segment.get("exchange_start"),
        "exchange_end": segment.get("exchange_end"),
        "member_exchange_ids": member_ids,
        "segment_text": build_segment_text(member_exchanges),
        "segmentation_metadata": {
            "segment_scope": segment.get("segment_scope"),
            "dominant_kc_id": primary_kc_id,
            "dominant_kc_name": primary_kc_name,
            "dominant_kc_role": segment.get("dominant_kc_role"),
            "dominant_branch": segment.get("dominant_branch"),
            "confidence_bands": segment.get("confidence_bands", []),
            "final_labels": segment.get("final_labels", []),
            "assignment_scopes": segment.get("assignment_scopes", []),
            "resolution_actions": segment.get("resolution_actions", []),
            "contains_auto_low": segment.get("contains_auto_low", False),
            "contains_branch_context": segment.get("contains_branch_context", False),
            "contains_broad_kc_set": segment.get("contains_broad_kc_set", False),
            "contains_context_dependent_rescue": segment.get("contains_context_dependent_rescue", False),
            "boundary_reasons": segment.get("boundary_reasons", {}),
        },
        "evaluation_target": {
            "mode": mode,
            "primary_kc_id": primary_kc_id,
            "primary_kc_name": primary_kc_name,
            "primary_kc_role": segment.get("dominant_kc_role"),
            "evaluator_kc_set": evaluator_kc_set,
            "instructions": evaluator_guidance(mode),
        },
        "kc_context": kc_context,
        "member_exchanges": member_packets,
        # Additive only -- computed read-only from member_exchanges, never alters segment_text,
        # KC assignment, or anything above. See focus_target_v1.py for the classification rules.
        "focus_target": compute_focus_target(member_packets),
        "segment_review": {
            "review_required": segment.get("review_required", False),
            "review_reasons": segment.get("review_reasons", []),
            "evaluator_notes": segment.get("evaluator_notes", []),
        },
        "packet_trace": {
            "primary_kc_id_source": primary_id_source,
            "primary_kc_name_source": primary_name_source,
            "segment_field_alias_contract": {
                "primary_id_aliases": PRIMARY_ID_ALIASES,
                "primary_name_aliases": PRIMARY_NAME_ALIASES,
            },
        },
    }


def audit_packets(
    packets: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    exchanges: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    missing_review: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(packets) != len(segments):
        errors.append(f"packet_count_mismatch:packets={len(packets)}:segments={len(segments)}")

    packet_member_ids: list[str] = []
    for packet in packets:
        sid = packet.get("segment_id")
        mode = (packet.get("evaluation_target") or {}).get("mode")
        target = packet.get("evaluation_target") or {}
        meta = packet.get("segmentation_metadata") or {}
        kc_set = target.get("evaluator_kc_set") or []
        primary = target.get("primary_kc_id")

        if not packet.get("segment_text"):
            errors.append(f"empty_segment_text:{sid}")

        if not kc_set and mode != "missing_kc_review":
            errors.append(f"empty_evaluator_kc_set:{sid}")

        if mode == "single_kc_primary" and not primary:
            errors.append(f"single_kc_missing_primary:{sid}")

        if mode == "single_kc_primary" and primary and kc_set and primary != kc_set[0]:
            warnings.append(f"single_kc_primary_not_first_in_kc_set:{sid}:{primary}:{kc_set[0]}")

        if mode == "broad_kc_set_low_confidence_anchor":
            if meta.get("dominant_kc_role") != "low_confidence_anchor":
                warnings.append(f"broad_mode_without_low_confidence_role:{sid}")

        packet_member_ids.extend(packet.get("member_exchange_ids") or [])

    exchange_ids = [row.get("exchange_id") for row in exchanges]
    if packet_member_ids != exchange_ids:
        errors.append("packet_member_exchange_order_or_coverage_mismatch")

    assignment_ids = [row.get("exchange_id") for row in assignments]
    if set(assignment_ids) != set(exchange_ids):
        errors.append("assignment_exchange_set_mismatch")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "packet_count": len(packets),
        "segment_count": len(segments),
        "exchange_count": len(exchanges),
        "assignment_count": len(assignments),
        "missing_kc_review_count": len(missing_review),
        "evaluation_mode_counts": dict(sorted(Counter((p.get("evaluation_target") or {}).get("mode") for p in packets).items())),
        "segment_scope_counts": dict(sorted(Counter((p.get("segmentation_metadata") or {}).get("segment_scope") for p in packets).items())),
        "primary_kc_id_source_counts": dict(sorted(Counter((p.get("packet_trace") or {}).get("primary_kc_id_source") for p in packets).items())),
    }


def build_packets(
    run_dir: str | Path,
    profiles_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = read_json(run / "summary.json")
    exchanges = read_jsonl(run / "exchanges.jsonl")
    assignments = read_jsonl(run / "contactless_exchange_assignments.jsonl")
    segments = read_jsonl(run / "contactless_segments.jsonl")
    missing_review = read_jsonl(run / "review_queue_missing_kc_only.jsonl")
    profiles = read_jsonl(profiles_path)

    exchanges_by_id = {row["exchange_id"]: row for row in exchanges}
    assignments_by_id = {row["exchange_id"]: row for row in assignments}
    profiles_by_id = {row["unit_id"]: row for row in profiles if row.get("unit_id")}

    packets = [build_packet(seg, exchanges_by_id, assignments_by_id, profiles_by_id) for seg in segments]
    packet_audit = audit_packets(packets, segments, exchanges, assignments, missing_review)

    scope_counts = Counter((p.get("segmentation_metadata") or {}).get("segment_scope") for p in packets)
    mode_counts = Counter((p.get("evaluation_target") or {}).get("mode") for p in packets)

    packet_summary = {
        "packet_pipeline": "segment_evaluation_packets_v1b_primary_alias_contract",
        "source_segmentation_pipeline": summary.get("pipeline"),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "profiles_path": str(profiles_path),
        "segment_count": len(segments),
        "packet_count": len(packets),
        "exchange_count": len(exchanges),
        "assignment_count": len(assignments),
        "missing_kc_review_count": len(missing_review),
        "segment_scope_counts": dict(sorted(scope_counts.items())),
        "evaluation_mode_counts": dict(sorted(mode_counts.items())),
    }

    write_jsonl(out / "segment_evaluation_packets.jsonl", packets)
    write_json(out / "packet_summary.json", packet_summary)
    write_json(out / "packet_audit.json", packet_audit)
    write_json(out / "packet_manifest.json", {
        "schema": "segment_evaluation_packet_manifest.v1b",
        "run_dir": str(run_dir),
        "profiles_path": str(profiles_path),
        "outputs": {
            "segment_evaluation_packets": "segment_evaluation_packets.jsonl",
            "packet_summary": "packet_summary.json",
            "packet_audit": "packet_audit.json",
        },
        "semantic_contract": {
            "single_kc_primary": "Judge primarily against primary_kc_id.",
            "branch_context_with_anchor": "Use primary_kc_id as context anchor and inspect evaluator_kc_set.",
            "broad_kc_set_low_confidence_anchor": "Do not treat primary_kc_id as exact truth; judge against KC set.",
        },
        "alias_contract": {
            "primary_id_aliases": PRIMARY_ID_ALIASES,
            "primary_name_aliases": PRIMARY_NAME_ALIASES,
        },
        "summary": packet_summary,
        "audit": packet_audit,
    })

    md = [
        "# Segment evaluation packet audit",
        "",
        f"- Status: `{packet_audit['status']}`",
        f"- Packet count: `{packet_audit['packet_count']}`",
        f"- Segment count: `{packet_audit['segment_count']}`",
        f"- Exchange count: `{packet_audit['exchange_count']}`",
        f"- Missing-KC review count: `{packet_audit['missing_kc_review_count']}`",
        f"- Evaluation modes: `{packet_audit['evaluation_mode_counts']}`",
        f"- Primary-KC ID sources: `{packet_audit['primary_kc_id_source_counts']}`",
    ]
    if packet_audit["errors"]:
        md.extend(["", "## Errors", ""])
        md.extend(f"- {x}" for x in packet_audit["errors"])
    if packet_audit["warnings"]:
        md.extend(["", "## Warnings", ""])
        md.extend(f"- {x}" for x in packet_audit["warnings"])
    Path(out / "packet_audit.md").write_text("\n".join(md), encoding="utf-8")

    if packet_audit["status"] != "PASS":
        raise RuntimeError("Segment evaluation packet audit failed: " + "; ".join(packet_audit["errors"][:10]))

    return {"summary": packet_summary, "out_dir": str(out)}
