"""Mixed-granularity judge packet preparation.

This is a post-hoc additive layer on top of a frozen segmentation baseline.
It does not alter segmentation boundaries, dominant KC selection, or prior
runtime artifacts. Instead, it prepares two evaluation-unit families:

1. Original KC-level segments for locally judgeable dimensions.
2. Topic-rollup units for dimensions that need a wider multi-turn arc.

The implementation is domain-generic. It uses only the frozen segmentation
artifacts, frozen KC profiles, and the additive topic rollup artifact.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .builder import (
    audit_packets,
    build_packet,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS


ARC_PARTIAL_DIMENSIONS = [
    "scaffolding_quality",
    "student_level_calibration",
]
ARC_TRUST_FLAGS = [
    "student_context_hallucination_present",
]

LOCAL_PARTIAL_DIMENSIONS = [item for item in PARTIAL_DIMENSIONS if item not in ARC_PARTIAL_DIMENSIONS]
LOCAL_TRUST_FLAGS = [item for item in TRUST_FLAGS if item not in ARC_TRUST_FLAGS]

LOCAL_FAMILY_ID = "kc_segment_local_dimensions_v1"
TOPIC_FAMILY_ID = "topic_rollup_arc_dimensions_v1"


def short_id(value: str) -> str:
    return value.rsplit("::", 1)[-1]


def read_topic_units(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique_nonempty(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value in (None, "", []):
            continue
        text = str(value)
        if text not in out:
            out.append(text)
    return out


def local_judge_plan() -> dict[str, Any]:
    return {
        "evaluation_family_id": LOCAL_FAMILY_ID,
        "evaluation_unit_type": "kc_segment",
        "recommended_partial_dimensions": list(LOCAL_PARTIAL_DIMENSIONS),
        "recommended_trust_flags": list(LOCAL_TRUST_FLAGS),
        "deferred_partial_dimensions": list(ARC_PARTIAL_DIMENSIONS),
        "deferred_trust_flags": list(ARC_TRUST_FLAGS),
        "rationale": (
            "Use original KC-level segments for dimensions that are primarily "
            "local to a single exchange or a short instructional response."
        ),
    }


def topic_judge_plan() -> dict[str, Any]:
    return {
        "evaluation_family_id": TOPIC_FAMILY_ID,
        "evaluation_unit_type": "topic_rollup",
        "recommended_partial_dimensions": list(ARC_PARTIAL_DIMENSIONS),
        "recommended_trust_flags": list(ARC_TRUST_FLAGS),
        "deferred_partial_dimensions": list(LOCAL_PARTIAL_DIMENSIONS),
        "deferred_trust_flags": list(LOCAL_TRUST_FLAGS),
        "rationale": (
            "Use topic-rollup units for arc-level dimensions that need a wider "
            "multi-turn window rather than a single leaf-KC segment."
        ),
    }


def overlay_local_packet(packet: dict[str, Any]) -> dict[str, Any]:
    packet = dict(packet)
    packet["packet_schema"] = "segment_evaluation_packet.v1b_mixed_granularity_overlay"
    packet["evaluation_unit_type"] = "kc_segment"
    packet["judge_plan"] = local_judge_plan()
    packet["granularity_metadata"] = {
        "evaluation_unit_type": "kc_segment",
        "source_segment_ids": [packet.get("segment_id")],
        "source_segment_count": 1,
        "source_topic_unit_ids": [],
        "granularity_note": "Original frozen v15 segment used directly for local-dimension judging.",
    }
    return packet


def mode_for_topic_unit(topic_unit: dict[str, Any]) -> tuple[str, str]:
    if bool(topic_unit.get("contains_fragmentation_risk")):
        return "broad_kc_set", "low_confidence_anchor"
    return "branch_context", "context_anchor"


def evaluator_kc_set_for_topic_unit(
    topic_unit: dict[str, Any],
    source_segments_by_short: dict[str, dict[str, Any]],
    assignments_by_short: dict[str, dict[str, Any]],
) -> list[str]:
    kc_ids: list[str] = []

    for short_segment_id in topic_unit.get("source_segment_ids", []):
        segment = source_segments_by_short.get(short_segment_id)
        if not segment:
            continue
        kc_ids.extend(segment.get("evaluator_kc_set") or [])

    if not kc_ids:
        for exchange_row in topic_unit.get("exchanges", []):
            exchange_id = str(exchange_row.get("exchange_id") or "")
            assignment = assignments_by_short.get(exchange_id)
            if assignment and assignment.get("resolved_kc_id"):
                kc_ids.append(str(assignment["resolved_kc_id"]))

    return unique_nonempty(kc_ids)


def source_segment_flags(
    topic_unit: dict[str, Any],
    source_segments_by_short: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_segments = [
        source_segments_by_short[short_segment_id]
        for short_segment_id in topic_unit.get("source_segment_ids", [])
        if short_segment_id in source_segments_by_short
    ]

    return {
        "contains_branch_context": any(bool(seg.get("contains_branch_context")) for seg in source_segments),
        "contains_broad_kc_set": any(bool(seg.get("contains_broad_kc_set")) for seg in source_segments),
        "contains_context_dependent_rescue": any(bool(seg.get("contains_context_dependent_rescue")) for seg in source_segments),
        "review_required": any(bool(seg.get("review_required")) for seg in source_segments) or bool(topic_unit.get("contains_fragmentation_risk")),
        "review_reasons": unique_nonempty(
            list(topic_unit.get("fragmentation_reasons") or [])
            + [reason for seg in source_segments for reason in (seg.get("review_reasons") or [])]
        ),
        "evaluator_notes": unique_nonempty(
            [
                "Topic-level additive rollup unit prepared for arc-level judging.",
                "Use the shared parent topic and member exchange sequence as the coherence window.",
                "Do not force a single leaf-KC interpretation when the unit spans multiple source segments.",
            ]
            + [note for seg in source_segments for note in (seg.get("evaluator_notes") or [])]
        ),
    }


def topic_unit_segment_like(
    topic_unit: dict[str, Any],
    dialogue_id: str,
    exchanges_by_short: dict[str, dict[str, Any]],
    assignments_by_short: dict[str, dict[str, Any]],
    source_segments_by_short: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exchange_short_ids = [str(row.get("exchange_id") or "") for row in topic_unit.get("exchanges", [])]
    full_exchange_ids = [
        str(exchanges_by_short[exchange_id]["exchange_id"])
        for exchange_id in exchange_short_ids
        if exchange_id in exchanges_by_short
    ]
    if not full_exchange_ids:
        raise RuntimeError(f"Topic unit has no recoverable exchanges: {topic_unit.get('topic_unit_id')}")

    first_exchange = exchanges_by_short[exchange_short_ids[0]]
    last_exchange = exchanges_by_short[exchange_short_ids[-1]]

    confidence_bands: list[str] = []
    final_labels: list[str] = []
    assignment_scopes: list[str] = []
    resolution_actions: list[str] = []
    contains_auto_low = False

    for exchange_id in exchange_short_ids:
        assignment = assignments_by_short.get(exchange_id) or {}
        band = assignment.get("confidence_band")
        label = assignment.get("final_label")
        scope = assignment.get("assignment_scope")
        action = assignment.get("resolution_action")
        if band:
            confidence_bands.append(str(band))
        if label:
            final_labels.append(str(label))
            if str(label) == "AUTO_LOW_KC":
                contains_auto_low = True
        if scope:
            assignment_scopes.append(str(scope))
        if action:
            resolution_actions.append(str(action))

    evaluator_kc_set = evaluator_kc_set_for_topic_unit(topic_unit, source_segments_by_short, assignments_by_short)
    segment_scope, dominant_kc_role = mode_for_topic_unit(topic_unit)
    source_flags = source_segment_flags(topic_unit, source_segments_by_short)

    boundary_reason = "topic_rollup_parent_topic"
    if topic_unit.get("topic_formation_source") == "adjacency_fallback_topic_vote":
        boundary_reason = "topic_rollup_parent_topic_with_adjacency_fallback"

    return {
        "segment_id": f"{dialogue_id}::{topic_unit['topic_unit_id']}",
        "dialogue_id": dialogue_id,
        "turn_start": first_exchange.get("turn_start"),
        "turn_end": last_exchange.get("turn_end"),
        "exchange_start": full_exchange_ids[0],
        "exchange_end": full_exchange_ids[-1],
        "member_exchange_ids": full_exchange_ids,
        "segment_scope": segment_scope,
        "dominant_kc_id": None,
        "dominant_kc_name": topic_unit.get("shared_parent_topic_node") or topic_unit.get("shared_parent_topic_label"),
        "dominant_kc_role": dominant_kc_role,
        "dominant_branch": topic_unit.get("shared_parent_topic_label"),
        "confidence_bands": confidence_bands,
        "final_labels": final_labels,
        "assignment_scopes": assignment_scopes,
        "resolution_actions": resolution_actions,
        "contains_auto_low": contains_auto_low,
        "contains_branch_context": source_flags["contains_branch_context"],
        "contains_broad_kc_set": source_flags["contains_broad_kc_set"],
        "contains_context_dependent_rescue": source_flags["contains_context_dependent_rescue"],
        "boundary_reasons": {
            "start": [boundary_reason],
            "aggregation": [boundary_reason],
        },
        "evaluator_kc_set": evaluator_kc_set,
        "kc_set": evaluator_kc_set,
        "review_required": source_flags["review_required"],
        "review_reasons": source_flags["review_reasons"],
        "evaluator_notes": source_flags["evaluator_notes"],
    }


def build_topic_packet(
    topic_unit: dict[str, Any],
    dialogue_id: str,
    exchanges_by_id: dict[str, dict[str, Any]],
    assignments_by_id: dict[str, dict[str, Any]],
    exchanges_by_short: dict[str, dict[str, Any]],
    assignments_by_short: dict[str, dict[str, Any]],
    source_segments_by_short: dict[str, dict[str, Any]],
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    segment_like = topic_unit_segment_like(
        topic_unit=topic_unit,
        dialogue_id=dialogue_id,
        exchanges_by_short=exchanges_by_short,
        assignments_by_short=assignments_by_short,
        source_segments_by_short=source_segments_by_short,
    )
    packet = build_packet(segment_like, exchanges_by_id, assignments_by_id, profiles_by_id)

    packet["packet_schema"] = "segment_evaluation_packet.v1b_mixed_granularity_overlay"
    packet["evaluation_unit_type"] = "topic_rollup"
    packet["judge_plan"] = topic_judge_plan()
    packet["topic_context"] = {
        "topic_unit_id": topic_unit.get("topic_unit_id"),
        "shared_parent_topic_node": topic_unit.get("shared_parent_topic_node"),
        "shared_parent_topic_path": list(topic_unit.get("shared_parent_topic_path") or []),
        "shared_parent_topic_label": topic_unit.get("shared_parent_topic_label"),
        "topic_formation_source": topic_unit.get("topic_formation_source"),
        "source_segment_ids": list(topic_unit.get("source_segment_ids") or []),
        "source_segment_count": topic_unit.get("source_segment_count"),
        "contains_fragmentation_risk": bool(topic_unit.get("contains_fragmentation_risk")),
        "fragmentation_reasons": list(topic_unit.get("fragmentation_reasons") or []),
        "topic_vote_summary": list(topic_unit.get("topic_vote_summary") or []),
    }
    packet["granularity_metadata"] = {
        "evaluation_unit_type": "topic_rollup",
        "source_segment_ids": list(topic_unit.get("source_segment_full_ids") or []),
        "source_segment_short_ids": list(topic_unit.get("source_segment_ids") or []),
        "source_segment_count": topic_unit.get("source_segment_count"),
        "source_topic_unit_ids": [topic_unit.get("topic_unit_id")],
        "granularity_note": (
            "Post-hoc topic-rollup unit prepared for arc-level judging. "
            "The underlying v15 segmentation remains unchanged."
        ),
    }
    packet["packet_trace"]["topic_rollup_trace"] = {
        "topic_unit_id": topic_unit.get("topic_unit_id"),
        "topic_formation_source": topic_unit.get("topic_formation_source"),
        "contains_fragmentation_risk": bool(topic_unit.get("contains_fragmentation_risk")),
        "source_segment_ids": list(topic_unit.get("source_segment_full_ids") or []),
    }
    return packet


def family_summary(
    *,
    family_id: str,
    packet_count: int,
    packets: list[dict[str, Any]],
    focus_partial_dimensions: list[str],
    focus_trust_flags: list[str],
) -> dict[str, Any]:
    return {
        "evaluation_family_id": family_id,
        "packet_count": packet_count,
        "evaluation_mode_counts": dict(sorted(Counter((packet.get("evaluation_target") or {}).get("mode") for packet in packets).items())),
        "segment_scope_counts": dict(sorted(Counter((packet.get("segmentation_metadata") or {}).get("segment_scope") for packet in packets).items())),
        "focus_partial_dimensions": list(focus_partial_dimensions),
        "focus_trust_flags": list(focus_trust_flags),
    }


def topic_family_audit(topic_units: list[dict[str, Any]], packets: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(topic_units) != len(packets):
        errors.append(f"topic_unit_packet_count_mismatch:units={len(topic_units)}:packets={len(packets)}")

    for packet in packets:
        segment_id = str(packet.get("segment_id") or "")
        topic_context = packet.get("topic_context") or {}
        if not topic_context.get("shared_parent_topic_path"):
            errors.append(f"missing_topic_path:{segment_id}")
        if not (packet.get("evaluation_target") or {}).get("evaluator_kc_set"):
            errors.append(f"empty_evaluator_kc_set:{segment_id}")
        if topic_context.get("contains_fragmentation_risk"):
            warnings.append(f"fragmentation_risk:{segment_id}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "topic_unit_count": len(topic_units),
        "packet_count": len(packets),
        "fragmentation_risk_count": sum(1 for unit in topic_units if bool(unit.get("contains_fragmentation_risk"))),
    }


def build_mixed_granularity_packets(
    *,
    run_dir: str | Path,
    profiles_path: str | Path,
    topic_rollup_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = read_json(run / "summary.json")
    exchanges = read_jsonl(run / "exchanges.jsonl")
    assignments = read_jsonl(run / "contactless_exchange_assignments.jsonl")
    segments = read_jsonl(run / "contactless_segments.jsonl")
    missing_review = read_jsonl(run / "review_queue_missing_kc_only.jsonl")
    profiles = read_jsonl(profiles_path)
    topic_units = read_topic_units(topic_rollup_path)

    dialogue_id = str(
        summary.get("dialogue_id")
        or (segments[0].get("dialogue_id") if segments else "")
        or (exchanges[0].get("dialogue_id") if exchanges else "")
        or run.name
    )
    excluded_exchange_ids = set(summary.get("excluded_meta_exchange_ids") or [])
    exchanges_by_id = {str(row["exchange_id"]): row for row in exchanges}
    assignments_by_id = {str(row["exchange_id"]): row for row in assignments}
    exchanges_by_short = {short_id(str(row["exchange_id"])): row for row in exchanges}
    assignments_by_short = {short_id(str(row["exchange_id"])): row for row in assignments}
    source_segments_by_short = {short_id(str(row["segment_id"])): row for row in segments}
    profiles_by_id = {str(row["unit_id"]): row for row in profiles if row.get("unit_id")}

    local_packets = [overlay_local_packet(build_packet(seg, exchanges_by_id, assignments_by_id, profiles_by_id)) for seg in segments]
    audit_exchanges = [row for row in exchanges if str(row.get("exchange_id")) not in excluded_exchange_ids]
    audit_assignments = [row for row in assignments if str(row.get("exchange_id")) not in excluded_exchange_ids]
    local_audit = audit_packets(local_packets, segments, audit_exchanges, audit_assignments, missing_review)
    local_summary = family_summary(
        family_id=LOCAL_FAMILY_ID,
        packet_count=len(local_packets),
        packets=local_packets,
        focus_partial_dimensions=LOCAL_PARTIAL_DIMENSIONS,
        focus_trust_flags=LOCAL_TRUST_FLAGS,
    )

    topic_packets = [
        build_topic_packet(
            topic_unit=topic_unit,
            dialogue_id=dialogue_id,
            exchanges_by_id=exchanges_by_id,
            assignments_by_id=assignments_by_id,
            exchanges_by_short=exchanges_by_short,
            assignments_by_short=assignments_by_short,
            source_segments_by_short=source_segments_by_short,
            profiles_by_id=profiles_by_id,
        )
        for topic_unit in topic_units
    ]
    topic_audit = topic_family_audit(topic_units, topic_packets)
    topic_summary = family_summary(
        family_id=TOPIC_FAMILY_ID,
        packet_count=len(topic_packets),
        packets=topic_packets,
        focus_partial_dimensions=ARC_PARTIAL_DIMENSIONS,
        focus_trust_flags=ARC_TRUST_FLAGS,
    )

    if local_audit.get("status") != "PASS":
        raise RuntimeError("Local mixed-granularity packet audit failed: " + "; ".join(local_audit.get("errors", [])[:10]))
    if topic_audit.get("status") != "PASS":
        raise RuntimeError("Topic mixed-granularity packet audit failed: " + "; ".join(topic_audit.get("errors", [])[:10]))

    local_dir = out / "kc_segment_local"
    topic_dir = out / "topic_rollup_arc"
    local_dir.mkdir(parents=True, exist_ok=True)
    topic_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(local_dir / "segment_evaluation_packets.jsonl", local_packets)
    write_json(local_dir / "packet_summary.json", local_summary)
    write_json(local_dir / "packet_audit.json", local_audit)

    write_jsonl(topic_dir / "segment_evaluation_packets.jsonl", topic_packets)
    write_json(topic_dir / "packet_summary.json", topic_summary)
    write_json(topic_dir / "packet_audit.json", topic_audit)

    mixed_summary = {
        "packet_pipeline": "segment_evaluation_packets_mixed_granularity_v1",
        "source_segmentation_pipeline": summary.get("pipeline"),
        "source_run_dir": str(run),
        "profiles_path": str(Path(profiles_path)),
        "topic_rollup_path": str(Path(topic_rollup_path)),
        "baseline_segment_count": len(segments),
        "topic_unit_count": len(topic_units),
        "local_packet_count": len(local_packets),
        "topic_packet_count": len(topic_packets),
        "local_family_id": LOCAL_FAMILY_ID,
        "topic_family_id": TOPIC_FAMILY_ID,
    }
    write_json(out / "mixed_granularity_summary.json", mixed_summary)
    write_json(out / "mixed_granularity_plan.json", {
        "plan_version": "mixed_granularity_judge_plan_v1",
        "local_family": local_judge_plan(),
        "topic_family": topic_judge_plan(),
        "source_run_dir": str(run),
        "topic_rollup_path": str(Path(topic_rollup_path)),
        "notes": [
            "This is a post-hoc additive preparation layer only.",
            "It does not change v15 segmentation boundaries or dominant_kc_id selection.",
            "Local-dimension packets and arc-dimension packets are prepared as separate families for downstream judging.",
            "Existing prompt/response contract remains unchanged and may need a later versioned adapter if the judge should score only a subset of dimensions per family.",
        ],
    })
    write_json(out / "mixed_granularity_manifest.json", {
        "schema": "mixed_granularity_evaluation_packet_manifest.v1",
        "packet_pipeline": "segment_evaluation_packets_mixed_granularity_v1",
        "outputs": {
            "local_packet_dir": "kc_segment_local",
            "topic_packet_dir": "topic_rollup_arc",
            "mixed_granularity_summary": "mixed_granularity_summary.json",
            "mixed_granularity_plan": "mixed_granularity_plan.json",
        },
        "local_family": local_summary,
        "topic_family": topic_summary,
        "local_audit": local_audit,
        "topic_audit": topic_audit,
    })

    report_lines = [
        "# Mixed-Granularity Judge Packet Prep v1",
        "",
        f"- Source run: `{run}`",
        f"- Topic rollup: `{Path(topic_rollup_path)}`",
        f"- Local packet family: `{LOCAL_FAMILY_ID}` with `{len(local_packets)}` packets",
        f"- Topic packet family: `{TOPIC_FAMILY_ID}` with `{len(topic_packets)}` packets",
        f"- Local focus dimensions: `{LOCAL_PARTIAL_DIMENSIONS}`",
        f"- Local focus trust flags: `{LOCAL_TRUST_FLAGS}`",
        f"- Topic focus dimensions: `{ARC_PARTIAL_DIMENSIONS}`",
        f"- Topic focus trust flags: `{ARC_TRUST_FLAGS}`",
        f"- Topic fragmentation-risk units carried forward for review: `{topic_audit['fragmentation_risk_count']}`",
        "",
        "## Notes",
        "",
        "- Local packets preserve the original v15 segment granularity.",
        "- Topic packets reuse the frozen topic rollup and keep explicit formation/fallback trace fields.",
        "- No gold labels are used in the production packet logic.",
    ]
    (out / "mixed_granularity_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "out_dir": str(out),
        "summary": mixed_summary,
        "local_summary": local_summary,
        "topic_summary": topic_summary,
        "local_audit": local_audit,
        "topic_audit": topic_audit,
    }
