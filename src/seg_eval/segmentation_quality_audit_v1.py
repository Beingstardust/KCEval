from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import Any


SEVERITY_RANK = {
    "evaluation_ready": 0,
    "review_recommended": 1,
    "hard_review_required": 2,
}

HARD_ACTIONS = {
    "pass2_broad_multibranch_closed_set",
    "pass2_context_dependent_anchor_context_only",
}

RECOMMENDED_ACTIONS = {
    "pass2_branch_guided_best_candidate",
    "pass2_context_dependent_anaphora_bridge_rescue",
    "pass2_left_context_continuation",
    "pass2_right_context_continuation",
}

META_RE = re.compile(
    r"\b("
    r"conversation|llm tutor|real human student|respond like|ask messy questions|teaching style|"
    r"simulate|simulation|dialogue|transcript"
    r")\b",
    re.IGNORECASE,
)

FORMULA_OR_CALC_RE = re.compile(
    r"\b("
    r"calculate|compute|formula|interval|z\s*=|z-statistic|laplace|smoothing|centroid|"
    r"manhattan|euclidean|cosine|f1|precision|recall|specificity|sensitivity|"
    r"prior|conditional probability|likelihood|entropy|gain ratio|information gain"
    r")\b",
    re.IGNORECASE,
)

RECAP_RE = re.compile(
    r"\b("
    r"connect all|final strategy|summarize|summary|recap|overall|all these exercises|"
    r"remember everything|four passes"
    r")\b",
    re.IGNORECASE,
)

TOPIC_SHIFT_RE = re.compile(
    r"\b("
    r"now exercise|exercise \d+|next|what about|switch|different from|instead|"
    r"classification|clustering|naive bayes|decision tree|random forest|k-means|dbscan|"
    r"confidence interval|z-statistic|precision|recall|f1|cosine|manhattan|euclidean"
    r")\b",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return default


def _join_text(*parts: Any) -> str:
    texts: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            texts.extend(str(x) for x in part)
        else:
            texts.append(str(part))
    return " ".join(texts)


def _short(text: str, n: int = 180) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= n else clean[: n - 3] + "..."


def _severity_max(values: list[str]) -> str:
    if not values:
        return "evaluation_ready"
    return max(values, key=lambda item: SEVERITY_RANK.get(item, 0))


def _extract_exchange_id(row: dict[str, Any]) -> str:
    return str(_first_present(row, ["exchange_id", "id"], ""))


def _extract_segment_id(row: dict[str, Any]) -> str:
    return str(_first_present(row, ["segment_id", "id"], ""))


def _extract_member_exchange_ids(segment: dict[str, Any]) -> list[str]:
    for key in ["member_exchange_ids", "exchange_ids", "exchanges"]:
        value = segment.get(key)
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    eid = _extract_exchange_id(item)
                    if eid:
                        result.append(eid)
                elif item:
                    result.append(str(item))
            return result
    start = segment.get("start_exchange_id")
    end = segment.get("end_exchange_id")
    if start and end and start == end:
        return [str(start)]
    return []


def _extract_assignment_fields(row: dict[str, Any]) -> dict[str, Any]:
    resolved_kc_id = _first_present(row, ["resolved_kc_id", "dominant_kc_id", "kc_id", "primary_kc_id"], "")
    resolved_kc_name = _first_present(row, ["resolved_kc_name", "dominant_kc_name", "kc_name", "primary_kc_name"], "")
    confidence = str(_first_present(row, ["confidence_band", "band", "final_confidence_band"], "") or "").lower()
    scope = str(_first_present(row, ["assignment_scope", "segment_scope", "scope"], "") or "").lower()
    action = str(_first_present(row, ["resolution_action", "action", "final_action"], "") or "")
    evaluator_kc_set = _as_list(_first_present(row, ["evaluator_kc_set", "kc_set", "secondary_kcs"], []))
    context_dependency = row.get("context_dependency") or row.get("context_dependency_flags") or row.get("context_flags") or []
    if isinstance(context_dependency, dict):
        context_dependency = context_dependency.get("flags") or context_dependency.get("labels") or list(context_dependency.values())
    return {
        "resolved_kc_id": resolved_kc_id,
        "resolved_kc_name": resolved_kc_name,
        "confidence_band": confidence,
        "assignment_scope": scope,
        "resolution_action": action,
        "evaluator_kc_set": evaluator_kc_set,
        "context_dependency": _as_list(context_dependency),
    }


def _extract_exchange_text(row: dict[str, Any], normalized_row: dict[str, Any] | None = None) -> dict[str, str]:
    norm = normalized_row or {}
    student = _first_present(row, ["student_text", "student", "student_preview"], None)
    tutor = _first_present(row, ["tutor_text", "tutor", "tutor_preview"], None)
    if student is None:
        student = _first_present(norm, ["student_text", "student", "student_preview"], "")
    if tutor is None:
        tutor = _first_present(norm, ["tutor_text", "tutor", "tutor_preview"], "")
    return {
        "student_text": str(student or ""),
        "tutor_text": str(tutor or ""),
    }


def _extract_style_tags(row: dict[str, Any], normalized_row: dict[str, Any] | None = None) -> list[str]:
    tags = _as_list(row.get("teaching_style_tags"))
    if not tags and normalized_row:
        tags = _as_list(normalized_row.get("teaching_style_tags"))
    return [str(tag) for tag in tags if tag not in (None, "")]


def classify_exchange(row: dict[str, Any], normalized_row: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = _extract_assignment_fields(row)
    text_fields = _extract_exchange_text(row, normalized_row)
    style_tags = _extract_style_tags(row, normalized_row)

    confidence = fields["confidence_band"]
    scope = fields["assignment_scope"]
    action = fields["resolution_action"]
    context_flags = [str(x) for x in fields["context_dependency"]]

    hard_reasons: list[str] = []
    recommended_reasons: list[str] = []
    suspect_reasons: list[str] = []

    if not fields["resolved_kc_id"]:
        hard_reasons.append("missing_resolved_kc")
    if confidence == "low":
        hard_reasons.append("low_confidence")
    if scope == "broad_kc_set":
        hard_reasons.append("broad_kc_set")
    if action in HARD_ACTIONS:
        hard_reasons.append(action)

    if scope == "branch_context":
        recommended_reasons.append("branch_context")
    if action in RECOMMENDED_ACTIONS:
        recommended_reasons.append(action)
    if any("possible_topic_shift" in flag for flag in context_flags):
        recommended_reasons.append("possible_topic_shift_context_flag")

    combined = _join_text(text_fields["student_text"], text_fields["tutor_text"])
    combined_lower = combined.lower()
    tags_lower = {tag.lower() for tag in style_tags}

    if META_RE.search(combined):
        suspect_reasons.append("meta_or_orientation_turn_forced_to_kc")
        hard_reasons.append("meta_or_orientation_turn")

    if action in RECOMMENDED_ACTIONS and FORMULA_OR_CALC_RE.search(combined):
        suspect_reasons.append("formula_or_calculation_turn_with_context_rescue")
        recommended_reasons.append("formula_or_calculation_context_rescue")

    if RECAP_RE.search(combined) or "recap_synthesis" in tags_lower:
        suspect_reasons.append("recap_or_final_strategy_forced_to_kc")
        recommended_reasons.append("recap_or_final_strategy")

    if action in RECOMMENDED_ACTIONS and TOPIC_SHIFT_RE.search(combined):
        suspect_reasons.append("context_rescue_with_possible_topic_shift")
        recommended_reasons.append("context_rescue_with_possible_topic_shift")

    if len(fields["evaluator_kc_set"]) >= 10:
        suspect_reasons.append("wide_evaluator_kc_set")
        recommended_reasons.append("wide_evaluator_kc_set")

    if hard_reasons:
        status = "hard_review_required"
    elif recommended_reasons:
        status = "review_recommended"
    else:
        status = "evaluation_ready"

    return {
        "exchange_id": _extract_exchange_id(row),
        "evaluation_readiness": status,
        "hard_reasons": sorted(set(hard_reasons)),
        "recommended_reasons": sorted(set(recommended_reasons)),
        "suspect_reasons": sorted(set(suspect_reasons)),
        "resolved_kc_id": fields["resolved_kc_id"],
        "resolved_kc_name": fields["resolved_kc_name"],
        "confidence_band": confidence,
        "assignment_scope": scope,
        "resolution_action": action,
        "student_preview": _short(text_fields["student_text"]),
        "tutor_preview": _short(text_fields["tutor_text"]),
        "teaching_style_tags": style_tags,
        "context_dependency": context_flags,
        "evaluator_kc_set_size": len(fields["evaluator_kc_set"]),
    }


def build_quality_audit(
    run_dir: str | Path,
    normalized_exchanges_path: str | Path | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    summary = _read_json(run / "summary.json")
    segments = _read_jsonl(run / "segments.jsonl")
    exchanges = _read_jsonl(run / "exchanges.jsonl")
    assignments = _read_jsonl(run / "contactless_exchange_assignments.jsonl")
    if not assignments:
        assignments = _read_jsonl(run / "exchange_assignments.jsonl")
    rescue_trace = _read_jsonl(run / "context_dependent_rescue_trace.jsonl")

    normalized_rows = _read_jsonl(Path(normalized_exchanges_path)) if normalized_exchanges_path else []
    normalized_by_id = {str(row.get("exchange_id")): row for row in normalized_rows if row.get("exchange_id")}

    exchange_text_by_id = {str(row.get("exchange_id")): row for row in exchanges if row.get("exchange_id")}
    assignment_rows: list[dict[str, Any]] = []

    for row in assignments:
        eid = _extract_exchange_id(row)
        merged = dict(row)
        if eid in exchange_text_by_id:
            for key, value in exchange_text_by_id[eid].items():
                merged.setdefault(key, value)
        assignment_rows.append(merged)

    if not assignment_rows and exchanges:
        assignment_rows = exchanges

    exchange_assessments = [
        classify_exchange(row, normalized_by_id.get(_extract_exchange_id(row)))
        for row in assignment_rows
    ]

    exchange_by_id = {row["exchange_id"]: row for row in exchange_assessments if row["exchange_id"]}

    segment_assessments: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = _extract_segment_id(segment)
        member_ids = _extract_member_exchange_ids(segment)
        members = [exchange_by_id[eid] for eid in member_ids if eid in exchange_by_id]
        status = _severity_max([member["evaluation_readiness"] for member in members])
        hard_reasons = sorted(set(reason for member in members for reason in member["hard_reasons"]))
        recommended_reasons = sorted(set(reason for member in members for reason in member["recommended_reasons"]))
        suspect_reasons = sorted(set(reason for member in members for reason in member["suspect_reasons"]))

        fields = _extract_assignment_fields(segment)

        segment_assessments.append({
            "segment_id": segment_id,
            "evaluation_readiness": status,
            "member_exchange_ids": member_ids,
            "resolved_kc_id": fields["resolved_kc_id"],
            "resolved_kc_name": fields["resolved_kc_name"],
            "confidence_bands": _as_list(_first_present(segment, ["confidence_bands", "confidence_band"], [])),
            "segment_scope": _first_present(segment, ["segment_scope", "assignment_scope", "scope"], ""),
            "resolution_actions": _as_list(_first_present(segment, ["resolution_actions", "resolution_action"], [])),
            "hard_reasons": hard_reasons,
            "recommended_reasons": recommended_reasons,
            "suspect_reasons": suspect_reasons,
        })

    exchange_status_counts = {
        "evaluation_ready": sum(1 for row in exchange_assessments if row["evaluation_readiness"] == "evaluation_ready"),
        "review_recommended": sum(1 for row in exchange_assessments if row["evaluation_readiness"] == "review_recommended"),
        "hard_review_required": sum(1 for row in exchange_assessments if row["evaluation_readiness"] == "hard_review_required"),
    }
    segment_status_counts = {
        "evaluation_ready": sum(1 for row in segment_assessments if row["evaluation_readiness"] == "evaluation_ready"),
        "review_recommended": sum(1 for row in segment_assessments if row["evaluation_readiness"] == "review_recommended"),
        "hard_review_required": sum(1 for row in segment_assessments if row["evaluation_readiness"] == "hard_review_required"),
    }

    suspect_examples = [
        row for row in exchange_assessments
        if row["suspect_reasons"] or row["evaluation_readiness"] != "evaluation_ready"
    ][:30]

    return {
        "schema_version": "segmentation_quality_audit_v1",
        "run_dir": str(run),
        "summary": summary,
        "input_counts": {
            "summary_exchange_count": summary.get("exchange_count"),
            "summary_segment_count": summary.get("segment_count"),
            "assignment_row_count": len(assignment_rows),
            "segment_row_count": len(segments),
            "rescue_trace_row_count": len(rescue_trace),
        },
        "exchange_status_counts": exchange_status_counts,
        "segment_status_counts": segment_status_counts,
        "exchange_assessments": exchange_assessments,
        "segment_assessments": segment_assessments,
        "suspect_assignment_examples": suspect_examples,
    }


def render_quality_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary") or {}
    exchange_counts = audit["exchange_status_counts"]
    segment_counts = audit["segment_status_counts"]

    lines = [
        "# Segmentation quality audit",
        "",
        "## Run identity",
        "",
        f"- Run directory: `{audit.get('run_dir')}`",
        f"- Pipeline: `{summary.get('pipeline', 'n/a')}`",
        f"- Exchange count: `{summary.get('exchange_count', 'n/a')}`",
        f"- Segment count: `{summary.get('segment_count', 'n/a')}`",
        "",
        "## Evaluation readiness summary",
        "",
        "| Level | Exchanges | Segments |",
        "|---|---:|---:|",
        f"| evaluation_ready | {exchange_counts['evaluation_ready']} | {segment_counts['evaluation_ready']} |",
        f"| review_recommended | {exchange_counts['review_recommended']} | {segment_counts['review_recommended']} |",
        f"| hard_review_required | {exchange_counts['hard_review_required']} | {segment_counts['hard_review_required']} |",
        "",
        "## Original uncertainty counters",
        "",
        f"- Confidence bands: `{summary.get('confidence_band_counts', {})}`",
        f"- Assignment scopes: `{summary.get('assignment_scope_counts', {})}`",
        f"- Resolution actions: `{summary.get('resolution_action_counts', {})}`",
        f"- Context-dependent rescue count: `{summary.get('context_dependent_rescue_count', 'n/a')}`",
        f"- Original missing-KC review count: `{summary.get('missing_kc_review_count', 'n/a')}`",
        "",
        "## Suspicious assignment examples",
        "",
        "| Exchange | Readiness | Resolved KC | Confidence | Scope | Action | Reasons | Student preview | Tutor preview |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    examples = audit.get("suspect_assignment_examples") or []
    if not examples:
        lines.append("| none | evaluation_ready | | | | | | | |")
    else:
        for row in examples:
            reasons = sorted(set(row.get("hard_reasons", []) + row.get("recommended_reasons", []) + row.get("suspect_reasons", [])))
            lines.append(
                "| `{}` | `{}` | `{}` {} | `{}` | `{}` | `{}` | {} | {} | {} |".format(
                    row.get("exchange_id", ""),
                    row.get("evaluation_readiness", ""),
                    row.get("resolved_kc_id", ""),
                    row.get("resolved_kc_name", ""),
                    row.get("confidence_band", ""),
                    row.get("assignment_scope", ""),
                    row.get("resolution_action", ""),
                    ", ".join(f"`{reason}`" for reason in reasons),
                    (row.get("student_preview") or "").replace("|", "\\|"),
                    (row.get("tutor_preview") or "").replace("|", "\\|"),
                )
            )

    lines.extend([
        "",
        "## Segment readiness map",
        "",
        "| Segment | Readiness | Resolved KC | Exchanges | Reasons |",
        "|---|---|---|---:|---|",
    ])

    for row in audit.get("segment_assessments", []):
        reasons = sorted(set(row.get("hard_reasons", []) + row.get("recommended_reasons", []) + row.get("suspect_reasons", [])))
        lines.append(
            "| `{}` | `{}` | `{}` {} | {} | {} |".format(
                row.get("segment_id", ""),
                row.get("evaluation_readiness", ""),
                row.get("resolved_kc_id", ""),
                row.get("resolved_kc_name", ""),
                len(row.get("member_exchange_ids", [])),
                ", ".join(f"`{reason}`" for reason in reasons) if reasons else "`none`",
            )
        )

    lines.append("")
    return "\n".join(lines)


def write_quality_audit_outputs(
    run_dir: str | Path,
    out_dir: str | Path,
    normalized_exchanges_path: str | Path | None = None,
    existing_focus_report_path: str | Path | None = None,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    audit = build_quality_audit(run_dir, normalized_exchanges_path)
    json_path = out / "segmentation_quality_audit.json"
    md_path = out / "segmentation_quality_audit.md"
    enhanced_report_path = out / "segmentation_focus_report_with_quality.md"

    _write_json(json_path, audit)
    audit_md = render_quality_audit_markdown(audit)
    md_path.write_text(audit_md, encoding="utf-8")

    existing_text = ""
    if existing_focus_report_path and Path(existing_focus_report_path).exists():
        existing_text = Path(existing_focus_report_path).read_text(encoding="utf-8")
    elif (Path(run_dir) / "segmentation_focus_report.md").exists():
        existing_text = (Path(run_dir) / "segmentation_focus_report.md").read_text(encoding="utf-8")

    enhanced_report_path.write_text(
        audit_md + "\n\n---\n\n" + existing_text,
        encoding="utf-8",
    )

    return {
        "quality_audit_json": str(json_path),
        "quality_audit_md": str(md_path),
        "enhanced_focus_report": str(enhanced_report_path),
    }
