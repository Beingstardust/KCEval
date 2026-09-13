from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import re


RECAP_RE = re.compile(
    r"\b(connect all|recap|summary|summarize|final strategy|overall|all these exercises)\b",
    re.IGNORECASE,
)

META_RE = re.compile(
    r"\b(conversation|llm tutor|real human student|dialogue|transcript|simulate|respond like)\b",
    re.IGNORECASE,
)

FORMULA_RE = re.compile(
    r"\b(calculate|compute|formula|interval|z-statistic|z\s*=|laplace|smoothing|centroid|"
    r"precision|recall|f1|specificity|sensitivity|prior|conditional probability|likelihood|"
    r"entropy|gain ratio|information gain|manhattan|euclidean|cosine|k-means)\b",
    re.IGNORECASE,
)

TOPIC_CUE_RE = re.compile(
    r"\b(classification|clustering|naive bayes|bayes|decision tree|random forest|k-means|"
    r"bisecting|confidence interval|z-statistic|confusion matrix|precision|recall|f1|"
    r"gain ratio|information gain|entropy|pruning|pessimistic|reduced-error|centroid|"
    r"manhattan|euclidean|cosine|laplace|prior|conditional)\b",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return default


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def norm_id(value: Any) -> str:
    return str(value or "")


def short(text: str, n: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= n else clean[: n - 3] + "..."


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]+", text or "")
        if len(token) >= 3
    }


def exchange_id(row: dict[str, Any]) -> str:
    return norm_id(first_present(row, ["exchange_id", "id", "unit_id"], ""))


def candidate_id(row: dict[str, Any]) -> str:
    return norm_id(first_present(row, ["unit_id", "kc_id", "candidate_kc_id", "resolved_kc_id", "dominant_kc_id", "id"], ""))


def candidate_name(row: dict[str, Any]) -> str:
    return norm_id(first_present(row, ["canonical_name", "kc_name", "candidate_kc_name", "resolved_kc_name", "dominant_kc_name", "name"], ""))


def candidate_score(row: dict[str, Any]) -> float | None:
    for key in [
        "composite_score",
        "fusion_score",
        "score",
        "final_score",
        "combined_score",
        "hybrid_retrieval_score",
        "bm25_score",
        "profile_native_score",
        "char_tfidf_score",
    ]:
        value = row.get(key)
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
    return None


def find_first_jsonl_with_any(run_dir: Path, name_hints: list[str]) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    for path in sorted(run_dir.glob("*.jsonl")):
        lower = path.name.lower()
        if any(hint in lower for hint in name_hints):
            rows = read_jsonl(path)
            if rows:
                return rows
    return []


def load_assignments(run_dir: Path) -> list[dict[str, Any]]:
    for name in [
        "contactless_exchange_assignments.jsonl",
        "exchange_assignments.jsonl",
        "resolved_assignments.jsonl",
        "assignments.jsonl",
    ]:
        path = run_dir / name
        rows = read_jsonl(path)
        if rows:
            return rows
    return find_first_jsonl_with_any(run_dir, ["assignment"])


def load_exchanges(run_dir: Path) -> list[dict[str, Any]]:
    for name in ["exchanges.jsonl", "exchange_units.jsonl"]:
        rows = read_jsonl(run_dir / name)
        if rows:
            return rows
    return find_first_jsonl_with_any(run_dir, ["exchange"])


def load_candidates_from_dir(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_dir.exists():
        return rows
    for path in sorted(run_dir.glob("*.jsonl")):
        lower = path.name.lower()
        if "candidate" in lower or "pool" in lower:
            try:
                rows.extend(read_jsonl(path))
            except Exception:
                pass
    return rows


def extract_nested_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for key in [
        "candidates",
        "candidate_pool",
        "top_candidates",
        "pass1_candidates",
        "kc_candidates",
        "ranked_candidates",
    ]:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidates.append(item)

    for key in ["candidate", "top1", "top_candidate", "final_candidate", "best_candidate"]:
        value = row.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    return candidates


def group_candidates(candidate_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in candidate_rows:
        eid = exchange_id(row)
        if not eid:
            for key in ["source_exchange_id", "exchange"]:
                if row.get(key):
                    eid = str(row.get(key))
                    break

        if not eid:
            continue

        nested = extract_nested_candidates(row)

        if nested:
            grouped.setdefault(eid, []).extend(nested)
        else:
            grouped.setdefault(eid, []).append(row)

    return grouped


def assignment_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolved_kc_id": first_present(row, ["resolved_kc_id", "dominant_kc_id", "kc_id", "primary_kc_id"], ""),
        "resolved_kc_name": first_present(row, ["resolved_kc_name", "dominant_kc_name", "kc_name", "primary_kc_name"], ""),
        "confidence_band": str(first_present(row, ["confidence_band", "band", "final_confidence_band"], "") or "").lower(),
        "assignment_scope": str(first_present(row, ["assignment_scope", "segment_scope", "scope"], "") or "").lower(),
        "resolution_action": str(first_present(row, ["resolution_action", "action", "final_action"], "") or ""),
        "evaluator_kc_set": as_list(first_present(row, ["evaluator_kc_set", "kc_set", "secondary_kcs"], [])),
    }


def row_text(row: dict[str, Any], exchange_row: dict[str, Any] | None = None, normalized_row: dict[str, Any] | None = None) -> dict[str, str]:
    exchange_row = exchange_row or {}
    normalized_row = normalized_row or {}

    student = first_present(row, ["student_text", "student", "student_preview"], None)
    tutor = first_present(row, ["tutor_text", "tutor", "tutor_preview"], None)

    if student is None:
        student = first_present(exchange_row, ["student_text", "student", "student_preview"], None)
    if tutor is None:
        tutor = first_present(exchange_row, ["tutor_text", "tutor", "tutor_preview"], None)

    if student is None:
        student = first_present(normalized_row, ["student_text", "student", "student_preview"], "")
    if tutor is None:
        tutor = first_present(normalized_row, ["tutor_text", "tutor", "tutor_preview"], "")

    return {
        "student_text": str(student or ""),
        "tutor_text": str(tutor or ""),
        "combined_text": f"{student or ''}\n{tutor or ''}",
    }


def flatten_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    cid = candidate_id(candidate)
    cname = candidate_name(candidate)
    score = candidate_score(candidate)
    branch = first_present(candidate, ["branch", "topic_branch", "topic_path"], "")
    action = first_present(candidate, ["action", "source", "route"], "")
    return {
        "rank": rank,
        "kc_id": cid,
        "kc_name": cname,
        "score": score,
        "branch": branch,
        "route_or_source": action,
    }


def select_candidates_for_exchange(
    assignment_row: dict[str, Any],
    external_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    nested = extract_nested_candidates(assignment_row)

    if nested:
        raw_candidates = nested
    else:
        raw_candidates = []
        for item in external_candidates:
            if isinstance(item, dict):
                item_nested = extract_nested_candidates(item)
                if item_nested:
                    raw_candidates.extend(item_nested)
                else:
                    raw_candidates.append(item)

    flattened = []
    for idx, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            continue
        flattened.append(flatten_candidate(candidate, idx + 1))

    def sort_key(item: dict[str, Any]):
        score = item.get("score")
        return (score is None, -(score or 0), item.get("rank") or 9999)

    flattened = sorted(flattened, key=sort_key)
    for idx, item in enumerate(flattened, start=1):
        item["rank"] = idx

    seen = set()
    deduped = []
    for item in flattened:
        key = (item.get("kc_id"), item.get("kc_name"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:15]


def classify_failure_mode(
    fields: dict[str, Any],
    candidates: list[dict[str, Any]],
    text: dict[str, str],
    readiness: str,
) -> tuple[list[str], list[str]]:
    modes: list[str] = []
    notes: list[str] = []

    resolved_id = str(fields.get("resolved_kc_id") or "")
    resolved_name = str(fields.get("resolved_kc_name") or "")
    confidence = fields.get("confidence_band")
    scope = fields.get("assignment_scope")
    action = fields.get("resolution_action")

    if not candidates:
        modes.append("candidate_trace_missing_or_not_exported")
        notes.append("No candidate list was available in the inspected artifacts for this exchange.")

    if confidence == "low":
        modes.append("low_confidence_assignment")

    if scope == "broad_kc_set":
        modes.append("broad_kc_set_fallback")

    if action in {
        "pass2_context_dependent_anaphora_bridge_rescue",
        "pass2_context_dependent_anchor_context_only",
        "pass2_left_context_continuation",
        "pass2_right_context_continuation",
    }:
        modes.append("context_rescue_or_continuation_used")

    if action == "pass2_context_dependent_anchor_context_only":
        modes.append("anchor_preserved_context_only")

    top_candidate = candidates[0] if candidates else None
    if top_candidate:
        top_id = str(top_candidate.get("kc_id") or "")
        if resolved_id and top_id and resolved_id != top_id:
            modes.append("resolved_assignment_not_top_local_candidate")
            notes.append(f"Top available candidate is {top_id}, but resolved KC is {resolved_id}.")

    combined = text.get("combined_text", "")

    if META_RE.search(combined):
        modes.append("meta_or_orientation_turn")
    if RECAP_RE.search(combined):
        modes.append("recap_or_strategy_turn")
    if FORMULA_RE.search(combined):
        modes.append("formula_or_calculation_turn")
    if TOPIC_CUE_RE.search(combined):
        cue_terms = sorted(set(match.group(0).lower() for match in TOPIC_CUE_RE.finditer(combined)))
        assigned_tokens = tokenize(resolved_name)
        cue_tokens = tokenize(" ".join(cue_terms))
        if cue_terms and assigned_tokens and not (cue_tokens & assigned_tokens):
            modes.append("local_topic_cues_weakly_match_resolved_name")
            notes.append("Explicit local topic cues do not overlap strongly with the resolved KC name.")

    if readiness == "hard_review_required" and not modes:
        modes.append("hard_review_without_specific_detector")

    return sorted(set(modes)), notes


def load_quality_assessments(quality_json_path: Path) -> dict[str, dict[str, Any]]:
    quality = read_json(quality_json_path)
    rows = quality.get("exchange_assessments") or []
    return {
        str(row.get("exchange_id")): row
        for row in rows
        if row.get("exchange_id")
    }


def build_candidate_diagnostic(
    run_root: str | Path,
    normalized_exchanges_path: str | Path | None = None,
    quality_json_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    v9c = root / "v9c"
    v9e = root / "v9e"

    v9c_summary = read_json(v9c / "summary.json")
    v9e_summary = read_json(v9e / "summary.json")

    v9e_assignments = load_assignments(v9e)
    v9c_assignments = load_assignments(v9c)
    v9e_exchanges = load_exchanges(v9e)
    v9c_exchanges = load_exchanges(v9c)
    normalized = read_jsonl(Path(normalized_exchanges_path)) if normalized_exchanges_path else []

    v9c_candidates = load_candidates_from_dir(v9c)
    external_candidates_by_exchange = group_candidates(v9c_candidates)

    v9c_by_id = {exchange_id(row): row for row in v9c_assignments if exchange_id(row)}
    exchange_by_id = {exchange_id(row): row for row in v9e_exchanges + v9c_exchanges if exchange_id(row)}
    normalized_by_id = {str(row.get("exchange_id")): row for row in normalized if row.get("exchange_id")}

    quality_by_id = load_quality_assessments(Path(quality_json_path)) if quality_json_path else {}

    diagnostics: list[dict[str, Any]] = []

    for row in v9e_assignments:
        eid = exchange_id(row)
        if not eid:
            continue

        fields = assignment_fields(row)
        quality = quality_by_id.get(eid, {})
        readiness = quality.get("evaluation_readiness") or (
            "hard_review_required"
            if fields["confidence_band"] == "low" or fields["assignment_scope"] == "broad_kc_set"
            else "evaluation_ready"
        )

        if readiness == "evaluation_ready":
            continue

        v9c_row = v9c_by_id.get(eid, {})
        text = row_text(row, exchange_by_id.get(eid), normalized_by_id.get(eid))

        candidates = select_candidates_for_exchange(
            assignment_row=v9c_row or row,
            external_candidates=external_candidates_by_exchange.get(eid, []),
        )

        modes, notes = classify_failure_mode(fields, candidates, text, readiness)

        diagnostics.append({
            "exchange_id": eid,
            "evaluation_readiness": readiness,
            "resolved_kc_id": fields["resolved_kc_id"],
            "resolved_kc_name": fields["resolved_kc_name"],
            "confidence_band": fields["confidence_band"],
            "assignment_scope": fields["assignment_scope"],
            "resolution_action": fields["resolution_action"],
            "evaluator_kc_set_size": len(fields["evaluator_kc_set"]),
            "student_preview": short(text["student_text"]),
            "tutor_preview": short(text["tutor_text"]),
            "candidate_trace_available": bool(candidates),
            "top_candidates": candidates[:8],
            "diagnostic_modes": modes,
            "diagnostic_notes": notes,
            "quality_reasons": sorted(set(
                as_list(quality.get("hard_reasons")) +
                as_list(quality.get("recommended_reasons")) +
                as_list(quality.get("suspect_reasons"))
            )),
        })

    mode_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}

    for row in diagnostics:
        readiness_counts[row["evaluation_readiness"]] = readiness_counts.get(row["evaluation_readiness"], 0) + 1
        for mode in row["diagnostic_modes"]:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        "schema_version": "segmentation_candidate_diagnostic_v1",
        "run_root": str(root),
        "v9c_summary": v9c_summary,
        "v9e_summary": v9e_summary,
        "diagnosed_exchange_count": len(diagnostics),
        "readiness_counts": readiness_counts,
        "diagnostic_mode_counts": dict(sorted(mode_counts.items())),
        "candidate_artifact_row_count": len(v9c_candidates),
        "diagnostics": diagnostics,
    }


def render_candidate_diagnostic_markdown(diag: dict[str, Any]) -> str:
    lines = [
        "# Candidate decision diagnostic audit",
        "",
        "## Run summary",
        "",
        f"- Run root: `{diag.get('run_root')}`",
        f"- V9C segment count: `{(diag.get('v9c_summary') or {}).get('segment_count', 'n/a')}`",
        f"- V9E segment count: `{(diag.get('v9e_summary') or {}).get('segment_count', 'n/a')}`",
        f"- Diagnosed exchange count: `{diag.get('diagnosed_exchange_count')}`",
        f"- Candidate artifact row count: `{diag.get('candidate_artifact_row_count')}`",
        "",
        "## Readiness counts among diagnosed exchanges",
        "",
        "| Readiness | Count |",
        "|---|---:|",
    ]

    for key, value in sorted((diag.get("readiness_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "## Diagnostic mode counts",
        "",
        "| Diagnostic mode | Count |",
        "|---|---:|",
    ])

    for key, value in sorted((diag.get("diagnostic_mode_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "## Exchange diagnostics",
        "",
        "| Exchange | Readiness | Resolved KC | Confidence | Scope | Action | Modes | Top candidates | Student preview | Tutor preview |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for row in diag.get("diagnostics", []):
        top = []
        for cand in row.get("top_candidates", [])[:5]:
            label = f"{cand.get('rank')}. {cand.get('kc_id')} {cand.get('kc_name')}"
            if cand.get("score") is not None:
                label += f" ({cand.get('score'):.4f})"
            top.append(label)

        lines.append(
            "| `{}` | `{}` | `{}` {} | `{}` | `{}` | `{}` | {} | {} | {} | {} |".format(
                row.get("exchange_id", ""),
                row.get("evaluation_readiness", ""),
                row.get("resolved_kc_id", ""),
                row.get("resolved_kc_name", ""),
                row.get("confidence_band", ""),
                row.get("assignment_scope", ""),
                row.get("resolution_action", ""),
                ", ".join(f"`{mode}`" for mode in row.get("diagnostic_modes", [])),
                "<br>".join(top).replace("|", "\\|") if top else "`no candidate trace`",
                str(row.get("student_preview", "")).replace("|", "\\|"),
                str(row.get("tutor_preview", "")).replace("|", "\\|"),
            )
        )

    lines.append("")
    return "\n".join(lines)


def write_candidate_diagnostic_outputs(
    run_root: str | Path,
    out_dir: str | Path,
    normalized_exchanges_path: str | Path | None = None,
    quality_json_path: str | Path | None = None,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    diag = build_candidate_diagnostic(
        run_root=run_root,
        normalized_exchanges_path=normalized_exchanges_path,
        quality_json_path=quality_json_path,
    )

    json_path = out / "candidate_decision_diagnostic.json"
    md_path = out / "candidate_decision_diagnostic.md"

    write_json(json_path, diag)
    md_path.write_text(render_candidate_diagnostic_markdown(diag), encoding="utf-8")

    return {
        "candidate_decision_diagnostic_json": str(json_path),
        "candidate_decision_diagnostic_md": str(md_path),
    }
