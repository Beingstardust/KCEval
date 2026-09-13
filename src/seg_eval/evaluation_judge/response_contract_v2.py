from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS, RUBRIC_VERSION

REQUIRED_PARTIAL_DIMENSIONS = list(PARTIAL_DIMENSIONS)
REQUIRED_TRUST_FLAGS = list(TRUST_FLAGS)

REQUIRED_FLAGS = [
    "major_correctness_error",
    "ungrounded_claim",
    "missed_student_need",
    "overly_answer_giving",
    "segment_boundary_problem",
    "needs_human_review",
]

ALLOWED_PARTIAL_SCORES = {0, 0.5, 1}
ALLOWED_TRUST_FLAGS = {0, 1}

ALLOWED_EVALUATION_MODES = {
    "single_kc_primary",
    "branch_context_with_anchor",
    "broad_kc_set_low_confidence_anchor",
}

ALLOWED_APPLICABILITY = {"applicable", "not_applicable", "unclear"}

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "contract_version": "segment_judge_response_contract_v2_scale001_dependency_aware",
    "rubric_version": RUBRIC_VERSION,
    "required_top_level_fields": [
        "segment_id",
        "evaluation_mode",
        "raw_dimension_scores",
        "dimension_applicability",
        "dependency_adjustments",
        "dimension_scores",
        "trust_flags",
        "micro_score_raw",
        "micro_score_dependency_adjusted",
        "trust_adjusted_score",
        "kc_grounding",
        "flags",
        "evidence_quotes",
        "rationale_short",
    ],
    "partial_dimension_ids": REQUIRED_PARTIAL_DIMENSIONS,
    "trust_flag_ids": REQUIRED_TRUST_FLAGS,
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def normalize_for_quote_match(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("â€œ", '"').replace("â€", '"').replace("â€™", "'")
    return text.strip()


def packet_segment_text(packet: dict[str, Any]) -> str:
    """Every form of the segment's own text a judge could legitimately quote from.

    The judge prompt (prompt_builder.exchange_lines) shows exchanges role-labelled --
    "[exid] Student: ...\\n[exid] Tutor: ...". A judge quoting a whole exchange naturally
    reproduces that "Student: .../Tutor: ..." framing. This function used to only concatenate
    the RAW student_text/tutor_text with no labels, so a verbatim, honest quote of exactly what
    the model was shown could still fail the verbatim-quote check. Both the raw and the
    role-labelled form are included now, so either quoting style validates -- this widens what
    counts as "found in text", not what the model is allowed to say.
    """
    parts: list[str] = []
    segment_text = packet.get("segment_text")
    if isinstance(segment_text, str) and segment_text.strip():
        parts.append(segment_text)

    exchanges = packet.get("member_exchanges")
    if isinstance(exchanges, list):
        for ex in exchanges:
            if not isinstance(ex, dict):
                continue
            for key in ["student_text", "tutor_text", "exchange_text", "text"]:
                value = ex.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
            student = ex.get("student_text")
            tutor = ex.get("tutor_text")
            if isinstance(student, str) and student.strip():
                parts.append(f"Student: {student}")
            if isinstance(tutor, str) and tutor.strip():
                parts.append(f"Tutor: {tutor}")
            if (isinstance(student, str) and student.strip()) or (isinstance(tutor, str) and tutor.strip()):
                parts.append(f"Student: {student or ''}\nTutor: {tutor or ''}")

    return "\n".join(parts)


def extract_response_payload(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    for key in ["judge_output", "response", "output"]:
        if key in row and isinstance(row[key], dict):
            return row[key], errors

    if "raw_response_text" in row and isinstance(row["raw_response_text"], str):
        text = row["raw_response_text"].strip()
        try:
            parsed = json.loads(text)
        except Exception as exc:
            errors.append(f"raw_response_text_not_json:{exc!r}")
            return None, errors
        if not isinstance(parsed, dict):
            errors.append("raw_response_text_json_not_object")
            return None, errors
        return parsed, errors

    if "segment_id" in row and "raw_dimension_scores" in row and "trust_flags" in row:
        return row, errors

    errors.append("could_not_find_response_payload")
    return None, errors


def _require_type(errors: list[str], path: str, value: Any, expected_type: type) -> bool:
    if not isinstance(value, expected_type):
        errors.append(f"{path}:expected_{expected_type.__name__}:got_{type(value).__name__}")
        return False
    return True


def _partial_score_or_null(errors: list[str], path: str, value: Any, applicability: str | None) -> None:
    if applicability == "not_applicable":
        if value is not None:
            errors.append(f"{path}:must_be_null_when_not_applicable")
        return

    if value is None:
        errors.append(f"{path}:null_only_allowed_when_not_applicable")
        return

    if not isinstance(value, (int, float)):
        errors.append(f"{path}:score_not_number")
        return
    if value not in ALLOWED_PARTIAL_SCORES:
        errors.append(f"{path}:score_not_allowed_0_0.5_1:{value}")


def _normalized_score(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        errors.append(f"{path}:not_number")
        return
    if value < 0.0 or value > 1.0:
        errors.append(f"{path}:out_of_range_0_1:{value}")


def _validate_dimension_block(errors: list[str], *, block_name: str, scores: Any, applicability: dict[str, str]) -> None:
    if not _require_type(errors, block_name, scores, dict):
        return

    extra_dims = sorted(set(scores.keys()) - set(REQUIRED_PARTIAL_DIMENSIONS))
    if extra_dims:
        errors.append(f"{block_name}.unexpected_dimensions:{extra_dims}")

    for key in REQUIRED_PARTIAL_DIMENSIONS:
        if key not in scores:
            errors.append(f"{block_name}.missing:{key}")
        else:
            _partial_score_or_null(errors, f"{block_name}.{key}", scores[key], applicability.get(key))


def validate_judge_response(
    response: dict[str, Any],
    *,
    expected_segment_id: str | None = None,
    expected_evaluation_mode: str | None = None,
    allowed_segment_text: str | None = None,
    allowed_kc_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []

    required = RESPONSE_JSON_SCHEMA["required_top_level_fields"]
    for key in required:
        if key not in response:
            errors.append(f"missing_required_field:{key}")

    if errors:
        return errors

    extra_keys = sorted(set(response.keys()) - set(required))
    if extra_keys:
        errors.append(f"unexpected_top_level_fields:{extra_keys}")

    segment_id = response.get("segment_id")
    if not _require_type(errors, "segment_id", segment_id, str):
        segment_id = None
    elif expected_segment_id and segment_id != expected_segment_id:
        errors.append(f"segment_id_mismatch:expected={expected_segment_id}:actual={segment_id}")

    evaluation_mode = response.get("evaluation_mode")
    if not _require_type(errors, "evaluation_mode", evaluation_mode, str):
        evaluation_mode = None
    elif evaluation_mode not in ALLOWED_EVALUATION_MODES:
        errors.append(f"evaluation_mode_not_allowed:{evaluation_mode}")
    elif expected_evaluation_mode and evaluation_mode != expected_evaluation_mode:
        errors.append(f"evaluation_mode_mismatch:expected={expected_evaluation_mode}:actual={evaluation_mode}")

    applicability_raw = response.get("dimension_applicability")
    applicability: dict[str, str] = {}
    if _require_type(errors, "dimension_applicability", applicability_raw, dict):
        extra_dims = sorted(set(applicability_raw.keys()) - set(REQUIRED_PARTIAL_DIMENSIONS))
        if extra_dims:
            errors.append(f"dimension_applicability.unexpected_dimensions:{extra_dims}")
        for key in REQUIRED_PARTIAL_DIMENSIONS:
            value = applicability_raw.get(key)
            if value not in ALLOWED_APPLICABILITY:
                errors.append(f"dimension_applicability.{key}:invalid:{value}")
            else:
                applicability[key] = value

    _validate_dimension_block(errors, block_name="raw_dimension_scores", scores=response.get("raw_dimension_scores"), applicability=applicability)
    _validate_dimension_block(errors, block_name="dimension_scores", scores=response.get("dimension_scores"), applicability=applicability)

    trust_flags = response.get("trust_flags")
    if _require_type(errors, "trust_flags", trust_flags, dict):
        extra_trust = sorted(set(trust_flags.keys()) - set(REQUIRED_TRUST_FLAGS))
        if extra_trust:
            errors.append(f"trust_flags.unexpected_flags:{extra_trust}")
        for key in REQUIRED_TRUST_FLAGS:
            value = trust_flags.get(key)
            if value not in ALLOWED_TRUST_FLAGS:
                errors.append(f"trust_flags.{key}:must_be_0_or_1")

    for key in ["micro_score_raw", "micro_score_dependency_adjusted", "trust_adjusted_score"]:
        _normalized_score(errors, key, response.get(key))

    adjustments = response.get("dependency_adjustments")
    if _require_type(errors, "dependency_adjustments", adjustments, list):
        for idx, item in enumerate(adjustments):
            if not isinstance(item, dict):
                errors.append(f"dependency_adjustments[{idx}]:not_object")
                continue
            required_adj = {"rule_id", "applied", "reason", "affected_dimensions"}
            missing = sorted(required_adj - set(item.keys()))
            if missing:
                errors.append(f"dependency_adjustments[{idx}].missing:{missing}")
            extra = sorted(set(item.keys()) - required_adj)
            if extra:
                errors.append(f"dependency_adjustments[{idx}].unexpected_fields:{extra}")
            if not isinstance(item.get("rule_id"), str) or not item.get("rule_id"):
                errors.append(f"dependency_adjustments[{idx}].rule_id:invalid")
            if not isinstance(item.get("applied"), bool):
                errors.append(f"dependency_adjustments[{idx}].applied:not_boolean")
            if not isinstance(item.get("reason"), str) or len(item.get("reason", "").strip()) < 8:
                errors.append(f"dependency_adjustments[{idx}].reason:too_short")
            affected = item.get("affected_dimensions")
            if not isinstance(affected, list):
                errors.append(f"dependency_adjustments[{idx}].affected_dimensions:not_list")
            else:
                for j, dim in enumerate(affected):
                    if dim not in REQUIRED_PARTIAL_DIMENSIONS:
                        errors.append(f"dependency_adjustments[{idx}].affected_dimensions[{j}]:unknown_dimension:{dim}")
                if item.get("applied") is True and len(affected) == 0:
                    errors.append(f"dependency_adjustments[{idx}].affected_dimensions:empty_when_applied")

    grounding = response.get("kc_grounding")
    if _require_type(errors, "kc_grounding", grounding, dict):
        if not isinstance(grounding.get("primary_kc_used"), bool):
            errors.append("kc_grounding.primary_kc_used:not_boolean")

        context_used = grounding.get("context_kcs_used")
        if _require_type(errors, "kc_grounding.context_kcs_used", context_used, list):
            for idx, item in enumerate(context_used):
                if not isinstance(item, str):
                    errors.append(f"kc_grounding.context_kcs_used[{idx}]:not_string")
                elif allowed_kc_ids is not None and item not in allowed_kc_ids:
                    errors.append(f"kc_grounding.context_kcs_used[{idx}]:kc_not_in_packet:{item}")

        wrong_claims = grounding.get("unsupported_or_wrong_kc_claims")
        if _require_type(errors, "kc_grounding.unsupported_or_wrong_kc_claims", wrong_claims, list):
            for idx, item in enumerate(wrong_claims):
                if not isinstance(item, str):
                    errors.append(f"kc_grounding.unsupported_or_wrong_kc_claims[{idx}]:not_string")

        extra_grounding = sorted(set(grounding.keys()) - {"primary_kc_used", "context_kcs_used", "unsupported_or_wrong_kc_claims"})
        if extra_grounding:
            errors.append(f"kc_grounding.unexpected_fields:{extra_grounding}")

    flags = response.get("flags")
    if _require_type(errors, "flags", flags, dict):
        extra_flags = sorted(set(flags.keys()) - set(REQUIRED_FLAGS))
        if extra_flags:
            errors.append(f"flags.unexpected_flags:{extra_flags}")
        for key in REQUIRED_FLAGS:
            if key not in flags:
                errors.append(f"flags.missing:{key}")
            elif not isinstance(flags[key], bool):
                errors.append(f"flags.{key}:not_boolean")

    quotes = response.get("evidence_quotes")
    if _require_type(errors, "evidence_quotes", quotes, list):
        if len(quotes) < 1:
            errors.append("evidence_quotes:empty")
        if len(quotes) > 6:
            errors.append(f"evidence_quotes:too_many:{len(quotes)}")

        segment_norm = normalize_for_quote_match(allowed_segment_text or "")
        for idx, quote in enumerate(quotes):
            if not isinstance(quote, str):
                errors.append(f"evidence_quotes[{idx}]:not_string")
                continue
            q = quote.strip()
            if len(q) < 4:
                errors.append(f"evidence_quotes[{idx}]:too_short")
            if len(q) > 500:
                errors.append(f"evidence_quotes[{idx}]:too_long")
            if allowed_segment_text:
                q_norm = normalize_for_quote_match(q)
                if q_norm and q_norm not in segment_norm:
                    errors.append(f"evidence_quotes[{idx}]:not_found_in_segment_text")

    rationale = response.get("rationale_short")
    if _require_type(errors, "rationale_short", rationale, str):
        if len(rationale.strip()) < 20:
            errors.append("rationale_short:too_short")
        if len(rationale) > 1400:
            errors.append("rationale_short:too_long")
        # "step-by-step reasoning" alone is too broad a marker: it's also the ordinary, legitimate
        # way to describe a TUTOR's scaffolding technique ("the tutor uses step-by-step
        # reasoning"), which a rationale should be free to say. Only flag it when the phrase is
        # self-referential -- the judge describing ITS OWN reasoning process -- which is what this
        # check exists to catch. "chain of thought" and "hidden reasoning" stay as plain substring
        # matches since neither has an equivalent legitimate meaning about the tutor.
        rationale_lower = rationale.lower()
        self_referential_reasoning = bool(re.search(
            r"\bmy\s+(?:step-by-step|internal|hidden)\s+reasoning\b|\bmy\s+chain\s+of\s+thought\b",
            rationale_lower,
        ))
        if self_referential_reasoning or any(
            marker in rationale_lower for marker in ["chain of thought", "hidden reasoning"]
        ):
            errors.append("rationale_short:mentions_hidden_reasoning")

    return errors


def packet_context_by_segment(packet_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(p["segment_id"]): p for p in packet_rows if p.get("segment_id")}


def prompt_context_by_segment(prompt_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(p["segment_id"]): p for p in prompt_rows if p.get("segment_id")}


def allowed_kc_ids_from_packet(packet: dict[str, Any]) -> set[str]:
    ids: set[str] = set()

    target = packet.get("evaluation_target")
    if isinstance(target, dict):
        for key in ["primary_kc_id", "resolved_kc_id", "dominant_kc_id"]:
            value = target.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
        values = target.get("evaluator_kc_set")
        if isinstance(values, list):
            ids.update(str(v) for v in values if v)

    context_rows = packet.get("kc_context")
    if isinstance(context_rows, list):
        for row in context_rows:
            if isinstance(row, dict):
                value = row.get("unit_id") or row.get("kc_id") or row.get("id")
                if value:
                    ids.add(str(value))

    return ids


def validate_response_rows(
    *,
    prompt_rows: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompts_by_segment = prompt_context_by_segment(prompt_rows)
    packets_by_segment = packet_context_by_segment(packet_rows)

    valid_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    seen_segments: set[str] = set()

    for idx, row in enumerate(response_rows):
        response, payload_errors = extract_response_payload(row)

        segment_id = None
        if isinstance(response, dict):
            segment_id = response.get("segment_id")
        if not segment_id:
            segment_id = row.get("segment_id") or row.get("source_segment_id") or f"row_{idx}"

        segment_id = str(segment_id)
        seen_segments.add(segment_id)

        prompt = prompts_by_segment.get(segment_id)
        packet = packets_by_segment.get(segment_id)

        errors = list(payload_errors)
        if response is not None:
            errors.extend(
                validate_judge_response(
                    response,
                    expected_segment_id=segment_id,
                    expected_evaluation_mode=prompt.get("evaluation_mode") if prompt else None,
                    allowed_segment_text=packet_segment_text(packet) if packet else None,
                    allowed_kc_ids=allowed_kc_ids_from_packet(packet) if packet else None,
                )
            )

        if prompt is None:
            errors.append("segment_id_not_found_in_prompts")
        if packet is None:
            errors.append("segment_id_not_found_in_packets")

        if errors:
            error_rows.append({
                "row_index": idx,
                "segment_id": segment_id,
                "errors": errors,
                "raw_row": row,
            })
        else:
            assert response is not None
            valid_rows.append({
                "segment_id": segment_id,
                "evaluation_mode": response["evaluation_mode"],
                "judge_output": response,
            })

    missing_segments = sorted(set(prompts_by_segment.keys()) - seen_segments)
    duplicate_count = len(response_rows) - len(seen_segments)

    summary = {
        "response_count": len(response_rows),
        "valid_count": len(valid_rows),
        "error_count": len(error_rows),
        "prompt_count": len(prompt_rows),
        "packet_count": len(packet_rows),
        "missing_response_segment_count": len(missing_segments),
        "duplicate_segment_response_count": duplicate_count,
        "missing_response_segments": missing_segments[:25],
        "status": "PASS" if not error_rows and not missing_segments and duplicate_count == 0 else "FAIL",
    }

    return valid_rows, error_rows, summary
