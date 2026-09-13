from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RUBRIC_VERSION = "segment_judge_rubric_v1"

REQUIRED_DIMENSIONS = [
    "kc_grounding_correctness",
    "student_need_alignment",
    "pedagogical_scaffolding",
    "clarity_and_coherence",
    "misconception_handling",
]

REQUIRED_FLAGS = [
    "major_correctness_error",
    "ungrounded_claim",
    "missed_student_need",
    "overly_answer_giving",
    "segment_boundary_problem",
    "needs_human_review",
]

ALLOWED_EVALUATION_MODES = {
    "single_kc_primary",
    "branch_context_with_anchor",
    "broad_kc_set_low_confidence_anchor",
}


RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "segment_id",
        "evaluation_mode",
        "overall_score",
        "dimension_scores",
        "kc_grounding",
        "flags",
        "evidence_quotes",
        "rationale_short",
    ],
    "properties": {
        "segment_id": {"type": "string", "minLength": 1},
        "evaluation_mode": {
            "type": "string",
            "enum": sorted(ALLOWED_EVALUATION_MODES),
        },
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "dimension_scores": {
            "type": "object",
            "additionalProperties": False,
            "required": REQUIRED_DIMENSIONS,
            "properties": {
                key: {"type": "integer", "minimum": 1, "maximum": 5}
                for key in REQUIRED_DIMENSIONS
            },
        },
        "kc_grounding": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "primary_kc_used",
                "context_kcs_used",
                "unsupported_or_wrong_kc_claims",
            ],
            "properties": {
                "primary_kc_used": {"type": "boolean"},
                "context_kcs_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unsupported_or_wrong_kc_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "flags": {
            "type": "object",
            "additionalProperties": False,
            "required": REQUIRED_FLAGS,
            "properties": {
                key: {"type": "boolean"}
                for key in REQUIRED_FLAGS
            },
        },
        "evidence_quotes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {"type": "string", "minLength": 4, "maxLength": 500},
        },
        "rationale_short": {
            "type": "string",
            "minLength": 20,
            "maxLength": 1200,
        },
    },
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

    return "\n".join(parts)


def extract_response_payload(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    if "judge_output" in row and isinstance(row["judge_output"], dict):
        return row["judge_output"], errors

    if "response" in row and isinstance(row["response"], dict):
        return row["response"], errors

    if "output" in row and isinstance(row["output"], dict):
        return row["output"], errors

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

    # Treat the row itself as the response object if it has the canonical keys.
    if "segment_id" in row and "overall_score" in row and "dimension_scores" in row:
        return row, errors

    errors.append("could_not_find_response_payload")
    return None, errors


def _require_type(errors: list[str], path: str, value: Any, expected_type: type) -> bool:
    if not isinstance(value, expected_type):
        errors.append(f"{path}:expected_{expected_type.__name__}:got_{type(value).__name__}")
        return False
    return True


def _score(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, int):
        errors.append(f"{path}:score_not_integer")
        return
    if value < 1 or value > 5:
        errors.append(f"{path}:score_out_of_range:{value}")


def validate_judge_response(
    response: dict[str, Any],
    *,
    expected_segment_id: str | None = None,
    expected_evaluation_mode: str | None = None,
    allowed_segment_text: str | None = None,
    allowed_kc_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []

    required = RESPONSE_JSON_SCHEMA["required"]
    for key in required:
        if key not in response:
            errors.append(f"missing_required_field:{key}")

    if errors:
        return errors

    extra_keys = sorted(set(response.keys()) - set(RESPONSE_JSON_SCHEMA["properties"].keys()))
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

    _score(errors, "overall_score", response.get("overall_score"))

    dim_scores = response.get("dimension_scores")
    if _require_type(errors, "dimension_scores", dim_scores, dict):
        extra_dims = sorted(set(dim_scores.keys()) - set(REQUIRED_DIMENSIONS))
        if extra_dims:
            errors.append(f"dimension_scores.unexpected_dimensions:{extra_dims}")
        for key in REQUIRED_DIMENSIONS:
            if key not in dim_scores:
                errors.append(f"dimension_scores.missing:{key}")
            else:
                _score(errors, f"dimension_scores.{key}", dim_scores[key])

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
        if len(rationale) > 1200:
            errors.append("rationale_short:too_long")
        if any(marker in rationale.lower() for marker in ["step-by-step reasoning", "chain of thought", "hidden reasoning"]):
            errors.append("rationale_short:mentions_hidden_reasoning")

    return errors


def packet_context_by_segment(packet_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for packet in packet_rows:
        segment_id = packet.get("segment_id")
        if segment_id:
            out[str(segment_id)] = packet
    return out


def prompt_context_by_segment(prompt_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for prompt in prompt_rows:
        segment_id = prompt.get("segment_id")
        if segment_id:
            out[str(segment_id)] = prompt
    return out


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
        if response is None:
            pass
        else:
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
