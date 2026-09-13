from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from .dialogue_parser import normalize_role, normalize_text
from .io_utils import as_text


LOW_CONFIDENCE_THRESHOLD = 0.75
HARD_BLOCKING_STATUSES = {
    "missing_student",
    "missing_tutor",
    "unknown_speaker_present",
    "empty_exchange_text",
}

SOFT_ONLY_STATUS = "low_confidence_parse"


STUDENT_LABELS = {"student", "s", "learner", "user", "human"}
TUTOR_LABELS = {"tutor", "t", "teacher", "assistant", "ai", "model"}

EXPLICIT_COLON_RE = re.compile(r"^\s*(student|tutor|s|t)\s*:\s*(.*)$", re.IGNORECASE)
EXPLICIT_BRACKET_RE = re.compile(r"^\s*\[(student|tutor|s|t)\]\s*(.*)$", re.IGNORECASE)
EXPLICIT_BRACE_RE = re.compile(r"^\s*(student|tutor|s|t)\s*\{\s*(.*)$", re.IGNORECASE)


@dataclass
class NormalizedTurn:
    conversation_id: str
    turn_id: str
    turn_index: int
    speaker: str
    text: str
    source_span: dict[str, Any]
    parser_confidence: float
    parser_notes: list[str]
    style_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedExchange:
    conversation_id: str
    exchange_id: str
    exchange_index: int
    student_turn_ids: list[str]
    tutor_turn_ids: list[str]
    student_text: str
    tutor_text: str
    teaching_style_tags: list[str]
    normalization_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_speaker(label: str) -> str:
    value = (label or "").strip().lower()
    if value in STUDENT_LABELS:
        return "student"
    if value in TUTOR_LABELS:
        return "tutor"
    normalized = normalize_role(value)
    if normalized in {"student", "tutor"}:
        return normalized
    return "unknown"


def unique_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def preview_text(text: str, limit: int = 120) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def detect_style_tags(speaker: str, text: str) -> list[str]:
    t = (text or "").lower()
    tags: list[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if speaker == "student":
        if "?" in t:
            add("student_clarification_question")
        if any(phrase in t for phrase in ["i am confused", "i'm confused", "i do not get", "i don't get", "i got stuck", "i am stuck"]):
            add("student_confusion")
        if any(phrase in t for phrase in ["i think", "my instinct", "so ", "i used", "i guess"]):
            add("student_partial_understanding")
        if any(phrase in t for phrase in ["right?", "so ", "means i can always", "is it simply"]):
            add("student_incorrect_attempt")
        if any(phrase in t for phrase in ["that", "this", "it works", "it fail", "same step", "previous idea"]):
            add("anaphora_ellipsis")
        return tags

    if speaker != "tutor":
        return tags

    if "?" in t and any(phrase in t for phrase in ["what do you think", "which part", "before i tell you", "why do you think", "how would you"]):
        add("socratic_questioning")
    if any(phrase in t for phrase in ["hint", "try focusing", "focus on", "useful hint", "next step"]):
        add("hint_scaffolding")
    if any(phrase in t for phrase in ["for example", "worked example", "tiny example", "start with a tiny example"]):
        add("worked_example")
    if any(phrase in t for phrase in ["step by step", "safe procedure", "first", "second", "third"]):
        add("step_by_step_procedure")
    if any(phrase in t for phrase in ["not always", "important correction", "depends on the conditions", "wrong conclusion", "not quite"]):
        add("misconception_correction")
    if any(phrase in t for phrase in ["think of it like", "analogy", "bridge", "imagine", "like organising"]):
        add("analogy_bridge")
    if any(phrase in t for phrase in ["contrast", "different from", "different questions", "compare", "versus"]):
        add("comparison_contrast")
    if any(phrase in t for phrase in ["yes, that is the right direction", "exactly", "correct", "good."]):
        add("confirmation_feedback")
    if any(phrase in t for phrase in ["pause and explain your reasoning", "reflect", "check whether that assumption fits"]):
        add("metacognitive_prompt")
    if any(phrase in t for phrase in ["big picture", "recap", "summary", "fit together"]):
        add("recap_synthesis")
    if any(phrase in t for phrase in ["prerequisite", "foundation", "depends on that earlier relationship", "repair the prerequisite"]):
        add("prerequisite_remediation")
    if any(phrase in t for phrase in ["keep that in mind", "active concept for the next question", "go back to"]):
        add("anaphora_setup")
    if any(phrase in t for phrase in ["that", "this", "it works", "previous idea", "same step", "those assumptions"]):
        add("anaphora_ellipsis")
    if any(phrase in t for phrase in ["reasoning shifts toward", "transition", "connect", "move from"]):
        add("multi_kc_transition")
    if not tags and any(phrase in t for phrase in [" is ", " means ", "the target idea", "i will explain"]):
        add("direct_explanation")
    return tags


def _finalize_text(parts: list[str]) -> str:
    text = "\n".join(parts)
    return normalize_text(text)


def _make_turn(
    conversation_id: str,
    turn_index: int,
    speaker: str,
    text_parts: list[str],
    source_span: dict[str, Any],
    parser_confidence: float,
    parser_notes: list[str],
) -> NormalizedTurn:
    text = _finalize_text(text_parts)
    notes = list(parser_notes)
    if speaker == "unknown" and "unknown_speaker" not in notes:
        notes.append("unknown_speaker")
    if parser_confidence < LOW_CONFIDENCE_THRESHOLD and "low_confidence_parse" not in notes:
        notes.append("low_confidence_parse")
    return NormalizedTurn(
        conversation_id=conversation_id,
        turn_id=f"{conversation_id}::turn_{turn_index:04d}",
        turn_index=turn_index,
        speaker=speaker,
        text=text,
        source_span=source_span,
        parser_confidence=round(parser_confidence, 3),
        parser_notes=unique_preserve(notes),
        style_tags=detect_style_tags(speaker, text),
    )


def _structured_rows_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]

    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        if all(any(key in item for key in ["turns", "messages", "dialogue", "conversation"]) for item in value):
            return list(value)

        if all("student" in item and "tutor" in item for item in value):
            turns: list[dict[str, Any]] = []

            def _message_text(obj: Any) -> str:
                if isinstance(obj, dict):
                    for key in ["text", "content", "message", "utterance"]:
                        val = obj.get(key)
                        if val not in (None, ""):
                            return as_text(val)
                    return ""
                return as_text(obj)

            def _message_metadata(obj: Any) -> dict[str, Any]:
                if not isinstance(obj, dict):
                    return {}
                return {
                    key: val
                    for key, val in obj.items()
                    if key not in {"text", "content", "message", "utterance"}
                }

            for pair_index, pair in enumerate(value):
                student_obj = pair.get("student")
                tutor_obj = pair.get("tutor")
                turns.append({
                    "role": "student",
                    "content": _message_text(student_obj),
                    "source_pair_index": pair_index,
                    "source_metadata": _message_metadata(student_obj),
                })
                turns.append({
                    "role": "tutor",
                    "content": _message_text(tutor_obj),
                    "source_pair_index": pair_index,
                    "source_metadata": _message_metadata(tutor_obj),
                })
            return [{"dialogue_id": "structured_pairs", "turns": turns}]

    raise ValueError("Unsupported structured JSON shape for dialogue normalization.")


def _structured_rows_from_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict) or not any(key in row for key in ["turns", "messages", "dialogue", "conversation"]):
            raise ValueError(f"JSONL line {line_index} does not contain turns/messages/dialogue/conversation.")
        rows.append(row)
    if not rows:
        raise ValueError("Structured JSONL input had no non-empty rows.")
    return rows


def _turns_from_structured_rows(
    rows: list[dict[str, Any]],
    conversation_id: str,
    source_name: str,
) -> tuple[list[NormalizedTurn], list[str]]:
    turns: list[NormalizedTurn] = []
    warnings: list[str] = []
    turn_index = 0

    if len(rows) > 1:
        warnings.append("multiple_structured_dialogue_rows_combined")

    for row_index, row in enumerate(rows):
        raw_turns = row.get("turns") or row.get("messages") or row.get("dialogue") or row.get("conversation")
        if not isinstance(raw_turns, list):
            raise ValueError(f"Structured row {row_index} has no list-valued turns/messages/dialogue/conversation.")

        for source_turn_index, raw in enumerate(raw_turns):
            if not isinstance(raw, dict):
                raise ValueError(f"Structured row {row_index} turn {source_turn_index} is not an object.")
            speaker = canonical_speaker(str(raw.get("role") or raw.get("speaker") or raw.get("from") or raw.get("author") or "unknown"))
            text = normalize_text(as_text(raw.get("content") or raw.get("text") or raw.get("message") or raw.get("utterance") or ""))
            notes = [f"structured_input:{source_name}", f"source_row_index:{row_index}", f"source_turn_index:{source_turn_index}"]
            confidence = 0.99
            if speaker == "unknown":
                confidence = 0.45
                notes.append("unknown_speaker_in_structured_input")
            turns.append(_make_turn(
                conversation_id=conversation_id,
                turn_index=turn_index,
                speaker=speaker,
                text_parts=[text],
                source_span={"row_index": row_index, "turn_index_in_row": source_turn_index},
                parser_confidence=confidence,
                parser_notes=notes,
            ))
            turn_index += 1
    return turns, warnings


def _parse_free_text_turns(text: str, conversation_id: str) -> tuple[list[NormalizedTurn], list[str]]:
    turns: list[NormalizedTurn] = []
    warnings: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    offsets: list[int] = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line) + 1

    current_speaker: str | None = None
    current_parts: list[str] = []
    current_notes: list[str] = []
    current_confidence = 0.0
    current_start_line = 0
    current_start_char = 0
    current_mode = ""
    inside_brace = False

    def flush(end_line: int, end_char: int) -> None:
        nonlocal current_speaker, current_parts, current_notes, current_confidence
        nonlocal current_start_line, current_start_char, current_mode
        if current_speaker is None:
            return
        turn = _make_turn(
            conversation_id=conversation_id,
            turn_index=len(turns),
            speaker=current_speaker,
            text_parts=current_parts,
            source_span={
                "line_start": current_start_line,
                "line_end": end_line,
                "char_start": current_start_char,
                "char_end": end_char,
            },
            parser_confidence=current_confidence,
            parser_notes=current_notes,
        )
        turns.append(turn)
        current_speaker = None
        current_parts = []
        current_notes = []
        current_confidence = 0.0
        current_start_line = 0
        current_start_char = 0
        current_mode = ""

    for line_index, line in enumerate(lines, start=1):
        stripped = line.strip()
        start_char = offsets[line_index - 1]
        end_char = start_char + len(line)

        if inside_brace:
            if "}" in line:
                content, _, tail = line.partition("}")
                current_parts.append(content)
                if tail.strip():
                    current_notes.append("text_after_closing_brace_ignored")
                    warnings.append(f"line_{line_index}:text_after_closing_brace_ignored")
                flush(line_index, end_char)
                inside_brace = False
                continue
            current_parts.append(line)
            continue

        colon_match = EXPLICIT_COLON_RE.match(line)
        bracket_match = EXPLICIT_BRACKET_RE.match(line)
        brace_match = EXPLICIT_BRACE_RE.match(line)

        if colon_match or bracket_match:
            match = colon_match or bracket_match
            assert match is not None
            flush(line_index - 1 if turns or current_parts else line_index, start_char)
            label = match.group(1)
            current_speaker = canonical_speaker(label)
            current_parts = [match.group(2)]
            current_notes = [f"speaker_marker:{'colon' if colon_match else 'bracket'}"]
            current_confidence = 0.98 if len(label) > 1 else 0.88
            current_start_line = line_index
            current_start_char = start_char
            current_mode = "explicit"
            continue

        if brace_match:
            flush(line_index - 1 if turns or current_parts else line_index, start_char)
            label = brace_match.group(1)
            rest = brace_match.group(2)
            current_speaker = canonical_speaker(label)
            current_notes = ["speaker_marker:brace"]
            current_confidence = 0.95 if len(label) > 1 else 0.85
            current_start_line = line_index
            current_start_char = start_char
            current_mode = "brace"
            if "}" in rest:
                content, _, tail = rest.partition("}")
                current_parts = [content]
                if tail.strip():
                    current_notes.append("text_after_closing_brace_ignored")
                    warnings.append(f"line_{line_index}:text_after_closing_brace_ignored")
                flush(line_index, end_char)
            else:
                current_parts = [rest]
                inside_brace = True
            continue

        if stripped == "":
            if current_speaker is not None and current_parts:
                current_parts.append("")
            continue

        if current_speaker is not None and current_mode in {"explicit", "brace", "unknown"}:
            current_parts.append(line)
            if current_mode == "unknown":
                current_confidence = min(current_confidence, 0.4)
                if "continued_unlabeled_block" not in current_notes:
                    current_notes.append("continued_unlabeled_block")
            else:
                current_confidence = min(current_confidence, 0.95 if current_confidence >= 0.95 else current_confidence)
                if "multiline_continuation" not in current_notes:
                    current_notes.append("multiline_continuation")
            continue

        flush(line_index - 1 if turns or current_parts else line_index, start_char)
        current_speaker = "unknown"
        current_parts = [line]
        current_notes = ["unlabeled_text_without_explicit_speaker"]
        current_confidence = 0.35
        current_start_line = line_index
        current_start_char = start_char
        current_mode = "unknown"

    if inside_brace:
        current_confidence = min(current_confidence, 0.3)
        current_notes.append("unclosed_brace_block")
        warnings.append("unclosed_brace_block")

    flush(len(lines), running)
    return turns, warnings


def build_normalized_exchanges(turns: list[NormalizedTurn]) -> list[NormalizedExchange]:
    exchanges: list[NormalizedExchange] = []
    index = 0

    while index < len(turns):
        student_turns: list[NormalizedTurn] = []
        tutor_turns: list[NormalizedTurn] = []
        member_turns: list[NormalizedTurn] = []

        current = turns[index]
        if current.speaker == "student":
            while index < len(turns) and turns[index].speaker == "student":
                student_turns.append(turns[index])
                member_turns.append(turns[index])
                index += 1
            while index < len(turns) and turns[index].speaker == "tutor":
                tutor_turns.append(turns[index])
                member_turns.append(turns[index])
                index += 1
        elif current.speaker == "tutor":
            while index < len(turns) and turns[index].speaker == "tutor":
                tutor_turns.append(turns[index])
                member_turns.append(turns[index])
                index += 1
        else:
            while index < len(turns) and turns[index].speaker == "unknown":
                member_turns.append(turns[index])
                index += 1

        student_text = normalize_text("\n".join(turn.text for turn in student_turns if turn.text))
        tutor_text = normalize_text("\n".join(turn.text for turn in tutor_turns if turn.text))
        style_tags = unique_preserve([
            tag
            for turn in member_turns
            for tag in turn.style_tags
        ])

        status = "ok"
        if any(turn.speaker == "unknown" for turn in member_turns):
            status = "unknown_speaker_present"
        elif not student_turns:
            status = "missing_student"
        elif not tutor_turns:
            status = "missing_tutor"
        elif not student_text or not tutor_text:
            status = "empty_exchange_text"
        elif any(turn.parser_confidence < LOW_CONFIDENCE_THRESHOLD for turn in member_turns):
            status = SOFT_ONLY_STATUS

        exchanges.append(NormalizedExchange(
            conversation_id=member_turns[0].conversation_id if member_turns else "unknown_conversation",
            exchange_id=f"{member_turns[0].conversation_id if member_turns else 'unknown_conversation'}::ex_{len(exchanges):04d}",
            exchange_index=len(exchanges),
            student_turn_ids=[turn.turn_id for turn in student_turns],
            tutor_turn_ids=[turn.turn_id for turn in tutor_turns],
            student_text=student_text,
            tutor_text=tutor_text,
            teaching_style_tags=style_tags,
            normalization_status=status,
        ))

    return exchanges


def _normalized_dialogue_turns_from_exchanges(exchanges: list[NormalizedExchange]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for exchange in exchanges:
        turns.append({
            "role": "student",
            "content": exchange.student_text,
            "metadata": {
                "source_exchange_id": exchange.exchange_id,
                "source_turn_ids": exchange.student_turn_ids,
                "teaching_style_tags": exchange.teaching_style_tags,
            },
        })
        turns.append({
            "role": "tutor",
            "content": exchange.tutor_text,
            "metadata": {
                "source_exchange_id": exchange.exchange_id,
                "source_turn_ids": exchange.tutor_turn_ids,
                "teaching_style_tags": exchange.teaching_style_tags,
            },
        })
    return turns


def build_normalized_dialogues_rows(
    conversation_id: str,
    exchanges: list[NormalizedExchange],
) -> list[dict[str, Any]]:
    return [{
        "dialogue_id": conversation_id,
        "metadata": {
            "normalization_pipeline": "dialogue_intake_v1",
            "exchange_count": len(exchanges),
        },
        "turns": _normalized_dialogue_turns_from_exchanges(exchanges),
    }]


def summarize_normalization(
    conversation_id: str,
    input_path: str | Path,
    input_format_detected: str,
    turns: list[NormalizedTurn],
    exchanges: list[NormalizedExchange],
    warnings: list[str],
    allow_low_confidence: bool,
) -> dict[str, Any]:
    speaker_counts = Counter(turn.speaker for turn in turns)
    status_counts = Counter(exchange.normalization_status for exchange in exchanges)
    low_turns = [turn for turn in turns if turn.parser_confidence < LOW_CONFIDENCE_THRESHOLD]
    hard_blockers = sorted(status for status in status_counts if status in HARD_BLOCKING_STATUSES)
    soft_only = any(status == SOFT_ONLY_STATUS for status in status_counts)
    ready = not hard_blockers and not soft_only and len(exchanges) > 0
    override_used = not ready and not hard_blockers and soft_only and allow_low_confidence

    return {
        "schema_version": "dialogue_normalization_audit_v1",
        "conversation_id": conversation_id,
        "input_path": str(input_path),
        "input_format_detected": input_format_detected,
        "turn_count": len(turns),
        "exchange_count": len(exchanges),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "normalization_status_counts": dict(sorted(status_counts.items())),
        "normalization_ready_for_segmentation": ready,
        "allow_low_confidence_requested": allow_low_confidence,
        "allow_low_confidence_override_used": override_used,
        "segmentation_hard_blockers": hard_blockers,
        "soft_warnings_present": soft_only,
        "normalized_dialogues_written": False,
        "parser_confidence_summary": {
            "min": round(min((turn.parser_confidence for turn in turns), default=0.0), 3),
            "max": round(max((turn.parser_confidence for turn in turns), default=0.0), 3),
            "avg": round(mean(turn.parser_confidence for turn in turns), 3) if turns else 0.0,
            "low_confidence_turn_count": len(low_turns),
        },
        "warnings": unique_preserve(warnings),
        "turn_notes": [
            {
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "parser_confidence": turn.parser_confidence,
                "parser_notes": turn.parser_notes,
                "preview": preview_text(turn.text),
            }
            for turn in turns
            if turn.parser_notes
        ],
        "exchange_notes": [
            {
                "exchange_id": exchange.exchange_id,
                "normalization_status": exchange.normalization_status,
                "student_turn_ids": exchange.student_turn_ids,
                "tutor_turn_ids": exchange.tutor_turn_ids,
                "teaching_style_tags": exchange.teaching_style_tags,
            }
            for exchange in exchanges
        ],
    }


def render_normalization_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Dialogue Normalization Audit",
        "",
        "## Summary",
        "",
        f"- Conversation ID: `{audit['conversation_id']}`",
        f"- Input path: `{audit['input_path']}`",
        f"- Input format detected: `{audit['input_format_detected']}`",
        f"- Turn count: `{audit['turn_count']}`",
        f"- Exchange count: `{audit['exchange_count']}`",
        f"- Normalization ready for segmentation: `{audit['normalization_ready_for_segmentation']}`",
        f"- Allow low confidence requested: `{audit['allow_low_confidence_requested']}`",
        f"- Allow low confidence override used: `{audit['allow_low_confidence_override_used']}`",
        f"- Normalized dialogues written: `{audit['normalized_dialogues_written']}`",
        f"- Status counts: `{audit['normalization_status_counts']}`",
        f"- Speaker counts: `{audit['speaker_counts']}`",
        f"- Parser confidence summary: `{audit['parser_confidence_summary']}`",
        "",
        "## Warnings",
        "",
    ]

    warning_lines = audit.get("warnings") or ["none"]
    for warning in warning_lines:
        lines.append(f"- {warning}")

    lines.extend([
        "",
        "## Turns",
        "",
        "| Turn | Speaker | Confidence | Style tags | Notes | Preview |",
        "|---|---|---:|---|---|---|",
    ])

    for turn_note in audit.get("turn_notes", []):
        turn_style_tags = next(
            (
                note.get("style_tags")
                for note in []
            ),
            None,
        )
        _ = turn_style_tags

    for turn_note in audit.get("turn_notes", []):
        notes = ", ".join(turn_note.get("parser_notes") or [])
        lines.append(
            "| `{turn_id}` | `{speaker}` | {confidence:.3f} | `{styles}` | `{notes}` | {preview} |".format(
                turn_id=turn_note["turn_id"],
                speaker=turn_note["speaker"],
                confidence=float(turn_note["parser_confidence"]),
                styles="see normalized_turns.jsonl",
                notes=notes or "none",
                preview=turn_note["preview"].replace("|", "\\|"),
            )
        )

    lines.extend([
        "",
        "## Exchanges",
        "",
        "| Exchange | Status | Student turns | Tutor turns | Teaching style tags |",
        "|---|---|---|---|---|",
    ])

    for exchange_note in audit.get("exchange_notes", []):
        lines.append(
            "| `{exchange_id}` | `{status}` | `{student_turns}` | `{tutor_turns}` | `{styles}` |".format(
                exchange_id=exchange_note["exchange_id"],
                status=exchange_note["normalization_status"],
                student_turns=",".join(exchange_note["student_turn_ids"]) or "none",
                tutor_turns=",".join(exchange_note["tutor_turn_ids"]) or "none",
                styles=", ".join(exchange_note["teaching_style_tags"]) or "none",
            )
        )

    return "\n".join(lines)


def normalize_dialogue_text(
    text: str,
    conversation_id: str,
    source_name: str = "<memory>",
    allow_low_confidence: bool = False,
) -> dict[str, Any]:
    turns, warnings = _parse_free_text_turns(text, conversation_id)
    exchanges = build_normalized_exchanges(turns)
    audit = summarize_normalization(
        conversation_id=conversation_id,
        input_path=source_name,
        input_format_detected="free_text",
        turns=turns,
        exchanges=exchanges,
        warnings=warnings,
        allow_low_confidence=allow_low_confidence,
    )
    return {
        "turns": turns,
        "exchanges": exchanges,
        "audit": audit,
    }


def normalize_dialogue_path(
    input_path: str | Path,
    conversation_id: str,
    allow_low_confidence: bool = False,
) -> dict[str, Any]:
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    format_detected = "free_text"
    warnings: list[str] = []
    turns: list[NormalizedTurn]

    if path.suffix.lower() == ".jsonl":
        rows = _structured_rows_from_jsonl(text)
        turns, warnings = _turns_from_structured_rows(rows, conversation_id, path.name)
        format_detected = "structured_jsonl"
    elif path.suffix.lower() == ".json":
        rows = _structured_rows_from_json(json.loads(text))
        turns, warnings = _turns_from_structured_rows(rows, conversation_id, path.name)
        format_detected = "structured_json"
    elif stripped.startswith("{") or stripped.startswith("["):
        try:
            rows = _structured_rows_from_json(json.loads(text))
            turns, warnings = _turns_from_structured_rows(rows, conversation_id, path.name)
            format_detected = "structured_json"
        except Exception:
            turns, warnings = _parse_free_text_turns(text, conversation_id)
            format_detected = "free_text"
    else:
        turns, warnings = _parse_free_text_turns(text, conversation_id)

    exchanges = build_normalized_exchanges(turns)
    audit = summarize_normalization(
        conversation_id=conversation_id,
        input_path=path,
        input_format_detected=format_detected,
        turns=turns,
        exchanges=exchanges,
        warnings=warnings,
        allow_low_confidence=allow_low_confidence,
    )
    return {
        "turns": turns,
        "exchanges": exchanges,
        "audit": audit,
    }
