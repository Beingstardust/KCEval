from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import Any


SPEAKER_LINE_RE = re.compile(
    r"^\s*(?P<label>"
    r"student|tutor|teacher|learner|pupil|s|t"
    r")\s*(?P<form>:|\{|\])?\s*(?P<body>.*)$",
    re.IGNORECASE,
)

BRACKET_SPEAKER_RE = re.compile(
    r"^\s*\[(?P<label>student|tutor|teacher|learner|pupil|s|t)\]\s*(?P<body>.*)$",
    re.IGNORECASE,
)

ROLE_MAP = {
    "student": "Student",
    "learner": "Student",
    "pupil": "Student",
    "s": "Student",
    "tutor": "Tutor",
    "teacher": "Tutor",
    "t": "Tutor",
}


@dataclass(frozen=True)
class ExtractedTurn:
    speaker: str
    original_label: str
    start_line_1based: int
    end_line_1based_inclusive: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationExtractionResult:
    input_path: str
    conversation_id: str
    extraction_method: str
    ready_for_normalization: bool
    total_input_lines: int
    extracted_turn_count: int
    student_turn_count: int
    tutor_turn_count: int
    first_turn_line_1based: int | None
    last_turn_line_1based_inclusive: int | None
    removed_prefix_line_count: int
    removed_suffix_line_count: int
    warnings: list[str]
    turns: list[ExtractedTurn]
    conversation_text: str

    def to_audit_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("conversation_text", None)
        return data


def _match_speaker_line(line: str) -> tuple[str, str] | None:
    bracket = BRACKET_SPEAKER_RE.match(line)
    if bracket:
        raw = bracket.group("label").lower()
        role = ROLE_MAP.get(raw)
        if role:
            return role, bracket.group("body").strip()

    match = SPEAKER_LINE_RE.match(line)
    if not match:
        return None

    raw = match.group("label").lower()
    form = match.group("form")
    body = match.group("body").strip()

    if raw not in ROLE_MAP:
        return None

    # Avoid treating prose lines beginning with "student" as labels unless they use clear label syntax.
    if raw in {"student", "tutor", "teacher", "learner", "pupil"} and form not in {":", "{"}:
        return None

    # Single-letter S/T must use a colon.
    if raw in {"s", "t"} and form != ":":
        return None

    if form == "{" and body.endswith("}"):
        body = body[:-1].strip()

    return ROLE_MAP[raw], body


def extract_student_tutor_turns(input_path: str | Path, conversation_id: str) -> ConversationExtractionResult:
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    warnings: list[str] = []
    turns: list[ExtractedTurn] = []

    current_speaker: str | None = None
    current_label: str | None = None
    current_start: int | None = None
    current_lines: list[str] = []

    def flush(end_line_1based: int) -> None:
        nonlocal current_speaker, current_label, current_start, current_lines, turns
        if current_speaker is None or current_start is None:
            return
        body = "\n".join(line.rstrip() for line in current_lines).strip()
        turns.append(
            ExtractedTurn(
                speaker=current_speaker,
                original_label=current_label or current_speaker,
                start_line_1based=current_start,
                end_line_1based_inclusive=end_line_1based,
                text=body,
            )
        )
        current_speaker = None
        current_label = None
        current_start = None
        current_lines = []

    for idx, line in enumerate(lines):
        line_no = idx + 1
        matched = _match_speaker_line(line)

        if matched:
            flush(line_no - 1)
            speaker, body = matched
            current_speaker = speaker
            current_label = speaker
            current_start = line_no
            current_lines = [body] if body else []
            continue

        if current_speaker is not None:
            current_lines.append(line)

    flush(len(lines))

    student_count = sum(1 for turn in turns if turn.speaker == "Student")
    tutor_count = sum(1 for turn in turns if turn.speaker == "Tutor")

    if len(turns) < 2:
        warnings.append("fewer_than_two_student_tutor_turns_detected")
    if student_count == 0:
        warnings.append("no_student_turn_detected")
    if tutor_count == 0:
        warnings.append("no_tutor_turn_detected")
    if turns and turns[0].speaker != "Student":
        warnings.append("first_detected_turn_is_not_student")
    if turns:
        for prev, cur in zip(turns, turns[1:]):
            if prev.speaker == cur.speaker:
                warnings.append(
                    f"consecutive_same_speaker_turns_detected_at_lines_{prev.start_line_1based}_{cur.start_line_1based}"
                )
                break

    ready = len(turns) >= 2 and student_count > 0 and tutor_count > 0

    conversation_lines: list[str] = []
    for turn in turns:
        conversation_lines.append(f"{turn.speaker}: {turn.text}".rstrip())
        conversation_lines.append("")
    conversation_text = "\n".join(conversation_lines).strip() + "\n" if conversation_lines else ""

    first_line = turns[0].start_line_1based if turns else None
    last_line = turns[-1].end_line_1based_inclusive if turns else None

    return ConversationExtractionResult(
        input_path=str(path),
        conversation_id=conversation_id,
        extraction_method="auto_student_tutor_turn_scan",
        ready_for_normalization=ready,
        total_input_lines=len(lines),
        extracted_turn_count=len(turns),
        student_turn_count=student_count,
        tutor_turn_count=tutor_count,
        first_turn_line_1based=first_line,
        last_turn_line_1based_inclusive=last_line,
        removed_prefix_line_count=(first_line - 1) if first_line else len(lines),
        removed_suffix_line_count=(len(lines) - last_line) if last_line else 0,
        warnings=warnings,
        turns=turns,
        conversation_text=conversation_text,
    )


def extract_conversation_region(
    input_path: str | Path,
    conversation_id: str,
    require_markers: bool = False,
    start_regex: str | None = None,
    end_regex: str | None = None,
) -> ConversationExtractionResult:
    # Backwards-compatible function name.
    # Markers are intentionally ignored in v1 auto mode.
    # The contract is now: extract explicit Student/Tutor turns automatically.
    if require_markers:
        # Kept for CLI compatibility, but no longer needed.
        pass
    return extract_student_tutor_turns(input_path=input_path, conversation_id=conversation_id)


def render_extraction_audit_markdown(result: ConversationExtractionResult) -> str:
    audit = result.to_audit_dict()
    lines = [
        "# Conversation region extraction audit",
        "",
        f"- Input path: `{audit['input_path']}`",
        f"- Conversation id: `{audit['conversation_id']}`",
        f"- Extraction method: `{audit['extraction_method']}`",
        f"- Ready for normalization: `{audit['ready_for_normalization']}`",
        f"- Total input lines: `{audit['total_input_lines']}`",
        f"- Extracted turn count: `{audit['extracted_turn_count']}`",
        f"- Student turn count: `{audit['student_turn_count']}`",
        f"- Tutor turn count: `{audit['tutor_turn_count']}`",
        f"- First turn line: `{audit['first_turn_line_1based']}`",
        f"- Last turn line inclusive: `{audit['last_turn_line_1based_inclusive']}`",
        f"- Removed prefix lines: `{audit['removed_prefix_line_count']}`",
        f"- Removed suffix lines: `{audit['removed_suffix_line_count']}`",
        "",
        "## Warnings",
        "",
    ]
    if audit["warnings"]:
        lines.extend([f"- `{warning}`" for warning in audit["warnings"]])
    else:
        lines.append("- none")
    lines.extend(["", "## Extracted turns", ""])
    for turn in result.turns:
        preview = turn.text.replace("\n", " ")[:160]
        lines.append(f"- `{turn.speaker}` lines `{turn.start_line_1based}-{turn.end_line_1based_inclusive}`: {preview}")
    lines.append("")
    return "\n".join(lines)


def write_extraction_outputs(result: ConversationExtractionResult, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conversation_path = out / "conversation_only.txt"
    turns_path = out / "conversation_only_turns.jsonl"
    audit_json_path = out / "conversation_extraction_audit.json"
    audit_md_path = out / "conversation_extraction_audit.md"

    conversation_path.write_text(result.conversation_text, encoding="utf-8")
    turns_path.write_text(
        "\n".join(json.dumps(turn.to_dict(), ensure_ascii=False, sort_keys=True) for turn in result.turns) + ("\n" if result.turns else ""),
        encoding="utf-8",
    )
    audit_json_path.write_text(json.dumps(result.to_audit_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    audit_md_path.write_text(render_extraction_audit_markdown(result), encoding="utf-8")

    return {
        "conversation_only": str(conversation_path),
        "conversation_only_turns": str(turns_path),
        "audit_json": str(audit_json_path),
        "audit_md": str(audit_md_path),
    }
