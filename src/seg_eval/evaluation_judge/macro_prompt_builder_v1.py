"""Judge prompt builder for the full-dialogue macro rubric (macro_rubric_v1).

Mirrors evaluation_judge/prompt_builder.py's shape (system_prompt/user_prompt/
build_judge_prompt_packets) at the dialogue grain instead of the segment grain. Kept separate
because the macro judge answers about a whole dialogue with a different dimension set, different
output schema, and no KC grounding at all -- these dimensions (adaptability, consistency,
outcome_completion, sequentiality) are about dialogue-level structure, not curriculum content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .macro_rubric_v1 import MACRO_DIMENSIONS, MACRO_OUTPUT_SCHEMA, MACRO_RUBRIC_VERSION, MACRO_SCALE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exchange_lines(packet: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for ex in packet.get("exchanges") or []:
        exid = ex.get("exchange_id")
        student = ex.get("student_text") or ""
        tutor = ex.get("tutor_text") or ""
        lines.append(f"[{exid}] Student: {student}\n[{exid}] Tutor: {tutor}")
    return lines


def system_prompt() -> str:
    return (
        "You are an evidence-grounded tutor-evaluation judge assessing an entire multi-turn "
        "dialogue for properties that only emerge across turns, not within any single exchange. "
        "Evaluate only the supplied dialogue. Do not use outside knowledge except general "
        "pedagogical reasoning. Return only valid JSON matching the requested schema. Do not "
        "include hidden chain-of-thought; give concise evidence pointers and a short rationale."
    )


def user_prompt(packet: dict[str, Any], *, dimensions: list[dict[str, str]] | None = None,
                output_schema: dict[str, Any] | None = None,
                rubric_version: str | None = None) -> str:
    # Defaults are v1 exactly, so a v1 build stays byte-identical to every frozen prompt file.
    dims = MACRO_DIMENSIONS if dimensions is None else dimensions
    schema = MACRO_OUTPUT_SCHEMA if output_schema is None else output_schema
    version = MACRO_RUBRIC_VERSION if rubric_version is None else rubric_version
    has_correctness = any(d["id"] == "factual_correctness" for d in dims)
    rubric = {
        "rubric_version": version,
        "scale": MACRO_SCALE,
        "dimensions": [
            {"id": d["id"], "definition": d["proposal_definition"], "question": d["question"],
             "applicability": d["applicability"]}
            for d in dims
        ],
        "required_output_schema": schema,
    }
    if has_correctness:
        # v2 puts correctness in scope, but quarantined into its own dimension so the other four
        # keep measuring what they measured in v1 and remain comparable across rubric versions.
        scope_line = (
            "- Factual correctness IS in scope here, but ONLY through the factual_correctness "
            "dimension; do not let correctness influence the other dimensions. Per-segment "
            "curriculum grounding remains the separate per-segment judge's job.\n\n"
        )
    else:
        scope_line = (
            "- Do not evaluate curriculum correctness or KC grounding here; that is handled by "
            "the separate per-segment judge. Evaluate only the 4 macro dimensions.\n\n"
        )

    return (
        "Evaluate the following FULL student-tutor dialogue as a whole.\n\n"
        "Important semantics:\n"
        f"- Each of the {len(dims)} dimensions asks about a property that emerges ACROSS multiple "
        "exchanges, not within one exchange in isolation.\n"
        "- Score each dimension using only 0, 0.5, 1, or null when not applicable (null "
        "requires dimension_applicability = not_applicable, and the reverse).\n"
        "- Use 0 = poor/absent/harmful, 0.5 = partial/mixed/weak, 1 = good/clearly present.\n"
        "- Every applicable dimension needs at least one evidence_pointers entry citing a real "
        "exchange_id and a verbatim quote from that exchange -- never a quote you invented.\n"
        "- Keep each quote SHORT (under ~15 words) and copy it EXACTLY, character for character, "
        "from the exchange text below -- never use an ellipsis (\"...\") to shorten or join a "
        "quote, and never merge two separated spans into one quote. A short exact phrase is "
        "always acceptable evidence; a long paraphrase or an ellipsis-joined quote is not.\n"
        + scope_line +
        "Rubric and output schema:\n"
        f"{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        "Dialogue exchanges, in order:\n"
        + "\n".join(exchange_lines(packet))
        + "\n\nDialogue packet:\n"
        f"{json.dumps({'dialogue_id': packet.get('dialogue_id'), 'exchange_count': packet.get('exchange_count')}, indent=2, ensure_ascii=False)}"
    )


def build_macro_judge_prompt_packets(packet_dir: str | Path, out_dir: str | Path, *,
                                     dimensions: list[dict[str, str]] | None = None,
                                     output_schema: dict[str, Any] | None = None,
                                     rubric_version: str | None = None) -> dict[str, Any]:
    packet_dir = Path(packet_dir)
    out_dir = Path(out_dir)
    schema = MACRO_OUTPUT_SCHEMA if output_schema is None else output_schema
    version = MACRO_RUBRIC_VERSION if rubric_version is None else rubric_version
    prompt_kind = "v1" if dimensions is None else "v2"

    packet_path = packet_dir / "dialogue_macro_packets.jsonl"
    if not packet_path.exists():
        raise FileNotFoundError(f"Missing dialogue_macro_packets.jsonl: {packet_path}")

    packets = read_jsonl(packet_path)
    prompt_rows: list[dict[str, Any]] = []

    for packet in packets:
        prompt_rows.append({
            "judge_prompt_id": f"{packet['dialogue_id']}::macro_judge_prompt_{prompt_kind}",
            "segment_id": packet["dialogue_id"],  # reuses the runner's generic "segment_id" slot
            "dialogue_id": packet["dialogue_id"],
            "rubric_version": version,
            "system_prompt": system_prompt(),
            "user_prompt": user_prompt(packet, dimensions=dimensions, output_schema=schema,
                                       rubric_version=version),
            "expected_output_schema": schema,
            "source_packet_sha256": stable_hash(packet),
        })

    summary = {
        "prompt_pipeline": f"dialogue_macro_judge_prompts_{prompt_kind}",
        "rubric_version": version,
        "packet_dir": str(packet_dir),
        "prompt_count": len(prompt_rows),
    }

    write_jsonl(out_dir / "dialogue_macro_judge_prompts.jsonl", prompt_rows)
    write_json(out_dir / "macro_judge_prompt_summary.json", summary)
    return summary
