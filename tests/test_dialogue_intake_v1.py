from __future__ import annotations

import json
from pathlib import Path

from seg_eval.dialogue_intake_v1 import (
    build_normalized_dialogues_rows,
    normalize_dialogue_path,
    normalize_dialogue_text,
)
from seg_eval.dialogue_parser import parse_dialogue


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_explicit_labels_and_multiline_are_preserved() -> None:
    fixture = FIXTURES / "raw_dialogue_messy_teaching_styles_001.txt"
    result = normalize_dialogue_path(fixture, "style_messy_001")

    turns = result["turns"]
    exchanges = result["exchanges"]
    audit = result["audit"]

    assert audit["normalization_ready_for_segmentation"] is True
    assert turns[0].speaker == "student"
    assert "It still feels abstract to me." in turns[0].text
    assert turns[1].speaker == "tutor"
    assert "I will explain what it means" in turns[1].text
    assert exchanges[0].normalization_status == "ok"


def test_shorthand_and_braces_parse_correctly() -> None:
    shorthand_text = "S: Can you show a small worked example for Binary Decision Tree?\nT: Yes. For Binary Decision Tree, start with a tiny example."
    shorthand = normalize_dialogue_text(shorthand_text, "short_001")

    assert [turn.speaker for turn in shorthand["turns"]] == ["student", "tutor"]
    assert shorthand["audit"]["normalization_ready_for_segmentation"] is True

    braces = normalize_dialogue_path(FIXTURES / "raw_dialogue_student_tutor_braces_001.txt", "brace_001")
    assert [turn.speaker for turn in braces["turns"]] == ["student", "tutor", "student", "tutor"]
    assert braces["audit"]["normalization_ready_for_segmentation"] is True


def test_low_confidence_transcript_is_audited_and_not_ready() -> None:
    result = normalize_dialogue_path(FIXTURES / "raw_dialogue_low_confidence_001.txt", "low_conf_001")
    audit = result["audit"]

    assert audit["normalization_ready_for_segmentation"] is False
    assert "unknown_speaker_present" in audit["segmentation_hard_blockers"]
    assert any(turn.speaker == "unknown" for turn in result["turns"])


def test_turn_and_exchange_ids_are_deterministic() -> None:
    fixture = FIXTURES / "raw_dialogue_messy_teaching_styles_001.txt"
    first = normalize_dialogue_path(fixture, "deterministic_001")
    second = normalize_dialogue_path(fixture, "deterministic_001")

    assert [turn.turn_id for turn in first["turns"]] == [turn.turn_id for turn in second["turns"]]
    assert [exchange.exchange_id for exchange in first["exchanges"]] == [exchange.exchange_id for exchange in second["exchanges"]]


def test_normalized_dialogues_are_compatible_with_parse_dialogue() -> None:
    fixture = FIXTURES / "raw_dialogue_messy_teaching_styles_001.txt"
    result = normalize_dialogue_path(fixture, "compat_001")
    rows = build_normalized_dialogues_rows("compat_001", result["exchanges"])

    assert len(rows) == 1
    parsed_turns = parse_dialogue(rows[0])
    assert len(parsed_turns) == len(result["exchanges"]) * 2
    assert parsed_turns[0].role == "student"
    assert parsed_turns[1].role == "tutor"


def test_structured_json_input_is_supported(tmp_path: Path) -> None:
    payload = {
        "dialogue_id": "structured_source",
        "turns": [
            {"role": "student", "content": "Can you explain Agglomerative (Bottom-Up) Clustering in simple terms?"},
            {"role": "tutor", "content": "Agglomerative (Bottom-Up) Clustering is the target idea here."},
        ],
    }
    path = tmp_path / "structured.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = normalize_dialogue_path(path, "structured_001")
    assert result["audit"]["input_format_detected"] == "structured_json"
    assert result["audit"]["normalization_ready_for_segmentation"] is True
