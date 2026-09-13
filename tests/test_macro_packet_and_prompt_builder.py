"""Tests for macro_packet_builder_v1.py and macro_prompt_builder_v1.py."""

from __future__ import annotations

from seg_eval.evaluation_packets.macro_packet_builder_v1 import build_macro_packet
from seg_eval.evaluation_judge.macro_prompt_builder_v1 import exchange_lines, user_prompt


def sample_exchanges():
    return [
        {"exchange_id": "ex_0000", "exchange_index": 0, "student_text": "meta instruction",
         "tutor_text": "ok", "turn_start": 0, "turn_end": 1},
        {"exchange_id": "ex_0001", "exchange_index": 1, "student_text": "How does X work?",
         "tutor_text": "X works like this.", "turn_start": 2, "turn_end": 3},
        {"exchange_id": "ex_0002", "exchange_index": 2, "student_text": "And Y?",
         "tutor_text": "Y works like that.", "turn_start": 4, "turn_end": 5},
    ]


def test_build_macro_packet_excludes_meta_exchanges():
    packet = build_macro_packet("dlg1", sample_exchanges(), excluded_exchange_ids={"ex_0000"})
    assert packet["exchange_count"] == 2
    ids = [ex["exchange_id"] for ex in packet["exchanges"]]
    assert ids == ["ex_0001", "ex_0002"]


def test_build_macro_packet_orders_by_exchange_index():
    exchanges = list(reversed(sample_exchanges()))
    packet = build_macro_packet("dlg1", exchanges, excluded_exchange_ids=set())
    ids = [ex["exchange_id"] for ex in packet["exchanges"]]
    assert ids == ["ex_0000", "ex_0001", "ex_0002"]


def test_build_macro_packet_no_exclusions_keeps_everything():
    packet = build_macro_packet("dlg1", sample_exchanges())
    assert packet["exchange_count"] == 3
    assert packet["dialogue_id"] == "dlg1"
    assert packet["packet_schema"] == "dialogue_macro_packet.v1"


def test_exchange_lines_role_labelled():
    packet = build_macro_packet("dlg1", sample_exchanges(), excluded_exchange_ids={"ex_0000"})
    lines = exchange_lines(packet)
    assert len(lines) == 2
    assert "[ex_0001] Student: How does X work?" in lines[0]
    assert "[ex_0001] Tutor: X works like this." in lines[0]


def test_user_prompt_contains_all_four_macro_dimensions():
    packet = build_macro_packet("dlg1", sample_exchanges())
    up = user_prompt(packet)
    for dim in ("adaptability", "consistency", "outcome_completion", "sequentiality"):
        assert dim in up


def test_user_prompt_never_mentions_kc_or_curriculum_grounding():
    # macro dimensions are dialogue-structure properties, not curriculum-correctness checks --
    # that's the per-segment judge's job, not this one's.
    packet = build_macro_packet("dlg1", sample_exchanges())
    up = user_prompt(packet)
    assert "kc_grounding" not in up
    assert "evaluator_kc_set" not in up
