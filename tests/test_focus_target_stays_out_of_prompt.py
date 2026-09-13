"""Locks in the redesign decided in
local_audits/evaluation_completion_20260824T093426Z/14_FOCUS_GUIDANCE_MINIMIZATION.md: after a
bounded experiment showed every tested way of exposing focus_target in the judge's prompt
measurably destabilized response-contract validity (52/52 -> 40-47/52), focus_target stays out
of the prompt entirely. It remains real, unremoved packet data (focus_target_v1.py's selection
rules are unchanged) -- these tests confirm it just never reaches compact_packet_context or
user_prompt, and that user_prompt is back to its plain, unparameterized pre-experiment form.
"""

from __future__ import annotations

import inspect

from seg_eval.evaluation_judge.prompt_builder import compact_packet_context, user_prompt


def packet_with_focus_target():
    return {
        "segment_id": "test::seg_0000",
        "focus_target": {
            "schema_version": "focus_target_v1", "target_type": "precise_focus_target",
            "pattern": "tutor_explanation", "focus_span": "X works like this.",
            "source_exchange_ids": ["ex_0000"], "signals": ["generic_question_student_turn"],
        },
        "member_exchanges": [{"exchange_id": "ex_0000", "student_text": "How does X work?",
                               "tutor_text": "X works like this."}],
    }


def test_compact_packet_context_never_includes_focus_target():
    ctx = compact_packet_context(packet_with_focus_target(), {})
    assert "focus_target" not in ctx


def test_user_prompt_takes_no_focus_guidance_parameter():
    sig = inspect.signature(user_prompt)
    assert list(sig.parameters) == ["context"]


def test_rendered_prompt_never_mentions_focus_target():
    ctx = compact_packet_context(packet_with_focus_target(), {})
    up = user_prompt(ctx)
    assert "focus_target" not in up
    assert "focus_span" not in up


def test_packet_focus_target_field_survives_untouched_for_code_use():
    # the packet itself (not the prompt context) still carries focus_target -- it's not deleted
    # from packets, only kept out of what the judge is shown.
    packet = packet_with_focus_target()
    assert packet["focus_target"]["target_type"] == "precise_focus_target"
