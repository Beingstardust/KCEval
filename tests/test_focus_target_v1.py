"""Tests for src/seg_eval/evaluation_packets/focus_target_v1.py."""

from __future__ import annotations

import pytest

from seg_eval.evaluation_packets.focus_target_v1 import (
    FocusTarget,
    classify_segment_focus,
    compute_focus_target,
)


def ex(exchange_id: str, student: str, tutor: str) -> dict:
    return {"exchange_id": exchange_id, "student_text": student, "tutor_text": tutor}


# ---------------------------------------------------------------------------
# FocusTarget construction validity
# ---------------------------------------------------------------------------


def test_invalid_target_type_rejected():
    with pytest.raises(ValueError):
        FocusTarget(target_type="not_a_real_type", pattern=None, focus_span="")


def test_invalid_pattern_rejected():
    with pytest.raises(ValueError):
        FocusTarget(target_type="broad_tutor_claim_target", pattern="not_a_real_pattern", focus_span="")


def test_precise_target_requires_a_pattern():
    with pytest.raises(ValueError):
        FocusTarget(target_type="precise_focus_target", pattern=None, focus_span="")


# ---------------------------------------------------------------------------
# confirmation (Scenario 1)
# ---------------------------------------------------------------------------


def test_confirmation_pattern_detected():
    exchanges = [ex("ex_0000",
        "Naive Bayes multiplies the prior by the likelihood of each feature, right?",
        "Exactly right! That's the core calculation.")]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "precise_focus_target"
    assert result.pattern == "confirmation"
    assert result.focus_span == exchanges[0]["student_text"]
    assert result.source_exchange_ids == ["ex_0000"]


def test_confirmation_not_triggered_when_tutor_turn_is_long():
    exchanges = [ex("ex_0000", "Is entropy always between 0 and 1?",
        "Exactly right, and let me also explain why: entropy is bounded by log2(k) for k "
        "classes, so with more than two classes it can exceed 1, which is a common point of "
        "confusion worth walking through in detail here.")]
    result = classify_segment_focus(exchanges)
    assert result.pattern != "confirmation"


def test_confirmation_not_triggered_when_tutor_also_corrects():
    exchanges = [ex("ex_0000", "So entropy is always between 0 and 1, right?",
        "Actually, that's not quite right -- it depends on the number of classes.")]
    result = classify_segment_focus(exchanges)
    assert result.pattern != "confirmation"


# ---------------------------------------------------------------------------
# tutor_correction (Scenario 3)
# ---------------------------------------------------------------------------


def test_tutor_correction_pattern_detected():
    exchanges = [ex("ex_0000",
        "So for spam detection, it's P(spam|'free'), not P('free'|spam), right?",
        "Actually, that's backwards. You need to calculate P('free'|spam), not P(spam|'free').")]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "precise_focus_target"
    assert result.pattern == "tutor_correction"
    assert "P(spam|'free')" in result.focus_span
    assert "P('free'|spam)" in result.focus_span
    assert result.focus_span.startswith("Student: ")
    assert "\nTutor: " in result.focus_span


def test_tutor_correction_not_triggered_without_a_student_assertion():
    # correction marker present but student turn is too short/thin to be a real claim
    exchanges = [ex("ex_0000", "ok", "Actually, that's not quite right.")]
    result = classify_segment_focus(exchanges)
    assert result.pattern != "tutor_correction"


def test_tutor_correction_not_triggered_by_pure_question_student_turn():
    exchanges = [ex("ex_0000", "How does Naive Bayes actually work in practice here?",
        "Actually, let me walk you through the formula step by step.")]
    # "actually" appears but the student turn is a pure question, not an assertion to correct
    result = classify_segment_focus(exchanges)
    assert result.pattern != "tutor_correction"


# ---------------------------------------------------------------------------
# self_correction (Scenario 2)
# ---------------------------------------------------------------------------


def test_self_correction_pattern_detected_in_later_exchange():
    exchanges = [
        ex("ex_0000", "First I find P(spam), then multiply by P('free'|ham), then compare.",
           "Good start, but step 2 is wrong. Should it be P('free'|spam)?"),
        ex("ex_0001", "Oh! It should be P('free'|spam)!", "Perfect, you got it."),
    ]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "precise_focus_target"
    assert result.pattern == "self_correction"
    assert result.focus_span == "Oh! It should be P('free'|spam)!"
    assert result.source_exchange_ids == ["ex_0001"]


def test_self_correction_not_triggered_in_first_exchange():
    # a self-correction marker in the FIRST exchange's student turn shouldn't count -- there's
    # nothing earlier for it to be correcting.
    exchanges = [ex("ex_0000", "Oh wait, actually I think entropy is bounded differently.",
                     "Let's work through that.")]
    result = classify_segment_focus(exchanges)
    assert result.pattern != "self_correction"


def test_self_correction_takes_priority_over_tutor_correction():
    # segment has both a plausible tutor_correction signal in exchange 0 AND a clear
    # self-correction marker in exchange 1 -- self_correction should win.
    exchanges = [
        ex("ex_0000", "So P(spam|'free') is what we compute directly, I believe.",
           "Actually, that's backwards -- you need P('free'|spam)."),
        ex("ex_0001", "Oh wait, I mean P('free'|spam), sorry I meant that.", "Exactly."),
    ]
    result = classify_segment_focus(exchanges)
    assert result.pattern == "self_correction"


# ---------------------------------------------------------------------------
# tutor_explanation (Scenario 4)
# ---------------------------------------------------------------------------


def test_tutor_explanation_pattern_detected():
    exchanges = [ex("ex_0000", "How does Naive Bayes work?",
        "It calculates P(class|features) using Bayes' theorem: find P(class) from training "
        "data, calculate P(feature|class) for each feature, multiply them together, and pick "
        "the class with highest probability.")]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "precise_focus_target"
    assert result.pattern == "tutor_explanation"
    assert result.focus_span == exchanges[0]["tutor_text"]


def test_tutor_explanation_not_triggered_when_student_makes_a_claim():
    exchanges = [ex("ex_0000", "I think Naive Bayes uses P(features|class) = 0.5 always.",
        "Not quite -- that value depends on the training data, it isn't fixed.")]
    result = classify_segment_focus(exchanges)
    # this should be caught as tutor_correction, not tutor_explanation
    assert result.pattern == "tutor_correction"


def test_confusion_phrase_triggers_tutor_explanation():
    exchanges = [ex("ex_0000", "I'm confused about how splitting works.",
        "Let's break it down: a split partitions the data by a threshold on one attribute.")]
    result = classify_segment_focus(exchanges)
    assert result.pattern == "tutor_explanation"


# ---------------------------------------------------------------------------
# fallback behaviour
# ---------------------------------------------------------------------------


def test_empty_segment_is_insufficient_or_ambiguous():
    result = classify_segment_focus([])
    assert result.target_type == "insufficient_or_ambiguous_target"
    assert result.pattern is None
    assert result.focus_span == ""


def test_no_tutor_content_is_no_domain_claim():
    exchanges = [ex("ex_0000", "Just thinking out loud here, not really asking anything.", "")]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "no_domain_claim"


def test_ambiguous_multi_turn_falls_back_to_broad_not_a_forced_pattern():
    # a longer, substantive back-and-forth with no clean marker of any single pattern --
    # must not be forced into one of the four precise patterns.
    exchanges = [
        ex("ex_0000",
           "Can we go over how bagging and boosting differ in terms of how they combine weak learners?",
           "Bagging trains learners independently on bootstrap samples and averages their "
           "predictions, which reduces variance. Boosting trains learners sequentially, each "
           "one focusing on the previous one's errors, which reduces bias."),
        ex("ex_0001",
           "And which one is more prone to overfitting on noisy data?",
           "Boosting tends to be more sensitive to noisy data and outliers because later "
           "learners keep focusing on hard-to-fit points, which can include mislabeled examples."),
    ]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "broad_tutor_claim_target"
    assert result.pattern is None
    assert "Bagging trains learners independently" in result.focus_span
    assert "Boosting tends to be more sensitive" in result.focus_span
    assert set(result.source_exchange_ids) == {"ex_0000", "ex_0001"}


def test_broad_fallback_never_used_when_a_precise_signal_exists():
    # sanity: the fallback path must not accidentally fire ahead of a clear confirmation.
    exchanges = [ex("ex_0000", "Is a leaf node one with entropy 0?", "Exactly.")]
    result = classify_segment_focus(exchanges)
    assert result.target_type == "precise_focus_target"


# ---------------------------------------------------------------------------
# never skip correctness checking: every classification produces SOME target
# ---------------------------------------------------------------------------


def test_never_returns_a_target_type_outside_the_defined_four():
    from seg_eval.evaluation_packets.focus_target_v1 import TARGET_TYPES

    cases = [
        [],
        [ex("a", "", "")],
        [ex("a", "hello", "hi")],
        [ex("a", "How does X work?", "It works like this.")],
        [ex("a", "So X=5, right?", "Actually X=6.")],
        [ex("a", "So X=5, right?", "Actually X=6."), ex("b", "Oh wait, I mean X=6.", "Right.")],
    ]
    for exchanges in cases:
        result = classify_segment_focus(exchanges)
        assert result.target_type in TARGET_TYPES


# ---------------------------------------------------------------------------
# packet-embeddable dict form
# ---------------------------------------------------------------------------


def test_compute_focus_target_returns_plain_dict_with_schema_version():
    exchanges = [ex("ex_0000", "Is a leaf node one with entropy 0?", "Exactly.")]
    d = compute_focus_target(exchanges)
    assert isinstance(d, dict)
    assert d["schema_version"] == "focus_target_v1"
    assert d["target_type"] == "precise_focus_target"
    assert d["pattern"] == "confirmation"
    assert d["source_exchange_ids"] == ["ex_0000"]
    assert isinstance(d["signals"], list) and d["signals"]


def test_focus_span_is_always_verbatim_substring_of_source_text():
    # the module's whole design premise: never synthesize new text. Check it holds for every
    # non-broad pattern by confirming the focus_span is contained in the concatenated raw
    # student/tutor text (broad target legitimately reformats with "Tutor: " prefixes, checked
    # separately by construction above).
    exchanges = [ex("ex_0000",
        "First I find P(spam), then multiply by P('free'|ham), then compare.",
        "Good start, but step 2 is wrong. Should it be P('free'|spam)?")]
    result = classify_segment_focus(exchanges)
    raw = exchanges[0]["student_text"] + exchanges[0]["tutor_text"]
    for line in result.focus_span.split("\n"):
        content = line.split(": ", 1)[-1]
        assert content in raw or content == ""
