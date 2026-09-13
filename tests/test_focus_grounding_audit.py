"""Tests for src/seg_eval/evaluation_judge/focus_grounding_audit.py.

Pure deterministic computation -- no model, no contract validation involved. These tests only
check the arithmetic/matching logic against constructed packets and responses.
"""

from __future__ import annotations

from seg_eval.evaluation_judge.focus_grounding_audit import compute_focus_grounding_audit


def packet_with_focus_target(target_type, pattern=None, focus_span=""):
    return {
        "focus_target": {
            "target_type": target_type, "pattern": pattern, "focus_span": focus_span,
            "source_exchange_ids": [], "signals": [],
        },
    }


def test_reads_target_type_and_pattern_directly_from_packet():
    packet = packet_with_focus_target("precise_focus_target", pattern="tutor_explanation",
                                       focus_span="Entropy measures impurity.")
    audit = compute_focus_grounding_audit(packet, {"evidence_quotes": []})
    assert audit["target_type"] == "precise_focus_target"
    assert audit["pattern"] == "tutor_explanation"


def test_anchored_true_when_a_quote_is_within_focus_span():
    packet = packet_with_focus_target(
        "precise_focus_target", pattern="tutor_explanation",
        focus_span="Entropy measures how mixed the class labels are in a node.",
    )
    response = {"evidence_quotes": ["measures how mixed the class labels are"]}
    audit = compute_focus_grounding_audit(packet, response)
    assert audit["computed_anchored_to_focus_span"] is True
    assert audit["matched_evidence_quotes"] == ["measures how mixed the class labels are"]


def test_anchored_false_when_no_quote_is_within_focus_span():
    packet = packet_with_focus_target(
        "precise_focus_target", pattern="tutor_explanation", focus_span="Entropy measures impurity.",
    )
    response = {"evidence_quotes": ["something entirely unrelated to the focus span"]}
    audit = compute_focus_grounding_audit(packet, response)
    assert audit["computed_anchored_to_focus_span"] is False
    assert audit["matched_evidence_quotes"] == []


def test_empty_focus_span_never_reports_anchored():
    packet = packet_with_focus_target("broad_tutor_claim_target", focus_span="")
    response = {"evidence_quotes": ["anything"]}
    audit = compute_focus_grounding_audit(packet, response)
    assert audit["focus_span_present"] is False
    assert audit["computed_anchored_to_focus_span"] is False


def test_no_evidence_quotes_is_not_anchored():
    packet = packet_with_focus_target("precise_focus_target", focus_span="Some text here.")
    audit = compute_focus_grounding_audit(packet, {"evidence_quotes": []})
    assert audit["computed_anchored_to_focus_span"] is False


def test_missing_focus_target_on_packet_handled_gracefully():
    audit = compute_focus_grounding_audit({}, {"evidence_quotes": ["x"]})
    assert audit["target_type"] is None
    assert audit["computed_anchored_to_focus_span"] is False


def test_normalization_is_case_and_whitespace_insensitive():
    packet = packet_with_focus_target(
        "precise_focus_target", focus_span="Entropy   measures   IMPURITY in a node.",
    )
    response = {"evidence_quotes": ["entropy measures impurity"]}
    audit = compute_focus_grounding_audit(packet, response)
    assert audit["computed_anchored_to_focus_span"] is True


def test_method_field_documents_this_is_not_model_reported():
    packet = packet_with_focus_target("precise_focus_target", focus_span="text")
    audit = compute_focus_grounding_audit(packet, {"evidence_quotes": []})
    assert "not model-reported" in audit["method"].lower()
