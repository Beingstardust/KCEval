"""Tests for src/seg_eval/evaluation_judge/response_contract_v2.py.

No coverage existed for this module before this pass (confirmed by a repo-wide search during
the evaluation-completion audit -- see
local_audits/evaluation_completion_20260824T093426Z/06_CURRENT_EVALUATION_GAPS.md item 8), even
though it is the contract every judge response in the repo is validated against. These tests
build one canonical valid response and mutate it per-test, rather than re-deriving the schema,
so each test isolates exactly one contract rule.
"""

from __future__ import annotations

import copy

from seg_eval.evaluation_judge.response_contract_v2 import (
    allowed_kc_ids_from_packet,
    extract_response_payload,
    packet_segment_text,
    validate_judge_response,
    validate_response_rows,
)
from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS

SEGMENT_TEXT = (
    "Student: What does entropy measure in a decision tree split?\n"
    "Tutor: Entropy measures how mixed the class labels are in a node; a pure node has entropy 0."
)


def _all_dims(value):
    return {d: value for d in PARTIAL_DIMENSIONS}


def _all_applicable():
    return {d: "applicable" for d in PARTIAL_DIMENSIONS}


def _all_trust(value):
    return {f: value for f in TRUST_FLAGS}


def valid_response() -> dict:
    return {
        "segment_id": "seg_0000",
        "evaluation_mode": "single_kc_primary",
        "raw_dimension_scores": _all_dims(1),
        "dimension_applicability": _all_applicable(),
        "dependency_adjustments": [],
        "dimension_scores": _all_dims(1),
        "trust_flags": _all_trust(0),
        "micro_score_raw": 1.0,
        "micro_score_dependency_adjusted": 1.0,
        "trust_adjusted_score": 1.0,
        "kc_grounding": {
            "primary_kc_used": True,
            "context_kcs_used": [],
            "unsupported_or_wrong_kc_claims": [],
        },
        "flags": {
            "major_correctness_error": False, "ungrounded_claim": False,
            "missed_student_need": False, "overly_answer_giving": False,
            "segment_boundary_problem": False, "needs_human_review": False,
        },
        "evidence_quotes": ["Entropy measures how mixed the class labels are in a node"],
        "rationale_short": "The tutor gave a clear, correct definition of entropy for this split question.",
    }


PACKET = {"segment_id": "seg_0000", "segment_text": SEGMENT_TEXT}


def test_valid_response_has_no_errors():
    assert validate_judge_response(
        valid_response(), expected_segment_id="seg_0000",
        expected_evaluation_mode="single_kc_primary", allowed_segment_text=SEGMENT_TEXT,
    ) == []


def test_missing_required_field_detected():
    r = valid_response()
    del r["rationale_short"]
    errors = validate_judge_response(r)
    assert any("missing_required_field:rationale_short" in e for e in errors)


def test_unexpected_top_level_field_detected():
    r = valid_response()
    r["extra_made_up_field"] = 123
    errors = validate_judge_response(r)
    assert any("unexpected_top_level_fields" in e for e in errors)


def test_segment_id_mismatch_detected():
    r = valid_response()
    errors = validate_judge_response(r, expected_segment_id="seg_9999")
    assert any("segment_id_mismatch" in e for e in errors)


def test_evaluation_mode_not_in_allowed_set():
    r = valid_response()
    r["evaluation_mode"] = "made_up_mode"
    errors = validate_judge_response(r)
    assert any("evaluation_mode_not_allowed" in e for e in errors)


def test_evaluation_mode_mismatch_with_prompt():
    r = valid_response()  # response says single_kc_primary
    errors = validate_judge_response(r, expected_evaluation_mode="branch_context_with_anchor")
    assert any("evaluation_mode_mismatch" in e for e in errors)


def test_score_must_be_null_when_not_applicable():
    r = valid_response()
    r["dimension_applicability"]["proactive_clarification"] = "not_applicable"
    # raw_dimension_scores still has a numeric value for it -- contract violation
    errors = validate_judge_response(r)
    assert any("must_be_null_when_not_applicable" in e for e in errors)


def test_null_score_requires_not_applicable():
    r = valid_response()
    r["raw_dimension_scores"]["proactive_clarification"] = None
    r["dimension_scores"]["proactive_clarification"] = None
    # applicability still says "applicable" -- contract violation
    errors = validate_judge_response(r)
    assert any("null_only_allowed_when_not_applicable" in e for e in errors)


def test_score_outside_allowed_scale_rejected():
    r = valid_response()
    r["raw_dimension_scores"]["error_detection"] = 0.75
    errors = validate_judge_response(r)
    assert any("score_not_allowed_0_0.5_1" in e for e in errors)


def test_missing_dimension_in_score_block_detected():
    r = valid_response()
    del r["raw_dimension_scores"]["error_detection"]
    errors = validate_judge_response(r)
    assert any("raw_dimension_scores.missing:error_detection" in e for e in errors)


def test_extra_dimension_in_score_block_detected():
    r = valid_response()
    r["raw_dimension_scores"]["not_a_real_dimension"] = 1
    errors = validate_judge_response(r)
    assert any("unexpected_dimensions" in e for e in errors)


def test_trust_flag_must_be_0_or_1():
    r = valid_response()
    r["trust_flags"]["curriculum_hallucination_present"] = True  # bool, not 0/1 int
    errors = validate_judge_response(r)
    # Python bool is a subclass of int and True == 1, so this specific case is accepted --
    # document that explicitly rather than assume a bare bool is rejected.
    assert errors == []


def test_trust_flag_invalid_value_rejected():
    r = valid_response()
    r["trust_flags"]["curriculum_hallucination_present"] = 2
    errors = validate_judge_response(r)
    assert any("trust_flags.curriculum_hallucination_present:must_be_0_or_1" in e for e in errors)


def test_aggregate_score_out_of_range_rejected():
    r = valid_response()
    r["micro_score_raw"] = 1.5
    errors = validate_judge_response(r)
    assert any("micro_score_raw:out_of_range_0_1" in e for e in errors)


def test_dependency_adjustment_missing_fields():
    r = valid_response()
    r["dependency_adjustments"] = [{"rule_id": "D_X"}]  # missing applied/reason/affected_dimensions
    errors = validate_judge_response(r)
    assert any("dependency_adjustments[0].missing" in e for e in errors)


def test_dependency_adjustment_applied_true_requires_nonempty_affected():
    r = valid_response()
    r["dependency_adjustments"] = [{
        "rule_id": "D_X", "applied": True, "reason": "because reasons", "affected_dimensions": [],
    }]
    errors = validate_judge_response(r)
    assert any("affected_dimensions:empty_when_applied" in e for e in errors)


def test_dependency_adjustment_reason_too_short():
    r = valid_response()
    r["dependency_adjustments"] = [{
        "rule_id": "D_X", "applied": False, "reason": "no", "affected_dimensions": [],
    }]
    errors = validate_judge_response(r)
    assert any("reason:too_short" in e for e in errors)


def test_dependency_adjustment_unknown_affected_dimension():
    r = valid_response()
    r["dependency_adjustments"] = [{
        "rule_id": "D_X", "applied": True, "reason": "because it applies here",
        "affected_dimensions": ["not_a_real_dimension"],
    }]
    errors = validate_judge_response(r)
    assert any("unknown_dimension" in e for e in errors)


def test_kc_grounding_context_kc_must_be_in_allowed_set():
    r = valid_response()
    r["kc_grounding"]["context_kcs_used"] = ["KC_NOT_IN_PACKET"]
    errors = validate_judge_response(r, allowed_kc_ids={"KC_REAL_001"})
    assert any("kc_not_in_packet" in e for e in errors)


def test_flags_missing_field_detected():
    r = valid_response()
    del r["flags"]["needs_human_review"]
    errors = validate_judge_response(r)
    assert any("flags.missing:needs_human_review" in e for e in errors)


def test_evidence_quotes_must_be_verbatim_substring():
    r = valid_response()
    r["evidence_quotes"] = ["Entropy measures how mixed the labels roughly are"]  # paraphrase
    errors = validate_judge_response(r, allowed_segment_text=SEGMENT_TEXT)
    assert any("not_found_in_segment_text" in e for e in errors)


def test_evidence_quote_matches_role_labelled_form():
    # the judge prompt shows exchanges as "Student: .../Tutor: ..."; a quote reproducing that
    # framing verbatim (matching what the model was actually shown) must validate even though
    # packet.segment_text itself has no role labels. Regression test for a real failure found
    # on a live run: 4/4 evidence quotes in one response were exact copies of full exchanges
    # in this labelled form and all failed before packet_segment_text() included it.
    packet = {
        "segment_text": "",
        "member_exchanges": [
            {"student_text": "What is entropy?", "tutor_text": "It measures label impurity."},
        ],
    }
    text = packet_segment_text(packet)
    r = valid_response()
    r["evidence_quotes"] = ["Student: What is entropy?\nTutor: It measures label impurity."]
    errors = validate_judge_response(r, allowed_segment_text=text)
    assert not any("not_found_in_segment_text" in e for e in errors)


def test_rationale_step_by_step_describing_tutor_is_not_flagged():
    # "step-by-step reasoning" describing the TUTOR's scaffolding technique is normal,
    # legitimate rationale text, not a hidden-chain-of-thought leak. Regression test for a real
    # false positive found on a live run.
    r = valid_response()
    r["rationale_short"] = (
        "The tutor scaffolds well, using step-by-step reasoning to walk the student through "
        "the calculation without giving the answer away too early."
    )
    errors = validate_judge_response(r)
    assert not any("mentions_hidden_reasoning" in e for e in errors)


def test_rationale_self_referential_step_by_step_is_still_flagged():
    r = valid_response()
    r["rationale_short"] = (
        "Following my step-by-step reasoning through this segment, the tutor did fine here."
    )
    errors = validate_judge_response(r)
    assert any("mentions_hidden_reasoning" in e for e in errors)


def test_evidence_quotes_ellipsis_truncation_fails():
    r = valid_response()
    r["evidence_quotes"] = ["Entropy measures how mixed the class labels are..."]
    errors = validate_judge_response(r, allowed_segment_text=SEGMENT_TEXT)
    assert any("not_found_in_segment_text" in e for e in errors)


def test_evidence_quotes_empty_rejected():
    r = valid_response()
    r["evidence_quotes"] = []
    errors = validate_judge_response(r)
    assert any("evidence_quotes:empty" in e for e in errors)


def test_evidence_quotes_too_many_rejected():
    r = valid_response()
    r["evidence_quotes"] = ["ok quote here"] * 7
    errors = validate_judge_response(r, allowed_segment_text=None)
    assert any("evidence_quotes:too_many" in e for e in errors)


def test_rationale_too_short_rejected():
    r = valid_response()
    r["rationale_short"] = "too short"
    errors = validate_judge_response(r)
    assert any("rationale_short:too_short" in e for e in errors)


def test_rationale_mentions_hidden_reasoning_rejected():
    r = valid_response()
    r["rationale_short"] = "Following my hidden chain of thought reasoning, the tutor did fine here."
    errors = validate_judge_response(r)
    assert any("mentions_hidden_reasoning" in e for e in errors)


# ---------------------------------------------------------------------------
# extract_response_payload / packet_segment_text / allowed_kc_ids_from_packet
# ---------------------------------------------------------------------------


def test_extract_payload_from_wrapped_judge_output_key():
    row = {"judge_output": valid_response()}
    payload, errors = extract_response_payload(row)
    assert errors == []
    assert payload["segment_id"] == "seg_0000"


def test_extract_payload_from_raw_response_text_json_string():
    import json
    row = {"raw_response_text": json.dumps(valid_response())}
    payload, errors = extract_response_payload(row)
    assert errors == []
    assert payload["segment_id"] == "seg_0000"


def test_extract_payload_invalid_json_string_reports_error():
    row = {"raw_response_text": "not json at all {"}
    payload, errors = extract_response_payload(row)
    assert payload is None
    assert any("raw_response_text_not_json" in e for e in errors)


def test_extract_payload_bare_response_row():
    row = valid_response()  # already has segment_id/raw_dimension_scores/trust_flags at top level
    payload, errors = extract_response_payload(row)
    assert errors == []
    assert payload is row


def test_extract_payload_unrecognized_shape():
    payload, errors = extract_response_payload({"nothing_useful": 1})
    assert payload is None
    assert errors == ["could_not_find_response_payload"]


def test_packet_segment_text_concatenates_segment_text_and_member_exchanges():
    # not an either/or fallback -- both sources are unioned, so a quote can be matched whether
    # it came from the assembled segment_text or from an individual member exchange's turns.
    packet = {"segment_text": "hello world", "member_exchanges": [{"student_text": "also this"}]}
    text = packet_segment_text(packet)
    assert "hello world" in text
    assert "also this" in text


def test_packet_segment_text_falls_back_to_member_exchanges():
    packet = {"member_exchanges": [
        {"student_text": "Q1"}, {"tutor_text": "A1"},
    ]}
    text = packet_segment_text(packet)
    assert "Q1" in text and "A1" in text


def test_allowed_kc_ids_from_packet_collects_primary_and_context():
    packet = {
        "evaluation_target": {"primary_kc_id": "KC_A", "evaluator_kc_set": ["KC_B", "KC_C"]},
        "kc_context": [{"unit_id": "KC_D"}],
    }
    ids = allowed_kc_ids_from_packet(packet)
    assert ids == {"KC_A", "KC_B", "KC_C", "KC_D"}


# ---------------------------------------------------------------------------
# validate_response_rows: batch-level PASS/FAIL, missing/duplicate detection
# ---------------------------------------------------------------------------


def test_validate_response_rows_pass_when_everything_lines_up():
    prompt_rows = [{"segment_id": "seg_0000", "evaluation_mode": "single_kc_primary"}]
    packet_rows = [PACKET]
    response_rows = [{"judge_output": valid_response()}]
    valid_rows, error_rows, summary = validate_response_rows(
        prompt_rows=prompt_rows, packet_rows=packet_rows, response_rows=response_rows,
    )
    assert summary["status"] == "PASS"
    assert len(valid_rows) == 1
    assert error_rows == []


def test_validate_response_rows_flags_missing_segment():
    prompt_rows = [
        {"segment_id": "seg_0000", "evaluation_mode": "single_kc_primary"},
        {"segment_id": "seg_0001", "evaluation_mode": "single_kc_primary"},
    ]
    packet_rows = [PACKET, {"segment_id": "seg_0001", "segment_text": "other text"}]
    response_rows = [{"judge_output": valid_response()}]  # only seg_0000 answered
    _, _, summary = validate_response_rows(
        prompt_rows=prompt_rows, packet_rows=packet_rows, response_rows=response_rows,
    )
    assert summary["status"] == "FAIL"
    assert summary["missing_response_segment_count"] == 1
    assert "seg_0001" in summary["missing_response_segments"]


def test_validate_response_rows_flags_duplicate_segment():
    prompt_rows = [{"segment_id": "seg_0000", "evaluation_mode": "single_kc_primary"}]
    packet_rows = [PACKET]
    response_rows = [{"judge_output": valid_response()}, {"judge_output": valid_response()}]
    _, _, summary = validate_response_rows(
        prompt_rows=prompt_rows, packet_rows=packet_rows, response_rows=response_rows,
    )
    assert summary["duplicate_segment_response_count"] == 1
    assert summary["status"] == "FAIL"


def test_validate_response_rows_one_bad_one_good():
    prompt_rows = [
        {"segment_id": "seg_0000", "evaluation_mode": "single_kc_primary"},
        {"segment_id": "seg_0001", "evaluation_mode": "single_kc_primary"},
    ]
    packet_rows = [PACKET, {"segment_id": "seg_0001", "segment_text": "other text"}]
    bad = valid_response()
    bad["segment_id"] = "seg_0001"
    del bad["rationale_short"]
    response_rows = [{"judge_output": valid_response()}, {"judge_output": bad}]
    valid_rows, error_rows, summary = validate_response_rows(
        prompt_rows=prompt_rows, packet_rows=packet_rows, response_rows=response_rows,
    )
    assert len(valid_rows) == 1
    assert len(error_rows) == 1
    assert summary["status"] == "FAIL"
    assert summary["valid_count"] == 1
    assert summary["error_count"] == 1
