"""Tests for macro_rubric_v1 + macro_response_contract_v1."""

from __future__ import annotations

import copy

from seg_eval.evaluation_judge.macro_response_contract_v1 import validate_macro_response
from seg_eval.evaluation_judge.macro_rubric_v1 import MACRO_DIMENSION_IDS, MACRO_DIMENSIONS

EXCHANGES = {
    "ex_0000": "Student: How does entropy relate to information gain? Tutor: Information gain is the reduction in entropy after a split.",
    "ex_0001": "Student: So lower entropy is better? Tutor: Yes, a pure node has entropy 0, which is the best case.",
}


def valid_macro_response() -> dict:
    return {
        "dialogue_id": "dm1_rich_dialogue",
        "dimension_scores": {
            "adaptability": 1, "consistency": 1, "outcome_completion": 1, "sequentiality": 1,
        },
        "dimension_applicability": {d: "applicable" for d in MACRO_DIMENSION_IDS},
        "evidence_pointers": [
            {"dimension": "adaptability", "exchange_id": "ex_0001",
             "quote": "a pure node has entropy 0"},
            {"dimension": "consistency", "exchange_id": "ex_0000",
             "quote": "Information gain is the reduction in entropy after a split"},
            {"dimension": "outcome_completion", "exchange_id": "ex_0001",
             "quote": "which is the best case"},
            {"dimension": "sequentiality", "exchange_id": "ex_0000",
             "quote": "Information gain is the reduction in entropy after a split"},
        ],
        "rationale_short": "The tutor built the explanation logically across two exchanges without contradiction.",
    }


def test_macro_dimensions_match_proposal_ids():
    assert MACRO_DIMENSION_IDS == [
        "adaptability", "consistency", "outcome_completion", "sequentiality",
    ]
    assert len(MACRO_DIMENSIONS) == 4


def test_valid_macro_response_has_no_errors():
    errors = validate_macro_response(
        valid_macro_response(), expected_dialogue_id="dm1_rich_dialogue",
        exchange_text_by_id=EXCHANGES,
    )
    assert errors == []


def test_missing_field_detected():
    r = valid_macro_response()
    del r["rationale_short"]
    assert any("missing_required_field:rationale_short" in e for e in validate_macro_response(r))


def test_dialogue_id_mismatch_detected():
    errors = validate_macro_response(valid_macro_response(), expected_dialogue_id="other_dialogue")
    assert any("dialogue_id_mismatch" in e for e in errors)


def test_score_null_requires_not_applicable():
    r = valid_macro_response()
    r["dimension_scores"]["adaptability"] = None
    errors = validate_macro_response(r)
    assert any("null_only_allowed_when_not_applicable" in e for e in errors)


def test_not_applicable_requires_null_score():
    r = valid_macro_response()
    r["dimension_applicability"]["sequentiality"] = "not_applicable"
    errors = validate_macro_response(r)
    assert any("sequentiality:must_be_null_when_not_applicable" in e for e in errors)
    # remove its evidence pointer too, since not_applicable dims shouldn't need one and the
    # dangling applicable-evidence check is orthogonal to this test
    r["evidence_pointers"] = [p for p in r["evidence_pointers"] if p["dimension"] != "sequentiality"]
    errors2 = validate_macro_response(r)
    assert not any("missing_for_applicable_dimension:sequentiality" in e for e in errors2)


def test_score_out_of_scale_rejected():
    r = valid_macro_response()
    r["dimension_scores"]["consistency"] = 0.25
    errors = validate_macro_response(r)
    assert any("score_not_allowed_0_0.5_1" in e for e in errors)


def test_applicable_dimension_without_evidence_rejected():
    r = valid_macro_response()
    r["evidence_pointers"] = [p for p in r["evidence_pointers"] if p["dimension"] != "consistency"]
    errors = validate_macro_response(r)
    assert any("missing_for_applicable_dimension:consistency" in e for e in errors)


def test_evidence_quote_must_match_its_claimed_exchange():
    r = valid_macro_response()
    # claim evidence is from ex_0000 but the quote is actually only in ex_0001
    r["evidence_pointers"][0] = {
        "dimension": "adaptability", "exchange_id": "ex_0000",
        "quote": "a pure node has entropy 0",
    }
    errors = validate_macro_response(r, exchange_text_by_id=EXCHANGES)
    assert any("not_found_in_exchange" in e for e in errors)


def test_evidence_pointer_unknown_dimension_rejected():
    r = valid_macro_response()
    r["evidence_pointers"].append(
        {"dimension": "not_a_real_macro_dim", "exchange_id": "ex_0000", "quote": "Student"}
    )
    errors = validate_macro_response(r, exchange_text_by_id=EXCHANGES)
    assert any("dimension:unknown" in e for e in errors)


def test_rationale_too_short_rejected():
    r = valid_macro_response()
    r["rationale_short"] = "too short"
    assert any("rationale_short:too_short" in e for e in validate_macro_response(r))


def test_unexpected_top_level_field_rejected():
    r = valid_macro_response()
    r["extra_field"] = 1
    assert any("unexpected_top_level_fields" in e for e in validate_macro_response(r))
