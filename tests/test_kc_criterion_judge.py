"""Tests for the KC-specific criterion judging layer (kc_criterion_judge_v1).

The integrity property matters most here: scoring must use LIBRARY weights, never the model's
echoed copy, so a model cannot change a criterion's weight or reattribute it to another KC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.kc_criterion_judge_v1 import (  # noqa: E402
    load_reviewer_criteria, library_criteria_by_id, candidate_criteria_for_packet,
    validate_criterion_response, user_prompt, REVIEWED_LIBRARY,
)
from seg_eval.aggregation.kc_criterion_scoring import (  # noqa: E402
    CriterionJudgment, score_kc_criteria, check_criterion_integrity,
)

LIB = REPO_ROOT / REVIEWED_LIBRARY
requires_lib = pytest.mark.skipif(not LIB.exists(), reason="reviewed library not present")

PACKET = {
    "segment_id": "d::seg_0000",
    "evaluation_target": {"primary_kc_id": "KC_A", "evaluator_kc_set": ["KC_A", "KC_B"]},
    "member_exchanges": [{"exchange_id": "ex_1", "student_text": "s", "tutor_text": "t"}],
}
CRITERIA = {
    "KC_A": [{"criterion_id": "KC_A::crit_00", "kc_id": "KC_A", "criterion_text": "does a good thing",
               "polarity": "positive", "weight": 2}],
    "KC_B": [{"criterion_id": "KC_B::crit_00", "kc_id": "KC_B", "criterion_text": "does a bad thing",
               "polarity": "negative", "weight": -1}],
}


# ---------------------------------------------------------------------------
# candidate selection
# ---------------------------------------------------------------------------

def test_candidates_drawn_from_primary_and_evaluator_set_deduplicated():
    got = candidate_criteria_for_packet(PACKET, CRITERIA)
    assert [c["criterion_id"] for c in got] == ["KC_A::crit_00", "KC_B::crit_00"]


def test_candidates_include_non_primary_matched_kcs_when_primary_has_no_criteria():
    packet = {
        "segment_id": "d::seg_multi",
        "evaluation_target": {
            "primary_kc_id": "KC_NO_CRITERIA",
            "evaluator_kc_set": ["KC_NO_CRITERIA", "KC_B", "KC_C"],
        },
        "member_exchanges": [{"exchange_id": "ex_1", "student_text": "s", "tutor_text": "t"}],
    }
    criteria = {
        "KC_B": [{"criterion_id": "KC_B::crit_00", "kc_id": "KC_B", "criterion_text": "b",
                  "polarity": "positive", "weight": 1}],
        "KC_C": [{"criterion_id": "KC_C::crit_00", "kc_id": "KC_C", "criterion_text": "c",
                  "polarity": "positive", "weight": 1}],
    }

    got = candidate_criteria_for_packet(packet, criteria)

    assert [c["criterion_id"] for c in got] == ["KC_B::crit_00", "KC_C::crit_00"]


def test_no_candidates_when_no_kc_has_criteria():
    assert candidate_criteria_for_packet(PACKET, {}) == []


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

def test_prompt_tells_judge_to_rate_occurrence_not_desirability():
    """Negative-polarity criteria are inverted downstream; if the judge also inverts them the
    sign is applied twice."""
    p = user_prompt(PACKET, candidate_criteria_for_packet(PACKET, CRITERIA))
    assert "Do NOT try to decide whether that is" in p
    assert "inverts it automatically downstream" in p


def test_prompt_contains_criteria_and_segment_text():
    p = user_prompt(PACKET, candidate_criteria_for_packet(PACKET, CRITERIA))
    assert "does a good thing" in p and "does a bad thing" in p
    assert "Tutor: t" in p


# ---------------------------------------------------------------------------
# response contract
# ---------------------------------------------------------------------------

def _resp(judgments, sid="d::seg_0000"):
    return {"segment_id": sid, "criterion_judgments": judgments}


def test_valid_response_has_no_errors():
    crits = candidate_criteria_for_packet(PACKET, CRITERIA)
    r = _resp([
        {"criterion_id": "KC_A::crit_00", "kc_id": "KC_A", "weight": 2, "applicability": "applicable", "rating": 1},
        {"criterion_id": "KC_B::crit_00", "kc_id": "KC_B", "weight": -1, "applicability": "not_applicable", "rating": None},
    ])
    assert validate_criterion_response(r, expected_segment_id="d::seg_0000", expected_criteria=crits) == []


def test_rating_must_be_null_when_not_applicable():
    crits = candidate_criteria_for_packet(PACKET, CRITERIA)
    r = _resp([
        {"criterion_id": "KC_A::crit_00", "kc_id": "KC_A", "weight": 2, "applicability": "not_applicable", "rating": 0},
        {"criterion_id": "KC_B::crit_00", "kc_id": "KC_B", "weight": -1, "applicability": "applicable", "rating": 0},
    ])
    errs = validate_criterion_response(r, expected_segment_id="d::seg_0000", expected_criteria=crits)
    assert any("must_be_null_when_not_applicable" in e for e in errs)


def test_illegal_rating_rejected():
    crits = candidate_criteria_for_packet(PACKET, CRITERIA)
    r = _resp([
        {"criterion_id": "KC_A::crit_00", "kc_id": "KC_A", "weight": 2, "applicability": "applicable", "rating": 0.75},
        {"criterion_id": "KC_B::crit_00", "kc_id": "KC_B", "weight": -1, "applicability": "applicable", "rating": 0},
    ])
    errs = validate_criterion_response(r, expected_segment_id="d::seg_0000", expected_criteria=crits)
    assert any("not_0_0.5_1" in e for e in errs)


def test_missing_and_invented_criteria_are_both_caught():
    crits = candidate_criteria_for_packet(PACKET, CRITERIA)
    r = _resp([
        {"criterion_id": "KC_INVENTED::crit_99", "kc_id": "KC_X", "weight": 1, "applicability": "applicable", "rating": 1},
    ])
    errs = validate_criterion_response(r, expected_segment_id="d::seg_0000", expected_criteria=crits)
    assert any("missing" in e for e in errs)
    assert any("unexpected" in e for e in errs)


def test_segment_id_mismatch_caught():
    crits = candidate_criteria_for_packet(PACKET, CRITERIA)
    errs = validate_criterion_response(_resp([], sid="other"), expected_segment_id="d::seg_0000",
                                        expected_criteria=crits)
    assert any("segment_id_mismatch" in e for e in errs)


# ---------------------------------------------------------------------------
# integrity: library truth beats the model's echo
# ---------------------------------------------------------------------------

def test_reweighted_criterion_is_caught():
    truth = library_criteria_by_id(CRITERIA)
    errs = check_criterion_integrity(criterion_id="KC_A::crit_00", reported_kc_id="KC_A",
                                      reported_weight=99, library_criteria_by_id=truth)
    assert any("weight mismatch" in e for e in errs)


def test_reattributed_criterion_is_caught():
    truth = library_criteria_by_id(CRITERIA)
    errs = check_criterion_integrity(criterion_id="KC_A::crit_00", reported_kc_id="KC_ELSEWHERE",
                                      reported_weight=2, library_criteria_by_id=truth)
    assert any("kc_id mismatch" in e for e in errs)


def test_invented_criterion_is_caught():
    truth = library_criteria_by_id(CRITERIA)
    errs = check_criterion_integrity(criterion_id="KC_MADE_UP::crit_00", reported_kc_id="KC_A",
                                      reported_weight=1, library_criteria_by_id=truth)
    assert any("not found in library truth" in e for e in errs)


# ---------------------------------------------------------------------------
# scoring semantics reaching K_s
# ---------------------------------------------------------------------------

def test_negative_polarity_is_inverted_so_absence_scores_well():
    """A misconception criterion rated 0 (did NOT occur) should give full quality."""
    js = [CriterionJudgment("c", "KC_B", -1, "applicable", 0)]
    assert score_kc_criteria(js).kc_s == 1.0
    js = [CriterionJudgment("c", "KC_B", -1, "applicable", 1)]
    assert score_kc_criteria(js).kc_s == 0.0


def test_not_applicable_excluded_never_zero():
    js = [
        CriterionJudgment("a", "KC_A", 2, "applicable", 1),
        CriterionJudgment("b", "KC_B", -1, "not_applicable", None),
    ]
    r = score_kc_criteria(js)
    assert r.kc_s == 1.0  # would be < 1 if the n/a were coerced to 0
    assert r.not_applicable_count == 1


def test_weight_magnitude_dominates_proportionally():
    js = [
        CriterionJudgment("a", "KC_A", 2, "applicable", 0),
        CriterionJudgment("b", "KC_A", 1, "applicable", 1),
    ]
    assert score_kc_criteria(js).kc_s == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# real library
# ---------------------------------------------------------------------------

@requires_lib
def test_real_library_yields_104_criteria_over_91_kcs():
    c = load_reviewer_criteria(LIB)
    assert len(c) == 91
    assert sum(len(v) for v in c.values()) == 104


@requires_lib
def test_real_criterion_ids_are_unique_and_stable():
    c = load_reviewer_criteria(LIB)
    ids = [x["criterion_id"] for v in c.values() for x in v]
    assert len(ids) == len(set(ids))
    assert load_reviewer_criteria(LIB) == c  # deterministic across calls


@requires_lib
def test_real_polarity_always_agrees_with_weight_sign():
    """The audit found this held across all 104; CriterionJudgment treats disagreement as an
    integrity error, so a violation would break scoring rather than pass silently."""
    for v in load_reviewer_criteria(LIB).values():
        for c in v:
            expected = "positive" if c["weight"] > 0 else "negative"
            assert c["polarity"] == expected, c["criterion_id"]
