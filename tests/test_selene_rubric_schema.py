"""Tests for the grammar-enforced Selene rubric wire format.

These assert that the two measured contract violations are structurally unrepresentable, and
that the wire->contract mapping neither invents nor alters a value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS  # noqa: E402
from seg_eval.evaluation_judge.selene_rubric_schema_v1 import (  # noqa: E402
    build_wire_schema, to_contract_shape, allowed_kc_ids_for_packet,
    WIRE_FORMAT_INSTRUCTION,
)

PACKET = {
    "segment_id": "d::seg_0000",
    "evaluation_target": {"primary_kc_id": "KC_A", "evaluator_kc_set": ["KC_A", "KC_B"]},
    "kc_context": [{"unit_id": "KC_C"}],
}


# ---------------------------------------------------------------------------
# structural impossibility of the two measured violations
# ---------------------------------------------------------------------------

def test_not_applicable_with_a_score_is_not_representable():
    """35 of Selene's 53 contract failures were a number scored on a not_applicable dimension.
    The oneOf makes that state ungeneratable rather than merely discouraged."""
    schema = build_wire_schema(["KC_A"])
    dim = schema["properties"]["dimensions"]["properties"]["error_detection"]
    branches = dim["oneOf"]
    assert len(branches) == 2

    na = next(b for b in branches if b["properties"]["applicability"].get("const") == "not_applicable")
    assert na["properties"]["raw"] == {"type": "null"}
    assert na["properties"]["adjusted"] == {"type": "null"}

    scored = next(b for b in branches if "enum" in b["properties"]["applicability"])
    assert scored["properties"]["applicability"]["enum"] == ["applicable", "unclear"]
    assert scored["properties"]["raw"]["enum"] == [0, 0.5, 1]
    assert scored["properties"]["adjusted"]["enum"] == [0, 0.5, 1]


def test_context_kcs_constrained_to_packet_identifiers():
    """68 failures were citing KC ids absent from the packet. An enum makes that ungeneratable."""
    schema = build_wire_schema(["KC_A", "KC_B"])
    items = schema["properties"]["kc_grounding"]["properties"]["context_kcs_used"]["items"]
    assert items["enum"] == ["KC_A", "KC_B"]


def test_no_enum_when_packet_has_no_kcs():
    """A packet with no KCs must not produce an empty enum, which no string could satisfy."""
    items = build_wire_schema([])["properties"]["kc_grounding"]["properties"]["context_kcs_used"]["items"]
    assert "enum" not in items


def test_allowed_ids_gathered_from_every_packet_source():
    assert allowed_kc_ids_for_packet(PACKET) == ["KC_A", "KC_B", "KC_C"]


def test_every_rubric_dimension_and_trust_flag_present():
    schema = build_wire_schema(["KC_A"])
    assert set(schema["properties"]["dimensions"]["properties"]) == set(PARTIAL_DIMENSIONS)
    assert set(schema["properties"]["trust_flags"]["properties"]) == set(TRUST_FLAGS)


def test_instruction_states_both_rules():
    assert "MUST be null" in WIRE_FORMAT_INSTRUCTION
    assert "only contain KC identifiers that appear in this" in WIRE_FORMAT_INSTRUCTION


# ---------------------------------------------------------------------------
# the mapping must be lossless, and must not invent
# ---------------------------------------------------------------------------

def _wire(**over):
    base = {
        "segment_id": "d::seg_0000",
        "evaluation_mode": "single_kc_primary",
        "dimensions": {d: {"applicability": "applicable", "raw": 1, "adjusted": 1}
                        for d in PARTIAL_DIMENSIONS},
        "trust_flags": {f: 0 for f in TRUST_FLAGS},
        "dependency_adjustments": [],
        "micro_score_raw": 1.0,
        "micro_score_dependency_adjusted": 1.0,
        "trust_adjusted_score": 1.0,
        "kc_grounding": {"primary_kc_used": True, "context_kcs_used": ["KC_A"],
                          "unsupported_or_wrong_kc_claims": []},
        "flags": {"major_correctness_error": False, "ungrounded_claim": False,
                   "missed_student_need": False, "overly_answer_giving": False,
                   "segment_boundary_problem": False, "needs_human_review": False},
        "evidence_quotes": ["a quote"],
        "rationale_short": "a rationale long enough to pass",
    }
    base.update(over)
    return base


def test_mapping_splits_paired_dimensions_into_contract_maps():
    dims = {d: {"applicability": "applicable", "raw": 0.5, "adjusted": 1} for d in PARTIAL_DIMENSIONS}
    dims["proactive_clarification"] = {"applicability": "not_applicable", "raw": None, "adjusted": None}
    c = to_contract_shape(_wire(dimensions=dims))
    assert c["raw_dimension_scores"]["error_detection"] == 0.5
    assert c["dimension_scores"]["error_detection"] == 1
    assert c["dimension_applicability"]["proactive_clarification"] == "not_applicable"
    assert c["raw_dimension_scores"]["proactive_clarification"] is None
    assert c["dimension_scores"]["proactive_clarification"] is None


def test_mapped_payload_satisfies_the_real_contract():
    from seg_eval.evaluation_judge.response_contract_v2 import validate_judge_response
    c = to_contract_shape(_wire())
    errs = validate_judge_response(c, expected_segment_id="d::seg_0000",
                                    expected_evaluation_mode="single_kc_primary")
    assert errs == [], errs


def test_the_previously_failing_combination_now_validates():
    """The exact shape that failed 35 times: not_applicable with null scores."""
    from seg_eval.evaluation_judge.response_contract_v2 import validate_judge_response
    dims = {d: {"applicability": "applicable", "raw": 1, "adjusted": 1} for d in PARTIAL_DIMENSIONS}
    dims["proactive_clarification"] = {"applicability": "not_applicable", "raw": None, "adjusted": None}
    c = to_contract_shape(_wire(dimensions=dims))
    errs = validate_judge_response(c, expected_segment_id="d::seg_0000",
                                    expected_evaluation_mode="single_kc_primary")
    assert not any("proactive_clarification" in e for e in errs), errs


def test_mapping_does_not_invent_missing_fields():
    """A field absent from the wire payload must stay absent, so the contract validator reports
    it rather than this mapping papering over it."""
    w = _wire()
    del w["rationale_short"]
    assert "rationale_short" not in to_contract_shape(w)


def test_mapping_preserves_values_exactly():
    w = _wire(micro_score_raw=0.4321, evidence_quotes=["x", "y"])
    c = to_contract_shape(w)
    assert c["micro_score_raw"] == 0.4321
    assert c["evidence_quotes"] == ["x", "y"]
    assert c["kc_grounding"]["context_kcs_used"] == ["KC_A"]
