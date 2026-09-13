"""Tests for the grounded factuality pass and the deterministic trust cap.

The properties asserted here are the ones the design rests on (see 27_..._CITATIONS.md):
absence-is-not-error, verbatim quoting, verdict/contradiction consistency, and a cap that can
only ever lower a score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.grounded_factuality_v1 import (  # noqa: E402
    ALLOWED_VERDICTS, REVIEWED_LIBRARY, load_reference_by_kc, reference_for_packet,
    validate_factuality_response, material_contradiction_count, user_prompt, system_prompt,
    tutor_lines, segment_lines,
)
from seg_eval.aggregation.deterministic_aggregation import (  # noqa: E402
    apply_deterministic_trust_cap, TRUST_CAP_SINGLE_MATERIAL, TRUST_CAP_REPEATED_MATERIAL,
    TRUST_CAP_MINOR_DEVIATION,
)

LIB = REPO_ROOT / REVIEWED_LIBRARY
requires_lib = pytest.mark.skipif(not LIB.exists(), reason="reviewed library not present")

PACKET = {
    "segment_id": "d::seg_0000",
    "evaluation_target": {"primary_kc_id": "KC_A", "evaluator_kc_set": ["KC_A", "KC_B", "KC_C", "KC_D"]},
    "member_exchanges": [
        {"exchange_id": "ex_1", "student_text": "is entropy 1 for a pure node?",
         "tutor_text": "A pure node has entropy 1."},
    ],
}
REF = {
    "KC_A": {"what_tutor_should_explain": ["a value of 0 represents a perfectly pure interval"], "scope_status": "grounded"},
    "KC_B": {"red_flags": ["claiming high entropy means high quality"], "scope_status": "partial"},
    "KC_C": {"what_tutor_should_explain": ["something else"], "scope_status": "partial"},
    "KC_D": {"what_tutor_should_explain": ["yet another thing"], "scope_status": "partial"},
}


# ---------------------------------------------------------------------------
# reference assembly
# ---------------------------------------------------------------------------

def test_reference_is_capped_to_keep_the_prompt_narrow():
    """The pass's whole advantage is a small prompt; an unbounded evaluator_kc_set would
    reintroduce the bloat that broke the KC-grounding ablation."""
    got = reference_for_packet(PACKET, REF, max_kcs=3)
    assert len(got) == 3
    assert got[0]["kc_id"] == "KC_A"  # primary KC always first


def test_reference_empty_when_no_kc_has_content():
    assert reference_for_packet(PACKET, {}) == []


def test_tutor_lines_exclude_student_text():
    lines = tutor_lines(PACKET)
    assert len(lines) == 1
    assert "Tutor:" in lines[0] and "Student:" not in lines[0]


def test_segment_lines_include_both_roles():
    assert any("Student:" in l for l in segment_lines(PACKET))
    assert any("Tutor:" in l for l in segment_lines(PACKET))


# ---------------------------------------------------------------------------
# prompt content — the guardrails that keep this measuring correctness
# ---------------------------------------------------------------------------

def test_prompt_does_not_show_a_candidate_mistake_list():
    """r3's central defect: showing candidate mistakes turned the task into list-matching, and
    recall fell to 0.000 at BOTH 8B and 14B. r4 supplies the reference as curriculum content."""
    p = user_prompt(PACKET, reference_for_packet(PACKET, REF))
    assert "P1" not in p
    assert "candidate" not in p.lower()


def test_prompt_requires_a_stateable_correction():
    """The mechanism that forces reasoning about the fact rather than pattern-matching a flag."""
    p = user_prompt(PACKET, reference_for_packet(PACKET, REF))
    assert "If you cannot state the correction, it is not an error" in p
    assert "correct_version" in p


def test_prompt_says_most_segments_have_no_error():
    """Counterweight to the r1/r2 pull, where an error-hunting frame produced false positives."""
    p = user_prompt(PACKET, reference_for_packet(PACKET, REF))
    assert "Most segments contain no error at all" in p


def test_prompt_excludes_incompleteness_and_teaching_quality():
    p = user_prompt(PACKET, reference_for_packet(PACKET, REF))
    assert "Being brief, or not explaining something fully, is NOT an error" in p
    assert "not judging teaching quality" in p


def test_prompt_scopes_judgment_to_tutor_not_student():
    p = user_prompt(PACKET, reference_for_packet(PACKET, REF))
    assert "The student's own statements are never a tutor error" in p


def test_prompt_counts_endorsing_a_student_error():
    p = user_prompt(PACKET, reference_for_packet(PACKET, REF))
    assert "endorsing a student's incorrect claim IS a tutor error" in p


# ---------------------------------------------------------------------------
# r4 response contract + code-derived verdict
# ---------------------------------------------------------------------------

def _err(quote="A pure node has entropy 1.", wrong="inverted", correction="entropy 0 is pure",
         severity="material"):
    return {"tutor_quote": quote, "what_is_wrong": wrong,
            "correct_version": correction, "severity": severity}


def _er(errors, sid="d::seg_0000"):
    return {"segment_id": sid, "errors": errors,
            "rationale_short": "assessed against the curriculum reference."}


def test_valid_error_response():
    from seg_eval.evaluation_judge.grounded_factuality_v1 import validate_error_response
    assert validate_error_response(_er([_err()]), expected_segment_id="d::seg_0000",
                                    allowed_tutor_text=TUTOR_TEXT) == []


def test_flag_without_a_correction_is_rejected():
    """A flag whose correction cannot be stated is exactly the unreasoned pattern-match that
    produced r1/r2's false positives."""
    from seg_eval.evaluation_judge.grounded_factuality_v1 import validate_error_response
    e = _err(); e.pop("correct_version")
    errs = validate_error_response(_er([e]), expected_segment_id="d::seg_0000",
                                    allowed_tutor_text=TUTOR_TEXT)
    assert any("correct_version:required" in x for x in errs)


def test_fabricated_quote_is_rejected():
    from seg_eval.evaluation_judge.grounded_factuality_v1 import validate_error_response
    errs = validate_error_response(_er([_err(quote="words the tutor never said")]),
                                    expected_segment_id="d::seg_0000",
                                    allowed_tutor_text=TUTOR_TEXT)
    assert any("not_found_in_tutor_text" in x for x in errs)


def test_verdict_is_derived_in_code_not_self_reported():
    from seg_eval.evaluation_judge.grounded_factuality_v1 import derive_verdict_from_errors
    assert derive_verdict_from_errors(_er([])) == ("grounded", 0)
    assert derive_verdict_from_errors(_er([_err()])) == ("contradicted", 1)
    assert derive_verdict_from_errors(_er([_err(severity="minor")])) == ("minor_deviation", 0)
    assert derive_verdict_from_errors(_er([_err(), _err()])) == ("contradicted", 2)


# ---------------------------------------------------------------------------
# response contract
# ---------------------------------------------------------------------------

TUTOR_TEXT = "A pure node has entropy 1."


def _r(verdict, contradictions, sid="d::seg_0000", rationale="Because the reference says otherwise."):
    return {"segment_id": sid, "verdict": verdict, "contradictions": contradictions,
            "rationale_short": rationale}


def test_valid_grounded_response():
    assert validate_factuality_response(_r("grounded", []), expected_segment_id="d::seg_0000",
                                         allowed_tutor_text=TUTOR_TEXT) == []


def test_valid_contradicted_response():
    r = _r("contradicted", [{"tutor_quote": "A pure node has entropy 1.",
                              "conflicts_with": "reference says 0 is pure", "severity": "material"}])
    assert validate_factuality_response(r, expected_segment_id="d::seg_0000",
                                         allowed_tutor_text=TUTOR_TEXT) == []


def test_contradicted_without_contradictions_rejected():
    errs = validate_factuality_response(_r("contradicted", []), expected_segment_id="d::seg_0000")
    assert any("no_contradictions_listed" in e for e in errs)


def test_grounded_with_contradictions_rejected():
    r = _r("grounded", [{"tutor_quote": "A pure node has entropy 1.",
                          "conflicts_with": "x", "severity": "material"}])
    errs = validate_factuality_response(r, expected_segment_id="d::seg_0000")
    assert any("verdict_grounded_but_contradictions_listed" in e for e in errs)


def test_fabricated_quote_is_rejected():
    """A contradiction must be asserted against words the tutor actually said."""
    r = _r("contradicted", [{"tutor_quote": "the tutor never said this at all",
                              "conflicts_with": "x", "severity": "material"}])
    errs = validate_factuality_response(r, expected_segment_id="d::seg_0000",
                                         allowed_tutor_text=TUTOR_TEXT)
    assert any("not_found_in_tutor_text" in e for e in errs)


def test_invalid_verdict_rejected():
    errs = validate_factuality_response(_r("made_up", []), expected_segment_id="d::seg_0000")
    assert any("verdict:invalid" in e for e in errs)


def test_invalid_severity_rejected():
    r = _r("contradicted", [{"tutor_quote": "A pure node has entropy 1.",
                              "conflicts_with": "x", "severity": "catastrophic"}])
    errs = validate_factuality_response(r, expected_segment_id="d::seg_0000",
                                         allowed_tutor_text=TUTOR_TEXT)
    assert any("severity:invalid" in e for e in errs)


def test_segment_id_mismatch_rejected():
    errs = validate_factuality_response(_r("grounded", [], sid="other"),
                                         expected_segment_id="d::seg_0000")
    assert any("segment_id_mismatch" in e for e in errs)


def test_material_contradiction_count():
    r = _r("contradicted", [
        {"tutor_quote": "a", "conflicts_with": "x", "severity": "material"},
        {"tutor_quote": "b", "conflicts_with": "y", "severity": "minor"},
        {"tutor_quote": "c", "conflicts_with": "z", "severity": "material"},
    ])
    assert material_contradiction_count(r) == 2


# ---------------------------------------------------------------------------
# deterministic trust cap
# ---------------------------------------------------------------------------

def test_grounded_verdict_applies_no_cap():
    res = apply_deterministic_trust_cap(0.95, "grounded", 0)
    assert res.capped_score == 0.95 and not res.cap_applied


def test_insufficient_reference_applies_no_cap():
    """Thin curriculum coverage is not tutor failure."""
    res = apply_deterministic_trust_cap(0.95, "insufficient_reference", 0)
    assert res.capped_score == 0.95 and not res.cap_applied


def test_single_material_contradiction_caps():
    res = apply_deterministic_trust_cap(0.95, "contradicted", 1)
    assert res.capped_score == TRUST_CAP_SINGLE_MATERIAL
    assert res.cap_applied


def test_repeated_material_contradictions_cap_harder():
    res = apply_deterministic_trust_cap(0.95, "contradicted", 3)
    assert res.capped_score == TRUST_CAP_REPEATED_MATERIAL


def test_minor_deviation_caps_lightly():
    res = apply_deterministic_trust_cap(0.95, "minor_deviation", 0)
    assert res.capped_score == TRUST_CAP_MINOR_DEVIATION


def test_cap_never_raises_a_score():
    """A cap is a ceiling. A segment already below it must be untouched."""
    res = apply_deterministic_trust_cap(0.20, "contradicted", 1)
    assert res.capped_score == 0.20
    assert not res.cap_applied


def test_no_verdict_leaves_score_unchanged():
    res = apply_deterministic_trust_cap(0.88, None, 0)
    assert res.capped_score == 0.88 and not res.cap_applied
    assert "no factuality verdict" in res.reason


def test_contradicted_with_only_minor_contradictions_caps_lightly():
    res = apply_deterministic_trust_cap(0.95, "contradicted", 0)
    assert res.capped_score == TRUST_CAP_MINOR_DEVIATION


# ---------------------------------------------------------------------------
# real library
# ---------------------------------------------------------------------------

@requires_lib
def test_real_reference_covers_all_159_kcs():
    ref = load_reference_by_kc(LIB)
    assert len(ref) == 159


@requires_lib
def test_real_reference_carries_only_curriculum_content_fields():
    """Review status, provenance and matching cues have no place in a factuality reference."""
    allowed = {"what_tutor_should_explain", "common_confusions_or_errors", "red_flags",
                "scope_status", "kc_id"}
    for block in load_reference_by_kc(LIB).values():
        assert set(block) <= allowed


BUILT = sorted((REPO_ROOT / "data/processed/judge_prompts").glob("v35_tv_*_factuality_20260829"))


@pytest.mark.skipif(not BUILT, reason="factuality prompts not built")
def test_built_prompts_stay_narrow():
    """The separate-pass design only works if the prompt stays far below the main judge's ~21k
    chars, which is what made the reference unaffordable there."""
    for d in BUILT:
        rows = [json.loads(l) for l in (d / "grounded_factuality_prompts.jsonl").open(encoding="utf-8")]
        assert rows
        mean = sum(len(r["user_prompt"]) for r in rows) / len(rows)
        assert mean < 9000, f"{d.name}: mean prompt {mean:.0f} chars is approaching main-judge size"
