"""Tests for the split replacement of ``solution_control``.

The construct being defended: an escape hatch is allowed to exist, but it must be NAMED and
QUOTED, so that it can be audited and counted rather than applied silently. Those are the
properties asserted here.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.answer_revelation_v1 import (  # noqa: E402
    build_schema, derive_solution_control, validate_response, build_user_prompt,
    revelation_rate, ALLOWED_REVELATION, ALLOWED_WARRANT, WARRANT_NONE,
    REVELATION_NOT, REVELATION_PARTIAL, REVELATION_FULL,
)

TUTOR_TEXT = ("Tutor: Let us think about it. The result you want is forty two. "
              "Student: oh I see. Tutor: you had already worked out the first half yourself.")


def _r(**over):
    base = {"segment_id": "d::seg_0000", "revelation": REVELATION_NOT, "revealed_quote": None,
            "warrant": WARRANT_NONE, "warrant_evidence": None,
            "reasoning": "the tutor asked a question instead of answering"}
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# the derived score -- policy lives in code, not in a model's reading of a compound question
# ---------------------------------------------------------------------------

def test_not_revealed_scores_full():
    assert derive_solution_control(_r())[0] == 1.0


def test_full_revelation_without_warrant_scores_zero():
    s, why = derive_solution_control(_r(revelation=REVELATION_FULL,
                                        revealed_quote="The result you want is forty two."))
    assert s == 0.0
    assert "no warrant" in why


def test_partial_revelation_without_warrant_scores_half():
    s, _ = derive_solution_control(_r(revelation=REVELATION_PARTIAL,
                                      revealed_quote="The result you want is forty two."))
    assert s == 0.5


def test_a_named_warrant_restores_the_score():
    """Revealing genuinely can be right teaching -- but the reason must be named."""
    s, why = derive_solution_control(_r(
        revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
        warrant="student_stuck_after_multiple_attempts", warrant_evidence="you had already worked"))
    assert s == 1.0
    assert "warranted" in why


def test_every_allowed_warrant_is_scoreable():
    for w in ALLOWED_WARRANT:
        s, _ = derive_solution_control(_r(revelation=REVELATION_FULL, revealed_quote="x" * 10,
                                          warrant=w))
        assert s == (0.0 if w == WARRANT_NONE else 1.0)


def test_unusable_values_return_none_not_a_default_score():
    assert derive_solution_control(_r(revelation="nonsense"))[0] is None
    assert derive_solution_control(_r(revelation=REVELATION_FULL, warrant="nonsense"))[0] is None


# ---------------------------------------------------------------------------
# the quote requirement -- a claim you cannot quote is not a claim
# ---------------------------------------------------------------------------

def test_revelation_without_a_quote_is_rejected():
    errs = validate_response(_r(revelation=REVELATION_FULL, revealed_quote=None),
                             expected_segment_id="d::seg_0000")
    assert any("revealed_quote:required_when_revealed" in e for e in errs)


def test_quote_must_appear_in_the_segment():
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="something never said in the segment"),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT)
    assert any("not_found_in_segment_text" in e for e in errs)


def test_a_real_quote_passes():
    """A fully-formed unexcused finding: both the revelation and the no_warrant judgement quoted."""
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant_evidence="Student: oh I see."),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT)
    assert errs == [], errs


def test_claimed_warrant_must_also_be_evidenced():
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant="student_stuck_after_multiple_attempts", warrant_evidence=None),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT)
    assert any("warrant_evidence:required_when_warrant_claimed" in e for e in errs)


def test_not_revealed_may_not_smuggle_in_a_quote_or_warrant():
    errs = validate_response(_r(revealed_quote="The result you want is forty two."),
                             expected_segment_id="d::seg_0000")
    assert any("must_be_null_when_not_revealed" in e for e in errs)
    errs2 = validate_response(_r(warrant="segment_is_expository_by_design"),
                              expected_segment_id="d::seg_0000")
    assert any("must_be_no_warrant_when_not_revealed" in e for e in errs2)


def test_segment_id_mismatch_is_caught():
    errs = validate_response(_r(), expected_segment_id="other")
    assert any("segment_id_mismatch" in e for e in errs)


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------

def test_schema_branches_cover_exactly_the_legal_shapes():
    branches = build_schema()["oneOf"]
    assert len(branches) == 4
    for b in branches:
        assert b["additionalProperties"] is False
        assert set(b["required"]) >= {"revelation", "revealed_quote", "warrant",
                                       "warrant_evidence", "reasoning"}


def test_not_revealed_branch_cannot_carry_a_string_quote():
    """Selene emitted the STRING "null" on 7 of the first 15 units under a nullable-string field.
    The branch must type the field as null outright so that shape is ungeneratable."""
    na = next(b for b in build_schema()["oneOf"]
              if b["properties"]["revelation"].get("const") == REVELATION_NOT)
    assert na["properties"]["revealed_quote"] == {"type": "null"}
    assert na["properties"]["warrant_evidence"] == {"type": "null"}
    assert na["properties"]["warrant"]["const"] == WARRANT_NONE


def test_a_claimed_warrant_branch_requires_real_evidence_text():
    warranted = next(b for b in build_schema()["oneOf"]
                     if "enum" in b["properties"]["warrant"])
    assert WARRANT_NONE not in warranted["properties"]["warrant"]["enum"]
    assert warranted["properties"]["warrant_evidence"]["minLength"] >= 8
    assert warranted["properties"]["revealed_quote"]["minLength"] >= 8


def test_no_warrant_carries_the_same_evidence_burden_as_a_warrant():
    """no_warrant is the branch that COSTS the tutor points. Requiring evidence only for the
    excusing branch made no_warrant the path of least resistance: it was used on 58 of 71 units and
    every measured false flag rode on that default."""
    revealed_unexcused = next(
        b for b in build_schema()["oneOf"]
        if b["properties"]["warrant"].get("const") == WARRANT_NONE
        and b["properties"]["revealed_quote"] != {"type": "null"})
    assert revealed_unexcused["properties"]["warrant_evidence"]["minLength"] >= 8


def test_unevidenced_no_warrant_is_a_contract_violation():
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant=WARRANT_NONE, warrant_evidence=None),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT)
    assert any("required_to_support_no_warrant" in e for e in errs)


def test_prompt_states_the_symmetric_burden_and_normalises_expository_teaching():
    p = build_user_prompt(segment_id="s", segment_text="t")
    assert "including no_warrant" in p
    assert "finding against the tutor" in p
    assert "common and legitimate" in p


def test_every_revelation_and_warrant_value_remains_reachable():
    branches = build_schema()["oneOf"]
    revs, warrants = set(), set()
    for b in branches:
        r, w = b["properties"]["revelation"], b["properties"]["warrant"]
        revs |= {r["const"]} if "const" in r else set(r["enum"])
        warrants |= {w["const"]} if "const" in w else set(w["enum"])
    assert revs == set(ALLOWED_REVELATION)
    assert warrants == set(ALLOWED_WARRANT)


# ---------------------------------------------------------------------------
# the prompt must not reintroduce the escape hatches it exists to remove
# ---------------------------------------------------------------------------

def test_prompt_asks_the_observation_before_the_judgement():
    p = build_user_prompt(segment_id="s", segment_text="t")
    assert p.index("QUESTION 1") < p.index("QUESTION 2")
    assert "NOT about whether it was good teaching" in p


def test_prompt_does_not_contain_the_compound_escape_wording():
    """'unnecessary' and 'when guided reasoning is appropriate' are the two hatches that made the
    original dimension score below chance. Neither may reappear."""
    p = build_user_prompt(segment_id="s", segment_text="t").lower()
    assert "unnecessary" not in p
    assert "when guided reasoning is appropriate" not in p


def test_prompt_is_domain_generic():
    p = build_user_prompt(segment_id="s", segment_text="t").lower()
    for term in ("naive bayes", "clustering", "math", "algebra", "data mining"):
        assert term not in p


# ---------------------------------------------------------------------------
# the MRBench-comparable statistic
# ---------------------------------------------------------------------------

def test_revelation_rate_matches_mrbench_shape():
    rows = [_r(), _r(revelation=REVELATION_FULL, revealed_quote="x" * 10),
            _r(revelation=REVELATION_FULL, revealed_quote="x" * 10,
               warrant="segment_is_expository_by_design", warrant_evidence="y" * 10),
            _r(revelation=REVELATION_PARTIAL, revealed_quote="x" * 10)]
    st = revelation_rate(rows)
    assert st["n"] == 4
    assert st["revealed_rate"] == 0.75
    assert st["unwarranted_revelation_rate"] == 0.5   # the warranted one does not count


STUDENT_TEXT = "Student: oh I see. I think the first half is forty."


def test_prompt_demands_a_verbatim_student_quote():
    assert "VERBATIM quote of something the STUDENT said" in build_user_prompt(
        segment_id="s", segment_text="t")


def test_warrant_evidence_must_quote_the_student_not_the_tutor():
    """9 of 12 expository warrants failed by DESCRIBING the segment instead of quoting it. Every
    warrant is a claim about the student, so the student's own words are what can settle it."""
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant="student_stuck_after_multiple_attempts",
           warrant_evidence="Let us think about it."),   # the TUTOR said this
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT,
        allowed_student_text=STUDENT_TEXT)
    assert any("not_found_in_student_text" in e for e in errs)


def test_a_real_student_quote_passes():
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant_evidence="I think the first half is forty."),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT,
        allowed_student_text=STUDENT_TEXT)
    assert errs == [], errs


def test_a_segment_with_no_student_turn_is_structurally_expository():
    """Verified from the packet, not taken on the model's word, so it cannot be a free escape."""
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant="segment_is_expository_by_design", warrant_evidence=None),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT,
        allowed_student_text="")
    assert errs == [], errs


def test_no_student_turn_does_not_excuse_a_different_warrant():
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant="student_asked_directly_after_genuine_effort", warrant_evidence=None),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT,
        allowed_student_text="")
    assert any("warrant_evidence" in e for e in errs)


def test_student_text_of_collects_only_student_turns():
    from seg_eval.evaluation_judge.answer_revelation_v1 import student_text_of
    pkt = {"member_exchanges": [{"student_text": "hello", "tutor_text": "TUTOR SAID THIS"},
                                {"student_text": "  ", "tutor_text": "x"},
                                {"student_text": "world", "tutor_text": "y"}]}
    out = student_text_of(pkt)
    assert "hello" in out and "world" in out
    assert "TUTOR SAID THIS" not in out


def test_grammar_permits_expository_with_null_evidence():
    """Without this branch the model cannot express "there is no student turn to quote" and writes
    a placeholder string to satisfy minLength -- it emitted
    "null - no student turn is applicable" verbatim. The grammar must permit the shape; the
    validator decides whether the claim was true."""
    b = [x for x in build_schema()["oneOf"]
         if x["properties"]["warrant"].get("const") == "segment_is_expository_by_design"]
    assert len(b) == 1
    assert b[0]["properties"]["warrant_evidence"] == {"type": "null"}
    assert b[0]["properties"]["revealed_quote"]["minLength"] >= 8


def test_null_expository_is_rejected_when_student_turns_exist():
    """Permitted by the grammar, checked against the packet -- so it cannot become a free escape."""
    errs = validate_response(
        _r(revelation=REVELATION_FULL, revealed_quote="The result you want is forty two.",
           warrant="segment_is_expository_by_design", warrant_evidence=None),
        expected_segment_id="d::seg_0000", allowed_tutor_text=TUTOR_TEXT,
        allowed_student_text=STUDENT_TEXT)
    assert any("student_turns_exist_so_expository_needs_a_quote" in e for e in errs)
