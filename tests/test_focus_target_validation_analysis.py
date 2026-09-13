"""Tests for src/seg_eval/evaluation_judge/focus_target_validation_analysis.py."""

from __future__ import annotations

import pytest

from seg_eval.evaluation_judge.focus_target_validation_analysis import (
    FocusValidationRow,
    broad_fallback_appropriateness,
    cohens_kappa,
    focus_target_validation_row_from_worksheet_row,
    missed_material_claim_rate,
    precise_target_correctness,
    target_status_agreement,
)


def row(seg_id, system_type, human_type=None, precise_ok=None, broad_ok=None, missed=None):
    return FocusValidationRow(
        segment_id=seg_id, dialogue="dm1", system_target_type=system_type,
        human_target_type=human_type, precise_span_correct=precise_ok,
        broad_fallback_appropriate=broad_ok, material_claim_missed=missed,
    )


# ---------------------------------------------------------------------------
# target_status_agreement
# ---------------------------------------------------------------------------


def test_perfect_agreement():
    rows = [
        row("a", "precise_focus_target", human_type="precise_focus_target"),
        row("b", "broad_tutor_claim_target", human_type="broad_tutor_claim_target"),
    ]
    result = target_status_agreement(rows)
    assert result.n_rated == 2
    assert result.raw_agreement == 1.0
    assert result.cohens_kappa == pytest.approx(1.0)


def test_no_agreement():
    rows = [
        row("a", "precise_focus_target", human_type="broad_tutor_claim_target"),
        row("b", "broad_tutor_claim_target", human_type="precise_focus_target"),
    ]
    result = target_status_agreement(rows)
    assert result.raw_agreement == 0.0
    assert result.cohens_kappa == pytest.approx(-1.0)


def test_unrated_rows_excluded_from_agreement():
    rows = [
        row("a", "precise_focus_target", human_type="precise_focus_target"),
        row("b", "precise_focus_target", human_type=None),  # not yet rated
    ]
    result = target_status_agreement(rows)
    assert result.n_rated == 1


def test_no_rated_rows_returns_none_metrics():
    result = target_status_agreement([row("a", "precise_focus_target", human_type=None)])
    assert result.n_rated == 0
    assert result.raw_agreement is None
    assert result.cohens_kappa is None


def test_kappa_below_chance_agreement_is_negative():
    # systematic disagreement should score worse than chance (negative kappa), not just "not 1.0"
    pairs = [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")]
    k = cohens_kappa(pairs)
    assert k is not None
    assert k < 0


def test_kappa_none_for_single_pair():
    assert cohens_kappa([("A", "A")]) is None


# ---------------------------------------------------------------------------
# precise_target_correctness / broad_fallback_appropriateness
# ---------------------------------------------------------------------------


def test_precise_correctness_basic_rate():
    rows = [
        row("a", "precise_focus_target", precise_ok="Y"),
        row("b", "precise_focus_target", precise_ok="Y"),
        row("c", "precise_focus_target", precise_ok="N"),
        row("d", "broad_tutor_claim_target", precise_ok="Y"),  # wrong type, excluded
    ]
    result = precise_target_correctness(rows)
    assert result.n_applicable == 3
    assert result.n_correct == 2
    assert result.rate == pytest.approx(2 / 3)


def test_precise_correctness_na_and_unrated_excluded_from_rate_but_counted():
    rows = [
        row("a", "precise_focus_target", precise_ok="Y"),
        row("b", "precise_focus_target", precise_ok="NA"),
        row("c", "precise_focus_target", precise_ok=None),
    ]
    result = precise_target_correctness(rows)
    assert result.n_applicable == 1
    assert result.rate == 1.0
    assert result.n_not_applicable == 1
    assert result.n_unrated == 1


def test_precise_correctness_no_candidates_returns_none_rate():
    result = precise_target_correctness([row("a", "broad_tutor_claim_target")])
    assert result.n_applicable == 0
    assert result.rate is None


def test_broad_fallback_appropriateness_basic():
    rows = [
        row("a", "broad_tutor_claim_target", broad_ok="Y"),
        row("b", "broad_tutor_claim_target", broad_ok="N"),
    ]
    result = broad_fallback_appropriateness(rows)
    assert result.n_applicable == 2
    assert result.rate == 0.5


def test_broad_fallback_ignores_precise_rows():
    rows = [
        row("a", "precise_focus_target", broad_ok="Y"),  # wrong type field, should be ignored
    ]
    result = broad_fallback_appropriateness(rows)
    assert result.n_applicable == 0


# ---------------------------------------------------------------------------
# missed_material_claim_rate
# ---------------------------------------------------------------------------


def test_missed_claim_rate_across_all_target_types():
    rows = [
        row("a", "precise_focus_target", missed="Y"),
        row("b", "broad_tutor_claim_target", missed="N"),
        row("c", "precise_focus_target", missed="N"),
    ]
    result = missed_material_claim_rate(rows)
    assert result.n_rated == 3
    assert result.n_missed == 1
    assert result.rate == pytest.approx(1 / 3)


def test_missed_claim_rate_unrated_excluded():
    rows = [row("a", "precise_focus_target", missed=None)]
    result = missed_material_claim_rate(rows)
    assert result.n_rated == 0
    assert result.rate is None


# ---------------------------------------------------------------------------
# worksheet row parsing
# ---------------------------------------------------------------------------


def test_row_from_worksheet_values():
    values = {
        "Segment ID": "seg_0001", "Dialogue": "dm2",
        "System target_type": "precise_focus_target",
        "STEP1: Your target type": "precise_focus_target",
        "Is the selected precise span correct? (Y/N/NA)": "Y",
        "Is broad fallback appropriate? (Y/N/NA)": None,
        "Was a material claim missed? (Y/N)": "N",
    }
    r = focus_target_validation_row_from_worksheet_row(values)
    assert r.segment_id == "seg_0001"
    assert r.dialogue == "dm2"
    assert r.system_target_type == "precise_focus_target"
    assert r.human_target_type == "precise_focus_target"
    assert r.precise_span_correct == "Y"
    assert r.material_claim_missed == "N"


def test_row_from_worksheet_values_blank_step1_is_none():
    values = {"Segment ID": "seg_0001", "System target_type": "broad_tutor_claim_target",
              "STEP1: Your target type": None}
    r = focus_target_validation_row_from_worksheet_row(values)
    assert r.human_target_type is None
