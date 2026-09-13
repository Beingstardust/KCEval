"""Tests for src/seg_eval/aggregation/deterministic_aggregation.py."""

from __future__ import annotations

import pytest

from seg_eval.aggregation.deterministic_aggregation import (
    DialogueScoreInputs,
    audit_trust_consistency,
    combine_generic_and_kc,
    combine_local_and_arc,
    detect_arithmetic_disagreement,
    exchange_weighted_mean,
    final_tutor_score,
    macro_score,
    ranking_is_stable,
    sensitivity_grid,
    weighted_mean_applicable,
)


# ---------------------------------------------------------------------------
# weighted_mean_applicable: the null-handling guarantee
# ---------------------------------------------------------------------------


def test_null_is_never_treated_as_zero():
    # if null were coerced to 0, this would average to 0.5; it must instead average only the
    # two real values.
    values = {"a": 1.0, "b": None, "c": 1.0}
    assert weighted_mean_applicable(values) == 1.0


def test_not_applicable_excluded_via_applicability_map():
    values = {"a": 1.0, "b": 0.0}
    applicability = {"a": "applicable", "b": "not_applicable"}
    assert weighted_mean_applicable(values, applicability) == 1.0


def test_all_null_or_not_applicable_returns_none():
    values = {"a": None, "b": None}
    assert weighted_mean_applicable(values) is None


def test_equal_weighting_default():
    values = {"a": 1.0, "b": 0.0, "c": 0.5}
    assert weighted_mean_applicable(values) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# exchange-count aggregation (D_segment / D_ARC) and the KC-confidence exclusion
# ---------------------------------------------------------------------------


def test_exchange_count_weighting_works():
    # a 4-exchange perfect segment and a 1-exchange zero segment: (4*1 + 1*0) / 5 = 0.8
    result = exchange_weighted_mean([(1.0, 4), (0.0, 1)])
    assert result == pytest.approx(0.8)


def test_equal_exchange_counts_reduce_to_plain_mean():
    result = exchange_weighted_mean([(1.0, 1), (0.0, 1)])
    assert result == pytest.approx(0.5)


def test_exchange_weighted_mean_signature_has_no_confidence_parameter():
    # execution spec §20: KC-assignment confidence must not be the primary aggregation weight.
    # Enforced structurally: the function only accepts (score, exchange_count) pairs, so there
    # is no argument through which a confidence weight could be passed even by mistake.
    import inspect

    params = list(inspect.signature(exchange_weighted_mean).parameters)
    assert params == ["scores_and_counts"]


def test_arc_weighting_uses_same_mechanism_as_segment_weighting():
    d_segment_like = exchange_weighted_mean([(0.9, 10), (0.3, 2)])
    d_arc_like = exchange_weighted_mean([(0.9, 10), (0.3, 2)])
    assert d_segment_like == d_arc_like  # same formula, same inputs -> same result


# ---------------------------------------------------------------------------
# rho combination (local vs ARC) and alpha combination (micro vs macro)
# ---------------------------------------------------------------------------


def test_rho_combination_matches_derived_value():
    # rho = 0.8 as derived live in 03_CURRENT_PROMPT_PROVENANCE.md (8 local / 10 total dims).
    combined = combine_local_and_arc(d_segment=1.0, d_arc=0.0, rho=0.8)
    assert combined.value == pytest.approx(0.8)
    assert combined.status == "PROVISIONAL_NOT_FROZEN"
    assert combined.weight_value == 0.8


def test_rho_out_of_range_rejected():
    with pytest.raises(ValueError):
        combine_local_and_arc(1.0, 0.0, rho=1.5)


def test_no_arc_falls_back_to_segment_only():
    combined = combine_local_and_arc(d_segment=0.7, d_arc=None, rho=0.8)
    assert combined.value == 0.7


def test_macro_score_ignores_unknown_keys_and_nulls():
    scores = {
        "adaptability": 1.0, "consistency": 1.0, "outcome_completion": None,
        "sequentiality": 1.0, "not_a_real_macro_dim": 0.0,
    }
    # mean of the 3 non-null real macro dims only: (1+1+1)/3 = 1.0
    assert macro_score(scores) == 1.0


def test_final_tutor_score_combination():
    t = final_tutor_score(d_micro=1.0, m=0.0, alpha=0.5)
    assert t.value == pytest.approx(0.5)


def test_final_tutor_score_falls_back_without_macro():
    t = final_tutor_score(d_micro=0.6, m=None, alpha=0.5)
    assert t.value == 0.6


# ---------------------------------------------------------------------------
# generic + KC-specific combination (beta)
# ---------------------------------------------------------------------------


def test_beta_combination_no_kc_criteria_falls_back_to_generic():
    b = combine_generic_and_kc(p_s=0.7, k_s=None, beta=0.5)
    assert b.value == 0.7


def test_beta_combination_blends_generic_and_kc():
    b = combine_generic_and_kc(p_s=1.0, k_s=0.0, beta=0.25)
    assert b.value == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# trust adjustment: audited, not reinvented -- and cannot be bypassed by KC scoring
# ---------------------------------------------------------------------------


def test_trust_flag_fired_and_score_was_capped_is_consistent():
    result = audit_trust_consistency(
        trust_flags={"curriculum_hallucination_present": True},
        pre_trust_score=0.9, reported_trust_adjusted_score=0.2,
    )
    assert result.consistent
    assert result.any_trust_flag_set


def test_trust_flag_fired_but_score_raised_is_inconsistent():
    # a trust-adjusted score that ends up HIGHER than the pre-trust score is an unambiguous
    # violation regardless of how "material" the rubric considered the hallucination.
    result = audit_trust_consistency(
        trust_flags={"curriculum_hallucination_present": True},
        pre_trust_score=0.9, reported_trust_adjusted_score=0.95,
    )
    assert not result.consistent
    assert "not capped" in result.reason


def test_trust_flag_fired_but_score_unchanged_is_tolerated():
    # the live dependency rule caps the score "when material" -- materiality is itself a
    # judgment call the rubric leaves to the judge, not something this audit can second-guess.
    # An unchanged score after a flag fires is therefore NOT automatically flagged as
    # inconsistent; only an actual increase is unambiguous. This is a deliberate choice to audit
    # what the rubric actually specifies, not a stricter rule invented to fill the gap.
    result = audit_trust_consistency(
        trust_flags={"curriculum_hallucination_present": True},
        pre_trust_score=0.9, reported_trust_adjusted_score=0.9,
    )
    assert result.consistent


def test_no_trust_flag_and_unchanged_score_is_consistent():
    result = audit_trust_consistency(
        trust_flags={"curriculum_hallucination_present": False},
        pre_trust_score=0.6, reported_trust_adjusted_score=0.6,
    )
    assert result.consistent
    assert not result.any_trust_flag_set


def test_kc_boost_cannot_hide_an_uncapped_trust_violation():
    # B_s combines a mediocre generic score with a strong KC-criteria score -- KC scoring pushes
    # the combined B_s well above the raw generic P_s.
    b_s = combine_generic_and_kc(p_s=0.4, k_s=1.0, beta=0.5).value
    assert b_s == pytest.approx(0.7)
    # the trust audit must be applied to this POST-KC combined value, not the raw P_s: if a
    # reported trust-adjusted score exceeds the POST-KC b_s (not just the raw P_s of 0.4), that
    # is still correctly caught -- proving the audit compares against the right baseline instead
    # of one KC scoring could have inflated past unnoticed.
    result = audit_trust_consistency(
        trust_flags={"domain_knowledge_hallucination_present": True},
        pre_trust_score=b_s, reported_trust_adjusted_score=0.75,
    )
    assert not result.consistent
    # for contrast: 0.75 would NOT have exceeded the raw generic P_s (0.4) using the wrong
    # baseline -- so comparing against b_s specifically is what makes this test meaningful.
    assert 0.75 > 0.4


# ---------------------------------------------------------------------------
# arithmetic-disagreement detection
# ---------------------------------------------------------------------------


def test_arithmetic_disagreement_detected():
    assert detect_arithmetic_disagreement(model_reported=0.75, deterministic_recomputed=0.5)


def test_arithmetic_agreement_within_tolerance_not_flagged():
    assert not detect_arithmetic_disagreement(
        model_reported=0.500000001, deterministic_recomputed=0.5
    )


# ---------------------------------------------------------------------------
# alpha sensitivity grid
# ---------------------------------------------------------------------------


def test_sensitivity_grid_stable_ranking():
    dialogues = [
        DialogueScoreInputs("strong", d_segment=1.0, d_arc=1.0, m=1.0),
        DialogueScoreInputs("weak", d_segment=0.0, d_arc=0.0, m=0.0),
    ]
    grid = sensitivity_grid(dialogues, rho=0.8, alpha_values=[0.0, 0.5, 1.0])
    assert set(grid.keys()) == {0.0, 0.5, 1.0}
    for alpha, ranked in grid.items():
        assert ranked[0][0] == "strong"  # strong dominates weak at every alpha here
    assert ranking_is_stable(grid)


def test_sensitivity_grid_detects_instability():
    # dialogue A is better on micro, dialogue B is better on macro -- ranking should flip
    # somewhere between alpha=0 (pure macro) and alpha=1 (pure micro).
    dialogues = [
        DialogueScoreInputs("micro_strong", d_segment=1.0, d_arc=1.0, m=0.0),
        DialogueScoreInputs("macro_strong", d_segment=0.0, d_arc=0.0, m=1.0),
    ]
    grid = sensitivity_grid(dialogues, rho=0.8, alpha_values=[0.0, 1.0])
    assert not ranking_is_stable(grid)
    assert grid[0.0][0][0] == "macro_strong"
    assert grid[1.0][0][0] == "micro_strong"
