"""Tests for src/seg_eval/aggregation/kc_criterion_scoring.py.

Two layers: synthetic unit tests covering every scoring-mechanics guarantee (fast, no I/O), and
one integration test against the real 104 draft criteria in the live reviewed library, which
regression-checks the audit's own finding (polarity always agrees with weight's sign) rather
than just asserting it once by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seg_eval.aggregation.kc_criterion_scoring import (
    CriterionJudgment,
    CriterionScoringError,
    check_criterion_integrity,
    criterion_quality,
    score_kc_criteria,
    stratify_criterion_outcomes,
)


def _j(criterion_id, weight, rating, applicability="applicable", kc_id="KC_TEST_001", polarity=None):
    return CriterionJudgment(
        criterion_id=criterion_id, kc_id=kc_id, weight=weight,
        applicability=applicability, rating=rating, polarity=polarity,
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_zero_weight_rejected():
    with pytest.raises(CriterionScoringError):
        _j("c1", weight=0, rating=1)


def test_not_applicable_requires_null_rating():
    with pytest.raises(CriterionScoringError):
        _j("c1", weight=1, rating=1, applicability="not_applicable")


def test_applicable_requires_non_null_rating():
    with pytest.raises(CriterionScoringError):
        _j("c1", weight=1, rating=None, applicability="applicable")


def test_applicable_rejects_out_of_scale_rating():
    with pytest.raises(CriterionScoringError):
        _j("c1", weight=1, rating=0.75, applicability="applicable")


def test_not_applicable_with_null_rating_is_valid():
    j = _j("c1", weight=2, rating=None, applicability="not_applicable")
    assert j.rating is None


def test_polarity_disagreeing_with_weight_sign_rejected():
    with pytest.raises(CriterionScoringError):
        _j("c1", weight=2, rating=1, polarity="negative")


def test_polarity_agreeing_with_weight_sign_accepted():
    j = _j("c1", weight=2, rating=1, polarity="positive")
    assert j.polarity == "positive"


# ---------------------------------------------------------------------------
# criterion_quality
# ---------------------------------------------------------------------------


def test_positive_criterion_scoring_direction():
    # weight > 0: presence of desired behavior (rating=1) is the best outcome.
    assert criterion_quality(rating=1, weight=2) == 1.0
    assert criterion_quality(rating=0, weight=2) == 0.0
    assert criterion_quality(rating=0.5, weight=2) == 0.5


def test_negative_criterion_scoring_direction():
    # weight < 0: presence of the misconception (rating=1) is the WORST outcome.
    assert criterion_quality(rating=1, weight=-2) == 0.0
    assert criterion_quality(rating=0, weight=-2) == 1.0
    assert criterion_quality(rating=0.5, weight=-2) == 0.5


def test_violating_negative_criterion_harms_score():
    triggered = score_kc_criteria([_j("neg", weight=-3, rating=1)]).kc_s
    not_triggered = score_kc_criteria([_j("neg", weight=-3, rating=0)]).kc_s
    assert triggered == 0.0
    assert not_triggered == 1.0
    assert not_triggered > triggered


def test_satisfying_positive_criterion_improves_score():
    satisfied = score_kc_criteria([_j("pos", weight=3, rating=1)]).kc_s
    unsatisfied = score_kc_criteria([_j("pos", weight=3, rating=0)]).kc_s
    assert satisfied == 1.0
    assert unsatisfied == 0.0
    assert satisfied > unsatisfied


# ---------------------------------------------------------------------------
# score_kc_criteria: weighting, magnitude ordering, not_applicable exclusion, scale invariance
# ---------------------------------------------------------------------------


def test_not_applicable_excluded_from_denominator():
    result = score_kc_criteria([
        _j("a", weight=2, rating=1),
        _j("b", weight=5, rating=None, applicability="not_applicable"),
    ])
    # if b's weight leaked into the denominator this would not be 1.0
    assert result.kc_s == 1.0
    assert result.applicable_count == 1
    assert result.not_applicable_count == 1


def test_unclear_applicability_is_scored_not_excluded():
    result = score_kc_criteria([_j("a", weight=2, rating=1, applicability="unclear")])
    assert result.kc_s == 1.0
    assert result.applicable_count == 1
    assert result.unclear_count == 1


def test_all_not_applicable_yields_none_not_zero():
    result = score_kc_criteria([
        _j("a", weight=1, rating=None, applicability="not_applicable"),
        _j("b", weight=2, rating=None, applicability="not_applicable"),
    ])
    assert result.kc_s is None
    assert result.applicable_count == 0


def test_weight_magnitude_ordering_matters():
    # a mild positive success (weight 1) should move the score less than a critical positive
    # success (weight 3) when mixed with an unrelated failure of the OTHER magnitude.
    mild_dominates = score_kc_criteria([
        _j("mild_pass", weight=1, rating=1),
        _j("critical_fail", weight=3, rating=0),
    ]).kc_s
    critical_dominates = score_kc_criteria([
        _j("critical_pass", weight=3, rating=1),
        _j("mild_fail", weight=1, rating=0),
    ]).kc_s
    # both cases have one pass and one fail; the case where the PASS carries the bigger weight
    # must score higher than the case where the FAIL carries the bigger weight.
    assert critical_dominates > mild_dominates
    assert mild_dominates == pytest.approx(1 / 4)
    assert critical_dominates == pytest.approx(3 / 4)


def test_more_mild_criteria_do_not_change_score_scale():
    # a single perfect weight-1 criterion and ten perfect weight-1 criteria must score the same.
    one = score_kc_criteria([_j("c0", weight=1, rating=1)]).kc_s
    ten = score_kc_criteria([_j(f"c{i}", weight=1, rating=1) for i in range(10)]).kc_s
    assert one == ten == 1.0


def test_mixed_polarity_weighted_mean():
    # one perfect positive (w=2) and one fully-triggered negative (w=-1):
    # numerator = 2*1 + 1*0 = 2, denominator = 2 + 1 = 3
    result = score_kc_criteria([
        _j("pos", weight=2, rating=1),
        _j("neg", weight=-1, rating=1),
    ])
    assert result.kc_s == pytest.approx(2 / 3)


def test_score_pools_applicable_criteria_across_multiple_matched_kcs():
    result = score_kc_criteria([
        _j("a", weight=1, rating=1, kc_id="KC_A"),
        _j("b", weight=3, rating=0, kc_id="KC_B"),
    ])

    assert result.kc_s == pytest.approx(1 / 4)
    assert result.applicable_count == 2
    assert result.total_weight_magnitude == 4.0


# ---------------------------------------------------------------------------
# severity stratification
# ---------------------------------------------------------------------------


def test_stratify_reports_only_observed_magnitudes():
    judgments = [_j("a", weight=1, rating=0), _j("b", weight=2, rating=1)]
    strat = stratify_criterion_outcomes(judgments)
    assert set(strat.keys()) == {1, 2}
    assert 3 not in strat  # never manufacture an unused tier


def test_stratify_full_vs_any_shortfall():
    judgments = [
        _j("full_fail_pos", weight=2, rating=0),   # positive, fully absent -> full shortfall
        _j("partial_pos", weight=2, rating=0.5),   # positive, partial -> any but not full
        _j("triggered_neg", weight=-2, rating=1),  # negative, fully triggered -> full shortfall
    ]
    strat = stratify_criterion_outcomes(judgments)
    assert strat[2]["positive"]["n"] == 2
    assert strat[2]["positive"]["any_shortfall"] == 2
    assert strat[2]["positive"]["full_shortfall"] == 1
    assert strat[2]["negative"]["n"] == 1
    assert strat[2]["negative"]["full_shortfall"] == 1


# ---------------------------------------------------------------------------
# integrity checking (model cannot invent/reweight a criterion)
# ---------------------------------------------------------------------------


def test_integrity_check_passes_for_matching_echo():
    library = {"c1": {"kc_id": "KC_A", "weight": 2}}
    errors = check_criterion_integrity(
        criterion_id="c1", reported_kc_id="KC_A", reported_weight=2,
        library_criteria_by_id=library,
    )
    assert errors == []


def test_integrity_check_catches_invented_criterion():
    errors = check_criterion_integrity(
        criterion_id="does_not_exist", reported_kc_id="KC_A", reported_weight=2,
        library_criteria_by_id={},
    )
    assert errors and "not found in library truth" in errors[0]


def test_integrity_check_catches_reweighted_criterion():
    library = {"c1": {"kc_id": "KC_A", "weight": 2}}
    errors = check_criterion_integrity(
        criterion_id="c1", reported_kc_id="KC_A", reported_weight=-2,  # model flipped the sign
        library_criteria_by_id=library,
    )
    assert any("weight mismatch" in e for e in errors)


def test_integrity_check_catches_kc_id_mismatch():
    library = {"c1": {"kc_id": "KC_A", "weight": 2}}
    errors = check_criterion_integrity(
        criterion_id="c1", reported_kc_id="KC_B", reported_weight=2,
        library_criteria_by_id=library,
    )
    assert any("kc_id mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# Integration: real draft criteria from the live reviewed library
# ---------------------------------------------------------------------------

_LIVE_LIBRARY = (
    Path(__file__).resolve().parents[1]
    / "data" / "input" / "frozen_library_incoming" / "kc_library_reviewed_v2"
    / "2026-08-10_155459" / "frozen_reviewed_library.jsonl"
)


@pytest.mark.skipif(not _LIVE_LIBRARY.exists(), reason="live reviewed library not present")
def test_live_draft_criteria_polarity_matches_weight_sign_and_are_scorable():
    rows = [json.loads(line) for line in _LIVE_LIBRARY.open(encoding="utf-8")]
    all_criteria = [c for r in rows for c in (r.get("reviewer_criteria") or [])]
    assert len(all_criteria) == 104, (
        "criterion count in the live library changed since the audit -- re-run "
        "05_KC_CRITERIA_WEIGHT_AUDIT.md's derivation, don't just bump this number"
    )

    judgments = []
    for c in all_criteria:
        # every live draft criterion must construct as a valid, fully-applicable judgment when
        # given a real rating -- this is what proves the schema this module assumes actually
        # matches production data, not a hand-picked example.
        j = CriterionJudgment(
            criterion_id=f"{c['kc_id']}::{hash(c['criterion_text']) & 0xffff}",
            kc_id=c["kc_id"], weight=c["weight"], applicability="applicable",
            rating=1, polarity=c["polarity"],
        )
        judgments.append(j)

    magnitudes = {abs(j.weight) for j in judgments}
    assert magnitudes == {1, 2}, (
        "observed magnitude range changed -- if |weight|=3 now exists, that's real news, "
        "update 05_KC_CRITERIA_WEIGHT_AUDIT.md rather than just this assertion"
    )

    result = score_kc_criteria(judgments)
    assert result.applicable_count == 104
    assert 0.0 <= result.kc_s <= 1.0
