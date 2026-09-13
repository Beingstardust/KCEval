"""Tests for exact score attribution back to segments, exchange units and dimensions.

The central property is that attribution is *exact*: because the aggregation chain is linear,
per-dimension contributions must sum back to the score they explain. If that ever drifts, a
developer is sent to the wrong place in the dialogue with no way to notice, so it is asserted
directly rather than eyeballed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.aggregation.score_provenance_v1 import (  # noqa: E402
    build_score_provenance, rank_improvement_targets, dimension_rollup, verify_reconstruction,
)


def _packet(sid, n_ex, unit="kc_segment", kc="KC_X"):
    return {
        "segment_id": sid,
        "evaluation_unit_type": unit,
        "member_exchange_ids": [f"{sid}::ex_{i:04d}" for i in range(n_ex)],
        "turn_start": 1, "turn_end": n_ex * 2,
        "evaluation_target": {"primary_kc_id": kc, "primary_kc_name": "a name"},
    }


def _judge(scores, applic=None, quotes=("q1",), rationale="because"):
    return {
        "dimension_scores": scores,
        "dimension_applicability": applic or {d: "applicable" for d in scores},
        "evidence_quotes": list(quotes),
        "rationale_short": rationale,
    }


def _units(*specs):
    return [{"packet": p, "judge_output": j} for p, j in specs]


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------

def test_contributions_sum_back_to_the_score_local_only():
    units = _units(
        (_packet("s1", 3), _judge({"a": 1.0, "b": 0.5})),
        (_packet("s2", 1), _judge({"a": 0.0, "b": 1.0})),
    )
    prov = build_score_provenance(tutor_id="t", units=units, alpha=0.8, beta=0.0, rho=0.8)
    ok, _ = verify_reconstruction(prov)
    assert ok
    # D_segment = (0.75*3 + 0.5*1)/4 = 0.6875, and with no arc and no macro T = that
    assert prov.d_segment == pytest.approx(0.6875)
    assert prov.final_score == pytest.approx(0.6875)


def test_contributions_sum_back_with_both_families_and_macro():
    units = _units(
        (_packet("s1", 3), _judge({"a": 1.0, "b": 0.5})),
        (_packet("s2", 2), _judge({"a": 0.5, "b": 0.5})),
        (_packet("t1", 4, unit="topic_unit"), _judge({"c": 0.5, "d": 1.0})),
    )
    prov = build_score_provenance(tutor_id="t", units=units, alpha=0.8, beta=0.0, rho=0.8,
                                  macro=0.9)
    ok, _ = verify_reconstruction(prov)
    assert ok
    assert prov.d_segment is not None and prov.d_arc is not None
    expected_micro = 0.8 * prov.d_segment + 0.2 * prov.d_arc
    assert prov.d_micro == pytest.approx(expected_micro)
    assert prov.final_score == pytest.approx(0.8 * expected_micro + 0.2 * 0.9)


def test_per_dimension_coefficients_sum_to_the_segment_coefficient():
    units = _units((_packet("s1", 2), _judge({"a": 1.0, "b": 0.0, "c": 0.5})))
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    seg = prov.segments[0]
    assert sum(d.coefficient for d in seg.dimensions) == pytest.approx(seg.coefficient)


def test_exchange_count_drives_segment_weight_not_segment_count():
    """A segment covering more exchanges must weigh more -- execution spec's aggregation rule."""
    units = _units(
        (_packet("big", 9), _judge({"a": 0.0})),
        (_packet("small", 1), _judge({"a": 1.0})),
    )
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    assert prov.d_segment == pytest.approx(0.1)
    big = next(s for s in prov.segments if s.segment_id == "big")
    assert big.segment_weight == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# null handling -- the standing "never treat null as 0" rule
# ---------------------------------------------------------------------------

def test_not_applicable_dimension_is_excluded_not_zeroed():
    both = _units((_packet("s1", 1), _judge({"a": 1.0, "b": 1.0})))
    with_na = _units((_packet("s1", 1), _judge(
        {"a": 1.0, "b": None}, {"a": "applicable", "b": "not_applicable"})))
    p1 = build_score_provenance(tutor_id="t", units=both, alpha=1.0, beta=0.0, rho=1.0)
    p2 = build_score_provenance(tutor_id="t", units=with_na, alpha=1.0, beta=0.0, rho=1.0)
    assert p1.final_score == pytest.approx(1.0)
    assert p2.final_score == pytest.approx(1.0)   # not 0.5
    assert [d.dimension for d in p2.segments[0].dimensions] == ["a"]


def test_unit_with_no_applicable_dimensions_does_not_crash_or_score_zero():
    units = _units((_packet("s1", 1), _judge({"a": None}, {"a": "not_applicable"})))
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    assert prov.segments[0].p_s is None
    assert prov.segments[0].contribution == 0.0


def test_no_scorable_units_yields_undefined_score_with_a_note():
    prov = build_score_provenance(tutor_id="t", units=[], alpha=1.0, beta=0.0, rho=1.0)
    assert prov.final_score is None
    assert any("no scorable units" in n for n in prov.notes)


# ---------------------------------------------------------------------------
# traceability -- the developer-facing contract
# ---------------------------------------------------------------------------

def test_every_target_names_exchange_units_so_it_is_locatable():
    units = _units(
        (_packet("s1", 3), _judge({"a": 0.0, "b": 1.0})),
        (_packet("s2", 2), _judge({"a": 0.5, "b": 0.5})),
    )
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    targets = rank_improvement_targets(prov)
    assert targets
    for t in targets:
        assert t.member_exchange_ids, "a target with no exchange ids is not actionable"
        assert t.segment_id
        assert 0.0 <= t.current_score < 1.0


def test_targets_are_ranked_by_recoverable_score():
    units = _units(
        (_packet("big", 8), _judge({"a": 0.0})),
        (_packet("small", 1), _judge({"a": 0.0})),
    )
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    targets = rank_improvement_targets(prov)
    # the same failure in a longer segment is worth more, and must rank first
    assert targets[0].segment_id == "big"
    assert targets[0].recoverable_score > targets[1].recoverable_score


def test_a_perfect_tutor_has_no_improvement_targets():
    units = _units((_packet("s1", 2), _judge({"a": 1.0, "b": 1.0})))
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    assert rank_improvement_targets(prov) == []


def test_recoverable_score_is_what_fixing_it_actually_regains():
    """The headline promise: 'fix this and you gain X' must be literally true."""
    units = _units(
        (_packet("s1", 2), _judge({"a": 0.0, "b": 1.0})),
        (_packet("s2", 2), _judge({"a": 1.0, "b": 1.0})),
    )
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    target = rank_improvement_targets(prov)[0]

    fixed = _units(
        (_packet("s1", 2), _judge({"a": 1.0, "b": 1.0})),
        (_packet("s2", 2), _judge({"a": 1.0, "b": 1.0})),
    )
    after = build_score_provenance(tutor_id="t", units=fixed, alpha=1.0, beta=0.0, rho=1.0)
    assert after.final_score - prov.final_score == pytest.approx(target.recoverable_score)


def test_dimension_rollup_aggregates_across_segments():
    units = _units(
        (_packet("s1", 1), _judge({"a": 0.0, "b": 1.0})),
        (_packet("s2", 1), _judge({"a": 0.0, "b": 1.0})),
    )
    roll = dimension_rollup(build_score_provenance(
        tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0))
    assert roll[0]["dimension"] == "a"
    assert sorted(roll[0]["segments_below_1"]) == ["s1", "s2"]
    assert roll[1]["recoverable"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# trust cap -- a cap must actually reduce the score and be visible in the trace
# ---------------------------------------------------------------------------

def test_trust_cap_lowers_the_segment_score_and_is_recorded():
    units = [{"packet": _packet("s1", 1), "judge_output": _judge({"a": 1.0, "b": 1.0}),
              "trust_cap": {"capped_score": 0.5, "cap": 0.5,
                            "reason": "one material contradiction"}}]
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    assert prov.segments[0].b_s == pytest.approx(0.5)
    assert prov.segments[0].trust_cap_applied == 0.5
    assert "material contradiction" in prov.segments[0].trust_cap_reason
    assert prov.final_score == pytest.approx(0.5)


def test_trust_cap_never_raises_a_score():
    units = [{"packet": _packet("s1", 1), "judge_output": _judge({"a": 0.0}),
              "trust_cap": {"capped_score": 0.5, "cap": 0.5, "reason": "cap"}}]
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    assert prov.final_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# beta routing
# ---------------------------------------------------------------------------

def test_beta_splits_the_segment_between_generic_and_kc_criteria():
    units = [{"packet": _packet("s1", 1), "judge_output": _judge({"a": 1.0}), "k_s": 0.0}]
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.25, rho=1.0)
    assert prov.segments[0].b_s == pytest.approx(0.75)
    # only the generic share of the segment coefficient reaches the rubric dimensions
    assert prov.segments[0].dimensions[0].coefficient == pytest.approx(0.75)


def test_absent_kc_criteria_leave_the_generic_score_untouched():
    units = _units((_packet("s1", 1), _judge({"a": 1.0})))
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.9, rho=1.0)
    assert prov.segments[0].b_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# genericity
# ---------------------------------------------------------------------------

def test_module_carries_no_domain_specific_vocabulary():
    src = (REPO_ROOT / "src/seg_eval/aggregation/score_provenance_v1.py").read_text(encoding="utf-8")
    for term in ("naive bayes", "clustering", "classification", "laplace", "data mining"):
        assert term not in src.lower(), f"domain term {term!r} leaked into a generic module"


def test_arbitrary_dimension_names_work():
    """Dimension names arrive as data; nothing may be hardcoded to this project's rubric."""
    units = _units((_packet("s1", 1), _judge({"totally_made_up": 0.5, "another": 1.0})))
    prov = build_score_provenance(tutor_id="t", units=units, alpha=1.0, beta=0.0, rho=1.0)
    assert {d.dimension for d in prov.segments[0].dimensions} == {"totally_made_up", "another"}
    assert prov.final_score == pytest.approx(0.75)
