"""Tests for cross-family judge agreement as a confidence and triage signal."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.judge_panel_v1 import (  # noqa: E402
    compare_judges, triage_summary, confidence_index, panel_disagreement_by_dimension,
    CONFIDENCE_HIGH, CONFIDENCE_LOW,
)

DIMS = ["a", "b"]


def _jo(**scores):
    return {"dimension_scores": dict(scores)}


def test_agreement_and_disagreement_are_labelled():
    p = {"s1": _jo(a=1.0, b=0.5)}
    s = {"s1": _jo(a=1.0, b=1.0)}
    d = {x.dimension: x for x in compare_judges(primary=p, secondary=s, dimensions=DIMS)}
    assert d["a"].agreed and d["a"].confidence == CONFIDENCE_HIGH
    assert not d["b"].agreed and d["b"].confidence == CONFIDENCE_LOW


def test_null_is_its_own_category_not_a_wildcard():
    """A null and a 0.0 are different answers; treating null as matching anything would inflate
    agreement exactly where the judges are most likely to be confused."""
    p = {"s1": _jo(a=None)}
    s = {"s1": _jo(a=0.0)}
    assert compare_judges(primary=p, secondary=s, dimensions=["a"])[0].agreed is False
    both_null = compare_judges(primary={"s1": _jo(a=None)}, secondary={"s1": _jo(a=None)},
                                dimensions=["a"])
    assert both_null[0].agreed is True


def test_only_segments_present_in_both_are_compared():
    p = {"s1": _jo(a=1.0), "s2": _jo(a=1.0)}
    s = {"s1": _jo(a=1.0)}
    out = compare_judges(primary=p, secondary=s, dimensions=["a"])
    assert [x.segment_id for x in out] == ["s1"]


def test_out_of_family_dimensions_are_skipped_not_counted_as_agreement():
    """Two judges both leaving an inapplicable dimension null is not evidence of agreement."""
    p = {"s1": _jo(a=1.0, b=None)}
    s = {"s1": _jo(a=1.0, b=None)}
    out = compare_judges(primary=p, secondary=s, dimensions=DIMS,
                         applicable_dimensions={"s1": ["a"]})
    assert [x.dimension for x in out] == ["a"]


def test_triage_summary_reports_the_review_cost():
    p = {"s1": _jo(a=1.0, b=1.0), "s2": _jo(a=1.0, b=1.0)}
    s = {"s1": _jo(a=1.0, b=0.0), "s2": _jo(a=1.0, b=1.0)}
    t = triage_summary(compare_judges(primary=p, secondary=s, dimensions=DIMS))
    assert t["n_decisions"] == 4
    assert t["n_flagged"] == 1
    assert t["flag_rate"] == 0.25
    assert t["flagged"][0]["dimension"] == "b"


def test_confidence_index_is_keyed_for_report_attachment():
    p = {"s1": _jo(a=1.0)}
    s = {"s1": _jo(a=0.0)}
    idx = confidence_index(compare_judges(primary=p, secondary=s, dimensions=["a"]))
    assert idx[("s1", "a")] == CONFIDENCE_LOW


def test_disagreement_by_dimension_ranks_the_worst_first():
    p = {"s1": _jo(a=1.0, b=1.0), "s2": _jo(a=1.0, b=1.0)}
    s = {"s1": _jo(a=0.0, b=1.0), "s2": _jo(a=0.5, b=1.0)}
    ranked = panel_disagreement_by_dimension(compare_judges(primary=p, secondary=s, dimensions=DIMS))
    assert ranked[0]["dimension"] == "a"
    assert ranked[0]["disagreement_rate"] == 1.0
    assert ranked[1]["disagreement_rate"] == 0.0


def test_expected_accuracy_differs_by_confidence_label():
    p = {"s1": _jo(a=1.0, b=1.0)}
    s = {"s1": _jo(a=1.0, b=0.0)}
    out = {x.dimension: x for x in compare_judges(primary=p, secondary=s, dimensions=DIMS)}
    assert out["a"].expected_accuracy > out["b"].expected_accuracy


def test_module_is_generic_over_dimension_names():
    p = {"s1": {"dimension_scores": {"anything_at_all": 1.0}}}
    s = {"s1": {"dimension_scores": {"anything_at_all": 1.0}}}
    out = compare_judges(primary=p, secondary=s, dimensions=["anything_at_all"])
    assert out[0].agreed


def test_agreement_is_documented_as_confidence_not_validity():
    """Measured on the live corpus: solution_control has one of the lowest disagreement rates while
    being the most broken dimension against human labels. The module must say so, because reading a
    low rate as a healthy dimension is the natural and wrong inference."""
    src = (REPO_ROOT / "src/seg_eval/evaluation_judge/judge_panel_v1.py").read_text(encoding="utf-8")
    assert "never *validity*" in src
    assert "solution_control" in src
