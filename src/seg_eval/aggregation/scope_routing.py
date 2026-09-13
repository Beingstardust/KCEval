"""Which scope scores which rubric dimension. Versioned, so a routing change never moves a
published number.

v1  The routing every published result used (mixed_granularity_builder_v1): arc scope scores
    scaffolding_quality and student_level_calibration, everything else is segment scope.
v2  Adds proactive_clarification to arc scope. Whether a tutor resolves an ambiguous request
    before answering is a pattern across turns: a single exchange rarely holds both the ambiguity
    and the choice to clarify it. On single exchanges the dimension is almost never applicable,
    which is what both judges and the human annotator found.

Default  v2 is the framework assignment for new work (CURRENT_ROUTING), adopted by the author on
    2026-09-11: proactive clarification is a topic-scope construct even though this corpus never
    exercises it. Scripts that reproduce published or preregistered numbers pass v1 explicitly.

Routing is applied at READ time. The judge prompt asks every dimension on every unit, so moving a
dimension between scopes needs no new judge call: v2 reads proactive_clarification from arc units
and ignores it on segment units.

routed_unit_score is what makes a routing real. The judge-reported unit scalar (micro_score_raw,
trust_adjusted_score) averages all ten dimensions on every unit, so without it every unit score
would include dimensions its scope does not own and a routing change would move nothing.
"""
from __future__ import annotations

from seg_eval.aggregation.deterministic_aggregation import weighted_mean_applicable
from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS
from seg_eval.evaluation_packets.mixed_granularity_builder_v1 import (
    ARC_PARTIAL_DIMENSIONS as _V1_ARC_DIMS,
    ARC_TRUST_FLAGS as _V1_ARC_FLAGS,
)

ROUTING_VERSIONS = ("v1", "v2")
CURRENT_ROUTING = "v2"  # framework assignment for new work; reproduction scripts pin v1 explicitly
_ARC_DIMS = {
    "v1": list(_V1_ARC_DIMS),
    "v2": list(_V1_ARC_DIMS) + ["proactive_clarification"],
}
_ARC_FLAGS = {
    "v1": list(_V1_ARC_FLAGS),
    "v2": list(_V1_ARC_FLAGS),
}


def _check(version: str) -> None:
    if version not in ROUTING_VERSIONS:
        raise ValueError("unknown routing version %r; known: %s" % (version, ROUTING_VERSIONS))


def arc_dimensions(version: str = CURRENT_ROUTING) -> list[str]:
    _check(version)
    return list(_ARC_DIMS[version])


def local_dimensions(version: str = CURRENT_ROUTING) -> list[str]:
    arc = set(arc_dimensions(version))
    return [d for d in PARTIAL_DIMENSIONS if d not in arc]


def arc_trust_flags(version: str = CURRENT_ROUTING) -> list[str]:
    _check(version)
    return list(_ARC_FLAGS[version])


def local_trust_flags(version: str = CURRENT_ROUTING) -> list[str]:
    arc = set(arc_trust_flags(version))
    return [f for f in TRUST_FLAGS if f not in arc]


def family_of(dimension: str, version: str = CURRENT_ROUTING) -> str:
    return "arc" if dimension in arc_dimensions(version) else "local"


def routed_unit_score(judge_output: dict, family: str, version: str = CURRENT_ROUTING) -> float | None:
    """Applicable-mean of ONLY the dimensions `family` owns under `version`.

    Uses the dependency-adjusted dimension_scores, the same input as compute_tutor_scores and
    analyze_grounded_factuality. Nulls and not_applicable are excluded, never scored 0. None when
    the unit has no applicable owned dimension, so the caller can drop it instead of scoring it.
    """
    if family not in ("local", "arc"):
        raise ValueError("family must be local or arc, got %r" % family)
    dims = arc_dimensions(version) if family == "arc" else local_dimensions(version)
    scores = judge_output.get("dimension_scores") or {}
    return weighted_mean_applicable({d: scores.get(d) for d in dims},
                                    judge_output.get("dimension_applicability") or {})
