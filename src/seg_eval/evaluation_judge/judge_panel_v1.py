"""Cross-family judge agreement as a per-decision confidence and triage signal.

MEASURED RESULT
---------------
Running Qwen3-8B and Selene-70B -- deliberately different model families -- over the same 640
dimension decisions on the 104 matched human-labelled units:

    panel agrees      567 decisions (88.6%)   either judge is 92.1% accurate
    panel disagrees    73 decisions (11.4%)   Qwen 39.7%, Selene 57.5%

**40.8% of Selene's errors fall inside the 11.4% of decisions the panel disagrees on** -- a 3.6x
enrichment. Disagreement is therefore a cheap and well-calibrated detector of the decisions most
likely to be wrong.

WHAT THIS IS AND IS NOT
-----------------------
It is NOT a claim that a panel is more accurate than one judge. That comparison is easy to botch:
scoring "did either judge get it right" against "did this one judge get it right" gives the panel
two shots at the answer and guarantees a flattering number. Everything above is like-for-like --
the same single judge scored separately on the agree and disagree subsets.

What it IS: a confidence label attached to each decision, and a triage rule. Review the 11.4% of
decisions flagged by disagreement and you inspect roughly a ninth of the output while covering
about two fifths of the residual error.

WHY CROSS-FAMILY
----------------
Panel composition, not panel size, drives reliability: same-family judges share training lineage
and so share systematic biases, which averaging amplifies rather than cancels. The 2026 result
"Nine Judges, Two Effective Votes" finds a 9-judge, 7-family panel carries only about two
independent votes, so a small cross-family pair captures most of the available benefit -- which is
what is used here. See doc 31 for citations.

GENERICITY
----------
No domain knowledge and no rubric knowledge: this compares two label maps whatever their keys are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

CONFIDENCE_HIGH = "panel_agreement"
CONFIDENCE_LOW = "panel_disagreement"

# measured on the 104 matched units; recorded so a consumer can see the basis of the label
MEASURED_ACCURACY_WHEN_AGREED = 0.921
MEASURED_ACCURACY_WHEN_DISAGREED = 0.575
MEASURED_ERROR_CAPTURE_RATE = 0.408
MEASURED_FLAG_RATE = 0.114
MEASUREMENT_BASIS = "640 dimension decisions over 104 matched human-labelled units, 2026-08-30"


@dataclass(frozen=True)
class DecisionConfidence:
    """One dimension decision, with whether the panel backed it."""

    segment_id: str
    dimension: str
    primary_score: float | None
    secondary_score: float | None
    agreed: bool
    confidence: str
    expected_accuracy: float


def _score(judge_output: dict[str, Any], dimension: str) -> float | None:
    return (judge_output.get("dimension_scores") or {}).get(dimension)


def _comparable(value: float | None) -> str:
    """null and a numeric score are different answers, so null must compare as its own category."""
    return "N/A" if value is None else str(value)


def compare_judges(
    *,
    primary: dict[str, dict[str, Any]],
    secondary: dict[str, dict[str, Any]],
    dimensions: Iterable[str],
    applicable_dimensions: dict[str, Iterable[str]] | None = None,
) -> list[DecisionConfidence]:
    """Label every decision the two judges both made.

    ``primary`` and ``secondary`` map segment_id -> judge output. ``applicable_dimensions`` may map
    segment_id -> the dimensions in scope for that unit, so an out-of-family dimension is skipped
    rather than counted as agreement on a shared null.
    """
    dims = list(dimensions)
    out: list[DecisionConfidence] = []
    for sid in sorted(set(primary) & set(secondary)):
        in_scope = set(applicable_dimensions[sid]) if applicable_dimensions and sid in applicable_dimensions else set(dims)
        for d in dims:
            if d not in in_scope:
                continue
            a, b = _score(primary[sid], d), _score(secondary[sid], d)
            agreed = _comparable(a) == _comparable(b)
            out.append(DecisionConfidence(
                segment_id=sid, dimension=d, primary_score=a, secondary_score=b, agreed=agreed,
                confidence=CONFIDENCE_HIGH if agreed else CONFIDENCE_LOW,
                expected_accuracy=(MEASURED_ACCURACY_WHEN_AGREED if agreed
                                   else MEASURED_ACCURACY_WHEN_DISAGREED),
            ))
    return out


def triage_summary(decisions: list[DecisionConfidence]) -> dict[str, Any]:
    """How much review the disagreement flag asks for, and what it is expected to buy."""
    n = len(decisions)
    flagged = [d for d in decisions if not d.agreed]
    return {
        "n_decisions": n,
        "n_flagged": len(flagged),
        "flag_rate": len(flagged) / n if n else None,
        "expected_error_capture": MEASURED_ERROR_CAPTURE_RATE,
        "measurement_basis": MEASUREMENT_BASIS,
        "flagged": [{"segment_id": d.segment_id, "dimension": d.dimension,
                     "primary": d.primary_score, "secondary": d.secondary_score}
                    for d in flagged],
    }


def confidence_index(decisions: list[DecisionConfidence]) -> dict[tuple[str, str], str]:
    """(segment_id, dimension) -> confidence label, for attaching to a feedback report."""
    return {(d.segment_id, d.dimension): d.confidence for d in decisions}


def panel_disagreement_by_dimension(decisions: list[DecisionConfidence]) -> list[dict[str, Any]]:
    """Which dimensions the two families most often read differently.

    MEASURED CAVEAT -- do not read a low rate here as a healthy dimension. ``solution_control`` has
    one of the LOWEST disagreement rates in the live corpus (6.9%) while being the most broken
    dimension in it (kappa -0.07 against human labels): both families default to "1" together, so
    they agree on the wrong answer. This was expected to flag compound-question defects and does
    not.

    Agreement therefore measures *confidence*, never *validity*. Two judges from different families
    still share a blind spot whenever the question itself invites one -- the correlated-error
    problem behind "Nine Judges, Two Effective Votes". Only labels from outside the model
    population, i.e. the human validation set, can tell you a dimension is measuring the wrong
    thing. Use this ranking to find hard cases, not broken constructs.
    """
    agg: dict[str, dict[str, Any]] = {}
    for d in decisions:
        a = agg.setdefault(d.dimension, {"dimension": d.dimension, "n": 0, "disagreements": 0})
        a["n"] += 1
        a["disagreements"] += (not d.agreed)
    for a in agg.values():
        a["disagreement_rate"] = a["disagreements"] / a["n"] if a["n"] else None
    return sorted(agg.values(), key=lambda a: -(a["disagreement_rate"] or 0))
