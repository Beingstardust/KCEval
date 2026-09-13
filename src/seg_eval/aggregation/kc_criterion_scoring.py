"""Signed-weight KC-specific criterion scoring.

Semantics verified against the live system, not assumed — see
``local_audits/evaluation_completion_20260824T093426Z/05_KC_CRITERIA_WEIGHT_AUDIT.md`` for the
full audit. Summary of what was verified there and is encoded here:

- Each criterion carries a signed ``weight`` (int, nonzero) and a redundant ``polarity`` string
  ("positive"/"negative") that always agrees with the weight's sign in the 104 real draft
  criteria checked. This module trusts ``weight``'s sign as authoritative and treats a
  disagreeing ``polarity`` as a data-integrity error, not silently ignorable.
- Observed magnitudes in live data are only 1 and 2. |weight| == 3 is not forbidden by the
  schema and this module does not special-case it away — it is simply unseen so far.
- A positive-weight criterion describes a *desired* behavior: the criterion being true
  (``rating`` close to 1) is good.
- A negative-weight criterion describes an *undesirable* behavior / misconception: the criterion
  being true is bad, so its quality contribution is inverted (``1 - rating``).
- Criteria are atomic checklist items (one behavioral claim each), not partial-credit rubric
  dimensions, so the natural rating scale is pass/fail, with 0.5 available for genuinely partial
  cases the same way the rest of this rubric uses it. ``not_applicable`` criteria are excluded
  from the score entirely (never coerced to 0).
- As of the audit, zero criteria in the repo are expert-approved — this module has real
  candidate data to test against (the 104 drafts) but nothing yet flows to a live judge prompt.
  This module is deliberately not the thing that decides that; it only scores whatever
  criterion judgments it's given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Applicability = Literal["applicable", "not_applicable", "unclear"]
ALLOWED_RATINGS = (0, 0.5, 1)


class CriterionScoringError(ValueError):
    """Raised for a judgment that cannot be scored as given (bad weight, bad rating pairing)."""


@dataclass(frozen=True)
class CriterionJudgment:
    """One judged instance of one KC-specific criterion against one segment.

    ``weight`` and ``kc_id`` here are what the JUDGE echoed back (execution spec §10's
    integrity requirement) — validate them against library truth with
    :func:`check_criterion_integrity` before trusting them for scoring in a real pipeline.
    This dataclass itself only enforces internal consistency (rating vs. applicability), not
    agreement with any external library.
    """

    criterion_id: str
    kc_id: str
    weight: int
    applicability: Applicability
    rating: float | None
    polarity: str | None = None  # optional, cross-checked against weight's sign if provided

    def __post_init__(self) -> None:
        if self.weight == 0:
            raise CriterionScoringError(
                f"criterion {self.criterion_id}: weight must be nonzero, got 0"
            )
        if self.polarity is not None:
            expected = "positive" if self.weight > 0 else "negative"
            if self.polarity != expected:
                raise CriterionScoringError(
                    f"criterion {self.criterion_id}: polarity={self.polarity!r} disagrees with "
                    f"weight={self.weight} (expected polarity={expected!r})"
                )
        if self.applicability == "not_applicable":
            if self.rating is not None:
                raise CriterionScoringError(
                    f"criterion {self.criterion_id}: rating must be null when not_applicable, "
                    f"got {self.rating!r}"
                )
        else:
            if self.rating not in ALLOWED_RATINGS:
                raise CriterionScoringError(
                    f"criterion {self.criterion_id}: rating must be one of {ALLOWED_RATINGS} "
                    f"when applicability={self.applicability!r}, got {self.rating!r}"
                )


def criterion_quality(rating: float, weight: int) -> float:
    """Map a raw pass/fail-ish rating to a "higher is better" quality contribution.

    weight > 0 (desired behavior): quality = rating  (presence is good)
    weight < 0 (undesirable behavior / misconception): quality = 1 - rating  (absence is good)
    """
    if weight == 0:
        raise CriterionScoringError("weight must be nonzero")
    return float(rating) if weight > 0 else 1.0 - float(rating)


SeverityBucket = dict[str, int]  # {"n": int, "any_shortfall": int, "full_shortfall": int}


def stratify_criterion_outcomes(
    judgments: list[CriterionJudgment],
) -> dict[int, dict[str, SeverityBucket]]:
    """Break down scored (applicable/unclear) judgments by |weight| and polarity.

    Returns ``{magnitude: {"positive": bucket, "negative": bucket}}`` for every magnitude that
    actually occurs in ``judgments`` — nothing is assumed present (no hardcoded 1/2/3 keys), so
    this reports honestly whether a given magnitude tier is used at all, per the audit's own
    finding that |weight|=3 does not currently occur.

    "any_shortfall" = quality < 1 (criterion not fully satisfied). "full_shortfall" = quality
    == 0 (total failure — a fully absent desired behavior, or a fully triggered misconception).
    This is what lets severe failures be reported explicitly instead of hiding inside a mean
    (execution spec §25).
    """
    out: dict[int, dict[str, SeverityBucket]] = {}
    for j in judgments:
        if j.applicability == "not_applicable":
            continue
        mag = abs(j.weight)
        polarity = "positive" if j.weight > 0 else "negative"
        bucket = out.setdefault(mag, {
            "positive": {"n": 0, "any_shortfall": 0, "full_shortfall": 0},
            "negative": {"n": 0, "any_shortfall": 0, "full_shortfall": 0},
        })[polarity]
        q = criterion_quality(j.rating, j.weight)
        bucket["n"] += 1
        if q < 1.0:
            bucket["any_shortfall"] += 1
        if q == 0.0:
            bucket["full_shortfall"] += 1
    return out


@dataclass(frozen=True)
class KCCriterionScore:
    kc_s: float | None  # None when there is nothing applicable to score
    applicable_count: int
    not_applicable_count: int
    unclear_count: int
    total_weight_magnitude: float
    severity: dict[int, dict[str, SeverityBucket]] = field(default_factory=dict)


def score_kc_criteria(judgments: list[CriterionJudgment]) -> KCCriterionScore:
    """K_s = sum(|w_i| * quality_i) / sum(|w_i|) over applicable-or-unclear criteria.

    The judgments may come from several KCs matched to the same segment. They are pooled at the
    criterion level; this scorer never filters to, groups by, or reweights the dominant KC.

    ``unclear`` applicability is scored, not excluded — this matches the existing convention in
    ``response_contract_v2._partial_score_or_null``, where only ``not_applicable`` triggers the
    null-score rule; ``unclear`` still requires and uses a real rating. Track how much of the
    score rests on unclear-applicability judgments via the returned counts, rather than
    excluding them silently.

    Adding more mild criteria does not change the score's scale (it stays in [0, 1] regardless
    of how many criteria are summed) because both numerator and denominator scale with the same
    |w_i| terms — this is the property execution spec §33 asks to be tested explicitly.
    """
    included = [j for j in judgments if j.applicability != "not_applicable"]
    not_applicable_count = len(judgments) - len(included)
    unclear_count = sum(1 for j in included if j.applicability == "unclear")

    if not included:
        return KCCriterionScore(
            kc_s=None,
            applicable_count=0,
            not_applicable_count=not_applicable_count,
            unclear_count=unclear_count,
            total_weight_magnitude=0.0,
        )

    denom = sum(abs(j.weight) for j in included)
    numer = sum(abs(j.weight) * criterion_quality(j.rating, j.weight) for j in included)
    k_s = numer / denom

    return KCCriterionScore(
        kc_s=k_s,
        applicable_count=len(included),
        not_applicable_count=not_applicable_count,
        unclear_count=unclear_count,
        total_weight_magnitude=float(denom),
        severity=stratify_criterion_outcomes(included),
    )


def check_criterion_integrity(
    *, criterion_id: str, reported_kc_id: str, reported_weight: int,
    library_criteria_by_id: dict[str, dict],
) -> list[str]:
    """Compare what a judge echoed back against library truth (execution spec §10).

    ``library_criteria_by_id`` maps criterion_id -> {"kc_id": ..., "weight": ...} built from the
    approved library, never from the model's own output. Returns a list of error strings; empty
    means the echo matches. A model cannot invent, rewrite, or silently reweight a criterion
    without this catching it, because scoring never trusts the model's copy of kc_id/weight —
    only the library's.
    """
    errors: list[str] = []
    truth = library_criteria_by_id.get(criterion_id)
    if truth is None:
        errors.append(f"criterion_id {criterion_id!r} not found in library truth (invented?)")
        return errors
    if truth["kc_id"] != reported_kc_id:
        errors.append(
            f"criterion {criterion_id}: kc_id mismatch, library={truth['kc_id']!r} "
            f"reported={reported_kc_id!r}"
        )
    if truth["weight"] != reported_weight:
        errors.append(
            f"criterion {criterion_id}: weight mismatch, library={truth['weight']!r} "
            f"reported={reported_weight!r}"
        )
    return errors
