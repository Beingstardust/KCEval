"""Segment -> episode -> dialogue -> tutor-score aggregation.

Every function here is pure arithmetic over judge-produced values. None of it re-asks a model
anything, and none of it invents a constant the live rubric doesn't specify — where the rubric
is qualitative only (the trustworthiness cap; see ``audit_trust_consistency``), this module
audits the judge's own number for internal consistency instead of recomputing one from a
formula that does not exist anywhere in the repo (see
``local_audits/evaluation_completion_20260824T093426Z/06_CURRENT_EVALUATION_GAPS.md``).

``alpha`` (local+ARC-vs-macro) and ``beta`` (generic-vs-KC-specific) are explicitly provisional,
per execution spec §17/§23 — every function that takes them requires the caller to pass a value
rather than defaulting to one, so a provisional number can never silently become "the" number.
"""

from __future__ import annotations

from dataclasses import dataclass


def weighted_mean_applicable(
    values: dict[str, float | None],
    applicability: dict[str, str] | None = None,
    dimension_weights: dict[str, float] | None = None,
) -> float | None:
    """Generic P_s / R_e: normalized weighted mean over applicable, non-null dimensions.

    ``null`` (or an explicit ``not_applicable`` in ``applicability``) is EXCLUDED from both
    numerator and denominator — never coerced to 0. This is a direct, tested guard against the
    execution spec's explicit warning ("never treat null = 0"). Returns None (not 0.0) when
    nothing is applicable, so a caller can distinguish "scored zero" from "nothing to score".

    ``dimension_weights`` defaults to 1.0 for every dimension (equal weight), matching execution
    spec §16's "unless the live rubric already defines unequal weights, start with equal weight".
    No live rubric-defined unequal weighting was found in this audit pass; this default is the
    honest current state, not a placeholder for one that exists and was skipped.
    """
    numer = 0.0
    denom = 0.0
    for dim, value in values.items():
        if applicability is not None and applicability.get(dim) == "not_applicable":
            continue
        if value is None:
            continue
        w = 1.0 if dimension_weights is None else dimension_weights.get(dim, 1.0)
        numer += w * float(value)
        denom += w
    if denom == 0.0:
        return None
    return numer / denom


@dataclass(frozen=True)
class ProvisionalCombination:
    """Wraps a combined score together with the provisional weight that produced it.

    Carrying the weight alongside the number (rather than just returning a float) is what makes
    "PROVISIONAL_NOT_FROZEN" (execution spec §17) an unavoidable part of reading the result, not
    a comment someone can lose track of.
    """

    value: float
    weight_name: str
    weight_value: float
    status: str = "PROVISIONAL_NOT_FROZEN"


def combine_generic_and_kc(p_s: float, k_s: float | None, beta: float) -> ProvisionalCombination:
    """B_s = (1 - beta) * P_s + beta * K_s, or B_s = P_s when there are no applicable KC criteria."""
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if k_s is None:
        value = p_s
    else:
        value = (1.0 - beta) * p_s + beta * k_s
    return ProvisionalCombination(value=value, weight_name="beta", weight_value=beta)


def exchange_weighted_mean(scores_and_counts: list[tuple[float, int]]) -> float | None:
    """D_segment / D_ARC: each unit contributes in proportion to its exchange count, not to
    segmentation's confidence in it.

    ``scores_and_counts`` is a list of (unit_score, n_exchanges). Deliberately does NOT accept a
    confidence/KC-band weight — execution spec §20 is explicit that KC-assignment confidence
    must not be the primary aggregation weight, so there is no parameter here to smuggle it in
    through. Use confidence for stratified reporting separately, not this function.
    """
    total_n = sum(n for _, n in scores_and_counts)
    if total_n == 0:
        return None
    return sum(score * n for score, n in scores_and_counts) / total_n


def combine_local_and_arc(d_segment: float, d_arc: float | None, rho: float) -> ProvisionalCombination:
    """D_micro = rho * D_segment + (1 - rho) * D_ARC.

    ``rho`` must be derived from the live rubric's actual local/ARC partial-dimension counts
    (execution spec §21), not hardcoded — see
    ``local_audits/evaluation_completion_20260824T093426Z/03_CURRENT_PROMPT_PROVENANCE.md``,
    which derives rho = 8/10 = 0.8 from the live split at audit time. If the rubric's dimension
    split ever changes, recompute rho from it again rather than reusing 0.8 from memory.

    When there are no ARC-scorable units at all for this dialogue (``d_arc is None``), falls
    back to D_segment alone rather than raising, since a dialogue too short to form an episode
    still has a defined micro score.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if d_arc is None:
        value = d_segment
    else:
        value = rho * d_segment + (1.0 - rho) * d_arc
    return ProvisionalCombination(value=value, weight_name="rho", weight_value=rho)


MACRO_DIMENSIONS = ("adaptability", "consistency", "outcome_completion", "sequentiality")


def macro_score(dimension_scores: dict[str, float | None]) -> float | None:
    """M = mean of the four macro dimensions that are actually scored (non-null).

    Uses :func:`weighted_mean_applicable` under the hood (equal weight, null-excluded) rather
    than a bespoke four-way average, so the "never treat null as 0" guarantee applies here too.
    Warns by omission, not by exception, if a caller passes a dimension name outside the known
    four — extra keys are ignored, matching this module's general "don't invent structure" stance
    only in the direction of not crashing on harmless extra context, not in silently including
    unknown dimensions in the mean (they're excluded entirely, present or not).
    """
    known_only = {k: v for k, v in dimension_scores.items() if k in MACRO_DIMENSIONS}
    return weighted_mean_applicable(known_only)


def final_tutor_score(d_micro: float, m: float | None, alpha: float) -> ProvisionalCombination:
    """T = alpha * D_micro + (1 - alpha) * M.

    Falls back to D_micro alone when M is None (no macro-evaluable dialogue yet), same reasoning
    as :func:`combine_local_and_arc`.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if m is None:
        value = d_micro
    else:
        value = alpha * d_micro + (1.0 - alpha) * m
    return ProvisionalCombination(value=value, weight_name="alpha", weight_value=alpha)


# ---------------------------------------------------------------------------
# Trustworthiness: audit, don't reinvent.
#
# The live rubric's D_TRUSTWORTHINESS_CAPS_PEDAGOGICAL_USEFULNESS rule is qualitative only
# ("Cap pedagogical usefulness and force human review when the hallucination is material") --
# no numeric cap value exists anywhere in rubric_v2.py or its dependency rules (checked by direct
# import + inspection, see 06_CURRENT_EVALUATION_GAPS.md item 1's sibling finding). The judge
# itself produces `trust_adjusted_score` as part of the response contract, presumably applying
# this qualitative rule at judgment time. This module does not invent a cap formula to replace
# that. Instead: audit that the judge's own reported number is internally consistent with the
# flags it also reported, and surface a disagreement rather than silently trusting or recomputing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustConsistencyResult:
    consistent: bool
    any_trust_flag_set: bool
    reason: str | None = None


def audit_trust_consistency(
    trust_flags: dict[str, bool | int],
    pre_trust_score: float,
    reported_trust_adjusted_score: float,
    tolerance: float = 1e-9,
) -> TrustConsistencyResult:
    """Check the judge's own trust-adjusted score against its own flags, without recomputing a cap.

    Rule audited (directly from D_TRUSTWORTHINESS_CAPS_PEDAGOGICAL_USEFULNESS's stated effect,
    nothing invented beyond it): if any trust flag fired, the trust-adjusted score must be
    <= the pre-trust score (it must have actually been capped, not left unchanged or raised).
    If no trust flag fired, the two scores should match (nothing should have been adjusted).
    """
    any_flag = any(bool(v) for v in trust_flags.values())
    if any_flag:
        if reported_trust_adjusted_score > pre_trust_score + tolerance:
            return TrustConsistencyResult(
                consistent=False, any_trust_flag_set=True,
                reason=(
                    f"a trust flag fired but trust_adjusted_score "
                    f"({reported_trust_adjusted_score}) exceeds the pre-trust score "
                    f"({pre_trust_score}) -- it was not capped"
                ),
            )
        return TrustConsistencyResult(consistent=True, any_trust_flag_set=True)

    if abs(reported_trust_adjusted_score - pre_trust_score) > tolerance:
        return TrustConsistencyResult(
            consistent=False, any_trust_flag_set=False,
            reason=(
                f"no trust flag fired but trust_adjusted_score "
                f"({reported_trust_adjusted_score}) differs from the pre-trust score "
                f"({pre_trust_score})"
            ),
        )
    return TrustConsistencyResult(consistent=True, any_trust_flag_set=False)


@dataclass(frozen=True)
class TrustCapResult:
    capped_score: float
    cap_applied: bool
    cap_value: float | None
    material_contradictions: int
    verdict: str | None
    reason: str


# Deterministic trust cap constants. PROVISIONAL in the same sense as alpha/beta/rho: chosen by
# argument, not fitted to data, and stated here so the framework's penalty for material factual
# error is a property of the framework rather than a number an LLM chose per response.
#
# WHY A CAP RATHER THAN A SUBTRACTION: rubric_v2's own dependency rule reads "Cap pedagogical
# usefulness and force human review when the hallucination is material" -- a ceiling, not a
# deduction. A subtraction would let a tutor with excellent pedagogy and a material factual error
# still outscore a merely mediocre one, which is the exact inversion measured in
# 24_TUTOR_VARIANT_RESULTS.md (T3, 15 planted errors, outranked T2 which had none).
#
# WHY THESE VALUES: a single material contradiction means a student could act on false
# information from this segment, so the segment cannot be counted as good teaching -- 0.5 is the
# rubric's own "partial/mixed/weak" anchor. Repeated material contradictions in one segment drop
# it to the rubric's "poor/absent/misleading/harmful" anchor, 0.25 being half of that again.
# These are argued positions open to revision, NOT empirical findings.
TRUST_CAP_SINGLE_MATERIAL = 0.5
TRUST_CAP_REPEATED_MATERIAL = 0.25
TRUST_CAP_MINOR_DEVIATION = 0.9


def apply_deterministic_trust_cap(
    score: float,
    factuality_verdict: str | None,
    material_contradictions: int = 0,
) -> TrustCapResult:
    """Cap a unit's score based on a grounded factuality verdict, deterministically.

    Replaces reliance on the judge's self-reported ``trust_adjusted_score``, which was measured
    to be a near-constant ~4-5% offset applied whether or not any trust flag fired (T1: zero
    flags, still -0.0422), and which the literature independently characterises as
    high-variance and self-inconsistent (see 27_..._CITATIONS.md).

    A cap never raises a score: ``min`` is applied, so a segment that already scored below the
    cap is unaffected. ``insufficient_reference`` deliberately applies no cap -- absence of a
    usable reference is not evidence of error, and penalising it would make thin curriculum
    coverage look like tutor failure.
    """
    if factuality_verdict is None:
        return TrustCapResult(score, False, None, material_contradictions, None,
                               "no factuality verdict available; score unchanged")

    if factuality_verdict == "contradicted" and material_contradictions >= 2:
        cap = TRUST_CAP_REPEATED_MATERIAL
        reason = f"{material_contradictions} material contradictions of the curriculum reference"
    elif factuality_verdict == "contradicted" and material_contradictions == 1:
        cap = TRUST_CAP_SINGLE_MATERIAL
        reason = "one material contradiction of the curriculum reference"
    elif factuality_verdict == "contradicted":
        # verdict says contradicted but every listed contradiction was marked minor
        cap = TRUST_CAP_MINOR_DEVIATION
        reason = "contradicted verdict with no contradiction marked material"
    elif factuality_verdict == "minor_deviation":
        cap = TRUST_CAP_MINOR_DEVIATION
        reason = "minor deviation from the curriculum reference"
    else:  # grounded, insufficient_reference
        return TrustCapResult(score, False, None, material_contradictions, factuality_verdict,
                               f"verdict={factuality_verdict}; no cap applies")

    capped = min(score, cap)
    return TrustCapResult(capped, capped < score, cap, material_contradictions,
                           factuality_verdict, reason)


def detect_arithmetic_disagreement(
    model_reported: float, deterministic_recomputed: float, tolerance: float = 1e-6,
) -> bool:
    """True when the judge's self-reported aggregate disagrees with our independent recompute.

    Execution spec §18: "Compare: model-reported calculated score / deterministic recalculated
    score ... Use the deterministic result for reporting. Never alter the underlying judge
    labels to make arithmetic match." This function only detects; callers decide what to do with
    a True result (flag it, report it), and must never use it as a reason to edit a label.
    """
    return abs(model_reported - deterministic_recomputed) > tolerance


# ---------------------------------------------------------------------------
# alpha/beta sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogueScoreInputs:
    dialogue_id: str
    d_segment: float
    d_arc: float | None
    m: float | None


def sensitivity_grid(
    dialogues: list[DialogueScoreInputs],
    rho: float,
    alpha_values: list[float],
) -> dict[float, list[tuple[str, float]]]:
    """For each alpha, compute every dialogue's T and return them ranked (highest T first).

    Only sweeps alpha (local+ARC vs. macro), since beta (generic-vs-KC-specific) currently has
    no live KC-specific criteria to combine with (see the criteria audit) — sweeping it would be
    sweeping a parameter that can't yet change any real number. Re-add a beta sweep once
    approved criteria exist to score against.

    Execution spec §24: "generate a table showing whether tutor ordering changes under
    reasonable alpha values. If rankings change dramatically, report instability rather than
    hiding it." This function produces the ranked-order table; interpreting instability is a
    reporting step, not something this function silently smooths over.

    NOTE: as of this audit pass, the repo has evaluated only ONE dialogue end-to-end (the dm1
    "rich_dialogue" run). A ranking-order comparison is meaningless over a single dialogue --
    this function is built and tested against synthetic multi-dialogue fixtures so it is ready
    the moment a second real dialogue's D_micro/M exist, not because a real multi-dialogue
    ranking already does.
    """
    out: dict[float, list[tuple[str, float]]] = {}
    for alpha in alpha_values:
        scored = []
        for d in dialogues:
            micro = combine_local_and_arc(d.d_segment, d.d_arc, rho).value
            t = final_tutor_score(micro, d.m, alpha).value
            scored.append((d.dialogue_id, t))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        out[alpha] = scored
    return out


def ranking_is_stable(grid: dict[float, list[tuple[str, float]]]) -> bool:
    """True iff every alpha in the grid produces the same dialogue ORDER (ties broken by id)."""
    orders = [tuple(dialogue_id for dialogue_id, _ in ranked) for ranked in grid.values()]
    return len(set(orders)) <= 1
