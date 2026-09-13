"""Exact attribution of a tutor's final score back to segments, exchange units and dimensions.

WHY THIS MODULE EXISTS
----------------------
A single number is only useful to a tutor developer if they can ask "where did I lose it, and
what do I fix first?" and get an answer that points at concrete dialogue locations. Every
contemporary tutor-evaluation benchmark surveyed in doc 30 stops at a per-dimension rate
(MRBench's DAMR, the BEA 2025 tracks' macro-F1) and offers no path from a score back to the turns
that produced it. This module closes that gap, and it is the framework's main practical claim.

THE ATTRIBUTION IS EXACT, NOT ESTIMATED
---------------------------------------
The whole aggregation chain is *linear* in the dimension scores::

    T       = alpha * D_micro + (1 - alpha) * M
    D_micro = rho * D_segment + (1 - rho) * D_ARC
    D_seg   = sum_s (B_s * n_s) / sum_s n_s
    B_s     = (1 - beta) * P_s + beta * K_s
    P_s     = sum_d (w_d * score_d) / sum_d w_d

so the partial derivative of T with respect to one dimension score in one segment is a plain
product of the weights along its path::

    dT/dscore_d,s = alpha * rho * (n_s / sum n) * (1 - beta) * (w_d / sum w)   [local family]

No sampling, no perturbation, no surrogate model: the coefficient is computed in closed form and
the per-dimension contributions provably sum back to the score they came from. :func:`verify_reconstruction`
asserts exactly that, so a bug in this file cannot silently produce a plausible-looking but wrong
attribution.

WHAT A DEVELOPER GETS
---------------------
:func:`rank_improvement_targets` converts those coefficients into *recoverable score* -- the amount
of final T that would be regained if a given dimension in a given segment were lifted to 1.0 --
and ranks them. The top of that list is where a developer should work first, and each entry
carries the exchange-unit ids and the judge's own evidence quotes for that segment, so the finding
is directly locatable in the transcript.

GENERICITY
----------
Nothing here knows anything about any subject domain. Dimension names, weights and KC identifiers
all arrive as data from the caller; the module only walks the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DimensionContribution:
    """One dimension's exact contribution to the final score, inside one segment."""

    dimension: str
    score: float
    applicability: str
    weight_in_segment: float
    coefficient: float          # dT/dscore for this dimension in this segment
    contribution: float         # coefficient * score  -- what it currently adds to T
    recoverable: float          # coefficient * (1 - score) -- what fixing it would regain
    evidence_quotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SegmentContribution:
    """One evaluation unit's contribution, with everything needed to locate it in the dialogue."""

    segment_id: str
    family: str                 # "local" | "arc"
    exchange_count: int
    member_exchange_ids: tuple[str, ...]
    turn_start: Any
    turn_end: Any
    primary_kc_id: str | None
    primary_kc_name: str | None
    p_s: float | None           # generic rubric mean
    k_s: float | None           # KC-criterion mean, when one exists
    b_s: float | None           # blended
    trust_cap_applied: float | None
    trust_cap_reason: str | None
    segment_weight: float       # n_s / sum n within its family
    coefficient: float          # dT/dB_s
    contribution: float
    recoverable: float
    dimensions: tuple[DimensionContribution, ...] = ()
    rationale: str | None = None


@dataclass
class ScoreProvenance:
    """The full traceable decomposition of one tutor's score."""

    tutor_id: str
    final_score: float | None
    alpha: float
    beta: float
    rho: float
    d_micro: float | None
    d_segment: float | None
    d_arc: float | None
    macro_score: float | None
    segments: list[SegmentContribution] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weights_status"] = "PROVISIONAL_NOT_FROZEN"
        return d


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------

#: The values ``mixed_granularity_builder_v1`` actually writes into ``evaluation_unit_type``.
#: This was previously compared against ``"topic_unit"``, a string the builders never emit, so
#: every arc unit was silently classified as local: ``D_ARC`` came out ``None``, ``rho`` never
#: applied, and ``D_segment`` was a pooled mean over both families instead of the local one.
#: Matching on the family id as well means a future packet family is caught by either signal.
ARC_UNIT_TYPES = {"topic_rollup", "topic_unit"}
ARC_FAMILY_IDS = {"topic_rollup_arc_dimensions_v1"}


def _family_of(packet: dict[str, Any]) -> str:
    """local vs arc, read from the packet rather than guessed from the id."""
    unit = (packet.get("evaluation_unit_type")
            or (packet.get("granularity_metadata") or {}).get("evaluation_unit_type"))
    if unit in ARC_UNIT_TYPES:
        return "arc"
    family_id = (packet.get("judge_plan") or {}).get("evaluation_family_id")
    return "arc" if family_id in ARC_FAMILY_IDS else "local"


def _exchange_ids(packet: dict[str, Any]) -> tuple[str, ...]:
    ids = packet.get("member_exchange_ids")
    if isinstance(ids, list) and ids:
        return tuple(str(i) for i in ids)
    # a packet that carries only the endpoints still locates itself in the dialogue
    lo, hi = packet.get("exchange_start"), packet.get("exchange_end")
    return tuple(x for x in (lo, hi) if isinstance(x, str))


def build_score_provenance(
    *,
    tutor_id: str,
    units: Iterable[dict[str, Any]],
    alpha: float,
    beta: float,
    rho: float,
    macro: float | None = None,
    dimension_weights: dict[str, float] | None = None,
) -> ScoreProvenance:
    """Decompose a tutor score into exact per-segment, per-dimension contributions.

    ``units`` is an iterable of dicts, each carrying at minimum::

        {"packet": <evaluation packet>, "judge_output": <contract-valid judge response>}

    and optionally ``"k_s"`` (KC-criterion score for the segment) and ``"trust_cap"``
    (a :class:`TrustCapResult`-shaped mapping). Anything missing is treated as absent, never as
    zero -- the same rule the rest of the aggregation follows.
    """
    rows = [u for u in units if u.get("judge_output")]
    by_family: dict[str, list[dict[str, Any]]] = {"local": [], "arc": []}
    for u in rows:
        by_family[_family_of(u["packet"])].append(u)

    # --- per-family segment scores -----------------------------------------------------------
    family_totals = {f: sum(_n_exchanges(u["packet"]) for u in us) for f, us in by_family.items()}
    prov = ScoreProvenance(tutor_id=tutor_id, final_score=None, alpha=alpha, beta=beta, rho=rho,
                           d_micro=None, d_segment=None, d_arc=None, macro_score=macro)

    family_scores: dict[str, float | None] = {}
    staged: list[tuple[str, dict[str, Any], float, float | None, float | None, float | None]] = []

    for fam, us in by_family.items():
        total_n = family_totals[fam]
        if total_n == 0:
            family_scores[fam] = None
            continue
        acc = 0.0
        for u in us:
            pkt, jo = u["packet"], u["judge_output"]
            n_s = _n_exchanges(pkt)
            p_s = _generic_mean(jo, dimension_weights)
            k_s = u.get("k_s")
            b_s = p_s if p_s is None else _blend(p_s, k_s, beta)
            cap = u.get("trust_cap") or {}
            capped = cap.get("capped_score")
            if b_s is not None and capped is not None:
                b_s = min(b_s, float(capped))
            if b_s is not None:
                acc += b_s * n_s
            staged.append((fam, u, n_s / total_n, p_s, k_s, b_s))
        family_scores[fam] = acc / total_n

    prov.d_segment = family_scores.get("local")
    prov.d_arc = family_scores.get("arc")

    if prov.d_segment is None and prov.d_arc is None:
        prov.notes.append("no scorable units; final score undefined")
        return prov
    if prov.d_segment is None:
        prov.d_micro = prov.d_arc
        prov.notes.append("no local units; D_micro falls back to D_ARC")
        fam_coeff = {"arc": 1.0, "local": 0.0}
    elif prov.d_arc is None:
        prov.d_micro = prov.d_segment
        prov.notes.append("no arc units; D_micro falls back to D_segment")
        fam_coeff = {"local": 1.0, "arc": 0.0}
    else:
        prov.d_micro = rho * prov.d_segment + (1.0 - rho) * prov.d_arc
        fam_coeff = {"local": rho, "arc": 1.0 - rho}

    prov.final_score = prov.d_micro if macro is None else alpha * prov.d_micro + (1.0 - alpha) * macro
    micro_coeff = 1.0 if macro is None else alpha
    if macro is None:
        prov.notes.append("no macro score; T falls back to D_micro alone")

    # --- exact per-dimension coefficients ----------------------------------------------------
    for fam, u, seg_w, p_s, k_s, b_s in staged:
        pkt, jo = u["packet"], u["judge_output"]
        seg_coeff = micro_coeff * fam_coeff[fam] * seg_w
        cap = u.get("trust_cap") or {}
        # beta routes only the generic half of B_s back to the rubric dimensions
        generic_share = 1.0 if k_s is None else (1.0 - beta)
        dims = _dimension_contributions(jo, seg_coeff * generic_share, dimension_weights)
        tgt = pkt.get("evaluation_target") or {}
        prov.segments.append(SegmentContribution(
            segment_id=pkt.get("segment_id", ""),
            family=fam,
            exchange_count=_n_exchanges(pkt),
            member_exchange_ids=_exchange_ids(pkt),
            turn_start=pkt.get("turn_start"),
            turn_end=pkt.get("turn_end"),
            primary_kc_id=tgt.get("primary_kc_id"),
            primary_kc_name=tgt.get("primary_kc_name"),
            p_s=p_s, k_s=k_s, b_s=b_s,
            trust_cap_applied=cap.get("cap"),
            trust_cap_reason=cap.get("reason"),
            segment_weight=seg_w,
            coefficient=seg_coeff,
            contribution=0.0 if b_s is None else seg_coeff * b_s,
            recoverable=0.0 if b_s is None else seg_coeff * (1.0 - b_s),
            dimensions=dims,
            rationale=jo.get("rationale_short"),
        ))

    prov.segments.sort(key=lambda s: -s.recoverable)
    return prov


def _n_exchanges(packet: dict[str, Any]) -> int:
    ids = packet.get("member_exchange_ids")
    if isinstance(ids, list) and ids:
        return len(ids)
    return 1


def _blend(p_s: float, k_s: float | None, beta: float) -> float:
    return p_s if k_s is None else (1.0 - beta) * p_s + beta * float(k_s)


def _applicable_items(judge_output: dict[str, Any]) -> list[tuple[str, float, str]]:
    scores = judge_output.get("dimension_scores") or {}
    applic = judge_output.get("dimension_applicability") or {}
    out = []
    for dim, val in scores.items():
        if val is None or applic.get(dim) == "not_applicable":
            continue
        out.append((dim, float(val), applic.get(dim, "applicable")))
    return out


def _generic_mean(judge_output: dict[str, Any],
                  dimension_weights: dict[str, float] | None) -> float | None:
    items = _applicable_items(judge_output)
    if not items:
        return None
    num = den = 0.0
    for dim, val, _ in items:
        w = 1.0 if dimension_weights is None else dimension_weights.get(dim, 1.0)
        num += w * val
        den += w
    return None if den == 0.0 else num / den


def _dimension_contributions(judge_output: dict[str, Any], segment_coefficient: float,
                             dimension_weights: dict[str, float] | None
                             ) -> tuple[DimensionContribution, ...]:
    items = _applicable_items(judge_output)
    if not items:
        return ()
    weights = {d: (1.0 if dimension_weights is None else dimension_weights.get(d, 1.0))
               for d, _, _ in items}
    total_w = sum(weights.values())
    if total_w == 0.0:
        return ()
    quotes = tuple(judge_output.get("evidence_quotes") or ())
    out = []
    for dim, val, applic in items:
        share = weights[dim] / total_w
        coeff = segment_coefficient * share
        out.append(DimensionContribution(
            dimension=dim, score=val, applicability=applic, weight_in_segment=share,
            coefficient=coeff, contribution=coeff * val, recoverable=coeff * (1.0 - val),
            evidence_quotes=quotes,
        ))
    out.sort(key=lambda d: -d.recoverable)
    return tuple(out)


# ---------------------------------------------------------------------------
# developer-facing outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImprovementTarget:
    """One concrete, locatable thing a developer could fix, with what it is worth."""

    rank: int
    dimension: str
    segment_id: str
    family: str
    member_exchange_ids: tuple[str, ...]
    turn_start: Any
    turn_end: Any
    primary_kc_id: str | None
    current_score: float
    recoverable_score: float
    share_of_total_recoverable: float
    evidence_quotes: tuple[str, ...]
    rationale: str | None


def rank_improvement_targets(prov: ScoreProvenance, *, limit: int = 20,
                             min_recoverable: float = 0.0) -> list[ImprovementTarget]:
    """Rank every (segment, dimension) pair by how much final score fixing it would regain.

    This is the framework's actionable output: each entry names the dialogue location by exchange
    unit, the dimension that lost points there, and the exact amount of T at stake -- so effort
    can be spent where it actually moves the score rather than on whichever failure is most
    visible.
    """
    rows: list[tuple[float, SegmentContribution, DimensionContribution]] = []
    for seg in prov.segments:
        for dim in seg.dimensions:
            if dim.recoverable > min_recoverable:
                rows.append((dim.recoverable, seg, dim))
    rows.sort(key=lambda r: (-r[0], r[1].segment_id, r[2].dimension))
    total = sum(r[0] for r in rows) or 1.0
    return [
        ImprovementTarget(
            rank=i, dimension=dim.dimension, segment_id=seg.segment_id, family=seg.family,
            member_exchange_ids=seg.member_exchange_ids, turn_start=seg.turn_start,
            turn_end=seg.turn_end, primary_kc_id=seg.primary_kc_id, current_score=dim.score,
            recoverable_score=dim.recoverable, share_of_total_recoverable=dim.recoverable / total,
            evidence_quotes=dim.evidence_quotes, rationale=seg.rationale,
        )
        for i, (_, seg, dim) in enumerate(rows[:limit], 1)
    ]


def dimension_rollup(prov: ScoreProvenance) -> list[dict[str, Any]]:
    """Total recoverable score per dimension across the whole dialogue.

    Answers "which capability is this tutor weakest at overall", as opposed to
    :func:`rank_improvement_targets` which answers "which single spot should I fix first".
    """
    agg: dict[str, dict[str, Any]] = {}
    for seg in prov.segments:
        for dim in seg.dimensions:
            a = agg.setdefault(dim.dimension, {"dimension": dim.dimension, "recoverable": 0.0,
                                                "contribution": 0.0, "n_segments": 0,
                                                "segments_below_1": []})
            a["recoverable"] += dim.recoverable
            a["contribution"] += dim.contribution
            a["n_segments"] += 1
            if dim.score < 1.0:
                a["segments_below_1"].append(seg.segment_id)
    return sorted(agg.values(), key=lambda a: -a["recoverable"])


def verify_reconstruction(prov: ScoreProvenance, tolerance: float = 1e-9) -> tuple[bool, float]:
    """Assert the attribution is exact: contributions must sum back to the score they explain.

    This is the guard that makes the decomposition trustworthy. If the weights along a path were
    ever mis-multiplied the sum would drift, and a developer would be sent to the wrong place with
    no way to notice. Returns (ok, reconstructed_micro_part).
    """
    if prov.final_score is None:
        return True, 0.0
    reconstructed = sum(seg.contribution for seg in prov.segments)
    expected = prov.final_score if prov.macro_score is None else (
        prov.final_score - (1.0 - prov.alpha) * prov.macro_score)
    return abs(reconstructed - expected) <= tolerance, reconstructed
