"""pool_application.py

Applies the generative tie-breaker to an already-built candidate pool,
BEFORE pass 1 / pass 2 run, so the entire downstream chain (local
assigner, context resolver, multilabel, segmentation) sees one consistent
ranking.

Why here and not as a post-hoc override of the final resolution: v31's
central lesson was that changing candidate-pool ranks does not
automatically change the final resolution, because pass 1 / pass 2 apply
their own eligibility logic on top. Overriding the resolved id after the
fact would invert that problem -- the resolution would change while
segments, multilabel sets, and abstention flags were still computed from
the old ranking. Reordering the pool in place keeps every downstream
consumer consistent, and whether the resolution actually follows is then
an empirical question the gold check answers rather than an assumption.

The gate is deliberately conservative and two-sided: the tie-breaker only
intervenes where the pipeline is UNCERTAIN (small top1-top2 score gap) and
the tie-breaker itself is CONFIDENT. Both thresholds are caller-supplied,
never defaulted silently, because they are design decisions that must be
justified from non-gold evidence.

Domain- and model-agnostic: no KC id, course term, or model-family
assumption appears here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .option_scoring import TieBreakCandidate, tie_break


@dataclass
class PoolTieBreakOutcome:
    exchange_id: str
    gated_in: bool
    original_top1: str | None
    chosen: str | None
    changed: bool
    top_probability: float
    probability_margin: float
    pipeline_margin: float
    reason: str


def _order_averaged_probabilities(scorer, exchange_text, candidates, max_definition_chars, prior_exchange_text=None, include_contrast_notes=False, contrast_notes_top_n=None):
    """Score both presentation orders and average. Option order is set by the
    pipeline's own ranking, so a position-biased model would otherwise just
    echo the pipeline; averaging over the forward and reversed orders cancels
    first-order position effects.

    `contrast_notes_top_n`, if given, is resolved to a fixed set of unit_ids
    from `candidates` (the pipeline-rank order, BEFORE either forward or
    reverse presentation) once, here -- not re-derived per call. That
    matters: re-deriving "first N of the list passed to tie_break" inside
    each call would select the pipeline's top-N candidates on the forward
    pass but its BOTTOM-N on the reversed pass, breaking the symmetry the
    order-averaging is built to have. The set is identical for both calls.
    """
    notes_ids = None
    if contrast_notes_top_n is not None:
        notes_ids = {c.unit_id for c in candidates[:contrast_notes_top_n]}
    forward = tie_break(scorer, exchange_text, candidates, max_definition_chars=max_definition_chars,
                        prior_exchange_text=prior_exchange_text, include_contrast_notes=include_contrast_notes,
                        contrast_notes_unit_ids=notes_ids)
    reverse = tie_break(scorer, exchange_text, list(reversed(candidates)), max_definition_chars=max_definition_chars,
                        prior_exchange_text=prior_exchange_text, include_contrast_notes=include_contrast_notes,
                        contrast_notes_unit_ids=notes_ids)
    averaged = {
        c.unit_id: (forward.probabilities.get(c.unit_id, 0.0) + reverse.probabilities.get(c.unit_id, 0.0)) / 2.0
        for c in candidates
    }
    ordered = sorted(averaged.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_p = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return top_id, top_p, top_p - runner_up, averaged


def apply_tie_breaker_to_pool(
    pool: dict[str, Any],
    scorer,
    profiles_by_id: dict[str, dict[str, Any]],
    top_k: int,
    max_pipeline_margin: float,
    min_probability: float,
    min_probability_margin: float,
    max_definition_chars: int = 400,
    prior_exchange_text: str | None = None,
    include_contrast_notes: bool = False,
    contrast_notes_top_n: int | None = None,
) -> PoolTieBreakOutcome:
    """Reorders `pool["candidates"]` in place when the gate fires.

    max_pipeline_margin: only intervene when the pipeline's own top1-top2
        `final_rank_score` gap is at or below this (i.e. the pipeline is
        genuinely undecided). Above it, the pipeline is left alone.
    min_probability / min_probability_margin: the tie-breaker must clear
        both to overrule; otherwise the existing order stands.
    contrast_notes_top_n: v38 scoped fix -- if set, contrast notes are only
        shown for the top N candidates by PIPELINE rank (before any
        presentation-order reversal), not all `top_k` options. None
        (default) preserves v36's behaviour of notes on every option.
    """
    exchange_id = pool.get("exchange_id", "")
    candidates = pool.get("candidates") or []
    if len(candidates) < 2:
        return PoolTieBreakOutcome(exchange_id, False, None, None, False, 0.0, 0.0, 0.0, "insufficient_candidates")

    ranked = sorted(candidates, key=lambda c: c.get("final_pool_rank") or 10**9)
    original_top1 = ranked[0].get("unit_id")
    pipeline_margin = float(ranked[0].get("final_rank_score") or 0.0) - float(ranked[1].get("final_rank_score") or 0.0)

    if pipeline_margin > max_pipeline_margin:
        return PoolTieBreakOutcome(
            exchange_id, False, original_top1, None, False, 0.0, 0.0, pipeline_margin, "pipeline_confident"
        )

    considered = ranked[:top_k]
    tb_candidates = [
        TieBreakCandidate(
            unit_id=c["unit_id"],
            canonical_name=c.get("canonical_name") or (profiles_by_id.get(c["unit_id"], {}) or {}).get("canonical_name") or "",
            definition_or_summary=(profiles_by_id.get(c["unit_id"], {}) or {}).get("definition_or_summary") or "",
            topic_path=(profiles_by_id.get(c["unit_id"], {}) or {}).get("topic_path") or [],
            contrast_notes=((profiles_by_id.get(c["unit_id"], {}) or {}).get("negative_profile") or {}).get("sibling_contrast_notes") or [],
        )
        for c in considered
    ]

    exchange_text = (
        f"Student: {(pool.get('student_text_raw') or '').strip()}\n"
        f"Tutor: {(pool.get('tutor_text_raw') or '').strip()}"
    )

    chosen, top_p, prob_margin, _ = _order_averaged_probabilities(
        scorer, exchange_text, tb_candidates, max_definition_chars, prior_exchange_text, include_contrast_notes,
        contrast_notes_top_n,
    )

    if top_p < min_probability or prob_margin < min_probability_margin:
        return PoolTieBreakOutcome(
            exchange_id, True, original_top1, chosen, False, top_p, prob_margin, pipeline_margin,
            "tie_breaker_not_confident",
        )
    if chosen == original_top1:
        return PoolTieBreakOutcome(
            exchange_id, True, original_top1, chosen, False, top_p, prob_margin, pipeline_margin, "agreed_with_pipeline"
        )

    # Promote the chosen candidate above the current top. A small epsilon over
    # the incumbent's score is enough: the intent is to change the ORDER, not
    # to distort the score scale that downstream branch-share computations
    # read. Every other field on every candidate is left untouched.
    incumbent_score = float(ranked[0].get("final_rank_score") or 0.0)
    for candidate in candidates:
        if candidate.get("unit_id") == chosen:
            candidate["final_rank_score"] = round(incumbent_score + 1e-4, 6)
            candidate["tie_breaker_promoted"] = True
            candidate["tie_breaker_probability"] = round(top_p, 6)
            break

    candidates.sort(key=lambda c: (-float(c.get("final_rank_score") or 0.0), str(c.get("unit_id") or "")))
    for idx, candidate in enumerate(candidates, start=1):
        candidate["final_pool_rank"] = idx

    pool["tie_breaker_applied"] = True
    pool["tie_breaker_original_top1"] = original_top1
    pool["tie_breaker_chosen"] = chosen
    pool["top_candidate_ids"] = [c["unit_id"] for c in candidates[:10]]

    return PoolTieBreakOutcome(
        exchange_id, True, original_top1, chosen, True, top_p, prob_margin, pipeline_margin, "promoted"
    )
