"""context_resolver_v35.py

v13's pass-2 resolver, with ONE scoped guard added: the
`pass2_broad_multibranch_closed_set` branch may no longer overrule a
candidate that the generative tie-breaker deliberately promoted to rank 1.

WHY (measured, not assumed)
---------------------------
Every resolution stage was scored on how often it resolves to something
*other* than the candidate pool's top-1, and whether that override helped
or hurt, pooled over both dialogues (n=102). See
`data/gold/downstream_override_diagnosis.md`:

    stage                                overrides   BETTER   WORSE
    pass1_closed_set                          1         0        0
    pass2_broad_multibranch_closed_set        3         0        2
    (all other stages)                        0         -        -

That branch overrides three times in 102 exchanges and has **never once
been right**. It is the only stage in the pipeline that meaningfully
overrides at all, so this guard is supported by that branch's entire
override record rather than by the two exchanges it happens to fix. n=3
overrides is small, and that is stated plainly rather than dressed up.

The branch itself is otherwise left intact: when it *agrees* with the pool
top-1 (21 of its 24 firings) nothing changes, and it still applies fully
whenever the tie-breaker did not promote anything for that exchange.

WHY A WRAPPER, NOT A FORK
-------------------------
`context_resolver_v13.py` is untouched, per this project's versioned/
additive discipline. Copying its ~100-line pass-2 body to change one
branch would create a second place for the logic to drift; wrapping it
keeps exactly one implementation.

This corrects the row *inside* the pass-2 step, before
`annotate_low_confidence_v15`, `annotate_resolved_kc_ids_v17`,
`annotate_abstention_v24` and segmentation run -- so every downstream
consumer sees one consistent value. That is the distinction from a
post-hoc override of the final output, which would leave segments and
multilabel sets computed from a stale ranking (the v31 lesson).
"""

from __future__ import annotations

from typing import Any

from .context_resolver_v13 import resolve_pass2_v13

GUARDED_ACTION = "pass2_broad_multibranch_closed_set"


def _promoted_top1(pool: dict[str, Any]) -> dict[str, Any] | None:
    """The pool's rank-1 candidate, but only if the tie-breaker put it there.

    `tie_breaker_promoted` is set by tie_breaker/pool_application.py, which
    only promotes after the verdict clears BOTH the minimum-probability and
    the order-stability margin bars -- so a promoted candidate is
    high-confidence by construction and needs no second threshold here.
    """
    candidates = pool.get("candidates") or []
    if not candidates:
        return None
    top = min(candidates, key=lambda c: c.get("final_pool_rank") or 10**9)
    if not top.get("tie_breaker_promoted"):
        return None
    return top


def resolve_pass2_v35(pass1_rows: list[dict], pools: list[dict]) -> tuple[list[dict], list[dict]]:
    resolved, novelty = resolve_pass2_v13(pass1_rows, pools)

    for row, pool in zip(resolved, pools):
        if row.get("resolution_action") != GUARDED_ACTION:
            continue
        top = _promoted_top1(pool)
        if not top:
            continue
        if row.get("resolved_kc_id") == top.get("unit_id"):
            continue

        row["resolution_reasons"] = list(row.get("resolution_reasons") or []) + [
            "broad_multibranch_override_suppressed_by_tie_breaker_promotion",
            f"superseded_{row.get('resolved_kc_id')}",
        ]
        row["resolved_kc_id"] = top.get("unit_id")
        row["resolved_kc_name"] = top.get("canonical_name")
        row["branch"] = top.get("branch") or row.get("branch")
        row["resolution_action"] = "pass2_tie_breaker_promotion_retained"

    return resolved, novelty
