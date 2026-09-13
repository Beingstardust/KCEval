"""meta_exclusion_v15.py

Phase 4 (v15): exclude pipeline-detected meta/admin exchanges from
segmentation entirely.

Uses the pipeline's OWN runtime meta-detection logic (local_assigner_v12.py,
gates 1-2 inside assign_pass1_v12, lines ~194-200: `_is_non_kc` for
low_content_or_admin_exchange, `_is_non_kc_meta_content` for
meta_dialogue_framing_content_weak_curriculum_evidence) -- NOT the gold
CSV's non_kc_meta label. This generalizes to dialogues with no gold labels.

Gate 3 (recap_synthesis_broad_multibranch_detected) is deliberately NOT
excluded: per local_assigner_v12.py's own docstring, a recap/synthesis
exchange "has curriculum content (unlike NON_KC_META) but no single
dominant KC" -- it is real content spanning many KCs, not admin/setup
noise, so it should keep participating in segmentation (as it already does,
absorbed into a neighbouring segment by segmenter_v3's non_kc_absorbed
rule).

This is applied as a filtering step between KC resolution (pass2) and
segmentation (segmenter_v3), not as an edit to segmenter_v3.py itself:
meta exchanges are removed from the exchange/assignment lists BEFORE
build_contactless_segments runs, so segmenter_v3 never sees them and they
cannot form their own segment or get absorbed into a neighbour's segment.
KC-resolution outputs (contactless_exchange_assignments.jsonl) are left
untouched -- every exchange, including excluded ones, still gets a row
there, so evaluate_segmentation_v2.py's non_kc_meta scoring rule (which
reads assignments, not segments) is unaffected.

Known side effect on boundary F1
---------------------------------
metrics.boundary_precision_recall_f1 compares boundary vectors built from
segment start/end positions. If a meta exchange sits at position i and is
removed from segmentation entirely, no predicted segment ends at position i
any more, so the predicted boundary vector loses a boundary that gold still
has there (gold always treats the meta exchange as its own segment_group).
This shows up as one additional false negative in the boundary metric per
excluded meta exchange that was previously a segment boundary. This is an
inherent consequence of "exclude from segmentation entirely" as specified,
not a bug in this filter -- reported explicitly in the v15 scoring report
rather than compensated for.
"""

from __future__ import annotations

from typing import Any


ADMIN_META_REASONS = {
    "low_content_or_admin_exchange",
    "meta_dialogue_framing_content_weak_curriculum_evidence",
}


def _resolution_reasons(row: dict[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for key in ("decision_reasons", "resolution_reasons"):
        values = row.get(key) or []
        if isinstance(values, list):
            reasons.update(str(v) for v in values)
    return reasons


def is_pipeline_detected_meta(row: dict[str, Any]) -> bool:
    """True only for admin/setup meta exchanges (gates 1-2). Recap/synthesis
    (gate 3) is intentionally excluded from this check -- see module docstring.
    """
    if row.get("final_label") != "NON_KC":
        return False
    reasons = _resolution_reasons(row)
    return bool(reasons & ADMIN_META_REASONS)


def exclude_meta_from_segmentation(
    exchanges: list[Any],
    assignments: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]], list[str]]:
    """Return (filtered_exchanges, filtered_assignments, excluded_exchange_ids).

    exchanges/assignments must be full, unfiltered, position-aligned lists
    (same lists passed to segmenter_v3.build_contactless_segments). The
    filtered lists are for segmentation only -- callers must still write
    the original, unfiltered `assignments` to
    contactless_exchange_assignments.jsonl.
    """
    filtered_exchanges: list[Any] = []
    filtered_assignments: list[dict[str, Any]] = []
    excluded_ids: list[str] = []

    for exchange, row in zip(exchanges, assignments):
        if is_pipeline_detected_meta(row):
            excluded_ids.append(row.get("exchange_id"))
            continue
        filtered_exchanges.append(exchange)
        filtered_assignments.append(row)

    return filtered_exchanges, filtered_assignments, excluded_ids
