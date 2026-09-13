"""segment_multilabel_v16.py

Phase 3 (v16): adds a segment-level multi-label aggregate field,
`segment_kc_ids`, additive alongside the existing `dominant_kc_id` (left
unchanged) and `kc_set` (left unchanged, untouched, used only by
segmenter_v3's stabilisation rule as before).

Reconciliation with the existing kc_set field (per Phase 0 finding)
----------------------------------------------------------------------------
Phase 0 traced `kc_set` (populated in context_resolver_v11.py's
`_make_final_from_pass1_v9`, consumed by segmenter_v3.py's `_same_segment`)
and found it is the raw top-6-by-composite-score slice of an exchange's
candidate pool, with NO relevance filtering -- confirmed on live v15 data:
ex_0013's kc_set includes `KC_FSEL_STAT_003` (Feature Selection), completely
unrelated to the Naive Bayes conditional-probability content of that
exchange. It exists purely so segmenter_v3 can ask "does this segment's
established dominant KC appear anywhere in this row's nearby candidates" for
merge decisions -- a structural signal, not a claim that every KC in it is
a genuine secondary concept.

`segment_kc_ids` here is a DIFFERENT field, deliberately not reusing or
renaming `kc_set`, because conflating them would silently change the meaning
of data segmenter_v3 already depends on. `segment_kc_ids` is the
deduplicated union of member exchanges' `resolved_kc_ids` (Phase 2's
relevance-filtered dominant+secondary list), not a raw top-N pool slice.

This module does not touch segment boundaries or dominant_kc_id selection --
purely additive metadata attached after segmenter_v3.build_contactless_segments
runs.
"""

from __future__ import annotations

from typing import Any


def _unique_kc_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for e in entries:
        kc_id = e.get("kc_id")
        if not kc_id or kc_id in seen:
            continue
        seen.add(kc_id)
        out.append({"kc_id": kc_id, "kc_name": e.get("kc_name")})
    return out


def annotate_segment_kc_ids_v16(
    segments: list[dict[str, Any]],
    assignments_by_exchange_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for seg in segments:
        member_ids = seg.get("member_exchange_ids") or []
        all_entries: list[dict[str, Any]] = []
        for eid in member_ids:
            row = assignments_by_exchange_id.get(eid)
            if not row:
                continue
            all_entries.extend(row.get("resolved_kc_ids") or [])
        seg["segment_kc_ids"] = _unique_kc_entries(all_entries)
        seg["segment_dominant_kc_id"] = seg.get("dominant_kc_id")
    return segments
