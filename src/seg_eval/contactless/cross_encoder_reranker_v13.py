"""cross_encoder_reranker_v13.py

Fix 12 follow-up: v13 dense retrieval raises composite scores before the
cross-encoder stage. The inherited exact-match bypass must therefore use
the pre-dense composite score, so dense retrieval cannot make the bypass
fire on rows that would otherwise be reranked.
"""

from __future__ import annotations

from typing import Any

from .cross_encoder_reranker import (
    CE_WEIGHT,
    CROSS_ENCODER_AVAILABLE,
    EXACT_MATCH_BYPASS_THRESHOLD,
    KCCrossEncoderReranker,
    _build_kc_document,
    _minmax,
)


class KCCrossEncoderRerankerV13(KCCrossEncoderReranker):
    """v13 cross-encoder reranker with a dense-safe bypass condition."""

    def rerank(
        self,
        exchange_text: str,
        candidates: list[dict[str, Any]],
        profiles_by_id: dict[str, dict[str, Any]],
        top_k: int = 25,
        ce_weight: float = CE_WEIGHT,
    ) -> list[dict[str, Any]]:
        if len(candidates) <= 1:
            for c in candidates:
                c.setdefault("cross_encoder_score", None)
                c.setdefault(
                    "ce_reranked_composite",
                    float(c.get("composite_score") or 0),
                )
                c.setdefault("ce_bypass_score_basis", "not_applicable")
            return candidates[:top_k]

        top = candidates[0]
        bypass_score = float(
            top.get("pre_dense_composite_score", top.get("composite_score")) or 0
        )
        if bool(top.get("exact_canonical_match")) and bypass_score >= EXACT_MATCH_BYPASS_THRESHOLD:
            for c in candidates:
                c["cross_encoder_score"] = None
                c["ce_reranked_composite"] = float(c.get("composite_score") or 0)
                c["ce_bypass_score_basis"] = "pre_dense_composite_score"
            return candidates[:top_k]

        pairs: list[tuple[str, str]] = []
        for c in candidates:
            profile = profiles_by_id.get(c.get("unit_id") or "") or {}
            pairs.append((exchange_text, _build_kc_document(profile)))

        raw_ce: list[float] = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        ).tolist()

        ce_norm = _minmax(raw_ce)
        comp_norm = _minmax([float(c.get("composite_score") or 0) for c in candidates])

        for i, c in enumerate(candidates):
            c["cross_encoder_score"] = round(raw_ce[i], 6)
            blended = ce_weight * ce_norm[i] + (1.0 - ce_weight) * comp_norm[i]
            c["ce_reranked_composite"] = round(blended, 6)
            c["ce_bypass_score_basis"] = "not_bypassed"

        candidates.sort(key=lambda c: c["ce_reranked_composite"], reverse=True)
        return candidates[:top_k]
