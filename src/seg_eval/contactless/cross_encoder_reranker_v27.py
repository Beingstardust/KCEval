"""cross_encoder_reranker_v27.py

Fix 27: removes the exact-match bypass inherited from v13/base.

Root cause (found via candidate_pool inspection against the ablation-1
159-KC library, see data/gold/v27_pipeline_performance_assessment.md):
the exact-match bypass treats "this KC's canonical name appears verbatim
somewhere in the exchange text" as sufficient grounds to skip semantic
reranking entirely. That is true when the exchange is actually about that
KC, but false when the name is a passing/contrastive mention inside an
explanation of a *different* KC -- e.g. a tutor explaining the overall
Naive Bayes classification procedure necessarily says the words "prior
probability" as one term in the formula, which previously caused the
exchange to resolve to "Prior Probability" instead of the KC it was
actually about ("NB Classification Phase"). Confirmed as the direct cause
of 5/20 wrong resolutions (100% of the confidently-wrong, high-confidence-
band failures) in the v26-logic/159-KC-library run, with the correct KC
present in the candidate pool in every one of those 5 cases -- a ranking
failure, not a retrieval-recall failure.

The bypass was also effectively dead code in its v13 form for the
opposite reason on other exchanges: `EXACT_MATCH_BYPASS_THRESHOLD = 6.0`
was calibrated against `pre_dense_composite_score`
(range ~tens, dominated by identity_component), which exact matches
routinely clear -- so the bypass fired far more often than the "no real
competition exists" case it was presumably meant for.

Fix: always run the cross-encoder and use the existing blend
(ce_weight * ce_norm + (1 - ce_weight) * comp_norm). Exact-match candidates
keep a real advantage because their composite_score (identity-component-
driven RRF rank) still feeds comp_norm -- so unambiguous exact matches
(no topically competitive alternative) still win on the blend -- but a
cross-encoder-confirmed better semantic fit can now outrank a merely-
mentioned name, which it could never do before this fix regardless of how
strong the semantic mismatch was.

19/24 of the pre-fix run's exact-override exchanges were already correct;
this change is validated by re-scoring the full 60-exchange gold set
after the change, not assumed safe a priori (see the report above for the
before/after comparison).
"""

from __future__ import annotations

from typing import Any

from .cross_encoder_reranker import CE_WEIGHT, KCCrossEncoderReranker, _build_kc_document, _minmax


class KCCrossEncoderRerankerV27(KCCrossEncoderReranker):
    """v27 cross-encoder reranker: no exact-match bypass, always reranks."""

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
                c.setdefault("ce_reranked_composite", float(c.get("composite_score") or 0))
                c.setdefault("ce_bypass_score_basis", "not_applicable")
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
            c["ce_bypass_score_basis"] = "not_bypassed_v27"

        candidates.sort(key=lambda c: c["ce_reranked_composite"], reverse=True)
        return candidates[:top_k]
