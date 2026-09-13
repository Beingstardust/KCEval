"""cross_encoder_reranker.py

Semantic reranker for KC candidates using a pretrained cross-encoder.

Stage position in the pipeline:
  BM25 + char-ngram TF-IDF + profile native matcher
  -> composite scoring -> top-K candidates
  -> THIS MODULE -> final reranked top-K

Why this helps:
  The first-stage retrievers (BM25, char-ngram) require surface-level
  lexical overlap between exchange text and KC profile text. When a
  teacher explains a concept using analogies, informal phrasing, or
  domain jargon that does not appear verbatim in the KC matching cues,
  the first stage fails even when the exchange is semantically about
  that KC. A cross-encoder jointly encodes the exchange text and KC
  profile document through cross-attention, enabling semantic matching
  beyond exact or near-exact term overlap.

What this module does NOT replace:
  Pass 2 (context_resolver_v9 + anaphora_bridge_postprocessor_v9f)
  handles analogy-based teaching, anaphora, and context-dependent
  exchanges using neighbouring exchange KCs as additional evidence.
  The cross-encoder runs inside the candidate pool builder (before
  pass 1) and produces better-ordered candidates only. It does not
  touch pass 2 logic.

GPU support:
  Explicitly detects CUDA at init time. Falls back to CPU if CUDA is
  unavailable. batch_size=64 on GPU (RTX 5060, 8 GB VRAM) is safe for
  MiniLM-L6-v2 (22M params, ~90 MB VRAM). On CPU, batch_size=32.

Domain-agnostic:
  KC document text uses only generic schema fields: canonical_name,
  definition_or_summary, canonical_terms, matching_cues,
  likely_dialogue_surface_forms, formula_or_symbol_forms.
  The default model (ms-marco-MiniLM-L-6-v2) is trained on general
  English retrieval pairs, not any specific domain.

Optional dependency:
  sentence-transformers is not a hard dependency. If not installed,
  instantiation raises ImportError with a clear install instruction.
  candidate_pool_v12 falls back to v10 composite ranking silently.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from sentence_transformers import CrossEncoder as _CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    CROSS_ENCODER_AVAILABLE = False


DEFAULT_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GPU_BATCH_SIZE   = 64
CPU_BATCH_SIZE   = 32
CE_WEIGHT        = 0.55
EXACT_MATCH_BYPASS_THRESHOLD = 6.0

IDENTITY_FIELDS = [
    "canonical_terms",
    "matching_cues",
    "likely_dialogue_surface_forms",
    "formula_or_symbol_forms",
]


def _build_kc_document(profile: dict[str, Any]) -> str:
    """Build rich text from a KC profile for the cross-encoder document side."""
    parts: list[str] = []
    name = (profile.get("canonical_name") or "").strip()
    if name:
        parts.append(name)
    defn = (profile.get("definition_or_summary") or "").strip()
    if defn:
        parts.append(defn)
    pos = profile.get("positive_profile") or {}
    for field in IDENTITY_FIELDS:
        values = pos.get(field)
        if isinstance(values, list):
            parts.extend(str(v).strip() for v in values if str(v).strip())
        elif isinstance(values, str) and values.strip():
            parts.append(values.strip())
    leaf = (profile.get("context_profile") or {}).get("leaf_topic") or ""
    if leaf:
        parts.append(leaf.strip())
    return " | ".join(p for p in parts if p)


def _minmax(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _detect_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class KCCrossEncoderReranker:
    """Cross-encoder reranker for KC candidates.

    Usage:
        reranker = KCCrossEncoderReranker()
        profiles_by_id = {p["unit_id"]: p for p in profiles}
        candidates = reranker.rerank(exchange_text, candidates, profiles_by_id)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking.\n"
                "Install with: pip install sentence-transformers\n"
                "If using the project venv: .venv\\Scripts\\pip install sentence-transformers"
            )
        self.device     = _detect_device()
        self.model      = _CrossEncoder(model_name, max_length=512, device=self.device)
        self.model_name = model_name
        self.batch_size = GPU_BATCH_SIZE if self.device == "cuda" else CPU_BATCH_SIZE

    def rerank(
        self,
        exchange_text: str,
        candidates: list[dict[str, Any]],
        profiles_by_id: dict[str, dict[str, Any]],
        top_k: int = 25,
        ce_weight: float = CE_WEIGHT,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by blending cross-encoder scores with composite scores.

        Args:
            exchange_text: full exchange text (student + tutor concatenated)
            candidates: top-K candidates with composite_score set
            profiles_by_id: unit_id -> full KC profile dict
            top_k: number of candidates to return
            ce_weight: cross-encoder score weight in blend

        Returns:
            Reranked candidates with cross_encoder_score and
            ce_reranked_composite fields added.
        """
        if len(candidates) <= 1:
            for c in candidates:
                c.setdefault("cross_encoder_score", None)
                c.setdefault("ce_reranked_composite",
                             float(c.get("composite_score") or 0))
            return candidates[:top_k]

        # Skip reranking when top candidate already has strong exact identity.
        # Prevents CE from introducing noise into already-correct assignments.
        top = candidates[0]
        if (
            bool(top.get("exact_canonical_match"))
            and float(top.get("composite_score") or 0) >= EXACT_MATCH_BYPASS_THRESHOLD
        ):
            for c in candidates:
                c["cross_encoder_score"]   = None
                c["ce_reranked_composite"] = float(c.get("composite_score") or 0)
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

        ce_norm   = _minmax(raw_ce)
        comp_norm = _minmax([float(c.get("composite_score") or 0) for c in candidates])

        for i, c in enumerate(candidates):
            c["cross_encoder_score"]   = round(raw_ce[i], 6)
            blended = ce_weight * ce_norm[i] + (1.0 - ce_weight) * comp_norm[i]
            c["ce_reranked_composite"] = round(blended, 6)

        candidates.sort(key=lambda c: c["ce_reranked_composite"], reverse=True)
        return candidates[:top_k]
