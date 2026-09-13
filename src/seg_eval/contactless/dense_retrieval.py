"""dense_retrieval.py

Fix 12: optional dense bi-encoder retrieval over KC profile documents.

This module is retrieval/ranking only. It builds one text document per KC
profile from generic matching-profile fields, embeds those documents once,
and scores exchange text by cosine-like dot product against normalized
embeddings when the installed encoder supports normalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    DENSE_RETRIEVAL_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _SentenceTransformer = None  # type: ignore[assignment]
    DENSE_RETRIEVAL_AVAILABLE = False


DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"

PROFILE_FIELDS = [
    "canonical_terms",
    "matching_cues",
    "likely_dialogue_surface_forms",
    "formula_or_symbol_forms",
    "matched_surface_terms_from_evidence",
    "child_surface_terms",
]


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype("float32", copy=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype("float32", copy=False)
    return (vector / norm).astype("float32", copy=False)


def _detect_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class KCDenseRetriever:
    """Dense retriever for KC profile candidates."""

    def __init__(
        self,
        profiles: list[dict[str, Any]],
        model_name: str = DEFAULT_DENSE_MODEL,
        device: str | None = None,
    ) -> None:
        if not DENSE_RETRIEVAL_AVAILABLE or _SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers and torch are required for dense retrieval."
            )

        self.profiles = profiles
        self.unit_ids = [p["unit_id"] for p in profiles]
        self.model_name = model_name
        self.device = device or _detect_device()
        self.model = _SentenceTransformer(model_name, device=self.device)
        self.documents = [self._build_kc_document(p) for p in profiles]
        self.embeddings = self._encode_documents(self.documents)

    def _build_kc_document(self, profile: dict[str, Any]) -> str:
        """Build KC-side text from generic matching-profile fields."""
        parts: list[str] = []

        parts.extend(_as_text_list(profile.get("canonical_name")))
        parts.extend(_as_text_list(profile.get("definition_or_summary")))

        positive = profile.get("positive_profile") or {}
        for field in PROFILE_FIELDS:
            parts.extend(_as_text_list(positive.get(field)))

        context = profile.get("context_profile") or {}
        parts.extend(_as_text_list(context.get("leaf_topic")))

        return " | ".join(p for p in parts if p)

    def _encode_documents(self, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, 0), dtype="float32")

        try:
            encoded = self.model.encode(
                documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(encoded, dtype="float32")
        except TypeError:
            encoded = self.model.encode(
                documents,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return _normalize_matrix(np.asarray(encoded, dtype="float32"))

    def _encode_query(self, query_text: str) -> np.ndarray:
        try:
            encoded = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(encoded, dtype="float32")[0]
        except TypeError:
            encoded = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return _normalize_vector(np.asarray(encoded, dtype="float32")[0])

    def score_query(self, query_text: str, top_k: int = 40) -> list[dict[str, Any]]:
        if not (query_text or "").strip():
            return []
        if self.embeddings.size == 0:
            return []

        query = self._encode_query(query_text)
        scores = self.embeddings @ query
        limit = max(0, min(int(top_k), len(self.unit_ids)))
        if limit <= 0:
            return []

        ranked = np.argsort(-scores)[:limit]
        rows: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked, start=1):
            rows.append(
                {
                    "unit_id": self.unit_ids[int(idx)],
                    "dense_score": round(float(scores[int(idx)]), 6),
                    "dense_rank": rank,
                }
            )
        return rows
