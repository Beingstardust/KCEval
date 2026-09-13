"""hybrid_retrieval_v2.py

v1 -> v2 changes:
  Fix 3a: FieldedBM25Index.score_query corpus-frequency guard is now
    field-aware instead of a single global 0.45 cutoff applied uniformly
    across all fields before scoring. The guard previously dropped query
    tokens appearing in >=45% of all KC profiles' combined fields *before*
    any field-level scoring happened, which meant a token like
    "classification" could be dropped from the query entirely even though
    it is highly discriminative when it appears in the canonical/identity
    field of a specific KC. v2 instead applies the frequency check
    per-field at scoring time: canonical/identity fields use a relaxed
    0.65 ceiling, support/topic fields keep 0.50, and the definition field
    keeps the original 0.45. This lets canonical-field exact hits survive
    even when the underlying token is common in definitions/support text
    across many KCs.

  Fix 3c: search_views now multiplies hybrid_retrieval_score by a small
    agreement penalty when only one retrieval source (bm25 or char) fired
    for a candidate. This pushes multi-signal candidates above
    single-signal ones without changing the underlying per-source scores,
    which keeps each index's calibration independent and inspectable.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


COMMON_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "hers", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "ours", "out", "over", "per", "she", "should", "so", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "why", "will",
    "with", "within", "without", "would", "you", "your", "yours", "yes",
    "no", "ok", "okay", "right", "actually", "basically", "just", "maybe",
    "please", "thanks", "thank", "exercise", "exercises", "question",
    "questions", "answer", "answers", "source", "material", "materials",
    "example", "examples", "task", "tasks", "problem", "problems", "sheet",
    "sheets"
}

# Per-field corpus-frequency ceilings for the BM25 query-token guard.
# A token is dropped from scoring against a given field only if its
# document frequency (within that field's documents) exceeds the field's
# ceiling. Canonical/identity fields get a much higher ceiling because a
# term can be common across a domain's KC set while still being
# discriminative for any one canonical name (e.g. "classification" appears
# in many KC definitions but is the identity term for exactly one KC).
FIELD_FREQUENCY_CEILINGS = {
    "canonical": 0.65,
    "identity": 0.65,
    "dialogue_forms": 0.55,
    "topic": 0.50,
    "support": 0.50,
    "definition": 0.45,
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"([a-z])[-_/]([a-z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    out = []
    for tok in normalize_text(text).split():
        if len(tok) < 2:
            continue
        if tok in COMMON_RETRIEVAL_STOPWORDS:
            continue
        out.append(tok)
    return out


def profile_field_values(profile: dict[str, Any], field: str) -> list[str]:
    if field == "canonical_name":
        value = profile.get("canonical_name")
    elif field == "definition_or_summary":
        value = profile.get("definition_or_summary")
    elif field == "topic_path":
        value = profile.get("topic_path") or []
    elif field == "leaf_topic":
        value = (profile.get("context_profile") or {}).get("leaf_topic")
    elif field in {
        "canonical_terms",
        "matching_cues",
        "likely_dialogue_surface_forms",
        "formula_or_symbol_forms",
        "matched_surface_terms_from_evidence",
        "child_surface_terms",
    }:
        value = (profile.get("positive_profile") or {}).get(field)
    else:
        value = (profile.get("negative_profile") or {}).get(field)

    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


@dataclass(frozen=True)
class HybridHit:
    unit_id: str
    bm25_score: float
    char_score: float
    hybrid_score: float
    bm25_rank: int | None
    char_rank: int | None
    matched_bm25_terms: list[str]


class FieldedBM25Index:
    """Small in-process BM25F-style index over frozen matching-profile fields.

    v2: the corpus-frequency query guard is field-aware (see
    FIELD_FREQUENCY_CEILINGS) instead of a single global cutoff applied
    before any field scoring. This is intentionally local and
    dependency-free. It uses BM25 scoring per field and combines fields
    with fixed generic weights. It does not hardcode any course terms.
    """

    field_weights = {
        "canonical": 2.8,
        "identity": 2.4,
        "dialogue_forms": 1.6,
        "topic": 1.0,
        "support": 0.9,
        "definition": 0.55,
    }

    def __init__(self, profiles: list[dict[str, Any]], k1: float = 1.4, b: float = 0.75):
        self.profiles = profiles
        self.k1 = k1
        self.b = b
        self.unit_ids = [p["unit_id"] for p in profiles]
        self.doc_count = max(len(profiles), 1)
        self.field_tokens: dict[str, list[list[str]]] = {field: [] for field in self.field_weights}
        self.avg_len: dict[str, float] = {}
        # Per-field document frequency, used for both IDF and the
        # field-aware frequency guard.
        self.field_df: dict[str, Counter[str]] = {field: Counter() for field in self.field_weights}

        for profile in profiles:
            fields = self._profile_fields(profile)
            for field in self.field_weights:
                field_toks = tokens(" ".join(fields[field]))
                self.field_tokens[field].append(field_toks)
                self.field_df[field].update(set(field_toks))

        for field, docs in self.field_tokens.items():
            lengths = [len(d) for d in docs]
            self.avg_len[field] = sum(lengths) / max(len(lengths), 1) or 1.0

    def _profile_fields(self, profile: dict[str, Any]) -> dict[str, list[str]]:
        return {
            "canonical": [str(profile.get("canonical_name") or "")] + profile_field_values(profile, "canonical_terms"),
            "identity": (
                profile_field_values(profile, "matching_cues")
                + profile_field_values(profile, "formula_or_symbol_forms")
            ),
            "dialogue_forms": profile_field_values(profile, "likely_dialogue_surface_forms"),
            "topic": profile_field_values(profile, "topic_path") + profile_field_values(profile, "leaf_topic"),
            "support": (
                profile_field_values(profile, "matched_surface_terms_from_evidence")
                + profile_field_values(profile, "child_surface_terms")
            ),
            "definition": profile_field_values(profile, "definition_or_summary"),
        }

    def _idf(self, tok: str, field: str) -> float:
        df = self.field_df[field].get(tok, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def _field_ceiling_ok(self, tok: str, field: str) -> bool:
        ceiling = FIELD_FREQUENCY_CEILINGS.get(field, 0.45)
        df = self.field_df[field].get(tok, 0)
        return (df / self.doc_count) < ceiling

    def score_query(self, query: str) -> list[dict[str, Any]]:
        q_tokens = list(dict.fromkeys(tokens(query)))
        if not q_tokens:
            return []

        scores = np.zeros(len(self.unit_ids), dtype=float)
        matched_terms_by_doc: list[set[str]] = [set() for _ in self.unit_ids]

        for field, weight in self.field_weights.items():
            # Field-aware guard: drop only tokens that are too common
            # *within this field* rather than dropping a token globally
            # for every field just because it is common somewhere.
            field_q_tokens = [t for t in q_tokens if self._field_ceiling_ok(t, field)]
            if not field_q_tokens:
                continue

            docs = self.field_tokens[field]
            avgdl = self.avg_len[field]
            for idx, doc_tokens in enumerate(docs):
                if not doc_tokens:
                    continue
                tf = Counter(doc_tokens)
                dl = len(doc_tokens)
                field_score = 0.0
                for tok in field_q_tokens:
                    freq = tf.get(tok, 0)
                    if freq <= 0:
                        continue
                    denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                    field_score += self._idf(tok, field) * ((freq * (self.k1 + 1.0)) / denom)
                    matched_terms_by_doc[idx].add(tok)
                scores[idx] += weight * field_score

        max_score = float(scores.max()) if len(scores) else 0.0
        rows = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            rows.append({
                "unit_id": self.unit_ids[idx],
                "bm25_score_raw": round(float(score), 6),
                "bm25_score": round(float(score) / max_score, 6) if max_score > 0 else 0.0,
                "matched_bm25_terms": sorted(matched_terms_by_doc[idx])[:20],
            })
        rows.sort(key=lambda r: r["bm25_score_raw"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["bm25_rank"] = rank
        return rows


class CharTfidfIndex:
    """Character n-gram TF-IDF surface-similarity index.

    char_wb n-grams give a lightweight typo/morphology/surface-form signal and
    complement BM25 without adding a neural dependency.
    """

    def __init__(self, profiles: list[dict[str, Any]]):
        self.profiles = profiles
        self.unit_ids = [p["unit_id"] for p in profiles]
        self.docs = [self._profile_text(p) for p in profiles]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, norm="l2")
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def _profile_text(self, profile: dict[str, Any]) -> str:
        parts: list[str] = []
        for field in [
            "canonical_name",
            "canonical_terms",
            "matching_cues",
            "likely_dialogue_surface_forms",
            "formula_or_symbol_forms",
            "matched_surface_terms_from_evidence",
            "child_surface_terms",
            "topic_path",
            "leaf_topic",
            "definition_or_summary",
        ]:
            parts.extend(profile_field_values(profile, field))
        return " \n ".join(parts)

    def score_query(self, query: str) -> list[dict[str, Any]]:
        if not (query or "").strip():
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        rows = []
        for idx, score in enumerate(sims):
            if score <= 0.0:
                continue
            rows.append({
                "unit_id": self.unit_ids[idx],
                "char_score": round(float(score), 6),
            })
        rows.sort(key=lambda r: r["char_score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["char_rank"] = rank
        return rows


class HybridRetrievalIndex:
    def __init__(self, profiles: list[dict[str, Any]]):
        self.profiles = profiles
        self.bm25 = FieldedBM25Index(profiles)
        self.char = CharTfidfIndex(profiles)

    def search_views(
        self,
        student_text: str,
        tutor_text: str,
        combined_text: str,
        top_k: int = 40,
    ) -> dict[str, Any]:
        bm25_student = self.bm25.score_query(student_text)
        bm25_tutor = self.bm25.score_query(tutor_text)
        bm25_combined = self.bm25.score_query(combined_text)
        char_student = self.char.score_query(student_text)
        char_tutor = self.char.score_query(tutor_text)
        char_combined = self.char.score_query(combined_text)

        def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {r["unit_id"]: r for r in rows}

        views = {
            "bm25_student": by_id(bm25_student),
            "bm25_tutor": by_id(bm25_tutor),
            "bm25_combined": by_id(bm25_combined),
            "char_student": by_id(char_student),
            "char_tutor": by_id(char_tutor),
            "char_combined": by_id(char_combined),
        }

        ids: set[str] = set()
        for rows in [bm25_student, bm25_tutor, bm25_combined, char_student, char_tutor, char_combined]:
            ids.update(r["unit_id"] for r in rows[:top_k])

        rows = []
        for uid in ids:
            bs = float((views["bm25_student"].get(uid) or {}).get("bm25_score") or 0.0)
            bt = float((views["bm25_tutor"].get(uid) or {}).get("bm25_score") or 0.0)
            bc = float((views["bm25_combined"].get(uid) or {}).get("bm25_score") or 0.0)
            cs = float((views["char_student"].get(uid) or {}).get("char_score") or 0.0)
            ct = float((views["char_tutor"].get(uid) or {}).get("char_score") or 0.0)
            cc = float((views["char_combined"].get(uid) or {}).get("char_score") or 0.0)

            bm25_score = max(bs * 1.25, bt * 0.75, bc)
            char_score = max(cs * 1.15, ct * 0.65, cc)

            source_agreement = 0
            evidence_sources = []
            if bm25_score >= 0.20:
                source_agreement += 1
                evidence_sources.append("bm25")
            if char_score >= 0.12:
                source_agreement += 1
                evidence_sources.append("char_tfidf")
            if bs >= 0.12 or cs >= 0.08:
                source_agreement += 1
                evidence_sources.append("student_view")

            matched_terms: set[str] = set()
            for key in ["bm25_student", "bm25_tutor", "bm25_combined"]:
                matched_terms.update((views[key].get(uid) or {}).get("matched_bm25_terms") or [])

            bm25_ranks = [
                (views[key].get(uid) or {}).get("bm25_rank")
                for key in ["bm25_student", "bm25_tutor", "bm25_combined"]
                if (views[key].get(uid) or {}).get("bm25_rank") is not None
            ]
            char_ranks = [
                (views[key].get(uid) or {}).get("char_rank")
                for key in ["char_student", "char_tutor", "char_combined"]
                if (views[key].get(uid) or {}).get("char_rank") is not None
            ]

            # Fix 3c: penalise candidates supported by only a single
            # retrieval source (bm25 OR char, not both). This nudges
            # multi-signal candidates above single-signal ones in the
            # combined ranking without altering either index's own
            # calibration.
            raw_hybrid = (0.65 * bm25_score) + (0.35 * char_score)
            single_source = (bm25_score > 0 and char_score == 0.0) or (char_score > 0 and bm25_score == 0.0)
            agreement_penalty = 0.70 if (single_source and source_agreement <= 1) else 1.0
            hybrid_score = raw_hybrid * agreement_penalty

            rows.append({
                "unit_id": uid,
                "bm25_score": round(bm25_score, 6),
                "char_score": round(char_score, 6),
                "hybrid_retrieval_score": round(hybrid_score, 6),
                "hybrid_retrieval_score_raw": round(raw_hybrid, 6),
                "single_source_penalty_applied": agreement_penalty < 1.0,
                "bm25_rank": min(bm25_ranks) if bm25_ranks else None,
                "char_rank": min(char_ranks) if char_ranks else None,
                "evidence_sources": evidence_sources,
                "hybrid_agreement_count": source_agreement,
                "matched_bm25_terms": sorted(matched_terms)[:20],
                "hybrid_role_scores": {
                    "bm25_student": round(bs, 6),
                    "bm25_tutor": round(bt, 6),
                    "bm25_combined": round(bc, 6),
                    "char_student": round(cs, 6),
                    "char_tutor": round(ct, 6),
                    "char_combined": round(cc, 6),
                },
            })

        rows.sort(key=lambda r: (r["hybrid_retrieval_score"], r["hybrid_agreement_count"]), reverse=True)
        return {
            "hybrid_candidates": rows[:top_k],
            "bm25_top_ids": [r["unit_id"] for r in bm25_combined[:10]],
            "char_top_ids": [r["unit_id"] for r in char_combined[:10]],
        }
