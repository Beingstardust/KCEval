"""candidate_pool_v28.py

v28 = v27's un-bypassed reranking, with the reranker MODEL swapped from
`cross-encoder/ms-marco-MiniLM-L-6-v2` (generic web query/passage
relevance) to `BAAI/bge-reranker-v2-m3`. v27 proved the bypass fix alone
was insufficient: with the bypass removed, ms-marco-MiniLM still could not
separate closely-related siblings sharing dense technical vocabulary
(e.g. Naive Bayes' Prior Probability / Conditional Probability / NB
Classification Phase / NB Learning Phase) -- confirmed net-negative
against gold (66.7% -> 61.7% acceptable), with the correct KC sometimes
scoring *highest* in raw cross-encoder terms yet still losing after the
composite-score blend picked a third, still-wrong candidate. See
data/gold/v27_exact_override_fix_validation.md for that evidence.

BAAI/bge-reranker-v2-m3 could not be loaded locally (Windows pagefile /
available-RAM constraints on the dev machine -- confirmed via direct
CrossEncoder load attempts, not assumed) so this run executes on the
Sofja HPC login node instead (94GB free RAM at time of run vs ~1.7GB
free locally).

Everything below this point is unchanged from candidate_pool_v27.py except
the reranker model_name passed to KCCrossEncoderRerankerV27.

Original v26 docstring, still accurate for everything else in this file:

v26 = v25's RRF candidate architecture, with one scoped fix:
`has_profile_native_corroboration`'s third condition changed from
`support_component > 0.0` to `candidate_state in {"candidate",
"support_only"}`. The profile-native matcher's own `candidate_state`
already uses a `support_score >= 3.0` bar to decide `support_only`
(profile_native_matcher_v9.py, `_score_profile`); `support_component > 0.0`
was a separate, much lower, never-independently-justified threshold on the
exact same underlying quantity. This was found (not fixed) in the v25 gap-
closure session while explaining why ex_0026/ex_0039 still didn't change
after the identity-path and support-fallback fixes: a residual
`support_component` as low as 2.76 -- correctly below the 3.0 bar
`candidate_state` itself uses, hence downgraded to `reject` -- still
satisfied the old `>0.0` check, keeping the candidate typed-eligible
regardless. This change makes the boolean consistent with the bar
`candidate_state` already enforces, rather than inventing a separate one.
See v26_short_token_final_closure_report.md for the full re-run and diff.

Nothing else in this file differs from candidate_pool_v25.py's RRF/fusion
logic.

Implemented design choices from v19_architecture_research_report.md and the
follow-up Task 0 analysis (unchanged from v21):
- exact canonical matches short-circuit before fusion-based reranking
- profile-native is a fused family, not a gate-only side channel
- profile-native family uses student+tutor views for fusion
- combined profile-native view is retained for audit, but not fused as an
  equal-weight family member because Task 0 found it to be highly tutor-
  dominated and therefore redundant/noisy to fuse alongside tutor
- lexical family is BM25 + char-TFIDF fused by RRF
- dense retrieval remains an independent family
- downstream audit fields from v13-v18 remain visible where useful
- exact RRF ties are resolved by stable unit_id ordering via rrf_fusion_v2

No v9-v22 runtime files are modified.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any
import os
import warnings

from seg_eval.profile_native_matcher_v9 import ProfileNativeMatcherV9
from seg_eval.profile_text_preprocessor_v4 import preprocess_exchange_text
from .cross_encoder_reranker_v27 import KCCrossEncoderRerankerV27
from .cross_encoder_reranker_v13 import CROSS_ENCODER_AVAILABLE
from .dense_retrieval import DENSE_RETRIEVAL_AVAILABLE, KCDenseRetriever
from .hybrid_retrieval_v2 import HybridRetrievalIndex
from .profile_index import LoadedProfiles
from .rrf_fusion_v2 import (
    FAMILY_SUPPORT_RANK_CUTOFF_V21,
    RRF_K_V21,
    FusedRank,
    fused_rank_map,
)


PROFILE_FAMILY_VIEWS_V21 = ("student", "tutor")


def _to_dicts(items: list[Any]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if hasattr(item, "to_dict"):
            out.append(item.to_dict())
        elif hasattr(item, "__dataclass_fields__"):
            out.append(asdict(item))
        else:
            out.append(dict(item))
    return out


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = row.get("unit_id")
        if uid and uid not in out:
            out[uid] = row
    return out


def _full_rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        row["unit_id"]: idx
        for idx, row in enumerate(rows, start=1)
        if row.get("unit_id")
    }


def _state_rank(state: str | None) -> int:
    return {"candidate": 3, "support_only": 2, "reject": 1}.get(state or "", 0)


def _best_state(*states: str | None) -> str:
    return sorted([s or "absent" for s in states], key=_state_rank, reverse=True)[0]


def _top_ids(rows: list[dict[str, Any]], limit: int = 10) -> list[str]:
    out = []
    for row in rows[:limit]:
        uid = row.get("unit_id")
        if uid and uid not in out:
            out.append(uid)
    return out


def _merge_evidence(rows: list[dict[str, Any]], key: str, limit: int = 12) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        for ev in row.get(key) or []:
            marker = (ev.get("field"), ev.get("match_kind"), ev.get("matched_text"), ev.get("phrase"))
            if marker in seen:
                continue
            seen.add(marker)
            out.append(ev)
            if len(out) >= limit:
                return out
    return out


def _merge_fields(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for field, values in (row.get(key) or {}).items():
            for value in values:
                merged[field].add(str(value))
    return {k: sorted(v) for k, v in merged.items()}


def _has_profile_native_evidence(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return (
        bool(row.get("exact_canonical_match"))
        or float(row.get("identity_score") or 0.0) > 0.0
        or float(row.get("support_score") or 0.0) > 0.0
    )


def _positive_profile_rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    rank = 0
    for row in rows:
        if not _has_profile_native_evidence(row):
            continue
        uid = row.get("unit_id")
        if not uid or uid in out:
            continue
        rank += 1
        out[uid] = rank
    return out


def _family_support(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def _best_fused_candidate(fused: dict[str, FusedRank]) -> dict[str, int]:
    return {uid: row.rank for uid, row in fused.items()}


@dataclass
class CandidatePoolBuilderV28:
    profiles: LoadedProfiles
    internal_top_k: int = 60
    use_cross_encoder: bool = True
    use_dense_retrieval: bool = True
    rrf_k: int = RRF_K_V21
    family_support_rank_cutoff: int = FAMILY_SUPPORT_RANK_CUTOFF_V21

    def __post_init__(self) -> None:
        self.matcher = ProfileNativeMatcherV9(self.profiles.profiles)
        self.hybrid = HybridRetrievalIndex(self.profiles.profiles)

        self.profiles_by_id: dict[str, dict[str, Any]] = {
            p["unit_id"]: p for p in self.profiles.profiles
        }
        self.profile_count = len(self.profiles.profiles)

        self.reranker: KCCrossEncoderRerankerV27 | None = None
        if self.use_cross_encoder:
            if CROSS_ENCODER_AVAILABLE:
                try:
                    reranker_model = os.environ.get("SEG_EVAL_RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
                    self.reranker = KCCrossEncoderRerankerV27(model_name=reranker_model)
                except Exception as exc:
                    warnings.warn(
                        "Cross-encoder reranking is disabled after initialization "
                        f"failed: {exc}",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "sentence-transformers is not installed. Cross-encoder "
                    "reranking is disabled; pipeline falls back to fused ranking.",
                    stacklevel=2,
                )

        self.dense: KCDenseRetriever | None = None
        if self.use_dense_retrieval:
            if DENSE_RETRIEVAL_AVAILABLE:
                try:
                    dense_model = os.environ.get("SEG_EVAL_DENSE_MODEL_PATH", "BAAI/bge-small-en-v1.5")
                    self.dense = KCDenseRetriever(self.profiles.profiles, model_name=dense_model)
                except Exception as exc:
                    warnings.warn(
                        "Dense retrieval is disabled after initialization "
                        f"failed: {exc}",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "sentence-transformers or torch is not installed. Dense "
                    "retrieval is disabled; pipeline falls back to profile+lexical fusion.",
                    stacklevel=2,
                )

    def build_for_exchange(self, exchange: Any, top_k: int = 25) -> dict[str, Any]:
        student_pre = preprocess_exchange_text(getattr(exchange, "student_text", "") or "")
        tutor_pre = preprocess_exchange_text(getattr(exchange, "tutor_text", "") or "")
        combined_pre = preprocess_exchange_text(getattr(exchange, "exchange_text", "") or "")

        full_profile_depth = self.profile_count
        dense_depth = full_profile_depth

        student_rows = _to_dicts(self.matcher.match_exchange(student_pre["match_text"], top_k=full_profile_depth))
        tutor_rows = _to_dicts(self.matcher.match_exchange(tutor_pre["match_text"], top_k=full_profile_depth))
        combined_rows = _to_dicts(self.matcher.match_exchange(combined_pre["match_text"], top_k=full_profile_depth))

        student_full_ranks = _full_rank_map(student_rows)
        tutor_full_ranks = _full_rank_map(tutor_rows)
        combined_full_ranks = _full_rank_map(combined_rows)

        hybrid_results = self.hybrid.search_views(
            student_text=student_pre["match_text"],
            tutor_text=tutor_pre["match_text"],
            combined_text=combined_pre["match_text"],
            top_k=full_profile_depth,
        )
        hybrid_by_id = _by_id(hybrid_results["hybrid_candidates"])

        dense_results = (
            self.dense.score_query(combined_pre["match_text"], top_k=dense_depth)
            if self.dense is not None
            else []
        )
        dense_by_id = _by_id(dense_results)

        views = {
            "student": _by_id(student_rows),
            "tutor": _by_id(tutor_rows),
            "combined": _by_id(combined_rows),
        }

        profile_student_rank_map = _positive_profile_rank_map(student_rows)
        profile_tutor_rank_map = _positive_profile_rank_map(tutor_rows)
        profile_combined_rank_map = _positive_profile_rank_map(combined_rows)
        profile_family_fused = fused_rank_map(
            {
                "profile_student": profile_student_rank_map,
                "profile_tutor": profile_tutor_rank_map,
            },
            k=self.rrf_k,
        )
        profile_family_rank_map = _best_fused_candidate(profile_family_fused)

        bm25_rank_map = {
            row["unit_id"]: int(row["bm25_rank"])
            for row in hybrid_results["hybrid_candidates"]
            if row.get("unit_id") and row.get("bm25_rank") is not None
        }
        char_rank_map = {
            row["unit_id"]: int(row["char_rank"])
            for row in hybrid_results["hybrid_candidates"]
            if row.get("unit_id") and row.get("char_rank") is not None
        }
        lexical_family_fused = fused_rank_map(
            {
                "bm25": bm25_rank_map,
                "char_tfidf": char_rank_map,
            },
            k=self.rrf_k,
        )
        lexical_family_rank_map = _best_fused_candidate(lexical_family_fused)

        dense_rank_map = {
            row["unit_id"]: int(row["dense_rank"])
            for row in dense_results
            if row.get("unit_id") and row.get("dense_rank") is not None
        }

        outer_family_fused = fused_rank_map(
            {
                "profile_native": profile_family_rank_map,
                "lexical": lexical_family_rank_map,
                "dense": dense_rank_map,
            },
            k=self.rrf_k,
        )
        outer_rrf_rank_map = _best_fused_candidate(outer_family_fused)

        profile_candidate_ids = set(profile_student_rank_map) | set(profile_tutor_rank_map) | set(profile_combined_rank_map)
        lexical_candidate_ids = set(hybrid_by_id)
        dense_candidate_ids = set(dense_rank_map)
        candidate_ids = profile_candidate_ids | lexical_candidate_ids | dense_candidate_ids

        candidates: list[dict[str, Any]] = []
        for uid in sorted(candidate_ids):
            profile = self.profiles.profile(uid)
            if not profile:
                continue

            s = views["student"].get(uid)
            t = views["tutor"].get(uid)
            c = views["combined"].get(uid)
            present = [row for row in [s, t, c] if row]
            h = hybrid_by_id.get(uid) or {}
            dense_hit = dense_by_id.get(uid) or {}

            if present:
                rep = sorted(
                    present,
                    key=lambda row: (
                        _state_rank(row.get("candidate_state")),
                        float(row.get("score") or 0.0),
                        float(row.get("identity_score") or 0.0),
                        float(row.get("support_score") or 0.0),
                    ),
                    reverse=True,
                )[0]
                base_state = _best_state(
                    s.get("candidate_state") if s else None,
                    t.get("candidate_state") if t else None,
                    c.get("candidate_state") if c else None,
                )
            else:
                rep = {
                    "unit_id": uid,
                    "canonical_name": profile.get("canonical_name"),
                    "topic_path": profile.get("topic_path") or [],
                    "candidate_state": "lexical_only",
                    "score": 0.0,
                    "identity_score": 0.0,
                    "support_score": 0.0,
                    "negative_score": 0.0,
                    "exact_canonical_match": False,
                }
                base_state = "lexical_only"

            student_identity = float(s.get("identity_score") or 0.0) if s else 0.0
            tutor_identity = float(t.get("identity_score") or 0.0) if t else 0.0
            combined_identity = float(c.get("identity_score") or 0.0) if c else 0.0
            student_support = float(s.get("support_score") or 0.0) if s else 0.0
            tutor_support = float(t.get("support_score") or 0.0) if t else 0.0
            combined_support = float(c.get("support_score") or 0.0) if c else 0.0
            student_negative = float(s.get("negative_score") or 0.0) if s else 0.0
            tutor_negative = float(t.get("negative_score") or 0.0) if t else 0.0
            combined_negative = float(c.get("negative_score") or 0.0) if c else 0.0
            exact = any(bool(row.get("exact_canonical_match")) for row in present)

            # Preserve legacy-style audit fields for regression tracing.
            intent_component = max(student_identity * 1.35, combined_identity)
            coverage_component = min(4.0, tutor_support * 0.75 + tutor_identity * 0.45)
            support_component = min(4.0, max(combined_support, student_support * 0.7 + tutor_support * 0.45))
            identity_component = max(student_identity, tutor_identity * 0.7, combined_identity * 0.85)
            negative_component = max(student_negative, tutor_negative, combined_negative)
            exact_bonus = 4.0 if exact else 0.0
            bm25_score = float(h.get("bm25_score") or 0.0)
            char_score = float(h.get("char_score") or 0.0)
            dense_score = float(dense_hit.get("dense_score") or 0.0) if self.dense is not None else 0.0
            dense_rank = dense_hit.get("dense_rank") if self.dense is not None else None
            hybrid_retrieval_score = float(h.get("hybrid_retrieval_score") or 0.0)
            hybrid_agreement_count = int(h.get("hybrid_agreement_count") or 0)
            profile_signal = 1 if _has_profile_native_evidence(rep) else 0
            signal_agreement_count = profile_signal + (1 if bm25_score >= 0.20 else 0) + (1 if char_score >= 0.12 else 0)
            lexical_component = min(4.0, (2.4 * bm25_score) + (1.6 * char_score))
            dense_component = min(4.0, 3.2 * dense_score) if self.dense is not None else 0.0
            pre_dense_composite = (
                intent_component
                + coverage_component
                + support_component
                + identity_component
                + exact_bonus
                + lexical_component
                - negative_component
            )
            legacy_composite = pre_dense_composite + dense_component

            branch = self.profiles.branch(uid)
            leaf = self.profiles.leaf(uid)
            tutor_only_noncanonical_identity = (
                not exact and student_identity <= 0.0 and tutor_identity > 0.0 and combined_identity > 0.0
            )

            has_profile_native_corroboration = (
                exact
                or identity_component > 0.0
                or base_state in {"candidate", "support_only"}
            )
            has_lexical_support = _family_support(lexical_family_rank_map.get(uid), self.family_support_rank_cutoff)
            has_dense_support = _family_support(dense_rank_map.get(uid), self.family_support_rank_cutoff)
            source_family_supports = {
                "profile_native": has_profile_native_corroboration,
                "lexical": has_lexical_support,
                "dense": has_dense_support,
            }
            source_family_count = sum(1 for supported in source_family_supports.values() if supported)
            has_nonlexical_family_support = has_profile_native_corroboration or has_dense_support
            eligible_for_dominant = bool(
                exact
                or has_profile_native_corroboration
                or (source_family_count >= 2 and has_nonlexical_family_support)
            )

            evidence_sources: list[str] = []
            if bm25_score > 0.0:
                evidence_sources.append("bm25")
            if char_score > 0.0:
                evidence_sources.append("char_tfidf")
            if has_profile_native_corroboration:
                evidence_sources.append("profile_native")
            if has_dense_support:
                evidence_sources.append("dense_retrieval")

            profile_fused_row = profile_family_fused.get(uid)
            lexical_fused_row = lexical_family_fused.get(uid)
            outer_fused_row = outer_family_fused.get(uid)

            candidates.append(
                {
                    "unit_id": uid,
                    "canonical_name": rep.get("canonical_name") or profile.get("canonical_name"),
                    "topic_path": rep.get("topic_path") or profile.get("topic_path") or [],
                    "branch": branch,
                    "leaf_topic": leaf,
                    "candidate_state": base_state,
                    "composite_score": round(float(outer_fused_row.rrf_score if outer_fused_row else 0.0), 6),
                    "legacy_composite_score": round(float(legacy_composite), 6),
                    "pre_dense_composite_score": round(float(pre_dense_composite), 6),
                    "intent_component": round(intent_component, 6),
                    "coverage_component": round(coverage_component, 6),
                    "support_component": round(support_component, 6),
                    "identity_component": round(identity_component, 6),
                    "negative_component": round(negative_component, 6),
                    "lexical_component": round(lexical_component, 6),
                    "dense_component": round(dense_component, 6),
                    "bm25_score": round(bm25_score, 6),
                    "char_score": round(char_score, 6),
                    "dense_score": round(dense_score, 6),
                    "dense_rank": dense_rank,
                    "dense_family_rank": dense_rank,
                    "hybrid_retrieval_score": round(hybrid_retrieval_score, 6),
                    "hybrid_agreement_count": hybrid_agreement_count,
                    "signal_agreement_count": signal_agreement_count,
                    "evidence_sources": evidence_sources,
                    "matched_bm25_terms": list(h.get("matched_bm25_terms") or []),
                    "bm25_rank": h.get("bm25_rank"),
                    "char_rank": h.get("char_rank"),
                    "hybrid_role_scores": h.get("hybrid_role_scores") or {},
                    "exact_canonical_match": exact,
                    "exact_override_candidate": exact,
                    "tutor_only_noncanonical_identity": tutor_only_noncanonical_identity,
                    "cross_encoder_score": None,
                    "ce_reranked_composite": None,
                    "final_rank_score": round(float(outer_fused_row.rrf_score if outer_fused_row else 0.0), 6),
                    "profile_family_rank": profile_family_rank_map.get(uid),
                    "lexical_family_rank": lexical_family_rank_map.get(uid),
                    "outer_rrf_rank": outer_rrf_rank_map.get(uid),
                    "profile_family_rrf_score": round(float(profile_fused_row.rrf_score if profile_fused_row else 0.0), 6),
                    "lexical_family_rrf_score": round(float(lexical_fused_row.rrf_score if lexical_fused_row else 0.0), 6),
                    "outer_rrf_score": round(float(outer_fused_row.rrf_score if outer_fused_row else 0.0), 6),
                    "family_source_counts": {
                        "profile_native": profile_fused_row.source_count if profile_fused_row else 0,
                        "lexical": lexical_fused_row.source_count if lexical_fused_row else 0,
                        "outer": outer_fused_row.source_count if outer_fused_row else 0,
                    },
                    "family_source_ranks": {
                        "profile_native": dict(profile_fused_row.source_ranks) if profile_fused_row else {},
                        "lexical": dict(lexical_fused_row.source_ranks) if lexical_fused_row else {},
                        "outer": dict(outer_fused_row.source_ranks) if outer_fused_row else {},
                    },
                    "family_support_flags": dict(source_family_supports),
                    "source_family_count": source_family_count,
                    "has_profile_native_corroboration": has_profile_native_corroboration,
                    "has_dense_support": has_dense_support,
                    "has_lexical_support": has_lexical_support,
                    "has_nonlexical_family_support": has_nonlexical_family_support,
                    "eligible_for_dominant": eligible_for_dominant,
                    "eligible_for_broad": eligible_for_dominant,
                    "role_states": {
                        "student": s.get("candidate_state") if s else "absent",
                        "tutor": t.get("candidate_state") if t else "absent",
                        "combined": c.get("candidate_state") if c else "absent",
                    },
                    "role_scores": {
                        "student": {
                            "identity": student_identity,
                            "support": student_support,
                            "negative": student_negative,
                            "score": float(s.get("score") or 0.0) if s else 0.0,
                        },
                        "tutor": {
                            "identity": tutor_identity,
                            "support": tutor_support,
                            "negative": tutor_negative,
                            "score": float(t.get("score") or 0.0) if t else 0.0,
                        },
                        "combined": {
                            "identity": combined_identity,
                            "support": combined_support,
                            "negative": combined_negative,
                            "score": float(c.get("score") or 0.0) if c else 0.0,
                        },
                    },
                    "identity_evidence": _merge_evidence(present, "identity_evidence"),
                    "support_evidence": _merge_evidence(present, "support_evidence"),
                    "negative_evidence": _merge_evidence(present, "negative_evidence"),
                    "matched_fields": _merge_fields(present, "matched_fields"),
                    "negative_fields": _merge_fields(present, "negative_fields"),
                    "source_view_ranks": {
                        "student": student_full_ranks.get(uid),
                        "tutor": tutor_full_ranks.get(uid),
                        "combined": combined_full_ranks.get(uid),
                        "bm25": h.get("bm25_rank"),
                        "char_tfidf": h.get("char_rank"),
                        "dense": dense_rank,
                    },
                    "profile_positive_ranks": {
                        "student": profile_student_rank_map.get(uid),
                        "tutor": profile_tutor_rank_map.get(uid),
                        "combined": profile_combined_rank_map.get(uid),
                    },
                }
            )

        candidates.sort(
            key=lambda row: (
                -float(row.get("outer_rrf_score") or 0.0),
                -int(row.get("source_family_count") or 0),
                -float(row.get("profile_family_rrf_score") or 0.0),
                str(row.get("unit_id") or ""),
            )
        )

        # v27/v28: exact_override_active is still computed and recorded (audit /
        # reporting fields below), but it no longer skips reranking. An exact
        # canonical-name match still carries real weight -- it drives
        # identity_component/composite_score upstream, which feeds comp_norm
        # in the blend below -- but it is no longer immune to a cross-encoder
        # correction when the name match is a passing mention rather than
        # the exchange's actual topic. See module docstring for the evidence.
        exact_override_active = any(bool(c.get("exact_override_candidate")) for c in candidates)

        rerank_candidates = list(candidates)
        if self.reranker is not None and rerank_candidates:
            exchange_text = (
                str(getattr(exchange, "student_text", "") or "")
                + " "
                + str(getattr(exchange, "tutor_text", "") or "")
            ).strip()
            rerank_candidates = self.reranker.rerank(
                exchange_text=exchange_text,
                candidates=rerank_candidates,
                profiles_by_id=self.profiles_by_id,
                top_k=len(rerank_candidates),
            )
        else:
            for c in rerank_candidates:
                c["ce_reranked_composite"] = float(c.get("composite_score") or 0.0)

        for c in rerank_candidates:
            c["final_rank_score"] = round(
                float(c.get("ce_reranked_composite") if c.get("ce_reranked_composite") is not None else c.get("composite_score") or 0.0),
                6,
            )
        candidates = sorted(
            rerank_candidates,
            key=lambda c: (
                -float(c.get("final_rank_score") or 0.0),
                str(c.get("unit_id") or ""),
            ),
        )

        for idx, row in enumerate(candidates, start=1):
            row["final_pool_rank"] = idx

        final_candidates = candidates[:top_k]

        total_score = sum(max(0.0, float(c.get("final_rank_score") or 0.0)) for c in final_candidates) or 1.0
        branch_scores: dict[str, float] = defaultdict(float)
        for cand in final_candidates:
            branch_scores[cand["branch"]] += max(0.0, float(cand.get("final_rank_score") or 0.0))
        branch_items = sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)
        top_branch_share = round(branch_items[0][1] / total_score, 6) if branch_items else 0.0
        active_branch_count = sum(1 for _, score in branch_items if score / total_score >= 0.12)

        return {
            "dialogue_id": getattr(exchange, "dialogue_id"),
            "exchange_id": getattr(exchange, "exchange_id"),
            "exchange_index": getattr(exchange, "exchange_index"),
            "student_text_raw": getattr(exchange, "student_text", "") or "",
            "tutor_text_raw": getattr(exchange, "tutor_text", "") or "",
            "exchange_text_raw": getattr(exchange, "exchange_text", "") or "",
            "student_preprocessing": student_pre,
            "tutor_preprocessing": tutor_pre,
            "combined_preprocessing": combined_pre,
            "candidate_count": len(final_candidates),
            "full_candidate_count": len(candidates),
            "top_candidate_ids": [c["unit_id"] for c in final_candidates[:10]],
            "cross_encoder_active": self.reranker is not None,
            "dense_retrieval_active": self.dense is not None,
            "exact_override_active": exact_override_active,
            "rrf_k": self.rrf_k,
            "family_support_rank_cutoff": self.family_support_rank_cutoff,
            "profile_family_views_used": list(PROFILE_FAMILY_VIEWS_V21),
            "branch_scores": dict(branch_items[:10]),
            "top_branch_share": top_branch_share,
            "active_branch_count": active_branch_count,
            "broad_multibranch_signal": active_branch_count >= 3 and top_branch_share < 0.45,
            "hybrid_bm25_top_ids": hybrid_results["bm25_top_ids"],
            "hybrid_char_top_ids": hybrid_results["char_top_ids"],
            "dense_top_ids": [r["unit_id"] for r in dense_results[:10]],
            "candidates": final_candidates,
        }
