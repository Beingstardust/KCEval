"""pipeline_v35.py

v35 = v34a's ungated tie-breaker plus three measured changes, each with its
own evidence trail:

1. FIXED OPTION SCORING. v33/v34a scored a token position the model was not
   using: only 4% of probability mass sat on the option letters (the model
   spent its first token restating "Answer"), and 0% on a thinking model.
   The cue now lives in an assistant-turn PREFILL. Measured letter mass
   0.0398 -> 1.0000. See option_scoring.HFOptionScorer.

2. PRECEDING-EXCHANGE CONTEXT for the judge. Only visible once (1) was
   fixed -- under the broken scorer this signal was buried and looked like
   nothing. Tie-breaker pick accuracy 72.5% -> 78.4% pooled, improving BOTH
   dialogues, with multi_kc improving on both (dm1 66.7->88.9%, dm2
   38.9->50.0%).

3. OVERRIDE GUARD (context_resolver_v35): pass2_broad_multibranch may no
   longer overrule a tie-breaker-promoted candidate. That branch overrides
   3 times in 102 exchanges and has never once been right.

Retrieval is unchanged: CandidatePoolBuilderV28 verbatim, unedited library.
The gold primary already sits in the candidate pool 98% of the time, so
nothing here targets retrieval.

No v9-v34 runtime file is modified.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seg_eval.dialogue_parser import parse_dialogue
from seg_eval.matching_profiles import read_jsonl
from .artifact_semantics_v9c import annotate_contactless_artifacts
from .artifacts import write_json, write_jsonl, write_report
from .abstention_gate_v24 import annotate_abstention_v24
from .candidate_pool_v28 import CandidatePoolBuilderV28
from .context_resolver_v35 import resolve_pass2_v35
from .exchange_units import build_strict_st_exchanges
from .local_assigner_v13 import assign_pass1_v13
from .meta_exclusion_v15 import exclude_meta_from_segmentation
from .multilabel_kc_v17 import annotate_resolved_kc_ids_v17
from .pre_fusion_non_kc_gate_v1 import (
    classify_pre_fusion_non_kc,
    make_pre_fusion_non_kc_pass1,
    make_pre_fusion_non_kc_pool,
)
from .profile_index import LoadedProfiles
from .segment_multilabel_v16 import annotate_segment_kc_ids_v16
from .semantic_flags_v15 import annotate_low_confidence_v15
from .segmenter_v3 import build_contactless_segments


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)


def run_contactless_pipeline_v35(
    profiles_path: str | Path,
    dialogues_jsonl: str | Path,
    out_dir: str | Path,
    top_k: int = 25,
    internal_top_k: int = 60,
    use_cross_encoder: bool = True,
    use_dense_retrieval: bool = True,
    tie_breaker_model_path: str | None = None,
    tie_breaker_top_k: int = 5,
    tie_breaker_max_pipeline_margin: float = 0.25,
    tie_breaker_min_probability: float = 0.5,
    tie_breaker_min_probability_margin: float = 0.15,
    use_prior_context: bool = True,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()

    profiles = LoadedProfiles.from_path(str(profiles_path))
    builder = CandidatePoolBuilderV28(
        profiles=profiles,
        internal_top_k=internal_top_k,
        use_cross_encoder=use_cross_encoder,
        use_dense_retrieval=use_dense_retrieval,
    )
    profiles_by_id = {p["unit_id"]: p for p in profiles.profiles}

    scorer = None
    apply_tie_breaker_to_pool = None
    if tie_breaker_model_path:
        # Imported lazily so that v33 with no tie-breaker has exactly v28's
        # dependency surface. Failure to load is raised, never swallowed --
        # v29 silently degrading to v28 behaviour for a full 55-minute job is
        # precisely the failure mode this refuses to repeat.
        from seg_eval.tie_breaker.option_scoring import HFOptionScorer
        from seg_eval.tie_breaker.pool_application import apply_tie_breaker_to_pool as _apply

        scorer = HFOptionScorer(tie_breaker_model_path)
        apply_tie_breaker_to_pool = _apply
        print(f"tie_breaker_active model={tie_breaker_model_path} device={scorer.device}")

    dialogue_rows = read_jsonl(dialogues_jsonl)

    all_turns, all_exchanges, all_pools = [], [], []
    all_pass1, all_contactless, all_novelty = [], [], []
    all_segments, all_review = [], []
    all_excluded_meta_ids: list[str] = []
    pre_fusion_gate_hits = 0
    tie_breaker_outcomes: list[dict[str, Any]] = []

    for dialogue_obj in dialogue_rows:
        turns = parse_dialogue(dialogue_obj)
        exchanges = build_strict_st_exchanges(turns)

        pools: list[dict[str, Any]] = []
        pass1: list[dict[str, Any]] = []

        for exchange in exchanges:
            gate = classify_pre_fusion_non_kc(exchange)
            if gate:
                pre_fusion_gate_hits += 1
                pool = make_pre_fusion_non_kc_pool(exchange, gate)
                row = make_pre_fusion_non_kc_pass1(exchange, pool, gate["reason"])
            else:
                pool = builder.build_for_exchange(exchange, top_k=top_k)
                if apply_tie_breaker_to_pool is not None:
                    prior_text = None
                    if use_prior_context and pools:
                        pv = pools[-1]
                        prior_text = (
                            "Student: " + (pv.get('student_text_raw') or '').strip()
                            + "\n" + "Tutor: " + (pv.get('tutor_text_raw') or '').strip()
                        )
                    outcome = apply_tie_breaker_to_pool(
                        pool=pool,
                        scorer=scorer,
                        profiles_by_id=profiles_by_id,
                        top_k=tie_breaker_top_k,
                        max_pipeline_margin=tie_breaker_max_pipeline_margin,
                        min_probability=tie_breaker_min_probability,
                        min_probability_margin=tie_breaker_min_probability_margin,
                        prior_exchange_text=prior_text,
                    )
                    tie_breaker_outcomes.append(asdict(outcome))
                    if outcome.changed:
                        print(
                            f"tie_breaker CHANGED {outcome.exchange_id}: "
                            f"{outcome.original_top1} -> {outcome.chosen} (p={outcome.top_probability:.3f})"
                        )
                # pass 1 runs AFTER any reordering, so every downstream stage
                # sees one consistent ranking.
                row = assign_pass1_v13(exchange, pool)
            pools.append(pool)
            pass1.append(row)

        contactless, novelty = resolve_pass2_v35(pass1, pools)
        contactless = annotate_low_confidence_v15(contactless, pools)
        contactless = annotate_resolved_kc_ids_v17(contactless, pools)
        contactless = annotate_abstention_v24(contactless, pools)

        seg_exchanges, seg_assignments, excluded_ids = exclude_meta_from_segmentation(exchanges, contactless)
        all_excluded_meta_ids.extend(excluded_ids)

        segments, review = build_contactless_segments(seg_exchanges, seg_assignments)
        contactless, segments = annotate_contactless_artifacts(contactless, segments)

        assignments_by_id = {row["exchange_id"]: row for row in contactless}
        segments = annotate_segment_kc_ids_v16(segments, assignments_by_id)

        all_turns.extend([_obj_to_dict(t) for t in turns])
        all_exchanges.extend([_obj_to_dict(e) for e in exchanges])
        all_pools.extend(pools)
        all_pass1.extend(pass1)
        all_contactless.extend(contactless)
        all_novelty.extend(novelty)
        all_segments.extend(segments)
        all_review.extend(review)

    label_counts = Counter(r.get("final_label") for r in all_contactless)
    pass1_counts = Counter(r.get("pass1_label") for r in all_pass1)
    action_counts = Counter(r.get("resolution_action") for r in all_contactless)
    band_counts = Counter(r.get("confidence_band") for r in all_contactless)
    low_confidence_count = sum(1 for r in all_contactless if r.get("low_confidence_no_strong_match"))
    secondary_kc_counts = Counter(len(r.get("resolved_kc_ids") or []) - 1 for r in all_contactless if r.get("resolved_kc_ids"))
    exchanges_with_secondaries = sum(1 for r in all_contactless if len(r.get("resolved_kc_ids") or []) > 1)
    exact_override_count = sum(1 for pool in all_pools if pool.get("exact_override_active"))
    abstained_count = sum(1 for r in all_contactless if r.get("resolution_status") == "abstained_low_confidence")

    summary = {
        "pipeline": "contactless_two_pass_v35_fixed_scorer_context_override_guard",
        "tie_breaker_model_path": tie_breaker_model_path,
        "tie_breaker_active": scorer is not None,
        "tie_breaker_top_k": tie_breaker_top_k,
        "tie_breaker_max_pipeline_margin": tie_breaker_max_pipeline_margin,
        "tie_breaker_min_probability": tie_breaker_min_probability,
        "tie_breaker_min_probability_margin": tie_breaker_min_probability_margin,
        "tie_breaker_uses_prior_context": use_prior_context,
        "tie_breaker_gated_in_count": sum(1 for o in tie_breaker_outcomes if o["gated_in"]),
        "tie_breaker_changed_count": sum(1 for o in tie_breaker_outcomes if o["changed"]),
        "abstained_low_confidence_count": abstained_count,
        "dialogue_count": len(dialogue_rows),
        "turn_count": len(all_turns),
        "exchange_count": len(all_exchanges),
        "profile_count": len(profiles.profiles),
        "segment_count": len(all_segments),
        "excluded_meta_exchange_count": len(all_excluded_meta_ids),
        "excluded_meta_exchange_ids": all_excluded_meta_ids,
        "low_confidence_no_strong_match_count": low_confidence_count,
        "exchanges_with_secondary_kcs": exchanges_with_secondaries,
        "secondary_kc_count_distribution": {str(k): v for k, v in sorted(secondary_kc_counts.items())},
        "missing_kc_review_count": len(all_review),
        "cross_encoder_active": builder.reranker is not None,
        "dense_retrieval_active": builder.dense is not None,
        "exact_override_exchange_count": exact_override_count,
        "pre_fusion_non_kc_gate_hits": pre_fusion_gate_hits,
        "rrf_k": builder.rrf_k,
        "family_support_rank_cutoff": builder.family_support_rank_cutoff,
        "profile_family_views_used": ["student", "tutor"],
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    for k, v in sorted(pass1_counts.items()):
        summary[f"pass1_{k}"] = v
    for k, v in sorted(label_counts.items()):
        summary[f"final_{k}"] = v
    for k, v in sorted(action_counts.items()):
        summary[f"action_{k}"] = v
    for k, v in sorted(band_counts.items()):
        summary[f"band_{k}"] = v

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_json(
        out / "run_manifest.json",
        {
            "pipeline": summary["pipeline"],
            "profiles_path": str(profiles_path),
            "dialogues_jsonl": str(dialogues_jsonl),
            "parameters": {
                "top_k": top_k,
                "internal_top_k": internal_top_k,
                "cross_encoder_active": builder.reranker is not None,
                "dense_retrieval_active": builder.dense is not None,
                "tie_breaker_model_path": tie_breaker_model_path,
                "tie_breaker_top_k": tie_breaker_top_k,
                "tie_breaker_max_pipeline_margin": tie_breaker_max_pipeline_margin,
                "tie_breaker_min_probability": tie_breaker_min_probability,
                "tie_breaker_min_probability_margin": tie_breaker_min_probability_margin,
                "strict_st_exchange_construction": True,
                "domain_specific_terms_hardcoded": False,
                "rrf_k": builder.rrf_k,
                "family_support_rank_cutoff": builder.family_support_rank_cutoff,
                "profile_family_views_used": ["student", "tutor"],
            },
            "summary": summary,
        },
    )
    write_json(out / "tie_breaker_outcomes.json", tie_breaker_outcomes)
    write_jsonl(out / "turns.jsonl", all_turns)
    write_jsonl(out / "exchanges.jsonl", all_exchanges)
    write_jsonl(out / "candidate_pool.jsonl", all_pools)
    write_jsonl(out / "pass1_local_assignments.jsonl", all_pass1)
    write_jsonl(out / "contactless_exchange_assignments.jsonl", all_contactless)
    write_jsonl(out / "novelty_scores.jsonl", all_novelty)
    write_jsonl(out / "contactless_segments.jsonl", all_segments)
    write_jsonl(out / "review_queue_missing_kc_only.jsonl", all_review)
    write_jsonl(out / "exchange_assignments.jsonl", all_contactless)
    write_jsonl(out / "segments.jsonl", all_segments)
    write_jsonl(out / "review_queue.jsonl", all_review)
    write_report(out / "segment_grounding_report.md", summary, all_segments)
    return {"summary": summary, "out_dir": str(out)}
