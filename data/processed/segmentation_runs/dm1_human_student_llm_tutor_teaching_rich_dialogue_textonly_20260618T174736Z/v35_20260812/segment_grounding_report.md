# Contactless KC segmentation report

## Summary

- `pipeline`: `contactless_two_pass_v35_fixed_scorer_context_override_guard`
- `tie_breaker_model_path`: `models/Qwen3-4B-Instruct-2507`
- `tie_breaker_active`: `True`
- `tie_breaker_top_k`: `5`
- `tie_breaker_max_pipeline_margin`: `999.0`
- `tie_breaker_min_probability`: `0.5`
- `tie_breaker_min_probability_margin`: `0.5`
- `tie_breaker_uses_prior_context`: `True`
- `tie_breaker_gated_in_count`: `60`
- `tie_breaker_changed_count`: `15`
- `abstained_low_confidence_count`: `20`
- `dialogue_count`: `1`
- `turn_count`: `126`
- `exchange_count`: `63`
- `profile_count`: `159`
- `segment_count`: `35`
- `excluded_meta_exchange_count`: `1`
- `excluded_meta_exchange_ids`: `['dm1_human_student_llm_tutor_teaching_rich_dialogue_textonly_20260618T174736Z::ex_0000']`
- `low_confidence_no_strong_match_count`: `3`
- `exchanges_with_secondary_kcs`: `60`
- `secondary_kc_count_distribution`: `{'3': 60}`
- `missing_kc_review_count`: `0`
- `cross_encoder_active`: `True`
- `dense_retrieval_active`: `True`
- `exact_override_exchange_count`: `24`
- `pre_fusion_non_kc_gate_hits`: `3`
- `rrf_k`: `11`
- `family_support_rank_cutoff`: `25`
- `profile_family_views_used`: `['student', 'tutor']`
- `started_utc`: `2026-08-12T18:06:27.926310+00:00`
- `finished_utc`: `2026-08-12T18:46:17.911194+00:00`
- `pass1_AUTO_HIGH_KC`: `14`
- `pass1_AUTO_MEDIUM_KC`: `23`
- `pass1_NEEDS_PASS2`: `23`
- `pass1_NON_KC`: `3`
- `final_AUTO_HIGH_KC`: `14`
- `final_AUTO_LOW_KC`: `23`
- `final_AUTO_MEDIUM_KC`: `23`
- `final_NON_KC`: `3`
- `action_pass1_closed_set`: `37`
- `action_pass1_protected_from_continuation_relative_evidence_v19`: `6`
- `action_pass2_branch_guided_best_candidate`: `1`
- `action_pass2_broad_multibranch_closed_set`: `12`
- `action_pass2_same_neighbour_rescue`: `1`
- `action_pass2_tie_breaker_promotion_retained`: `3`
- `action_pending_pass2`: `3`
- `band_high`: `14`
- `band_low`: `23`
- `band_medium`: `23`
- `band_none`: `3`

## Segments

| # | Turns | Exchanges | Dominant KC | Confidence | Labels | Review |
|---:|---|---:|---|---|---|---|
| 0 | 2-7 | 3 | `KC_CLU_CORE_004` Classification vs. Clustering (Distinction) | low | AUTO_LOW_KC | False |
| 1 | 8-9 | 1 | `KC_CLF_UND_003` Training Set vs. Test Set Split | low | AUTO_LOW_KC | False |
| 2 | 10-13 | 2 | `KC_CLF_UND_007` Attribute/Variable Types (Numerical, Categorical, Ordinal) | low | AUTO_LOW_KC | False |
| 3 | 14-15 | 1 | `KC_CLF_NB_001` Bayes' Theorem | low | AUTO_LOW_KC | False |
| 4 | 16-17 | 1 | `KC_EVAL_COMP_004` Friedman Test | medium | AUTO_MEDIUM_KC | False |
| 5 | 18-19 | 1 | `KC_CLF_NB_003` Conditional Probability (Likelihood) | low | AUTO_LOW_KC | False |
| 6 | 20-21 | 1 | `KC_CLF_NB_006` NB Classification Phase | medium | AUTO_MEDIUM_KC | False |
| 7 | 22-23 | 1 | `KC_CLF_NB_004` Naive Independence Assumption | high | AUTO_HIGH_KC | False |
| 8 | 24-25 | 1 | `KC_CLF_NB_002` Prior Probability | medium | AUTO_MEDIUM_KC | False |
| 9 | 26-27 | 1 | `KC_CLF_NB_003` Conditional Probability (Likelihood) | medium | AUTO_MEDIUM_KC | False |
| 10 | 28-37 | 5 | `KC_CLF_NB_007` Zero-Frequency Problem | high,low,medium | AUTO_HIGH_KC,AUTO_LOW_KC,AUTO_MEDIUM_KC | False |
| 11 | 38-41 | 2 | `KC_CLF_DT_006` Information Gain | high | AUTO_HIGH_KC | False |
| 12 | 42-43 | 1 | `KC_CLF_DT_008` Gain Ratio | high | AUTO_HIGH_KC | False |
| 13 | 44-47 | 2 | `KC_CLF_DT_005` Entropy (Node) | medium | AUTO_MEDIUM_KC | False |
| 14 | 48-49 | 1 | `KC_CLF_DT_008` Gain Ratio | high | AUTO_HIGH_KC | False |
| 15 | 50-55 | 3 | `KC_CLF_PRUNE_003` Pessimistic Error Estimate | low,medium | AUTO_LOW_KC,AUTO_MEDIUM_KC | False |
| 16 | 56-57 | 1 | `KC_CLF_PRUNE_004` Reduced Error Pruning | high | AUTO_HIGH_KC | False |
| 17 | 58-59 | 1 | `KC_EVAL_IMBAL_001` Class Imbalance Problem | low | AUTO_LOW_KC | False |
| 18 | 60-61 | 1 | `KC_CLF_UND_006` Target Attribute | medium | AUTO_MEDIUM_KC | False |
| 19 | 62-63 | 1 | `KC_EVAL_BASIC_001` Confusion Matrix | high | AUTO_HIGH_KC | False |
| 20 | 64-65 | 1 | `KC_EVAL_BASIC_003` Precision | high | AUTO_HIGH_KC | False |
| 21 | 66-67 | 1 | `KC_EVAL_BASIC_004` Recall (Sensitivity) | medium | AUTO_MEDIUM_KC | False |
| 22 | 68-69 | 1 | `KC_EVAL_BASIC_006` F-Measure | medium | AUTO_MEDIUM_KC | False |
| 23 | 70-77 | 4 | `KC_EVAL_COMP_002` Confidence Interval for Accuracy | low,medium | AUTO_LOW_KC,AUTO_MEDIUM_KC | False |
| 24 | 78-79 | 1 | `KC_EVAL_ENS_002` Majority Voting | low | AUTO_LOW_KC | False |
| 25 | 80-83 | 2 | `KC_EVAL_ENS_003` Random Forest | high,low | AUTO_HIGH_KC,AUTO_LOW_KC | False |
| 26 | 84-85 | 1 | `KC_EVAL_SAMP_003` k-Fold Cross Validation | low | AUTO_LOW_KC | False |
| 27 | 86-89 | 2 | `KC_CLU_CORE_004` Classification vs. Clustering (Distinction) | low | AUTO_LOW_KC | False |
| 28 | 90-103 | 7 | `KC_CLU_KM_001` K-Means Algorithm | low,medium | AUTO_LOW_KC,AUTO_MEDIUM_KC | False |
| 29 | 104-105 | 1 | `KC_CLU_SIM_004` Manhattan Distance | high | AUTO_HIGH_KC | False |
| 30 | 106-107 | 1 | `KC_CLU_SIM_005` Cosine Similarity | high | AUTO_HIGH_KC | False |
| 31 | 108-109 | 1 | `KC_CLU_KM_005` K-Means Limitations | low | AUTO_LOW_KC | False |
| 32 | 110-115 | 3 | `KC_CLU_KM_006` Bisecting K-Means | high,low,none | AUTO_HIGH_KC,AUTO_LOW_KC,NON_KC | False |
| 33 | 116-119 | 2 | `KC_CLF_UND_006` Target Attribute | high,low | AUTO_HIGH_KC,AUTO_LOW_KC | False |
| 34 | 120-125 | 3 | `KC_CLF_NB_007` Zero-Frequency Problem | low,medium,none | AUTO_LOW_KC,AUTO_MEDIUM_KC,NON_KC | False |