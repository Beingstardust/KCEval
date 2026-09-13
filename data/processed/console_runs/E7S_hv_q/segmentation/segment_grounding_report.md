# Contactless KC segmentation report

## Summary

- `pipeline`: `contactless_two_pass_v35_fixed_scorer_context_override_guard`
- `tie_breaker_model_path`: `/beegfs1/home/hpcuser/projects/seg_eval_v28_run/models/Qwen3-4B-Instruct-2507`
- `tie_breaker_active`: `True`
- `tie_breaker_top_k`: `5`
- `tie_breaker_max_pipeline_margin`: `999.0`
- `tie_breaker_min_probability`: `0.5`
- `tie_breaker_min_probability_margin`: `0.5`
- `tie_breaker_uses_prior_context`: `True`
- `tie_breaker_gated_in_count`: `44`
- `tie_breaker_changed_count`: `14`
- `abstained_low_confidence_count`: `13`
- `dialogue_count`: `1`
- `turn_count`: `88`
- `exchange_count`: `44`
- `profile_count`: `159`
- `segment_count`: `42`
- `excluded_meta_exchange_count`: `0`
- `excluded_meta_exchange_ids`: `[]`
- `low_confidence_no_strong_match_count`: `1`
- `exchanges_with_secondary_kcs`: `44`
- `secondary_kc_count_distribution`: `{'3': 44}`
- `missing_kc_review_count`: `0`
- `cross_encoder_active`: `True`
- `dense_retrieval_active`: `True`
- `exact_override_exchange_count`: `31`
- `pre_fusion_non_kc_gate_hits`: `0`
- `rrf_k`: `11`
- `family_support_rank_cutoff`: `25`
- `profile_family_views_used`: `['student', 'tutor']`
- `started_utc`: `2026-09-01T19:33:33.393441+00:00`
- `finished_utc`: `2026-09-01T20:09:32.541647+00:00`
- `pass1_AUTO_HIGH_KC`: `11`
- `pass1_AUTO_MEDIUM_KC`: `18`
- `pass1_NEEDS_PASS2`: `15`
- `final_AUTO_HIGH_KC`: `11`
- `final_AUTO_LOW_KC`: `15`
- `final_AUTO_MEDIUM_KC`: `18`
- `action_pass1_closed_set`: `29`
- `action_pass1_protected_from_continuation_relative_evidence_v19`: `3`
- `action_pass2_branch_guided_best_candidate`: `1`
- `action_pass2_broad_multibranch_closed_set`: `9`
- `action_pass2_tie_breaker_promotion_retained`: `2`
- `band_high`: `11`
- `band_low`: `15`
- `band_medium`: `18`

## Segments

| # | Turns | Exchanges | Dominant KC | Confidence | Labels | Review |
|---:|---|---:|---|---|---|---|
| 0 | 0-1 | 1 | `KC_CLF_UND_002` Querying Phase | low | AUTO_LOW_KC | False |
| 1 | 2-3 | 1 | `KC_CLF_UND_003` Training Set vs. Test Set Split | low | AUTO_LOW_KC | False |
| 2 | 4-5 | 1 | `KC_CLF_UND_006` Target Attribute | high | AUTO_HIGH_KC | False |
| 3 | 6-7 | 1 | `KC_CLF_UND_007` Attribute/Variable Types (Numerical, Categorical, Ordinal) | low | AUTO_LOW_KC | False |
| 4 | 8-9 | 1 | `KC_CLF_NB_001` Bayes' Theorem | medium | AUTO_MEDIUM_KC | False |
| 5 | 10-11 | 1 | `KC_CLF_NB_003` Conditional Probability (Likelihood) | medium | AUTO_MEDIUM_KC | False |
| 6 | 12-13 | 1 | `KC_CLF_NB_009` NB for Numerical Attributes (Gaussian NB) | medium | AUTO_MEDIUM_KC | False |
| 7 | 14-15 | 1 | `KC_CLF_NB_002` Prior Probability | medium | AUTO_MEDIUM_KC | False |
| 8 | 16-19 | 2 | `KC_CLF_NB_007` Zero-Frequency Problem | low,medium | AUTO_LOW_KC,AUTO_MEDIUM_KC | False |
| 9 | 20-21 | 1 | `KC_CLF_DT_006` Information Gain | high | AUTO_HIGH_KC | False |
| 10 | 22-23 | 1 | `KC_CLF_DT_008` Gain Ratio | high | AUTO_HIGH_KC | False |
| 11 | 24-25 | 1 | `KC_FSEL_GOOD_006` Shannon Entropy (Uncertainty Measure) | medium | AUTO_MEDIUM_KC | False |
| 12 | 26-27 | 1 | `KC_CLF_DT_011` Binary Decision Tree | medium | AUTO_MEDIUM_KC | False |
| 13 | 28-29 | 1 | `KC_CLF_PRUNE_003` Pessimistic Error Estimate | medium | AUTO_MEDIUM_KC | False |
| 14 | 30-31 | 1 | `KC_CLF_PRUNE_004` Reduced Error Pruning | high | AUTO_HIGH_KC | False |
| 15 | 32-33 | 1 | `KC_EVAL_BASIC_001` Confusion Matrix | high | AUTO_HIGH_KC | False |
| 16 | 34-35 | 1 | `KC_EVAL_BASIC_004` Recall (Sensitivity) | medium | AUTO_MEDIUM_KC | False |
| 17 | 36-37 | 1 | `KC_EVAL_COMP_002` Confidence Interval for Accuracy | medium | AUTO_MEDIUM_KC | False |
| 18 | 38-39 | 1 | `KC_EVAL_COMP_001` McNemar Test | low | AUTO_LOW_KC | False |
| 19 | 40-41 | 1 | `KC_EVAL_COMP_002` Confidence Interval for Accuracy | medium | AUTO_MEDIUM_KC | False |
| 20 | 42-43 | 1 | `KC_EVAL_ENS_003` Random Forest | low | AUTO_LOW_KC | False |
| 21 | 44-45 | 1 | `KC_EVAL_COMP_003` Comparing Two Models on Independent Test Sets | low | AUTO_LOW_KC | False |
| 22 | 46-47 | 1 | `KC_EVAL_BASIC_004` Recall (Sensitivity) | low | AUTO_LOW_KC | False |
| 23 | 48-49 | 1 | `KC_EVAL_ROC_005` Threshold Effect on Precision, Recall, F1 | high | AUTO_HIGH_KC | False |
| 24 | 50-51 | 1 | `KC_EVAL_ROC_006` Cost-Based Model Selection via ROC | medium | AUTO_MEDIUM_KC | False |
| 25 | 52-53 | 1 | `KC_CLU_KM_001` K-Means Algorithm | medium | AUTO_MEDIUM_KC | False |
| 26 | 54-55 | 1 | `KC_CLU_SIM_005` Cosine Similarity | high | AUTO_HIGH_KC | False |
| 27 | 56-57 | 1 | `KC_CLU_KM_001` K-Means Algorithm | medium | AUTO_MEDIUM_KC | False |
| 28 | 58-59 | 1 | `KC_CLU_KM_006` Bisecting K-Means | high | AUTO_HIGH_KC | False |
| 29 | 60-61 | 1 | `KC_CLU_DBS_008` DBSCAN Cluster Definition | high | AUTO_HIGH_KC | False |
| 30 | 62-63 | 1 | `KC_CLU_DBS_006` Density-Reachable | high | AUTO_HIGH_KC | False |
| 31 | 64-65 | 1 | `KC_CLU_DBS_001` Core Point | high | AUTO_HIGH_KC | False |
| 32 | 66-67 | 1 | `KC_CLU_HIER_002` Agglomerative (Bottom-Up) Clustering | medium | AUTO_MEDIUM_KC | False |
| 33 | 68-71 | 2 | `KC_CLU_HIER_004` MIN (Single Linkage) | low,medium | AUTO_LOW_KC,AUTO_MEDIUM_KC | False |
| 34 | 72-73 | 1 | `KC_CLU_EVAL_009` External Index: Purity | low | AUTO_LOW_KC | False |
| 35 | 74-75 | 1 | `KC_CLU_EVAL_010` External Index: Rand Index / Jaccard | low | AUTO_LOW_KC | False |
| 36 | 76-77 | 1 | `KC_CLU_EVAL_004` Separation | low | AUTO_LOW_KC | False |
| 37 | 78-79 | 1 | `KC_CLU_EVAL_005` Silhouette Coefficient | low | AUTO_LOW_KC | False |
| 38 | 80-81 | 1 | `KC_FSEL_FUND_001` Feature Selection Definition | medium | AUTO_MEDIUM_KC | False |
| 39 | 82-83 | 1 | `KC_FSEL_GOOD_002` Pearson Product-Moment Correlation | low | AUTO_LOW_KC | False |
| 40 | 84-85 | 1 | `KC_FSEL_FW_004` Filter vs. Wrapper Trade-off | medium | AUTO_MEDIUM_KC | False |
| 41 | 86-87 | 1 | `KC_EVAL_COMP_003` Comparing Two Models on Independent Test Sets | low | AUTO_LOW_KC | False |