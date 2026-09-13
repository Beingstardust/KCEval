# Tutor feedback report — hv_q

**Final score T = 0.7717**  (alpha=0.8, beta=0.2, rho=0.8 — PROVISIONAL_NOT_FROZEN)

| component | value |
|---|---|
| D_segment (local) | 0.7215 |
| D_ARC (arc) | 0.9729 |
| D_micro | 0.7717 |
| M (macro) | — |

Attribution reconstructs the score exactly: **True**

> note: no macro score; T falls back to D_micro alone

## Where the score was lost, ranked

Each row is a concrete place to work. *Recoverable* is how much final T would be regained by lifting that dimension to 1.0 in that segment.

| # | dimension | segment | exchanges | score | recoverable | share |
|---:|---|---|---|---:|---:|---:|
| 1 | error_detection | seg_0018 | ex_0019 | 0.00 | 0.0020 | 4.6% |
| 2 | error_localisation | seg_0018 | ex_0019 | 0.00 | 0.0020 | 4.6% |
| 3 | error_reasoning | seg_0018 | ex_0019 | 0.00 | 0.0020 | 4.6% |
| 4 | actionability_moving_forward | seg_0021 | ex_0022 | 0.50 | 0.0015 | 3.5% |
| 5 | clarity_cognitive_load | seg_0021 | ex_0022 | 0.50 | 0.0015 | 3.5% |
| 6 | scaffolding_quality | seg_0021 | ex_0022 | 0.50 | 0.0015 | 3.5% |
| 7 | solution_control | seg_0021 | ex_0022 | 0.50 | 0.0015 | 3.5% |
| 8 | student_level_calibration | seg_0021 | ex_0022 | 0.50 | 0.0015 | 3.5% |
| 9 | actionability_moving_forward | seg_0041 | ex_0043 | 0.50 | 0.0015 | 3.5% |
| 10 | clarity_cognitive_load | seg_0041 | ex_0043 | 0.50 | 0.0015 | 3.5% |
| 11 | student_level_calibration | seg_0041 | ex_0043 | 0.50 | 0.0015 | 3.5% |
| 12 | clarity_cognitive_load | seg_0005 | ex_0005 | 0.50 | 0.0012 | 2.8% |
| 13 | student_level_calibration | seg_0005 | ex_0005 | 0.50 | 0.0012 | 2.8% |
| 14 | clarity_cognitive_load | seg_0006 | ex_0006 | 0.50 | 0.0012 | 2.8% |
| 15 | clarity_cognitive_load | seg_0013 | ex_0014 | 0.50 | 0.0012 | 2.8% |
| 16 | clarity_cognitive_load | seg_0024 | ex_0025 | 0.50 | 0.0012 | 2.8% |
| 17 | clarity_cognitive_load | seg_0027 | ex_0028 | 0.50 | 0.0012 | 2.8% |
| 18 | actionability_moving_forward | seg_0032 | ex_0033 | 0.50 | 0.0012 | 2.8% |
| 19 | clarity_cognitive_load | seg_0032 | ex_0033 | 0.50 | 0.0012 | 2.8% |
| 20 | student_level_calibration | seg_0032 | ex_0033 | 0.50 | 0.0012 | 2.8% |
| 21 | clarity_cognitive_load | seg_0037 | ex_0039 | 0.50 | 0.0012 | 2.8% |
| 22 | solution_control | seg_0016 | ex_0017 | 0.50 | 0.0010 | 2.3% |
| 23 | clarity_cognitive_load | seg_0018 | ex_0019 | 0.50 | 0.0010 | 2.3% |
| 24 | scaffolding_quality | seg_0018 | ex_0019 | 0.50 | 0.0010 | 2.3% |
| 25 | solution_control | seg_0018 | ex_0019 | 0.50 | 0.0010 | 2.3% |
| 26 | student_level_calibration | seg_0018 | ex_0019 | 0.50 | 0.0010 | 2.3% |
| 27 | actionability_moving_forward | seg_0019 | ex_0020 | 0.50 | 0.0010 | 2.3% |
| 28 | clarity_cognitive_load | seg_0019 | ex_0020 | 0.50 | 0.0010 | 2.3% |
| 29 | error_localisation | seg_0019 | ex_0020 | 0.50 | 0.0010 | 2.3% |
| 30 | clarity_cognitive_load | topic_unit_0005 | ex_0014, ex_0015 | 0.50 | 0.0008 | 1.7% |
| 31 | clarity_cognitive_load | topic_unit_0014 | ex_0028, ex_0029 | 0.50 | 0.0008 | 1.7% |
| 32 | error_detection | topic_unit_0009 | ex_0022 | 0.00 | 0.0005 | 1.2% |
| 33 | error_localisation | topic_unit_0009 | ex_0022 | 0.00 | 0.0005 | 1.2% |
| 34 | error_reasoning | topic_unit_0009 | ex_0022 | 0.00 | 0.0005 | 1.2% |
| 35 | actionability_moving_forward | topic_unit_0021 | ex_0043 | 0.50 | 0.0004 | 0.9% |
| 36 | clarity_cognitive_load | topic_unit_0021 | ex_0043 | 0.50 | 0.0004 | 0.9% |
| 37 | student_level_calibration | topic_unit_0021 | ex_0043 | 0.50 | 0.0004 | 0.9% |
| 38 | actionability_moving_forward | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 0.6% |
| 39 | clarity_cognitive_load | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 0.6% |
| 40 | scaffolding_quality | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 0.6% |

## Weakest capabilities overall

| dimension | total recoverable | segments scored | segments below 1.0 |
|---|---:|---:|---:|
| clarity_cognitive_load | 0.0157 | 64 | 15 |
| student_level_calibration | 0.0071 | 64 | 7 |
| actionability_moving_forward | 0.0059 | 64 | 6 |
| solution_control | 0.0038 | 64 | 4 |
| error_localisation | 0.0035 | 23 | 3 |
| scaffolding_quality | 0.0028 | 64 | 3 |
| error_detection | 0.0025 | 24 | 2 |
| error_reasoning | 0.0025 | 23 | 2 |
| local_coherence_relevance | 0.0000 | 64 | 0 |

## Evidence for the top findings

### 1. error_detection — seg_0018 (turns 38–39, KC KC_EVAL_COMP_001)
- scored **0.00**, worth **0.0020** of T
- exchange units: `E7S_hv_q::ex_0019`
- judge rationale: The tutor provides a clear and actionable response to the student's question, but does not address any potential errors or misconceptions. The response is relevant and coherent, but may be slightly overloaded for the learner. The tutor could improve by providing more guidance on how to interpret the results of the Z-test.
  - evidence: "Compute Z = (pA−pB)/√(2p̄(1−p̄)/N) for each dataset and compare its magnitude to 1.64"

### 2. error_localisation — seg_0018 (turns 38–39, KC KC_EVAL_COMP_001)
- scored **0.00**, worth **0.0020** of T
- exchange units: `E7S_hv_q::ex_0019`
- judge rationale: The tutor provides a clear and actionable response to the student's question, but does not address any potential errors or misconceptions. The response is relevant and coherent, but may be slightly overloaded for the learner. The tutor could improve by providing more guidance on how to interpret the results of the Z-test.
  - evidence: "Compute Z = (pA−pB)/√(2p̄(1−p̄)/N) for each dataset and compare its magnitude to 1.64"

### 3. error_reasoning — seg_0018 (turns 38–39, KC KC_EVAL_COMP_001)
- scored **0.00**, worth **0.0020** of T
- exchange units: `E7S_hv_q::ex_0019`
- judge rationale: The tutor provides a clear and actionable response to the student's question, but does not address any potential errors or misconceptions. The response is relevant and coherent, but may be slightly overloaded for the learner. The tutor could improve by providing more guidance on how to interpret the results of the Z-test.
  - evidence: "Compute Z = (pA−pB)/√(2p̄(1−p̄)/N) for each dataset and compare its magnitude to 1.64"

### 4. actionability_moving_forward — seg_0021 (turns 44–45, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0015** of T
- exchange units: `E7S_hv_q::ex_0022`
- judge rationale: The tutor provides a clear comparison between the two models, M1 and M2, based on their AUC values. However, the explanation could be improved by providing more context or scaffolding for the student to understand the significance of the AUC values. The response is relevant and coherent, but lacks clarity in terms of cognitive load, as the tutor assumes the student is familiar with the concept of AUC and its interpretation.
  - evidence: "M1 comes out around 0.92 — strong separation between the positive and negative instances across virtually every threshold."

### 5. clarity_cognitive_load — seg_0021 (turns 44–45, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0015** of T
- exchange units: `E7S_hv_q::ex_0022`
- judge rationale: The tutor provides a clear comparison between the two models, M1 and M2, based on their AUC values. However, the explanation could be improved by providing more context or scaffolding for the student to understand the significance of the AUC values. The response is relevant and coherent, but lacks clarity in terms of cognitive load, as the tutor assumes the student is familiar with the concept of AUC and its interpretation.
  - evidence: "M1 comes out around 0.92 — strong separation between the positive and negative instances across virtually every threshold."
