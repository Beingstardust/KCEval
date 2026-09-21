# Tutor feedback report — hv_q

**Final score T = 0.7732**  (alpha=0.8, beta=0.2, rho=0.8 — PROVISIONAL_NOT_FROZEN)

| component | value |
|---|---|
| D_segment (local) | 0.7208 |
| D_ARC (arc) | 0.9830 |
| D_micro | 0.7732 |
| M (macro) | — |

Attribution reconstructs the score exactly: **True**

> note: no macro score; T falls back to D_micro alone

## Where the score was lost, ranked

Each row is a concrete place to work. *Recoverable* is how much final T would be regained by lifting that dimension to 1.0 in that segment.

| # | dimension | segment | exchanges | score | recoverable | share |
|---:|---|---|---|---:|---:|---:|
| 1 | error_detection | seg_0018 | ex_0019 | 0.00 | 0.0026 | 5.8% |
| 2 | error_localisation | seg_0018 | ex_0019 | 0.00 | 0.0026 | 5.8% |
| 3 | error_reasoning | seg_0018 | ex_0019 | 0.00 | 0.0026 | 5.8% |
| 4 | actionability_moving_forward | seg_0021 | ex_0022 | 0.50 | 0.0023 | 5.1% |
| 5 | clarity_cognitive_load | seg_0021 | ex_0022 | 0.50 | 0.0023 | 5.1% |
| 6 | solution_control | seg_0021 | ex_0022 | 0.50 | 0.0023 | 5.1% |
| 7 | actionability_moving_forward | seg_0041 | ex_0043 | 0.50 | 0.0023 | 5.1% |
| 8 | clarity_cognitive_load | seg_0041 | ex_0043 | 0.50 | 0.0023 | 5.1% |
| 9 | clarity_cognitive_load | seg_0005 | ex_0005 | 0.50 | 0.0018 | 4.0% |
| 10 | clarity_cognitive_load | seg_0006 | ex_0006 | 0.50 | 0.0018 | 4.0% |
| 11 | clarity_cognitive_load | seg_0013 | ex_0014 | 0.50 | 0.0018 | 4.0% |
| 12 | clarity_cognitive_load | seg_0024 | ex_0025 | 0.50 | 0.0018 | 4.0% |
| 13 | clarity_cognitive_load | seg_0027 | ex_0028 | 0.50 | 0.0018 | 4.0% |
| 14 | actionability_moving_forward | seg_0032 | ex_0033 | 0.50 | 0.0018 | 4.0% |
| 15 | clarity_cognitive_load | seg_0032 | ex_0033 | 0.50 | 0.0018 | 4.0% |
| 16 | clarity_cognitive_load | seg_0037 | ex_0039 | 0.50 | 0.0018 | 4.0% |
| 17 | solution_control | seg_0016 | ex_0017 | 0.50 | 0.0013 | 2.9% |
| 18 | clarity_cognitive_load | seg_0018 | ex_0019 | 0.50 | 0.0013 | 2.9% |
| 19 | solution_control | seg_0018 | ex_0019 | 0.50 | 0.0013 | 2.9% |
| 20 | actionability_moving_forward | seg_0019 | ex_0020 | 0.50 | 0.0013 | 2.9% |
| 21 | clarity_cognitive_load | seg_0019 | ex_0020 | 0.50 | 0.0013 | 2.9% |
| 22 | error_localisation | seg_0019 | ex_0020 | 0.50 | 0.0013 | 2.9% |
| 23 | scaffolding_quality | topic_unit_0009 | ex_0022 | 0.50 | 0.0011 | 2.5% |
| 24 | student_level_calibration | topic_unit_0009 | ex_0022 | 0.50 | 0.0011 | 2.5% |
| 25 | student_level_calibration | topic_unit_0021 | ex_0043 | 0.50 | 0.0011 | 2.5% |

## Weakest capabilities overall

| dimension | total recoverable | segments scored | segments below 1.0 |
|---|---:|---:|---:|
| clarity_cognitive_load | 0.0199 | 42 | 11 |
| actionability_moving_forward | 0.0077 | 42 | 4 |
| solution_control | 0.0049 | 42 | 3 |
| error_localisation | 0.0039 | 14 | 2 |
| error_detection | 0.0026 | 15 | 1 |
| error_reasoning | 0.0026 | 14 | 1 |
| student_level_calibration | 0.0023 | 22 | 2 |
| scaffolding_quality | 0.0011 | 22 | 1 |
| local_coherence_relevance | 0.0000 | 42 | 0 |

## Evidence for the top findings

### 1. error_detection — seg_0018 (turns 38–39, KC KC_EVAL_COMP_001)
- scored **0.00**, worth **0.0026** of T
- exchange units: `E7S_hv_q::ex_0019`
- judge rationale: The tutor provides a clear and actionable response to the student's question, but does not address any potential errors or misconceptions. The response is relevant and coherent, but may be slightly overloaded for the learner. The tutor could improve by providing more guidance on how to interpret the results of the Z-test.
  - evidence: "Compute Z = (pA−pB)/√(2p̄(1−p̄)/N) for each dataset and compare its magnitude to 1.64"

### 2. error_localisation — seg_0018 (turns 38–39, KC KC_EVAL_COMP_001)
- scored **0.00**, worth **0.0026** of T
- exchange units: `E7S_hv_q::ex_0019`
- judge rationale: The tutor provides a clear and actionable response to the student's question, but does not address any potential errors or misconceptions. The response is relevant and coherent, but may be slightly overloaded for the learner. The tutor could improve by providing more guidance on how to interpret the results of the Z-test.
  - evidence: "Compute Z = (pA−pB)/√(2p̄(1−p̄)/N) for each dataset and compare its magnitude to 1.64"

### 3. error_reasoning — seg_0018 (turns 38–39, KC KC_EVAL_COMP_001)
- scored **0.00**, worth **0.0026** of T
- exchange units: `E7S_hv_q::ex_0019`
- judge rationale: The tutor provides a clear and actionable response to the student's question, but does not address any potential errors or misconceptions. The response is relevant and coherent, but may be slightly overloaded for the learner. The tutor could improve by providing more guidance on how to interpret the results of the Z-test.
  - evidence: "Compute Z = (pA−pB)/√(2p̄(1−p̄)/N) for each dataset and compare its magnitude to 1.64"

### 4. actionability_moving_forward — seg_0021 (turns 44–45, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0023** of T
- exchange units: `E7S_hv_q::ex_0022`
- judge rationale: The tutor provides a clear comparison between the two models, M1 and M2, based on their AUC values. However, the explanation could be improved by providing more context or scaffolding for the student to understand the significance of the AUC values. The response is relevant and coherent, but lacks clarity in terms of cognitive load, as the tutor assumes the student is familiar with the concept of AUC and its interpretation.
  - evidence: "M1 comes out around 0.92 — strong separation between the positive and negative instances across virtually every threshold."

### 5. clarity_cognitive_load — seg_0021 (turns 44–45, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0023** of T
- exchange units: `E7S_hv_q::ex_0022`
- judge rationale: The tutor provides a clear comparison between the two models, M1 and M2, based on their AUC values. However, the explanation could be improved by providing more context or scaffolding for the student to understand the significance of the AUC values. The response is relevant and coherent, but lacks clarity in terms of cognitive load, as the tutor assumes the student is familiar with the concept of AUC and its interpretation.
  - evidence: "M1 comes out around 0.92 — strong separation between the positive and negative instances across virtually every threshold."
