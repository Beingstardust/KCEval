# Tutor feedback report — hv_p

**Final score T = 0.8495**  (alpha=0.8, beta=0.2, rho=0.8 — PROVISIONAL_NOT_FROZEN)

| component | value |
|---|---|
| D_segment (local) | 0.8169 |
| D_ARC (arc) | 0.9799 |
| D_micro | 0.8495 |
| M (macro) | — |

Attribution reconstructs the score exactly: **True**

> note: no macro score; T falls back to D_micro alone

## Where the score was lost, ranked

Each row is a concrete place to work. *Recoverable* is how much final T would be regained by lifting that dimension to 1.0 in that segment.

| # | dimension | segment | exchanges | score | recoverable | share |
|---:|---|---|---|---:|---:|---:|
| 1 | actionability_moving_forward | seg_0041 | ex_0043 | 0.50 | 0.0015 | 6.8% |
| 2 | clarity_cognitive_load | seg_0041 | ex_0043 | 0.50 | 0.0015 | 6.8% |
| 3 | solution_control | seg_0041 | ex_0043 | 0.50 | 0.0015 | 6.8% |
| 4 | student_level_calibration | seg_0041 | ex_0043 | 0.50 | 0.0015 | 6.8% |
| 5 | clarity_cognitive_load | seg_0005 | ex_0005 | 0.50 | 0.0012 | 5.5% |
| 6 | student_level_calibration | seg_0005 | ex_0005 | 0.50 | 0.0012 | 5.5% |
| 7 | clarity_cognitive_load | seg_0006 | ex_0006 | 0.50 | 0.0012 | 5.5% |
| 8 | clarity_cognitive_load | seg_0024 | ex_0025 | 0.50 | 0.0012 | 5.5% |
| 9 | clarity_cognitive_load | seg_0037 | ex_0039 | 0.50 | 0.0012 | 5.5% |
| 10 | error_reasoning | seg_0015 | ex_0016 | 0.50 | 0.0010 | 4.5% |
| 11 | error_localisation | seg_0019 | ex_0020 | 0.50 | 0.0010 | 4.5% |
| 12 | actionability_moving_forward | seg_0021 | ex_0022 | 0.50 | 0.0008 | 3.6% |
| 13 | clarity_cognitive_load | seg_0021 | ex_0022 | 0.50 | 0.0008 | 3.6% |
| 14 | scaffolding_quality | seg_0021 | ex_0022 | 0.50 | 0.0008 | 3.6% |
| 15 | solution_control | seg_0021 | ex_0022 | 0.50 | 0.0008 | 3.6% |
| 16 | student_level_calibration | seg_0021 | ex_0022 | 0.50 | 0.0008 | 3.6% |
| 17 | actionability_moving_forward | topic_unit_0021 | ex_0043 | 0.50 | 0.0004 | 1.9% |
| 18 | clarity_cognitive_load | topic_unit_0021 | ex_0043 | 0.50 | 0.0004 | 1.9% |
| 19 | student_level_calibration | topic_unit_0021 | ex_0043 | 0.50 | 0.0004 | 1.9% |
| 20 | clarity_cognitive_load | topic_unit_0006 | ex_0019 | 0.50 | 0.0003 | 1.3% |
| 21 | solution_control | topic_unit_0006 | ex_0019 | 0.50 | 0.0003 | 1.3% |
| 22 | actionability_moving_forward | topic_unit_0007 | ex_0020 | 0.50 | 0.0003 | 1.3% |
| 23 | clarity_cognitive_load | topic_unit_0007 | ex_0020 | 0.50 | 0.0003 | 1.3% |
| 24 | error_localisation | topic_unit_0007 | ex_0020 | 0.50 | 0.0003 | 1.3% |
| 25 | actionability_moving_forward | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 1.3% |
| 26 | clarity_cognitive_load | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 1.3% |
| 27 | scaffolding_quality | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 1.3% |
| 28 | solution_control | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 1.3% |
| 29 | student_level_calibration | topic_unit_0009 | ex_0022 | 0.50 | 0.0003 | 1.3% |

## Weakest capabilities overall

| dimension | total recoverable | segments scored | segments below 1.0 |
|---|---:|---:|---:|
| clarity_cognitive_load | 0.0084 | 63 | 10 |
| student_level_calibration | 0.0042 | 63 | 5 |
| actionability_moving_forward | 0.0033 | 63 | 5 |
| solution_control | 0.0029 | 63 | 4 |
| error_localisation | 0.0013 | 28 | 2 |
| scaffolding_quality | 0.0011 | 63 | 2 |
| error_reasoning | 0.0010 | 28 | 1 |
| error_detection | 0.0000 | 29 | 0 |
| local_coherence_relevance | 0.0000 | 63 | 0 |

## Evidence for the top findings

### 1. actionability_moving_forward — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0015** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 2. clarity_cognitive_load — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0015** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 3. solution_control — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0015** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 4. student_level_calibration — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0015** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 5. clarity_cognitive_load — seg_0005 (turns 10–11, KC KC_CLF_NB_003)
- scored **0.50**, worth **0.0012** of T
- exchange units: `E7S_hv_p::ex_0005`
- judge rationale: The tutor provides a clear explanation of how Naive Bayes decides the class, using Bayes' theorem and explaining the 'naive' assumption. The response is relevant and coherent, but may be slightly overloaded for some learners due to the technical terms used.
  - evidence: "In the classification phase it picks the class y that maximizes P(y) · Π P(xi|y) over all attributes xi, using Bayes' theorem and dropping the shared denominator P(X) since it is the same for every cl"
