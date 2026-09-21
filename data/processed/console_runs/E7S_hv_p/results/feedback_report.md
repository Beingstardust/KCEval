# Tutor feedback report — hv_p

**Final score T = 0.8493**  (alpha=0.8, beta=0.2, rho=0.8 — PROVISIONAL_NOT_FROZEN)

| component | value |
|---|---|
| D_segment (local) | 0.8164 |
| D_ARC (arc) | 0.9812 |
| D_micro | 0.8493 |
| M (macro) | — |

Attribution reconstructs the score exactly: **True**

> note: no macro score; T falls back to D_micro alone

## Where the score was lost, ranked

Each row is a concrete place to work. *Recoverable* is how much final T would be regained by lifting that dimension to 1.0 in that segment.

| # | dimension | segment | exchanges | score | recoverable | share |
|---:|---|---|---|---:|---:|---:|
| 1 | actionability_moving_forward | seg_0041 | ex_0043 | 0.50 | 0.0023 | 9.6% |
| 2 | clarity_cognitive_load | seg_0041 | ex_0043 | 0.50 | 0.0023 | 9.6% |
| 3 | solution_control | seg_0041 | ex_0043 | 0.50 | 0.0023 | 9.6% |
| 4 | clarity_cognitive_load | seg_0005 | ex_0005 | 0.50 | 0.0018 | 7.7% |
| 5 | clarity_cognitive_load | seg_0006 | ex_0006 | 0.50 | 0.0018 | 7.7% |
| 6 | clarity_cognitive_load | seg_0024 | ex_0025 | 0.50 | 0.0018 | 7.7% |
| 7 | clarity_cognitive_load | seg_0037 | ex_0039 | 0.50 | 0.0018 | 7.7% |
| 8 | error_reasoning | seg_0015 | ex_0016 | 0.50 | 0.0013 | 5.5% |
| 9 | error_localisation | seg_0019 | ex_0020 | 0.50 | 0.0013 | 5.5% |
| 10 | scaffolding_quality | topic_unit_0009 | ex_0022 | 0.50 | 0.0012 | 5.3% |
| 11 | student_level_calibration | topic_unit_0009 | ex_0022 | 0.50 | 0.0012 | 5.3% |
| 12 | student_level_calibration | topic_unit_0021 | ex_0043 | 0.50 | 0.0012 | 5.3% |
| 13 | actionability_moving_forward | seg_0021 | ex_0022 | 0.50 | 0.0010 | 4.4% |
| 14 | clarity_cognitive_load | seg_0021 | ex_0022 | 0.50 | 0.0010 | 4.4% |
| 15 | solution_control | seg_0021 | ex_0022 | 0.50 | 0.0010 | 4.4% |

## Weakest capabilities overall

| dimension | total recoverable | segments scored | segments below 1.0 |
|---|---:|---:|---:|
| clarity_cognitive_load | 0.0106 | 42 | 6 |
| solution_control | 0.0033 | 42 | 2 |
| actionability_moving_forward | 0.0033 | 42 | 2 |
| student_level_calibration | 0.0025 | 21 | 2 |
| error_reasoning | 0.0013 | 18 | 1 |
| error_localisation | 0.0013 | 18 | 1 |
| scaffolding_quality | 0.0012 | 21 | 1 |
| error_detection | 0.0000 | 19 | 0 |
| local_coherence_relevance | 0.0000 | 42 | 0 |

## Evidence for the top findings

### 1. actionability_moving_forward — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0023** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 2. clarity_cognitive_load — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0023** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 3. solution_control — seg_0041 (turns 86–87, KC KC_EVAL_COMP_003)
- scored **0.50**, worth **0.0023** of T
- exchange units: `E7S_hv_p::ex_0043`
- judge rationale: The tutor provides a clear and relevant summary of the key concepts covered, effectively connecting the dots for the student. However, the explanation could be improved by providing more specific examples or steps for the student to follow, which would enhance actionability and clarity.
  - evidence: "how well does what the model found match what's actually true?"

### 4. clarity_cognitive_load — seg_0005 (turns 10–11, KC KC_CLF_NB_003)
- scored **0.50**, worth **0.0018** of T
- exchange units: `E7S_hv_p::ex_0005`
- judge rationale: The tutor provides a clear explanation of how Naive Bayes decides the class, using Bayes' theorem and explaining the 'naive' assumption. The response is relevant and coherent, but may be slightly overloaded for some learners due to the technical terms used.
  - evidence: "In the classification phase it picks the class y that maximizes P(y) · Π P(xi|y) over all attributes xi, using Bayes' theorem and dropping the shared denominator P(X) since it is the same for every cl"

### 5. clarity_cognitive_load — seg_0006 (turns 12–13, KC KC_CLF_NB_009)
- scored **0.50**, worth **0.0018** of T
- exchange units: `E7S_hv_p::ex_0006`
- judge rationale: The tutor provides a clear explanation of how Naive Bayes handles numeric attributes, using relevant concepts from the primary KC. The response is coherent and relevant to the student's question, demonstrating good scaffolding quality and actionability. However, the explanation may be slightly overloaded for the learner, resulting in a moderate clarity score.
  - evidence: "For a numeric attribute you don't count frequencies — you estimate a class-conditional density instead."
