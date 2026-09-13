# stress_v1 — dimension stress-test corpus

Generated from the course source PDFs with the prompt in
`docs/evaluation_completeness_v2/31_STRESS_TEST_CORPUS_GENERATION_PROMPT.md`, with one deliberate
substitution by the author: planted errors are anchored to the **source documents the KC library was
built from**, not to the library itself. That is what makes library coverage measurable rather than
assumed. See doc 32 for the adjudication.

| file | what it is |
|---|---|
| `dialogue.json` | 88 exchanges, no annotation of any kind |
| `ground_truth.json` | 163 records as generated, `detectability` in {pdf_required, parametric} |
| `ground_truth_adjudicated.json` | the same, plus `library_status` set by reading the library |

**`exchange_index` is 1-BASED.** The pipeline's `ex_NNNN` ids are 0-based. Subtract one when
normalizing, or every ground-truth record lands on the wrong exchange.
