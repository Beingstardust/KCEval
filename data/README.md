# Data: contents, provenance, and licensing

This directory contains a curated subset of the source project's `data/` tree: exactly the frozen
artifacts read by the scripts in `../REPRODUCE.md`, plus the dialogue corpora and KC library those
scripts and the pipeline/judge code operate on. It is **not** the full ~250 MB `data/` tree of the
source project (which also holds many superseded experimental runs, intermediate retrieval-candidate
dumps, and internal audit workbooks not needed here).

## Layout

- `input/dialogues/` — the two source dialogues and all five tutor-variant / condition dialogues
  described in the paper's Section 4 ("Corpora" and "Conditions"), as plain JSON: `tv_a` (REF)
  through `tv_e` (EXP-REG), `sv1`/`sv2` (second dialogue / its corrupted replication copy), and
  `hv_p`/`hv_q` (the replication console runs, clean/corrupted).
- `input/frozen_library_incoming/kc_library_reviewed_v2/` — the frozen, expert-reviewed 159-KC
  library used by every reported experiment.
- `input/bea_mrbench_v3/` — the BEA 2025 / MRBench v3 dev split used for the external control in
  Section 5.4, vendored under its original licence (see below).
- `gold/` — expert gold KC assignments, human-validation exports (two annotators), injected-error
  ground truth, and frozen result artifacts (`e7_results.json`, `kc_assignment_baselines.json`,
  `bea_external_validation.json`, etc.) referenced directly in `REPRODUCE.md`.
- `processed/` — frozen judge prompts/responses and evaluation packets for the five tutor-variant
  conditions and the replication console runs, plus the deterministic re-analyses this cleanup's
  own verification pass wrote back out (`processed/publication_final/`).

## Constructed dialogues: no real student data

Per the paper's Experimental Setup (Section 4): *"The learner turns were constructed by the authors
with a publicly available tool rather than collected from students, and the tutor turns are an AI
tutor's responses to them; no learner participants were recruited and no personal data were
processed."* This cleanup re-confirmed that reading by inspecting the dialogue files directly (no
names, no identifying detail beyond the constructed pedagogical content) before including them in
full. They are safe to publish as-is on that basis.

## Model weights: not included

No model weights are redistributed. See the main `README.md`'s "Models" section for exact
HuggingFace model IDs and the pinned Selene revision. `gold/judge_weight_identity_sofja_vs_ants.json`
records the provenance check that established the two serving copies of Selene were byte-identical
snapshots (account-specific filesystem paths from the original two serving hosts were redacted from
this file and from every judge-response file that carried them; the hashes, revision, and model IDs
are untouched).

## Licensing

**This is a proposal, not a finalized decision** — the source project's authors have not stated a
license for this data, and this section should be read as a recommended default pending their
confirmation, per the brief that produced this repository.

- **Own artifacts** (the constructed dialogues, the frozen KC library, human annotations, judge
  prompts/responses, and every derived analysis file): proposed **CC BY 4.0**. These are the
  authors' own constructed/collected material and carry no known third-party rights.
- **`input/bea_mrbench_v3/`** (the BEA 2025 / MRBench v3 dev split): this is **not** the authors'
  data. It is vendored from `github.com/kaushal0494/UnifyingAITutorEvaluation` under its stated
  licence, **CC BY-SA 4.0** (see `input/bea_mrbench_v3/PROVENANCE.json` for the exact source URL,
  download date, and sha256). Anything derived from it and redistributed here — including
  `gold/e1_raw/*`, `gold/bea_external_validation.json`, and
  `processed/judge_prompts/bea_arm_a_20260902/bea_gold.jsonl` — should be treated as inheriting the
  share-alike obligation, and attribution to Kochmar et al. (BEA @ ACL 2025) and the original
  repository must be kept wherever this subset is used. This is a different licence from the rest
  of this repository's data and is called out separately for that reason. Only the frozen gold
  labels and judge responses are vendored here, not the full rendered judge prompts (regenerable
  from the dev split with `scripts/build_grounded_factuality_prompts.py`-style tooling), to keep
  the vendored footprint to what is actually needed to verify the reported numbers.
- **The KC library's `evidence_spans`/`evidence_map` fields**: **flagged, not resolved.** These
  fields quote or closely paraphrase the underlying course's textbook/slide material to support the
  factuality pass. This cleanup checked all 159 entries (388 evidence spans): the longest is 250
  characters and the median is 119, reading as short single-sentence paraphrases rather than
  extended verbatim quotation, and no raw slide/textbook source text is stored anywhere in this
  repository (only these short derived claim strings plus an opaque internal evidence id). That is
  a sampled, heuristic finding, not a copyright clearance — the course materials themselves are not
  the authors' to license, and whether these short paraphrased claims require separate permission
  from the course/institution is the authors' call to make, not one this cleanup can make for them.
