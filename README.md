# KCEval

KCEval evaluates complete AI-tutor dialogues against a frozen, expert-reviewed course Knowledge
Component (KC) library. It matches a dialogue's exchanges to the library, groups them into
KC-aligned segments and broader topic units, scores tutor behaviour at both scopes against generic
pedagogical dimensions and KC-specific expert criteria, checks tutor content against curriculum
evidence with an independent factuality pass, and keeps every score traceable back to the exchanges
that produced it. This is the companion repository for the paper *"KCEval: Curriculum-Indexed,
Scope-Aware and Traceable Evaluation of AI Tutor Dialogues"* (submitted, ACM SAC 2027, AI for
Education track; currently under double-blind review).

**Start here:** [`REPRODUCE.md`](REPRODUCE.md) maps every number reported in the paper to the exact
command that reproduces it. Everything in this README is orientation for that file.

## What is (and is not) in this repository

This repository ships the **evaluation and analysis code and the frozen intermediate artifacts**
that the paper's tables are computed from — recomputing them needs nothing beyond a Python 3.10+
standard library. It also ships the **KC-indexing/segmentation pipeline and the LLM-judge-calling
code** as the actual implementation of the method in Section 3, for inspection and for anyone with
GPU access to rerun; see [`REPRODUCE.md`](REPRODUCE.md#pipeline--judge-layer-requires-gpu). It does
**not** ship model weights (Selene-1-Llama-3.3-70B, the bge retrieval/rerank models, or the
tie-breaker model) — see [Models](#models) — and it does not ship the internal engineering history
(220+ iteration scripts, ~60 internal audit documents, an annotation-portal web app) that produced
this project over its lifetime; see [What was left out](#what-was-left-out-and-why).

## Install

```bash
git clone https://github.com/Beingstardust/KCEval.git
cd KCEval
pip install -e .
```

The core reproduction path (everything in `REPRODUCE.md` except the pipeline/judge layer) has **no
third-party dependencies**. Two optional extras cover the rest:

```bash
pip install -e ".[segmentation]"   # torch, transformers, sentence-transformers, scikit-learn — to
                                    # rerun the KC-indexing pipeline or the naive-retrieval baselines
pip install -e ".[workbooks]"      # openpyxl — only needed to read the .xlsx gold-annotation workbooks
```

## Quickstart

```bash
python scripts/build_e5_agreement_table.py          # Table 2: human/judge agreement
python scripts/analyze_v35_vs_b3_mcnemar.py          # Table 1: KC-indexing exact/acceptable agreement
python scripts/analyze_kc_induction_comparison.py    # Table 1: partition agreement (ARI)
python scripts/recompute_e7_with_definition_factuality.py  # Table 3: rescore under the definition-grounded pass
python scripts/analyze_scope_routing_ablation.py     # Section 5.4: routed-versus-unrouted ablation
python scripts/compute_option_b_final_tutor_score.py # supplementary: final tutor score grid
```

Table 3 takes two steps, since the trust cap fires on the factuality verdict and the replication
therefore has to be rescored rather than restated. `REPRODUCE.md` gives the second command and
explains what the first one validates before it reports anything.

Each prints its numbers directly to the terminal, and several assert their own reproduction of a
frozen artifact before printing anything else (see `REPRODUCE.md` for what each one asserts and
against what). Run `pytest` from the repository root to run the unit tests for the pruned module
set (mostly deterministic-aggregation, response-contract and packet-building logic; the tests do
not need a GPU). One test, `test_conversation_region_extractor_v1.py::test_auto_student_tutor_turn_scan_excludes_header_footer`,
fails on a pre-existing bug in a text-extraction helper used only during original corpus
construction (not by anything in `REPRODUCE.md`); it was confirmed to fail identically in the
source project before this repository was built, so it was left as-is rather than silently fixed.

## Repository layout

```
src/seg_eval/
  aggregation/        deterministic scoring: P_s/K_s/B_s/C_s, trust cap, D_segment/D_topic/T,
                       scope routing (Section 3.7)
  evaluation_judge/    prompt builders, response contracts/schemas, and the Selene HTTP client for
                       every judge pass (segment/topic rubric, KC criteria, factuality, macro)
  evaluation_packets/  builds the per-scope evaluation packets the judge prompts are built from
  contactless/         the v35 KC-indexing pipeline (retrieval, fusion, cross-encoder rerank,
                       resolution, tie-breaker application, segment/topic grouping) — Section 3.3
  tie_breaker/         the generative tie-breaker's option scoring and pool application
scripts/               one entrypoint per REPRODUCE.md row, plus the judge-calling and pipeline
                       entrypoints (see REPRODUCE.md for which is which)
data/                  frozen corpora, KC library, and judge/segmentation outputs — see data/README.md
tests/                 unit tests for the modules above (pruned to match)
REPRODUCE.md           paper claim -> exact command map
```

## Data

See [`data/README.md`](data/README.md) for what is included, its provenance, and its licensing —
including one item the paper's authors have not made a final call on (the KC library's evidence
fields) and one vendored third-party dataset under its own licence (BEA 2025 / MRBench v3, CC
BY-SA 4.0). The two dialogue corpora are entirely constructed (no real student data; see the
paper's Experimental Setup, Section 4) and are included in full at `data/input/dialogues/`.

## Models

Nothing in this repository redistributes model weights. The paper's single evaluation judge is
**Selene-1-Llama-3.3-70B** (HuggingFace `AtlaAI/Selene-1-Llama-3.3-70B`, revision
`37ad448f3d1ab7579cbccb0d8418b2808573a30b`), served with vLLM under tensor-parallel 2, bf16, greedy
decoding. The KC-indexing pipeline's retrieval/rerank stage uses `BAAI/bge-small-en-v1.5` (dense
retrieval) and `BAAI/bge-reranker-v2-m3` (cross-encoder rerank); the indexing tie-breaker uses
`Qwen/Qwen3-4B-Instruct-2507`. `data/gold/judge_weight_identity_sofja_vs_ants.json` documents how
the exact Selene weights snapshot was verified byte-identical across the two machines this project
served it from (account-specific filesystem paths redacted; the verification method and hashes are
intact).

## Flagged for the author

A few things this cleanup found are judgment calls the paper's authors should make explicitly
rather than have silently resolved by whoever built this repository:

1. **The KC library's `evidence_spans`/`evidence_map` fields** (short factual claims attached to
   each KC, used by the factuality pass) were checked for verbatim copyrighted course-material
   text: across all 159 entries and 388 evidence spans, the longest is 250 characters and the
   median is 119 — these read as short paraphrased single-sentence claims, not extended quotation,
   and no raw slide/textbook text blob is stored anywhere in this repository. That is a sampled,
   heuristic check (length is a proxy, not a copyright determination), not a legal clearance —
   confirm this reading holds before treating the library's evidence fields as clear to publish
   verbatim in a camera-ready or an anonymized mirror.
2. **Two apparent script duplicates were kept rather than collapsed**, because it was not possible
   to establish from the artifacts alone which one produced the shipped frozen judge responses:
   `scripts/run_grounded_factuality_judge.py` vs `scripts/run_selene_factuality_judge.py`, and
   `scripts/validate_segment_judge_responses.py` vs `scripts/validate_segment_judge_responses_v2.py`.
   Both members of each pair are included and both resolve their imports cleanly; picking one as
   canonical (or documenting why both are needed) is a decision for whoever maintains this repo
   next.
3. **`src/seg_eval/profile_native_matcher_v10.py`** (an already-drafted false-negative fix to the
   matcher used by the shipped v35 pipeline) exists in the source repository but is **not**
   reachable from the frozen v35 entrypoint and was excluded here on that basis. It is not simply
   dead code, though — the source project's own notes describe it as "ready for next round," i.e.
   prepared future work. It is excluded from this snapshot because it was not part of what produced
   the reported numbers, not because it is superseded; a future release that adopts it should say so
   explicitly rather than swap it in silently.
4. **Data licensing** (own artifacts vs. the vendored BEA subset vs. the KC library) is proposed,
   not finalized — see `data/README.md`. The code `LICENSE` (Apache-2.0) and `CITATION.cff` are
   reasonable defaults, not a claim that the authors have chosen them.

## What was left out, and why

The source project (a single-author MSc thesis kernel with roughly 189 commits and heavy iteration)
contains a great deal that never reached, or was superseded before, the frozen configuration that
produced this paper's numbers. This was established by tracing forward from the actual reproduction
entrypoints (both by running the pure-Python analysis scripts under an import/file-access tracer,
and by a static import trace of the GPU-dependent pipeline entrypoint), not by guessing from file
age or version suffix. Left out on that basis:

- **Superseded pipeline generations.** The `contactless/` pipeline shipped here (v28 base +
  v13/v35 context resolution + the v24 abstention gate, per the static trace of
  `run_contactless_segment_dialogues_v35.py`) is one of many. An entire earlier "profile pipeline"
  generation (`profile_pipeline_v1`-`v7s`, `profile_matcher_v1`-`v3`, `profile_assignment_v4/v7/v7s`,
  `profile_segmenter_v4`-`v7`, `profile_native_matcher_v4/v6/v7/v8`, `segmenter.py`,
  `retriever_tfidf.py`, and more) and seventeen other versions of the segmentation runner script
  (`run_contactless_segment_dialogues_v9`...`v37` other than `v35`) are not reachable from the
  frozen entrypoint and are excluded. `scripts/compute_tutor_scores.py` is explicitly marked
  superseded in its own docstring ("SUPERSEDED BY THE SINGLE-JUDGE POLICY") and is excluded on that
  stated basis, not an inferred one.
- **Two unrelated side-projects with negative or inconclusive results**, per the project's own
  records, and never referenced by the paper's Method or Results: an adapter-training service
  (`src/seg_eval/adapter_training/`) and a ColBERT late-interaction retrieval experiment
  (`src/seg_eval/late_interaction/`). Also excluded: `src/seg_eval/style_coverage/` and
  `src/seg_eval/console/` (an internal Streamlit-based run-orchestration UI for driving HPC jobs,
  unrelated to any reported number).
- **~150 one-off scripts**: HPC/cluster job orchestration and Slurm log tooling, exploratory
  ablations beyond the ones this paper reports, adapter-training and ColBERT experiment drivers,
  library-merge/leakage-audit tooling from a separate library-construction line of work, and
  annotation-portal (Flask) code for a human-annotation web app. None of these are imported by, or
  needed to run, anything in `REPRODUCE.md`.
- **`local_audits/` (~60 documents) and `docs/publication_final/` / `docs/evaluation_completeness_v2/`**
  (the internal audit and manuscript-editing trail): excluded wholesale. These document real
  internal back-and-forth, corrections, and superseded claims that were never meant to be read as a
  public record, and two of them (`docs/evaluation_completeness_v2/{annotation_site,e5_site}/`)
  contain a `secrets.env` with a live study key and admin token for the annotation web app. Their
  *useful* content — the claim-to-evidence mapping that says which script and artifact backs each
  reported number — was extracted and rewritten into `REPRODUCE.md` rather than shipped verbatim.
  This is a judgment call (the brief that produced this repository asked for it to be flagged): if
  the authors want a redacted version of any specific internal document preserved for transparency,
  it was not silently destroyed — it still exists in the private source repository.
- **Two scripts contained a hardcoded personal path** during this cleanup and were fixed rather
  than shipped as-is: `scripts/baseline_kc_assignment.py` (a `--model` default pointing at a local
  temp-cache path) now defaults to the HuggingFace model id `BAAI/bge-small-en-v1.5`.
  `scripts/build_merged_library_v3.py` and `scripts/run_library_swap_segmentation.py` reference a
  second, unrelated local project directory outside this repository and were excluded (they are
  also outside the traced reproduction set).

## License

Code is licensed under Apache-2.0 (see `LICENSE`); see `data/README.md` for data licensing. See
`CITATION.cff` for how to cite this work — it currently carries a placeholder author list because
the paper is under double-blind review; update it at camera-ready.
