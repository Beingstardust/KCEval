# Reproducing every reported number

This file maps each number reported in the paper to the exact command that reproduces it, the
frozen artifact the command reads, and what it prints. All commands below are run from the
repository root with a plain Python 3.10+ interpreter (**no third-party dependencies are needed**
for anything in this file unless stated otherwise) after

```bash
pip install -e .
```

Every script in this file is read-only over `data/` except for the small `data/processed/` and
`docs/` side-files it writes back out as its own recomputed evidence (named in each entry below);
none of them makes a network or model call. Each one re-derives its numbers from the frozen judge
/ segmentation outputs under `data/` and, where the paper states an assertion (e.g. "the all-ten
basis reproduces the preregistered replication numbers exactly"), the script checks that assertion
itself and raises if it fails, rather than only printing a number.

For the parts of the pipeline that genuinely need a GPU and a served model (building the KC index
from scratch, calling the LLM judge), see [Reproducing the pipeline / judge layer](#pipeline--judge-layer-requires-gpu)
below; the frozen outputs those runs produced are what every command in this section reads.

## Table 1 — KC indexing against expert gold (`tab:kc`)

75.5% exact / 85.3% acceptable agreement, ARI 0.729 vs 0.437 (induced) vs 0.354 (naive), 73/74/50/75
groups, the 14.7-point margin and both McNemar tests.

```bash
python scripts/analyze_v35_vs_b3_mcnemar.py
python scripts/analyze_kc_induction_comparison.py
```

Reads `data/gold/kc_assignment_baselines.json` (the frozen per-exchange predictions of the deployed
pipeline and both naive baselines), `data/gold/kc_induction/{assign,induce}_responses.jsonl` (the
LLM-induction comparator), `data/gold/gold_struct_dm{1,2}.json` (expert gold), and
`data/processed/segmentation_runs/*/v35_20260812/exchange_assignments.jsonl`. The McNemar script
prints "reconstruction against frozen pooled exact rates: EXACT" before reporting the paired test.
`kc_assignment_baselines.json` is itself a frozen artifact of `scripts/baseline_kc_assignment.py`,
which needs a sentence-transformers encoder (`BAAI/bge-small-en-v1.5`) to regenerate from scratch —
see the pipeline section below.

## Table 2 — Human agreement on the focal dimensions (`tab:agree`)

Error reasoning AC2 0.831 / 0.743, error localisation 0.135 / 0.531, error detection 0.135 / 0.345,
scaffolding quality 0.346 / 0.350, plus the weighted-kappa row and the inside/outside-CI table.

```bash
python scripts/analyze_second_annotator_agreement.py
python scripts/build_e5_agreement_table.py
```

The first regenerates `data/gold/e5_second_annotator/agreement.json` from the three raw annotation
sources (`data/gold/human_validation_extracted/judge_validation_extracted.jsonl` = A1,
`data/gold/e5_second_annotator/a2_export_20260910.jsonl` = A2, and the frozen Selene judge
responses under `data/processed/judge_responses/selene_fixed_dm{1,2}_*`). The second reads that
file plus `data/gold/dimension_informativeness.json` and prints
"Judge-H1 reproduction ... EXACT on every dimension" before the table.

## Table 3 — Preregistered corruption replication (`tab:repl`)

Segment drop −0.0955 [−0.153, −0.040] p=0.0008; localisation recall 0.812, precision 0.722;
cross-scope widening +0.0853 [+0.028, +0.145] p=0.0037.

```bash
python scripts/analyze_scope_routing_ablation.py
```

Reads `data/gold/e7_results.json` (the frozen preregistered result) and the two replication console
runs `data/processed/console_runs/E7S_hv_p` (clean) / `E7S_hv_q` (corrupted). Prints
"all10 basis reproduces the preregistered H7.1 and H7.3 exactly (asserted)" — this is a hard
assertion, not a description — before showing the routed-basis rows described below. Writes
`data/processed/publication_final/scope_routing_ablation.json`.

## Table 4 — Final tutor score (`tab:final-tutor`)

The α ∈ {0.5 … 1.0} grid for REF / PED-DEG / FACT-CORR / DLG-DEG / EXP-REG, and the replication's
micro-only score (0.8658 clean, 0.7969 corrupted).

```bash
python scripts/compute_option_b_final_tutor_score.py
```

Reads the five tutor-variant evaluation packets and Selene judge responses under
`data/processed/evaluation_packets/v35_tv_{a,b,c,d,e}_eval_*` and
`data/processed/judge_responses/v35_tv_{a,b,c,d,e}_*_selene*`. Writes
`data/processed/publication_final/option_b_final_tutor_score.{csv,json}`.

## Section 5.4 — Cross-scope discrepancy, sensitivity and external control

```bash
python scripts/sp1_resolution_decomposition_full.py      # the 78% widening decomposition, +0.0187 (p=0.231)
python scripts/analyze_beta_sensitivity_and_caps.py        # beta grid, cap-binding counts (35/36 of 42)
python scripts/build_dimension_validation_matrix.py         # prerequisite for the next line (see above)
python scripts/analyze_construct_scope_contrast.py          # per-construct owned-vs-other-scope table
python scripts/analyze_e1_bea.py                            # BEA external control: -0.081, +0.010, -0.096, -0.140
python scripts/compute_routed_tutor_scores.py               # full routed-vs-unrouted numbers (see below)
```

`compute_routed_tutor_scores.py` is the script the paper's §5.4 refers to as: *"Full
routed-versus-unrouted numbers, including the per-dimension breakdown, are provided in the project
repository."* It recomputes every development tutor arm's `D_segment`, `D_ARC` and `T` under five
bases (the published judge scalar, the uncapped all-ten mean, and routed v1/v2 with and without the
trust cap) and under both the as-built and topic-merged arc definitions, so the effect of routing
and of the trust cap can be read apart from each other. Nothing here is a new model call; it is a
deterministic re-read of the same frozen Selene responses `compute_option_b_final_tutor_score.py`
uses.

`analyze_e1_bea.py` additionally needs `data/gold/e1_raw/{arm_a,arm_b}_responses.jsonl` and
`data/processed/judge_prompts/bea_arm_a_20260902/bea_gold.jsonl`, all included; the BEA 2025 dev
split itself is vendored at `data/input/bea_mrbench_v3/` under its original CC BY-SA 4.0 licence
(see `data/README.md`).

## Supporting / sanity scripts

```bash
python scripts/build_dimension_validation_matrix.py  # per-dimension validity table over all ten
                                                       # rubric dimensions (extends Table 2 to the
                                                       # six non-focal dimensions); run this BEFORE
                                                       # analyze_construct_scope_contrast.py, which
                                                       # asserts against the CSV this writes to
                                                       # docs/publication_final/ (created fresh; not
                                                       # the internal docs/publication_final tree
                                                       # from the source project, which is not part
                                                       # of this repository — see below)
python scripts/closeout_status.py    # cross-checks G1-G8 (segmentation, per-scope dimensions,
                                      # KC criteria, aggregation, traceability) against the frozen
                                      # artifacts; not a paper table, included for transparency
```

## Pipeline / judge layer (requires GPU)

The scripts above all recompute paper numbers from frozen, already-collected judge and
segmentation output. Two further layers of code are included as the actual implementation of the
method described in Section 3, but are **not required to reproduce any reported number** and need
a GPU plus a served model to execute:

- **KC indexing / segmentation** (`src/seg_eval/contactless/`, entrypoint
  `scripts/run_contactless_segment_dialogues_v35.py`) needs `bge-small-en-v1.5` (dense retrieval),
  `bge-reranker-v2-m3` (cross-encoder rerank) and `Qwen3-4B-Instruct-2507` (tie-breaker). Install
  the `segmentation` extra (`pip install -e ".[segmentation]"`). **The tie-breaker gate defaults in
  this script are the published ones** (`--tie-breaker-max-pipeline-margin 999.0`,
  `--tie-breaker-min-probability-margin 0.5`); the script's own docstring explains a real
  regression this project hit when those defaults were wrong, and the correct values are asserted
  at startup — do not override them without a reason.
- **Judge calling** (`src/seg_eval/evaluation_judge/`, entrypoints
  `scripts/run_selene_rubric_judge.py`, `run_kc_criterion_judge.py`,
  `run_grounded_factuality_judge.py` / `run_selene_factuality_judge.py`, `run_macro_judge.py`, and
  the matching `build_*_prompts.py` / `validate_*_responses.py` scripts) needs a running vLLM
  OpenAI-compatible endpoint serving **Selene-1-Llama-3.3-70B**, revision
  `37ad448f3d1ab7579cbccb0d8418b2808573a30b`, bf16, tensor-parallel 2, greedy decoding
  (`temperature=0, top_p=1`). Do not redistribute the weights; pull them from HuggingFace at the
  pinned revision. `data/gold/judge_weight_identity_sofja_vs_ants.json` records how the exact
  weights snapshot was verified across two serving hosts, with host-local account paths redacted.

Two script pairs in this layer look duplicated and are shipped as-is rather than silently
collapsed — see `README.md`'s "Flagged for the author" section:
`run_grounded_factuality_judge.py` vs `run_selene_factuality_judge.py`, and
`validate_segment_judge_responses.py` vs `validate_segment_judge_responses_v2.py`.

## Verification actually performed when this repository was built

Every command in the "pure recompute" sections above was executed from this cleaned checkout (not
merely inspected) and its output compared, value by value, against the numbers quoted in the
manuscript's Abstract, Table 1 (`tab:kc`), Table 2 (`tab:agree`), Table 3 (`tab:repl`), Table 4
(`tab:final-tutor`) and Section 5.4. All matched exactly. The GPU-dependent pipeline/judge layer was
not re-executed (no GPU in the build environment); instead, every local import in
`run_contactless_segment_dialogues_v35.py`'s dependency tree was resolved statically with no error,
and the frozen artifacts it would have produced (`kc_assignment_baselines.json`, the
`exchange_assignments.jsonl` files, the Selene judge response files) were checksummed against the
values recorded in this project's internal audit trail at freeze time and matched exactly.
