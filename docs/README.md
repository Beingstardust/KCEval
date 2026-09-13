# docs/

Everything in this directory is **generated output**, written by scripts in `../scripts/` when
this repository's reproduction was verified (see `../REPRODUCE.md`), at the exact paths those
scripts are hardcoded to write to:

- `publication_final/02_VALIDATED_CORE_DIMENSIONS.csv` — written by
  `scripts/build_dimension_validation_matrix.py`.
- `open_problems_research/SP-01_DECOMPOSITION.{json,csv}` — written by
  `scripts/sp1_resolution_decomposition_full.py`.
- `open_problems_research/_repro/v35_vs_b3_mcnemar.json` — written by
  `scripts/analyze_v35_vs_b3_mcnemar.py`.

This is **not** the source project's internal `docs/publication_final/` or `local_audits/` trail —
those (an internal manuscript-editing and audit history, including material never meant for public
release) were deliberately excluded from this repository; see the "What was left out" section of
the top-level `README.md`. These four files exist here only because that is where the shipped
scripts happen to write their own recomputed output, and are committed so a reader can compare
their own rerun's output against what was checked when this repository was built.
