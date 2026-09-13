#!/usr/bin/env python3
"""CLI runner for the v35 contactless pipeline (v28 + generative tie-breaker).

With --tie-breaker-model-path omitted the pipeline is behaviourally identical to v28, which is what
makes the A/B honest.

THE TIE-BREAKER GATE DEFAULTS ARE THE PUBLISHED ONES, AND THAT IS A CORRECTION
------------------------------------------------------------------------------
This script used to default to ``--tie-breaker-max-pipeline-margin 0.25`` and
``--tie-breaker-min-probability-margin 0.15``. **No published result was ever produced with those
values.** Both frozen v35 gold runs record 999.0 and 0.5 in their own ``summary.json`` -- the v34a
"ungated" design, in which the tie-breaker is consulted on every exchange and the order-stability
guard is the sole abstention mechanism. That deletion is what took the project from 69.6% to 75.5%
exact.

The gap was not cosmetic. Re-running the deployed profiles with the old defaults resolved 6 of 60
dm1 exchanges differently and scored 80.0/80.0 against the published 81.7/85.0 -- silently, with no
error, because a gated-out exchange simply keeps its pre-tie-breaker answer. Anyone re-running v35
with the old defaults would have got a different and worse pipeline than the one in the report.

The defaults now match the published configuration, and the effective settings are echoed at
startup so a mismatch can never again be invisible. Pass the old values explicitly if you want the
gated variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seg_eval.contactless.pipeline_v35 import run_contactless_pipeline_v35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-path", required=True)
    ap.add_argument("--dialogues-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--internal-top-k", type=int, default=60)
    ap.add_argument("--no-cross-encoder", action="store_true")
    ap.add_argument("--no-dense-retrieval", action="store_true")
    ap.add_argument("--tie-breaker-model-path", default=None)
    ap.add_argument("--tie-breaker-top-k", type=int, default=5)
    ap.add_argument("--tie-breaker-max-pipeline-margin", type=float, default=999.0)
    ap.add_argument("--tie-breaker-min-probability", type=float, default=0.5)
    ap.add_argument("--tie-breaker-min-probability-margin", type=float, default=0.5)
    ap.add_argument("--no-prior-context", action="store_true")
    args = ap.parse_args()

    if args.tie_breaker_model_path:
        print("  tie-breaker: %s" % args.tie_breaker_model_path)
        print("    top_k=%d  min_probability=%.3g  max_pipeline_margin=%.4g  "
              "min_probability_margin=%.3g"
              % (args.tie_breaker_top_k, args.tie_breaker_min_probability,
                 args.tie_breaker_max_pipeline_margin, args.tie_breaker_min_probability_margin))
        if args.tie_breaker_max_pipeline_margin < 999.0:
            print("    NOTE: gated. The published v35 runs used max_pipeline_margin=999.0 "
                  "(ungated) and min_probability_margin=0.5.")
    else:
        print("  tie-breaker: OFF (pipeline is behaviourally v28)")

    result = run_contactless_pipeline_v35(
        profiles_path=args.profiles_path,
        dialogues_jsonl=args.dialogues_jsonl,
        out_dir=args.out_dir,
        top_k=args.top_k,
        internal_top_k=args.internal_top_k,
        use_cross_encoder=not args.no_cross_encoder,
        use_dense_retrieval=not args.no_dense_retrieval,
        tie_breaker_model_path=args.tie_breaker_model_path,
        tie_breaker_top_k=args.tie_breaker_top_k,
        tie_breaker_max_pipeline_margin=args.tie_breaker_max_pipeline_margin,
        tie_breaker_min_probability=args.tie_breaker_min_probability,
        tie_breaker_min_probability_margin=args.tie_breaker_min_probability_margin,
        use_prior_context=not args.no_prior_context,
    )
    print("V35_PIPELINE_COMPLETE")
    print(f"OUT_DIR={result['out_dir']}")
    for k, v in result["summary"].items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
