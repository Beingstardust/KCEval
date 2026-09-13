"""Build the three-way agreement table for the paper (H1-H2, Judge-H1, Judge-H2). NO MODEL CALLS.

    python scripts/build_e5_agreement_table.py

The paper's published agreement table (judge against one annotator) was produced by
analyze_dimension_informativeness.py. This reuses that module's own metric functions and repeats
its pairing convention exactly:

  * a unit contributes to a dimension only if that dimension is IN SCOPE for the unit's family;
  * "not applicable" is a category, not a missing value (canon maps None and N/A alike to "N/A"),
    so n is the number of units where the rater expressed any judgement including n/a;
  * weighted Cohen's kappa and Gwet's AC2 over the ordinal levels present;
  * entropy is computed on the JUDGE's labels, as an informativeness diagnostic.

The published Judge-H1 column is recomputed here and asserted against the artifact the paper was
written from, so the two new columns cannot silently use a different convention.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_dimension_informativeness as di  # noqa: E402
from seg_eval.evaluation_judge.response_contract_v2 import validate_response_rows  # noqa: E402
from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS  # noqa: E402
from seg_eval.evaluation_packets.mixed_granularity_builder_v1 import (  # noqa: E402
    ARC_PARTIAL_DIMENSIONS, LOCAL_PARTIAL_DIMENSIONS,
)

# (packet dir, prompt dir, response dir) — the same four families the published analysis used.
# Only CONTRACT-VALID responses carry judge_output, which is what makes n 76 local / 37 arc.
FAM = [
    ("v35_dm1_eval_20260820/kc_segment_local", "v35_dm1_kc_segment_local_20260820",
     "selene_fixed_dm1_kc_segment_local_20260830"),
    ("v35_dm1_eval_20260820/topic_rollup_arc", "v35_dm1_topic_rollup_arc_20260820",
     "selene_fixed_dm1_topic_rollup_arc_20260830"),
    ("v35_dm2_eval_20260827/kc_segment_local", "v35_dm2_kc_segment_local_20260827",
     "selene_fixed_dm2_kc_segment_local_20260830"),
    ("v35_dm2_eval_20260827/topic_rollup_arc", "v35_dm2_topic_rollup_arc_20260827",
     "selene_fixed_dm2_topic_rollup_arc_20260830"),
]
A1_PATH = REPO / "data/gold/human_validation_extracted/judge_validation_extracted.jsonl"
A2_PATH = REPO / "data/gold/e5_second_annotator/a2_export_20260910.jsonl"
PUBLISHED = REPO / "data/gold/dimension_informativeness.json"

LABEL = {
    "error_detection": "Error detection", "error_reasoning": "Error reasoning",
    "error_localisation": "Error localisation", "solution_control": "Solution control",
    "scaffolding_quality": "Scaffolding quality", "actionability_moving_forward": "Actionability",
    "local_coherence_relevance": "Local coherence",
    "clarity_cognitive_load": "Clarity / cognitive load",
    "student_level_calibration": "Student-level calibration",
    "proactive_clarification": "Proactive clarification",
}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def canon(v) -> str:
    if v is None:
        return "N/A"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else str(f)


def main() -> None:
    judge = {}
    for pk, pr, rd in FAM:
        v, _e, _s = validate_response_rows(
            prompt_rows=jl(REPO / "data/processed/judge_prompts" / pr / "segment_judge_prompts.jsonl"),
            packet_rows=jl(REPO / "data/processed/evaluation_packets" / pk / "segment_evaluation_packets.jsonl"),
            response_rows=jl(REPO / "data/processed/judge_responses" / rd / "judge_responses.jsonl"))
        for r in v:
            judge[r["segment_id"]] = r["judge_output"]
    a1 = {r["review_item_id"]: r for r in jl(A1_PATH)}
    a2 = {r["item_id"]: (r.get("scores") or {}) for r in jl(A2_PATH)}
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))["partial_dimensions"]

    rows, mismatches, INSIDE = [], [], []
    for d in PARTIAL_DIMENSIONS:
        p_hh, p_j1, p_j2, jvals = [], [], [], []
        for rid, h in a1.items():
            scope = LOCAL_PARTIAL_DIMENSIONS if h.get("family") == "local" else ARC_PARTIAL_DIMENSIONS
            if d not in scope:
                continue
            hv = (h.get("human_dimension_scores") or {}).get(d)
            if not hv or rid not in judge:
                continue
            h1 = canon(hv)
            jv = canon((judge[rid].get("dimension_scores") or {}).get(d))
            p_j1.append((h1, jv))
            jvals.append(jv)
            sv = (a2.get(rid) or {}).get(d)
            if sv is not None:
                h2 = canon(None if str(sv).upper() in ("NA", "N/A") else sv)
                p_hh.append((h1, h2))
                p_j2.append((h2, jv))
        if not p_j1:
            continue

        def stat(pairs):
            return (di.weighted_kappa(pairs), di.gwet_ac2(pairs), len(pairs))

        wk1, ac1, n1 = stat(p_j1)
        wkh, ach, nh = stat(p_hh)
        wk2, ac2_, n2 = stat(p_j2)
        ent = di.entropy(jvals)
        # Interval on the human-human estimate: the template permits the "comparable to
        # inter-annotator agreement" wording only where the judge estimate falls inside it.
        import random as _r
        hh_lo, hh_hi = di.boot_ci(p_hh, di.gwet_ac2, _r.Random(di.SEED)) if p_hh else (None, None)
        hhk_lo, hhk_hi = di.boot_ci(p_hh, di.weighted_kappa, _r.Random(di.SEED)) if p_hh else (None, None)
        inside_ac = (hh_lo is not None and ac2_ is not None and hh_lo <= ac2_ <= hh_hi)
        inside_k = (hhk_lo is not None and wk2 is not None and hhk_lo <= wk2 <= hhk_hi)
        INSIDE.append((d, hh_lo, hh_hi, ac2_, inside_ac, hhk_lo, hhk_hi, wk2, inside_k))

        # reproduction check against the artifact the paper was written from
        ref = (pub.get(d) or {}).get("reliability") or {}
        for name, got, want in (("n", n1, ref.get("n")),
                                ("w-kappa", wk1, ref.get("weighted_kappa")),
                                ("AC2", ac1, ref.get("gwet_ac2"))):
            if want is None or got is None:
                continue
            if abs(round(got, 4) - want) > 5e-4:
                mismatches.append("%s %s: recomputed %.4f vs published %.4f" % (d, name, got, want))
        rows.append((d, nh, n1, wkh, ach, wk1, ac1, wk2, ac2_, ent))

    print("Judge-H1 reproduction against data/gold/dimension_informativeness.json: %s"
          % ("EXACT on every dimension" if not mismatches else "MISMATCH"))
    for m in mismatches:
        print("   " + m)
    print()
    print("%-26s %5s %5s | %7s %6s | %7s %6s | %7s %6s | %6s"
          % ("dimension", "n_hh", "n_j", "kap_HH", "AC2", "kap_JH1", "AC2", "kap_JH2", "AC2", "entr"))
    for d, nh, n1, wkh, ach, wk1, ac1, wk2, ac2_, ent in rows:
        f = lambda x: " n/d " if x is None else "%+.3f" % x
        print("%-26s %5d %5d | %7s %6s | %7s %6s | %7s %6s | %6.3f"
              % (LABEL.get(d, d), nh, n1, f(wkh), f(ach), f(wk1), f(ac1), f(wk2), f(ac2_), ent))

    print("\n%% LaTeX rows (H1-H2 | Judge-H1 | Judge-H2), same convention as the published table")
    for d, nh, n1, wkh, ach, wk1, ac1, wk2, ac2_, ent in rows:
        g = lambda x: "--" if x is None else ("$%+.3f$" % x).replace("+", "\\phantom{-}") \
            if x >= 0 else "$%+.3f$" % x
        print("%-26s & %d & %s & %s & %s & %s & %s & %s & %.3f \\\\"
              % (LABEL.get(d, d), n1, g(wkh), g(ach), g(wk1), g(ac1), g(wk2), g(ac2_), ent))

    print("")
    print("%% Judge-H2 inside the human-human interval? (permitted-wording test)")
    print("  %-26s %-26s %-9s %-7s %-26s %-9s %s" % ("dimension", "HH AC2 95% CI", "J-H2 AC2", "inside",
                                                     "HH kappa 95% CI", "J-H2 kap", "inside"))
    for d, lo, hi, v, ok, klo, khi, kv, kok in INSIDE:
        fmt = lambda a, b: "--" if a is None else "[%+.3f, %+.3f]" % (a, b)
        print("  %-26s %-26s %-9s %-7s %-26s %-9s %s"
              % (LABEL.get(d, d), fmt(lo, hi), "--" if v is None else "%+.3f" % v, "YES" if ok else "no",
                 fmt(klo, khi), "--" if kv is None else "%+.3f" % kv, "YES" if kok else "no"))


if __name__ == "__main__":
    main()
