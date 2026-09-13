"""Missing paired test: v35 vs B3 (naive dense retrieval over matching profiles), exact agreement.

WHY THIS EXISTS
---------------
`kc_assignment_baselines.json` records the headline margin as v35 (75.49%) minus **B3** (60.78%)
= +14.7pp, but the only paired significance test it stores is `mcnemar_v35_vs_B2` -- against a
DIFFERENT baseline (B2, 59.80%). The manuscript presents both as one "best naive baseline". They
are not the same comparison, so the margin currently has no paired test attached to it.

This script computes the missing one. It is a FORENSIC analysis of already-frozen predictions:

* no model is run, no embedding recomputed, no matching code touched;
* predictions are read verbatim from the frozen artifact's `corpora.<dm>.predictions` block;
* scoring reproduces `baseline_kc_assignment.score()` exactly -- `exact` means the predicted KC id
  equals gold `primary`; every system's recorded `offset` is 0 in the frozen artifact, which this
  script asserts rather than assumes.

The pre-existing `mcnemar_v35_vs_B2` result remains valid and untouched; it simply describes
v35 vs B2 and must be labelled that way.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CORPORA = {"dm1": "data/gold/gold_struct_dm1.json",
           "dm2": "data/gold/gold_struct_dm2.json"}


def load_gold(path: str) -> dict[int, dict]:
    raw = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    return {int(k.split("_")[1]): v for k, v in raw.items()}


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    """Exact binomial test on the discordant pairs, as the frozen artifact's B2 test uses."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="data/gold/kc_assignment_baselines.json")
    ap.add_argument("--system", default="B3_dense_profile")
    ap.add_argument("--out", default="docs/open_problems_research/_repro/v35_vs_b3_mcnemar.json")
    args = ap.parse_args()

    art = json.loads((REPO_ROOT / args.baselines).read_text(encoding="utf-8"))

    per_corpus, tot = {}, {"both_correct": 0, "v35_only": 0, "other_only": 0, "both_wrong": 0, "n": 0}
    for name, goldpath in CORPORA.items():
        gold = load_gold(goldpath)
        block = art["corpora"][name]
        # The frozen scoring picks the best offset per system; assert it is 0 for both systems here,
        # because a nonzero offset would mean the two systems are indexed against gold differently
        # and a paired table over raw indices would silently compare different exchanges.
        for sysname in (args.system, "v35"):
            off = block[sysname].get("offset")
            if off != 0:
                raise SystemExit(f"{name}/{sysname}: offset {off} != 0; paired table would misalign")

        pv = block["predictions"]["v35"]
        po = block["predictions"][args.system]
        idx = sorted(set(int(i) for i in pv) & set(int(i) for i in po))
        idx = [i for i in idx if i in gold]

        cell = {"both_correct": 0, "v35_only": 0, "other_only": 0, "both_wrong": 0}
        for i in idx:
            g = gold[i].get("primary")
            a = pv[str(i)] == g
            b = po[str(i)] == g
            cell["both_correct" if (a and b) else
                 "v35_only" if a else
                 "other_only" if b else "both_wrong"] += 1
        cell["n"] = len(idx)
        cell["v35_exact_n"] = cell["both_correct"] + cell["v35_only"]
        cell["other_exact_n"] = cell["both_correct"] + cell["other_only"]
        per_corpus[name] = cell
        for k in tot:
            tot[k] += cell[k]

    b, c = tot["v35_only"], tot["other_only"]
    p = exact_mcnemar_two_sided(b, c)
    n = tot["n"]
    v35_exact = (tot["both_correct"] + b) / n
    oth_exact = (tot["both_correct"] + c) / n

    # Cross-check against the frozen artifact so a silent divergence cannot pass unnoticed.
    recorded_v35 = art["pooled"]["v35"]["exact"]
    recorded_oth = art["pooled"][args.system]["exact"]
    ok_v35 = abs(v35_exact - recorded_v35) < 1e-9
    ok_oth = abs(oth_exact - recorded_oth) < 1e-9
    if not (ok_v35 and ok_oth):
        raise SystemExit(f"RECONSTRUCTION FAILED: v35 {v35_exact} vs {recorded_v35}; "
                         f"{args.system} {oth_exact} vs {recorded_oth}")

    out = {
        "analysis": f"paired exact McNemar, v35 vs {args.system}, exact agreement",
        "status": "FORENSIC - reads frozen predictions only; no model run, no code changed",
        "why": "the +14.7pp headline margin is v35 minus B3, but the only stored paired test was "
               "mcnemar_v35_vs_B2 against a different baseline",
        "source_artifact": args.baselines,
        "scoring": "exact = predicted KC id equals gold primary; identical to "
                   "baseline_kc_assignment.score(); all recorded offsets are 0 and are asserted",
        "per_corpus": per_corpus,
        "pooled": {
            "n": n,
            "both_correct": tot["both_correct"],
            "v35_correct_other_wrong": b,
            "v35_wrong_other_correct": c,
            "both_wrong": tot["both_wrong"],
            "discordant": b + c,
            f"v35_exact": v35_exact,
            f"{args.system}_exact": oth_exact,
            "margin_pp": 100.0 * (v35_exact - oth_exact),
            "p_two_sided_exact": p,
            "significant_at_0_05": p < 0.05,
            "test": "exact McNemar (two-sided binomial on discordant pairs)",
        },
        "reconstruction_exact_against_frozen_artifact": True,
    }

    print("=" * 84)
    print(f"  PAIRED EXACT McNEMAR - v35 vs {args.system}   (exact agreement)")
    print("=" * 84)
    for name, cell in per_corpus.items():
        print(f"  {name}: n={cell['n']:3d}  both_correct {cell['both_correct']:3d}  "
              f"v35_only {cell['v35_only']:3d}  other_only {cell['other_only']:3d}  "
              f"both_wrong {cell['both_wrong']:3d}")
    print()
    print(f"  POOLED n={n}")
    print(f"    both correct                 {tot['both_correct']:3d}")
    print(f"    v35 correct / {args.system:<18s} wrong   {b:3d}")
    print(f"    v35 wrong   / {args.system:<18s} correct {c:3d}")
    print(f"    both wrong                   {tot['both_wrong']:3d}")
    print(f"    discordant                   {b + c:3d}")
    print()
    print(f"    v35 exact  {v35_exact*100:.1f}%   {args.system} exact  {oth_exact*100:.1f}%   "
          f"margin +{100*(v35_exact-oth_exact):.1f} pp")
    print(f"    exact two-sided McNemar p = {p:.7f}   "
          f"{'SIGNIFICANT' if p < 0.05 else 'NOT significant'} at 0.05")
    print(f"\n  reconstruction against frozen pooled exact rates: EXACT")

    outp = REPO_ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  WROTE {outp}")


if __name__ == "__main__":
    main()
