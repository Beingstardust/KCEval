"""Tier 2 — frozen expert KC library vs LLM-induced per-dialogue KCs (DialogueKT-style).

Three systems, one gold, one set of metrics:

  v35          our five-pass pipeline against the frozen expert-reviewed 159-KC library
  B2_dense     naive top-1 dense retrieval against the same library (Tier 1's baseline)
  induced      KCs generated per dialogue by the same judge model, then tagged (DialogueKT-style)

The induced system uses its OWN vocabulary, so nothing can be scored by string match against our
gold KC ids. Demanding it emit our ids would hand it the library that is under test. Every system is
therefore scored on the PARTITION it induces -- which exchanges it groups together as sharing a
knowledge component -- against the expert gold partition.

Metrics, all label-space free and all symmetric in the label vocabulary:
  Adjusted Rand Index   chance-corrected pair agreement; 0 = chance, 1 = identical partition
  V-measure             harmonic mean of homogeneity and completeness
  homogeneity           each predicted group contains only one gold KC
  completeness          each gold KC lands in only one predicted group

Homogeneity and completeness are reported separately because they fail in opposite directions and
the failure mode is the interesting part: a system that emits one group per exchange is perfectly
homogeneous and useless, and one that emits a single group is perfectly complete and useless. Group
counts are printed beside every score so those degenerate cases are visible rather than hidden
inside a V-measure.

Exact-match accuracy is reported ONLY for the two systems that predict in our label space, and is
never compared against the induced baseline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import comb, log
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_DM1 = ("data/processed/segmentation_runs/"
        "dm1_human_student_llm_tutor_teaching_rich_dialogue_textonly_20260618T174736Z")
_DM2 = "data/processed/segmentation_runs/dm2_gold_holdout_dialogue_20260811"

CORPORA = {
    "dm1": {"gold": "data/gold/gold_struct_dm1.json",
            "v35": f"{_DM1}/v35_20260812/exchange_assignments.jsonl"},
    "dm2": {"gold": "data/gold/gold_struct_dm2.json",
            "v35": f"{_DM2}/v35_20260812/exchange_assignments.jsonl"},
}


def jl(p) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def adjusted_rand(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    tbl = Counter(zip(a, b))
    ra, rb = Counter(a), Counter(b)
    idx = sum(comb(v, 2) for v in tbl.values())
    sa = sum(comb(v, 2) for v in ra.values())
    sb = sum(comb(v, 2) for v in rb.values())
    total = comb(n, 2)
    exp = sa * sb / total if total else 0.0
    mx = (sa + sb) / 2
    return (idx - exp) / (mx - exp) if mx != exp else 0.0


def _entropy(labels: list) -> float:
    n = len(labels)
    return -sum((c / n) * log(c / n) for c in Counter(labels).values() if c) if n else 0.0


def _mutual_info(a: list, b: list) -> float:
    n = len(a)
    if not n:
        return 0.0
    tbl, ra, rb = Counter(zip(a, b)), Counter(a), Counter(b)
    return sum((c / n) * log((c / n) / ((ra[x] / n) * (rb[y] / n)))
               for (x, y), c in tbl.items() if c)


def v_measure(gold: list, pred: list) -> tuple[float, float, float]:
    hg, hp = _entropy(gold), _entropy(pred)
    mi = _mutual_info(gold, pred)
    homogeneity = 1.0 if hg == 0 else mi / hg
    completeness = 1.0 if hp == 0 else mi / hp
    v = (0.0 if homogeneity + completeness == 0
         else 2 * homogeneity * completeness / (homogeneity + completeness))
    return homogeneity, completeness, v


def load_gold(path) -> dict[int, str]:
    raw = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    return {int(k.split("_")[1]): v.get("primary") for k, v in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-responses", default="data/gold/kc_induction/assign_responses.jsonl")
    ap.add_argument("--induce-responses", default="data/gold/kc_induction/induce_responses.jsonl")
    ap.add_argument("--tier1", default="data/gold/kc_assignment_baselines.json")
    ap.add_argument("--out", default="data/gold/kc_induction_comparison.json")
    args = ap.parse_args()

    induced_sets = {}
    for r in jl(REPO_ROOT / args.induce_responses):
        c = str(r.get("segment_id", "")).split("::")[-1]
        if r.get("json_extracted"):
            induced_sets[c] = json.loads(r["raw_response_text"])["knowledge_components"]

    assign: dict[str, dict[int, str]] = defaultdict(dict)
    for r in jl(REPO_ROOT / args.assign_responses):
        if not r.get("json_extracted"):
            continue
        parts = str(r.get("segment_id", "")).split("::")
        if len(parts) < 3:
            continue
        corpus, ex = parts[1], parts[2]
        assign[corpus][int(ex.split("_")[1])] = json.loads(r["raw_response_text"])["kc_name"]

    results: dict = {
        "question": "does a frozen, expert-reviewed KC library beat generating KCs per dialogue?",
        "comparator": "DialogueKT-style (LAK 2025) LLM-induced per-dialogue KCs",
        "why_partition_metrics": "the induced system uses its own vocabulary; string-matching it "
                                 "against our gold ids would require handing it the library under "
                                 "test",
        "model_held_constant": "induction uses Selene-70B, the same model KCEval judges with, so "
                               "the contrast is library-vs-generated rather than model-vs-model",
        "corpora": {},
    }

    print("=" * 96)
    print("TIER 2 - FROZEN EXPERT LIBRARY vs LLM-INDUCED KCs (partition agreement)")
    print("=" * 96)

    pooled = defaultdict(lambda: {"gold": [], "pred": []})
    for name, cfg in CORPORA.items():
        gold = load_gold(cfg["gold"])
        v35 = {int(r["exchange_index"]): r.get("resolved_kc_id")
               for r in jl(REPO_ROOT / cfg["v35"])}
        ind = assign.get(name, {})

        # Tier 1's naive baseline enters the same partition space, so all three systems are scored
        # by one metric rather than v35 being compared to two different things.
        b2 = {}
        t1 = REPO_ROOT / args.tier1
        if t1.exists():
            preds = (json.loads(t1.read_text(encoding="utf-8"))["corpora"]
                     .get(name, {}).get("predictions", {}).get("B2_dense_plain", {}))
            b2 = {int(k): v for k, v in preds.items()}

        # v35 aligns to gold at a +1 index offset (established in the Tier 1 scorer and the
        # reproduction check); the induced labels were built from the same exchange files as v35.
        systems = {"v35": v35, "B2_dense": b2, "induced": ind}
        block = {"induced_kc_count": len(induced_sets.get(name, [])),
                 "gold_kc_count": len(set(v for v in gold.values() if v))}
        for sysname, pred in systems.items():
            best = None
            for off in (0, 1, 2):
                idx = [i for i in sorted(pred) if (i + off) in gold and gold[i + off]
                       and pred[i] is not None]
                if len(idx) < 5:
                    continue
                g = [gold[i + off] for i in idx]
                p = [pred[i] for i in idx]
                ari = adjusted_rand(g, p)
                if best is None or ari > best["ari"]:
                    h, c, v = v_measure(g, p)
                    best = {"offset": off, "n": len(idx), "ari": ari, "homogeneity": h,
                            "completeness": c, "v_measure": v,
                            "predicted_groups": len(set(p)), "gold_groups": len(set(g))}
                    pooled[sysname]["gold"] = pooled[sysname]["gold"] + [f"{name}:{x}" for x in g]
                    pooled[sysname]["pred"] = pooled[sysname]["pred"] + [f"{name}:{x}" for x in p]
            block[sysname] = best
        results["corpora"][name] = block

        print(f"\n  {name}   gold KCs {block['gold_kc_count']}   "
              f"induced KCs {block['induced_kc_count']}")
        print(f"    {'system':10s} {'n':>4s} {'ARI':>7s} {'V':>7s} {'homog':>7s} {'compl':>7s} "
              f"{'groups':>7s}")
        for sysname in ("v35", "B2_dense", "induced"):
            b = block.get(sysname)
            if not b:
                print(f"    {sysname:10s}  (no usable predictions)")
                continue
            print(f"    {sysname:10s} {b['n']:4d} {b['ari']:7.3f} {b['v_measure']:7.3f} "
                  f"{b['homogeneity']:7.3f} {b['completeness']:7.3f} {b['predicted_groups']:7d}")

    print(f"\n  POOLED (dm1 + dm2)")
    print(f"    {'system':10s} {'n':>4s} {'ARI':>7s} {'V':>7s} {'homog':>7s} {'compl':>7s} "
          f"{'groups':>7s}")
    results["pooled"] = {}
    for sysname in ("v35", "B2_dense", "induced"):
        g, p = pooled[sysname]["gold"], pooled[sysname]["pred"]
        if not g:
            continue
        ari = adjusted_rand(g, p)
        h, c, v = v_measure(g, p)
        results["pooled"][sysname] = {"n": len(g), "ari": ari, "v_measure": v, "homogeneity": h,
                                      "completeness": c, "predicted_groups": len(set(p)),
                                      "gold_groups": len(set(g))}
        print(f"    {sysname:10s} {len(g):4d} {ari:7.3f} {v:7.3f} {h:7.3f} {c:7.3f} "
              f"{len(set(p)):7d}")

    # Uncertainty on the ARI gap. A with-replacement bootstrap is the wrong tool for a
    # pair-counting metric: duplicated exchanges manufacture pairs that never existed and inflate
    # agreement. A leave-one-out jackknife perturbs the sample without creating spurious pairs.
    #
    # n_dialogue = 2. This interval is CONDITIONAL on these two dialogues and supports no claim
    # about a population of dialogues, exactly as doc 44 requires elsewhere in this project.
    for other in ("induced", "B2_dense"):
        if "v35" not in results["pooled"] or other not in results["pooled"]:
            continue
        gv, pv = pooled["v35"]["gold"], pooled["v35"]["pred"]
        go, po = pooled[other]["gold"], pooled[other]["pred"]
        if len(gv) != len(go):
            results[f"v35_minus_{other}_ari"] = {
                "difference": results["pooled"]["v35"]["ari"] - results["pooled"][other]["ari"],
                "jackknife": "not computed: the two systems scored different exchange sets",
            }
            continue
        full = adjusted_rand(gv, pv) - adjusted_rand(go, po)
        n = len(gv)
        vals = []
        for i in range(n):
            gv_i = gv[:i] + gv[i + 1:]
            pv_i = pv[:i] + pv[i + 1:]
            go_i = go[:i] + go[i + 1:]
            po_i = po[:i] + po[i + 1:]
            vals.append(adjusted_rand(gv_i, pv_i) - adjusted_rand(go_i, po_i))
        mean = sum(vals) / n
        se = (sum((v - mean) ** 2 for v in vals) * (n - 1) / n) ** 0.5
        lo, hi = full - 1.96 * se, full + 1.96 * se
        results[f"v35_minus_{other}_ari"] = {
            "difference": full, "jackknife_se": se, "ci_95": [lo, hi],
            "excludes_zero": lo > 0 or hi < 0, "n_exchanges": n,
            "interval_type": "conditional leave-one-out jackknife over exchanges; n_dialogue = 2, "
                             "so this supports no claim about a population of dialogues",
        }
        print(f"\n  v35 ARI - {other} ARI = {full:+.3f}   jackknife SE {se:.3f}   "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]   "
              f"{'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")

    tier1 = REPO_ROOT / args.tier1
    if tier1.exists():
        t = json.loads(tier1.read_text(encoding="utf-8"))
        results["tier1_reference"] = {
            "v35_exact": t["pooled"]["v35"]["exact"],
            "best_naive_exact": max(t["pooled"][k]["exact"] for k in
                                    ("B0_random", "B1_majority", "B2_dense_plain",
                                     "B3_dense_profile")),
            "note": "exact-match accuracy exists only in our label space and is NOT comparable to "
                    "the induced system; carried here for context only",
        }

    (REPO_ROOT / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWROTE {REPO_ROOT / args.out}")


if __name__ == "__main__":
    main()
