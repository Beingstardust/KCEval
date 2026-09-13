"""E2 — dimension reliability x corpus informativeness.

Two DIFFERENT properties, never conflated (doc 48 §2):

  RELIABILITY     can the dimension be judged consistently against human labels?
  INFORMATIVENESS does this corpus exercise the dimension at all?

A dimension can be pedagogically necessary and empirically unexercised. The permitted wording for
that case is "not demonstrated to discriminate in this corpus" -- never "useless".

Statistics differ by scale. The 10 partial-credit dimensions get ordinal treatment (weighted kappa,
Gwet AC2). The 3 binary trust flags get binary treatment (prevalence, precision/recall, AC1), and
are reported as UNDERPOWERED rather than given manufactured inference when the corpus contains too
few positives.

Frozen artifacts only. No model calls.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS  # noqa: E402
from seg_eval.evaluation_judge.response_contract_v2 import validate_response_rows  # noqa: E402
from seg_eval.evaluation_packets.mixed_granularity_builder_v1 import (  # noqa: E402
    LOCAL_PARTIAL_DIMENSIONS, ARC_PARTIAL_DIMENSIONS,
)

SEED = 1234
N_BOOT = 10_000
MIN_POSITIVES_FOR_INFERENCE = 5


def jl(p) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


# ---------------------------------------------------------------------------
# agreement statistics
# ---------------------------------------------------------------------------

def _levels(pairs: list[tuple[str, str]]) -> list[str]:
    return sorted({v for pair in pairs for v in pair})


def _ordinal_weight(a: str, b: str, levels: list[str]) -> float:
    """Linear disagreement weight on the ordinal scale; 'N/A' is treated as a separate nominal
    category (maximum disagreement with any numeric level), because not-applicable is a different
    kind of answer, not an extreme score."""
    if a == b:
        return 0.0
    if a == "N/A" or b == "N/A":
        return 1.0
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return 1.0
    numeric = sorted({float(x) for x in levels if x != "N/A"})
    span = (max(numeric) - min(numeric)) or 1.0
    return abs(fa - fb) / span


def weighted_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    levels = _levels(pairs)
    n = len(pairs)
    m1, m2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    obs = sum(_ordinal_weight(a, b, levels) for a, b in pairs) / n
    exp = sum(m1[a] * m2[b] * _ordinal_weight(a, b, levels)
              for a in levels for b in levels) / (n * n)
    if exp == 0:
        return None
    return 1.0 - obs / exp


def gwet_ac2(pairs: list[tuple[str, str]]) -> float | None:
    """Gwet's AC2 -- ordinal, weighted, prevalence-robust. Reported ALONGSIDE kappa, never instead
    of it: the peer-reviewed comparison (PubMed 37234937) shows AC1/AC2 remain marginal-dependent in
    the opposite direction, so substituting one for the other trades one bias for another."""
    if not pairs:
        return None
    levels = _levels(pairs)
    q = len(levels)
    if q < 2:
        return None
    n = len(pairs)
    pa = sum(1.0 - _ordinal_weight(a, b, levels) for a, b in pairs) / n
    counts = Counter()
    for a, b in pairs:
        counts[a] += 0.5
        counts[b] += 0.5
    pi = {lv: counts[lv] / n for lv in levels}
    # Gwet AC2:  pe = [Tw / (q(q-1))] * sum_k pi_k (1 - pi_k)
    # An earlier version appended "/(q-1)*q" here, which inflated pe and produced impossible
    # values (AC2 = -3.39). Fixed; AC2 is now bounded as it should be.
    tw = sum(1.0 - _ordinal_weight(a, b, levels) for a in levels for b in levels)
    norm = tw / (q * (q - 1)) if q > 1 else 1.0
    pe = norm * sum(pi[lv] * (1 - pi[lv]) for lv in levels)
    if pe >= 1.0:
        return None
    return (pa - pe) / (1 - pe)


def boot_ci(pairs: list[tuple[str, str]], fn, rng: random.Random,
            n: int = N_BOOT) -> tuple[float | None, float | None]:
    if len(pairs) < 8:
        return (None, None)
    vals = []
    for _ in range(n):
        s = [rng.choice(pairs) for _ in pairs]
        v = fn(s)
        if v is not None:
            vals.append(v)
    if len(vals) < n * 0.5:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


# ---------------------------------------------------------------------------
# informativeness statistics
# ---------------------------------------------------------------------------

def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    c = Counter(values)
    n = len(values)
    return max(0.0, -sum((v / n) * math.log2(v / n) for v in c.values() if v))


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gold/dimension_informativeness.json")
    args = ap.parse_args()
    rng = random.Random(SEED)

    FAM = [
        ("v35_dm1_eval_20260820/kc_segment_local", "v35_dm1_kc_segment_local_20260820",
         "selene_fixed_dm1_kc_segment_local_20260830", "local"),
        ("v35_dm1_eval_20260820/topic_rollup_arc", "v35_dm1_topic_rollup_arc_20260820",
         "selene_fixed_dm1_topic_rollup_arc_20260830", "arc"),
        ("v35_dm2_eval_20260827/kc_segment_local", "v35_dm2_kc_segment_local_20260827",
         "selene_fixed_dm2_kc_segment_local_20260830", "local"),
        ("v35_dm2_eval_20260827/topic_rollup_arc", "v35_dm2_topic_rollup_arc_20260827",
         "selene_fixed_dm2_topic_rollup_arc_20260830", "arc"),
    ]
    judge: dict[str, dict] = {}
    for pk, pr, rd, _fam in FAM:
        v, _e, _s = validate_response_rows(
            prompt_rows=jl(REPO_ROOT / "data/processed/judge_prompts" / pr / "segment_judge_prompts.jsonl"),
            packet_rows=jl(REPO_ROOT / "data/processed/evaluation_packets" / pk / "segment_evaluation_packets.jsonl"),
            response_rows=jl(REPO_ROOT / "data/processed/judge_responses" / rd / "judge_responses.jsonl"))
        for r in v:
            judge[r["segment_id"]] = r["judge_output"]

    human = {r["review_item_id"]: r for r in
             jl(REPO_ROOT / "data/gold/human_validation_extracted/judge_validation_extracted.jsonl")}

    # tutor-variant outputs, for between-arm variance under known perturbation
    arms: dict[str, dict[str, dict]] = {}
    for t in "abc":
        arms[f"tv_{t}"] = {}
        for fam in ("kc_segment_local", "topic_rollup_arc"):
            pk = f"v35_tv_{t}_eval_20260829/{fam}"
            pr = rd = f"v35_tv_{t}_{fam.replace('kc_segment_local','kc_segment_local').replace('topic_rollup_arc','topic_rollup_arc')}_20260829"
            v, _e, _s = validate_response_rows(
                prompt_rows=jl(REPO_ROOT / "data/processed/judge_prompts" / pr / "segment_judge_prompts.jsonl"),
                packet_rows=jl(REPO_ROOT / "data/processed/evaluation_packets" / pk / "segment_evaluation_packets.jsonl"),
                response_rows=jl(REPO_ROOT / "data/processed/judge_responses" / rd / "judge_responses.jsonl"))
            for r in v:
                arms[f"tv_{t}"][r["segment_id"]] = r["judge_output"]

    def canon(v) -> str:
        if v is None:
            return "N/A"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        return str(int(f)) if f == int(f) else str(f)

    results = {"partial_dimensions": {}, "trust_flags": {},
               "seed": SEED, "n_bootstrap": N_BOOT,
               "note": "reliability and informativeness are separate properties; see doc 48 §2"}

    print("=" * 100)
    print("E2 — DIMENSION RELIABILITY x CORPUS INFORMATIVENESS")
    print("=" * 100)
    print(f"\n{'dimension':30s} {'n':>4s} {'agree':>6s} {'w-kappa':>8s} {'AC2':>7s} "
          f"{'entropy':>8s} {'modal':>6s} {'btw-arm var':>11s}")

    for d in PARTIAL_DIMENSIONS:
        pairs = []
        for rid, h in human.items():
            scope = LOCAL_PARTIAL_DIMENSIONS if h.get("family") == "local" else ARC_PARTIAL_DIMENSIONS
            if d not in scope:
                continue
            hv = (h.get("human_dimension_scores") or {}).get(d)
            if not hv or rid not in judge:
                continue
            pairs.append((canon(hv), canon((judge[rid].get("dimension_scores") or {}).get(d))))
        if not pairs:
            continue

        raw = sum(a == b for a, b in pairs) / len(pairs)
        wk, ac2 = weighted_kappa(pairs), gwet_ac2(pairs)
        wk_lo, wk_hi = boot_ci(pairs, weighted_kappa, rng)
        ac_lo, ac_hi = boot_ci(pairs, gwet_ac2, rng)

        judge_vals = [b for _, b in pairs]
        ent = entropy(judge_vals)
        modal = Counter(judge_vals).most_common(1)[0][1] / len(judge_vals)
        na_rate = sum(1 for v in judge_vals if v == "N/A") / len(judge_vals)

        arm_means = []
        for arm, out in arms.items():
            vals = [float(x) for x in
                    ((o.get("dimension_scores") or {}).get(d) for o in out.values())
                    if x is not None]
            if vals:
                arm_means.append(sum(vals) / len(vals))
        btw = variance(arm_means)

        results["partial_dimensions"][d] = {
            "scale": "ordinal_0_0.5_1",
            "reliability": {"n": len(pairs), "raw_agreement": round(raw, 4),
                            "weighted_kappa": None if wk is None else round(wk, 4),
                            "weighted_kappa_ci": [wk_lo, wk_hi],
                            "gwet_ac2": None if ac2 is None else round(ac2, 4),
                            "gwet_ac2_ci": [ac_lo, ac_hi],
                            "human_marginals": dict(Counter(a for a, _ in pairs)),
                            "judge_marginals": dict(Counter(judge_vals))},
            "informativeness": {"entropy_bits": round(ent, 4),
                                "modal_class_proportion": round(modal, 4),
                                "na_rate": round(na_rate, 4),
                                "between_arm_variance": round(btw, 6),
                                "arm_means": [round(m, 4) for m in arm_means]},
        }
        f = lambda x: "  n/d" if x is None else f"{x:+.3f}"
        print(f"{d:30s} {len(pairs):>4d} {raw:>6.3f} {f(wk):>8s} {f(ac2):>7s} "
              f"{ent:>8.3f} {modal:>6.3f} {btw:>11.6f}")

    # --- trust flags: binary scale, separate treatment ------------------------------------
    print(f"\n{'trust flag (binary)':40s} {'n':>4s} {'judge pos':>10s} {'prevalence':>11s} {'status':>14s}")
    for flag in TRUST_FLAGS:
        vals = []
        for sid, o in judge.items():
            v = (o.get("trust_flags") or {}).get(flag)
            if v is not None:
                vals.append(int(v))
        pos = sum(vals)
        prev = pos / len(vals) if vals else 0.0
        underpowered = pos < MIN_POSITIVES_FOR_INFERENCE
        arm_rates = []
        for arm, out in arms.items():
            av = [int(x) for x in ((o.get("trust_flags") or {}).get(flag) for o in out.values())
                  if x is not None]
            if av:
                arm_rates.append(sum(av) / len(av))
        results["trust_flags"][flag] = {
            "scale": "binary",
            "n_scored": len(vals), "judge_positives": pos, "prevalence": round(prev, 4),
            "between_arm_variance": round(variance(arm_rates), 6),
            "arm_positive_rates": [round(r, 4) for r in arm_rates],
            "inference_status": "UNDERPOWERED_NO_INFERENCE" if underpowered else "reportable",
            "reason": (f"only {pos} positive(s) in {len(vals)} scored units; "
                       f"below the pre-declared minimum of {MIN_POSITIVES_FOR_INFERENCE}. "
                       "Reliability statistics are undefined rather than estimated.")
            if underpowered else "sufficient positives for binary reliability statistics",
        }
        status = "UNDERPOWERED" if underpowered else "reportable"
        print(f"{flag:40s} {len(vals):>4d} {pos:>10d} {prev:>11.4f} {status:>14s}")

    # --- 2x2 classification ----------------------------------------------------------------
    # Informativeness has TWO independent signals and they must not be ANDed into one verdict:
    #   corpus entropy      -- does the NATURAL corpus vary on this dimension?
    #   perturbation response -- does it move under the DESIGNED tv_a/b/c degradation?
    # A dimension can be flat on the natural corpus yet respond strongly to designed degradation.
    # That is "not exercised by this corpus", which is materially different from "measures nothing".
    print("\n" + "=" * 100)
    print("CLASSIFICATION")
    print("  reliability   : weighted-kappa bootstrap CI lower bound > 0")
    print("  corpus variety: entropy >= 0.5 bits on the human-labelled corpus")
    print("  perturbation  : between-arm variance > 1e-3 under the known tv_a/b/c degradation")
    print("=" * 100)
    grid: dict[str, list[str]] = {}
    for d, r in results["partial_dimensions"].items():
        lo, _hi = r["reliability"]["weighted_kappa_ci"]
        reliable = lo is not None and lo > 0
        inf = r["informativeness"]
        varied = inf["entropy_bits"] >= 0.5
        responds = inf["between_arm_variance"] > 1e-3
        key = (("reliable" if reliable else "unreliable") + "|" +
               ("varied" if varied else "flat") + "|" +
               ("responds" if responds else "inert"))
        grid.setdefault(key, []).append(d)
        r["classification"] = key
        r["informativeness"]["corpus_varied"] = varied
        r["informativeness"]["responds_to_perturbation"] = responds
        r["reliability"]["reliable"] = reliable

    labels = {
        "reliable|varied|responds": "empirically strong: reliably judged, corpus exercises it, responds to degradation",
        "reliable|flat|responds": "reliably judged and responsive to designed degradation, but NOT exercised by the natural corpus",
        "reliable|flat|inert": "reliably judged, but not demonstrated to discriminate in this corpus",
        "reliable|varied|inert": "reliably judged and varied, but did not move under designed degradation",
        "unreliable|varied|responds": "responds to degradation but evaluator agreement not established",
        "unreliable|flat|responds": "responds to designed degradation only; agreement not established, corpus flat",
        "unreliable|flat|inert": "not demonstrated to discriminate in this corpus, agreement not established",
        "unreliable|varied|inert": "corpus varies but neither agreement nor perturbation response established",
    }
    for k in sorted(grid):
        print(f"\n  [{k}]  {labels.get(k, k)}")
        for d in grid[k]:
            r = results["partial_dimensions"][d]
            print(f"    - {d:30s} (entropy {r['informativeness']['entropy_bits']:.3f}, "
                  f"btw-arm var {r['informativeness']['between_arm_variance']:.5f})")
    results["classification_grid"] = grid
    results["classification_labels"] = labels

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWROTE {out}")
    print("\nWording rule: a dimension in a low-informativeness cell is 'not demonstrated to")
    print("discriminate in this corpus' -- never 'useless'.")


if __name__ == "__main__":
    main()
