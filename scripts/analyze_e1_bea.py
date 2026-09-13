"""E1 — BEA 2025 / MRBench v3 dev-set external criterion validity (doc 46).

Arms:
  A  KCEval frozen rubric, thin input adapter, no KC grounding
  B  plain Selene on BEA's own definitions and label set   [MANDATORY control, doc 46 §2]
  C  published majority baseline                            [figures from doc 46 §2]

Metrics (doc 46 §4): exact macro-F1; lenient macro-F1 (Yes + To some extent vs No); per-class
precision/recall/F1 with particular attention to "To some extent", the minority and hardest class;
confusion matrices; weighted kappa; Gwet AC2; cluster bootstrap over the 300 dev dialogues, 10,000
resamples, seed 1234, IDENTICAL resamples across arms so A-vs-B is paired.

Multiplicity: Bonferroni across the four tracks, alpha = 0.0125 (doc 44 §9).

Comparison hierarchy (doc 46 §5): PRIMARY A vs B; SECONDARY A vs C; the published test leaderboard
and the Fleiss kappa 0.65 human figure are CONTEXT ONLY and are never used as thresholds here.

WHAT THIS CANNOT SHOW (doc 46 §9). Evaluation is on the dev split, which the competing teams
developed on and which KCEval sees zero-shot. No leaderboard placement, no ranking against
competition systems, and no claim that E1 validates segmentation, KC routing, ARC, macro, the trust
cap or attribution -- E1 exercises the rubric/judge layer only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 1234
N_BOOT = 10_000
ALPHA_BONFERRONI = 0.0125

LABELS = ["Yes", "To some extent", "No"]
# Doc 46 §3, frozen mapping. KCEval partial score -> BEA label.
SCORE_TO_LABEL = {"1": "Yes", "0.5": "To some extent", "0": "No"}
TRACK_TO_DIMENSION = {
    "Mistake_Identification": "error_detection",
    "Mistake_Location": "error_localisation",
    "Actionability": "actionability_moving_forward",
    "Providing_Guidance": "scaffolding_quality",
}
# Doc 46 §2, published majority baseline, listed in the document's own track order.
MAJORITY_F1 = {"Mistake_Identification": 0.2827, "Mistake_Location": 0.2450,
               "Providing_Guidance": 0.2313, "Actionability": 0.2198}


def jl(p) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def canon_score(v) -> str | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return str(int(f)) if f == int(f) else str(f)


def macro_f1(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """Unweighted mean of per-class F1 over `labels`, matching the BEA reporting convention."""
    if not pairs:
        return 0.0
    total = 0.0
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / len(labels)


def lenient(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    m = {"Yes": "Yes-ish", "To some extent": "Yes-ish", "No": "No"}
    return [(m[g], m[p]) for g, p in pairs]


def per_class(pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for lab in LABELS:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"support": tp + fn, "predicted": tp + fp, "precision": prec, "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return out


def _wt(a: str, b: str) -> float:
    """Ordinal (linear) weight on the 3-point scale, as doc 44 §11 uses elsewhere."""
    i, j = LABELS.index(a), LABELS.index(b)
    return 1.0 - abs(i - j) / (len(LABELS) - 1)


def weighted_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(_wt(g, p) for g, p in pairs) / n
    gc, pc = Counter(g for g, _ in pairs), Counter(p for _, p in pairs)
    pe = sum(_wt(a, b) * (gc[a] / n) * (pc[b] / n) for a in LABELS for b in LABELS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac2(pairs: list[tuple[str, str]]) -> float | None:
    """Prevalence-robust companion to kappa. Reported together, never alone (doc 44 §11)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(_wt(g, p) for g, p in pairs) / n
    q = len(LABELS)
    marg = Counter()
    for g, p in pairs:
        marg[g] += 1
        marg[p] += 1
    pi = {lab: marg[lab] / (2 * n) for lab in LABELS}
    tw = sum(1.0 - _wt(a, b) for a in LABELS for b in LABELS)
    norm = tw / (q * (q - 1)) if q > 1 else 1.0
    pe = norm * sum(pi[lab] * (1 - pi[lab]) for lab in LABELS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", default="data/gold/e1_raw/arm_a_responses.jsonl")
    ap.add_argument("--arm-b", default="data/gold/e1_raw/arm_b_responses.jsonl")
    ap.add_argument("--gold", default="data/processed/judge_prompts/bea_arm_a_20260902/bea_gold.jsonl")
    ap.add_argument("--out", default="data/gold/bea_external_validation.json")
    args = ap.parse_args()
    rng = random.Random(SEED)

    gold = {g["segment_id"]: g for g in jl(REPO_ROOT / args.gold)}
    if not gold:
        raise SystemExit(f"no gold at {args.gold}")

    def payload(r):
        if not r.get("json_extracted"):
            return None
        try:
            return json.loads(r["raw_response_text"])
        except (TypeError, ValueError, KeyError):
            return None

    pred_a: dict[str, dict[str, str]] = {}
    for r in jl(REPO_ROOT / args.arm_a):
        p = payload(r)
        if not p:
            continue
        scores = p.get("dimension_scores") or {}
        row = {}
        for track, dim in TRACK_TO_DIMENSION.items():
            lab = SCORE_TO_LABEL.get(canon_score(scores.get(dim)) or "")
            if lab:
                row[track] = lab
        pred_a[r["segment_id"]] = row

    pred_b: dict[str, dict[str, str]] = {}
    for r in jl(REPO_ROOT / args.arm_b):
        p = payload(r)
        if not p:
            continue
        pred_b[r["segment_id"]] = {t: p[t] for t in TRACK_TO_DIMENSION if p.get(t) in LABELS}

    # Unit lists per track, per arm, keyed by dialogue for the cluster bootstrap.
    by_dialogue: dict[str, list[str]] = defaultdict(list)
    for sid, g in gold.items():
        by_dialogue[g["conversation_id"]].append(sid)
    dialogues = sorted(by_dialogue)

    results: dict = {
        "design": {"dialogues": len(dialogues), "responses": len(gold),
                   "arms": {"A": "KCEval frozen rubric, no KC grounding",
                            "B": "plain Selene, official BEA definitions (mandatory control)",
                            "C": "published majority baseline (doc 46 §2)"},
                   "seed": SEED, "n_bootstrap": N_BOOT,
                   "multiplicity": f"Bonferroni across 4 tracks, alpha={ALPHA_BONFERRONI}",
                   "split": "dev only; test gold is not public, so no leaderboard claim"},
        "coverage": {"arm_a_units": len(pred_a), "arm_b_units": len(pred_b),
                     "gold_units": len(gold)},
        "tracks": {},
    }

    print("=" * 96)
    print("E1 - BEA / MRBench v3 DEV EXTERNAL CRITERION VALIDITY")
    print("=" * 96)
    print(f"  dialogues {len(dialogues)}   responses {len(gold)}   "
          f"arm A usable {len(pred_a)}   arm B usable {len(pred_b)}")

    boot_index = [[rng.randrange(len(dialogues)) for _ in dialogues] for _ in range(N_BOOT)]

    for track in TRACK_TO_DIMENSION:
        pairs = {"A": [], "B": []}
        # The A-vs-B contrast must run on units BOTH arms labelled. Arm A's rubric can return
        # not_applicable where arm B, asked a direct question, always emits a label, so arm A
        # scores 24-83 fewer units per track. Differencing two macro-F1 values computed on
        # different unit sets is not a paired comparison, and if arm A abstains on the harder
        # cases it would be scored on an easier subset. So the contrast uses the intersection and
        # the per-arm figures below stay on each arm's own full set, labelled as such.
        common: dict[str, dict[str, list]] = {d: {"A": [], "B": []} for d in dialogues}
        per_dialogue: dict[str, dict[str, list]] = {d: {"A": [], "B": []} for d in dialogues}
        n_common = 0
        for sid, g in gold.items():
            gl = g.get(track)
            if gl not in LABELS:
                continue
            la = (pred_a.get(sid) or {}).get(track)
            lb = (pred_b.get(sid) or {}).get(track)
            for arm, pl in (("A", la), ("B", lb)):
                if pl in LABELS:
                    pairs[arm].append((gl, pl))
                    per_dialogue[g["conversation_id"]][arm].append((gl, pl))
            if la in LABELS and lb in LABELS:
                n_common += 1
                common[g["conversation_id"]]["A"].append((gl, la))
                common[g["conversation_id"]]["B"].append((gl, lb))

        blk = {"n_scored": {a: len(pairs[a]) for a in ("A", "B")},
               "n_common_both_arms": n_common,
               "arm_a_abstentions_vs_b": len(pairs["B"]) - n_common,
               "gold_distribution": dict(Counter(g for g, _ in pairs["A"])),
               "majority_baseline_f1": MAJORITY_F1[track]}
        for arm in ("A", "B"):
            blk[arm] = {
                "exact_macro_f1": macro_f1(pairs[arm], LABELS),
                "lenient_macro_f1": macro_f1(lenient(pairs[arm]), ["Yes-ish", "No"]),
                "per_class": per_class(pairs[arm]),
                "weighted_kappa": weighted_kappa(pairs[arm]),
                "gwet_ac2": gwet_ac2(pairs[arm]),
                "predicted_distribution": dict(Counter(p for _, p in pairs[arm])),
                "confusion": {g: dict(Counter(p for gg, p in pairs[arm] if gg == g))
                              for g in LABELS},
            }

        # Paired cluster bootstrap: same dialogue resample for both arms, so the A-vs-B difference
        # is paired and the interval reflects dialogue-level dependence rather than treating 2,476
        # responses as independent.
        diffs, a_s = [], []
        for idx in boot_index:
            pa, pb, fa_full = [], [], []
            for i in idx:
                d = dialogues[i]
                pa.extend(common[d]["A"])
                pb.extend(common[d]["B"])
                fa_full.extend(per_dialogue[d]["A"])
            diffs.append(macro_f1(pa, LABELS) - macro_f1(pb, LABELS))
            a_s.append(macro_f1(fa_full, LABELS))
        diffs.sort(); a_s.sort()
        lo_i, hi_i = int(0.00625 * N_BOOT), int(0.99375 * N_BOOT)  # Bonferroni-adjusted 98.75%
        p_two = 2 * min(sum(1 for d in diffs if d <= 0), sum(1 for d in diffs if d >= 0)) / N_BOOT
        common_a = [p for d in dialogues for p in common[d]["A"]]
        common_b = [p for d in dialogues for p in common[d]["B"]]
        blk["A_vs_B"] = {
            "computed_on": "intersection of units both arms labelled",
            "n": n_common,
            "arm_a_macro_f1_on_common": macro_f1(common_a, LABELS),
            "arm_b_macro_f1_on_common": macro_f1(common_b, LABELS),
            "difference_exact_macro_f1": macro_f1(common_a, LABELS) - macro_f1(common_b, LABELS),
            "ci_98_75": [diffs[lo_i], diffs[hi_i]],
            "bootstrap_p_two_sided": min(1.0, p_two),
            "excludes_zero": diffs[lo_i] > 0 or diffs[hi_i] < 0,
            "interval_note": "98.75% paired cluster-bootstrap interval over dialogues, "
                             "Bonferroni-adjusted for 4 tracks",
        }
        blk["A_vs_C"] = {
            "difference": blk["A"]["exact_macro_f1"] - MAJORITY_F1[track],
            "arm_a_ci_98_75": [a_s[lo_i], a_s[hi_i]],
            "exceeds_majority": a_s[lo_i] > MAJORITY_F1[track],
            "note": "C is a published point figure with no interval of its own, so this compares "
                    "arm A's interval against a constant rather than testing a difference",
        }
        results["tracks"][track] = blk

        print(f"\n  {track}  (n={len(pairs['A'])} scored; gold "
              f"{blk['gold_distribution']})")
        for arm in ("A", "B"):
            b = blk[arm]
            tse = b["per_class"]["To some extent"]
            print(f"    {arm}: exact macro-F1 {b['exact_macro_f1']:.4f}   "
                  f"lenient {b['lenient_macro_f1']:.4f}   "
                  f"w-kappa {b['weighted_kappa']:.3f}   AC2 {b['gwet_ac2']:.3f}")
            print(f"       'To some extent': support {tse['support']:4d}  "
                  f"predicted {tse['predicted']:4d}  P {tse['precision']:.3f}  "
                  f"R {tse['recall']:.3f}  F1 {tse['f1']:.3f}")
        ab = blk["A_vs_B"]
        print(f"    paired on n={ab['n']} both-arm units "
              f"(arm A abstained on {blk['arm_a_abstentions_vs_b']} that B labelled): "
              f"A {ab['arm_a_macro_f1_on_common']:.4f} vs B {ab['arm_b_macro_f1_on_common']:.4f}")
        print(f"    A-B = {ab['difference_exact_macro_f1']:+.4f}  "
              f"CI98.75 [{ab['ci_98_75'][0]:+.4f}, {ab['ci_98_75'][1]:+.4f}]  "
              f"p={ab['bootstrap_p_two_sided']:.4f}  "
              f"{'EXCLUDES 0' if ab['excludes_zero'] else 'includes 0'}")
        ac = blk["A_vs_C"]
        print(f"    A-C = {ac['difference']:+.4f} vs majority {MAJORITY_F1[track]:.4f}  "
              f"{'A exceeds majority' if ac['exceeds_majority'] else 'not established'}")

    # ---- E1-KG: known groups, expert vs novice human tutors -------------------------
    kg = {}
    for track in TRACK_TO_DIMENSION:
        row = {}
        for arm, pred in (("A", pred_a), ("B", pred_b)):
            grp = {}
            for tutor in ("Expert", "Novice"):
                sel = [(gold[s][track], (pred.get(s) or {}).get(track))
                       for s in gold if gold[s]["tutor"] == tutor]
                sel = [(g, p) for g, p in sel if g in LABELS and p in LABELS]
                # "Yes" rate is the natural ordinal summary of predicted pedagogical quality.
                grp[tutor] = {"n": len(sel),
                              "predicted_yes_rate": (sum(1 for _, p in sel if p == "Yes") / len(sel)
                                                     if sel else None),
                              "gold_yes_rate": (sum(1 for g, _ in sel if g == "Yes") / len(sel)
                                                if sel else None)}
            row[arm] = grp
        kg[track] = row
    results["known_groups_E1KG"] = {
        "hypothesis": "expert human tutors should score above novice human tutors",
        "by_track": kg,
        "note": "Novice n is 76 and Expert n is 300, so this is descriptive; it is reported because "
                "doc 47 §4 asks for it at zero extra compute, not as a powered test.",
    }
    print("\n  E1-KG known groups (predicted 'Yes' rate, Expert vs Novice):")
    for track, row in kg.items():
        e_a, n_a = row["A"]["Expert"], row["A"]["Novice"]
        e_b, n_b = row["B"]["Expert"], row["B"]["Novice"]
        def f(x): return "n/a" if x["predicted_yes_rate"] is None else f"{x['predicted_yes_rate']:.3f}"
        print(f"    {track:26s} A: E {f(e_a)} vs N {f(n_a)}   B: E {f(e_b)} vs N {f(n_b)}   "
              f"gold: E {e_a['gold_yes_rate']:.3f} vs N {n_a['gold_yes_rate']:.3f}")

    (REPO_ROOT / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWROTE {REPO_ROOT / args.out}")


if __name__ == "__main__":
    main()
