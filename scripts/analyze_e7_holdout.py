"""E7 — the three pre-registered hypotheses (doc 45 §9), tested by doc 44's paired method.

H7.1  factual sensitivity        D_segment(H2) < D_segment(H1)          one-sided, alpha = 0.05
H7.2  localisation               factuality recall lower bound > base rate
H7.3  resolution interaction     R(H2) > R(H1),  R = D_ARC - D_segment  [the central claim]

Doc 44: bootstrap 10,000 resamples of the exchange ids, percentile interval, seed 1234; exact paired
permutation by sign-flip, 100,000 draws, "two-sided unless a direction is pre-registered". All three
are directional, so the decision p-values here are one-sided; the two-sided p is reported beside
them. No multiplicity correction (doc 45 §9): three pre-registered, distinct, directional
hypotheses, all reported regardless of outcome.

H7.2 deliberately reuses dm1's operationalisation from ``analyze_factuality_localisation.py``
verbatim -- SEGMENT-level precision and recall, a flag being ``derived_verdict == "contradicted"``.
Doc 45 §9 anchors the comparison to dm1's reported 0.812/0.867, so a different operationalisation
would make that comparison meaningless. Per-error recall over the 16 injected errors is reported
alongside as a secondary descriptive figure, because doc 45 §7 sizes the Wilson interval on n = 16.

Inference is CONDITIONAL on the dm2 dialogue. n_dialogue = 1, so no resampling scheme here supports
generalisation across dialogues, tutors or students. E7 confirms the mechanism on independent
material; it does not establish population generality. Both facts are printed together.
"""
from __future__ import annotations

import argparse
import json
import random
from math import sqrt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED = 1234
N_BOOT = 10_000
N_PERM = 100_000
RECONSTRUCTION_TOLERANCE = 1e-9


def jl(p) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bare(exchange_id: str) -> str:
    """`E7S_hv_p::ex_0008` -> `ex_0008`, so the two arms pair on a common key."""
    return str(exchange_id).split("::")[-1]


def family_contributions(segments: list[dict], family: str):
    """Exact per-exchange decomposition of a family score (doc 44 §3).

    ``b_s`` is read from the provenance rather than recomputed from ``p_s``/``k_s``: it is the
    authoritative post-blend, post-trust-cap value the aggregator actually used, and recomputing it
    would silently diverge if the cap interacts with the blend. Each exchange in segment s carries
    ``b_s / N`` because the aggregator already weights segments by their exchange count.
    """
    rows = [(seg["b_s"], [bare(i) for i in seg.get("member_exchange_ids") or []])
            for seg in segments
            if seg.get("family") == family and seg.get("b_s") is not None
            and seg.get("member_exchange_ids")]
    total = sum(len(ids) for _, ids in rows)
    if not total:
        return {}, None
    contrib: dict[str, float] = {}
    for b, ids in rows:
        for e in ids:
            contrib[e] = contrib.get(e, 0.0) + b / total
    return contrib, sum(b * len(ids) for b, ids in rows) / total


def paired(deltas: dict[str, float], rng: random.Random, direction: str):
    """Bootstrap interval + sign-flip permutation on the summed paired difference.

    `direction` is the pre-registered sign of the alternative ("less" or "greater"); the decision
    p-value is one-sided in that direction, per doc 44's "two-sided unless a direction is
    pre-registered".
    """
    ids = sorted(deltas)
    n = len(ids)
    obs = sum(deltas[i] for i in ids)
    draws = sorted(sum(deltas[rng.choice(ids)] for _ in range(n)) for _ in range(N_BOOT))
    lo, hi = draws[int(0.025 * N_BOOT)], draws[int(0.975 * N_BOOT)]

    vals = [deltas[i] for i in ids]
    ge_abs = ge_signed = 0
    for _ in range(N_PERM):
        t = sum(v if rng.random() < 0.5 else -v for v in vals)
        ge_abs += abs(t) >= abs(obs)
        ge_signed += (t <= obs) if direction == "less" else (t >= obs)
    p_two = (ge_abs + 1) / (N_PERM + 1)
    p_one = (ge_signed + 1) / (N_PERM + 1)
    return {"observed": obs, "ci": [lo, hi], "p_one_sided": p_one, "p_two_sided": p_two,
            "p_one_sided_mc_ci": list(wilson(ge_signed, N_PERM)), "n_pairs": n,
            "direction": direction}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/processed/console_runs/E7S_hv_p")
    ap.add_argument("--corrupt", default="data/processed/console_runs/E7S_hv_q")
    ap.add_argument("--injected", default="data/gold/dm2_holdout_injected_errors.json")
    ap.add_argument("--dm1", default="data/gold/factuality_localisation_tv_c.json")
    ap.add_argument("--out", default="data/gold/e7_results.json")
    args = ap.parse_args()
    rng = random.Random(SEED)

    prov, contrib, score = {}, {}, {}
    for arm, d in (("H1", args.clean), ("H2", args.corrupt)):
        p = json.loads((REPO_ROOT / d / "results" / "score_provenance.json")
                       .read_text(encoding="utf-8"))["provenance"]
        prov[arm] = p
        contrib[arm], score[arm] = {}, {}
        for fam, recorded in (("local", p["d_segment"]), ("arc", p["d_arc"])):
            c, s = family_contributions(p["segments"], fam)
            # The decomposition is the whole basis of the interval; if it does not reproduce the
            # aggregator's own number, every p-value below is meaningless. Refuse rather than report.
            if s is None or abs(s - recorded) > RECONSTRUCTION_TOLERANCE:
                raise SystemExit(f"RECONSTRUCTION FAILED for {arm}/{fam}: {s} vs recorded "
                                 f"{recorded}. Refusing to report inference on a broken decomposition.")
            if abs(sum(c.values()) - s) > RECONSTRUCTION_TOLERANCE:
                raise SystemExit(f"RECONSTRUCTION FAILED for {arm}/{fam}: contributions do not sum "
                                 "to the family score.")
            contrib[arm][fam], score[arm][fam] = c, s

    out: dict = {
        "method": "doc 44 paired exchange resampling + sign-flip permutation",
        "interval_type": "conditional exchange-resampling interval (dm2 only)",
        "n_dialogue": 1, "population_generalisation": False, "seed": SEED,
        "n_bootstrap": N_BOOT, "n_permutations": N_PERM,
        "aggregation": {k: prov["H1"][k] for k in ("alpha", "beta", "rho")},
        "reconstruction_exact": True,
        "family_scores": {a: dict(score[a], R=score[a]["arc"] - score[a]["local"])
                          for a in ("H1", "H2")},
    }

    print("=" * 86)
    print("E7 - HELD-OUT CORRUPTION REPLICATION (dm2, 44 exchanges, 16 injected errors)")
    print("=" * 86)
    print(f"  aggregation: alpha={prov['H1']['alpha']} beta={prov['H1']['beta']} "
          f"rho={prov['H1']['rho']}   reconstruction exact against recorded D_segment/D_ARC")
    for arm, what in (("H1", "clean control"), ("H2", "corrupted")):
        s = score[arm]
        print(f"  {arm} ({what:14s}): D_segment {s['local']:.4f}   D_ARC {s['arc']:.4f}   "
              f"R {s['arc'] - s['local']:+.4f}")

    # ---- H7.1  factual sensitivity ----------------------------------------------------
    shared = sorted(set(contrib["H1"]["local"]) & set(contrib["H2"]["local"]))
    r1 = paired({e: contrib["H2"]["local"][e] - contrib["H1"]["local"][e] for e in shared},
                rng, "less")
    r1["supported"] = r1["observed"] < 0 and r1["p_one_sided"] < 0.05
    out["H7.1"] = r1
    print("\n  H7.1  D_segment(H2) < D_segment(H1)   [factual sensitivity]")
    print(f"        H2-H1 = {r1['observed']:+.4f}   CI [{r1['ci'][0]:+.4f}, {r1['ci'][1]:+.4f}]"
          f"   p(1-sided)={r1['p_one_sided']:.5f}  (2-sided {r1['p_two_sided']:.5f})"
          f"   n={r1['n_pairs']}")
    print(f"        -> {'SUPPORTED' if r1['supported'] else 'NOT SUPPORTED'}")

    # ---- H7.2  localisation -----------------------------------------------------------
    manifest = json.loads((REPO_ROOT / args.injected).read_text(encoding="utf-8"))
    # The corpus manifest numbers exchanges from 1; the console renumbers from 0. The offset is not
    # assumed -- it was confirmed by verbatim span match on all 16 injected errors, each matching
    # exactly one console exchange (see doc 52). Held here as an explicit, checked constant.
    inj_console = {"ex_%04d" % (r["exchange_index"] - 1) for r in manifest["injected_errors"]}
    err_by_console = {}
    for r in manifest["injected_errors"]:
        err_by_console.setdefault("ex_%04d" % (r["exchange_index"] - 1), []).append(r)

    def localisation(run_dir: str) -> dict:
        """dm1's segment-level confusion matrix, computed for one arm."""
        seg_ex = {p["segment_id"]: {bare(i) for i in p.get("member_exchange_ids") or []}
                  for p in jl(REPO_ROOT / run_dir / "evaluation_packets/kc_segment_local"
                              / "segment_evaluation_packets.jsonl")}
        flagged = {r["segment_id"]: r.get("derived_verdict") == "contradicted"
                   for r in jl(REPO_ROOT / run_dir / "judge_responses/factuality"
                               / "grounded_factuality_responses.jsonl")}
        tp = fp = fn = tn = 0
        caught = total_err = 0
        per_category: dict[str, list[int]] = {}
        for sid, ex in seg_ex.items():
            has = bool(ex & inj_console)
            hit = flagged.get(sid, False)
            tp += has and hit
            fn += has and not hit
            fp += (not has) and hit
            tn += (not has) and not hit
            for e in ex & inj_console:
                for r in err_by_console[e]:
                    caught += hit
                    total_err += 1
                    c = per_category.setdefault(r["category"], [0, 0])
                    c[0] += hit
                    c[1] += 1
        n_seg = tp + fp + fn + tn
        base = (tp + fn) / n_seg if n_seg else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        covered = set().union(*seg_ex.values()) if seg_ex else set()
        return {
            "segments": n_seg, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "base_rate": base,
            "flagged_total": tp + fp,
            "precision": prec, "precision_ci": list(wilson(tp, tp + fp)),
            "recall": rec, "recall_ci": list(wilson(tp, tp + fn)),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "enrichment": prec / base if base else None,
            "per_error_recall": caught / total_err if total_err else 0.0,
            "per_error_recall_ci": list(wilson(caught, total_err)),
            "errors_caught": caught, "errors_total": total_err,
            "per_category_hit_rate": {k: {"caught": v[0], "total": v[1]}
                                      for k, v in sorted(per_category.items())},
            "injected_exchanges_not_in_any_segment": sorted(inj_console - covered),
        }

    def show(label: str, r: dict) -> None:
        print(f"        {label}: segments {r['segments']}   "
              f"containing an injected error {r['tp'] + r['fn']}   "
              f"base rate {r['base_rate']:.1%}   flags raised {r['flagged_total']}")
        print(f"                              flagged   not-flagged")
        print(f"          has injected error  {r['tp']:>7d}   {r['fn']:>11d}")
        print(f"          clean               {r['fp']:>7d}   {r['tn']:>11d}")
        print(f"          precision {r['precision']:.3f} "
              f"[{r['precision_ci'][0]:.3f}, {r['precision_ci'][1]:.3f}]   "
              f"recall {r['recall']:.3f} "
              f"[{r['recall_ci'][0]:.3f}, {r['recall_ci'][1]:.3f}]   "
              f"F1 {r['f1']:.3f}   enrichment {r['enrichment'] or 0:.2f}x")

    # H2 carries the pre-registered decision. H1 is the same statistic on material with NO injected
    # errors: every flag it raises is a false positive by construction, so it is the floor that H2's
    # numbers must be read against. It is descriptive and changes no pre-registered rule.
    loc2 = localisation(args.corrupt)
    loc1 = localisation(args.clean)
    h72 = loc2["recall_ci"][0] > loc2["base_rate"]
    dm1 = json.loads((REPO_ROOT / args.dm1).read_text(encoding="utf-8")) \
        if (REPO_ROOT / args.dm1).exists() else None

    out["H7.2"] = dict(
        loc2,
        operationalisation="segment-level, flag = derived_verdict == 'contradicted' "
                           "(identical to dm1's analyze_factuality_localisation.py)",
        clean_arm_false_positive_floor=loc1,
        flags_gained_by_corruption=loc2["flagged_total"] - loc1["flagged_total"],
        dm1_comparison=dm1,
        decision_rule="Wilson lower bound of recall > base rate",
        supported=bool(h72),
    )
    print("\n  H7.2  localisation of the injected errors   [segment-level, as dm1]")
    if loc2["injected_exchanges_not_in_any_segment"]:
        print("        NOTE injected exchange(s) in no scored segment: "
              + ", ".join(loc2["injected_exchanges_not_in_any_segment"]))
    show("H2 corrupted", loc2)
    print(f"        secondary, per-error recall over {loc2['errors_total']} injected errors: "
          f"{loc2['per_error_recall']:.3f} "
          f"[{loc2['per_error_recall_ci'][0]:.3f}, {loc2['per_error_recall_ci'][1]:.3f}]")
    print("\n        control - the same statistic on the arm with NO injected errors, where every")
    print("        flag is a false positive by construction:")
    show("H1 clean    ", loc1)
    print(f"        flags gained by corruption: {loc2['flagged_total']} - "
          f"{loc1['flagged_total']} = {loc2['flagged_total'] - loc1['flagged_total']}")
    if dm1:
        print(f"\n        dm1 for reference: precision {dm1['precision']:.3f}  "
              f"recall {dm1['recall']:.3f}  base rate {dm1['base_rate']:.1%}")
    print(f"        -> {'SUPPORTED' if h72 else 'NOT SUPPORTED'} "
          f"(rule: recall lower bound {loc2['recall_ci'][0]:.3f} > "
          f"base rate {loc2['base_rate']:.3f})")

    # ---- H7.3  resolution interaction -------------------------------------------------
    # R must be computed on a COMMON SUPPORT with a COMMON denominator, or the two arms are not
    # comparable and the difference-in-differences does not estimate R(H2) - R(H1).
    #
    # This is not a stylistic preference. The ARC family covers 40 of 44 exchanges in the clean arm
    # but all 44 in the corrupted arm: corruption gave ex_0010-ex_0013 an ARC unit they did not have
    # when clean. Dividing each arm's ARC contributions by its own family total therefore scales the
    # clean arm by 44/40 relative to the corrupted arm, inflating R(H1) purely as an artifact of
    # normalisation. An earlier version of this script did exactly that and reported DiD = -0.0116,
    # the opposite sign to the family scores it was supposed to decompose. That disagreement is what
    # exposed the error.
    #
    # H7.1 is unaffected: the local family covers all 44 exchanges in both arms, so its contributions
    # already share a denominator.
    b_of: dict[str, dict[str, dict[str, float]]] = {}
    for arm, d in (("H1", args.clean), ("H2", args.corrupt)):
        b_of[arm] = {"local": {}, "arc": {}}
        for seg in prov[arm]["segments"]:
            fam = seg.get("family")
            if fam in b_of[arm] and seg.get("b_s") is not None:
                for i in seg.get("member_exchange_ids") or []:
                    b_of[arm][fam][bare(i)] = float(seg["b_s"])
    support = sorted(set(b_of["H1"]["local"]) & set(b_of["H1"]["arc"])
                     & set(b_of["H2"]["local"]) & set(b_of["H2"]["arc"]))
    if not support:
        raise SystemExit("H7.3: no exchange is covered by both families in both arms")
    n_s = len(support)
    restricted = {arm: {fam: sum(b_of[arm][fam][e] for e in support) / n_s
                        for fam in ("local", "arc")} for arm in ("H1", "H2")}

    did = {e: ((b_of["H2"]["arc"][e] - b_of["H2"]["local"][e])
               - (b_of["H1"]["arc"][e] - b_of["H1"]["local"][e])) / n_s for e in support}

    # The guard that would have caught the normalisation bug on the first run: a paired statistic
    # must reproduce the difference of the scores it claims to decompose. When it does not, it is
    # measuring something else, whatever its p-value says.
    expected = ((restricted["H2"]["arc"] - restricted["H2"]["local"])
                - (restricted["H1"]["arc"] - restricted["H1"]["local"]))
    if abs(sum(did.values()) - expected) > RECONSTRUCTION_TOLERANCE:
        raise SystemExit(f"H7.3 DECOMPOSITION FAILED: paired sum {sum(did.values())} does not "
                         f"reproduce R|S(H2) - R|S(H1) = {expected}. Refusing to report.")

    r3 = paired(did, rng, "greater")
    r3["supported"] = r3["observed"] > 0 and r3["p_one_sided"] < 0.05
    r3["common_support_size"] = n_s
    r3["restricted_family_scores"] = {
        a: dict(restricted[a], R=restricted[a]["arc"] - restricted[a]["local"])
        for a in ("H1", "H2")}
    r3["arc_coverage"] = {a: len(b_of[a]["arc"]) for a in ("H1", "H2")}
    r3["excluded_from_support"] = sorted(
        (set(b_of["H1"]["local"]) | set(b_of["H2"]["local"])) - set(support))
    r3["note"] = ("computed on the common support with a common denominator; see the comment in "
                  "this script for why per-arm family normalisation is invalid here")
    out["H7.3"] = r3
    print("\n  H7.3  R(H2) > R(H1)   [difference-in-differences; the central claim]")
    print(f"        ARC coverage differs by arm: H1 {r3['arc_coverage']['H1']} exchanges, "
          f"H2 {r3['arc_coverage']['H2']}. Restricted to the common support, n={n_s}"
          + (f", excluding {', '.join(r3['excluded_from_support'])}"
             if r3["excluded_from_support"] else ""))
    for a, what in (("H1", "clean control"), ("H2", "corrupted")):
        rr = r3["restricted_family_scores"][a]
        print(f"        {a} ({what:14s})|S: D_segment {rr['local']:.4f}   D_ARC {rr['arc']:.4f}   "
              f"R {rr['R']:+.4f}")
    print(f"        DiD = {r3['observed']:+.4f}   CI [{r3['ci'][0]:+.4f}, {r3['ci'][1]:+.4f}]"
          f"   p(1-sided)={r3['p_one_sided']:.5f}  (2-sided {r3['p_two_sided']:.5f})"
          f"   n={r3['n_pairs']}")
    print(f"        -> {'SUPPORTED' if r3['supported'] else 'NOT SUPPORTED'}")

    passed = [k for k in ("H7.1", "H7.2", "H7.3") if out[k]["supported"]]
    out["supported"] = passed
    print("\n" + "=" * 86)
    print(f"  {len(passed)}/3 pre-registered hypotheses supported: {', '.join(passed) or 'none'}")
    print("  Intervals are CONDITIONAL on the dm2 dialogue. n_dialogue = 1: E7 confirms the")
    print("  mechanism on independent material; it does NOT establish population generality.")

    (REPO_ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWROTE {REPO_ROOT / args.out}")


if __name__ == "__main__":
    main()
