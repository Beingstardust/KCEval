"""SP-1 — full decomposition of the resolution gap across every available condition.

STATUS: FORENSIC. Reads frozen judge outputs and frozen console provenance. No model is run, no
score is recomputed by a different rule, no code in the evaluated pipeline is touched.

WHAT IS BEING SEPARATED
-----------------------
The manuscript reads R = D_topic - D_segment as a resolution effect. Two deterministic
transformations are applied to the LOCAL family and never to the TOPIC family:

    KC-specific criteria blend :  b_blend = (1 - beta) * p_s + beta * k_s
    factuality trust cap       :  C_s     = cap(b_blend, factuality verdict)

so R contains a post-processing component that has nothing to do with conversational scope. This
script computes, per condition:

    local  raw          exchange-weighted mean of p_s over local units
    local  post-criteria exchange-weighted mean of b_blend
    local  final         exchange-weighted mean of C_s   (= published D_segment)
    topic  raw           exchange-weighted mean of p_s over topic units
    topic  final         exchange-weighted mean of topic score (= published D_ARC)

and attributes R into: raw-judge difference, criteria contribution, cap contribution.

TWO DATA SOURCES, ONE CONVENTION
--------------------------------
* E7 arms are console runs; their `score_provenance.json` stores p_s, k_s and b_s per unit, where
  b_s is already post-blend AND post-cap. The criteria/cap split is recovered by recomputing the
  blend from the stored p_s/k_s and beta, then attributing the residual to the cap.
* Tutor-variant arms have no console provenance; their judge responses are read through
  `analyze_tutor_variant_ranking.collect_family`, p_s via `weighted_mean_applicable`, and the cap
  via `apply_deterministic_trust_cap` -- the same calls `analyze_grounded_factuality.py` makes.

Exchange weighting is used everywhere, matching the aggregator.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from seg_eval.aggregation.deterministic_aggregation import (  # noqa: E402
    weighted_mean_applicable, exchange_weighted_mean,
)

_spec = importlib.util.spec_from_file_location("_tvr", SCRIPTS / "analyze_tutor_variant_ranking.py")
_tvr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tvr)

CONSOLE_ARMS = {
    "replication_clean":     ("data/processed/console_runs/E7S_hv_p", 0.2),
    "replication_corrupted": ("data/processed/console_runs/E7S_hv_q", 0.2),
}
# beta = 0 for the tutor-variant study (manuscript: beta=0.2 only in the corruption replication)
TV_ARMS = {"reference_REF": "T1_strong",
           "pedagogical_degradation_PED_DEG": "T2_answer_dumping",
           "factual_corruption_FACT_CORR": "T3_subtly_wrong"}


def ew(pairs):
    """exchange-weighted mean of (value, n_exchanges)."""
    num = den = 0.0
    for v, n in pairs:
        if v is None:
            continue
        num += v * n
        den += n
    return (num / den) if den else None


def from_console(run_dir: str, beta: float) -> dict:
    prov = json.loads((REPO_ROOT / run_dir / "results" / "score_provenance.json")
                      .read_text(encoding="utf-8"))["provenance"]
    segs = prov["segments"]
    loc = [s for s in segs if s.get("family") == "local"]
    arc = [s for s in segs if s.get("family") == "arc"]

    def n_of(s):
        return s.get("exchange_count") or len(s.get("member_exchange_ids") or []) or 1

    raw_l, blend_l, final_l, raw_a, final_a = [], [], [], [], []
    n_capped = 0
    for s in loc:
        p, k, b = s.get("p_s"), s.get("k_s"), s.get("b_s")
        n = n_of(s)
        raw_l.append((p, n))
        blend = p if (k is None or p is None) else (1 - beta) * p + beta * k
        blend_l.append((blend, n))
        final_l.append((b, n))
        if b is not None and blend is not None and b < blend - 1e-9:
            n_capped += 1
    for s in arc:
        n = n_of(s)
        raw_a.append((s.get("p_s"), n))
        final_a.append((s.get("b_s"), n))

    return {"local_raw": ew(raw_l), "local_post_criteria": ew(blend_l), "local_final": ew(final_l),
            "topic_raw": ew(raw_a), "topic_final": ew(final_a),
            "n_local": len(loc), "n_topic": len(arc), "n_local_capped": n_capped,
            "n_local_with_k_s": sum(1 for s in loc if s.get("k_s") is not None),
            "n_topic_with_k_s": sum(1 for s in arc if s.get("k_s") is not None),
            "topic_b_equals_p_all": all(
                s.get("b_s") is not None and s.get("p_s") is not None
                and abs(s["b_s"] - s["p_s"]) < 1e-12 for s in arc),
            "beta": beta, "source": "console score_provenance"}


def from_tv(variant: str, beta: float = 0.0) -> dict:
    from seg_eval.aggregation.deterministic_aggregation import apply_deterministic_trust_cap
    fams = _tvr.VARIANTS[variant]
    out = {}
    fact = _load_factuality(variant)
    for fam_key, tag in (("local", "local"), ("arc", "arc")):
        rows, _ = _tvr.collect_family(*fams[fam_key])
        raw, final = [], []
        n_capped = 0
        for r in rows:
            jo = r["judge_output"]
            p = weighted_mean_applicable(jo.get("dimension_scores") or {},
                                         jo.get("dimension_applicability") or {})
            n = max(1, len(r.get("member_exchange_ids") or []))
            raw.append((p, n))
            if fam_key == "local" and p is not None:
                f = fact.get(r["segment_id"])
                res = apply_deterministic_trust_cap(p, f["verdict"] if f else None,
                                                    f["material"] if f else 0)
                final.append((res.capped_score, n))
                if res.cap_applied:
                    n_capped += 1
            else:
                final.append((p, n))
        out[fam_key] = {"raw": ew(raw), "final": ew(final), "n": len(rows), "capped": n_capped}
    return {"local_raw": out["local"]["raw"],
            "local_post_criteria": out["local"]["raw"],   # beta = 0, so blend is identity
            "local_final": out["local"]["final"],
            "topic_raw": out["arc"]["raw"], "topic_final": out["arc"]["final"],
            "n_local": out["local"]["n"], "n_topic": out["arc"]["n"],
            "n_local_capped": out["local"]["capped"],
            "n_local_with_k_s": 0, "n_topic_with_k_s": 0, "topic_b_equals_p_all": True,
            "beta": beta, "source": "judge_responses via collect_family"}


# The factuality run tag the tutor-variant analysis uses (argparse default in
# analyze_grounded_factuality.py). Passing the wrong tag silently returns {"status": "not_run"} and
# no cap is applied anywhere, which would make every tutor-variant row a raw-score row wearing a
# post-cap label. Asserted below rather than trusted.
FACT_TAG = "r4_14b_20260830"


def _load_factuality(variant: str) -> dict:
    """Reuse the loader analyze_grounded_factuality uses, so verdicts match that analysis."""
    spec = importlib.util.spec_from_file_location(
        "_gf", SCRIPTS / "analyze_grounded_factuality.py")
    gf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf)
    fact, stats = gf.load_factuality(variant, FACT_TAG)
    if not fact or stats.get("status") == "not_run":
        raise SystemExit(f"factuality not loaded for {variant} at tag {FACT_TAG}: {stats}")
    return fact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="docs/open_problems_research/SP-01_DECOMPOSITION.json")
    ap.add_argument("--out-csv", default="docs/open_problems_research/SP-01_DECOMPOSITION.csv")
    args = ap.parse_args()

    rows = {}
    for name, (d, beta) in CONSOLE_ARMS.items():
        rows[name] = from_console(d, beta)
    for name, variant in TV_ARMS.items():
        try:
            rows[name] = from_tv(variant)
        except Exception as e:                      # noqa: BLE001
            rows[name] = {"error": f"{type(e).__name__}: {e}"}

    for name, r in rows.items():
        if "error" in r:
            continue
        lr, lc, lf = r["local_raw"], r["local_post_criteria"], r["local_final"]
        tr, tf = r["topic_raw"], r["topic_final"]
        r["R_reported"] = None if (tf is None or lf is None) else tf - lf
        r["R_raw"] = None if (tr is None or lr is None) else tr - lr
        r["criteria_contribution"] = None if (lr is None or lc is None) else lr - lc
        r["cap_contribution"] = None if (lc is None or lf is None) else lc - lf
        if r["R_reported"] is not None and r["R_raw"] is not None:
            r["post_processing_component"] = r["R_reported"] - r["R_raw"]
            r["post_processing_share"] = (
                r["post_processing_component"] / r["R_reported"] if r["R_reported"] else None)

    print("=" * 108)
    print("  SP-1 - RESOLUTION GAP DECOMPOSITION, ALL CONDITIONS   (forensic; frozen artifacts only)")
    print("=" * 108)
    hdr = (f"  {'condition':34s} {'loc raw':>8s} {'loc+crit':>9s} {'loc fin':>8s} "
           f"{'top raw':>8s} {'top fin':>8s} {'R_rep':>8s} {'R_raw':>8s} {'crit':>7s} {'cap':>7s}")
    print(hdr)
    print("  " + "-" * 104)
    for name, r in rows.items():
        if "error" in r:
            print(f"  {name:34s} ERROR: {r['error'][:60]}")
            continue
        f = lambda x: "  n/a  " if x is None else f"{x:+.4f}" if abs(x) < 1 else f"{x:.4f}"
        g = lambda x: "  n/a  " if x is None else f"{x:.4f}"
        print(f"  {name:34s} {g(r['local_raw']):>8s} {g(r['local_post_criteria']):>9s} "
              f"{g(r['local_final']):>8s} {g(r['topic_raw']):>8s} {g(r['topic_final']):>8s} "
              f"{f(r['R_reported']):>8s} {f(r['R_raw']):>8s} "
              f"{f(r['criteria_contribution']):>7s} {f(r['cap_contribution']):>7s}")
    print("  " + "-" * 104)
    print(f"  {'condition':34s} {'n_loc':>6s} {'n_top':>6s} {'capped':>7s} {'k_s loc':>8s} "
          f"{'k_s top':>8s} {'topic b==p all':>15s} {'beta':>5s}")
    for name, r in rows.items():
        if "error" in r:
            continue
        print(f"  {name:34s} {r['n_local']:>6d} {r['n_topic']:>6d} {r['n_local_capped']:>7d} "
              f"{r['n_local_with_k_s']:>8d} {r['n_topic_with_k_s']:>8d} "
              f"{str(r['topic_b_equals_p_all']):>15s} {r['beta']:>5.1f}")

    outj = REPO_ROOT / args.out_json
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps({
        "analysis": "SP-1 resolution-gap decomposition across all available conditions",
        "status": "FORENSIC - frozen artifacts only, no model run, no pipeline change",
        "definitions": {
            "local_raw": "exchange-weighted mean of p_s over local units",
            "local_post_criteria": "exchange-weighted mean of (1-beta)*p_s + beta*k_s",
            "local_final": "exchange-weighted mean of C_s (post-cap) = published D_segment",
            "topic_raw": "exchange-weighted mean of p_s over topic units",
            "topic_final": "exchange-weighted mean of topic score = published D_ARC",
            "R_reported": "topic_final - local_final",
            "R_raw": "topic_raw - local_raw",
            "criteria_contribution": "local_raw - local_post_criteria",
            "cap_contribution": "local_post_criteria - local_final",
            "post_processing_component": "R_reported - R_raw",
        },
        "conditions": rows,
    }, indent=2), encoding="utf-8")

    cols = ["condition", "source", "beta", "n_local", "n_topic", "n_local_capped",
            "n_local_with_k_s", "n_topic_with_k_s", "topic_b_equals_p_all",
            "local_raw", "local_post_criteria", "local_final", "topic_raw", "topic_final",
            "R_reported", "R_raw", "criteria_contribution", "cap_contribution",
            "post_processing_component", "post_processing_share"]
    outc = REPO_ROOT / args.out_csv
    with outc.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, r in rows.items():
            if "error" in r:
                continue
            w.writerow({"condition": name, **{c: r.get(c) for c in cols if c != "condition"}})
    print(f"\n  WROTE {outj}\n  WROTE {outc}")


if __name__ == "__main__":
    main()
