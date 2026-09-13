"""Tutor scores under versioned scope routing and either arc definition. NO MODEL CALLS.

    python scripts/compute_routed_tutor_scores.py
    python scripts/compute_routed_tutor_scores.py --json data/gold/tutor_variant_ranking/routed_tutor_scores.json

Reads the Selene-70B responses (doc 45 made Selene the rubric judge).

WHY
---
The published T is built from the judge-reported unit scalar, trust_adjusted_score. That scalar
averages all ten rubric dimensions on every unit, segment or arc, so scope routing (which scope
owns which dimension) never reached T. Moving proactive_clarification to arc scope, or giving arcs
a real multi-exchange definition, changes nothing in T through that scalar.

Here every unit score is recomputed from the dimensions its scope owns (scope_routing), segment
units are capped by the deterministic trust cap on the grounded factuality verdict (the cap the
framework specifies in deterministic_aggregation), and the framework formula aggregates:

    D_segment = exchange_weighted_mean(segment unit scores)
    D_ARC     = exchange_weighted_mean(arc unit scores)
    D_micro   = rho * D_segment + (1 - rho) * D_ARC
    T         = alpha * D_micro + (1 - alpha) * M

ATTRIBUTION
-----------
Each basis changes exactly one thing from the one above it, so every movement in T has one cause:

    judge_scalar         trust_adjusted_score as the judge reported it (the published basis)
    all_dims_uncapped    code mean of all ten dimensions, no cap      (is the scalar that mean?)
    routed_v1_uncapped   only the dimensions the scope owns, v1        (routing enforced)
    routed_v1            plus the deterministic factuality cap         (cap)
    routed_v2            proactive_clarification owned by arc scope    (the v2 move)

and each is computed on both arc definitions:

    asbuilt              consecutive-only topic rollup (published)
    merged_min2          topic unit anywhere in the dialogue, at least 2 exchanges

Arc units have no grounded factuality verdict (that pass runs on segments). The documented rule
for a missing verdict is: score unchanged. So the cap never touches an arc unit.

Factuality quotes are validated against the segment tutor text. analyze_grounded_factuality built
its packet path from the run tag, a directory that does not exist, so it validated no quotes; the
number of verdicts this stricter check rejects is printed per arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seg_eval.aggregation.deterministic_aggregation import (  # noqa: E402
    apply_deterministic_trust_cap, combine_local_and_arc, exchange_weighted_mean,
    final_tutor_score, weighted_mean_applicable,
)
from seg_eval.aggregation.pass_loading_v1 import load_macro_score  # noqa: E402
from seg_eval.aggregation.scope_routing import routed_unit_score  # noqa: E402
from seg_eval.evaluation_judge.grounded_factuality_v1 import (  # noqa: E402
    derive_verdict_from_errors, validate_error_response,
)
import analyze_tutor_variant_ranking as az  # noqa: E402

RHO, ALPHA = 0.8, 0.8
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ARMS = {"T1_strong": ("tv_a", "20260829"), "T2_answer_dumping": ("tv_b", "20260829"),
        "T3_subtly_wrong": ("tv_c", "20260829"), "T4_macro_degraded": ("tv_d", "20260909"),
        "T5_expert_register": ("tv_e", "20260909")}
FACT_DIR = {"tv_a": "v35_tv_a_factuality_selene_20260830", "tv_b": "v35_tv_b_factuality_selene_20260830",
            "tv_c": "v35_tv_c_factuality_selene_20260830", "tv_d": "v35_tv_d_factuality_selene_20260909",
            "tv_e": "v35_tv_e_factuality_selene_20260909"}
ARC_DEFS = ("asbuilt", "merged_min2")
MACRO_VERSION = "v1"          # v2 reads the macro pass that includes factual_correctness
ARC_LABEL = {"asbuilt": "asbuilt", "merged_min2": "merged"}
# basis -> None (judge scalar) or (dimension set, cap on?)
BASES = {
    "judge_scalar": None,
    "all_dims_uncapped": ("all", False),
    "routed_v1_uncapped": ("v1", False),
    "routed_v1": ("v1", True),
    "routed_v2": ("v2", True),
}


def load_family(tag: str, stamp: str, fam: str, arcdef: str | None) -> list[dict]:
    if fam == "local":
        pk = f"data/processed/evaluation_packets/v35_{tag}_eval_{stamp}/kc_segment_local"
        pr = f"data/processed/judge_prompts/v35_{tag}_kc_segment_local_{stamp}"
        rs = f"data/processed/judge_responses/v35_{tag}_kc_segment_local_selene"
    elif arcdef == "asbuilt":
        pk = f"data/processed/evaluation_packets/v35_{tag}_eval_{stamp}/topic_rollup_arc"
        pr = f"data/processed/judge_prompts/v35_{tag}_topic_rollup_arc_{stamp}"
        rs = f"data/processed/judge_responses/v35_{tag}_topic_rollup_arc_selene"
    else:
        pk = f"data/processed/evaluation_packets/v35_{tag}_eval_{stamp}_{arcdef}/topic_rollup_arc"
        pr = f"data/processed/judge_prompts/v35_{tag}_topic_rollup_arc_{stamp}_{arcdef}"
        rs = f"data/processed/judge_responses/v35_{tag}_topic_rollup_arc_{arcdef}_selene"
    if not (REPO / rs / "judge_responses.jsonl").exists():
        return []
    rows, _ = az.collect_family(pk, pr, rs)
    return rows


def load_factuality(tag: str, stamp: str):
    """segment_id -> (verdict, material), and a contract-status dict. (None, None) if not run.

    A verdict is accepted when the fields the VERDICT rests on are valid: errors is a well-formed
    list, every tutor_quote appears verbatim in the segment tutor text, severities are legal. The
    Selene pass omits `rationale_short` on every response (payload carries only errors and
    segment_id), which fails the full response contract without touching any of that evidence, so
    it is counted as a deviation and reported rather than silently discarding the whole pass. Any
    other contract error rejects the response.

    Quotes ARE checked here. analyze_grounded_factuality built its packet path from the run tag, a
    directory that does not exist, so it passed allowed_tutor_text=None and verified no quotes.
    """
    f = REPO / "data/processed/judge_responses" / FACT_DIR[tag] / "grounded_factuality_responses.jsonl"
    if not f.exists():
        return None, None
    pk = REPO / (f"data/processed/evaluation_packets/v35_{tag}_eval_{stamp}/kc_segment_local/"
                 "segment_evaluation_packets.jsonl")
    packets = {}
    for line in pk.open(encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            packets[p["segment_id"]] = p
    out = {}
    st = {"accepted": 0, "rejected": 0, "unparseable": 0, "total": 0,
          "missing_rationale_only": 0, "reject_reasons": {}}
    for line in f.open(encoding="utf-8"):
        if not line.strip():
            continue
        st["total"] += 1
        row = json.loads(line)
        sid = row["segment_id"]
        try:
            payload = json.loads(row["raw_response_text"])
        except Exception:
            st["unparseable"] += 1
            continue
        pkt = packets.get(sid, {})
        tutor_text = " ".join(ex.get("tutor_text") or "" for ex in (pkt.get("member_exchanges") or [])
                              if isinstance(ex, dict))
        errs = validate_error_response(payload, expected_segment_id=sid,
                                       allowed_tutor_text=tutor_text or None)
        blocking = [e for e in errs if not e.startswith("rationale_short:")]
        if len(errs) > len(blocking):
            st["missing_rationale_only"] += 1
        if blocking:
            st["rejected"] += 1
            key = blocking[0].split(":")[-1]
            st["reject_reasons"][key] = st["reject_reasons"].get(key, 0) + 1
            continue
        st["accepted"] += 1
        out[sid] = derive_verdict_from_errors(payload)
    return out, st


def macro(tag: str, stamp: str):
    sfx = "_selene" if MACRO_VERSION == "v1" else "_v2_selene"
    f = REPO / ("data/processed/judge_responses/v35_%s_macro_%s%s/macro_judge_responses.jsonl"
                % (tag, stamp, sfx))
    if not f.exists():
        return None
    m, _ = load_macro_score(f)
    return m


def unit_scores(rows: list[dict], family: str, basis: str, fact):
    """[(score, n_exchanges)] and the number of units the cap lowered."""
    spec = BASES[basis]
    out, capped = [], 0
    for r in rows:
        jo = r["judge_output"]
        n = max(1, len(r.get("member_exchange_ids") or []))
        if spec is None:
            v = jo.get("trust_adjusted_score")
            s = float(v) if isinstance(v, (int, float)) else None
        else:
            dims, cap = spec
            if dims == "all":
                s = weighted_mean_applicable(jo.get("dimension_scores") or {},
                                             jo.get("dimension_applicability") or {})
            else:
                s = routed_unit_score(jo, family, dims)
            if s is not None and cap and family == "local" and fact is not None:
                vm = fact.get(r["segment_id"])
                res = apply_deterministic_trust_cap(s, vm[0] if vm else None, vm[1] if vm else 0)
                capped += 1 if (res.cap_applied and res.capped_score < s) else 0
                s = res.capped_score
        if s is not None:
            out.append((s, n))
    return out, capped


def dialogue(local_rows, arc_rows, basis, fact, m) -> dict:
    loc, capped = unit_scores(local_rows, "local", basis, fact)
    arc, _ = unit_scores(arc_rows, "arc", basis, fact)
    d_seg = exchange_weighted_mean(loc) if loc else None
    d_arc = exchange_weighted_mean(arc) if arc else None
    d_micro = combine_local_and_arc(d_seg, d_arc, RHO).value if d_seg is not None else None
    t = ({str(a): final_tutor_score(d_micro, m, a).value for a in ALPHAS}
         if d_micro is not None and m is not None else {})
    return {"D_segment": d_seg, "D_ARC": d_arc, "D_micro": d_micro, "M": m, "T": t,
            "n_local_scored": len(loc), "n_arc_scored": len(arc), "units_capped": capped}


def routed_dialogue_scores(local_rows, arc_rows, version: str, fact) -> dict:
    """Drop-in for the az.dialogue_scores keys closeout_status G7 reads, on the routed basis."""
    r = dialogue(local_rows, arc_rows, "routed_" + version, fact, None)
    return {"d_local_trust_adjusted": r["D_segment"], "d_arc_trust_adjusted": r["D_ARC"],
            "d_micro_trust_adjusted": r["D_micro"]}


def applicable(rows, dim) -> int:
    return sum(1 for r in rows
               if isinstance((r["judge_output"].get("dimension_scores") or {}).get(dim), (int, float)))


def n_exchanges(rows) -> int:
    return sum(len(r.get("member_exchange_ids") or []) for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--macro", choices=("v1", "v2"), default="v1",
                    help="macro rubric version behind M; v2 adds factual_correctness")
    a = ap.parse_args()
    global MACRO_VERSION
    MACRO_VERSION = a.macro
    print("macro rubric: %s" % MACRO_VERSION)

    loaded = {}
    print("%-20s %5s %8s %7s   %-26s %-30s %s" % ("arm", "seg", "arc_old", "arc_new",
                                                  "exchanges covered by arcs", "factuality ok/rej/total", "M"))
    for name, (tag, stamp) in ARMS.items():
        local = load_family(tag, stamp, "local", None)
        if not local:
            print("%-20s no Selene segment responses on disk yet -- arm skipped" % name)
            continue
        arcs = {d: load_family(tag, stamp, "arc", d) for d in ARC_DEFS}
        fact, fst = load_factuality(tag, stamp)
        m = macro(tag, stamp)
        loaded[name] = (local, arcs, fact, m)
        cov = "  ".join("%s %d/%d" % (ARC_LABEL[d], n_exchanges(arcs[d]), n_exchanges(local))
                        for d in ARC_DEFS if arcs[d])
        print("%-20s %5d %8d %7s   %-26s %-30s %s" % (
            name, len(local), len(arcs["asbuilt"]),
            len(arcs["merged_min2"]) if arcs["merged_min2"] else "-", cov or "-",
            ("%d ok / %d rej / %d (%d missing rationale)"
             % (fst["accepted"], fst["rejected"], fst["total"], fst["missing_rationale_only"]))
            if fst else "NOT RUN (no cap on this arm)",
            "%.3f" % m if m is not None else "-"))

    configs = {}
    for d in ARC_DEFS:
        for b in BASES:
            per = {n: dialogue(v[0], v[1][d], b, v[2], v[3]) for n, v in loaded.items() if v[1][d]}
            if per:
                configs["%s/%s" % (ARC_LABEL[d], b)] = per

    names = list(loaded)
    short = {n: n.split("_")[0] for n in names}
    for metric in ("T", "D_segment", "D_ARC"):
        print("\n%s%s" % (metric, " (alpha=%.1f)" % ALPHA if metric == "T" else ""))
        print("  %-28s" % "arcs / basis" + "".join("%8s" % short[n] for n in names)
              + "    T1-T2   T1-T3   order")
        for key, per in configs.items():
            val = {}
            for n in names:
                if n not in per:
                    val[n] = None
                elif metric == "T":
                    val[n] = per[n]["T"].get(str(ALPHA))
                else:
                    val[n] = per[n][metric]
            cells = "".join("%8s" % ("%.3f" % val[n] if val[n] is not None else "-") for n in names)

            def sep(x, y):
                if val.get(x) is None or val.get(y) is None:
                    return "     -  "
                return "%+8.3f" % (val[x] - val[y])
            ranked = sorted([n for n in names if val[n] is not None], key=lambda n: -val[n])
            print("  %-28s%s  %s%s   %s" % (key, cells, sep("T1_strong", "T2_answer_dumping"),
                                             sep("T1_strong", "T3_subtly_wrong"),
                                             " > ".join(short[n] for n in ranked)))

    print("\nfactuality verdicts, and what the deterministic cap does with them:")
    print("  %-20s%10s%8s%14s%9s%9s" % ("arm", "grounded", "minor", "contradicted", "insuff", "capped"))
    for n in names:
        fact = loaded[n][2] or {}
        c = {}
        for v, _m in fact.values():
            c[v] = c.get(v, 0) + 1
        per = configs.get("asbuilt/routed_v1", {}).get(n)
        print("  %-20s%10d%8d%14d%9d%9s" % (
            n, c.get("grounded", 0), c.get("minor_deviation", 0), c.get("contradicted", 0),
            c.get("insufficient_reference", 0), per["units_capped"] if per else "-"))
    print("  minor_deviation caps at 0.9 and covers most units, so it acts as a near-uniform")
    print("  ceiling. The discrimination comes from contradicted: 0.5, or 0.25 when repeated.")

    print("\nproactive_clarification applicable (non-null); routing v2 reads it from arc units:")
    for n, (local, arcs, _, _) in loaded.items():
        merged = arcs["merged_min2"]
        print("  %-20s segment %2d/%-3d  arc asbuilt %2d/%-3d  arc merged %s" % (
            n, applicable(local, "proactive_clarification"), len(local),
            applicable(arcs["asbuilt"], "proactive_clarification"), len(arcs["asbuilt"]),
            ("%d/%d" % (applicable(merged, "proactive_clarification"), len(merged))) if merged
            else "not run yet"))

    if a.json:
        Path(a.json).write_text(json.dumps({"rho": RHO, "alpha": ALPHA, "configs": configs},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
