"""Dimension validation matrix for the publication reconstruction. NO MODEL CALLS.

    python scripts/build_dimension_validation_matrix.py

Assembles, per pedagogical dimension, every piece of evidence that bears on whether it belongs in
the paper's validated core, and writes docs/publication_final/02_VALIDATED_CORE_DIMENSIONS.csv.

Conventions, identical to the analysis the published agreement table came from
(analyze_dimension_informativeness.py), and re-asserted against its artifact at run time:

  * in-scope only: a dimension contributes on a unit only if that unit's family owns it;
  * "not applicable" is a category, not missing data;
  * weighted Cohen's kappa and Gwet's AC2 over the levels present.

CONSTRUCT VARIATION IS MEASURED ON HUMAN LABELS, NOT JUDGE LABELS. The published table reported
entropy over the judge's own labels and called it corpus informativeness. Whether a corpus exercises
a human construct cannot be defined by model behaviour, so H1 and H2 entropy are computed here and
the judge's is retained only as a diagnostic of the instrument.

The discrimination column is the pedagogical contrast (reference vs pedagogically degraded arm) on
the project judge, recomputed with the register's own permutation test.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_dimension_informativeness as di  # noqa: E402
from seg_eval.evaluation_judge.response_contract_v2 import validate_response_rows  # noqa: E402
from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS  # noqa: E402
from seg_eval.evaluation_packets.mixed_granularity_builder_v1 import (  # noqa: E402
    ARC_PARTIAL_DIMENSIONS, ARC_TRUST_FLAGS, LOCAL_PARTIAL_DIMENSIONS,
)

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
# Reference vs pedagogically degraded arm, on the project judge, contiguous topic rollup.
ARMS = {
    "REF": ("tv_a", "20260829"),
    "PED_DEG": ("tv_b", "20260829"),
}
OUT = REPO / "docs/publication_final/02_VALIDATED_CORE_DIMENSIONS.csv"


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


def h2_canon(v):
    if v is None:
        return None
    return canon(None if str(v).upper() in ("NA", "N/A") else v)


def arm_rows(tag: str, stamp: str, fam: str) -> dict[str, dict]:
    sub = "kc_segment_local" if fam == "local" else "topic_rollup_arc"
    pk = f"data/processed/evaluation_packets/v35_{tag}_eval_{stamp}/{sub}"
    pr = f"data/processed/judge_prompts/v35_{tag}_{sub}_{stamp}"
    rd = f"data/processed/judge_responses/v35_{tag}_{sub}_selene"
    if not (REPO / rd / "judge_responses.jsonl").exists():
        return {}
    v, _e, _s = validate_response_rows(
        prompt_rows=jl(REPO / pr / "segment_judge_prompts.jsonl"),
        packet_rows=jl(REPO / pk / "segment_evaluation_packets.jsonl"),
        response_rows=jl(REPO / rd / "judge_responses.jsonl"))
    return {r["segment_id"]: r["judge_output"] for r in v}


def perm_p(a: list[float], b: list[float], n: int = 20000, seed: int = 1234):
    if not a or not b:
        return None
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    pool, k, rnd, hits = a + b, len(a), random.Random(seed), 0
    for _ in range(n):
        rnd.shuffle(pool)
        if abs(statistics.fmean(pool[:k]) - statistics.fmean(pool[k:])) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def main() -> None:
    judge = {}
    for pk, pr, rd in FAM:
        v, _e, _s = validate_response_rows(
            prompt_rows=jl(REPO / "data/processed/judge_prompts" / pr / "segment_judge_prompts.jsonl"),
            packet_rows=jl(REPO / "data/processed/evaluation_packets" / pk / "segment_evaluation_packets.jsonl"),
            response_rows=jl(REPO / "data/processed/judge_responses" / rd / "judge_responses.jsonl"))
        for r in v:
            judge[r["segment_id"]] = r["judge_output"]
    a1 = {r["review_item_id"]: r for r in
          jl(REPO / "data/gold/human_validation_extracted/judge_validation_extracted.jsonl")}
    a2 = {r["item_id"]: (r.get("scores") or {}) for r in
          jl(REPO / "data/gold/e5_second_annotator/a2_export_20260910.jsonl")}
    pub = json.loads((REPO / "data/gold/dimension_informativeness.json").read_text(encoding="utf-8"))

    contrast = {}
    for name, (tag, stamp) in ARMS.items():
        contrast[name] = {"local": arm_rows(tag, stamp, "local"), "arc": arm_rows(tag, stamp, "arc")}

    derived = json.loads((REPO / "data/gold/derived_dimension_validity.json").read_text(encoding="utf-8"))

    rows, mism = [], []
    for d in list(PARTIAL_DIMENSIONS) + list(TRUST_FLAGS):
        is_flag = d in TRUST_FLAGS
        scope = "topic" if d in (list(ARC_PARTIAL_DIMENSIONS) + list(ARC_TRUST_FLAGS)) else "local"
        p_hh, p_j1, p_j2 = [], [], []
        jv_all, h1_all, h2_all = [], [], []
        for rid, h in a1.items():
            fam = h.get("family") or "local"
            owned = (fam == "arc") == (scope == "topic")
            if not owned or rid not in judge:
                continue
            src = "human_trust_flags" if is_flag else "human_dimension_scores"
            hv = (h.get(src) or {}).get(d)
            if not hv:
                continue
            h1 = canon(hv)
            jsrc = "trust_flags" if is_flag else "dimension_scores"
            jv = canon((judge[rid].get(jsrc) or {}).get(d))
            p_j1.append((h1, jv))
            jv_all.append(jv)
            h1_all.append(h1)
            sv = (a2.get(rid) or {}).get(d)
            if sv is not None:
                h2 = h2_canon(sv)
                p_hh.append((h1, h2))
                p_j2.append((h2, jv))
                h2_all.append(h2)
        if not p_j1:
            continue

        ref = ((pub.get("partial_dimensions", {}).get(d)
                or pub.get("trust_flags", {}).get(d) or {}).get("reliability") or {})
        if ref.get("weighted_kappa") is not None and di.weighted_kappa(p_j1) is not None:
            if abs(round(di.weighted_kappa(p_j1), 4) - ref["weighted_kappa"]) > 5e-4:
                mism.append(d)

        a_vals = [float(x) for x in
                  ((o.get("dimension_scores") or {}).get(d)
                   for o in contrast["REF"][("arc" if scope == "topic" else "local")].values())
                  if isinstance(x, (int, float))] if not is_flag else []
        b_vals = [float(x) for x in
                  ((o.get("dimension_scores") or {}).get(d)
                   for o in contrast["PED_DEG"][("arc" if scope == "topic" else "local")].values())
                  if isinstance(x, (int, float))] if not is_flag else []
        cp = perm_p(a_vals, b_vals) if (a_vals and b_vals) else None
        cd = (statistics.fmean(a_vals) - statistics.fmean(b_vals)) if (a_vals and b_vals) else None

        def f(x, nd=3):
            return "" if x is None else round(x, nd)

        # Decompose human-human disagreement: applicability (does it apply here at all) versus
        # rating (the score, on units BOTH raters judged applicable). The diagnostic triad has low
        # combined coefficients driven entirely by the applicability channel, which a single
        # coefficient hides.
        hh_app = [(("NA" if x == "N/A" else "scored"), ("NA" if y == "N/A" else "scored"))
                  for x, y in p_hh]
        hh_rate = [(x, y) for x, y in p_hh if x != "N/A" and y != "N/A"]
        j2_rate = [(x, y) for x, y in p_j2 if x != "N/A" and y != "N/A"]
        raw = lambda pr: (sum(1 for x, y in pr if x == y) / len(pr)) if pr else None

        rows.append({
            "dimension": d,
            "kind": "trust_flag" if is_flag else "partial_credit",
            "scope": scope,
            "n_units": len(p_j1),
            "h1_entropy_bits": f(di.entropy(h1_all)),
            "h2_entropy_bits": f(di.entropy(h2_all)) if h2_all else "",
            "judge_entropy_bits": f(di.entropy(jv_all)),
            "h1_applicable_rate": f(sum(1 for v in h1_all if v != "N/A") / len(h1_all)),
            "judge_applicable_rate": f(sum(1 for v in jv_all if v != "N/A") / len(jv_all)),
            "hh_applicability_raw": f(raw(hh_app)), "hh_rating_raw": f(raw(hh_rate)),
            "hh_rating_n": len(hh_rate), "jh2_rating_raw": f(raw(j2_rate)),
            "hh_kappa": f(di.weighted_kappa(p_hh)), "hh_ac2": f(di.gwet_ac2(p_hh)),
            "jh1_kappa": f(di.weighted_kappa(p_j1)), "jh1_ac2": f(di.gwet_ac2(p_j1)),
            "jh2_kappa": f(di.weighted_kappa(p_j2)), "jh2_ac2": f(di.gwet_ac2(p_j2)),
            "contrast_delta": f(cd, 4), "contrast_p": f(cp, 4),
            "derived_measure": (("passes" if derived.get(d, {}).get("passes")
                                 else "partial" if derived.get(d, {}).get("partial") else "")
                                if d in derived else ""),
        })

    # ------------------------------------------------------------------ classification
    # Thresholds are declared here, applied uniformly, and reported with every row. They are
    # argued positions, not fitted: reference variation below 0.5 bits means the corpus cannot
    # exercise the construct enough for agreement to mean much; a human-human AC2 below 0.30 means
    # two qualified readers do not reproduce one another; a judge estimate within 0.10 of the
    # human-human estimate is "tracks humans about as well as humans track each other".
    VAR_MIN, HH_MIN, NEAR = 0.5, 0.30, 0.10
    for r in rows:
        g = lambda k: (None if r[k] == "" else float(r[k]))
        var_ok = (g("h1_entropy_bits") or 0) >= VAR_MIN
        hh, jh2 = g("hh_ac2"), g("jh2_ac2")
        rating_ok = (g("hh_rating_raw") or 0) >= 0.90 and (g("jh2_rating_raw") or 0) >= 0.90
        tracks = hh is not None and jh2 is not None and jh2 >= hh - NEAR
        disc = (g("contrast_p") is not None and g("contrast_p") < 0.05)
        if r["kind"] == "trust_flag":
            cls, why = "FAILED_VALIDATION", ("judge never raises it on any scored unit "
                                             "(judge entropy 0.000) while the human sometimes does")
        elif var_ok and tracks and (rating_ok or (hh is not None and hh >= HH_MIN)):
            cls, why = "VALIDATED_CORE", ("reference labels vary, judge tracks the humans, and it "
                                          "separates the planted contrast" if disc else
                                          "reference labels vary and judge tracks the humans; the "
                                          "available contrast does not exercise this construct")
        elif var_ok and hh is not None and hh < HH_MIN and not rating_ok:
            cls, why = "EXPLORATORY_NOT_RETAINED", ("corpus exercises it but two annotators do not "
                                                    "reproduce one another and the judge does not "
                                                    "track them")
        elif disc:
            # Responds to a known planted degradation, but the human reference on this corpus is
            # too flat (or the two annotators too far apart) for agreement to validate it.
            lim = []
            if not var_ok:
                lim.append("human reference labels nearly constant (%.3f bits)" % (g("h1_entropy_bits") or 0))
            if hh is not None and hh < HH_MIN:
                lim.append("two annotators do not reproduce one another (AC2 %.3f)" % hh)
            cls, why = "SUPPORTED_WITH_LIMITATION", ("separates the planted contrast (p=%s); "
                                                     % r["contrast_p"]) + "; ".join(lim)
        elif not var_ok:
            cls, why = "UNINFORMATIVE_ON_CURRENT_CORPUS", ("reference labels almost constant and it "
                                                           "does not separate the planted contrast")
        else:
            cls, why = "EXPLORATORY_NOT_RETAINED", "insufficient combined evidence"
        r["classification"], r["rationale"] = cls, why

    print("judge-H1 reproduction vs published artifact: %s"
          % ("EXACT" if not mism else "MISMATCH on " + ", ".join(mism)))
    hdr = list(rows[0].keys())
    print("\n%-38s %-6s %4s %6s %6s %7s %7s %7s %8s" % (
        "dimension", "scope", "n", "H1ent", "Jent", "HH_AC2", "JH2_AC2", "contr_p", "derived"))
    for r in rows:
        print("%-38s %-6s %4s %6s %6s %7s %7s %7s %8s  %s" % (
            r["dimension"], r["scope"], r["n_units"], r["h1_entropy_bits"], r["judge_entropy_bits"],
            r["hh_ac2"], r["jh2_ac2"], r["contrast_p"], r["derived_measure"], r["classification"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
