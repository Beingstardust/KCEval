"""Second-annotator (E5) agreement: human vs human, and each human vs the Selene judge.

    python scripts/analyze_second_annotator_agreement.py
    python scripts/analyze_second_annotator_agreement.py --json data/gold/e5_second_annotator/agreement.json

NO MODEL CALLS. Reads three sources for the SAME 115 units:

    A1      data/gold/human_validation_extracted/judge_validation_extracted.jsonl  (initials SM)
    A2      data/gold/e5_second_annotator/a2_export_20260910.jsonl                 (the E5 site)
    judge   data/processed/judge_responses/selene_fixed_dm{1,2}_{local,arc}_20260830

WHY IT IS NEEDED
----------------
Every human-validation number in this project rests on ONE annotator. A single rater cannot show
whether the rubric is reproducible or whether that rater simply applied a private reading of it.
The judge-human agreement already reported is only interpretable against a human-human ceiling:
a judge that matches A1 as well as A2 does is at the limit of what the rubric supports.

TWO CHANNELS, REPORTED SEPARATELY
---------------------------------
applicability   does the dimension apply to this unit at all (scored vs n/a). For the sparse
                dimensions this is where nearly all the information is.
rating          the 0 / 0.5 / 1 score, on units BOTH raters scored. Restricting to that subset is
                what keeps a rating disagreement from being an applicability disagreement wearing
                a different hat.

Chance-corrected twice, on purpose. Cohen kappa collapses toward 0 when one category dominates
(the prevalence paradox) and most dimensions here are ~90% "1". Gwet AC1/AC2 is reported beside it
because it does not degrade that way. Raw agreement is shown so both corrections stay auditable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seg_eval.aggregation import scope_routing as SR  # noqa: E402
from seg_eval.evaluation_judge.focus_target_validation_analysis import cohens_kappa  # noqa: E402
import analyze_dimension_informativeness as di  # noqa: E402

A1_PATH = REPO / "data/gold/human_validation_extracted/judge_validation_extracted.jsonl"
A2_PATH = REPO / "data/gold/e5_second_annotator/a2_export_20260910.jsonl"
JUDGE_DIRS = [
    "data/processed/judge_responses/selene_fixed_dm1_kc_segment_local_20260830",
    "data/processed/judge_responses/selene_fixed_dm1_topic_rollup_arc_20260830",
    "data/processed/judge_responses/selene_fixed_dm2_kc_segment_local_20260830",
    "data/processed/judge_responses/selene_fixed_dm2_topic_rollup_arc_20260830",
]
NA = "NA"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def norm(v) -> str | None:
    """Every rater onto one category set: 1 / 0.5 / 0 / NA. None means the rater left it blank."""
    if v is None:
        return NA
    if isinstance(v, (int, float)):
        f = float(v)
        return "1" if f == 1 else ("0" if f == 0 else ("0.5" if f == 0.5 else str(f)))
    s = str(v).strip()
    if s == "":
        return None
    if s.upper() in ("NA", "N/A", "NOT_APPLICABLE"):
        return NA
    try:
        return norm(float(s))
    except ValueError:
        return None


def load_a1() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    out, fam = {}, {}
    for r in load_jsonl(A1_PATH):
        vals = {}
        for src in ("human_dimension_scores", "human_trust_flags"):
            for k, v in (r.get(src) or {}).items():
                n = norm(v)
                if n is not None:
                    vals[k] = n
        out[r["review_item_id"]] = vals
        fam[r["review_item_id"]] = r.get("family") or "local"
    return out, fam


def load_a2() -> tuple[dict[str, dict[str, str]], dict[str, float]]:
    out, secs = {}, {}
    for r in load_jsonl(A2_PATH):
        vals = {}
        for k, v in (r.get("scores") or {}).items():
            n = norm(v)
            if n is not None:
                vals[k] = n
        out[r["item_id"]] = vals
        secs[r["item_id"]] = float(r.get("seconds") or 0)
    return out, secs


def load_judge() -> dict[str, dict[str, str]]:
    out = {}
    for d in JUDGE_DIRS:
        f = REPO / d / "judge_responses.jsonl"
        if not f.exists():
            continue
        for r in load_jsonl(f):
            try:
                jo = json.loads(r["raw_response_text"])
            except Exception:
                continue                       # contract-invalid response: no scores to compare
            vals = {}
            for src in ("dimension_scores", "trust_flags"):
                for k, v in (jo.get(src) or {}).items():
                    n = norm(v)
                    if n is not None:
                        vals[k] = n
            app = jo.get("dimension_applicability") or {}
            for k, v in app.items():
                if v == "not_applicable":
                    vals[k] = NA
            out[r["segment_id"]] = vals
    return out


def pair_stats(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0}
    raw = sum(1 for a, b in pairs if a == b) / len(pairs)
    return {"n": len(pairs), "raw": raw,
            "kappa": cohens_kappa(pairs), "wkappa": di.weighted_kappa(pairs),
            "ac": di.gwet_ac2(pairs)}


def compare(x: dict[str, dict[str, str]], y: dict[str, dict[str, str]], items: list[str],
            scope: dict[str, set[str]]) -> dict:
    """Applicability and rating agreement, per dimension and pooled.

    Restricted to the dimensions IN SCOPE for each unit under routing v1. Without that restriction
    the comparison is not the same size for every pair: the judge is asked all ten dimensions on
    every unit, while each human was asked only the ones that unit's scope owns, so an unrestricted
    judge-human comparison adds ~600 out-of-scope cells the humans were never shown.
    """
    dims = sorted({d for i in items
                   for d in set(x.get(i, {})) & set(y.get(i, {})) & scope.get(i, set())})
    per, app_all, rate_all = {}, [], []
    for d in dims:
        app, rate = [], []
        for i in items:
            if d not in scope.get(i, set()):
                continue
            a, b = x.get(i, {}).get(d), y.get(i, {}).get(d)
            if a is None or b is None:
                continue
            app.append(("NA" if a == NA else "scored", "NA" if b == NA else "scored"))
            if a != NA and b != NA:
                rate.append((a, b))
        per[d] = {"applicability": pair_stats(app), "rating": pair_stats(rate),
                  "n_scored_x": sum(1 for p in app if p[0] == "scored"),
                  "n_scored_y": sum(1 for p in app if p[1] == "scored")}
        app_all += app
        rate_all += rate
    return {"per_dimension": per, "pooled_applicability": pair_stats(app_all),
            "pooled_rating": pair_stats(rate_all)}


def fmt(v, nd=3):
    return "-" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def show(title: str, res: dict) -> None:
    print("\n%s" % title)
    print("  %-34s%18s%30s" % ("", "applicability", "rating (both scored)"))
    print("  %-34s%6s%6s%6s   %6s%7s%7s%7s" % ("dimension", "n", "raw", "AC1", "n", "raw", "kappa", "AC2"))
    for d, r in res["per_dimension"].items():
        a, t = r["applicability"], r["rating"]
        print("  %-34s%6s%6s%6s   %6s%7s%7s%7s" % (
            d, a.get("n", 0), fmt(a.get("raw"), 2), fmt(a.get("ac"), 2),
            t.get("n", 0), fmt(t.get("raw"), 2), fmt(t.get("kappa"), 2), fmt(t.get("ac"), 2)))
    pa, pr = res["pooled_applicability"], res["pooled_rating"]
    print("  %-34s%6s%6s%6s   %6s%7s%7s%7s" % (
        "POOLED", pa.get("n", 0), fmt(pa.get("raw"), 2), fmt(pa.get("ac"), 2),
        pr.get("n", 0), fmt(pr.get("raw"), 2), fmt(pr.get("kappa"), 2), fmt(pr.get("ac"), 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "data/gold/e5_second_annotator/agreement.json"))
    args = ap.parse_args()

    (a1, fam), (a2, secs) = load_a1(), load_a2()
    judge = load_judge()
    items = sorted(set(a1) & set(a2))
    local_scope = set(SR.local_dimensions("v1")) | set(SR.local_trust_flags("v1"))
    arc_scope = set(SR.arc_dimensions("v1")) | set(SR.arc_trust_flags("v1"))
    scope = {i: (arc_scope if fam.get(i) == "arc" else local_scope) for i in items}
    print("units: %d segment-scope, %d arc-scope (scope decides which dimensions are compared)"
          % (sum(1 for i in items if fam.get(i) != "arc"), sum(1 for i in items if fam.get(i) == "arc")))
    print("items: A1 %d, A2 %d, shared %d; judge responses on %d of the shared items"
          % (len(a1), len(a2), len(items), sum(1 for i in items if i in judge)))
    total_min = sum(secs.get(i, 0) for i in items) / 60.0
    print("A2 effort: %.0f min total, median %.0f s per unit" % (
        total_min, sorted(secs.values())[len(secs) // 2] if secs else 0))

    res = {"n_items": len(items),
           "A1_vs_A2": compare(a1, a2, items, scope),
           "A1_vs_judge": compare(a1, judge, [i for i in items if i in judge], scope),
           "A2_vs_judge": compare(a2, judge, [i for i in items if i in judge], scope)}

    show("A1 vs A2  -- the human-human ceiling", res["A1_vs_A2"])
    show("A1 vs Selene judge", res["A1_vs_judge"])
    show("A2 vs Selene judge", res["A2_vs_judge"])

    print("\nthe question this study exists to answer:")
    for key in ("A1_vs_A2", "A1_vs_judge", "A2_vs_judge"):
        pr = res[key]["pooled_rating"]
        pa = res[key]["pooled_applicability"]
        print("  %-14s rating raw %s (n=%s)   applicability raw %s (n=%s)"
              % (key, fmt(pr.get("raw"), 3), pr.get("n"), fmt(pa.get("raw"), 3), pa.get("n")))

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
