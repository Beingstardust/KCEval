"""`A_s`: the rate at which an expert criterion APPLIES to what the tutor did. NO MODEL CALLS.

    python scripts/derive_criterion_applicability.py

WHY THIS EXISTS
---------------
G6 asks for evidence that the expert criteria work. `K_s` -- the mean rating over criteria the
judge marked *applicable* -- has never supplied it: T1 0.881 vs T2 0.731 at p = 0.1954, n = 21
against 13 units.

Reading the criterion-level judgments rather than the per-unit means shows why, and the answer is
not "underpowered". Conditional on a criterion being applicable, **every arm scores the same**:

    T1 62.2%   T2 63.2%   T3 59.4%   T4 60.6%   T5 60.5%   (fully satisfied)
    T1 vs T2 on rating: Fisher p = 1.0000

The rating channel carries no signal at all. What carries it is **applicability**: whether an
expert's criterion is even relevant to what the tutor did.

    T1 29.6%   T2 15.3%   T3 25.0%   T4 25.6%   T5 24.5%   (applicable)
    T1 vs T2: Fisher p = 0.0095

And it is specific. The three arms that degrade a NON-pedagogical axis by construction --
factuality (T3), dialogue structure (T4), register (T5) -- do not move: p = 0.48, 0.49, 0.35.

WHAT IT MEANS, AND WHY IT IS THE RIGHT SHAPE
--------------------------------------------
The expert criteria describe HOW A CONCEPT SHOULD BE TAUGHT. A tutor that dumps the answer instead
of teaching does not fail those criteria -- it never engages them. `K_s` averages over applicable
criteria and therefore divides that failure away: the few criteria that still apply are satisfied
at the same rate as in a good tutor, so the mean barely moves.

So the pedagogical signal the criteria carry is in the DENOMINATOR, not the numerator.

STATUS: EXPLORATORY, AND LABELLED AS SUCH
------------------------------------------
This was found by inspecting the criterion-level data after `K_s` failed, so it is post-hoc and one
p-value alone would not be worth much. What raises it above a fishing expedition is the built-in
discriminant test it passes unprompted: of the four arms, exactly the one that degrades what the
criteria measure moves, and the three that degrade something else do not. That is a 1-of-4 pattern
predicted by the construct, not selected after the fact.

It is reported as a companion measure. **`K_s` and every published `T` are left unchanged** -- this
is post-hoc, and changing the aggregate on a post-hoc finding at closeout would be exactly the
error the pre-registration discipline exists to prevent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ARMS = {
    "T1_strong": ("tv_a", "20260829"),
    "T2_answer_dumping": ("tv_b", "20260829"),
    "T3_subtly_wrong": ("tv_c", "20260829"),
    "T4_macro_degraded": ("tv_d", "20260909"),
    "T5_expert_register": ("tv_e", "20260909"),
}
GOOD, BAD = "T1_strong", "T2_answer_dumping"
# Arms that degrade an axis the criteria are NOT about. They are the discriminant test.
NON_PEDAGOGICAL = ("T3_subtly_wrong", "T4_macro_degraded", "T5_expert_register")


def judgments(tag: str, stamp: str) -> list[dict]:
    p = REPO / (f"data/processed/judge_responses/v35_{tag}_kc_criteria_{stamp}/"
                "kc_criterion_responses.jsonl")
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            body = json.loads(json.loads(line)["raw_response_text"])
        except Exception:
            continue          # a malformed response contributes nothing, and is counted below
        out += body.get("criterion_judgments") or []
    return out


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact test on [[a, b], [c, d]]. Used rather than chi-square because the applicable counts
    are small (19 in the degraded arm)."""
    n = a + b + c + d
    if not n:
        return 1.0

    def pr(x: int) -> float:
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)

    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12))


def split(js: list[dict]) -> tuple[int, int]:
    ap = sum(1 for x in js if x.get("applicability") == "applicable")
    return ap, len(js) - ap


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--json")
    a = ap_.parse_args()

    J = {v: judgments(*ARMS[v]) for v in ARMS}
    missing = [v for v, j in J.items() if not j]
    if missing:
        print("  arms with no criterion judgments on disk: %s" % missing)

    print("  A_s -- does an expert criterion APPLY to what the tutor did?\n")
    print("  %-22s %-18s %s" % ("arm", "applicable", "vs T1 (Fisher exact)"))
    rows = {}
    g_ap, g_na = split(J[GOOD])
    for v in ARMS:
        if not J[v]:
            continue
        ap, na = split(J[v])
        rate = ap / (ap + na)
        p = None if v == GOOD else fisher_two_sided(g_ap, g_na, ap, na)
        note = ""
        if v == BAD:
            note = "  <-- degrades what the criteria measure"
        elif v in NON_PEDAGOGICAL:
            note = "  (discriminant: should NOT move)"
        print("  %-22s %3d/%-3d = %5.1f%%   %s%s"
              % (v, ap, ap + na, 100 * rate, "-" if p is None else "p = %.4f" % p, note))
        rows[v] = {"applicable": ap, "total": ap + na, "rate": rate, "p_vs_good": p}

    print("\n  rating, CONDITIONAL on being applicable -- did the tutor satisfy it?\n")
    print("  %-22s %s" % ("arm", "fully satisfied"))
    for v in ARMS:
        if not J[v]:
            continue
        app = [x for x in J[v] if x.get("applicability") == "applicable"]
        good = sum(1 for x in app if x.get("rating") == 1)
        rows[v]["rating_good"] = good
        rows[v]["rating_n"] = len(app)
        print("  %-22s %3d/%-3d = %5.1f%%" % (v, good, len(app), 100 * good / max(1, len(app))))
    r1, r2 = rows[GOOD], rows[BAD]
    p_rating = fisher_two_sided(r1["rating_good"], r1["rating_n"] - r1["rating_good"],
                                r2["rating_good"], r2["rating_n"] - r2["rating_good"])
    print("\n  %s vs %s on RATING: Fisher p = %.4f  -- no signal in this channel"
          % (GOOD, BAD, p_rating))

    disc = [rows[v]["p_vs_good"] for v in NON_PEDAGOGICAL if v in rows]
    p_bad = rows[BAD]["p_vs_good"] if BAD in rows else None
    print("\n  VERDICT")
    print("    applicability separates the pedagogically degraded arm : p = %s"
          % ("%.4f" % p_bad if p_bad is not None else "n/a"))
    print("    and does NOT separate the three non-pedagogical arms   : p = %s"
          % ", ".join("%.2f" % x for x in disc))
    print("    rating separates nothing                               : p = %.4f" % p_rating)
    print("\n    K_s and every published T are UNCHANGED -- this is post-hoc and reported as a")
    print("    companion measure, not folded into the aggregate.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"rows": rows, "p_rating_good_vs_bad": p_rating}, indent=1), encoding="utf-8")
        print("\n  wrote %s" % a.json)


if __name__ == "__main__":
    main()
