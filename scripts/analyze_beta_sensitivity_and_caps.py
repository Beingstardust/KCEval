"""Beta sensitivity and trust-cap reconciliation, from frozen artifacts. NO MODEL CALLS.

    python scripts/analyze_beta_sensitivity_and_caps.py

Answers two questions the manuscript got wrong or asserted without checking.

1. CAP RECONCILIATION. The manuscript said the trust cap "lowers 39 of 42 segment units". That
   number came from the SP-1 premise audit field `local_units_where_b_below_p`, which compares the
   stored b_s against p_s. b_s is post-blend AND post-cap, so that count mixes two different
   mechanisms: the KC-criterion blend and the factuality cap. The console provenance stores
   `trust_cap_applied` per segment, so the two can be separated exactly.

2. BETA SENSITIVITY. The manuscript said no reported result depends on a weight. alpha and rho are
   genuinely absent from family-scope results, but beta enters D_segment through
   b_blend = (1-beta) p_s + beta k_s wherever a segment carries KC criteria. This recomputes the
   headline replication quantities over a declared beta grid. The grid is fixed in advance here and
   beta is NOT tuned: the purpose is sensitivity, not selection.

Topic units carry no k_s and are never capped in these runs (asserted below), so D_topic is
beta-invariant and the entire beta dependence of the cross-scope discrepancy runs through D_segment.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ARMS = {
    "clean": "data/processed/console_runs/E7S_hv_p",
    "corrupted": "data/processed/console_runs/E7S_hv_q",
}
BETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]   # 1.0 is a diagnostic endpoint, not an operating point
PUBLISHED_BETA = 0.2
OUT_CSV = REPO / "data/processed/publication_final/beta_sensitivity.csv"
SEED = 1234
N_PERM = 100000


def segments(run_dir: str) -> list[dict]:
    p = json.loads((REPO / run_dir / "results" / "score_provenance.json").read_text(encoding="utf-8"))
    return p["provenance"]["segments"]


def n_of(s: dict) -> int:
    return s.get("exchange_count") or len(s.get("member_exchange_ids") or []) or 1


def blend(s: dict, beta: float):
    p, k = s.get("p_s"), s.get("k_s")
    if p is None:
        return None
    if k is None:
        return p
    return (1.0 - beta) * p + beta * k


def value(s: dict, beta: float):
    """Post-blend, post-cap unit value at a given beta, using the stored cap value."""
    b = blend(s, beta)
    if b is None:
        return None
    cap = s.get("trust_cap_applied")
    return min(b, cap) if cap is not None else b


def ew(pairs) -> float | None:
    num = den = 0.0
    for v, n in pairs:
        if v is None:
            continue
        num += v * n
        den += n
    return (num / den) if den else None


def per_exchange(segs: list[dict], beta: float, fam: str) -> dict[str, float]:
    """exchange short id -> unit value, for one family."""
    out = {}
    for s in segs:
        if s.get("family") != fam:
            continue
        v = value(s, beta)
        if v is None:
            continue
        for eid in (s.get("member_exchange_ids") or []):
            out[eid.split("::")[-1]] = v
    return out


def sign_flip_p(diffs: list[float], seed: int = SEED, n: int = N_PERM) -> float:
    """One-sided paired sign-flip permutation test, direction 'greater'."""
    obs = sum(diffs) / len(diffs)
    rnd = random.Random(seed)
    hits = 0
    for _ in range(n):
        m = sum(d if rnd.random() < 0.5 else -d for d in diffs) / len(diffs)
        if m >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def main() -> None:
    segs = {arm: segments(d) for arm, d in ARMS.items()}

    # ---------------------------------------------------------------- 1. cap reconciliation
    print("=" * 96)
    print("CAP RECONCILIATION  (beta = %.1f, the value used in the replication)" % PUBLISHED_BETA)
    print("=" * 96)
    print("%-11s %6s %9s %11s %11s %12s %12s" % (
        "condition", "local", "cap set", "cap BINDING", "criteria", "any change", "unchanged"))
    recon = {}
    for arm, ss in segs.items():
        loc = [s for s in ss if s.get("family") == "local"]
        cap_set = [s for s in loc if s.get("trust_cap_applied") is not None]
        cap_binding = [s for s in loc
                       if s.get("trust_cap_applied") is not None
                       and blend(s, PUBLISHED_BETA) is not None
                       and s["trust_cap_applied"] < blend(s, PUBLISHED_BETA) - 1e-12]
        crit = [s for s in loc
                if s.get("k_s") is not None and s.get("p_s") is not None
                and abs(s["k_s"] - s["p_s"]) > 1e-12]
        changed = [s for s in loc
                   if s.get("p_s") is not None and value(s, PUBLISHED_BETA) is not None
                   and abs(value(s, PUBLISHED_BETA) - s["p_s"]) > 1e-12]
        recon[arm] = {"local": len(loc), "cap_set": len(cap_set), "cap_binding": len(cap_binding),
                      "criteria_differ": len(crit), "any_change": len(changed)}
        print("%-11s %6d %9d %11d %11d %12d %12d" % (
            arm, len(loc), len(cap_set), len(cap_binding), len(crit), len(changed),
            len(loc) - len(changed)))

    # topic-side premise, asserted rather than assumed
    for arm, ss in segs.items():
        arc = [s for s in ss if s.get("family") == "arc"]
        with_k = [s for s in arc if s.get("k_s") is not None]
        capped = [s for s in arc if s.get("trust_cap_applied") is not None]
        print("  %-9s topic units %d; with k_s %d; with a cap value %d"
              % (arm, len(arc), len(with_k), len(capped)))

    # ---------------------------------------------------------------- 2. beta sensitivity
    print()
    print("=" * 96)
    print("BETA SENSITIVITY  (topic side is beta-invariant; all dependence is through D_segment)")
    print("=" * 96)
    print("%6s %11s %11s %11s | %10s %10s %11s %9s" % (
        "beta", "D_seg clean", "D_seg corr", "corr-clean", "R clean", "R corr", "widening", "p(1-sided)"))
    rows = []
    for beta in BETA_GRID:
        d_seg = {}
        d_top = {}
        for arm, ss in segs.items():
            d_seg[arm] = ew([(value(s, beta), n_of(s)) for s in ss if s.get("family") == "local"])
            d_top[arm] = ew([(value(s, beta), n_of(s)) for s in ss if s.get("family") == "arc"])
        # paired exchange-level widening on the common support of both families and both arms
        loc = {arm: per_exchange(ss, beta, "local") for arm, ss in segs.items()}
        top = {arm: per_exchange(ss, beta, "arc") for arm, ss in segs.items()}
        common = (set(loc["clean"]) & set(loc["corrupted"])
                  & set(top["clean"]) & set(top["corrupted"]))
        diffs = [(top["corrupted"][e] - loc["corrupted"][e]) - (top["clean"][e] - loc["clean"][e])
                 for e in sorted(common)]
        widening = sum(diffs) / len(diffs)
        p = sign_flip_p(diffs)
        r_clean = (d_top["clean"] - d_seg["clean"])
        r_corr = (d_top["corrupted"] - d_seg["corrupted"])
        print("%6.1f %11.4f %11.4f %11.4f | %10.4f %10.4f %11.4f %9.4f" % (
            beta, d_seg["clean"], d_seg["corrupted"], d_seg["corrupted"] - d_seg["clean"],
            r_clean, r_corr, widening, p))
        rows.append({"beta": beta,
                     "d_segment_clean": round(d_seg["clean"], 6),
                     "d_segment_corrupted": round(d_seg["corrupted"], 6),
                     "segment_corrupted_minus_clean": round(d_seg["corrupted"] - d_seg["clean"], 6),
                     "d_topic_clean": round(d_top["clean"], 6),
                     "d_topic_corrupted": round(d_top["corrupted"], 6),
                     "R_clean": round(r_clean, 6), "R_corrupted": round(r_corr, 6),
                     "widening_paired": round(widening, 6),
                     "p_one_sided": round(p, 6), "n_pairs": len(diffs)})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % OUT_CSV)

    base = next(r for r in rows if abs(r["beta"] - PUBLISHED_BETA) < 1e-9)
    print("\nAt the published beta=%.1f: segment drop %.4f, widening %.4f (published: -0.0955, +0.0853)"
          % (PUBLISHED_BETA, base["segment_corrupted_minus_clean"], base["widening_paired"]))
    lo = min(r["segment_corrupted_minus_clean"] for r in rows if r["beta"] <= 0.4)
    hi = max(r["segment_corrupted_minus_clean"] for r in rows if r["beta"] <= 0.4)
    wlo = min(r["widening_paired"] for r in rows if r["beta"] <= 0.4)
    whi = max(r["widening_paired"] for r in rows if r["beta"] <= 0.4)
    print("Across beta in [0, 0.4]: segment drop ranges %.4f to %.4f; widening %.4f to %.4f"
          % (lo, hi, wlo, whi))
    print("Direction preserved on both quantities: %s"
          % ("YES" if hi < 0 and wlo > 0 else "NO"))


if __name__ == "__main__":
    main()
