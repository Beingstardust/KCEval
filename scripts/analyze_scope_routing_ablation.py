"""Scope-routing ablation on the preregistered replication, from frozen outputs. NO MODEL CALLS.

    python scripts/analyze_scope_routing_ablation.py

The preregistered pipeline scores every segment unit and every topic unit as the mean of ALL ten
rubric dimensions (asserted below: the stored p_s equals that mean on every unit). The judge
answered every dimension on every unit, so routing can be applied at read time without a new judge
call (src/seg_eval/aggregation/scope_routing.py). This recomputes H7.1 and H7.3 under three bases:

  all10      the preregistered basis; must reproduce data/gold/e7_results.json EXACTLY (asserted)
  routed_v1  each unit scored only on the dimensions its scope owns (the published routing lists)
  routed_v2  routed_v1 with proactive_clarification moved to topic scope

Everything else is held fixed: beta = 0.2 with the stored K_s, the stored trust-cap value per
segment, exchange weighting, the common-support widening with a common denominator, and the
statistics of analyze_e7_holdout.py (its paired(), with one seed-1234 RNG consumed in the published
order: H7.1, then H7.3). For each basis the two quantities are also computed on the unit score
alone (no blend, no cap), the analogue of the SP-1 raw-judge decomposition.

Output: data/processed/publication_final/scope_routing_ablation.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e7_holdout as e7  # noqa: E402  (functions and constants only; its main() is guarded)
from seg_eval.aggregation.deterministic_aggregation import weighted_mean_applicable  # noqa: E402
from seg_eval.aggregation.scope_routing import routed_unit_score  # noqa: E402

ARMS = {"H1": "data/processed/console_runs/E7S_hv_p", "H2": "data/processed/console_runs/E7S_hv_q"}
BETA = 0.2
BASES = ("all10", "routed_v1", "routed_v2")
OUT = REPO / "data/processed/publication_final/scope_routing_ablation.json"
PUBLISHED = REPO / "data/gold/e7_results.json"
TOL = 1e-9


def bare(i: str) -> str:
    return str(i).split("::")[-1]


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_outputs(run_dir: str) -> dict[tuple[str, str], dict]:
    out = {}
    for fam in ("local", "arc"):
        for r in jl(REPO / run_dir / "results" / f"validation_{fam}" / "valid_judge_responses.jsonl"):
            jo = r.get("judge_output") or r
            out[(fam, bare(r.get("segment_id") or jo.get("segment_id")))] = jo
    return out


def all_ten(jo: dict) -> float | None:
    return weighted_mean_applicable(dict(jo.get("dimension_scores") or {}),
                                    jo.get("dimension_applicability") or {})


def post(seg: dict, p: float | None) -> float | None:
    """Blend with the stored K_s at BETA, then apply the stored cap value."""
    if p is None:
        return None
    k = seg.get("k_s")
    b = p if k is None else (1.0 - BETA) * p + BETA * k
    cap = seg.get("trust_cap_applied")
    return min(b, cap) if cap is not None else b


def rescored(prov: dict, judge: dict, basis: str, processed: bool) -> tuple[list[dict], int]:
    segs, dropped = [], 0
    for s in prov["segments"]:
        fam = s.get("family")
        if fam not in ("local", "arc"):
            continue
        jo = judge.get((fam, bare(s["segment_id"])))
        if jo is None:
            raise SystemExit(f"no contract-valid judge output for {s['segment_id']}")
        if basis == "all10":
            p = s.get("p_s")
            if p is not None:
                assert abs(all_ten(jo) - p) < TOL, ("stored p_s is not the all-ten mean", s["segment_id"])
        else:
            p = routed_unit_score(jo, fam, basis.split("_")[1])
        dropped += p is None
        v = post(s, p) if processed else p
        if basis == "all10" and processed and s.get("b_s") is not None:
            assert abs(v - s["b_s"]) < TOL, ("blend/cap does not reproduce stored b_s", s["segment_id"])
        segs.append({**s, "b_s": v})
    return segs, dropped


def h71_h73(segs: dict[str, list[dict]]) -> dict:
    rng = random.Random(e7.SEED)          # consumed in the published order: H7.1, then H7.3
    contrib = {arm: e7.family_contributions(ss, "local")[0] for arm, ss in segs.items()}
    shared = sorted(set(contrib["H1"]) & set(contrib["H2"]))
    r1 = e7.paired({e: contrib["H2"][e] - contrib["H1"][e] for e in shared}, rng, "less")

    b_of = {arm: {"local": {}, "arc": {}} for arm in segs}
    for arm, ss in segs.items():
        for seg in ss:
            fam = seg.get("family")
            if fam in b_of[arm] and seg.get("b_s") is not None:
                for i in seg.get("member_exchange_ids") or []:
                    b_of[arm][fam][bare(i)] = float(seg["b_s"])
    support = sorted(set(b_of["H1"]["local"]) & set(b_of["H1"]["arc"])
                     & set(b_of["H2"]["local"]) & set(b_of["H2"]["arc"]))
    n_s = len(support)
    did = {e: ((b_of["H2"]["arc"][e] - b_of["H2"]["local"][e])
               - (b_of["H1"]["arc"][e] - b_of["H1"]["local"][e])) / n_s for e in support}
    r3 = e7.paired(did, rng, "greater")
    r3["common_support_size"] = n_s
    fam = {arm: {f: e7.family_contributions(ss, f)[1] for f in ("local", "arc")}
           for arm, ss in segs.items()}
    return {"H7.1": r1, "H7.3": r3, "family_scores": fam}


def main() -> None:
    prov = {arm: json.loads((REPO / d / "results" / "score_provenance.json")
                            .read_text(encoding="utf-8"))["provenance"] for arm, d in ARMS.items()}
    judge = {arm: judge_outputs(d) for arm, d in ARMS.items()}
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))

    results = {}
    for basis in BASES:
        for processed in (True, False):
            segs, dropped = {}, {}
            for arm in ARMS:
                segs[arm], dropped[arm] = rescored(prov[arm], judge[arm], basis, processed)
            r = h71_h73(segs)
            r["units_without_an_owned_applicable_dimension"] = dropped
            results[f"{basis}|{'post_processed' if processed else 'unit_score_only'}"] = r

    ref = results["all10|post_processed"]
    for h in ("H7.1", "H7.3"):
        for k in ("observed", "p_one_sided"):
            assert abs(ref[h][k] - pub[h][k]) < TOL, (h, k, ref[h][k], pub[h][k])
        assert all(abs(x - y) < TOL for x, y in zip(ref[h]["ci"], pub[h]["ci"])), (h, ref[h]["ci"])
    print("all10 basis reproduces the preregistered H7.1 and H7.3 exactly (asserted)\n")

    hdr = "%-10s %-15s | %8s %8s %9s %20s %8s | %8s %8s %9s %20s %8s"
    print(hdr % ("basis", "scores", "Dseg H1", "Dseg H2", "H7.1", "CI", "p", "Dtop H1", "Dtop H2",
                 "H7.3", "CI", "p"))
    for key, r in results.items():
        basis, kind = key.split("|")
        f, a, c = r["family_scores"], r["H7.1"], r["H7.3"]
        print("%-10s %-15s | %8.4f %8.4f %+9.4f %20s %8.4f | %8.4f %8.4f %+9.4f %20s %8.4f" % (
            basis, kind, f["H1"]["local"], f["H2"]["local"], a["observed"],
            "[%+.3f, %+.3f]" % tuple(a["ci"]), a["p_one_sided"],
            f["H1"]["arc"], f["H2"]["arc"], c["observed"], "[%+.3f, %+.3f]" % tuple(c["ci"]),
            c["p_one_sided"]))
    print()
    for basis in BASES:
        full = results[f"{basis}|post_processed"]["H7.3"]["observed"]
        raw = results[f"{basis}|unit_score_only"]["H7.3"]["observed"]
        print("%-10s widening %+.4f; unit scores alone %+.4f; share from blend and cap %.0f%%; "
              "support %d; units dropped (no owned applicable dimension) %s" % (
                  basis, full, raw, 100 * (1 - raw / full) if full else float("nan"),
                  results[f"{basis}|post_processed"]["H7.3"]["common_support_size"],
                  results[f"{basis}|post_processed"]["units_without_an_owned_applicable_dimension"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"beta": BETA, "seed": e7.SEED, "n_bootstrap": e7.N_BOOT,
                               "n_permutations": e7.N_PERM, "results": results},
                              indent=2), encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
