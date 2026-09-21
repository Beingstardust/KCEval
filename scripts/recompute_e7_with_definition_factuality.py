"""Recompute the E7/dm2 score-level results under the definition-grounded factuality checker.

RQ3's score-level predictions run through the deterministic trust cap, and the cap fires on the
factuality verdict. Swapping the checker therefore changes them, so they cannot be restated from the
published run.

This does NOT re-run the aggregation. The published `score_provenance.json` already records, per
segment, the pre-cap blend inputs (`p_s`, `k_s`), the cap that was applied, the resulting `b_s`, and
the exchange count. Since

    b_s = min(alpha_blend(p_s, k_s, beta), cap)

and the family score is the exchange-weighted mean of `b_s`, the only thing that has to change is
the cap, which is a pure function of the verdict.

The reconstruction is validated first: recomputing `b_s` from `p_s`, `k_s` and the OLD verdicts must
reproduce every recorded `b_s` and the recorded `d_segment` / `d_arc` exactly. If it does not, the
script refuses rather than report a number built on a broken reconstruction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from seg_eval.aggregation.deterministic_aggregation import apply_deterministic_trust_cap  # noqa: E402

TOL = 1e-9
ARMS = {"clean": "E7S_hv_p", "corrupt": "E7S_hv_q"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def blend(p_s: float, k_s: float | None, beta: float) -> float:
    return p_s if k_s is None else (1.0 - beta) * p_s + beta * k_s


def family_score(segs: list[dict], fam: str, key: str = "b_s") -> float | None:
    rows = [(s[key], s.get("exchange_count") or 0) for s in segs
            if s.get("family") == fam and s.get(key) is not None]
    tot = sum(n for _, n in rows)
    return sum(b * n for b, n in rows) / tot if tot else None


def old_verdicts(run: str) -> dict[str, tuple[str | None, int]]:
    out = {}
    for r in jl(REPO / f"data/processed/console_runs/{run}/judge_responses/factuality"
                / "grounded_factuality_responses.jsonl"):
        out[r["segment_id"]] = (r.get("derived_verdict"), int(r.get("material_contradictions") or 0))
    return out


def new_verdicts(arm: str) -> dict[str, tuple[str, int]]:
    """Definition-grounded verdicts, keyed by the console's segment id."""
    out = {}
    for r in jl(REPO / f"data/processed/definition_factuality_dm2_20260920/responses/{arm}"
                / "definition_factuality_responses.jsonl"):
        out[r["segment_id"]] = (r.get("derived_verdict"),
                                int(r.get("material_contradictions") or 0))
    return out


def recompute(segs: list[dict], verdicts: dict, beta: float) -> list[dict]:
    out = []
    for s in segs:
        pre = blend(s["p_s"], s.get("k_s"), beta) if s.get("p_s") is not None else None
        v, m = verdicts.get(s["segment_id"], (None, 0))
        if pre is None:
            out.append({**s, "b_s": None}); continue
        res = apply_deterministic_trust_cap(pre, v, m)
        out.append({**s, "b_s": res.capped_score, "_pre_cap": pre,
                    "_cap": res.cap_value, "_cap_bound": res.cap_applied, "_verdict": v})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/gold/e7_definition_factuality_rescore.json"))
    args = ap.parse_args()

    report: dict = {"note": "RQ3 score-level results under the definition-grounded checker",
                    "preregistration": "docs/evaluation_completeness_v2/67_DM4_REPAIRED_MECHANISM_RESULTS.md"}

    for arm, run in ARMS.items():
        prov = json.loads((REPO / f"data/processed/console_runs/{run}/results/score_provenance.json")
                          .read_text(encoding="utf-8"))["provenance"]
        beta = prov["beta"]
        segs = prov["segments"]

        # ---- 1. validate the reconstruction against the published run ----
        rebuilt = recompute(segs, old_verdicts(run), beta)
        bad = [s["segment_id"] for s, r in zip(segs, rebuilt)
               if s.get("b_s") is not None and abs(s["b_s"] - r["b_s"]) > TOL]
        if bad:
            raise SystemExit(f"{arm}: reconstruction failed on {len(bad)} segments, e.g. {bad[:5]}. "
                             "Refusing to report a rescore built on a broken reconstruction.")
        for fam, recorded in (("local", prov["d_segment"]), ("arc", prov["d_arc"])):
            got = family_score(rebuilt, fam)
            if got is None or abs(got - recorded) > TOL:
                raise SystemExit(f"{arm}/{fam}: rebuilt {got} != recorded {recorded}")

        # ---- 2. rescore under the definition-grounded verdicts ----
        nv = new_verdicts(arm)
        rescored = recompute(segs, nv, beta)
        old_caps = sum(1 for s in rebuilt if s.get("_cap_bound"))
        new_caps = sum(1 for s in rescored if s.get("_cap_bound"))

        report[arm] = {
            "run": run, "beta": beta,
            "segments_scored": sum(1 for s in segs if s.get("p_s") is not None),
            "old": {"d_segment": prov["d_segment"], "d_arc": prov["d_arc"],
                    "R": prov["d_arc"] - prov["d_segment"], "segments_capped": old_caps},
            "new": {"d_segment": family_score(rescored, "local"),
                    "d_arc": family_score(rescored, "arc"),
                    "segments_capped": new_caps},
            "new_verdict_counts": {v: sum(1 for x in nv.values() if x[0] == v)
                                   for v in sorted({x[0] for x in nv.values()})},
            "reconstruction_exact": True,
        }
        report[arm]["new"]["R"] = report[arm]["new"]["d_arc"] - report[arm]["new"]["d_segment"]

        # ---- 3. write a rescored run directory ----
        # `analyze_e7_holdout.py` reads a console run, not this summary, so the rescored
        # provenance has to exist on disk as a run of its own for the preregistered analysis to
        # be re-runnable against it. Everything except results/ is copied unchanged, since only
        # the factuality verdict and what the cap does with it differ.
        src = REPO / f"data/processed/console_runs/{run}"
        dst = REPO / f"data/processed/console_runs/{run}_defgrounded"
        (dst / "results").mkdir(parents=True, exist_ok=True)
        for sub in ("evaluation_packets", "judge_responses"):
            if (src / sub).is_dir():
                if (dst / sub).exists():
                    shutil.rmtree(dst / sub)
                shutil.copytree(src / sub, dst / sub)
        # the definition-grounded verdicts replace the old factuality responses, so a reader of
        # the rescored run sees the verdicts its scores were actually built from
        fact_dir = dst / "judge_responses" / "factuality"
        if fact_dir.is_dir():
            rows = jl(REPO / f"data/processed/definition_factuality_dm2_20260920/responses/{arm}"
                      / "definition_factuality_responses.jsonl")
            with (fact_dir / "grounded_factuality_responses.jsonl").open(
                    "w", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        full = json.loads((src / "results/score_provenance.json").read_text(encoding="utf-8"))
        by_id = {s["segment_id"]: s for s in rescored}
        for s in full["provenance"]["segments"]:
            r = by_id.get(s["segment_id"])
            if r is None or r.get("b_s") is None:
                continue
            s["b_s"] = r["b_s"]
            s["trust_cap_applied"] = r.get("_cap")
        full["provenance"]["d_segment"] = family_score(rescored, "local")
        full["provenance"]["d_arc"] = family_score(rescored, "arc")
        (dst / "results/score_provenance.json").write_text(
            json.dumps(full, indent=2), encoding="utf-8")
        print(f"{arm}: wrote {dst.relative_to(REPO)}")

    for k in ("old", "new"):
        c, d = report["clean"][k], report["corrupt"][k]
        report.setdefault("predictions", {})[k] = {
            "P1_segment_score_drop": d["d_segment"] - c["d_segment"],
            "P3_cross_scope_widening": d["R"] - c["R"],
            "segments_capped_corrupt_vs_clean": f"{d['segments_capped']} vs {c['segments_capped']}",
        }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["predictions"], indent=2))
    for arm in ARMS:
        r = report[arm]
        print(f"\n{arm}: verdicts {r['new_verdict_counts']}")
        print(f"  d_segment {r['old']['d_segment']:.4f} -> {r['new']['d_segment']:.4f}")
        print(f"  d_arc     {r['old']['d_arc']:.4f} -> {r['new']['d_arc']:.4f}")
        print(f"  capped    {r['old']['segments_capped']} -> {r['new']['segments_capped']}")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
