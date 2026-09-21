"""Doc 66 preflight: does the repaired Stage 2 still protect correct material?

Two things are checked, and only the first one gates.

GATE (safety). On dm1's clean arm, Stage 1 raised exactly one conflict, against a tutor statement
that is TRUE ("a perfectly mixed binary node ... has entropy 1"). Stage 2 v1 dropped it, and that
drop is the entire reason dm1's clean-arm flag rate is 0/37 rather than 1/37. Stage 2 v2 loosens the
"adds detail" carve-out, so it could plausibly stop protecting that statement. If v2 calls it
inconsistent, the repair has traded away the property that makes this mechanism worth having, and
the run must not proceed.

DIAGNOSTIC (does not gate). The three correct catches Stage 2 v1 dropped on dm3 (doc 65 section
2.2). This replays them under v2 only to learn whether the repair changes anything at all, so that a
fresh dialogue is not spent validating a fix with no effect. dm3 is a spent corpus and this is
declared development use of it: v2 is thereby exposed to dm3, and dm4 is the only clean estimate of
what v2 achieves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_definition_factuality import adjudicate  # noqa: E402

DM1_CLEAN = REPO / "data/processed/definition_factuality_cal_20260920/responses/tv_a/definition_factuality_responses.jsonl"
DM3_CORRUPT = REPO / "data/processed/definition_factuality_dm3_20260920/responses/corrupt/definition_factuality_responses.jsonl"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def collect(path: Path) -> list[tuple[str, dict]]:
    out = []
    for r in jl(path):
        for c in r.get("rejected_conflicts") or []:
            if c.get("stage") == "stage2_dropped":
                out.append((r["segment_id"].rsplit("::", 1)[-1], c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--out", default=str(REPO / "data/processed/definition_factuality_dm4_20260920/preflight_stage2_v2.json"))
    args = ap.parse_args()

    gate = collect(DM1_CLEAN)
    if len(gate) != 1:
        raise SystemExit(f"expected exactly 1 dm1 clean-arm Stage 2 drop, found {len(gate)}")

    report: dict = {"gate": [], "diagnostic": [], "preregistration":
                    "docs/evaluation_completeness_v2/66_REPAIRED_MECHANISM_PREREGISTRATION.md"}

    print("=== GATE: dm1 clean-arm protection ===", flush=True)
    passed = True
    for seg, c in gate:
        res = adjudicate(c["tutor_span"], c["definition_span"], args.base_url, variant="v2")
        inc = bool((res.payload or {}).get("inconsistent"))
        print(f"  {seg} inconsistent={inc} (must be False)")
        print(f"    tutor: {c['tutor_span'][:110]}")
        print(f"    why  : {(res.payload or {}).get('why')}")
        report["gate"].append({"segment_id": seg, "tutor_span": c["tutor_span"],
                               "inconsistent": inc, "why": (res.payload or {}).get("why")})
        if inc:
            passed = False

    print("\n=== DIAGNOSTIC: dm3 Stage 2 drops under v2 (does not gate) ===", flush=True)
    flipped = 0
    for seg, c in collect(DM3_CORRUPT):
        res = adjudicate(c["tutor_span"], c["definition_span"], args.base_url, variant="v2")
        inc = bool((res.payload or {}).get("inconsistent"))
        flipped += inc
        print(f"  {seg} v1=dropped -> v2 inconsistent={inc}")
        print(f"    tutor: {c['tutor_span'][:110]}")
        report["diagnostic"].append({"segment_id": seg, "tutor_span": c["tutor_span"],
                                     "now_inconsistent": inc,
                                     "why": (res.payload or {}).get("why")})

    report["gate_passed"] = passed
    report["diagnostic_flipped"] = f"{flipped}/{len(report['diagnostic'])}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGATE_PASSED={passed}  DIAGNOSTIC_FLIPPED={report['diagnostic_flipped']}")
    print(f"WROTE {out}")
    if not passed:
        raise SystemExit("GATE FAILED: Stage 2 v2 no longer protects correct material. Do not run dm4.")


if __name__ == "__main__":
    main()
