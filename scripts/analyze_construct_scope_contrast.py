"""Construct-level scope test on frozen development outputs. NO MODEL CALLS.

    python scripts/analyze_construct_scope_contrast.py

The judge prompt asks every rubric dimension on every unit, at segment scope and at topic scope.
So for each construct the planted pedagogical contrast (REF against PED-DEG: the reference tutor
against the rewrite with diagnosis, scaffolding and hints removed) can be measured at BOTH scopes
from the same frozen responses. If routing is right, a construct should separate the contrast at
the scope that owns it at least as well as at the other scope.

arm_rows and perm_p are copied from build_dimension_validation_matrix.py (contract-valid Selene
rows, contiguous topic rollup; two-sample permutation, 20,000 draws, seed 1234) rather than
imported, so running this cannot touch 02_VALIDATED_CORE_DIMENSIONS.csv. The owned-scope columns
must reproduce that file exactly; the script asserts it.

Output: data/processed/publication_final/construct_scope_contrast.csv
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from seg_eval.aggregation.scope_routing import family_of  # noqa: E402
from seg_eval.evaluation_judge.response_contract_v2 import validate_response_rows  # noqa: E402
from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS  # noqa: E402

OUT = REPO / "data/processed/publication_final/construct_scope_contrast.csv"
PUBLISHED = REPO / "docs/publication_final/02_VALIDATED_CORE_DIMENSIONS.csv"
ARMS = {"REF": ("tv_a", "20260829"), "PED_DEG": ("tv_b", "20260829")}


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def arm_rows(tag: str, stamp: str, fam: str) -> dict[str, dict]:
    """Copied from build_dimension_validation_matrix.py."""
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
    """Copied from build_dimension_validation_matrix.py."""
    if not a or not b:
        return None
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    pool, k, rnd, hits = a + b, len(a), random.Random(seed), 0
    for _ in range(n):
        rnd.shuffle(pool)
        if abs(statistics.fmean(pool[:k]) - statistics.fmean(pool[k:])) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def values(rows: dict, dim: str) -> list[float]:
    return [float(x) for x in ((o.get("dimension_scores") or {}).get(dim) for o in rows.values())
            if isinstance(x, (int, float))]


def verdict(own_sep: bool, other_sep: bool) -> str:
    if own_sep and not other_sep:
        return "owned scope only (routing supported)"
    if own_sep and other_sep:
        return "both scopes"
    if other_sep:
        return "other scope only (routing contradicted)"
    return "neither scope"


def main() -> None:
    data = {arm: {f: arm_rows(tag, stamp, f) for f in ("local", "arc")}
            for arm, (tag, stamp) in ARMS.items()}
    published = {r["dimension"]: r for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))}
    label = {"local": "segment", "arc": "topic"}

    print("units per arm:", {arm: {label[f]: len(v) for f, v in d.items()} for arm, d in data.items()})
    rows = []
    for d in PARTIAL_DIMENSIONS:
        rec = {"dimension": d,
               "owned_v1": label[family_of(d, "v1")],
               "owned_v2": label[family_of(d, "v2")]}
        sep = {}
        for fam in ("local", "arc"):
            a, b = values(data["REF"][fam], d), values(data["PED_DEG"][fam], d)
            units = len(data["REF"][fam]) + len(data["PED_DEG"][fam])
            lab = label[fam]
            rec[f"{lab}_n_scored"] = f"{len(a)}+{len(b)}"
            rec[f"{lab}_scored_rate"] = round((len(a) + len(b)) / units, 3) if units else None
            delta = round(statistics.fmean(a) - statistics.fmean(b), 4) if a and b else None
            p = round(perm_p(a, b), 4) if a and b else None
            rec[f"{lab}_delta"], rec[f"{lab}_p"] = delta, p
            sep[lab] = p is not None and p < 0.05 and delta is not None and delta > 0

        own = rec["owned_v1"]
        other = "topic" if own == "segment" else "segment"
        pub = published.get(d)
        if pub and pub.get("contrast_p") not in (None, "") and rec[f"{own}_p"] is not None:
            assert abs(float(pub["contrast_p"]) - rec[f"{own}_p"]) < 1e-4, (d, pub["contrast_p"], rec)
            assert abs(float(pub["contrast_delta"]) - rec[f"{own}_delta"]) < 1e-4, (d, pub["contrast_delta"], rec)
        rec["verdict_v1"] = verdict(sep[own], sep[other])
        rows.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("%-30s %-8s | %-18s | %-18s | %s" % ("dimension", "owned", "segment d / p", "topic d / p", "verdict"))
    for r in rows:
        seg = "%s / %s" % (r["segment_delta"], r["segment_p"])
        top = "%s / %s" % (r["topic_delta"], r["topic_p"])
        print("%-30s %-8s | %-18s | %-18s | %s" % (r["dimension"], r["owned_v1"], seg, top, r["verdict_v1"]))
    print("owned-scope columns reproduce 02_VALIDATED_CORE_DIMENSIONS.csv exactly (asserted)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
