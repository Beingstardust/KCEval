"""Doc 66: stage-by-stage analysis of a definition-grounded factuality holdout.

Generalised from the dm3 analyser so dm4 is scored by code frozen before its responses exist.

Two things are done differently from the dm3 version, both because of mistakes that version made.

1. **The ledger-to-packet id mapping is discovered, not assumed.** On dm3 the ledger numbered
   exchanges from ex_0001 and ingestion from ex_0000, and scoring against the raw ids produced a
   false "2/16 indexing ceiling" that nearly killed the corpus. Here each ledger entry is located by
   finding the packet exchange whose tutor text actually contains its corrupted span. The mapping is
   reported, and any entry that cannot be located is a hard error rather than a silent miss.

2. **The stage decomposition is computed by default.** Doc 65 had to reconstruct after the fact that
   the filters, not the detector, were the limiting component. Doc 66 section 5 commits to reporting
   it by design.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

PRIMARY_OVERLAP = 24
OVERLAP_SWEEP = (16, 24, 32)


def norm(t: str | None) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()


def jl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def short_id(raw: Any) -> str:
    return str(raw or "").rsplit("::", 1)[-1]


def overlap_len(a: str, b: str) -> int:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0
    if na in nb or nb in na:
        return min(len(na), len(nb))
    m = difflib.SequenceMatcher(None, na, nb).find_longest_match(0, len(na), 0, len(nb))
    return m.size


def load_packets(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """returns (exchange_id -> segment_id, exchange_id -> tutor_text)"""
    ex2seg, ex2txt = {}, {}
    for pkt in jl(path):
        for ex in pkt.get("member_exchanges") or []:
            k = short_id(ex.get("exchange_id"))
            ex2seg[k] = pkt["segment_id"]
            ex2txt[k] = ex.get("tutor_text") or ""
    return ex2seg, ex2txt


def locate(errors: list[dict], ex2txt: dict[str, str]) -> dict[str, str]:
    """Map each ledger exchange_id to the packet exchange_id actually carrying its corrupted span."""
    mapping, unresolved = {}, []
    for e in errors:
        span = norm(e["corrupted_span"])
        hits = [k for k, v in ex2txt.items() if span and span in norm(v)]
        if len(hits) == 1:
            mapping[e["exchange_id"]] = hits[0]
        else:
            unresolved.append((e["exchange_id"], len(hits)))
    if unresolved:
        raise SystemExit(
            "could not uniquely locate these ledger entries in the corrupt packets "
            f"(id, candidate count): {unresolved}. The corpus or the ingestion is wrong; "
            "do not score around it."
        )
    return mapping


def survivors(row: dict[str, Any], material_only: bool = False) -> list[dict[str, Any]]:
    cs = row.get("surviving_conflicts") or []
    return [c for c in cs if c.get("severity") == "material"] if material_only else cs


def all_conflicts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("surviving_conflicts") or []) + list(row.get("rejected_conflicts") or [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="e.g. dm4")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--responses", required=True, help="dir containing <arm>/definition_factuality_responses.jsonl")
    ap.add_argument("--packets", required=True, help="dir containing <corpus>_<arm>_*/segment_evaluation_packets.jsonl")
    ap.add_argument("--clean-arm", default="clean")
    ap.add_argument("--corrupt-arm", default="corrupt")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    errors = sorted(gold["injected_errors"], key=lambda r: r["exchange_id"])
    n_err = len(errors)

    resp = {}
    for arm in (args.clean_arm, args.corrupt_arm):
        p = Path(args.responses) / arm / "definition_factuality_responses.jsonl"
        if not p.exists():
            raise SystemExit(f"missing responses: {p}")
        resp[arm] = {r["segment_id"]: r for r in jl(p)}

    pk = {}
    for arm in (args.clean_arm, args.corrupt_arm):
        cands = sorted(Path(args.packets).glob(f"{args.corpus}_{arm}_*/segment_evaluation_packets.jsonl"))
        if not cands:
            raise SystemExit(f"no packets for arm {arm} under {args.packets}")
        pk[arm] = load_packets(cands[-1])

    ex2seg, ex2txt = pk[args.corrupt_arm]
    mapping = locate(errors, ex2txt)
    offsets = {int(k.split("_")[1]) - int(v.split("_")[1]) for k, v in mapping.items()}

    rep: dict[str, Any] = {
        "corpus": args.corpus,
        "preregistration": "docs/evaluation_completeness_v2/66_REPAIRED_MECHANISM_PREREGISTRATION.md",
        "injected_errors": n_err,
        "id_mapping": {"ledger_to_packet": mapping,
                       "uniform_offset": sorted(offsets)[0] if len(offsets) == 1 else None,
                       "note": "discovered by locating each corrupted span, not assumed"},
        "units": {a: len(resp[a]) for a in resp},
        "contract_valid": {a: sum(1 for r in resp[a].values()
                                  if r.get("schema_ok") and r.get("json_extracted"))
                           for a in resp},
    }

    # ---- pre-scoring catchability ceiling (doc 66 section 5, item 6) ----
    ceiling, no_kc = 0, []
    for e in errors:
        seg = ex2seg[mapping[e["exchange_id"]]]
        kcs = (resp[args.corrupt_arm].get(seg) or {}).get("reference_kc_ids") or []
        if e["kc_id"] in kcs:
            ceiling += 1
        else:
            no_kc.append(e["exchange_id"])
    rep["catchability_ceiling"] = {"ledger_kc_in_reference": f"{ceiling}/{n_err}",
                                   "errors_without_their_kc": no_kc}

    # ---- stage decomposition (doc 66 section 5, items 1-3) ----
    # A segment can carry MORE THAN ONE injected error (dm4: two segments do), so this maps to a
    # list. An earlier segment-to-error dict silently dropped the extra errors.
    err_by_seg: dict[str, list[dict]] = {}
    for e in errors:
        err_by_seg.setdefault(ex2seg[mapping[e["exchange_id"]]], []).append(e)
    rep["segments_with_multiple_errors"] = {
        s: [e["exchange_id"] for e in es] for s, es in err_by_seg.items() if len(es) > 1
    }
    stages: dict[str, Any] = {}
    for arm in (args.clean_arm, args.corrupt_arm):
        raised = mech = s2 = surv = 0
        reasons: Counter = Counter()
        on_err = {"raised": 0, "mech_rejected": 0, "stage2_dropped": 0, "survived": 0}
        for sid, r in resp[arm].items():
            es = err_by_seg.get(sid, []) if arm == args.corrupt_arm else []
            for c in all_conflicts(r):
                raised += 1
                hits_err = any(overlap_len(c.get("tutor_span"), e["corrupted_span"]) >= PRIMARY_OVERLAP
                               for e in es)
                on_err["raised"] += hits_err
                st = c.get("stage")
                if st == "mech_rejected":
                    mech += 1
                    on_err["mech_rejected"] += hits_err
                    for f in c.get("failures") or []:
                        reasons[f] += 1
                elif st == "stage2_dropped":
                    s2 += 1
                    on_err["stage2_dropped"] += hits_err
                else:
                    surv += 1
                    on_err["survived"] += hits_err
        stages[arm] = {"raised": raised, "mech_rejected": mech, "stage2_dropped": s2,
                       "survived": surv, "mech_reject_reasons": dict(reasons.most_common()),
                       "conflicts_localising_a_ledger_error": on_err}
    rep["stages"] = stages

    # ---- recall, per category, sensitivity ----
    def score(threshold: int) -> tuple[int, list[dict]]:
        hits, per = 0, []
        for e in errors:
            seg = ex2seg[mapping[e["exchange_id"]]]
            r = resp[args.corrupt_arm].get(seg) or {}
            best_s = max((overlap_len(c.get("tutor_span"), e["corrupted_span"])
                          for c in survivors(r)), default=0)
            best_a = max((overlap_len(c.get("tutor_span"), e["corrupted_span"])
                          for c in all_conflicts(r)), default=0)
            ok = best_s >= threshold
            hits += ok
            per.append({"exchange_id": e["exchange_id"], "packet_exchange_id": mapping[e["exchange_id"]],
                        "segment_id": seg, "category": e["category"], "kc_id": e["kc_id"],
                        "caught": ok, "best_overlap_chars": best_s,
                        "detected_by_stage1": best_a >= threshold})
            per[-1]["lost_to_filters"] = per[-1]["detected_by_stage1"] and not ok
        return hits, per

    hits, per = score(PRIMARY_OVERLAP)
    cats: dict[str, dict[str, int]] = {}
    for r in per:
        d = cats.setdefault(r["category"][:2], {"caught": 0, "stage1": 0, "total": 0})
        d["total"] += 1
        d["caught"] += int(r["caught"])
        d["stage1"] += int(r["detected_by_stage1"])
    s1_recall = sum(1 for r in per if r["detected_by_stage1"])

    clean_flagged = [s for s, r in resp[args.clean_arm].items() if survivors(r)]
    corrupt_flags = sum(len(survivors(r)) for r in resp[args.corrupt_arm].values())
    clean_flags = sum(len(survivors(r)) for r in resp[args.clean_arm].values())
    total_flags = corrupt_flags + clean_flags
    err_segs = set(err_by_seg)
    corrupt_clean_seg_flags = [s for s, r in resp[args.corrupt_arm].items()
                               if s not in err_segs and survivors(r)]

    def grp(gs: set[str], key: str, rows: list[dict] | None = None) -> str:
        rows = [r for r in (rows if rows is not None else per) if r["category"][:2] in gs]
        k = sum(int(r[key]) for r in rows)
        return f"{k}/{len(rows)}"

    # doc 66 section 6: recall restricted to errors whose tutor turn is at least as long as dm3's
    # shortest (83 chars). Defined by turn length alone and fixed before scoring.
    DM3_MIN_TURN = 83
    for r in per:
        r["tutor_turn_chars"] = len(ex2txt.get(r["packet_exchange_id"], ""))
        r["dm3_comparable"] = r["tutor_turn_chars"] >= DM3_MIN_TURN
    comp = [r for r in per if r["dm3_comparable"]]
    comp_hits = sum(1 for r in comp if r["caught"])
    rep["dm3_comparable_subset"] = {
        "definition": f"injected tutor turn >= {DM3_MIN_TURN} chars (dm3's shortest)",
        "n": len(comp),
        "recall": f"{comp_hits}/{len(comp)}",
        "stage1_recall": f"{sum(1 for r in comp if r['detected_by_stage1'])}/{len(comp)}",
        "C1_C3_C8_recall": grp({"C1", "C3", "C8"}, "caught", comp),
        "excluded_short_turn_errors": [r["exchange_id"] for r in per if not r["dm3_comparable"]],
    }

    rep["results"] = {
        "stage1_detector_recall": f"{s1_recall}/{n_err}",
        "final_recall": f"{hits}/{n_err}",
        "lost_to_filters": sum(1 for r in per if r["lost_to_filters"]),
        "precision_both_arms": (hits / total_flags) if total_flags else None,
        "clean_arm_segments_flagged": f"{len(clean_flagged)}/{len(resp[args.clean_arm])}",
        "corrupt_arm_error_free_segments_flagged":
            f"{len(corrupt_clean_seg_flags)}/{len(resp[args.corrupt_arm]) - len(err_segs)}",
        "per_category": dict(sorted(cats.items())),
        "per_error": per,
        "recall_sensitivity": {f"@{t}": f"{score(t)[0]}/{n_err}" for t in OVERLAP_SWEEP},
    }

    # ---- preregistered predictions (doc 66 section 4) ----
    c138 = grp({"C1", "C3", "C8"}, "caught")
    k, tot = (int(x) for x in c138.split("/"))
    rep["predictions"] = {
        "P1_clean_flag_rate": {"value": f"{len(clean_flagged)}/{len(resp[args.clean_arm])}",
                               "threshold": "<= 3/60", "met": len(clean_flagged) <= 3},
        "P2_recall": {"value": f"{hits}/{n_err}", "threshold": ">= 12/24",
                      "met": hits / n_err >= 0.5 if n_err else False},
        "P3_precision": {"value": rep["results"]["precision_both_arms"], "threshold": ">= 0.85",
                         "met": bool(total_flags) and hits / total_flags >= 0.85},
        "P4_C1_C3_C8_recall": {"value": c138, "threshold": ">= 0.50",
                               "met": (k / tot) >= 0.50 if tot else False},
        # doc 66 section 3.2: same group, excluding errors whose ledger KC was never retrieved,
        # which no checker could settle from the references it was given.
        "P4_C1_C3_C8_recall_retrieved_only": {
            "value": grp({"C1", "C3", "C8"}, "caught",
                         [r for r in per if r["exchange_id"] not in set(no_kc)]),
            "excluded": [x for x in no_kc],
        },
        "regression_floor": {"threshold": "recall >= 10/24 or the repair is withdrawn",
                             "breached": (hits / n_err) < (10/24) if n_err else True},
    }

    out = Path(args.out) if args.out else Path(args.responses).parent / "analysis" / f"{args.corpus}_definition_factuality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in rep.items() if k != "results"}, indent=2))
    print("\nSTAGE 1 detector", rep["results"]["stage1_detector_recall"],
          "-> FINAL", rep["results"]["final_recall"],
          f"(lost to filters: {rep['results']['lost_to_filters']})")
    print("per-category (caught/stage1/total):",
          json.dumps(rep["results"]["per_category"]))
    print("P1", rep["predictions"]["P1_clean_flag_rate"]["met"],
          "P2", rep["predictions"]["P2_recall"]["met"],
          "P3", rep["predictions"]["P3_precision"]["met"],
          "P4", rep["predictions"]["P4_C1_C3_C8_recall"]["met"])
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
