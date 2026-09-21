"""Doc 62: Stage 1 symmetric-quote contradiction check + Stage 2 blind adjudication.

Every Stage 1 conflict must survive three MECHANICAL checks before it reaches Stage 2, and Stage 2
sees only the two spans -- no dialogue, no rationale, no Stage 1 reasoning. That is
Prover-Verifier Deliberation's exclusively-skeptical verifier (arXiv 2605.25133); withholding the
rationale is what stops the second pass ratifying the first pass's reasoning.

The mechanical checks are the part that does not depend on a model's judgement:
    tutor_span      must appear verbatim in the segment's tutor text
    definition_span must appear verbatim in one of the supplied reviewed definitions
    correct_version must differ non-trivially from tutor_span

The third check exists because measured hollow flags restated the quote they "corrected"
(doc 59: 5 of 12 regenerated material flags, 0 of 4 deployed).
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from seg_eval.evaluation_judge.selene_client_v1 import call_selene, health  # noqa: E402

SEV = ["material", "minor"]


def norm(t: str | None) -> str:
    return " ".join((t or "").lower().split())


def stage1_schema() -> dict:
    conflict = {
        "type": "object",
        "properties": {
            "tutor_span": {"type": "string", "minLength": 8},
            "definition_span": {"type": "string", "minLength": 8},
            "kc_id": {"type": "string"},
            "what_is_wrong": {"type": "string", "minLength": 4},
            "correct_version": {"type": "string", "minLength": 4},
            "severity": {"type": "string", "enum": SEV},
        },
        "required": ["tutor_span", "definition_span", "kc_id", "what_is_wrong",
                     "correct_version", "severity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "conflicts": {"type": "array", "items": conflict},
            "rationale_short": {"type": "string"},
        },
        "required": ["segment_id", "conflicts", "rationale_short"],
        "additionalProperties": False,
    }


def stage2_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "inconsistent": {"type": "boolean"},
            "why": {"type": "string", "minLength": 4},
        },
        "required": ["inconsistent", "why"],
        "additionalProperties": False,
    }


def mechanical_checks(conf: dict, tutor_text: str, defs: dict, variant: str = "v1") -> list[str]:
    """v1 is the configuration reported in doc 65 and is kept byte-exact for reproducibility.

    v2 repairs the third check. v1 rejected a `correct_version` whose CHARACTER similarity to the
    tutor span was >= 0.90, which is unsatisfiable for the error classes this mechanism is best at:
    correcting a reversal, a direction or a single constant changes one or two words, so a perfect
    correction scores 0.92-0.98 against the error it fixes and was discarded (doc 65 section 2.1).

    v2 rejects only the degenerate case, where the correction is the quote: equal token sequences
    after normalisation. Any genuine correction alters at least one token in place and so survives.
    Note the comparison must be on the token SEQUENCE and not the multiset, because a reversal
    ("test"/"training" swapped) uses exactly the same words in different positions. Hollow flags
    that merely pad the quote now pass to Stage 2, which is a semantic judge and the right place to
    settle them.
    """
    bad: list[str] = []
    ts = norm(conf.get("tutor_span"))
    ds = norm(conf.get("definition_span"))
    cv = norm(conf.get("correct_version"))
    if not ts or ts not in norm(tutor_text):
        bad.append("tutor_span:not_verbatim_in_tutor_text")
    if not ds or not any(ds in norm(d) for d in defs.values()):
        bad.append("definition_span:not_verbatim_in_any_definition")
    if ts and cv:
        if variant == "v2":
            hollow = ts.split() == cv.split()
        else:
            sim = difflib.SequenceMatcher(None, ts, cv).ratio()
            hollow = sim >= 0.90 or ts in cv or cv in ts
        if hollow:
            bad.append("correct_version:restates_tutor_span")
    return bad


def adjudicate(tutor_span: str, definition_span: str, base_url: str, variant: str = "v1"):
    """v1 is the configuration reported in doc 65, kept byte-exact.

    v2 closes one hole. v1's carve-out for a tutor statement that "adds detail" was read as excusing
    a corrupted formula: doc 65 section 2.2 records three correct catches dropped with reasons of the
    form "provides a more detailed and mathematically precise definition ... consistent with the
    curriculum statement". The carve-out is kept, because it is what suppresses false alarms on
    correct material, and an exception is added for the case where both statements pin the same
    quantity differently.
    """
    system = ("You judge whether two statements are logically inconsistent. You are skeptical: "
              "say inconsistent only if they cannot both be true. Return only valid JSON.")
    detail_rule = (
        "- If the tutor statement adds detail, simplifies, or covers something the curriculum "
        "statement does not mention, they are NOT inconsistent.\n"
    )
    if variant == "v2":
        detail_rule += (
            "- BUT if both statements specify the SAME formula, constant, condition, ordering or "
            "scope, and they specify it differently, they ARE inconsistent. Being more detailed, "
            "more formal, or more precisely worded does not make a different value, a different "
            "term, a different direction or a different order compatible.\n"
        )
    user = (
        "CURRICULUM STATEMENT (authoritative):\n" + definition_span + "\n\n"
        "TUTOR STATEMENT:\n" + tutor_span + "\n\n"
        "Can both statements be true at the same time?\n"
        + detail_rule +
        "- Say inconsistent ONLY if the curriculum statement makes the tutor statement false.\n\n"
        "Return JSON: {\"inconsistent\": true or false, \"why\": \"<one sentence>\"}"
    )
    return call_selene(user, system_message=system, schema=stage2_schema(), base_url=base_url,
                       temperature=0.0, top_p=1.0, seed=None, max_tokens=300,
                       schema_name="blind_adjudication")


def run_retro(path: Path, out: Path, base_url: str) -> None:
    """Validate the filter independently, on flags produced by earlier arms."""
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    kept: dict[str, list[int]] = {}
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(rows, 1):
            res = adjudicate(r["tutor_span"], r["definition_span"], base_url)
            inc = bool((res.payload or {}).get("inconsistent"))
            src = r.get("source", "?")
            kept.setdefault(src, [0, 0])
            kept[src][0] += int(inc)
            kept[src][1] += 1
            fh.write(json.dumps({**r, "adjudicated_inconsistent": inc,
                                 "why": (res.payload or {}).get("why")}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [retro {i}/{len(rows)}] {src} {r.get('segment_id')} keep={inc}", flush=True)
    print("RETRO_KEPT=" + json.dumps({k: f"{v[0]}/{v[1]}" for k, v in kept.items()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--retro")
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--variant", choices=["v1", "v2"], default="v1",
                    help="v1 reproduces doc 65 exactly; v2 is the repaired configuration (doc 66)")
    args = ap.parse_args()
    print(f"variant={args.variant}", flush=True)

    if health(args.base_url) is None:
        raise SystemExit(f"Selene not reachable at {args.base_url}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.retro:
        run_retro(Path(args.retro), out, args.base_url)
        return

    rows = [json.loads(l) for l in Path(args.prompts).open(encoding="utf-8") if l.strip()]
    print(f"stage1 prompts={len(rows)}", flush=True)
    schema = stage1_schema()
    tally = {"raised": 0, "mech_rejected": 0, "stage2_dropped": 0, "survived": 0}
    reasons: dict[str, int] = {}

    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(rows, 1):
            res = call_selene(row["user_prompt"], system_message=row["system_prompt"],
                              schema=schema, base_url=args.base_url, temperature=0.0, top_p=1.0,
                              seed=None, max_tokens=2000, schema_name="definition_factuality")
            payload = res.payload or {}
            survivors, audit = [], []
            for c in (payload.get("conflicts") or []):
                tally["raised"] += 1
                bad = mechanical_checks(c, row["allowed_tutor_text"], row["definitions"],
                                        variant=args.variant)
                if bad:
                    tally["mech_rejected"] += 1
                    for b in bad:
                        reasons[b] = reasons.get(b, 0) + 1
                    audit.append({**c, "stage": "mech_rejected", "failures": bad})
                    continue
                ad = adjudicate(c["tutor_span"], c["definition_span"], args.base_url,
                                variant=args.variant)
                if bool((ad.payload or {}).get("inconsistent")):
                    tally["survived"] += 1
                    survivors.append({**c, "stage2_why": (ad.payload or {}).get("why")})
                else:
                    tally["stage2_dropped"] += 1
                    audit.append({**c, "stage": "stage2_dropped",
                                  "why": (ad.payload or {}).get("why")})
            material = sum(1 for c in survivors if c.get("severity") == "material")
            verdict = ("contradicted" if material
                       else ("minor_deviation" if survivors else "grounded"))
            fh.write(json.dumps({
                "segment_id": row["segment_id"],
                "reference_kc_ids": row["reference_kc_ids"],
                "model_path": "Selene-1-Llama-3.3-70B",
                "served_model_name": "selene-1-llama-3.3-70b",
                "decoding": {"temperature": 0.0, "top_p": 1.0, "greedy": True},
                "schema_ok": res.schema_ok,
                "json_extracted": bool(payload),
                "derived_verdict": verdict,
                "material_contradictions": material,
                "surviving_conflicts": survivors,
                "rejected_conflicts": audit,
                "raw_rationale": payload.get("rationale_short"),
                "client_error": res.error,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(rows)}] {row['segment_id'].split('::')[-1]} {verdict} "
                  f"mat={material} raised={len(payload.get('conflicts') or [])} "
                  f"kept={len(survivors)}", flush=True)

    print(f"OUT={out}")
    print("TALLY=" + json.dumps(tally))
    print("MECH_REJECT_REASONS=" + json.dumps(reasons))


if __name__ == "__main__":
    main()
