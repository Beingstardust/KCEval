"""Doc 62 Stage 1 prompts: segment vs the verbatim expert-reviewed definition.

No generated reference content. The definition is supplied as-is and is the authority a
contradiction must cite (see doc 61 P1/P2 and doc 62).
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/processed/claim_level_factuality_20260920/source_definitions/current_deployed_with_console_freeze_overrides.jsonl"
# Corpora. The prompt itself is frozen by doc 64; only paths vary here.
CORPORA = {
    "dm1": (
        REPO / "data/processed/definition_factuality_cal_20260920",
        {
            "tv_a": REPO / "data/processed/evaluation_packets/v35_tv_a_eval_20260829/kc_segment_local/segment_evaluation_packets.jsonl",
            "tv_c": REPO / "data/processed/evaluation_packets/v35_tv_c_eval_20260829/kc_segment_local/segment_evaluation_packets.jsonl",
        },
    ),
    # dm2 E7 corruption arms, for the RQ3 swap. The mechanism was designed on dm1 and confirmed on
    # dm3; it has never been run on dm2, so dm2 is unseen by it.
    "dm2": (
        REPO / "data/processed/definition_factuality_dm2_20260920",
        {
            "clean": REPO / "data/processed/console_runs/E7S_hv_p/evaluation_packets/kc_segment_local/segment_evaluation_packets.jsonl",
            "corrupt": REPO / "data/processed/console_runs/E7S_hv_q/evaluation_packets/kc_segment_local/segment_evaluation_packets.jsonl",
        },
    ),
    # dm4 holdout (doc 66). Packets sit at the run-directory root.
    "dm4": (
        REPO / "data/processed/definition_factuality_dm4_20260920",
        {
            "clean": REPO / "data/processed/evaluation_packets/dm4_clean_20260920/segment_evaluation_packets.jsonl",
            "corrupt": REPO / "data/processed/evaluation_packets/dm4_corrupt_20260920/segment_evaluation_packets.jsonl",
        },
    ),
    # dm3 holdout (doc 64). Packets sit at the run-directory root, not under kc_segment_local/.
    "dm3": (
        REPO / "data/processed/definition_factuality_dm3_20260920",
        {
            "clean": REPO / "data/processed/evaluation_packets/dm3_clean_20260920/segment_evaluation_packets.jsonl",
            "corrupt": REPO / "data/processed/evaluation_packets/dm3_corrupt_20260920/segment_evaluation_packets.jsonl",
        },
    ),
}
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "dm1"
OUT, PACKETS = CORPORA[CORPUS]
MAX_KCS = 3


def rj(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def segment_lines(pkt):
    out = []
    for ex in pkt.get("member_exchanges") or []:
        i = ex.get("exchange_id", "?")
        if (ex.get("student_text") or "").strip():
            out.append(f"[{i}] Student: {ex['student_text']}")
        if (ex.get("tutor_text") or "").strip():
            out.append(f"[{i}] Tutor: {ex['tutor_text']}")
    return "\n".join(out)


def tutor_text(pkt):
    return " ".join((e.get("tutor_text") or "") for e in (pkt.get("member_exchanges") or []))


def system_prompt():
    return ("You are a curriculum fact-checker. You judge only whether a tutor's statements conflict "
            "with the reviewed curriculum definitions supplied to you. Return only valid JSON. "
            "Do not include hidden chain-of-thought.")


def user_prompt(pkt, defs):
    block = "\n\n".join(f"[{d['kc_id']}] {d['canonical_name']}\n{d['definition']}" for d in defs)
    return (
        "REVIEWED CURRICULUM DEFINITIONS (authoritative; these are the only source of truth):\n"
        f"{block}\n\n"
        "TUTORING SEGMENT:\n" + segment_lines(pkt) + "\n\n"
        "Question: does the TUTOR state anything that CONFLICTS with the definitions above?\n\n"
        "Rules:\n"
        "- Judge the TUTOR's words only. A student's own statement is never a tutor error.\n"
        "- A tutor endorsing a student's incorrect claim IS a tutor error.\n"
        "- The definitions are NOT exhaustive. If the tutor says something the definitions simply do "
        "not cover, that is NOT a conflict. Say nothing about it.\n"
        "- Being brief, simplifying, using an analogy, or giving a different example is NOT a conflict.\n"
        "- Only a direct conflict with something a definition actually STATES counts.\n"
        "- A statement that REVERSES, SWAPS or INVERTS what a definition states IS a conflict, "
        "even if the wording is otherwise similar.\n\n"
        "For every conflict you report you MUST supply all three of:\n"
        "  tutor_span      - the tutor's exact words, copied VERBATIM from the segment\n"
        "  definition_span - the exact sentence or clause from a definition above, copied VERBATIM,\n"
        "                    that the tutor's words conflict with\n"
        "  correct_version - what the tutor should have said instead\n"
        "If you cannot copy a verbatim definition_span that the tutor actually conflicts with, then "
        "there is no conflict to report.\n\n"
        "Return ONE JSON object and nothing else:\n"
        "{\n"
        f'  "segment_id": "{pkt.get("segment_id")}",\n'
        '  "conflicts": [\n'
        '    {"tutor_span": "<verbatim tutor words>",\n'
        '     "definition_span": "<verbatim words from a definition above>",\n'
        '     "kc_id": "<which definition>",\n'
        '     "what_is_wrong": "<why these conflict>",\n'
        '     "correct_version": "<what the tutor should have said>",\n'
        '     "severity": "material | minor"}\n'
        "  ],\n"
        '  "rationale_short": "<1-3 sentences>"\n'
        "}\n"
        "- conflicts MUST be [] when nothing conflicts.\n"
        "- severity is 'material' if a student acting on the claim would be wrong.\n"
    )


def main():
    defs = {r["kc_id"]: r for r in rj(SRC)}
    manifest = {}
    for arm, pp in PACKETS.items():
        rows = []
        for pkt in rj(pp):
            t = pkt.get("evaluation_target") or {}
            kcs, seen = [], set()
            for k in [t.get("primary_kc_id"), *(t.get("evaluator_kc_set") or [])]:
                if k and k in defs and k not in seen:
                    seen.add(k); kcs.append(defs[k])
                if len(kcs) >= MAX_KCS:
                    break
            if not kcs:
                continue
            rows.append({
                "factuality_prompt_id": f"{pkt['segment_id']}::definition_factuality_v1",
                "segment_id": pkt["segment_id"],
                "reference_kc_ids": [d["kc_id"] for d in kcs],
                "definitions": {d["kc_id"]: d["definition"] for d in kcs},
                "allowed_tutor_text": tutor_text(pkt),
                "system_prompt": system_prompt(),
                "user_prompt": user_prompt(pkt, kcs),
            })
        p = OUT / f"prompts/{arm}/definition_factuality_prompts.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[arm] = {"units": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        print(f"WROTE {p.relative_to(REPO)} ({len(rows)} units)")
    prereg = ("docs/evaluation_completeness_v2/64_DM3_HOLDOUT_PREREGISTRATION.md" if CORPUS == "dm3"
              else "docs/evaluation_completeness_v2/62_DEFINITION_GROUNDED_FACTUALITY_PREREGISTRATION.md")
    (OUT / "prompt_manifest.json").write_text(json.dumps({
        "corpus": CORPUS, "preregistration": prereg,
        "reference": str(SRC.relative_to(REPO)), "max_kcs": MAX_KCS, "arms": manifest,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
