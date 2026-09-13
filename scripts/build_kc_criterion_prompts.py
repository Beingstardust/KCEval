"""Build KC-specific criterion-judgment prompts for each tutor arm's local evaluation units.

One prompt per unit, carrying that unit's candidate criteria (primary KC + evaluator_kc_set).
Units with no candidate criteria are skipped and recorded, so K_s is None for them rather than
silently 0.

Local family only: KC-specific criteria are per-KC claims, and arc units are deliberately
multi-KC rollups whose "primary KC" is a topic label rather than a leaf KC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.kc_criterion_judge_v1 import (  # noqa: E402
    CRITERION_CONTRACT_VERSION, REVIEWED_LIBRARY, load_reviewer_criteria,
    candidate_criteria_for_packet, system_prompt, user_prompt,
)

ARMS = ["tv_a", "tv_b", "tv_c"]


def _prompt_rows(packets: list[dict], criteria_by_kc: dict) -> tuple[list[dict], list[str]]:
    """Shared by the arm sweep and the single-corpus path so both build identical prompts."""
    rows, skipped = [], []
    for pkt in packets:
        criteria = candidate_criteria_for_packet(pkt, criteria_by_kc)
        if not criteria:
            skipped.append(pkt["segment_id"])
            continue
        rows.append({
            "criterion_prompt_id": f"{pkt['segment_id']}::kc_criterion_v1",
            "segment_id": pkt["segment_id"],
            "contract_version": CRITERION_CONTRACT_VERSION,
            "criterion_ids": [c["criterion_id"] for c in criteria],
            "system_prompt": system_prompt(),
            "user_prompt": user_prompt(pkt, criteria),
        })
    return rows, skipped


def _build_one(packets_path: Path, out_dir: Path, criteria_by_kc: dict) -> None:
    packets = [json.loads(l) for l in packets_path.open(encoding="utf-8") if l.strip()]
    rows, skipped = _prompt_rows(packets, criteria_by_kc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kc_criterion_prompts.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "contract_version": CRITERION_CONTRACT_VERSION,
        "units_total": len(packets),
        "units_with_criteria": len(rows),
        "units_skipped_no_criteria": skipped,
        "criterion_instances": sum(len(r["criterion_ids"]) for r in rows),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (out_dir / "kc_criterion_prompt_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"OUT={out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260829")
    # Explicit paths let a single arbitrary corpus be built (what the console drives). Omitting
    # them keeps the original three-arm sweep, so every earlier reproduction still runs unchanged.
    ap.add_argument("--packets", default="",
                    help="one segment_evaluation_packets.jsonl; overrides the tv_a/b/c sweep")
    ap.add_argument("--out", default="", help="output directory, required with --packets")
    ap.add_argument("--library", default="",
                    help="reviewed library to read reviewer_criteria from; "
                         "defaults to corpus_config's resolution")
    args = ap.parse_args()

    if bool(args.packets) != bool(args.out):
        ap.error("--packets and --out must be given together")

    library = Path(args.library) if args.library else (REPO_ROOT / REVIEWED_LIBRARY)
    criteria_by_kc = load_reviewer_criteria(library)
    total_criteria = sum(len(v) for v in criteria_by_kc.values())
    print(f"reviewer criteria loaded: {total_criteria} across {len(criteria_by_kc)} KCs "
          f"(NONE expert-approved -- see 05_KC_CRITERIA_WEIGHT_AUDIT.md)")

    if args.packets:
        _build_one(Path(args.packets), Path(args.out), criteria_by_kc)
        return

    summary = {}
    for arm in ARMS:
        packets_path = (REPO_ROOT / f"data/processed/evaluation_packets/v35_{arm}_eval_{args.tag}"
                        / "kc_segment_local/segment_evaluation_packets.jsonl")
        packets = [json.loads(l) for l in packets_path.open(encoding="utf-8") if l.strip()]

        rows, skipped = _prompt_rows(packets, criteria_by_kc)
        out_dir = REPO_ROOT / f"data/processed/judge_prompts/v35_{arm}_kc_criteria_{args.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "kc_criterion_prompts.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_crit = sum(len(r["criterion_ids"]) for r in rows)
        summary[arm] = {
            "units_total": len(packets), "units_with_criteria": len(rows),
            "units_skipped_no_criteria": skipped, "criterion_instances": n_crit,
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        }
        print(f"  {arm}: {len(rows)}/{len(packets)} units, {n_crit} criterion instances "
              f"-> {out_path.relative_to(REPO_ROOT)}")

    man = REPO_ROOT / f"data/processed/judge_prompts/kc_criteria_{args.tag}_manifest.json"
    man.write_text(json.dumps({
        "contract_version": CRITERION_CONTRACT_VERSION,
        "reviewed_library": REVIEWED_LIBRARY,
        "criteria_total": total_criteria,
        "criteria_expert_approved": 0,
        "approval_note": ("reviewer_criteria are DRAFT. kc_specific_criteria is empty on all 159 "
                           "KCs. Any K_s / B_s computed from these is provisional."),
        "arms": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MANIFEST={man.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
