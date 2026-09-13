"""Build grounded-factuality prompts for each tutor arm's local evaluation units.

One prompt per unit: the segment plus the curriculum reference for its KCs. Units whose KCs
carry no reference content are skipped and recorded, so their verdict is absent rather than
silently 'grounded'.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.grounded_factuality_v1 import (  # noqa: E402
    FACTUALITY_CONTRACT_VERSION, REVIEWED_LIBRARY, load_reference_by_kc,
    reference_for_packet, system_prompt, user_prompt, numbered_error_patterns, PROMPT_REVISION,
)

ARMS = ["tv_a", "tv_b", "tv_c"]


def _prompt_rows(packets: list[dict], ref: dict, max_kcs: int) -> tuple[list[dict], list[str]]:
    """Shared by the arm sweep and the single-corpus path so both build identical prompts."""
    rows, skipped = [], []
    for p in packets:
        r = reference_for_packet(p, ref, max_kcs=max_kcs)
        if not r:
            skipped.append(p["segment_id"])
            continue
        rows.append({
            "factuality_prompt_id": f"{p['segment_id']}::grounded_factuality_v1",
            "segment_id": p["segment_id"],
            "contract_version": FACTUALITY_CONTRACT_VERSION,
            "prompt_revision": PROMPT_REVISION,
            "reference_kc_ids": [b["kc_id"] for b in r],
            "system_prompt": system_prompt(),
            "user_prompt": user_prompt(p, r),
        })
    return rows, skipped


def _build_one(packets_path: Path, out_dir: Path, ref: dict, max_kcs: int) -> None:
    packets = [json.loads(l) for l in packets_path.open(encoding="utf-8") if l.strip()]
    rows, skipped = _prompt_rows(packets, ref, max_kcs)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grounded_factuality_prompts.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "contract_version": FACTUALITY_CONTRACT_VERSION,
        "units_total": len(packets),
        "units_with_reference": len(rows),
        "skipped_no_reference": skipped,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (out_dir / "factuality_prompt_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"OUT={out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260829")
    ap.add_argument("--max-kcs", type=int, default=3)
    # Explicit paths let a single arbitrary corpus be built (what the console drives). Omitting
    # them keeps the original three-arm sweep, so every earlier reproduction still runs unchanged.
    ap.add_argument("--packets", default="",
                    help="one segment_evaluation_packets.jsonl; overrides the tv_a/b/c sweep")
    ap.add_argument("--out", default="", help="output directory, required with --packets")
    ap.add_argument("--library", default="",
                    help="reviewed library to read evaluation_support from; "
                         "defaults to corpus_config's resolution")
    args = ap.parse_args()

    if bool(args.packets) != bool(args.out):
        ap.error("--packets and --out must be given together")

    library = Path(args.library) if args.library else (REPO_ROOT / REVIEWED_LIBRARY)
    ref = load_reference_by_kc(library)
    print(f"curriculum reference loaded for {len(ref)} KCs from {library}")

    if args.packets:
        _build_one(Path(args.packets), Path(args.out), ref, args.max_kcs)
        return

    summary = {}
    for arm in ARMS:
        pkts = [json.loads(l) for l in (
            REPO_ROOT / f"data/processed/evaluation_packets/v35_{arm}_eval_{args.tag}"
            / "kc_segment_local/segment_evaluation_packets.jsonl").open(encoding="utf-8") if l.strip()]
        rows, skipped = _prompt_rows(pkts, ref, args.max_kcs)
        out_dir = REPO_ROOT / f"data/processed/judge_prompts/v35_{arm}_factuality_{args.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "grounded_factuality_prompts.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        avg = sum(len(r["user_prompt"]) for r in rows) / max(1, len(rows))
        summary[arm] = {"units_total": len(pkts), "units_with_reference": len(rows),
                        "skipped_no_reference": skipped, "mean_prompt_chars": round(avg),
                        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}
        print(f"  {arm}: {len(rows)}/{len(pkts)} units, mean prompt {avg:.0f} chars")

    man = REPO_ROOT / f"data/processed/judge_prompts/factuality_{args.tag}_manifest.json"
    man.write_text(json.dumps({
        "contract_version": FACTUALITY_CONTRACT_VERSION,
        "reviewed_library": REVIEWED_LIBRARY,
        "max_kcs_per_prompt": args.max_kcs,
        "design_note": ("Holistic grounded verdict against a supplied reference, in a separate "
                        "narrow pass. NOT atomic decomposition (contraindicated) and NOT in the "
                        "main rubric prompt (measured harmful). See 27_..._CITATIONS.md."),
        "arms": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MANIFEST={man.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
