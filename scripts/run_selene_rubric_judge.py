"""Run the main segment rubric through Selene, using the EXISTING prompts byte-for-byte.

This is the decisive experiment for the re-baseline: the dm1/dm2 judge prompts already on disk
are exactly what Qwen3-8B saw, so sending them unchanged to Selene isolates the model as the
only variable. Agreement with the 115 human labels can then be compared directly against Qwen's.

The response schema is built from response_contract_v2's own constants rather than hand-written,
so grammar enforcement and contract validation cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.selene_client_v1 import call_selene, health  # noqa: E402
from seg_eval.evaluation_judge.selene_rubric_schema_v1 import (  # noqa: E402
    build_wire_schema, to_contract_shape, allowed_kc_ids_for_packet, WIRE_FORMAT_INSTRUCTION,
    WIRE_SCHEMA_VERSION,
)
from seg_eval.evaluation_judge.response_contract_v2 import (  # noqa: E402
    validate_judge_response, packet_segment_text, allowed_kc_ids_from_packet,
)


def resolve_model_revision(model_dir: str) -> tuple[str | None, str]:
    """Read the served weights' HuggingFace commit from the download metadata.

    Recording this at run time is the fix for a gap already found once on the Qwen r4 runs, where
    the revision had to be reconstructed afterwards from a Slurm log. Every ``*.metadata`` file
    written by ``hf download`` carries the commit on its first line; requiring all of them to
    agree means a partially-updated directory is reported rather than silently pinned to whichever
    file happened to be read first.
    """
    if not model_dir:
        return None, "not_recorded: --model-dir not supplied"
    meta = Path(model_dir) / ".cache" / "huggingface" / "download"
    if not meta.is_dir():
        return None, f"not_recorded: no download metadata under {meta}"
    revs = set()
    for f in meta.glob("*.metadata"):
        try:
            first = f.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if first:
            revs.add(first)
    if len(revs) == 1:
        return revs.pop(), f"read_from_hf_download_metadata ({len(list(meta.glob('*.metadata')))} files agree)"
    if not revs:
        return None, f"not_recorded: no revisions found in {meta}"
    return None, f"not_recorded: conflicting revisions in download metadata: {sorted(revs)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://gpu03:8001/v1")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-dir", default="",
                    help="served weights directory; used to record the model revision")
    args = ap.parse_args()

    revision, revision_source = resolve_model_revision(args.model_dir)
    if revision is None:
        print("WARNING: model revision unresolved -- responses will not be reproducibly pinned",
              file=sys.stderr, flush=True)

    if health(args.base_url) is None:
        raise SystemExit(f"Selene not reachable at {args.base_url}")

    packets = {json.loads(l)["segment_id"]: json.loads(l)
               for l in Path(args.packets).open(encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in Path(args.prompts).open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"prompts={len(rows)}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judge_responses.jsonl"

    n_ok = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(rows, 1):
            pkt = packets.get(row["segment_id"], {})
            # Per-packet schema: context_kcs_used is constrained to THIS packet's KC ids, so an
            # absent identifier is ungeneratable rather than caught afterwards.
            schema = build_wire_schema(allowed_kc_ids_for_packet(pkt))
            res = call_selene(
                row["user_prompt"] + WIRE_FORMAT_INSTRUCTION,
                system_message=row["system_prompt"], schema=schema,
                base_url=args.base_url, temperature=0.0, top_p=1.0, seed=None,
                max_tokens=args.max_tokens, schema_name="segment_judge_response",
            )
            contract = to_contract_shape(res.payload) if res.payload else None
            cerrs = validate_judge_response(
                contract, expected_segment_id=row["segment_id"],
                expected_evaluation_mode=row.get("evaluation_mode"),
                allowed_segment_text=packet_segment_text(pkt) if pkt else None,
                allowed_kc_ids=allowed_kc_ids_from_packet(pkt) if pkt else None,
            ) if contract else ["no_payload"]
            ok = contract is not None and not cerrs
            n_ok += ok
            fh.write(json.dumps({
                "judge_prompt_id": row.get("judge_prompt_id"),
                "segment_id": row["segment_id"],
                "rubric_version": row.get("rubric_version"),
                "evaluation_mode": row.get("evaluation_mode"),
                "model_path": "Selene-1-Llama-3.3-70B",
                "served_model_name": "selene-1-llama-3.3-70b",
                "model_revision": revision,
                "model_revision_source": revision_source,
                "wire_schema_version": WIRE_SCHEMA_VERSION,
                "decoding": {"temperature": 0.0, "top_p": 1.0, "greedy": True},
                "schema_enforced": True,
                "raw_response_text": json.dumps(contract, ensure_ascii=False) if contract else res.raw_text,
                "wire_response_text": res.raw_text,
                "contract_errors": cerrs,
                "json_extracted": ok,
                "schema_ok": res.schema_ok,
                "client_error": res.error,
                "generation_seconds": round(res.elapsed_s, 2),
                "usage": res.usage,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(rows)}] {row['segment_id']} contract_ok={ok} "
                  f"{res.elapsed_s:.1f}s" + ("" if ok else f" :: {cerrs[:2]}"), flush=True)

    print(f"OUT={out_path}  CONTRACT_VALID={n_ok}/{len(rows)}")


if __name__ == "__main__":
    main()
