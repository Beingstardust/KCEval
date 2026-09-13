"""Run the r4 grounded-factuality pass through Selene, on the existing prompts unchanged.

WHY THIS RUN EXISTS
-------------------
The factuality verdict feeds the deterministic trust cap, and the cap is a hard ``min()``: one
false "contradicted + material" permanently caps a segment at 0.5. That makes a false positive far
more damaging than a miss, so *precision* is the property that matters here, not recall.

The Qwen3-14B r4 run charges tv_a -- the real tutor, with no injected errors -- with 9 contradicted
segments and 11 material contradictions, against tv_c's 18 material for 15 deliberately injected
errors. A signal-to-noise ratio of 1.64:1 means the false-positive floor is consuming most of the
discrimination the cap is supposed to provide. Selene measured 0.000-0.083 false positives against
Qwen-14B's 0.250 on the same probe, which is why this arm is worth running.

The grammar mirrors the r4 contract: every flagged error MUST carry a verbatim tutor quote, a
statement of what is wrong, and the correction. "If you cannot state the correction, it is not an
error" is the mechanism that suppressed unreasoned pattern-matching in r1/r2, and here it is
enforced structurally rather than only asked for in prose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.selene_client_v1 import call_selene, health  # noqa: E402
from seg_eval.evaluation_judge.grounded_factuality_v1 import (  # noqa: E402
    ALLOWED_SEVERITY, PROMPT_REVISION, derive_verdict_from_errors,
)


def build_schema() -> dict:
    """Every field the r4 contract requires is required by the grammar too.

    ``errors`` is an array that may legitimately be empty -- that is what "grounded" means -- but
    an entry that exists cannot omit its justification.
    """
    error_obj = {
        "type": "object",
        "properties": {
            "tutor_quote": {"type": "string", "minLength": 8},
            "what_is_wrong": {"type": "string", "minLength": 4},
            "correct_version": {"type": "string", "minLength": 4},
            "severity": {"type": "string", "enum": sorted(ALLOWED_SEVERITY)},
        },
        "required": ["tutor_quote", "what_is_wrong", "correct_version", "severity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "errors": {"type": "array", "items": error_obj},
        },
        "required": ["segment_id", "errors"],
        "additionalProperties": False,
    }


def resolve_model_revision(model_dir: str) -> tuple[str | None, str]:
    """Same run-time provenance capture as the rubric runner; all metadata must agree."""
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
        return revs.pop(), "read_from_hf_download_metadata"
    if not revs:
        return None, f"not_recorded: no revisions found in {meta}"
    return None, f"not_recorded: conflicting revisions: {sorted(revs)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--model-dir", default="")
    args = ap.parse_args()

    if health(args.base_url) is None:
        raise SystemExit(f"Selene not reachable at {args.base_url}")

    revision, revision_source = resolve_model_revision(args.model_dir)
    rows = [json.loads(l) for l in Path(args.prompts).open(encoding="utf-8") if l.strip()]
    print(f"prompts={len(rows)} revision={revision}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grounded_factuality_responses.jsonl"
    schema = build_schema()

    counts = {"grounded": 0, "minor_deviation": 0, "contradicted": 0, "insufficient_reference": 0}
    material_total = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(rows, 1):
            res = call_selene(
                row["user_prompt"], system_message=row["system_prompt"], schema=schema,
                base_url=args.base_url, temperature=0.0, top_p=1.0, seed=None,
                max_tokens=args.max_tokens, schema_name="grounded_factuality_response",
            )
            payload = res.payload
            verdict, material = ("insufficient_reference", 0)
            if payload:
                verdict, material = derive_verdict_from_errors(payload)
            counts[verdict] = counts.get(verdict, 0) + 1
            material_total += material
            fh.write(json.dumps({
                "factuality_prompt_id": row.get("factuality_prompt_id"),
                "segment_id": row["segment_id"],
                "contract_version": row.get("contract_version"),
                "prompt_revision": row.get("prompt_revision", PROMPT_REVISION),
                "reference_kc_ids": row.get("reference_kc_ids"),
                "model_path": "Selene-1-Llama-3.3-70B",
                "served_model_name": "selene-1-llama-3.3-70b",
                "model_revision_resolved": revision,
                "model_revision_source": revision_source,
                "decoding": {"temperature": 0.0, "top_p": 1.0, "greedy": True},
                "schema_enforced": True,
                "raw_response_text": json.dumps(payload, ensure_ascii=False) if payload else res.raw_text,
                "json_extracted": payload is not None,
                "schema_ok": res.schema_ok,
                "derived_verdict": verdict,
                "material_contradictions": material,
                "client_error": res.error,
                "generation_seconds": round(res.elapsed_s, 2),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(rows)}] {row['segment_id']} {verdict} material={material} "
                  f"{res.elapsed_s:.1f}s", flush=True)

    print(f"OUT={out_path}")
    print(f"VERDICTS={counts} MATERIAL_TOTAL={material_total}")


if __name__ == "__main__":
    main()
