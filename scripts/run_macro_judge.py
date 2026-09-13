#!/usr/bin/env python3
"""Run the full-dialogue macro judge over built macro judge prompts.

Deliberately a standalone script, not an import from run_segment_judge.py, even though the core
mechanics (think-tag stripping, JSON extraction, greedy generation) are identical in spirit --
run_segment_judge.py is the proven 52/52 runner and this task's instructions are explicit about
not modifying the judge model or decoding configuration it uses. Duplicating ~150 lines here
keeps that file at zero risk of an accidental import-time side effect, rather than saving the
duplication at the cost of coupling two runners that serve different, independently-versioned
schemas (macro_rubric_v1 vs rubric_v2).

Same determinism discipline as run_segment_judge.py: greedy decoding, no sampling, pinned model
revision recorded in the output summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seg_eval.evaluation_judge.served_backend_v1 import (  # noqa: E402
    add_served_arguments, build_judge,
)

THINK_CLOSE = "</think>"

OUTPUT_INSTRUCTION = """

OUTPUT REQUIREMENTS (read carefully):
- Return ONE JSON object and nothing else. No prose, no markdown fences.
- Return an INSTANCE, not the schema. Do NOT echo `scale` or any field description.
- Top-level keys required: dialogue_id, dimension_scores, dimension_applicability,
  evidence_pointers, rationale_short. No other top-level keys.
- Score VALUES must be JSON numbers 0, 0.5 or 1, or null -- never strings.
- Set a dimension to null ONLY when it truly does not apply to this dialogue, and mark it
  "not_applicable" in dimension_applicability. This is a hard coupling: whenever
  dimension_applicability[d] == "not_applicable", dimension_scores[d] MUST be null, never a
  number. A non-null score paired with not_applicable is a contract error.
- Every dimension marked "applicable" needs at least one evidence_pointers entry with a real
  exchange_id from the dialogue and a verbatim quote copied from that exchange -- never a
  paraphrase, never an invented quote.
- Each quote must be an EXACT contiguous substring already present in that exchange's text --
  same words, same order, no ellipsis ("..."), no omission, no merging two separated spans into
  one quote. If a point needs two spans, use two separate evidence_pointers entries instead.
- rationale_short must be the LAST field in the object.
"""


def strip_thinking(text: str) -> str:
    idx = text.find(THINK_CLOSE)
    if idx == -1:
        return text
    return text[idx + len(THINK_CLOSE):]


def extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidate = fenced.group(1)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def normalize_payload(obj: dict, prompt_row: dict) -> dict:
    """Structural repair only -- coerce numeric strings, fill dialogue_id from the prompt. Never
    invents or alters a judgment. Mirrors run_segment_judge.py's normalize_payload's discipline,
    scoped to the macro schema's actual fields."""
    out = dict(obj)
    for junk in ("scale", "expected_output_schema", "dimensions"):
        out.pop(junk, None)

    def coerce(v):
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("null", "none", ""):
                return None
            try:
                f = float(t)
                return int(f) if f in (0.0, 1.0) else f
            except ValueError:
                return v
        return v

    if isinstance(out.get("dimension_scores"), dict):
        out["dimension_scores"] = {k: coerce(v) for k, v in out["dimension_scores"].items()}

    if not out.get("dialogue_id") and prompt_row.get("dialogue_id"):
        out["dialogue_id"] = prompt_row["dialogue_id"]

    return out


class JudgeModel:
    def __init__(self, model_path: str, max_new_tokens: int = 4096, revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, revision=revision)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.resolved_revision = revision or getattr(self.model.config, "_commit_hash", None)
        print(f"macro judge model={model_path} revision={self.resolved_revision} "
              f"device={self.device} dtype={dtype}", flush=True)

    def run(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the macro dialogue judge over built macro prompts.")
    ap.add_argument("--prompts", required=True, help="dialogue_macro_judge_prompts.jsonl")
    # No longer required: a served run supplies --base-url instead and never loads weights here.
    ap.add_argument("--model-path", default="")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    add_served_arguments(ap)
    args = ap.parse_args()

    prompts = read_jsonl(Path(args.prompts))
    print(f"prompts={len(prompts)}", flush=True)

    judge = build_judge(
        args,
        lambda: JudgeModel(args.model_path, max_new_tokens=args.max_new_tokens,
                           revision=args.model_revision),
        max_tokens=args.max_new_tokens,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "macro_judge_responses.jsonl"

    n_parsed = 0
    started = time.time()
    with open(responses_path, "w", encoding="utf-8") as fh:
        for i, p in enumerate(prompts, start=1):
            t0 = time.time()
            raw = judge.run(p.get("system_prompt") or "",
                            (p.get("user_prompt") or "") + OUTPUT_INSTRUCTION)
            obj = extract_json_object(strip_thinking(raw))
            payload = None
            if obj:
                n_parsed += 1
                payload = normalize_payload(json.loads(obj), p)
            truncated = obj is None or not raw.rstrip().endswith(("}", "```"))
            row = {
                "likely_truncated": bool(truncated),
                "judge_prompt_id": p.get("judge_prompt_id"),
                "dialogue_id": p.get("dialogue_id"),
                "rubric_version": p.get("rubric_version"),
                "model_path": args.model_path or getattr(judge, "model_path", ""),
                "model_revision_requested": args.model_revision,
                "model_revision_resolved": judge.resolved_revision,
                "raw_response_text": json.dumps(payload, ensure_ascii=False) if payload else raw,
                "raw_model_text": raw,
                "json_extracted": bool(obj),
                "generation_seconds": round(time.time() - t0, 2),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(prompts)}] {p.get('dialogue_id')} "
                  f"json={'ok' if obj else 'FAILED'} {row['generation_seconds']}s", flush=True)

    elapsed = time.time() - started
    summary = {
        "prompts": len(prompts),
        "json_extracted": n_parsed,
        "json_failed": len(prompts) - n_parsed,
        "model_path": args.model_path or getattr(judge, "model_path", ""),
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": judge.resolved_revision,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": round(elapsed, 1),
    }
    (out_dir / "macro_judge_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMACRO_JUDGE_RUN_COMPLETE  parsed={n_parsed}/{len(prompts)}  {elapsed:.0f}s")
    print(f"OUT={responses_path}")


if __name__ == "__main__":
    main()
