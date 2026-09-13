#!/usr/bin/env python3
"""Run the KC-specific criterion judge over built criterion prompts.

Same model, revision and greedy decoding as the segment judge -- this is an additional judging
pass over the same segments, not a different evaluator. Output shape mirrors
run_segment_judge.py: the model's exact text is preserved in `raw_response_text` and parsed
downstream, so a malformed response is recorded and auditable rather than coerced here.
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def strip_thinking(text: str) -> str:
    """Qwen3 emits a <think>...</think> block before its JSON; drop it before extraction."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return re.sub(r"^<think>.*?$", "", text, flags=re.S).strip()


def extract_json_object(text: str) -> tuple[dict | None, bool]:
    text = strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None, False
    depth, in_str, esc = 0, False, False
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
                try:
                    return json.loads(text[start:i + 1]), True
                except Exception:
                    return None, False
    return None, False


class JudgeModel:
    def __init__(self, model_path: str, max_new_tokens: int, revision: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, revision=revision)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.resolved_revision = revision or getattr(self.model.config, "_commit_hash", None)
        print(f"criterion judge model={model_path} revision={self.resolved_revision} "
              f"device={self.device} dtype={dtype}", flush=True)

    def generate(self, system: str, user: str) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    # No longer required: a served run supplies --base-url instead and never loads weights here.
    ap.add_argument("--model-path", default="")
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    add_served_arguments(ap)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.prompts))
    if args.limit:
        rows = rows[:args.limit]
    print(f"prompts={len(rows)}", flush=True)

    judge = build_judge(
        args,
        lambda: JudgeModel(args.model_path, args.max_new_tokens, args.model_revision),
        max_tokens=args.max_new_tokens,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kc_criterion_responses.jsonl"

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for i, row in enumerate(rows, 1):
            t0 = time.time()
            raw = judge.generate(row["system_prompt"], row["user_prompt"])
            parsed, ok = extract_json_object(raw)
            secs = round(time.time() - t0, 2)
            fh.write(json.dumps({
                "criterion_prompt_id": row["criterion_prompt_id"],
                "segment_id": row["segment_id"],
                "contract_version": row["contract_version"],
                "criterion_ids": row["criterion_ids"],
                "model_path": args.model_path or getattr(judge, "model_path", ""),
                "model_revision_requested": args.model_revision,
                "model_revision_resolved": judge.resolved_revision,
                "raw_model_text": raw,
                "raw_response_text": json.dumps(parsed, ensure_ascii=False) if ok else raw,
                "json_extracted": ok,
                "generation_seconds": secs,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(rows)}] {row['segment_id']} json={'ok' if ok else 'FAIL'} {secs}s",
                  flush=True)

    print(f"OUT={out_path}")


if __name__ == "__main__":
    main()
