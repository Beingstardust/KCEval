from __future__ import annotations

import argparse
import json

from seg_eval.evaluation_judge.macro_prompt_builder_v1 import build_macro_judge_prompt_packets


def main() -> None:
    ap = argparse.ArgumentParser(description="Build macro judge prompts from macro evaluation packets.")
    ap.add_argument("--packet-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = build_macro_judge_prompt_packets(packet_dir=args.packet_dir, out_dir=args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"OUT_DIR={args.out}")


if __name__ == "__main__":
    main()
