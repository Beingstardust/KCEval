from __future__ import annotations

import argparse
import json

from seg_eval.evaluation_judge.prompt_builder import build_judge_prompt_packets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build judge prompt packets from segment evaluation packets.")
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profiles", default=None)
    args = parser.parse_args()

    summary = build_judge_prompt_packets(
        packet_dir=args.packet_dir,
        out_dir=args.out,
        profiles_path=args.profiles,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"OUT_DIR={args.out}")


if __name__ == "__main__":
    main()
