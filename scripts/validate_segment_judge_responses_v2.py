from __future__ import annotations

import argparse
import json
from pathlib import Path

from seg_eval.evaluation_judge.response_contract_v2 import (
    RESPONSE_JSON_SCHEMA,
    read_jsonl,
    validate_response_rows,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate segment judge JSON responses against the V2 scale 0, 0.5, 1 dependency-aware judge response contract.")
    parser.add_argument("--prompts", required=True, help="Path to segment_judge_prompts.jsonl")
    parser.add_argument("--packets", required=True, help="Path to segment_evaluation_packets.jsonl")
    parser.add_argument("--responses", required=True, help="JSONL of judge outputs or wrapped judge responses")
    parser.add_argument("--out", required=True, help="Output directory for validation artifacts")
    args = parser.parse_args()

    prompt_rows = read_jsonl(args.prompts)
    packet_rows = read_jsonl(args.packets)
    response_rows = read_jsonl(args.responses)

    valid_rows, error_rows, summary = validate_response_rows(
        prompt_rows=prompt_rows,
        packet_rows=packet_rows,
        response_rows=response_rows,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    write_json(out / "judge_response_validation_summary.json", summary)
    write_json(out / "judge_response_schema_v2_scale001_dependency_aware.json", RESPONSE_JSON_SCHEMA)
    write_jsonl(out / "valid_judge_responses.jsonl", valid_rows)
    write_jsonl(out / "judge_response_validation_errors.jsonl", error_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
