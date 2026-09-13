from __future__ import annotations

import argparse
import json
import sys

from seg_eval.evaluation_judge.macro_response_contract_v1 import validate_macro_response

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def extract_payload(row: dict) -> dict | None:
    if isinstance(row.get("raw_response_text"), str):
        try:
            return json.loads(row["raw_response_text"])
        except Exception:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate macro judge responses against macro_response_contract_v1.")
    ap.add_argument("--packets", required=True, help="dialogue_macro_packets.jsonl")
    ap.add_argument("--responses", required=True, help="macro_judge_responses.jsonl")
    args = ap.parse_args()

    packets = {p["dialogue_id"]: p for p in read_jsonl(args.packets)}
    responses = read_jsonl(args.responses)

    valid = []
    errors = []
    for row in responses:
        payload = extract_payload(row)
        dialogue_id = row.get("dialogue_id")
        if payload is None:
            errors.append({"dialogue_id": dialogue_id, "errors": ["could_not_parse_response_json"]})
            continue
        packet = packets.get(dialogue_id)
        exchange_text_by_id = None
        if packet:
            exchange_text_by_id = {
                ex["exchange_id"]: f"{ex.get('student_text') or ''} {ex.get('tutor_text') or ''}"
                for ex in packet.get("exchanges", [])
            }
        err = validate_macro_response(
            payload, expected_dialogue_id=dialogue_id, exchange_text_by_id=exchange_text_by_id,
        )
        if err:
            errors.append({"dialogue_id": dialogue_id, "errors": err})
        else:
            valid.append({"dialogue_id": dialogue_id, "macro_output": payload})

    summary = {
        "response_count": len(responses),
        "valid_count": len(valid),
        "error_count": len(errors),
        "status": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(summary, indent=2))
    for e in errors:
        print(" ", e["dialogue_id"], e["errors"])

    if valid:
        print()
        print("=== valid macro outputs ===")
        for v in valid:
            print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
