"""Full-dialogue macro evaluation packet builder.

One packet per dialogue (not per segment) -- the macro rubric (adaptability, consistency,
outcome_completion, sequentiality) evaluates properties that emerge across the whole
conversation, per the thesis proposal section 6.5. Reads the same frozen `exchanges.jsonl` the
segment-level packet builder reads, so exchange text/order is identical to what local/ARC
packets are built from -- no new exchange construction, no segmentation change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .builder import read_json, read_jsonl, write_json, write_jsonl


def build_macro_packet(
    dialogue_id: str,
    exchanges: list[dict[str, Any]],
    excluded_exchange_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_exchange_ids = excluded_exchange_ids or set()
    ordered = sorted(
        (ex for ex in exchanges if str(ex.get("exchange_id")) not in excluded_exchange_ids),
        key=lambda ex: ex.get("exchange_index", 0),
    )
    packet_exchanges = [
        {
            "exchange_id": ex.get("exchange_id"),
            "exchange_index": ex.get("exchange_index"),
            "turn_start": ex.get("turn_start"),
            "turn_end": ex.get("turn_end"),
            "student_text": ex.get("student_text"),
            "tutor_text": ex.get("tutor_text"),
        }
        for ex in ordered
    ]
    return {
        "packet_schema": "dialogue_macro_packet.v1",
        "packet_id": f"macro_packet::{dialogue_id}",
        "dialogue_id": dialogue_id,
        "exchange_count": len(packet_exchanges),
        "exchanges": packet_exchanges,
    }


def build_macro_packets(run_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = read_json(run / "summary.json")
    exchanges = read_jsonl(run / "exchanges.jsonl")
    if not exchanges:
        raise FileNotFoundError(f"No exchanges.jsonl (or it was empty) under {run}")

    dialogue_id = str(
        summary.get("dialogue_id")
        or (exchanges[0].get("dialogue_id") if exchanges else "")
        or run.parent.name
    )
    excluded = set(str(x) for x in (summary.get("excluded_meta_exchange_ids") or []))

    packet = build_macro_packet(dialogue_id, exchanges, excluded)

    errors: list[str] = []
    if packet["exchange_count"] == 0:
        errors.append("empty_macro_packet_no_exchanges")
    for ex in packet["exchanges"]:
        if not ex.get("exchange_id"):
            errors.append("exchange_missing_id")
        if not (ex.get("student_text") or ex.get("tutor_text")):
            errors.append(f"exchange_missing_text:{ex.get('exchange_id')}")

    audit = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "dialogue_id": dialogue_id,
        "exchange_count": packet["exchange_count"],
        "excluded_exchange_count": len(excluded),
    }

    write_jsonl(out / "dialogue_macro_packets.jsonl", [packet])
    write_json(out / "macro_packet_audit.json", audit)
    write_json(out / "macro_packet_summary.json", {
        "packet_pipeline": "dialogue_macro_packets_v1",
        "source_run_dir": str(run),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "dialogue_id": dialogue_id,
        "exchange_count": packet["exchange_count"],
    })

    if audit["status"] != "PASS":
        raise RuntimeError("Macro packet audit failed: " + "; ".join(errors[:10]))

    return {"summary": audit, "out_dir": str(out)}
