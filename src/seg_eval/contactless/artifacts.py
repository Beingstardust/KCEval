from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_report(path: str | Path, summary: dict[str, Any], segments: list[dict[str, Any]]) -> None:
    lines = ["# Contactless KC segmentation report", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## Segments",
        "",
        "| # | Turns | Exchanges | Dominant KC | Confidence | Labels | Review |",
        "|---:|---|---:|---|---|---|---|",
    ])

    for i, seg in enumerate(segments):
        labels = ",".join(sorted(set(str(x) for x in seg.get("final_labels", []))))
        bands = ",".join(sorted(set(str(x) for x in seg.get("confidence_bands", []))))
        lines.append(
            f"| {i} | {seg['turn_start']}-{seg['turn_end']} | {len(seg['member_exchange_ids'])} | "
            f"`{seg.get('dominant_kc_id')}` {seg.get('dominant_kc_name') or ''} | "
            f"{bands} | {labels} | {seg.get('review_required')} |"
        )

    Path(path).write_text("\n".join(lines), encoding="utf-8")
