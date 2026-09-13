from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows


def load_matching_profiles(path: str | Path) -> list[dict[str, Any]]:
    profiles = read_jsonl(path)

    seen: set[str] = set()
    for profile in profiles:
        unit_id = profile.get("unit_id")
        if not unit_id:
            raise ValueError(f"Profile missing unit_id in {path}")
        if unit_id in seen:
            raise ValueError(f"Duplicate profile unit_id {unit_id} in {path}")
        seen.add(unit_id)

        if profile.get("unit_type") != "kc":
            raise ValueError(f"Profile {unit_id} is not a KC profile")

    return profiles
