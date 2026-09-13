from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import hashlib
import json
from pathlib import Path


TUTOR_UNDER_EVALUATION_POLICY_VERSION = "tutor_under_evaluation_policy_v1_black_box"
EVALUATOR_BACKEND_POLICY_VERSION = "evaluator_backend_policy_v1_model_agnostic"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_json(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected metadata object or None, got {type(value).__name__}")


def build_run_manifest(
    *,
    run_id: str,
    run_kind: str,
    prompt_path: str | Path,
    packet_path: str | Path,
    selected_prompt_path: str | Path,
    selected_packet_path: str | Path,
    output_dir: str | Path,
    prompt_rows: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    evaluated_tutor_metadata: dict[str, Any] | None = None,
    evaluator_backend_metadata: dict[str, Any] | None = None,
    scoring_contract_metadata: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    evaluated_tutor_metadata = normalize_json(evaluated_tutor_metadata)
    evaluator_backend_metadata = normalize_json(evaluator_backend_metadata)
    scoring_contract_metadata = normalize_json(scoring_contract_metadata)

    prompt_path = Path(prompt_path)
    packet_path = Path(packet_path)
    selected_prompt_path = Path(selected_prompt_path)
    selected_packet_path = Path(selected_packet_path)
    output_dir = Path(output_dir)

    selected_segment_ids = [str(row.get("segment_id")) for row in prompt_rows]
    selected_modes = {}
    for row in prompt_rows:
        mode = str(row.get("evaluation_mode"))
        selected_modes[mode] = selected_modes.get(mode, 0) + 1

    prompt_hashes = []
    for row in prompt_rows:
        system_prompt = str(row.get("system_prompt", ""))
        user_prompt = str(row.get("user_prompt", ""))
        prompt_hashes.append({
            "segment_id": row.get("segment_id"),
            "evaluation_mode": row.get("evaluation_mode"),
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(user_prompt),
        })

    return {
        "run_id": run_id,
        "run_kind": run_kind,
        "created_utc": utc_now_iso(),
        "policies": {
            "tutor_under_evaluation_policy": TUTOR_UNDER_EVALUATION_POLICY_VERSION,
            "evaluator_backend_policy": EVALUATOR_BACKEND_POLICY_VERSION,
            "metadata_visibility_rule": (
                "evaluated_tutor_metadata is recorded for reporting only and must not be injected "
                "into judge prompts unless a specific bias/identity experiment explicitly enables it."
            ),
        },
        "evaluated_tutor_metadata": {
            "metadata_visible_to_judge": False,
            "tutor_id": evaluated_tutor_metadata.get("tutor_id", "black_box_tutor"),
            "tutor_backend": evaluated_tutor_metadata.get("tutor_backend", "unknown"),
            "fine_tuned": evaluated_tutor_metadata.get("fine_tuned", "unknown"),
            "rag_used": evaluated_tutor_metadata.get("rag_used", "unknown"),
            "prompt_version": evaluated_tutor_metadata.get("prompt_version", "unknown"),
            "additional_metadata": evaluated_tutor_metadata.get("additional_metadata", {}),
        },
        "evaluator_backend_metadata": {
            "evaluator_status": evaluator_backend_metadata.get("evaluator_status", "not_selected"),
            "backend": evaluator_backend_metadata.get("backend", "not_selected"),
            "model_name": evaluator_backend_metadata.get("model_name", "not_selected"),
            "temperature": evaluator_backend_metadata.get("temperature", None),
            "json_mode": evaluator_backend_metadata.get("json_mode", None),
            "fine_tuned": evaluator_backend_metadata.get("fine_tuned", "not_selected"),
            "domain_fine_tuned": evaluator_backend_metadata.get("domain_fine_tuned", "not_selected"),
            "metadata_visible_to_judge": False,
            "selection_note": evaluator_backend_metadata.get(
                "selection_note",
                "No evaluator model has been selected or run in this preparation step.",
            ),
        },
        "scoring_contract_metadata": scoring_contract_metadata,
        "inputs": {
            "source_prompt_path": str(prompt_path),
            "source_packet_path": str(packet_path),
            "selected_prompt_path": str(selected_prompt_path),
            "selected_packet_path": str(selected_packet_path),
            "source_prompt_sha256": sha256_file(prompt_path) if prompt_path.exists() else None,
            "source_packet_sha256": sha256_file(packet_path) if packet_path.exists() else None,
            "selected_prompt_sha256": sha256_file(selected_prompt_path) if selected_prompt_path.exists() else None,
            "selected_packet_sha256": sha256_file(selected_packet_path) if selected_packet_path.exists() else None,
        },
        "selected_segments": {
            "count": len(prompt_rows),
            "segment_ids": selected_segment_ids,
            "mode_counts": selected_modes,
            "prompt_hashes": prompt_hashes,
        },
        "output_dir": str(output_dir),
        "notes": notes or [],
    }
