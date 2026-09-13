from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .rubric_v2 import DEPENDENCY_RULES, MICRO_DIMENSIONS as DIMENSIONS, MODE_POLICIES, OUTPUT_SCHEMA, PARTIAL_SCALE, RUBRIC_VERSION, TRUST_FLAG_SCALE
from .kc_grounding_conditions import (
    DEFAULT_CONDITION,
    condition_adds_curriculum_grounding,
    condition_shows_kc_context,
    grounding_block_for_kc,
)

# Policy text used in place of MODE_POLICIES when no KC context is supplied at all
# (grounding condition A). The stock mode policies all instruct the judge to evaluate against
# the primary KC or evaluator_kc_set, which is incoherent when those are deliberately empty --
# so A substitutes this. This is the ONE unavoidable prompt-text difference between A and B
# beyond the KC content itself, and it is recorded in the ablation report rather than hidden.
UNGROUNDED_MODE_POLICY = (
    "No curriculum knowledge-component reference is supplied for this segment. Judge the "
    "tutor's pedagogical quality from the segment text alone. Do not speculate about which "
    "curriculum unit this segment belongs to."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_profiles(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        unit_id = row.get("unit_id") or row.get("kc_id") or row.get("id")
        if unit_id:
            profiles[str(unit_id)] = row
    return profiles


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def evaluation_target(packet: dict[str, Any]) -> dict[str, Any]:
    return _dict(packet.get("evaluation_target"))


def segmentation_metadata(packet: dict[str, Any]) -> dict[str, Any]:
    return _dict(packet.get("segmentation_metadata"))


def get_first(packet: dict[str, Any], names: list[str], default: Any = None) -> Any:
    """
    Read direct fields first, then the known nested packet sections.

    V1 originally only read direct fields. Segment-evaluation packets store the
    evaluator mode under evaluation_target and scope/role under segmentation_metadata.
    """
    nested = [packet, evaluation_target(packet), segmentation_metadata(packet)]

    for source in nested:
        for name in names:
            if name in source and source[name] not in (None, "", []):
                return source[name]

    return default


def segment_id(packet: dict[str, Any]) -> str:
    return str(get_first(packet, ["segment_id", "id", "packet_id"], "UNKNOWN_SEGMENT"))


def evaluation_mode(packet: dict[str, Any]) -> str:
    mode = get_first(packet, ["evaluation_mode", "segment_evaluation_mode", "mode"], None)
    if mode:
        return str(mode)

    scope = get_first(packet, ["segment_scope", "assignment_scope"], None)
    dominant_role = get_first(packet, ["dominant_kc_role", "primary_kc_role"], None)

    if scope == "single_kc":
        return "single_kc_primary"
    if scope == "branch_context":
        return "branch_context_with_anchor"
    if scope == "broad_kc_set" or dominant_role == "low_confidence_anchor":
        return "broad_kc_set_low_confidence_anchor"

    return "branch_context_with_anchor"


def kc_ids(packet: dict[str, Any]) -> list[str]:
    ids = get_first(packet, ["evaluator_kc_set", "kc_set", "context_kc_ids"], [])
    if isinstance(ids, list) and ids:
        return [str(x) for x in ids if x]

    context_rows = packet.get("kc_context")
    if isinstance(context_rows, list) and context_rows:
        out: list[str] = []
        for row in context_rows:
            if isinstance(row, dict):
                unit_id = row.get("unit_id") or row.get("kc_id") or row.get("id")
                if unit_id and str(unit_id) not in out:
                    out.append(str(unit_id))
        if out:
            return out

    resolved = primary_kc_id(packet)
    return [str(resolved)] if resolved else []


def primary_kc_id(packet: dict[str, Any]) -> str | None:
    value = get_first(packet, ["primary_kc_id", "resolved_kc_id", "dominant_kc_id"], None)
    return str(value) if value else None


def packet_kc_context(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = packet.get("kc_context")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                unit_id = row.get("unit_id") or row.get("kc_id") or row.get("id")
                if unit_id:
                    out[str(unit_id)] = row
    return out


def profile_summary(
    unit_id: str,
    profiles: dict[str, dict[str, Any]],
    packet_context: dict[str, dict[str, Any]] | None = None,
    grounding_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render one KC for the judge's prompt.

    ``grounding_index`` is used only by KC-grounding condition C; when it is None (every other
    caller, including the frozen baseline path) this function's output is unchanged, so
    condition B stays byte-identical to the previously validated baseline prompts.
    """
    row = (packet_context or {}).get(unit_id) or profiles.get(unit_id, {})
    summary = {
        "unit_id": unit_id,
        "canonical_name": row.get("canonical_name") or row.get("name") or row.get("label"),
        "topic_path": row.get("topic_path") or row.get("source_hierarchy_path") or [],
        "definition_or_summary": row.get("definition_or_summary") or row.get("definition") or row.get("summary"),
        "kc_specific_criteria": row.get("kc_specific_criteria") or [],
    }
    if grounding_index is not None:
        block = grounding_block_for_kc(unit_id, grounding_index)
        if block:
            summary["curriculum_grounding"] = block
    return summary


def exchange_lines(packet: dict[str, Any]) -> list[str]:
    for key in ["member_exchanges", "exchanges", "exchange_units"]:
        rows = packet.get(key)
        if isinstance(rows, list) and rows:
            lines: list[str] = []
            for i, ex in enumerate(rows):
                if isinstance(ex, dict):
                    exid = ex.get("exchange_id") or ex.get("id") or f"exchange_{i}"
                    student = ex.get("student_text") or ex.get("student") or ex.get("student_turn_text")
                    tutor = ex.get("tutor_text") or ex.get("tutor") or ex.get("tutor_turn_text")
                    text = ex.get("exchange_text") or ex.get("text")
                    if student or tutor:
                        lines.append(f"[{exid}] Student: {student or ''}\n[{exid}] Tutor: {tutor or ''}")
                    elif text:
                        lines.append(f"[{exid}] {text}")
                    else:
                        lines.append(f"[{exid}] {json.dumps(ex, ensure_ascii=False)}")
                else:
                    lines.append(str(ex))
            return lines

    for key in ["segment_text", "text", "exchange_text"]:
        text = packet.get(key)
        if text:
            return [str(text)]

    return [json.dumps(packet, ensure_ascii=False, sort_keys=True)[:8000]]


def _confidence_from_metadata(packet: dict[str, Any]) -> dict[str, Any]:
    meta = segmentation_metadata(packet)
    labels = meta.get("final_labels")
    bands = meta.get("confidence_bands")
    actions = meta.get("resolution_actions")

    return {
        "final_labels": labels if isinstance(labels, list) else [],
        "confidence_bands": bands if isinstance(bands, list) else [],
        "resolution_actions": actions if isinstance(actions, list) else [],
    }


def _segmentation_context(packet: dict[str, Any], show_kc: bool = True) -> dict[str, Any]:
    """
    Preserve V9E teaching-style and context-resolution metadata for the judge.

    This is not a new segmentation decision. It only exposes already-existing packet
    metadata so context-dependent teaching moves, such as analogy, anaphora, ellipsis,
    prerequisite recall, recap, and pass2 bridge rescue, are not treated as noise.
    """
    meta = segmentation_metadata(packet)

    exchange_contexts: list[dict[str, Any]] = []
    member_exchanges = packet.get("member_exchanges")
    if isinstance(member_exchanges, list):
        for item in member_exchanges:
            if not isinstance(item, dict):
                continue

            assignment = item.get("assignment")
            if not isinstance(assignment, dict):
                assignment = {}

            context_dependency = assignment.get("context_dependency")
            if not isinstance(context_dependency, dict):
                context_dependency = {}

            # Under grounding condition A the per-exchange resolved KC identity is withheld
            # along with the rest of the KC context. Without this the "ungrounded" condition
            # is not actually ungrounded: resolved_kc_id/resolved_kc_name name the curriculum
            # unit for every exchange, which leaked a KC identifier into 35/35 dm1-local
            # prompts when this was first built. Everything else in this block (confidence
            # bands, context-dependency flags, resolution actions) is segmentation metadata
            # rather than curriculum grounding, so it stays in all conditions -- the ablation
            # isolates curriculum grounding, not segmentation metadata.
            exchange_contexts.append({
                "exchange_id": item.get("exchange_id"),
                "final_label": assignment.get("final_label"),
                "confidence_band": assignment.get("confidence_band"),
                "assignment_scope": assignment.get("assignment_scope"),
                "resolved_kc_id": assignment.get("resolved_kc_id") if show_kc else None,
                "resolved_kc_name": assignment.get("resolved_kc_name") if show_kc else None,
                "dominant_kc_role": assignment.get("dominant_kc_role"),
                "resolution_action": assignment.get("resolution_action"),
                "context_dependency": {
                    "is_context_dependent": bool(context_dependency.get("is_context_dependent", False)),
                    "flags": context_dependency.get("flags") if isinstance(context_dependency.get("flags"), list) else [],
                    "anaphora_hits": context_dependency.get("anaphora_hits") if isinstance(context_dependency.get("anaphora_hits"), list) else [],
                    "ellipsis_hits": context_dependency.get("ellipsis_hits") if isinstance(context_dependency.get("ellipsis_hits"), list) else [],
                    "analogy_hits": context_dependency.get("analogy_hits") if isinstance(context_dependency.get("analogy_hits"), list) else [],
                    "shift_hits": context_dependency.get("shift_hits") if isinstance(context_dependency.get("shift_hits"), list) else [],
                },
                "context_dependency_anchor_handling": assignment.get("context_dependency_anchor_handling"),
            })

    return {
        "purpose": (
            "Use this block to interpret segmentation uncertainty and context-dependent teaching style. "
            "Do not treat analogy, anaphora, ellipsis, recap, prerequisite recall, or pass2 bridge rescue as noise by default."
        ),
        "segment_scope": meta.get("segment_scope"),
        "dominant_kc_role": meta.get("dominant_kc_role"),
        "contains_context_dependent_rescue": bool(meta.get("contains_context_dependent_rescue", False)),
        "contains_branch_context": bool(meta.get("contains_branch_context", False)),
        "contains_broad_kc_set": bool(meta.get("contains_broad_kc_set", False)),
        "resolution_actions": meta.get("resolution_actions") if isinstance(meta.get("resolution_actions"), list) else [],
        "member_exchange_contexts": exchange_contexts,
    }


def compact_packet_context(
    packet: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    condition: str = DEFAULT_CONDITION,
    grounding_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the judge's prompt context.

    ``condition`` defaults to B_baseline and ``grounding_index`` to None, so every pre-existing
    caller gets byte-identical output to the previously validated baseline. Only the
    KC-grounding ablation passes anything else.
    """
    sid = segment_id(packet)
    mode = evaluation_mode(packet)
    p_kc = primary_kc_id(packet)
    kcs = kc_ids(packet)
    packet_profiles = packet_kc_context(packet)

    show_kc = condition_shows_kc_context(condition)
    grounding = grounding_index if condition_adds_curriculum_grounding(condition) else None

    return {
        "segment_id": sid,
        "evaluation_mode": mode,
        "mode_policy": (
            MODE_POLICIES.get(mode, MODE_POLICIES["branch_context_with_anchor"])
            if show_kc else UNGROUNDED_MODE_POLICY
        ),
        "segment_scope": get_first(packet, ["segment_scope", "assignment_scope"], None),
        "dominant_kc_role": get_first(packet, ["dominant_kc_role", "primary_kc_role"], None),
        "primary_kc_id": p_kc if show_kc else None,
        "primary_kc": (
            profile_summary(p_kc, profiles, packet_profiles, grounding) if (p_kc and show_kc) else None
        ),
        "evaluator_kc_set": (
            [profile_summary(kc, profiles, packet_profiles, grounding) for kc in kcs] if show_kc else []
        ),
        "confidence": _confidence_from_metadata(packet),
        "segmentation_context": _segmentation_context(packet, show_kc),
        "segment_exchanges": exchange_lines(packet),
        # focus_target is deliberately NOT included here. A bounded prompt-minimization
        # experiment (local_audits/evaluation_completion_20260824T093426Z/
        # 14_FOCUS_GUIDANCE_MINIMIZATION.md) found that exposing focus_target in the judge's
        # prompt context measurably destabilizes response-contract validity (52/52 -> 44-47/52)
        # regardless of how much or how little natural-language guidance accompanied it -- even
        # the bare structural field with zero guidance text cost 5/52. Per that finding,
        # focus_target stays deterministic packet/reference data for CODE use only (see
        # evaluation_judge/focus_grounding_audit.py, which reads it straight from the packet
        # rather than from anything the judge is shown) -- never shown to the judge.
        "packet_trace_fields": {
            "member_exchange_ids": get_first(packet, ["member_exchange_ids"], []),
            "contains_broad_kc_set": bool(get_first(packet, ["contains_broad_kc_set"], False)),
            "contains_branch_context": bool(get_first(packet, ["contains_branch_context"], False)),
            "missing_kc_review": bool(get_first(packet, ["missing_kc_review"], False)),
        },
    }


def system_prompt() -> str:
    return (
        "You are an evidence-grounded tutor-evaluation judge. "
        "Evaluate only the supplied segment packet. Do not use outside knowledge except general mathematical/pedagogical reasoning. "
        "Return only valid JSON matching the requested schema. Do not include hidden chain-of-thought; give concise evidence quotes and a short rationale."
    )


# A bounded prompt-minimization experiment tried exposing focus_target in this prompt, in four
# forms (a standalone 8-bullet section, and three variants integrated into the trust_flags
# bullet below: no guidance text, a moderate clause, a minimal clause). All four measurably
# destabilized response-contract validity relative to this unparameterized baseline (52/52 ->
# 40-47/52), regardless of how much or how little was said -- even zero guidance text cost 5/52
# just from focus_target's structural presence in context. See
# local_audits/evaluation_completion_20260824T093426Z/14_FOCUS_GUIDANCE_MINIMIZATION.md for the
# full account; every variant's prompts/responses/results are preserved there and on disk.
# Conclusion: focus_target stays out of the judge's prompt entirely. It remains available as
# deterministic packet/reference data for CODE to use directly (see
# evaluation_judge/focus_grounding_audit.py, which reads it from the packet, never from
# anything the judge is shown) -- this function is intentionally back to its pre-experiment,
# unparameterized form.
def user_prompt(context: dict[str, Any]) -> str:
    rubric = {
        "rubric_version": RUBRIC_VERSION,
        "partial_scale": PARTIAL_SCALE,
        "trust_flag_scale": TRUST_FLAG_SCALE,
        "dimensions": DIMENSIONS,
        "dependency_rules": DEPENDENCY_RULES,
        "required_output_schema": OUTPUT_SCHEMA,
    }

    return (
        "Evaluate the following student-tutor segment.\n\n"
        "Important segmentation semantics:\n"
        f"- evaluation_mode: {context['evaluation_mode']}\n"
        f"- mode_policy: {context['mode_policy']}\n"
        "- If dominant_kc_role is low_confidence_anchor, do not treat the primary KC as exact truth; use the evaluator_kc_set.\n"
        "- If evaluation_mode is single_kc_primary, judge mainly against the primary KC.\n"
        "- Use segmentation_context to interpret context-dependent teaching moves such as analogy, anaphora, ellipsis, recap, prerequisite recall, and pass2 bridge rescue.\n"
        "- Do not mark a segment as bad only because it required context rescue; judge whether the supplied segment and KC set are sufficient for fair evaluation.\n"
        "- Score partial-credit dimensions using only 0, 0.5, 1, or null when not applicable.\n"
        "- Use 0 = poor/absent/harmful, 0.5 = partial/mixed/weak, 1 = good/clearly present.\n"
        "- Score trust_flags as 0 or 1 only, where 1 means a material hallucination is present.\n"
        "- First produce raw_dimension_scores independently for each partial-credit dimension.\n"
        "- Then apply only justified dependency_rules and put final scores in dimension_scores.\n"
        "- Compute micro_score_raw, micro_score_dependency_adjusted, and trust_adjusted_score between 0.0 and 1.0.\n"
        "- Do not let trustworthiness failures average away; reflect them in trust_adjusted_score and flags.\n"
        "- Quote only from the supplied segment text when giving evidence.\n\n"
        "Rubric and output schema:\n"
        f"{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        "Segment packet:\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}"
    )


def build_judge_prompt_packets(
    packet_dir: str | Path,
    out_dir: str | Path,
    profiles_path: str | Path | None = None,
    condition: str = DEFAULT_CONDITION,
    grounding_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``condition``/``grounding_index`` default to the frozen baseline behavior; only the
    KC-grounding ablation passes anything else."""
    packet_dir = Path(packet_dir)
    out_dir = Path(out_dir)
    profiles_path = Path(profiles_path) if profiles_path else None

    packet_path = packet_dir / "segment_evaluation_packets.jsonl"
    if not packet_path.exists():
        raise FileNotFoundError(f"Missing segment_evaluation_packets.jsonl: {packet_path}")

    profiles = load_profiles(profiles_path)
    packets = read_jsonl(packet_path)

    prompt_rows: list[dict[str, Any]] = []
    pipeline_id = "segment_judge_prompts_v1b"

    for packet in packets:
        context = compact_packet_context(packet, profiles, condition, grounding_index)
        prompt_row = {
            "judge_prompt_id": f"{context['segment_id']}::judge_prompt_v1b",
            "segment_id": context["segment_id"],
            "rubric_version": RUBRIC_VERSION,
            "evaluation_mode": context["evaluation_mode"],
            "system_prompt": system_prompt(),
            "user_prompt": user_prompt(context),
            "expected_output_schema": OUTPUT_SCHEMA,
            "source_packet_sha256": stable_hash(packet),
            "prompt_context_sha256": stable_hash(context),
        }
        prompt_rows.append(prompt_row)

    mode_counts: dict[str, int] = {}
    errors: list[str] = []

    for row in prompt_rows:
        mode_counts[row["evaluation_mode"]] = mode_counts.get(row["evaluation_mode"], 0) + 1

        try:
            context_json = row["user_prompt"].split("Segment packet:\n", 1)[1]
            context = json.loads(context_json)
        except Exception as exc:
            errors.append(f"bad_context_json:{row['segment_id']}:{exc!r}")
            continue

        # Condition A deliberately supplies no KC context, so an empty evaluator_kc_set / absent
        # primary_kc is the intended state there, not an audit failure. These two checks stay
        # fully active for B and C.
        if condition_shows_kc_context(condition):
            if not context.get("evaluator_kc_set"):
                errors.append(f"empty_evaluator_kc_set:{row['segment_id']}")

            if context.get("evaluation_mode") == "single_kc_primary" and not context.get("primary_kc"):
                errors.append(f"single_kc_missing_primary:{row['segment_id']}")
        else:
            if context.get("evaluator_kc_set") or context.get("primary_kc"):
                errors.append(f"ungrounded_condition_leaked_kc_context:{row['segment_id']}")
            # Structural emptiness is not sufficient -- KC identity also reaches the prompt
            # through nested metadata (segmentation_context.member_exchange_contexts carries
            # resolved_kc_id/resolved_kc_name). Assert on the FULL rendered prompt text so any
            # future field that starts carrying a KC id fails the build instead of silently
            # contaminating the ungrounded condition.
            if "KC_" in row["user_prompt"]:
                errors.append(f"ungrounded_condition_kc_identifier_in_prompt_text:{row['segment_id']}")

        if context.get("evaluation_mode") == "broad_kc_set_low_confidence_anchor":
            if context.get("dominant_kc_role") != "low_confidence_anchor":
                errors.append(f"broad_mode_missing_low_confidence_anchor:{row['segment_id']}")

    packet_summary_path = packet_dir / "packet_summary.json"
    if packet_summary_path.exists():
        try:
            packet_summary = json.loads(packet_summary_path.read_text(encoding="utf-8"))
            expected_counts = packet_summary.get("evaluation_mode_counts")
            if isinstance(expected_counts, dict) and expected_counts != mode_counts:
                errors.append(f"mode_count_mismatch:expected={expected_counts}:actual={mode_counts}")
        except Exception as exc:
            errors.append(f"could_not_read_packet_summary:{exc!r}")

    summary = {
        "prompt_pipeline": pipeline_id,
        "rubric_version": RUBRIC_VERSION,
        "packet_dir": str(packet_dir),
        "profiles_path": str(profiles_path) if profiles_path else None,
        "source_packet_count": len(packets),
        "judge_prompt_count": len(prompt_rows),
        "evaluation_mode_counts": mode_counts,
        "kc_grounding_condition": condition,
    }

    audit = dict(summary)
    audit["status"] = "PASS" if not errors else "FAIL"
    audit["errors"] = errors

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "segment_judge_prompts.jsonl", prompt_rows)
    write_json(out_dir / "judge_prompt_summary.json", summary)
    write_json(out_dir / "judge_prompt_audit.json", audit)
    write_json(out_dir / "rubric_v1.json", {
        "rubric_version": RUBRIC_VERSION,
        "partial_scale": PARTIAL_SCALE,
        "trust_flag_scale": TRUST_FLAG_SCALE,
        "dimensions": DIMENSIONS,
        "mode_policies": MODE_POLICIES,
        "output_schema": OUTPUT_SCHEMA,
    })
    write_json(out_dir / "judge_prompt_manifest.json", {
        "prompt_pipeline": pipeline_id,
        "packet_dir": str(packet_dir),
        "profiles_path": str(profiles_path) if profiles_path else None,
        "outputs": [
            "segment_judge_prompts.jsonl",
            "judge_prompt_summary.json",
            "judge_prompt_audit.json",
            "rubric_v1.json",
            "judge_prompt_manifest.json",
        ],
    })

    if errors:
        raise RuntimeError("Judge prompt audit failed: " + "; ".join(errors[:10]))

    return summary
