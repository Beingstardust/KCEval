from __future__ import annotations

import re
from typing import Any
from .schema import Turn
from .io_utils import as_text


STUDENT_ALIASES = {"student", "learner", "user", "human"}
TUTOR_ALIASES = {"tutor", "assistant", "teacher", "ai", "model"}


def normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in STUDENT_ALIASES:
        return "student"
    if r in TUTOR_ALIASES:
        return "tutor"
    return r or "unknown"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_noise(text: str) -> list[str]:
    t = normalize_text(text).lower()
    flags: list[str] = []
    if not t:
        return ["empty_or_near_empty"]

    short_ack = {
        "ok", "okay", "yes", "yeah", "yep", "no", "nope", "thanks",
        "thank you", "got it", "cool", "fine", "alright", "right"
    }
    if t in short_ack:
        flags.append("acknowledgement_only")
    if t in {"hi", "hello", "hey", "good morning", "good evening"}:
        flags.append("greeting_only")
    if len(t.split()) <= 2 and not any(ch.isdigit() for ch in t) and "?" not in t:
        flags.append("low_content_short")
    return flags


def _extract_turns(dialogue_obj: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    dialogue_id = str(
        dialogue_obj.get("dialogue_id")
        or dialogue_obj.get("id")
        or dialogue_obj.get("conversation_id")
        or "dlg_000"
    )

    turns = (
        dialogue_obj.get("turns")
        or dialogue_obj.get("messages")
        or dialogue_obj.get("dialogue")
        or dialogue_obj.get("conversation")
    )
    if not isinstance(turns, list):
        raise ValueError(f"Dialogue {dialogue_id} has no list-valued turns/messages/dialogue/conversation field")
    return dialogue_id, turns


def parse_dialogue(dialogue_obj: dict[str, Any]) -> list[Turn]:
    dialogue_id, raw_turns = _extract_turns(dialogue_obj)
    parsed: list[Turn] = []

    for idx, raw in enumerate(raw_turns):
        if not isinstance(raw, dict):
            raise ValueError(f"Dialogue {dialogue_id} turn {idx} is not an object")
        role = normalize_role(str(raw.get("role") or raw.get("speaker") or raw.get("from") or raw.get("author") or "unknown"))
        text = as_text(raw.get("content") or raw.get("text") or raw.get("message") or raw.get("utterance") or "")
        norm = normalize_text(text)
        parsed.append(
            Turn(
                dialogue_id=dialogue_id,
                turn_id=idx,
                role=role,
                raw_text=text,
                norm_text=norm,
                noise_flags=detect_noise(norm),
            )
        )
    return parsed
