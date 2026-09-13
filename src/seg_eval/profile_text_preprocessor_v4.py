from __future__ import annotations

import re
from collections import Counter
from typing import Any


FILLER_WORDS = {
    "um", "uh", "erm", "hmm", "hm", "ah", "oh",
    "yeah", "yep", "yes", "nope", "okay", "ok", "right",
    "actually", "basically", "literally", "just", "maybe",
    "please", "thanks", "thank", "cool", "fine",
}

DISCOURSE_PHRASES = [
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "let me think",
    "wait",
    "hold on",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def remove_discourse_phrases(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    out = f" {text} "

    for phrase in DISCOURSE_PHRASES:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
        if pattern.search(out):
            removed.append(phrase)
            out = pattern.sub(" ", out)

    return normalize_whitespace(out), removed


def tokenize_light(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)?", text or "")


def preprocess_exchange_text(text: str) -> dict[str, Any]:
    raw = normalize_whitespace(text)
    without_phrases, removed_phrases = remove_discourse_phrases(raw)

    raw_tokens = tokenize_light(without_phrases)
    kept_tokens: list[str] = []
    removed_tokens: list[str] = []

    for tok in raw_tokens:
        low = tok.lower()

        # Keep technical-looking tokens, acronyms, formulas, and mixed alnum.
        technical_like = (
            any(ch.isdigit() for ch in tok)
            or tok.isupper()
            or "-" in tok
            or "_" in tok
            or "/" in tok
        )

        if low in FILLER_WORDS and not technical_like:
            removed_tokens.append(tok)
            continue

        kept_tokens.append(tok)

    match_text = " ".join(kept_tokens)

    return {
        "raw_text": raw,
        "match_text": match_text,
        "raw_token_count": len(raw_tokens),
        "match_token_count": len(kept_tokens),
        "removed_filler_tokens": dict(Counter(t.lower() for t in removed_tokens)),
        "removed_discourse_phrases": removed_phrases,
    }
