"""Client for a vLLM-served Selene judge with grammar-enforced JSON output.

Adopted from the KC-library pipeline's harness
(``~/projects/selene_judge/eval_code/run_selene_judge_v3.py`` on ANTS) rather than reinvented,
so both projects call the same model the same way. See
``local_audits/evaluation_completion_20260824T093426Z/28_SELENE_JUDGE_ADOPTION_FINDINGS.md``.

THE TRAP THIS MODULE EXISTS TO AVOID
------------------------------------
``response_format`` MUST sit at the top level of the request body. It must NOT be nested under
``extra_body`` -- that is an openai-python-SDK convention for merging extra fields, and is
meaningless when building the raw HTTP body by hand, which is what this module does.

That mistake fails SILENTLY: the request succeeds, the model returns plausible markdown-fenced
free-form JSON, and the schema is simply never enforced. The KC-library pipeline lost all 61
calls of its first real run to exactly this. A regression test asserts the payload shape.

WHY GRAMMAR ENFORCEMENT REPLACES POST-HOC PARSING
-------------------------------------------------
This project previously ran every judge through ``extract_json_object`` -- strip ``<think>``
tags, find the outermost brace pair, hope it parses. Several contract failures across earlier
runs were malformed output rather than bad judgment. With ``strict: True`` the server constrains
generation to the schema, so structural failures largely stop happening rather than being caught
afterwards.

TWO-TIER VALIDITY
-----------------
The KC pipeline records a grammar/schema failure separately from a contract violation the
grammar cannot express (an identity leak, a quote that is not actually in the source text, a
verdict inconsistent with its own evidence). A single boolean conflates two different problems
with different fixes, so :class:`SeleneResult` keeps them apart.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://gpu03:8001/v1"
SERVED_MODEL_NAME = "selene-1-llama-3.3-70b"


@dataclass(frozen=True)
class SeleneResult:
    """One judge call. ``schema_ok`` and ``contract_errors`` are deliberately separate: the first
    says the model produced the requested SHAPE, the second says the content satisfies rules the
    schema cannot encode."""
    raw_text: str
    payload: dict[str, Any] | None
    schema_ok: bool
    contract_errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def fully_valid(self) -> bool:
        return self.schema_ok and not self.contract_errors and self.error is None


def build_payload(
    *, model: str, system_message: str, user_prompt: str, schema: dict[str, Any] | None,
    temperature: float, top_p: float, seed: int | None, max_tokens: int,
    schema_name: str = "judge_response", json_object: bool = False,
) -> dict[str, Any]:
    """Build the raw request body.

    ``response_format`` is placed at the TOP LEVEL. Never move it under ``extra_body`` -- see the
    module docstring; that failure is silent.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    # A seed only affects sampling. At temperature 0 generation is greedy and the seed is inert;
    # sending it anyway would imply a reproducibility guarantee the parameter is not providing.
    if seed is not None and temperature > 0:
        payload["seed"] = seed
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
    elif json_object:
        # Plain JSON mode: constrain the output to *some* valid JSON object without constraining
        # its fields. This exists for the runners that carry their own prompt-level shape contract
        # and their own parser (macro, KC-criterion) rather than a wire schema.
        #
        # It is not cosmetic. Qwen3-8B emits well-formed JSON from the prompt instruction alone, so
        # the served backend never needed it; Selene-70B returns prose and every call fails with
        # `response_not_json_despite_schema`. Nine macro prompts produced zero parsed rows before
        # this was added. Same top-level placement rule as the schema branch above.
        payload["response_format"] = {"type": "json_object"}
    return payload


def call_selene(
    user_prompt: str,
    *,
    system_message: str,
    schema: dict[str, Any] | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = SERVED_MODEL_NAME,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    max_tokens: int = 3000,
    timeout_s: int = 600,
    schema_name: str = "judge_response",
    json_object: bool = False,
    contract_validator=None,
) -> SeleneResult:
    """One judged call. ``contract_validator`` takes the parsed payload and returns a list of
    contract-error strings, so the caller's own rules stay in the caller's module."""
    payload = build_payload(
        model=model, system_message=system_message, user_prompt=user_prompt, schema=schema,
        temperature=temperature, top_p=top_p, seed=seed, max_tokens=max_tokens,
        schema_name=schema_name, json_object=json_object,
    )
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        return SeleneResult("", None, False, elapsed_s=time.time() - t0,
                             error=f"http_{exc.code}:{detail}")
    except Exception as exc:  # connection refused, timeout, malformed response
        return SeleneResult("", None, False, elapsed_s=time.time() - t0, error=repr(exc))

    elapsed = time.time() - t0
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    try:
        parsed = json.loads(text)
        schema_ok = isinstance(parsed, dict)
    except Exception:
        # With strict grammar this should not happen; when it does it is a real signal that the
        # schema was not applied, not something to paper over with fuzzy extraction.
        return SeleneResult(text, None, False, elapsed_s=elapsed, usage=usage,
                             error="response_not_json_despite_schema")

    contract_errors = list(contract_validator(parsed)) if contract_validator else []
    return SeleneResult(text, parsed, schema_ok, contract_errors, elapsed, usage)


def health(base_url: str = DEFAULT_BASE_URL, timeout_s: int = 20) -> dict[str, Any] | None:
    """GET /models -- no inference cost, safe to poll."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
