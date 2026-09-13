"""A served-model backend with the same interface as the runners' in-process ``JudgeModel``.

WHY
---
Three judge passes (rubric, factuality, answer revelation) already talk to a vLLM server; two
(KC criteria, macro) only knew how to load weights in-process with transformers. That split was
fine when each pass was run by hand on whatever GPU was free, but it makes a single end-to-end
run impossible: a 70B model served on the job's GPUs leaves no room to also load one in-process,
so the two transformers-only passes could never run in the same reservation as the other three.

This class closes that gap without touching the passes' prompts, contracts or output rows. It
exposes exactly the methods those runners already call, so selecting it is a one-line change in
each runner and the in-process path stays byte-for-byte what it was.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not impose a JSON schema. The two runners it serves parse with their own
``extract_json_object`` and record ``json_extracted`` per row, and their downstream validators
read that. Enforcing a grammar here would change what those passes measure -- doc 32 found that
constrained decoding relocates evasion rather than removing it, so switching a pass to grammar
enforcement is a measurement decision, not a plumbing one. Making that change silently, as a side
effect of running on a server instead of locally, would be exactly the kind of drift the byte
identity guards elsewhere in this repo exist to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .selene_client_v1 import call_selene


def resolve_model_revision(model_dir: str) -> tuple[str | None, str]:
    """Read the served weights' HuggingFace commit from the download metadata.

    Identical logic to ``scripts/run_selene_rubric_judge.py``: every ``*.metadata`` file written
    by ``hf download`` carries the commit on its first line, and all of them must agree, so a
    partially-updated directory is reported rather than silently pinned to whichever file was
    read first.
    """
    if not model_dir:
        return None, "not_recorded: --model-dir not supplied"
    meta = Path(model_dir) / ".cache" / "huggingface" / "download"
    if not meta.is_dir():
        return None, f"not_recorded: no download metadata under {meta}"
    revs = set()
    for f in meta.glob("*.metadata"):
        try:
            first = f.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if first:
            revs.add(first)
    if len(revs) == 1:
        count = len(list(meta.glob("*.metadata")))
        return revs.pop(), f"read_from_hf_download_metadata ({count} files agree)"
    if not revs:
        return None, f"not_recorded: no revisions found in {meta}"
    return None, f"not_recorded: conflicting revisions in download metadata: {sorted(revs)}"


class ServedJudgeModel:
    """Drop-in replacement for a runner's in-process ``JudgeModel``.

    Both ``generate`` and ``run`` are provided because the two runners named the method
    differently; they are the same call.
    """

    def __init__(
        self,
        base_url: str,
        served_model_name: str,
        max_tokens: int = 4096,
        model_dir: str = "",
        timeout_s: int = 900,
    ) -> None:
        self.base_url = base_url
        self.served_model_name = served_model_name
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.model_path = model_dir or served_model_name
        self.resolved_revision, self.revision_source = resolve_model_revision(model_dir)
        self.last_error: str | None = None
        print(
            f"served judge base_url={base_url} model={served_model_name} "
            f"revision={self.resolved_revision} ({self.revision_source})",
            flush=True,
        )
        if self.resolved_revision is None:
            print("WARNING: model revision unresolved -- responses will not be reproducibly "
                  "pinned to a commit", flush=True)

    def generate(self, system: str, user: str) -> str:
        """Greedy generation. Returns the model's raw text, exactly as the in-process path does.

        A transport failure returns an empty string and records the error rather than raising:
        the runners write one output row per prompt and mark ``json_extracted`` false, so a failed
        call becomes a visible missing judgement instead of aborting the remaining prompts.
        """
        result = call_selene(
            user,
            system_message=system or "",
            schema=None,
            json_object=True,   # see build_payload: Selene returns prose without it
            base_url=self.base_url,
            model=self.served_model_name,
            temperature=0.0,
            top_p=1.0,
            seed=None,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
        )
        self.last_error = result.error
        if result.error:
            print(f"  call failed: {result.error}", flush=True)
            return ""
        return result.raw_text

    # The macro runner calls .run(); the criterion runner calls .generate().
    def run(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate(system_prompt, user_prompt)


def add_served_arguments(parser: Any) -> None:
    """The flags that switch a runner from in-process weights to a served endpoint.

    Kept in one place so both runners expose the same names, and so ``--model-path`` can stop
    being required without each runner inventing its own rule for when it is needed.
    """
    parser.add_argument(
        "--base-url", default="",
        help="OpenAI-compatible endpoint of a served judge. When given, weights are not loaded "
             "in-process and --model-path is not required.")
    parser.add_argument(
        "--served-model-name", default="selene-1-llama-3.3-70b",
        help="model name the server was started with (--served-model-name)")
    parser.add_argument(
        "--model-dir", default="",
        help="served weights directory, read only to record the model revision")


def build_judge(args: Any, in_process_factory, max_tokens: int):
    """Pick the backend from the parsed arguments.

    ``in_process_factory`` is called with no arguments and must return the runner's existing
    ``JudgeModel``; it is only invoked when no ``--base-url`` was given, so importing torch stays
    lazy and a served run needs no local CUDA at all.
    """
    if getattr(args, "base_url", ""):
        model = ServedJudgeModel(
            base_url=args.base_url,
            served_model_name=getattr(args, "served_model_name", "selene-1-llama-3.3-70b"),
            max_tokens=max_tokens,
            model_dir=getattr(args, "model_dir", "") or "",
        )
        return model
    if not getattr(args, "model_path", ""):
        raise SystemExit("either --model-path (in-process weights) or --base-url (served) "
                         "must be supplied")
    return in_process_factory()
