"""option_scoring.py

Model-agnostic, fully deterministic tie-breaker over an already-retrieved
candidate set.

WHY A GENERATIVE MODEL AND NOT A BETTER RERANKER
------------------------------------------------
The failure this addresses is not a relevance-estimation failure. On the
persistently-wrong exchanges, the WRONG candidate is genuinely relevant --
e.g. an exchange teaching Bayes' theorem via a medical-test example loses
to "Recall (Sensitivity)", which really is about true/false positive rates
on a test. A cross-encoder already ranks relevance well here
(bge-reranker-v2-m3 is what made v28 beat v27) and still picks wrong,
because relevance is the wrong objective. What separates the candidates is
whether the exchange is TEACHING a concept or merely MENTIONING it while
teaching something else. That is an instruction-following judgment, so a
dedicated reranker (Qwen3-Reranker, monoT5, ...) would optimize the same
objective that is already failing.

WHY OPTION-TOKEN LOGITS AND NOT FREE GENERATION
------------------------------------------------
The task is a choice over k already-retrieved candidates, so it is scored
as a multiple-choice question: build one prompt listing the options, run a
SINGLE forward pass, and compare the logits of the option letters at the
final position (the standard MMLU-style option-scoring technique). This:

  - is fully deterministic (no sampling, no temperature, no decoding loop);
  - cannot hallucinate a KC id -- the choice is constrained to the given
    options by construction, so no output parsing or retry logic is needed;
  - costs one prefill instead of an autoregressive generation, which keeps
    it cheap enough to run on CPU (no CUDA build required);
  - yields a per-option probability the caller can threshold on, rather
    than being forced to accept every verdict.

CALIBRATION WARNING -- measured, not theoretical
------------------------------------------------
The single-order probability produced here is NOT calibrated. Softmaxing
over a handful of option-letter logits saturates: measured mean top
probability is 0.988 across 60 real exchanges, and still 0.963 on the
subset whose verdict FLIPS when the options are presented in reverse
order. A single-order confidence threshold is therefore close to useless
as a gate -- it admits almost everything, including unstable verdicts.

What IS discriminative is the margin of the ORDER-AVERAGED distribution
(see pool_application.py): across the same 60 exchanges, order-stable
cases had averaged margin >= 0.651 and order-unstable cases <= 0.205, an
empty gap wide enough that any threshold inside it separates them
perfectly. Callers should gate on that averaged margin, not on the raw
per-call probability returned here.

DOMAIN AND MODEL AGNOSTICISM
-----------------------------
No KC id, course term, or library-specific string appears here. Candidates
are described only via generic frozen_matching_profile.v1 fields
(canonical_name, definition_or_summary, topic_path). Any causal LM with a
HuggingFace tokenizer can be swapped in by path; nothing below is specific
to a model family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# Option letters used for the multiple-choice framing. Chosen because each
# is a single token in every common tokenizer, which keeps the logit lookup
# unambiguous.
# A-P verified single-token (both bare and space-prefixed) for the Qwen3
# tokenizers in use. Extended past H because a measured 8.8% of gold KCs sit at
# pool rank 6-25 -- outside a top_k=5 option set the judge can never reach.
OPTION_LETTERS = [chr(ord("A") + i) for i in range(16)]

PROMPT_HEADER = (
    "You are labelling one exchange from a tutoring dialogue with the single "
    "knowledge concept it is PRIMARILY TEACHING.\n\n"
    "Important: an exchange often mentions terms belonging to other concepts "
    "while teaching something else. Worked examples, analogies, and formulas "
    "frequently borrow vocabulary from related concepts. Choose the concept "
    "the exchange is actually teaching, not one whose vocabulary merely "
    "appears in it.\n"
)


# The library ships an explicit provenance caveat in the same field as real
# contrast notes; it carries no disambiguating content and is filtered out.
BOILERPLATE_NOTE_MARKER = "use only as segmentation support"


def useful_contrast_notes(notes: list[str] | None) -> list[str]:
    return [n for n in (notes or []) if BOILERPLATE_NOTE_MARKER not in n.lower()]


@dataclass
class TieBreakCandidate:
    unit_id: str
    canonical_name: str
    definition_or_summary: str = ""
    topic_path: list[str] = field(default_factory=list)
    # negative_profile.sibling_contrast_notes -- purpose-built statements of how
    # this KC differs from its confusable siblings. Present with real content on
    # 138/159 (87%) of the library. The judge was never shown them.
    contrast_notes: list[str] = field(default_factory=list)


@dataclass
class TieBreakResult:
    chosen_unit_id: str | None
    probabilities: dict[str, float]
    margin: float
    prompt_token_count: int
    applied: bool
    reason: str


class CausalLMScorer(Protocol):
    """Anything that can score option letters for a prompt. Implemented by
    HFOptionScorer below; kept as a Protocol so a different backend (vLLM,
    llama.cpp, a remote endpoint) can be substituted without touching the
    prompt construction or the gating logic."""

    def score_options(self, prompt: str, option_letters: list[str]) -> tuple[dict[str, float], int]:
        ...


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_prompt(
    exchange_text: str,
    candidates: list[TieBreakCandidate],
    max_definition_chars: int = 400,
    max_exchange_chars: int = 1600,
    prior_exchange_text: str | None = None,
    max_prior_chars: int = 700,
    include_contrast_notes: bool = False,
    max_contrast_chars: int = 220,
    contrast_notes_unit_ids: set[str] | None = None,
) -> str:
    """`prior_exchange_text` supplies the preceding exchange as context.

    Measured effect (only visible once the scorer bug was fixed -- with the
    old 4% letter-mass scorer this signal was buried in noise and looked
    like nothing): tie-breaker pick accuracy 72.5% -> 78.4% pooled, and the
    `multi_kc` subgroup improves on BOTH dialogues (dm1 66.7%->88.9%,
    dm2 38.9%->50.0%). Many student turns are anaphoric ("how does that fix
    it?") and simply unreadable alone, which is why the preceding turn
    helps most exactly where several concepts compete.

    Note this is the JUDGE reading context, which is a different use from
    the earlier failed experiment that fed context to RETRIEVAL
    (data/gold/context_expansion_probe_report.md) -- there it added
    vocabulary noise; here it disambiguates a choice.

    `contrast_notes_unit_ids`, if given, restricts notes to only those
    unit_ids even when `include_contrast_notes=True` (v38 scoped fix, see
    data/gold/v36_contrast_notes_gold_report.md's recommendation). v36
    showed notes on every candidate destabilize the judge's forward/reverse
    agreement on some exchanges (added text length is a known source of
    MCQ position-sensitivity, arXiv:2402.14499) -- a real dm1 win offset by
    a dm2 regression, net not decisive. Restricting notes to only the
    candidates a decision is actually close between keeps the option text
    shorter for the rest of the set while still delivering the
    disambiguating content where it is needed. Default None = no
    restriction, i.e. IDENTICAL to v36 behaviour.
    """
    lines = [PROMPT_HEADER]
    if prior_exchange_text:
        lines += ["\nPrevious exchange (context only -- do NOT label this one):\n",
                  _truncate(prior_exchange_text, max_prior_chars), "\n"]
    lines += ["\nExchange to label:\n", _truncate(exchange_text, max_exchange_chars), "\n\nCandidate concepts:\n"]
    for letter, cand in zip(OPTION_LETTERS, candidates):
        topic = " > ".join(cand.topic_path) if cand.topic_path else ""
        header = f"{letter}. {cand.canonical_name}"
        if topic:
            header += f"  [{topic}]"
        lines.append(header + "\n")
        definition = _truncate(cand.definition_or_summary, max_definition_chars)
        if definition:
            lines.append(f"   {definition}\n")
        notes_allowed = contrast_notes_unit_ids is None or cand.unit_id in contrast_notes_unit_ids
        if include_contrast_notes and notes_allowed:
            for note in useful_contrast_notes(cand.contrast_notes)[:1]:
                lines.append(f"   How it differs: {_truncate(note, max_contrast_chars)}\n")
    # NB: no trailing "Answer:" here -- HFOptionScorer supplies it as an
    # assistant-turn PREFILL so the scored token position is exactly where the
    # letter goes. Putting the cue in the user turn instead made the model spend
    # its first token restating "Answer" (p=0.96), leaving only 4% of the
    # probability mass on the actual options. See the HFOptionScorer docstring.
    lines.append(
        "\nWhich concept is this exchange primarily teaching? "
        "Answer with a single letter."
    )
    return "".join(lines)


def tie_break(
    scorer: CausalLMScorer,
    exchange_text: str,
    candidates: list[TieBreakCandidate],
    min_probability: float = 0.0,
    min_margin: float = 0.0,
    max_definition_chars: int = 400,
    prior_exchange_text: str | None = None,
    include_contrast_notes: bool = False,
    contrast_notes_unit_ids: set[str] | None = None,
) -> TieBreakResult:
    """Score the candidates and return the chosen one.

    min_probability / min_margin let the caller ABSTAIN rather than accept a
    low-confidence verdict: if the winning option does not clear both bars,
    applied=False is returned and the caller should keep the existing
    ranking. Defaults are 0.0 (always apply) so that thresholds are an
    explicit, separately-justified decision rather than a silent default.
    """
    if not candidates:
        return TieBreakResult(None, {}, 0.0, 0, False, "no_candidates")
    if len(candidates) == 1:
        return TieBreakResult(candidates[0].unit_id, {candidates[0].unit_id: 1.0}, 1.0, 0, False, "single_candidate")

    usable = candidates[: len(OPTION_LETTERS)]
    prompt = build_prompt(exchange_text, usable, max_definition_chars=max_definition_chars,
                          prior_exchange_text=prior_exchange_text,
                          include_contrast_notes=include_contrast_notes,
                          contrast_notes_unit_ids=contrast_notes_unit_ids)
    letters = OPTION_LETTERS[: len(usable)]
    letter_probs, token_count = scorer.score_options(prompt, letters)

    by_unit = {cand.unit_id: letter_probs.get(letter, 0.0) for letter, cand in zip(letters, usable)}
    ordered = sorted(by_unit.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_p = ordered[0]
    runner_up_p = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_p - runner_up_p

    if top_p < min_probability or margin < min_margin:
        return TieBreakResult(top_id, by_unit, margin, token_count, False, "below_confidence_threshold")
    return TieBreakResult(top_id, by_unit, margin, token_count, True, "applied")


class HFOptionScorer:
    """HuggingFace causal-LM backend. One forward pass per call; reads the
    logits of the option letters at the final prompt position. Greedy by
    construction -- there is no sampling anywhere in this path.

    OUTPUT PREFILLING (added after a measured bug -- see below)
    -----------------------------------------------------------
    Naively scoring the first token after the chat template's generation
    prompt is a known-fragile technique. Measured on a real prompt from this
    project:

        Qwen3-4B-Instruct  argmax 'Answer' @0.96  -> only 0.0398 mass on A-E
        Qwen3-8B           argmax '<think>' @1.00 -> 0.0000 mass on A-E

    Both are the documented failure modes of first-token probability
    scoring: *misinterpretation* (the model spends the first token on a
    preamble like "Answer") and *misalignment* (it spends it on an
    unrelated token, here a reasoning-mode marker). See "Improving LLM
    First-Token Predictions in Multiple-Choice Question Answering via
    Output Prefilling" (arXiv:2505.15323) and "'My Answer is C':
    First-Token Probabilities Do Not Match Text Answers in
    Instruction-Tuned Language Models" (arXiv:2402.14499).

    The fix, per that literature, is to PREFILL the assistant turn so the
    scored position is exactly where the answer letter goes. Measured
    effect on the same prompt: letter mass 0.0398 -> 1.0000.

    Note this also flips which letter token is emitted: after a prefill
    ending in ':' the model emits ' A' (leading space), not 'A'. The
    variant is therefore selected to match the prefill, and verified
    empirically at init rather than assumed.

    Sequence-likelihood ("cloze") scoring of the candidate names was
    considered as an alternative and rejected: it reportedly aligns even
    worse with a model's actual answers than letter scoring does.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        dtype: str = "float32",
        use_chat_template: bool = True,
        assistant_prefill: str = "Answer:",
        # Default None = do NOT pass the flag. Measured, not assumed: on
        # Qwen3-8B, enable_thinking=False DESTROYS the signal that the prefill
        # recovers -- letter mass 0.9996 with the prefill alone, but 0.0000
        # with the prefill plus enable_thinking=False (the model then spends
        # its first token on markdown ' **'). The prefill alone is sufficient
        # for both a non-thinking and a hybrid-thinking model.
        enable_thinking: bool | None = None,
        min_letter_mass: float = 0.20,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # float32 by default on purpose: bf16 matmul is slow on CPUs without native
        # bf16 support, and this path is CPU-first (Sofja venv28 ships torch+cpu).
        torch_dtype = {"auto": "auto", "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.use_chat_template = use_chat_template and getattr(self.tokenizer, "chat_template", None) is not None
        self.assistant_prefill = assistant_prefill or ""
        self.enable_thinking = enable_thinking
        self.min_letter_mass = min_letter_mass
        # After a prefill ending in a non-space character the model emits the
        # letter with a leading space; with no prefill it emits the bare letter.
        self._prefer_space = bool(self.assistant_prefill) and not self.assistant_prefill.endswith(" ")
        self._letter_ids: dict[str, int] = {}
        self._mass_checked = False

    def _letter_token_id(self, letter: str) -> int:
        """Resolve the token id the model would actually emit for this option,
        choosing the spacing variant implied by the prefill. A letter that does
        not resolve to exactly one token is rejected loudly rather than
        silently mis-scored."""
        if letter in self._letter_ids:
            return self._letter_ids[letter]
        variants = (" " + letter, letter) if self._prefer_space else (letter, " " + letter)
        for variant in variants:
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                self._letter_ids[letter] = ids[0]
                return ids[0]
        raise ValueError(f"Option letter {letter!r} is not a single token for this tokenizer")

    def _render(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt + self.assistant_prefill
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], enable_thinking=self.enable_thinking, **kwargs
                ) + self.assistant_prefill
            except TypeError:
                pass  # template does not support the flag; fall through
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs) + self.assistant_prefill

    def score_options(self, prompt: str, option_letters: list[str]) -> tuple[dict[str, float], int]:
        import torch

        text = self._render(prompt)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1, :]

        ids = [self._letter_token_id(letter) for letter in option_letters]

        # Guard against the exact bug documented above: if essentially none of
        # the model's probability mass sits on the option letters, the softmax
        # below is ranking noise. Fail loudly ONCE rather than silently
        # returning a confident-looking but meaningless distribution -- silent
        # degradation has already cost this project a full wasted run.
        if not self._mass_checked:
            self._mass_checked = True
            full = torch.softmax(logits.float(), dim=-1)
            mass = float(sum(full[i].item() for i in ids))
            self.letter_mass = mass
            if mass < self.min_letter_mass:
                import warnings

                top = torch.topk(full, 3)
                offenders = [self.tokenizer.decode([i]) for i in top.indices.tolist()]
                warnings.warn(
                    f"OPTION-SCORING DEGRADED: only {mass:.4f} of probability mass is on the "
                    f"option letters; the model's top tokens are {offenders!r}. Scores from this "
                    f"model are effectively noise. Adjust assistant_prefill / enable_thinking "
                    f"for this model before trusting any result.",
                    stacklevel=2,
                )

        selected = logits[ids].float()
        probs = torch.softmax(selected, dim=-1).tolist()
        return dict(zip(option_letters, probs)), int(enc["input_ids"].shape[1])
