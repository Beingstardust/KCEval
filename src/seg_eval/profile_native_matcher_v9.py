"""profile_native_matcher_v9.py

v8 -> v9 changes (this session's Task 1a, closing one of the two channels
that let v8-rejected short-token matches back in):

v8 blocked short, uncorroborated identity matches from becoming identity
evidence, but did not stop the SAME evidence instance from falling through
to `support_evidence` via a pre-existing v7 mechanism: any identity-type
evidence that fails acceptance is unconditionally added to `support_score`
(`_score_profile`'s `else: support_evidence.append(ev); support_score +=
min(ev.score, 1.2)` branch, present unchanged since v7). Traced concretely
on ex_0039: the "NB" acronym match in the `canonical_terms` field is now
correctly rejected by v8 (`acronym_short_token_insufficient_corroboration`)
-- but it still adds 1.2 points to `support_score` through this fallback.

Investigated whether the SAME length+corroboration standard (v8's
`SHORT_TOKEN_LENGTH_THRESHOLD`) should also gate this fallback, using the
same corpus-wide, gold-free methodology: reran the matcher fresh across all
63 exchanges and checked dense-retrieval corroboration by shortest-matched-
token length for all 524 identity-type evidence instances that fall through
to support (`data/gold/task1a_support_fallback_audit.csv`). Unlike the
identity-acceptance path's clean, monotonic 62.5%->100% gradient, this much
larger, more heterogeneous population (mixing exact_phrase, ngram, acronym,
and rare-multi-token match kinds, each with its own pre-existing specificity
floor) shows **no clean length-based gradient** -- rates bounce
non-monotonically between 53.7% and 88.9% with no consistent break point.
Applying v8's length threshold to this whole population would not be
distributionally justified, and is NOT done here.

One narrower, still fully justified action was taken instead: evidence
instances that v8's OWN gate already rejected for insufficient corroboration
(identifiable by the reason strings v8 added:
`canonical_single_short_token_insufficient_corroboration`,
`acronym_short_token_insufficient_corroboration`,
`formula_or_symbol_short_token_insufficient_corroboration`) are now also
excluded from the support-score fallback. This is not a new, separately-
derived rule -- it is closing a backdoor around a decision the matcher
already made: if a match was judged to lack sufficient length or
corroboration to count as identity evidence, letting it back in at a lower
evidentiary bar (support) undermines the reason v8 rejected it in the first
place. All OTHER identity-type support-fallback evidence (i.e. everything
NOT carrying one of these three v8-specific reasons) is untouched, since the
distribution does not support gating it.

Also investigated (this session's Task 1b) whether the independent BM25
lexical index (`hybrid_retrieval_v2.py`, a separate module from this
matcher) needed a length-based term-weight discount, using the same
methodology: checked dense-corroboration rate by shortest-matched-BM25-term
length, both across all 1374 BM25-matched candidate rows and restricted to
the 60 candidates that are an exchange's actual resolved winner
(`data/gold/task1b_bm25_term_length_audit.csv`). Neither population shows a
usable length-based signal: the full population is flat and noisy (35-58%
corroboration at every length, 2 through 14, no gradient), and the winner-
restricted population is ~100% corroborated at every length INCLUDING
length 2-3 -- i.e. when a BM25-driven match actually wins, it already has
independent dense support essentially always, regardless of how short the
matched term is. This is the opposite of the identity-path finding, and it
means BM25 itself is not the unreliable component for "NB"/"SSE" -- **no
length-based BM25 discount was implemented**, because the evidence does not
support one. See the Task 1 report for what this implies for ex_0026/
ex_0039's final resolution.

v7 -> v8 changes (Task 2, short-token identity collision fix), retained
unchanged in v9:

Root cause, found by tracing the actual acceptance path for the documented
"NB"/"DT"/"k" collision cases, not assumed: `is_rare_token` (v7's Fix 2a
guard) only gates ONE acceptance path -- a bare single-token canonical_terms
value matched by exact phrase. It does NOT gate two other unconditional-
acceptance paths in `evidence_from_rep`:
  - `acronym_overlap` matches inside a canonical_terms phrase with >1 token
    (e.g. "NB" inside "NB Learning Phase") go straight to
    `canonical_exact_or_acronym_identity` with no rareness or length check
    at all.
  - `formula_or_symbol_forms` field matches (`exact or acronym_overlap`) go
    straight to `formula_or_symbol_identity`, also with no check.
Traced concretely: ex_0039's exchange text ("Using A=NB and B=DT...") uses
"NB"/"DT" as arbitrary classifier labels in a Z-test comparison exercise,
unrelated to Naive Bayes/Decision-Tree content; `raw_acronyms` extracts any
2+ character all-uppercase token with no document-frequency or length check,
and the acronym-overlap path above accepts the resulting match
unconditionally.

Distributional evidence (data/gold/task2_phase1_identity_evidence_audit.csv,
73 accepted identity-evidence instances across all 63 exchanges, all
144-profile pools, matcher run fresh): for each accepted match, computed the
shortest matched token's length and checked whether the SAME winning
candidate also had independent dense-retrieval corroboration (top-10% dense
rank) -- a proxy for "this identity hit reflects a genuinely central
concept," not gold. Result, by shortest-token length:

    len=2  n=8   dense-corroborated 62.5%
    len=3  n=13  dense-corroborated 84.6%
    len=4  n=14  dense-corroborated 85.7%
    len=5  n=12  dense-corroborated 100.0%
    len>=5 n=26  dense-corroborated 100.0%

A genuine break, not a gradual slope: corroboration first reaches 100% at
length 5 and stays there for every longer length observed. Also checked
whether library document frequency (the existing `is_rare_token` signal)
discriminates further within the length<=4 bucket: it does not -- multiple
same-df=21 short-token matches split roughly evenly between corroborated and
uncorroborated, so document frequency alone (v7's only check) cannot carry
this distinction; length adds real, otherwise-missing information. Breaking
down by match reason confirms the mechanism: `acronym_match` has the lowest
corroboration rate of any accepted-identity path (72.2%) and the shortest
median matched-token length (3) -- consistent with it being the path that
produced the documented "NB" collision.

Fix: `SHORT_TOKEN_LENGTH_THRESHOLD = 5` (the corpus-derived break point,
picked before checking which known exchanges it affects). Any identity
acceptance whose shortest matched token is below this length -- whether via
`canonical_terms` acronym-overlap, `canonical_terms` bare single-token, or
`formula_or_symbol_forms` exact/acronym match -- now additionally requires
at least one corroborating rare-token overlap elsewhere in the SAME match
(`_has_corroborating_rare_overlap`), reusing the existing `rare_overlap`
computation and the existing `rare_multi_token_overlap` design pattern
already present in this file for a similar purpose, rather than adding a new
cross-module dependency (e.g. calling out to dense retrieval from inside the
matcher, which would blur the source-family independence the v19 RRF
architecture relies on). Multi-token canonical/matching-cue phrases and
formula matches at or above the threshold length are unaffected -- this only
tightens the three paths shown above to have no rareness/length gate at all.

v6 -> v7 changes (Fix 2a), retained unchanged:
  - exact_phrase_match: removed the hard guard that blocked single-token phrases
    from returning True. Single-word KC canonical terms (Precision, Accuracy,
    Specificity, Entropy, Classification, etc.) could never get
    exact_canonical_match=True under v6 because of the `len(rep.tokens) < 2`
    guard. This meant the 4.0 exact_bonus was never applied for these KCs and
    any short-name KC had to rely purely on BM25/char-tfidf composite scores.
  - evidence_from_rep (canonical_terms branch): for single-token exact matches,
    require the token to be sufficiently rare (is_rare_token check) before
    accepting as identity evidence. This prevents common words like "data" or
    "model" from triggering identity evidence just because they appear in the
    exchange. Multi-token phrases keep existing behaviour unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import math
import re
from typing import Any


FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "hers", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "ours", "out", "over", "per", "she", "should", "so", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "why", "will",
    "with", "within", "without", "would", "you", "your", "yours"
}

GENERIC_PEDAGOGY_WORDS = {
    "answer", "ask", "basic", "basics", "case", "cases", "concept", "course",
    "definition", "example", "exercise", "explain", "important", "learn",
    "lesson", "method", "methods", "part", "parts", "problem", "problems",
    "question", "sheet", "step", "student", "teacher", "topic", "tutor",
    "understand"
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third", "fourth", "fifth", "single",
    "double", "triple"
}

IDENTITY_FIELDS = {
    "canonical_terms",
    "matching_cues",
    "likely_dialogue_surface_forms",
    "formula_or_symbol_forms",
}

SUPPORT_FIELDS = {
    "matched_surface_terms_from_evidence",
    "child_surface_terms",
    "definition_or_summary",
    "leaf_topic",
}

NEGATIVE_FIELDS = {
    "sibling_kc_names",
    "sibling_contrast_notes",
    "near_miss_or_gap_context",
}

FIELD_WEIGHTS = {
    "canonical_terms": 7.0,
    "matching_cues": 5.5,
    "likely_dialogue_surface_forms": 4.5,
    "formula_or_symbol_forms": 6.5,
    "matched_surface_terms_from_evidence": 3.2,
    "child_surface_terms": 1.8,
    "definition_or_summary": 0.7,
    "leaf_topic": 0.6,
    "sibling_kc_names": 5.5,
    "sibling_contrast_notes": 3.4,
    "near_miss_or_gap_context": 2.5,
}

FIELD_CAPS = {
    "canonical_terms": 8.0,
    "matching_cues": 7.0,
    "likely_dialogue_surface_forms": 5.5,
    "formula_or_symbol_forms": 7.5,
    "matched_surface_terms_from_evidence": 4.0,
    "child_surface_terms": 2.0,
    "definition_or_summary": 1.0,
    "leaf_topic": 0.8,
    "sibling_kc_names": 7.5,
    "sibling_contrast_notes": 4.5,
    "near_miss_or_gap_context": 3.0,
}


@dataclass(frozen=True)
class PhraseRep:
    field: str
    phrase: str
    phrase_norm: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    ngrams: frozenset[str]
    acronyms: frozenset[str]
    evidence_type: str
    weight: float
    cap: float


@dataclass
class ProfileRep:
    unit_id: str
    canonical_name: str
    topic_path: list[str]
    identity_core_tokens: frozenset[str]
    identity_phrases: list[PhraseRep]
    support_phrases: list[PhraseRep]
    negative_phrases: list[PhraseRep]


@dataclass
class Evidence:
    evidence_type: str
    field: str
    phrase: str
    matched_text: str
    match_kind: str
    score: float
    specificity: float
    accepted_as_identity: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateDecision:
    unit_id: str
    canonical_name: str
    topic_path: list[str]
    candidate_state: str
    score: float
    identity_score: float
    support_score: float
    negative_score: float
    exact_canonical_match: bool
    identity_evidence: list[dict[str, Any]]
    support_evidence: list[dict[str, Any]]
    negative_evidence: list[dict[str, Any]]
    matched_fields: dict[str, list[str]]
    negative_fields: dict[str, list[str]]
    score_components: dict[str, float]
    decision_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    # Split hyphenated and underscored letter-letter sequences so that
    # terms like "z-test", "k-fold", "t-test", "k-means" are treated as
    # separate tokens on both the exchange and profile sides.
    text = re.sub(r"([a-z])[-_]([a-z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str, remove_pedagogy: bool = True) -> list[str]:
    toks = [t for t in normalize_text(text).split() if len(t) >= 2]
    out = []
    for tok in toks:
        if tok in FUNCTION_WORDS or tok in NUMBER_WORDS:
            continue
        if remove_pedagogy and tok in GENERIC_PEDAGOGY_WORDS:
            continue
        out.append(tok)
    return out


def make_ngrams(tokens: list[str], min_n: int = 2, max_n: int = 4) -> set[str]:
    out: set[str] = set()
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            out.add(" ".join(tokens[i:i+n]))
    return out


def raw_acronyms(text: str) -> set[str]:
    """
    Return only real acronym/formula-like tokens.

    Accepted generically:
      - all-uppercase alphabetic tokens of length >= 2
      - formula-like alphanumeric tokens with at least one uppercase letter
        and at least one digit

    Rejected generically:
      - lowercase words
      - titlecase ordinary words
      - pure numbers
    """
    out: set[str] = set()

    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]{1,8}\b", text or ""):
        letters = "".join(ch for ch in token if ch.isalpha())
        has_digit = any(ch.isdigit() for ch in token)
        has_upper = any(ch.isupper() for ch in token)

        if not letters:
            continue

        if not has_digit and letters == letters.upper() and len(letters) >= 2:
            out.add(token.upper())
            continue

        if has_digit and has_upper:
            out.add(token.upper())

    return out


def field_values(profile: dict[str, Any], field: str) -> list[str]:
    if field == "definition_or_summary":
        value = profile.get("definition_or_summary")
    elif field == "leaf_topic":
        value = (profile.get("context_profile") or {}).get("leaf_topic")
    elif field in IDENTITY_FIELDS or field in SUPPORT_FIELDS:
        value = (profile.get("positive_profile") or {}).get(field)
    else:
        value = (profile.get("negative_profile") or {}).get(field)

    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    value = str(value).strip()
    return [value] if value else []


def exact_phrase_match(rep: PhraseRep, exchange_norm: str) -> bool:
    """
    v7 change: single-token phrases now return True when the token appears
    as a whole word in the exchange. Previously the guard `len < 2` blocked
    all single-token canonical terms from ever matching. Multi-token phrases
    keep the original exact-phrase regex behaviour.
    """
    if not rep.tokens:
        return False
    if len(rep.tokens) == 1:
        # Word-boundary match for single-token phrases.
        tok = rep.tokens[0]
        return bool(re.search(rf"(^|\s){re.escape(tok)}(\s|$)", exchange_norm))
    return bool(re.search(rf"(^| ){re.escape(rep.phrase_norm)}( |$)", exchange_norm))


def token_specificity(tok: str, token_df: Counter, n_profiles: int) -> float:
    return math.log((n_profiles + 1) / (token_df.get(tok, 0) + 1))


def phrase_specificity(phrase_norm: str, phrase_df: Counter, n_profiles: int) -> float:
    return math.log((n_profiles + 1) / (phrase_df.get(phrase_norm, 0) + 1))


def is_rare_token(tok: str, token_df: Counter, n_profiles: int) -> bool:
    df = token_df.get(tok, 0)
    return df <= max(2, int(0.03 * n_profiles)) or token_specificity(tok, token_df, n_profiles) >= math.log(4.0)


# v8 Task 2: corpus-derived break point in dense-corroboration rate by
# shortest-matched-token length (see module docstring for the full
# distributional evidence). Document frequency alone (is_rare_token) does
# not discriminate within the short-token range, so length is used as an
# additional, independent gate on top of it.
SHORT_TOKEN_LENGTH_THRESHOLD = 5


def _shortest_token_length(tokens) -> int:
    lengths = [len(t) for t in tokens if t]
    return min(lengths) if lengths else 0


def _has_corroborating_rare_overlap(matched_tokens: set[str], rare_overlap: set[str]) -> bool:
    """True when at least one rare-token match exists in this rep's evidence
    OUTSIDE the short span that is itself being evaluated for acceptance --
    i.e. the short token is not the only thing tying this candidate to the
    exchange."""
    return len(rare_overlap - matched_tokens) >= 1


# v9 Task 1a: the exact reason strings v8's short-token gate attaches when it
# rejects a match for identity. Reused here (not redefined) to close the
# support-score backdoor for those SAME rejected instances specifically.
_V8_SHORT_TOKEN_REJECTION_REASONS = {
    "canonical_single_short_token_insufficient_corroboration",
    "acronym_short_token_insufficient_corroboration",
    "formula_or_symbol_short_token_insufficient_corroboration",
}


def _rejected_for_short_token_insufficient_corroboration(ev: "Evidence") -> bool:
    return any(r in _V8_SHORT_TOKEN_REJECTION_REASONS for r in ev.reasons)


def build_phrase_rep(field: str, phrase: str, evidence_type: str) -> PhraseRep | None:
    toks = content_tokens(phrase)
    if not toks:
        return None

    phrase_norm = normalize_text(phrase)
    grams = make_ngrams(toks, 2, 4)

    if len(toks) >= 2:
        grams.add(" ".join(toks))

    return PhraseRep(
        field=field,
        phrase=str(phrase).strip(),
        phrase_norm=phrase_norm,
        tokens=tuple(toks),
        token_set=frozenset(toks),
        ngrams=frozenset(grams),
        acronyms=frozenset(raw_acronyms(phrase)),
        evidence_type=evidence_type,
        weight=FIELD_WEIGHTS[field],
        cap=FIELD_CAPS[field],
    )


def build_profile_rep(profile: dict[str, Any]) -> ProfileRep:
    identity_values = []
    identity_values.extend(field_values(profile, "canonical_terms"))
    identity_values.extend(field_values(profile, "matching_cues"))
    identity_values.extend(profile.get("source_hierarchy_path") or [])
    identity_core = frozenset(content_tokens(" ".join(identity_values)))

    identity_reps: list[PhraseRep] = []
    support_reps: list[PhraseRep] = []
    negative_reps: list[PhraseRep] = []

    for field in IDENTITY_FIELDS:
        for phrase in field_values(profile, field):
            rep = build_phrase_rep(field, phrase, "identity")
            if rep:
                identity_reps.append(rep)

    for field in SUPPORT_FIELDS:
        for phrase in field_values(profile, field):
            rep = build_phrase_rep(field, phrase, "support")
            if rep:
                support_reps.append(rep)

    for field in NEGATIVE_FIELDS:
        for phrase in field_values(profile, field):
            rep = build_phrase_rep(field, phrase, "negative")
            if rep:
                negative_reps.append(rep)

    return ProfileRep(
        unit_id=profile.get("unit_id"),
        canonical_name=profile.get("canonical_name"),
        topic_path=profile.get("topic_path") or [],
        identity_core_tokens=identity_core,
        identity_phrases=identity_reps,
        support_phrases=support_reps,
        negative_phrases=negative_reps,
    )


def build_stats(profile_reps: list[ProfileRep]) -> tuple[Counter, Counter, int]:
    token_df: Counter = Counter()
    phrase_df: Counter = Counter()

    for profile in profile_reps:
        tokens_seen: set[str] = set()
        phrases_seen: set[str] = set()

        for rep in profile.identity_phrases + profile.support_phrases + profile.negative_phrases:
            tokens_seen.update(rep.token_set)
            phrases_seen.update(rep.ngrams)
            if len(rep.tokens) >= 2:
                phrases_seen.add(rep.phrase_norm)

        token_df.update(tokens_seen)
        phrase_df.update(phrases_seen)

    return token_df, phrase_df, max(len(profile_reps), 1)


def best_ngram_match(rep: PhraseRep, exchange_ngrams: set[str], phrase_df: Counter, n_profiles: int) -> tuple[str | None, float]:
    overlaps = []
    for gram in rep.ngrams:
        if gram in exchange_ngrams:
            overlaps.append((gram, len(gram.split()), phrase_specificity(gram, phrase_df, n_profiles)))

    if not overlaps:
        return None, 0.0

    overlaps.sort(key=lambda x: (x[1], x[2]), reverse=True)
    gram, _, spec = overlaps[0]
    return gram, spec


def evidence_from_rep(
    profile: ProfileRep,
    rep: PhraseRep,
    exchange_norm: str,
    exchange_tokens: set[str],
    exchange_ngrams: set[str],
    exchange_acronyms: set[str],
    token_df: Counter,
    phrase_df: Counter,
    n_profiles: int,
) -> Evidence | None:
    exact = exact_phrase_match(rep, exchange_norm)
    acronym_overlap = rep.acronyms & exchange_acronyms
    ngram, ngram_spec = best_ngram_match(rep, exchange_ngrams, phrase_df, n_profiles)
    overlap = rep.token_set & exchange_tokens
    rare_overlap = {t for t in overlap if is_rare_token(t, token_df, n_profiles)}
    core_overlap = overlap & profile.identity_core_tokens

    match_kind = ""
    matched_text = ""
    specificity = 0.0
    multiplier = 0.0
    reasons: list[str] = []

    if exact:
        match_kind = "exact_phrase"
        matched_text = rep.phrase_norm
        specificity = phrase_specificity(rep.phrase_norm, phrase_df, n_profiles)
        multiplier = 1.0
        reasons.append("exact_phrase_match")
    elif acronym_overlap:
        match_kind = "acronym"
        matched_text = ",".join(sorted(acronym_overlap))
        specificity = max(token_specificity(a.lower(), token_df, n_profiles) for a in acronym_overlap)
        multiplier = 0.85
        reasons.append("acronym_match")
    elif ngram:
        match_kind = "ngram"
        matched_text = ngram
        specificity = ngram_spec
        multiplier = 0.55
        reasons.append("ngram_match")
    elif len(rare_overlap) >= 2:
        match_kind = "rare_token_set"
        matched_text = ",".join(sorted(rare_overlap))
        specificity = sum(token_specificity(t, token_df, n_profiles) for t in rare_overlap) / max(len(rare_overlap), 1)
        multiplier = 0.42
        reasons.append("rare_multi_token_overlap")
    else:
        return None

    accepted_identity = False

    if rep.evidence_type == "identity":
        matched_tokens = set(content_tokens(matched_text))
        matched_core = matched_tokens & profile.identity_core_tokens
        matched_token_count = len(matched_tokens)

        if rep.field == "canonical_terms":
            if exact or acronym_overlap:
                if len(rep.tokens) == 1:
                    # v7 Fix 2a: single-token canonical terms require rareness guard
                    # to prevent common words ("data", "model", "error") from
                    # triggering identity evidence.
                    # v8 Task 2: also require length >= SHORT_TOKEN_LENGTH_THRESHOLD,
                    # or, for shorter tokens, an independent corroborating rare-token
                    # overlap -- document frequency alone was shown not to
                    # discriminate reliably within the short-token range.
                    tok = rep.tokens[0]
                    long_enough = len(tok) >= SHORT_TOKEN_LENGTH_THRESHOLD
                    corroborated = long_enough or _has_corroborating_rare_overlap({tok}, rare_overlap)
                    if is_rare_token(tok, token_df, n_profiles) and corroborated:
                        accepted_identity = True
                        reasons.append("canonical_single_rare_token_identity")
                    elif is_rare_token(tok, token_df, n_profiles):
                        reasons.append("canonical_single_short_token_insufficient_corroboration")
                    else:
                        reasons.append("canonical_single_token_too_common_for_identity")
                elif acronym_overlap and not exact:
                    # v8 Task 2: an acronym match embedded in a multi-token
                    # canonical phrase (e.g. "NB" inside "NB Learning Phase")
                    # previously bypassed any rareness/length check entirely.
                    shortest = _shortest_token_length(a.lower() for a in acronym_overlap)
                    matched_lower = {a.lower() for a in acronym_overlap}
                    if shortest >= SHORT_TOKEN_LENGTH_THRESHOLD or _has_corroborating_rare_overlap(matched_lower, rare_overlap):
                        accepted_identity = True
                        reasons.append("canonical_exact_or_acronym_identity")
                    else:
                        reasons.append("acronym_short_token_insufficient_corroboration")
                else:
                    accepted_identity = True
                    reasons.append("canonical_exact_or_acronym_identity")
            else:
                reasons.append("partial_canonical_match_not_identity")

        elif rep.field == "formula_or_symbol_forms":
            if exact or acronym_overlap:
                # v8 Task 2: exact/acronym formula-or-symbol matches previously
                # bypassed any rareness/length check entirely -- the same gate
                # applied to canonical_terms acronym matches now applies here.
                span = rep.tokens if exact else list(acronym_overlap)
                shortest = _shortest_token_length(t.lower() for t in span)
                matched_lower = {t.lower() for t in span}
                if shortest >= SHORT_TOKEN_LENGTH_THRESHOLD or _has_corroborating_rare_overlap(matched_lower, rare_overlap):
                    accepted_identity = True
                    reasons.append("formula_or_symbol_identity")
                else:
                    reasons.append("formula_or_symbol_short_token_insufficient_corroboration")
            elif len(rare_overlap) >= 2:
                accepted_identity = True
                reasons.append("formula_or_symbol_identity")
            else:
                reasons.append("formula_match_not_specific_enough")

        elif rep.field in {"matching_cues", "likely_dialogue_surface_forms"}:
            if exact and len(rep.tokens) >= 2 and specificity >= math.log(6.0):
                accepted_identity = True
                reasons.append("exact_specific_profile_identity_phrase")

            elif (
                match_kind == "ngram"
                and matched_token_count >= 3
                and specificity >= math.log(10.0)
                and (len(matched_core) >= 2 or len(rare_overlap) >= 3)
            ):
                accepted_identity = True
                reasons.append("long_specific_identity_ngram")

            elif (
                match_kind == "rare_token_set"
                and len(rare_overlap) >= 3
                and specificity >= math.log(8.0)
                and len(matched_core) >= 2
            ):
                accepted_identity = True
                reasons.append("large_specific_identity_token_set")

            else:
                reasons.append("identity_field_match_treated_as_support_due_to_low_specificity")

        else:
            reasons.append("not_identity_field")

    score = rep.weight * multiplier * max(0.60, min(1.35, specificity / max(math.log(4.0), 1.0)))

    return Evidence(
        evidence_type=rep.evidence_type,
        field=rep.field,
        phrase=rep.phrase,
        matched_text=matched_text,
        match_kind=match_kind,
        score=round(score, 6),
        specificity=round(specificity, 6),
        accepted_as_identity=accepted_identity,
        reasons=reasons,
    )


def saturate(evidence: list[Evidence], cap: float, max_items: int = 2) -> tuple[float, list[Evidence]]:
    best_by_key: dict[tuple[str, str], Evidence] = {}

    for ev in evidence:
        key = (ev.match_kind, ev.matched_text)
        old = best_by_key.get(key)
        if old is None or ev.score > old.score:
            best_by_key[key] = ev

    selected = sorted(best_by_key.values(), key=lambda e: e.score, reverse=True)[:max_items]
    return min(cap, sum(e.score for e in selected)), selected


class ProfileNativeMatcherV9:
    def __init__(self, profiles: list[dict[str, Any]]):
        self.profile_reps = [build_profile_rep(p) for p in profiles]
        self.token_df, self.phrase_df, self.n_profiles = build_stats(self.profile_reps)

    def match_exchange(self, exchange_text: str, top_k: int = 10) -> list[CandidateDecision]:
        exchange_norm = normalize_text(exchange_text)
        exchange_token_list = content_tokens(exchange_text)
        exchange_tokens = set(exchange_token_list)
        exchange_ngrams = make_ngrams(exchange_token_list, 2, 4)
        exchange_acronyms = raw_acronyms(exchange_text)

        decisions = [
            self._score_profile(rep, exchange_norm, exchange_tokens, exchange_ngrams, exchange_acronyms)
            for rep in self.profile_reps
        ]

        decisions.sort(
            key=lambda d: (
                d.candidate_state == "candidate",
                d.exact_canonical_match,
                d.score,
                d.identity_score,
                d.support_score,
            ),
            reverse=True,
        )
        return decisions[:top_k]

    def _score_profile(
        self,
        profile: ProfileRep,
        exchange_norm: str,
        exchange_tokens: set[str],
        exchange_ngrams: set[str],
        exchange_acronyms: set[str],
    ) -> CandidateDecision:
        raw_identity_evidence: list[Evidence] = []
        support_evidence: list[Evidence] = []
        negative_evidence: list[Evidence] = []

        identity_score = 0.0
        support_score = 0.0
        negative_score = 0.0

        for field in IDENTITY_FIELDS:
            reps = [r for r in profile.identity_phrases if r.field == field]
            field_evidence = [
                ev for r in reps
                if (ev := evidence_from_rep(
                    profile, r, exchange_norm, exchange_tokens,
                    exchange_ngrams, exchange_acronyms,
                    self.token_df, self.phrase_df, self.n_profiles,
                ))
            ]

            field_score, selected = saturate(field_evidence, FIELD_CAPS[field], max_items=2)
            for ev in selected:
                if ev.accepted_as_identity:
                    raw_identity_evidence.append(ev)
                    identity_score += ev.score
                elif _rejected_for_short_token_insufficient_corroboration(ev):
                    # v9 Task 1a: don't let a match v8's own gate already
                    # judged too short/uncorroborated for identity back in
                    # through the support fallback. Still recorded in
                    # support_evidence for audit, but contributes no score.
                    support_evidence.append(ev)
                else:
                    support_evidence.append(ev)
                    support_score += min(ev.score, 1.2)

        for field in SUPPORT_FIELDS:
            reps = [r for r in profile.support_phrases if r.field == field]
            field_evidence = [
                ev for r in reps
                if (ev := evidence_from_rep(
                    profile, r, exchange_norm, exchange_tokens,
                    exchange_ngrams, exchange_acronyms,
                    self.token_df, self.phrase_df, self.n_profiles,
                ))
            ]
            field_score, selected = saturate(field_evidence, FIELD_CAPS[field], max_items=2)
            support_score += field_score
            support_evidence.extend(selected)

        for field in NEGATIVE_FIELDS:
            reps = [r for r in profile.negative_phrases if r.field == field]
            field_evidence = [
                ev for r in reps
                if (ev := evidence_from_rep(
                    profile, r, exchange_norm, exchange_tokens,
                    exchange_ngrams, exchange_acronyms,
                    self.token_df, self.phrase_df, self.n_profiles,
                ))
            ]
            field_score, selected = saturate(field_evidence, FIELD_CAPS[field], max_items=1)
            negative_score += field_score
            negative_evidence.extend(selected)

        exact_canonical = any(
            ev.field == "canonical_terms" and ev.match_kind in {"exact_phrase", "acronym"}
            for ev in raw_identity_evidence
        )

        if exact_canonical:
            identity_score += 4.0
            negative_score *= 0.35

        support_score_capped = min(4.0, support_score)
        net_score = identity_score + support_score_capped - negative_score

        matched_fields: dict[str, list[str]] = defaultdict(list)
        for ev in raw_identity_evidence + support_evidence:
            matched_fields[ev.field].append(ev.phrase)

        negative_fields: dict[str, list[str]] = defaultdict(list)
        for ev in negative_evidence:
            negative_fields[ev.field].append(ev.phrase)

        candidate_state = "reject"
        reasons: list[str] = []

        if not raw_identity_evidence and not support_evidence:
            reasons.append("no_profile_evidence")
        elif not raw_identity_evidence:
            if support_score >= 3.0:
                candidate_state = "support_only"
                reasons.append("support_evidence_without_identity")
            else:
                reasons.append("no_accepted_identity_evidence")
        elif identity_score < 3.0 and not exact_canonical:
            reasons.append("identity_score_below_floor")
        elif net_score < 3.0:
            reasons.append("net_score_below_floor")
        elif negative_score >= (identity_score + support_score_capped) * 0.70 and not exact_canonical:
            reasons.append("negative_evidence_too_high")
        else:
            candidate_state = "candidate"
            reasons.append("profile_native_identity_gate_passed")

        return CandidateDecision(
            unit_id=profile.unit_id,
            canonical_name=profile.canonical_name,
            topic_path=profile.topic_path,
            candidate_state=candidate_state,
            score=round(net_score, 6),
            identity_score=round(identity_score, 6),
            support_score=round(support_score, 6),
            negative_score=round(negative_score, 6),
            exact_canonical_match=exact_canonical,
            identity_evidence=[e.to_dict() for e in raw_identity_evidence],
            support_evidence=[e.to_dict() for e in support_evidence],
            negative_evidence=[e.to_dict() for e in negative_evidence],
            matched_fields={k: sorted(set(v)) for k, v in matched_fields.items()},
            negative_fields={k: sorted(set(v)) for k, v in negative_fields.items()},
            score_components={
                "identity_score": round(identity_score, 6),
                "support_score_raw": round(support_score, 6),
                "support_score_capped": round(support_score_capped, 6),
                "negative_score": round(negative_score, 6),
                "net_score": round(net_score, 6),
                "accepted_identity_count": len(raw_identity_evidence),
            },
            decision_reasons=reasons,
        )
