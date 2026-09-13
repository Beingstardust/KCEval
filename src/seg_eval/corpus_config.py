"""Domain/corpus configuration — the one place a domain-specific input path is resolved.

WHY THIS EXISTS
---------------
This is a *framework* for evaluating tutors, not a data-mining-specific tool. The evaluation
machinery (rubric dimensions, focus-target patterns, aggregation, factuality pass) is
domain-generic by construction: `rubric_v2`'s dimensions are pedagogical constructs
(error_detection, scaffolding_quality, clarity_cognitive_load), `macro_rubric_v1`'s are dialogue
properties (adaptability, consistency, sequentiality), and `focus_target_v1`'s cue regexes match
tutoring-dialogue markers ("oh wait", "actually I", "exactly") rather than any subject matter.

What IS domain-specific is the **input**: the knowledge-component library. Three modules had that
library's path hardcoded as a module-level constant, which meant retargeting the framework to a
different subject would require editing source files. That is the one thing this module fixes.

USAGE
-----
Point the framework at a different domain by setting one environment variable::

    SEG_EVAL_KC_LIBRARY=data/input/my_other_domain/reviewed_library.jsonl

or by calling :func:`set_kc_library_path` before building prompts. Nothing else needs to change:
the library's own schema (kc_id, canonical_name, evaluation_support, reviewer_criteria) is the
contract, and any domain that supplies that schema works.

KNOWN NON-DOMAIN LIMITATION
---------------------------
The cue regexes in ``focus_target_v1`` are English-language. That is a *language* dependency, not
a domain one -- they carry no subject matter -- but a non-English corpus would need them
localised. Recorded here so it is not mistaken for domain coupling.
"""

from __future__ import annotations

import os
from pathlib import Path

# The library this project was developed against. A DEFAULT, not a requirement -- overridable by
# environment variable or set_kc_library_path(). Kept repo-relative so it resolves the same way
# regardless of working directory.
DEFAULT_KC_LIBRARY = (
    "data/input/frozen_library_incoming/kc_library_reviewed_v2/2026-08-10_155459/"
    "frozen_reviewed_library.jsonl"
)

KC_LIBRARY_ENV_VAR = "SEG_EVAL_KC_LIBRARY"

_override: str | None = None


def set_kc_library_path(path: str | Path | None) -> None:
    """Point the framework at a different domain's KC library for this process."""
    global _override
    _override = str(path) if path is not None else None


def kc_library_path() -> str:
    """Resolution order: explicit override -> environment variable -> repo default."""
    if _override:
        return _override
    return os.environ.get(KC_LIBRARY_ENV_VAR) or DEFAULT_KC_LIBRARY


def resolve_kc_library(repo_root: str | Path) -> Path:
    """Absolute path to the active KC library. Relative paths resolve against ``repo_root`` so a
    caller can supply either a repo-relative or an absolute location."""
    p = Path(kc_library_path())
    return p if p.is_absolute() else Path(repo_root) / p
