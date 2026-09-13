"""Build T4_macro_degraded: a tutor degraded ONLY at dialogue level. NO MODEL CALLS.

Spec and pre-registered predictions:
``docs/evaluation_completeness_v2/27_MACRO_DEGRADATION_PREREGISTRATION.md`` -- frozen before this
text was authored.

WHY THIS CONDITION EXISTS
-------------------------
``consistency``, ``outcome_completion`` and ``sequentiality`` have scored exactly 1.0 on all seven
dialogues judged so far. Nothing has ever exercised them, so the framework's claim that tutoring
quality needs three scopes -- segment, topic arc, whole dialogue -- has never been tested against
the alternative that the macro layer is a redundant re-read of segment-level evidence.

T4 separates the two. It is a MINIMAL PAIR with T1 (dm1's real tutor) over dm1's identical student
turns: 52 of 63 tutor turns are byte-identical, and the 11 modified ones each stay locally
correct, complete and pedagogically sound when read on their own.

THE DESIGN CONSTRAINT THAT SHAPED EVERY REPLACEMENT
---------------------------------------------------
A content-level self-contradiction necessarily makes one of the two statements curriculum-false,
which the factuality pass catches and the trust cap charges -- contaminating the contrast with
exactly the segment-scope movement the experiment predicts is absent.

So every consistency violation here is a **self-misattribution about the dialogue**: the tutor
misreports what it or the student said earlier, while its curriculum content stays correct. A
claim about what was previously said has no curriculum truth value, so a grounding verifier cannot
see it and only reading across turns can.

The same rule governs the sequentiality violations: each answer is factually correct and is simply
delivered before the concepts it relies on have been introduced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data/input/dialogues/inbox/dm1_human_student_llm_tutor_teaching_rich_dialogue.json"
OUT_DIR = REPO_ROOT / "data/input/dialogues/inbox"
GROUND_TRUTH = REPO_ROOT / "data/gold/macro_degraded_variant_ground_truth.json"

# ---------------------------------------------------------------------------
# CONSISTENCY -- the tutor misreports the dialogue's own history. Curriculum content unchanged.
# ---------------------------------------------------------------------------
CONSISTENCY: dict[int, tuple[str, str]] = {
    17: (
        "Yes, treating income as a discrete value for this exercise calculation, which is the "
        "route you chose when I offered you the two options a moment ago. DB=N score = 7/10 * "
        "5/9 * 5/10 * 1/17, about 0.0114. DB=Y score = 3/10 * 4/5 * 1/6 * 2/13, about 0.00615. "
        "Since 0.0114 is larger, classify X as DB=N.",
        "Claims it already offered the discretize-vs-Gaussian choice and that the student picked "
        "one. That offer is not made until idx 18, one turn later. Arithmetic unchanged.",
    ),
    24: (
        "Yes. Graduation gives meaningful groups: Middle School is pure class 1, High School is "
        "mostly class 0, and College is mixed. Its Gain Ratio is about 0.436, while ID's Gain "
        "Ratio is about 0.231. So Gain Ratio pushes us away from ID-like memorization. This is "
        "the same conclusion you reached yourself back in Exercise 2, so you are staying "
        "consistent.",
        "Attributes the conclusion to the student in Exercise 2. Exercise 2 was Bayes and Naive "
        "Bayes; the student reached no such conclusion. Curriculum content unchanged.",
    ),
    34: (
        "F1 is 2PR/(P+R). With precision 0.5 and recall 0.667, F1 is about 0.571. Do not compute "
        "F1 from accuracy. Accuracy is (TP+TN)/total = 7/10 = 0.7, which is a separate metric. "
        "You had precision and recall the right way round when you first raised them, so this "
        "follows directly.",
        "At idx 32 the student said 'I mix up precision and recall' and the tutor had to define "
        "both. Claiming the student had them right contradicts the dialogue two turns earlier.",
    ),
    51: (
        "Using C1=(2.625,7.000) and C2=(4.071,1.786), the assignments stay the same. Since "
        "assignments and centroids no longer change, K-Means terminates. The final SSE is about "
        "42.830. That is the stopping rule I gave you before we started the worked example, "
        "rather than the fixed iteration count I mentioned then.",
        "The tutor never mentioned a fixed iteration count, and gave only one stopping rule. "
        "Invents a prior version of its own guidance. Numbers unchanged.",
    ),
    59: (
        "Correct. You separated the concepts cleanly: classification predicts a target attribute "
        "from labeled training data; K-Means groups unlabeled data by distance to centroids. "
        "That is a change from Exercise 6, where you told me clustering was a supervised method.",
        "The student never said clustering was supervised; at idx 43 the student asked and the "
        "tutor answered no. Misreports the student's earlier position.",
    ),
}

# ---------------------------------------------------------------------------
# SEQUENTIALITY -- correct answers delivered before their prerequisites are introduced.
# ---------------------------------------------------------------------------
SEQUENTIALITY: dict[int, tuple[str, str]] = {
    12: (
        "Yes. Count the target labels first. There are 3 DB=Y rows and 7 DB=N rows out of 10. So "
        "P(DB=Y)=3/10 and P(DB=N)=7/10. These are priors because they are known before looking "
        "at the new instance attributes. You will multiply each prior by the Laplace-smoothed "
        "conditionals, (count + 1)/(class_count + number_of_possible_values), when we score the "
        "instance.",
        "Uses Laplace smoothing and its formula at idx 12. The zero-frequency problem is not "
        "raised until idx 14 and smoothing is not defined until idx 16.",
    ),
    19: (
        "That is the problem. Information Gain can prefer attributes with many distinct values. "
        "ID is the obvious example: splitting by ID creates one-row pure leaves, so the weighted "
        "child entropy drops to 0 and the gain is maximal. Once you normalize by the split "
        "information, though, ID's ratio falls to about 0.231, which is why Gain Ratio is the "
        "criterion to trust here. But ID usually does not generalize to future examples.",
        "Relies on entropy, weighted child entropy, split information and Gain Ratio. Gain Ratio "
        "is not introduced until idx 21, entropy not until idx 22, weighting not until idx 23.",
    ),
    30: (
        "Because TP, FP, TN, and FN depend on the positive class. Here TP means true Y predicted "
        "Y, FN means true Y predicted N, FP means true N predicted Y, and TN means true N "
        "predicted N. That choice then propagates: specificity is TN/(TN+FP) and F1 is "
        "2PR/(P+R), so both flip meaning if you swap the positive class. If you use N as "
        "positive, every metric changes.",
        "Uses specificity and F1 at idx 30. Precision and recall are not defined until idx 32, "
        "sensitivity and specificity until idx 33, F1 until idx 34.",
    ),
    45: (
        "Choose K initial centroids; assign each point to the nearest centroid; recompute each "
        "centroid as the mean of assigned points; repeat assignment and recomputation until "
        "assignments or centroids stop changing. The objective is to reduce SSE, the sum of "
        "squared distances from each point to its assigned centroid. In the bisecting variant "
        "this same loop runs with K=2 inside each split, which is the version you will want for "
        "Exercise 6's later parts.",
        "Introduces bisecting K-Means at idx 45. It is not raised until idx 55, and the standard "
        "algorithm's worked example (idx 47-51) has not run yet.",
    ),
}

# ---------------------------------------------------------------------------
# OUTCOME COMPLETION -- both synthesis turns break off unresolved.
#
# PRE-REGISTERED LEAK (H-M4): trailing off is also locally poor actionability, so these two units
# are expected to cost segment-scope score. Predicted contribution to the segment gap: <= 0.03.
# ---------------------------------------------------------------------------
COMPLETION: dict[int, tuple[str, str]] = {
    57: (
        "Yes. Exercise 1 introduces supervised classification and data roles. Exercise 2 gives "
        "probabilistic classification with Bayes and Naive Bayes. Exercise 3 studies "
        "decision-tree split criteria and pruning estimates. Exercise 4 evaluates classifiers "
        "and compares models. Exercise 6 moves to unsupervised clustering, where distances and",
        "The synthesis breaks off mid-sentence and the connection across exercises is never "
        "completed.",
    ),
    62: (
        "Use four passes. First identify the task type: classification, probability, decision "
        "tree, evaluation, or clustering. Second write the formula before",
        "The dialogue's closing strategy breaks off mid-sentence, so the dialogue ends with no "
        "resolution.",
    ),
}

TARGETS = {
    **{i: ("consistency", *v) for i, v in CONSISTENCY.items()},
    **{i: ("sequentiality", *v) for i, v in SEQUENTIALITY.items()},
    **{i: ("outcome_completion", *v) for i, v in COMPLETION.items()},
}


def sabotage_checks(source: list[dict], t4: list[dict]) -> list[str]:
    """Pre-registration section 'Sabotage checks before submission', run here so a failure blocks
    the build rather than being discovered after GPU time is spent."""
    problems = []

    changed = [i for i in range(len(source))
               if source[i]["tutor"]["text"] != t4[i]["tutor"]["text"]]
    if changed != sorted(TARGETS):
        problems.append(f"tutor turns differ at {changed}, declared {sorted(TARGETS)}")

    for i in range(len(source)):
        if source[i]["student"]["text"] != t4[i]["student"]["text"]:
            problems.append(f"student turn {i} is not byte-identical to dm1")

    if len(t4) != len(source):
        problems.append(f"length changed: {len(source)} -> {len(t4)}")

    # A macro-only condition must not shorten the dialogue or empty a turn: an empty tutor turn
    # would be a segment-scope failure, not a dialogue-scope one.
    for i, item in enumerate(t4):
        if not item["tutor"]["text"].strip():
            problems.append(f"tutor turn {i} is empty")

    # Every declared property must actually be represented.
    covered = {TARGETS[i][0] for i in TARGETS}
    for prop in ("consistency", "sequentiality", "outcome_completion"):
        if prop not in covered:
            problems.append(f"no turn targets {prop}")

    return problems


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    n = len(source)
    bad = [i for i in TARGETS if i not in range(n)]
    if bad:
        raise RuntimeError(f"TARGETS references out-of-range indices: {bad}")

    t4, records = [], []
    for i, item in enumerate(source):
        student = item["student"]["text"]
        original = item["tutor"]["text"]
        if i in TARGETS:
            prop, text, description = TARGETS[i]
            t4.append({"student": {"text": student}, "tutor": {"text": text}})
            records.append({
                "exchange_index": i,
                "exchange_id": f"ex_{i:04d}",
                "macro_property_targeted": prop,
                "violation": description,
                "original_tutor_text": original,
                "variant_tutor_text": text,
            })
        else:
            t4.append({"student": {"text": student}, "tutor": {"text": original}})

    problems = sabotage_checks(source, t4)
    if problems:
        print("SABOTAGE CHECKS FAILED - nothing written:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    out = OUT_DIR / "tv_d_source.json"
    out.write_text(json.dumps(t4, indent=2, ensure_ascii=False), encoding="utf-8")

    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps({
        "schema": "macro_degraded_variant_ground_truth.v1",
        "preregistration": "docs/evaluation_completeness_v2/27_MACRO_DEGRADATION_PREREGISTRATION.md",
        "source_dialogue": str(SOURCE.relative_to(REPO_ROOT)),
        "variant": {
            "id": "T4_macro_degraded",
            "opaque_dialogue_id": "tv_d_20260909",
            "source": "tv_d_source.json",
            "design": "minimal pair with T1; degradation is dialogue-scope only",
        },
        "blinding_note": (
            "The opaque id tv_d carries no description of the condition. dialogue_id is embedded "
            "in every segment and exchange id and therefore appears inside each judge prompt, so "
            "a descriptive id would disclose the condition to a judge meant to score blind."
        ),
        "exchanges_total": n,
        "exchanges_modified": len(records),
        "exchanges_identical_to_T1": n - len(records),
        "targets_by_property": {
            prop: sorted(i for i in TARGETS if TARGETS[i][0] == prop)
            for prop in ("consistency", "sequentiality", "outcome_completion")
        },
        "preregistered_leak": (
            "H-M4: idx 57 and 62 trail off, which is locally poor actionability as well as a "
            "dialogue-level failure. Predicted contribution to the segment-scope gap: <= 0.03."
        ),
        "modified_exchanges": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  sabotage checks passed")
    print(f"  {n} exchanges, {len(records)} modified, {n - len(records)} identical to T1")
    for prop in ("consistency", "sequentiality", "outcome_completion"):
        idxs = sorted(i for i in TARGETS if TARGETS[i][0] == prop)
        print(f"    {prop:<20} {idxs}")
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    print(f"  wrote {GROUND_TRUTH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
