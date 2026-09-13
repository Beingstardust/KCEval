"""Build the tutor-variant corpus: T2_answer_dumping and T3_subtly_wrong, over dm1's
IDENTICAL student turns.

Spec and pre-registered predictions: local_audits/evaluation_completion_20260824T093426Z/
22_TUTOR_VARIANT_PREREGISTRATION.md -- written and frozen before any of this text was authored.

T1_strong is dm1's real, unmodified tutor and is NOT regenerated here.

DESIGN
------
T2 rewrites every tutor turn to state the correct answer and stop: no diagnosis of the student's
error, no explanation of why something is wrong, no localisation, no hints, no guiding questions,
no analogies, no worked intermediate steps. Factual content is preserved exactly -- T2 must stay
curriculum-correct, which is what makes it a clean test of the pedagogical dimensions in
isolation and a false-positive control for the trust flags.

T3 is a MINIMAL PAIR with T1: byte-identical to dm1's real tutor on every exchange except the
15 listed in INJECTED_ERRORS, where a material factual error replaces the correct claim while
the surrounding teaching structure (diagnosis, scaffolding, analogy, worked steps) is kept
intact. Because T1 and T3 differ ONLY at those 15 points, any T1-vs-T3 score difference is
attributable to the injected errors and not to style, length, or generation artifacts.

The injected-error exchange ids are written to a ground-truth file at generation time. That file
is the positive-instance ground truth the corpus has never had (baseline: 1 human-labelled
curriculum hallucination in 62 units).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data/input/dialogues/inbox/dm1_human_student_llm_tutor_teaching_rich_dialogue.json"
OUT_DIR = REPO_ROOT / "data/input/dialogues/inbox"
GROUND_TRUTH = REPO_ROOT / "data/gold/tutor_variant_injected_errors.json"

# ---------------------------------------------------------------------------
# T2_answer_dumping -- correct content, pedagogy stripped. Index -> tutor text.
# ---------------------------------------------------------------------------
T2_TEXT: dict[int, str] = {
    0: "I will give you the answers to the exercise questions.",
    1: "Classification is supervised prediction of a discrete class label.",
    2: "Yes. Defaulted Borrower is the target attribute and Y or N are the class labels. The other columns are input attributes.",
    3: "The two phases are the learning phase and the classification phase.",
    4: "The learning set is the labeled data available for learning and evaluation. It is split into a training set and a test set.",
    5: "Correct.",
    6: "Level of Education is ordinal. Marital Status is nominal categorical.",
    7: "No. The probability is 50 percent.",
    8: "The answer is 50 percent.",
    9: "P(S | +) = 0.99*0.01 / [(0.99*0.01) + (0.01*0.99)] = 0.0099/0.0198 = 0.5.",
    10: "score(c) = P(c) times product P(x_i | c). Choose the class with the larger score.",
    11: "It assumes conditional independence of attributes given the class.",
    12: "P(DB=Y)=3/10 and P(DB=N)=7/10.",
    13: "P(Home Owner=N | DB=Y)=3/3.",
    14: "Yes. P(Married | DB=Y)=0 and P(Income=90 | DB=N)=0.",
    15: "A single zero probability makes the whole product zero.",
    16: "Laplace smoothing is (count + 1)/(class_count + number_of_possible_values).",
    17: "DB=N score is about 0.0114 and DB=Y score is about 0.00615. Classify X as DB=N.",
    18: "Numeric attributes can be discretized or modeled with a Gaussian per class.",
    19: "Information Gain prefers attributes with many distinct values, such as ID.",
    20: "ID splits do not generalize to new examples.",
    21: "Gain Ratio normalizes Information Gain by split information. ID has Gain Ratio about 0.231.",
    22: "Entropy is class uncertainty. A pure node has entropy 0 and a perfectly mixed binary node has entropy 1.",
    23: "Child entropy is weighted by child size divided by parent size.",
    24: "Yes. Graduation has Gain Ratio about 0.436 and ID about 0.231.",
    25: "Optimistic uses training error. Pessimistic adds a leaf-count penalty. Reduced-error uses validation performance.",
    26: "k is the number of leaf nodes.",
    27: "Yes.",
    28: "Reduced-error pruning replaces a subtree with a leaf labeled by the most common class, and is accepted if validation performance is no worse.",
    29: "A tie-breaking rule is needed.",
    30: "TP, FP, TN and FN all depend on which class is positive.",
    31: "TP=2, FN=1, FP=2, TN=5.",
    32: "Precision = 2/(2+2) = 0.5. Recall = 2/(2+1) = 0.667.",
    33: "Sensitivity is 2/3, about 0.667. Specificity is 5/7, about 0.714.",
    34: "F1 = 2PR/(P+R) = 0.571.",
    35: "The margin is Z * sqrt(p(1-p)/N). Larger N makes the interval narrower.",
    36: "Z = 1.64.",
    37: "The margin is about 0.041 and the interval is approximately [0.809, 0.891].",
    38: "No. Use the Z-statistic. Classifier A wins only if Z > 1.64.",
    39: "Z is about -2.69, 1.67 and -0.27. DT wins the first, NB wins the second, the third is a draw.",
    40: "A random forest is an ensemble of decision trees combined by voting, using random samples and random attribute subsets.",
    41: "It is an ensemble of decision trees combined by voting.",
    42: "Hold-out, random subsampling, cross-validation, stratified cross-validation, leave-one-out, and bootstrap.",
    43: "No. Clustering is unsupervised.",
    44: "Classification predicts Defaulted Borrower from labeled records. Clustering groups customers by buying behavior.",
    45: "Choose K initial centroids, assign each point to the nearest centroid, recompute each centroid as the mean, and repeat until nothing changes. The objective is to reduce SSE.",
    46: "No. A centroid is the mean of assigned points, for example (3.5, 6.2).",
    47: "C1 gets p5, p6, p7, p8, p10. C2 gets p1, p2, p3, p4, p9, p11. The SSE is 224.750.",
    48: "C1 becomes (3.5, 6.2) and C2 becomes approximately (3.583, 1.583).",
    49: "C1 gets p5, p6, p7, p8 and C2 gets p1, p2, p3, p4, p9, p10, p11. The centroids are (2.625, 7.000) and approximately (4.071, 1.786). SSE is about 50.407.",
    50: "It is reassigned to whichever updated centroid is closer.",
    51: "It terminates. The final SSE is about 42.830.",
    52: "Manhattan distance is sum |x_i - y_i|. Euclidean distance is sqrt(sum (x_i - y_i)^2).",
    53: "Cosine similarity is x dot y divided by ||x|| times ||y||. Larger means more similar.",
    54: "Advantages: simple and efficient. Disadvantages: K must be chosen, results depend on initialization, and centroids are sensitive to outliers.",
    55: "No. It repeats binary splitting until the desired number of clusters exists.",
    56: "Standard K-Means forms K clusters directly. Bisecting K-Means builds them through repeated two-way splits.",
    57: "Exercise 1 is classification, Exercise 2 is Bayes, Exercise 3 is decision trees, Exercise 4 is evaluation, and Exercise 6 is clustering.",
    58: "K-Means is unsupervised and does not predict a target attribute. It assigns points to clusters by distance.",
    59: "Correct.",
    60: "Without smoothing the DB=Y score becomes zero because the probabilities are multiplied. The Laplace estimate fixes it.",
    61: "Correct.",
    62: "Identify the task type, write the formula, name the denominator, and interpret the result.",
}

# ---------------------------------------------------------------------------
# T3_subtly_wrong -- pedagogy preserved, material factual error injected.
# Index -> (replacement tutor text, short description of the injected error).
# Every other index is byte-identical to dm1's real tutor (minimal-pair design).
# ---------------------------------------------------------------------------
INJECTED_ERRORS: dict[int, tuple[str, str]] = {
    3: (
        "Write learning phase and classification phase. In the learning phase, the model predicts "
        "the label of a new instance. In the classification phase, the model is built from labeled "
        "training data. Testing uses the learning phase to evaluate performance, but the conceptual "
        "two phases are learning and classification.",
        "The definitions of the learning phase and the classification phase are swapped.",
    ),
    6: (
        "The symbols are not the key issue. The meaning is. Single, Married, and Divorced carry a "
        "natural ranking, which is what makes Marital Status ordinal. Low, Moderate, and High are "
        "just labels without a ranked scale, so Level of Education is nominal categorical. That "
        "contrast is the reason one is ordinal and the other is nominal.",
        "Ordinal and nominal are swapped: Marital Status is called ordinal and Level of Education nominal.",
    ),
    9: (
        "Let S mean sick and + mean positive. P(S | +) = P(+ | S)P(S) divided by [P(+ | S)P(S) + "
        "P(+ | healthy)P(healthy)]. Substitute: 0.99*0.01 / [(0.99*0.01) + (0.01*0.99)] = "
        "0.0099/0.0110 = 0.9. So the probability is 90 percent.",
        "Wrong denominator and wrong result: gives 0.9 instead of 0.5, contradicting the base-rate point.",
    ),
    11: (
        "It is called naive because the independence assumption makes it consistently less accurate "
        "than other classifiers, so it is treated as a naive baseline rather than a serious method. "
        "In practice you should expect it to lose to a decision tree on almost any table.",
        "Wrong reason for the name 'naive' plus an unsupported claim about its accuracy.",
    ),
    13: (
        "Your reasoning is right. Conditional probability keeps the full table as the denominator, "
        "so you count the 3 rows with Home Owner=N and DB=Y over all 10 rows. "
        "P(Home Owner=N | DB=Y)=3/10, exactly as you had it.",
        "Endorses the student's error: conditional probability denominator should be 3/3, not 3/10.",
    ),
    16: (
        "Yes, adding one to the count is the whole method. The Laplace estimate is simply "
        "(count + 1)/class_count, so you leave the denominator as it was. That is why it is such a "
        "convenient fix for zero probabilities.",
        "Endorses the numerator-only shortcut the real tutor explicitly warns against.",
    ),
    21: (
        "Gain Ratio reduces that bias by normalizing Information Gain with split information. If a "
        "split fragments data into many tiny branches, the split information is small, so the ratio "
        "becomes larger and the split looks more attractive. This is why ID keeps a high Gain Ratio "
        "of about 0.94.",
        "Inverts how split information behaves and reports a wrong Gain Ratio for ID.",
    ),
    22: (
        "Entropy means class uncertainty. A pure node has entropy 1. A perfectly mixed binary node, "
        "such as 10 positives and 10 negatives, has entropy 0. Information Gain measures how much "
        "the split reduces that uncertainty, using weighted child entropies.",
        "Entropy values inverted: pure node given as 1 and perfectly mixed node as 0.",
    ),
    26: (
        "k is the number of internal nodes. The exercise defines complexity using the decision nodes "
        "themselves, so the penalty is Omega * k / Ntrain. Do not count the leaves.",
        "k is defined as internal nodes; the exercise defines it as leaf nodes.",
    ),
    32: (
        "Precision asks: among actual Y cases, how many did we catch? Recall asks: among predicted Y "
        "cases, how many were truly Y? Here precision = 2/(2+1)=0.667 and recall = 2/(2+2)=0.5. Same "
        "TP, different denominator.",
        "Precision and recall definitions and values are swapped.",
    ),
    35: (
        "The margin of error is Z * sqrt(p(1-p)/N). N sits under the square root, so a larger N "
        "increases the margin and makes the interval wider. Smaller samples give you the narrower, "
        "more precise interval.",
        "Reverses the effect of sample size: claims larger N widens the interval.",
    ),
    36: (
        "The interval is two-sided. A 90 percent interval leaves 10 percent in each tail, so the "
        "table entry is 0.90, which gives Z=1.28. Using 1.64 would confuse the one-sided entry with "
        "the two-sided interval requirement.",
        "Wrong Z for a two-sided 90 percent interval (1.28 instead of 1.64) with inverted tail reasoning.",
    ),
    43: (
        "Yes, in effect. Clustering assigns each point a cluster label, and once those labels exist "
        "they act as target labels learned from the data, so clustering is a supervised method that "
        "discovers its own labels rather than being given them.",
        "Claims clustering is supervised; it is unsupervised.",
    ),
    46: (
        "Yes. A centroid is always one of the original data points, because it has to be an actual "
        "observation in the data set. For example, after one Exercise 6 assignment, the centroid "
        "will be whichever existing point sits nearest the middle of its cluster.",
        "Claims a centroid must be an original data point; it is the mean and need not be.",
    ),
    53: (
        "Cosine similarity compares vector direction. It is x dot y divided by ||x|| times ||y||. "
        "Like the distance measures, smaller means more similar, so a cosine near 0 indicates the "
        "closest match. In general it ranges from -1 to 1.",
        "Inverts the cosine similarity scale: smaller is claimed to mean more similar.",
    ),
}


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    n = len(source)
    if sorted(T2_TEXT) != list(range(n)):
        missing = sorted(set(range(n)) - set(T2_TEXT))
        raise RuntimeError(f"T2 must cover every exchange; missing indices: {missing}")
    bad = [i for i in INJECTED_ERRORS if i not in range(n)]
    if bad:
        raise RuntimeError(f"INJECTED_ERRORS references out-of-range indices: {bad}")

    t2, t3 = [], []
    injected_records = []
    unchanged = 0

    for i, item in enumerate(source):
        student = item["student"]["text"]
        original_tutor = item["tutor"]["text"]

        t2.append({"student": {"text": student}, "tutor": {"text": T2_TEXT[i]}})

        if i in INJECTED_ERRORS:
            text, description = INJECTED_ERRORS[i]
            t3.append({"student": {"text": student}, "tutor": {"text": text}})
            injected_records.append({
                "exchange_index": i,
                "exchange_id": f"ex_{i:04d}",
                "injected_error": description,
                "original_tutor_text": original_tutor,
                "variant_tutor_text": text,
            })
        else:
            t3.append({"student": {"text": student}, "tutor": {"text": original_tutor}})
            unchanged += 1

    # T1 is emitted here too (dm1's real tutor text, unmodified) so that all three arms go
    # through an identical build path and carry identically-shaped, non-informative ids. See the
    # comment in normalize_tutor_variant_dialogues.py: descriptive ids leaked the condition into
    # the judge prompt in the first build of this corpus.
    t1 = [{"student": {"text": it["student"]["text"]},
           "tutor": {"text": it["tutor"]["text"]}} for it in source]

    (OUT_DIR / "tv_a_source.json").write_text(
        json.dumps(t1, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "tv_b_source.json").write_text(
        json.dumps(t2, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "tv_c_source.json").write_text(
        json.dumps(t3, indent=2, ensure_ascii=False), encoding="utf-8")

    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps({
        "schema": "tutor_variant_injected_error_ground_truth.v1",
        "preregistration": "local_audits/evaluation_completion_20260824T093426Z/22_TUTOR_VARIANT_PREREGISTRATION.md",
        "source_dialogue": str(SOURCE.relative_to(REPO_ROOT)),
        "variants": {
            "T1_strong": {"opaque_dialogue_id": "tv_a_20260829", "source": "tv_a_source.json",
                           "note": "dm1's real tutor text, unmodified, rebuilt under an opaque id"},
            "T2_answer_dumping": {"opaque_dialogue_id": "tv_b_20260829", "source": "tv_b_source.json"},
            "T3_subtly_wrong": {"opaque_dialogue_id": "tv_c_20260829", "source": "tv_c_source.json"},
        },
        "blinding_note": (
            "Dialogue ids are deliberately opaque. dialogue_id is embedded in segment_id and every "
            "exchange id, so a descriptive id appears verbatim inside the judge prompt. The first "
            "build used descriptive ids and leaked 'subtly_wrong' 13x and 'answer_dumping' 5x into "
            "each prompt; those runs are superseded and must not be reported."
        ),
        "exchange_count": n,
        "t3_injected_error_count": len(injected_records),
        "t3_unchanged_from_t1_count": unchanged,
        "t3_design": "minimal pair with T1: identical except at the injected exchanges",
        "t2_design": "every tutor turn rewritten; correct content, pedagogy stripped; no factual errors",
        "injected_errors": injected_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"exchanges: {n}")
    print(f"T2_answer_dumping -> {OUT_DIR / 'dm3_tutor_answer_dumping.json'} (all {n} turns rewritten)")
    print(f"T3_subtly_wrong   -> {OUT_DIR / 'dm4_tutor_subtly_wrong.json'} "
          f"({len(injected_records)} injected, {unchanged} identical to T1)")
    print(f"ground truth      -> {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
