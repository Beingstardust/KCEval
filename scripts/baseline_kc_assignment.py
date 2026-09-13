"""Tier 1 — does the v35 KC-indexing pipeline earn its complexity?

The headline segmentation number (dm1 81.7% exact / dm2 66.7%, pooled 75.5%) has, until now, been
compared only against our OWN earlier versions (v28 -> v35, LIBRA 73.5%). There is no external or
even chance-level floor beneath it anywhere in the project, so we cannot presently say whether 75.5%
is good.

This is the same question E1 asked of the rubric layer -- does the added machinery beat the obvious
simple thing? -- applied to the part of KCEval that is genuinely ours rather than inherited from
MRBench.

BASELINES
---------
  B0  uniform random over the library            the arithmetic floor
  B1  most frequent gold primary KC              the degenerate-but-real floor a classifier must beat
  B2  naive dense retrieval, top-1               KC text = title + canonical name + reviewer
                                                 definition. "What you would build in an afternoon."
  B3  naive dense retrieval, top-1               KC text = the frozen matching profile. Separates
                                                 the value of the LIBRARY CONTENT from the value of
                                                 the PIPELINE LOGIC.

B2/B3 deliberately use bge-small-en-v1.5 -- the SAME dense model v35 uses -- so the contrast
isolates the five-pass logic, tie-breaker and branch handling, not embedding quality. What B2/B3
lack relative to v35 is every decision layer: no cross-encoder rerank, no tie-breaker, no branch
guidance, no novelty check, no abstention, no pass-2 resolution.

SCORING is identical to `verify_dm2_segmentation_reproduction.py`: exact = predicted equals gold
`primary`; acceptable = exact or predicted appears in gold `secondary`. All candidate index offsets
are scored and the best reported, so a misalignment is visible rather than silent.

This is an ABLATION, not a system comparison. It establishes whether the pipeline earns its
complexity; it does not compare KCEval to anyone else's system. Tier 2 does that.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from math import sqrt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 1234
N_SIM = 10_000

_DM1 = ("data/processed/segmentation_runs/"
        "dm1_human_student_llm_tutor_teaching_rich_dialogue_textonly_20260618T174736Z")
_DM2 = "data/processed/segmentation_runs/dm2_gold_holdout_dialogue_20260811"

CORPORA = {
    "dm1": {
        "gold": "data/gold/gold_struct_dm1.json",
        "exchanges": f"{_DM1}/normalized/normalized_exchanges.jsonl",
        "v35_assignments": f"{_DM1}/v35_20260812/exchange_assignments.jsonl",
        "v35_reported_exact": 0.817, "v35_reported_acceptable": 0.850,
    },
    "dm2": {
        "gold": "data/gold/gold_struct_dm2.json",
        "exchanges": f"{_DM2}/v35_20260812_packet_staging/exchanges.jsonl",
        "v35_assignments": f"{_DM2}/v35_20260812/exchange_assignments.jsonl",
        "v35_reported_exact": 0.667, "v35_reported_acceptable": 0.857,
    },
}


def jl(p) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_gold(path) -> dict[int, dict]:
    raw = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    return {int(k.split("_")[1]): v for k, v in raw.items()}


def score(pred: dict[int, str | None], gold: dict[int, dict]) -> dict:
    """Best over candidate offsets, exactly as the v35 reproduction check does."""
    best = None
    for off in (0, 1, 2):
        common = [i for i in pred if (i + off) in gold]
        if not common:
            continue
        ex = sum(1 for i in common if pred[i] == gold[i + off].get("primary"))
        ac = sum(1 for i in common
                 if pred[i] == gold[i + off].get("primary")
                 or (pred[i] and pred[i] in (gold[i + off].get("secondary") or [])))
        row = {"offset": off, "n": len(common), "exact_n": ex, "acceptable_n": ac,
               "exact": ex / len(common), "acceptable": ac / len(common)}
        if best is None or row["exact"] > best["exact"]:
            best = row
    return best or {"n": 0, "exact": 0.0, "acceptable": 0.0, "exact_n": 0, "acceptable_n": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library",
                    default="data/processed/console_runs/E7S_hv_p/library/reviewed_library.jsonl")
    ap.add_argument("--profiles",
                    default="data/processed/console_runs/E7S_hv_p/library"
                            "/frozen_matching_profiles.jsonl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="HuggingFace model id or local path for the sentence-transformers "
                         "encoder used by the B2/B3 naive-retrieval baselines (same encoder "
                         "the v35 pipeline's dense-retrieval stage uses).")
    ap.add_argument("--out", default="data/gold/kc_assignment_baselines.json")
    args = ap.parse_args()
    rng = random.Random(SEED)

    library = jl(REPO_ROOT / args.library)
    # The matching profiles key on `unit_id`, NOT `kc_id`. Keying on kc_id returns None for every
    # entry, B3 silently falls back to B2's text, and the two baselines come out byte-identical --
    # which is exactly what happened on the first run. Accept either key.
    profiles = {}
    for p in jl(REPO_ROOT / args.profiles):
        key = p.get("unit_id") or p.get("kc_id")
        if key:
            profiles[key] = p
    kc_ids = [k["kc_id"] for k in library]
    matched = sum(1 for k in kc_ids if k in profiles)
    print(f"library: {len(kc_ids)} KCs; matching profiles joined: {matched}")
    if matched < len(kc_ids):
        raise SystemExit(f"only {matched}/{len(kc_ids)} KCs joined to a matching profile; B3 would "
                         "silently degrade into B2")

    def kc_text_plain(k: dict) -> str:
        return " ".join(str(k.get(f) or "") for f in
                        ("title", "canonical_name", "reviewer_facing_definition")).strip()

    def kc_text_profile(k: dict) -> str:
        p = profiles.get(k["kc_id"]) or {}
        # Use the profile's own retrieval text, whatever fields it carries, rather than assuming a
        # schema: this baseline exists to measure what the profile engineering is worth.
        parts = [str(v) for key, v in p.items()
                 if isinstance(v, str) and key not in ("kc_id", "schema_version", "library_tier")]
        parts += [" ".join(str(x) for x in v) for v in p.values() if isinstance(v, list)
                  and all(isinstance(x, str) for x in v)]
        return " ".join(parts).strip() or kc_text_plain(k)

    from sentence_transformers import SentenceTransformer  # noqa: E402
    import numpy as np  # noqa: E402

    model = SentenceTransformer(args.model, device="cpu")

    def embed(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True, batch_size=32,
                            show_progress_bar=False, convert_to_numpy=True)

    texts_plain = [kc_text_plain(k) for k in library]
    texts_prof = [kc_text_profile(k) for k in library]
    differing = sum(1 for a, b in zip(texts_plain, texts_prof) if a != b)
    print(f"KCs whose profile text differs from plain text: {differing}/{len(library)}")
    if differing < len(library) // 2:
        raise SystemExit("B3's profile text is mostly identical to B2's; it would not be a "
                         "separate baseline. Check the profile join and field extraction.")
    emb_plain = embed(texts_plain)
    emb_prof = embed(texts_prof)

    results: dict = {
        "purpose": "Tier 1 ablation: does the v35 KC-indexing pipeline earn its complexity?",
        "what_this_is_not": "not a system comparison; it compares v35 against naive baselines on "
                            "our own expert gold",
        "dense_model": "BAAI/bge-small-en-v1.5 (the same dense model v35 uses, so the contrast "
                       "isolates pipeline logic rather than embedding quality)",
        "scoring": "identical to verify_dm2_segmentation_reproduction.py; best over index offsets",
        "seed": SEED, "library_size": len(kc_ids), "corpora": {},
    }

    print("=" * 96)
    print("TIER 1 - KC ASSIGNMENT BASELINES vs v35")
    print("=" * 96)

    pooled = {b: [0, 0, 0] for b in ("B0_random", "B1_majority", "B2_dense_plain",
                                     "B3_dense_profile", "v35")}

    for name, cfg in CORPORA.items():
        gold = load_gold(cfg["gold"])
        rows = jl(REPO_ROOT / cfg["exchanges"])
        idx_text = {}
        for r in rows:
            i = r.get("exchange_index")
            if i is None:
                continue
            idx_text[int(i)] = f"{r.get('student_text') or ''}\n{r.get('tutor_text') or ''}".strip()
        if not idx_text:
            raise SystemExit(f"{name}: no exchange text at {cfg['exchanges']}")

        order = sorted(idx_text)
        emb_ex = embed([idx_text[i] for i in order])

        # B0 uniform random: simulate rather than assume, so `acceptable` (which depends on how many
        # secondaries each gold entry has) is estimated rather than hand-derived.
        sims_e, sims_a = [], []
        for _ in range(N_SIM):
            pred = {i: rng.choice(kc_ids) for i in order}
            s = score(pred, gold)
            sims_e.append(s["exact"])
            sims_a.append(s["acceptable"])
        b0 = {"exact": sum(sims_e) / N_SIM, "acceptable": sum(sims_a) / N_SIM,
              "n": len(order), "note": f"mean of {N_SIM} simulated random assignments"}

        # B1 majority: the single most frequent gold primary, assigned to everything.
        majority = Counter(v.get("primary") for v in gold.values()).most_common(1)[0][0]
        b1 = score({i: majority for i in order}, gold)
        b1["kc_used"] = majority

        pred_b2 = {i: kc_ids[int(j)] for i, j in
                   zip(order, (emb_ex @ emb_plain.T).argmax(axis=1))}
        pred_b3 = {i: kc_ids[int(j)] for i, j in
                   zip(order, (emb_ex @ emb_prof.T).argmax(axis=1))}
        b2 = score(pred_b2, gold)
        b3 = score(pred_b3, gold)

        # v35 is recomputed from its own frozen per-exchange artifacts rather than trusting the
        # reported percentages, so the comparison and the paired test rest on the same scorer.
        pred_v35 = {int(r["exchange_index"]): r.get("resolved_kc_id")
                    for r in jl(REPO_ROOT / cfg["v35_assignments"])}
        v35 = score(pred_v35, gold)
        for label, got, want in (("exact", v35["exact"], cfg["v35_reported_exact"]),
                                 ("acceptable", v35["acceptable"], cfg["v35_reported_acceptable"])):
            if abs(got - want) > 0.005:
                raise SystemExit(f"{name}: recomputed v35 {label} {got:.3f} does not match the "
                                 f"reported {want:.3f}; the scorer or the artifact is wrong")
        n = v35["n"]

        # Paired McNemar on exact correctness, v35 vs the stronger naive baseline, over the
        # exchanges both scored. Unpaired interval overlap is not a test; discordant pairs are.
        off_v, off_b = v35["offset"], b2["offset"]
        shared = [i for i in order if i in pred_v35 and (i + off_v) in gold and (i + off_b) in gold]
        b_only = sum(1 for i in shared
                     if pred_v35[i] == gold[i + off_v]["primary"]
                     and pred_b2[i] != gold[i + off_b]["primary"])
        c_only = sum(1 for i in shared
                     if pred_v35[i] != gold[i + off_v]["primary"]
                     and pred_b2[i] == gold[i + off_b]["primary"])
        results.setdefault("_mcnemar_cells", {"b": 0, "c": 0, "n": 0})
        results["_mcnemar_cells"]["b"] += b_only
        results["_mcnemar_cells"]["c"] += c_only
        results["_mcnemar_cells"]["n"] += len(shared)

        for b in (b1, b2, b3):
            b["exact_ci"] = list(wilson(b["exact_n"], b["n"]))
            b["acceptable_ci"] = list(wilson(b["acceptable_n"], b["n"]))
        v35["exact_ci"] = list(wilson(v35["exact_n"], n))

        results["corpora"][name] = {"B0_random": b0, "B1_majority": b1, "B2_dense_plain": b2,
                                    "B3_dense_profile": b3, "v35": v35}
        # Persist per-exchange predictions so Tier 2 can score this baseline in the same
        # label-space-free partition metrics, rather than re-deriving the embeddings there.
        results["corpora"][name]["predictions"] = {
            "B2_dense_plain": {str(i): pred_b2[i] for i in sorted(pred_b2)},
            "B3_dense_profile": {str(i): pred_b3[i] for i in sorted(pred_b3)},
            "v35": {str(i): pred_v35[i] for i in sorted(pred_v35)},
        }
        for key, b in (("B0_random", b0), ("B1_majority", b1), ("B2_dense_plain", b2),
                       ("B3_dense_profile", b3), ("v35", v35)):
            pooled[key][0] += b.get("exact_n", b["exact"] * b["n"])
            pooled[key][1] += b.get("acceptable_n", b["acceptable"] * b["n"])
            pooled[key][2] += b["n"]

        print(f"\n  {name}  (n={n}, gold has {len(gold)} exchanges)")
        print(f"    {'baseline':22s} {'exact':>8s} {'acceptable':>11s}")
        for label, b in (("B0 uniform random", b0), (f"B1 majority KC", b1),
                         ("B2 dense, plain text", b2), ("B3 dense, profile", b3),
                         ("v35 full pipeline", v35)):
            print(f"    {label:22s} {b['exact']:7.1%} {b['acceptable']:10.1%}")

    print(f"\n  POOLED (dm1 + dm2)")
    print(f"    {'baseline':22s} {'exact':>8s} {'acceptable':>11s}   {'exact 95% CI':>18s}")
    results["pooled"] = {}
    for key, label in (("B0_random", "B0 uniform random"), ("B1_majority", "B1 majority KC"),
                       ("B2_dense_plain", "B2 dense, plain text"),
                       ("B3_dense_profile", "B3 dense, profile"),
                       ("v35", "v35 full pipeline")):
        e, a, n = pooled[key]
        lo, hi = wilson(round(e), n)
        results["pooled"][key] = {"exact": e / n, "acceptable": a / n, "n": n,
                                  "exact_ci": [lo, hi]}
        print(f"    {label:22s} {e / n:7.1%} {a / n:10.1%}   [{lo:6.1%}, {hi:6.1%}]")

    v = results["pooled"]["v35"]["exact"]
    best_naive = max(results["pooled"][k]["exact"] for k in
                     ("B0_random", "B1_majority", "B2_dense_plain", "B3_dense_profile"))
    results["v35_margin_over_best_naive"] = v - best_naive
    print(f"\n  v35 pooled exact {v:.1%} vs best naive baseline {best_naive:.1%}  "
          f"-> margin {v - best_naive:+.1%}")

    # Exact two-sided McNemar (binomial on discordant pairs). Overlapping Wilson intervals are not
    # a test; this is the paired one.
    cells = results.pop("_mcnemar_cells")
    b, c, n_shared = cells["b"], cells["c"], cells["n"]
    from math import comb
    m = b + c
    if m:
        p = min(1.0, 2 * sum(comb(m, k) for k in range(0, min(b, c) + 1)) / (2 ** m))
    else:
        p = 1.0
    results["mcnemar_v35_vs_B2"] = {
        "paired_exchanges": n_shared,
        "v35_right_B2_wrong": b, "v35_wrong_B2_right": c, "discordant": m,
        "p_two_sided_exact": p, "significant_at_0_05": p < 0.05,
        "test": "exact McNemar (two-sided binomial on discordant pairs)",
    }
    print(f"\n  paired McNemar, v35 vs B2 (dense, plain text), n={n_shared} exchanges:")
    print(f"    v35 right / B2 wrong: {b}    v35 wrong / B2 right: {c}    discordant: {m}")
    print(f"    exact two-sided p = {p:.5f}  -> "
          f"{'SIGNIFICANT' if p < 0.05 else 'NOT significant'} at 0.05")

    (REPO_ROOT / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWROTE {REPO_ROOT / args.out}")


if __name__ == "__main__":
    main()
