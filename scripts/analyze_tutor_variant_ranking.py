"""SUPERSEDED BY THE SINGLE-JUDGE POLICY (doc 49). The tables in this module point at the
Qwen response directories and are kept only because published results were computed from
them. The projects authoritative tutor scores come from scripts/compute_routed_tutor_scores.py,
which reads the Selene passes for all five arms.

Test the pre-registered predictions in 22_TUTOR_VARIANT_PREREGISTRATION.md against the
tutor-variant judge runs.

This is the framework's first multi-tutor run and the first test of whether it can recover a
known-correct quality ordering. `deterministic_aggregation.py` (final_tutor_score,
sensitivity_grid, ranking_is_stable) has been built and unit-tested since an earlier phase but
has never had real multi-tutor data to run on.

Each prediction is evaluated mechanically and reported PASS / FAIL / UNDETERMINED. Nothing is
reinterpreted after the fact: the predictions were frozen before any variant text was authored.

    P1  ranking on trust-adjusted score: T1 > T2 > T3
    P2  the dissociation: T3 ~= T1 on pedagogical dimensions, separated by trust flags
    P3  dimension-specific movement, incl. the T2 false-positive control (T2 trust flags stay 0)
    P4  trust-flag recall > 0 against the injected-error ground truth
    P5  ranking stability across alpha

Usage:
    python scripts/analyze_tutor_variant_ranking.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seg_eval.evaluation_judge.rubric_v2 import PARTIAL_DIMENSIONS, TRUST_FLAGS  # noqa: E402
from seg_eval.evaluation_judge.response_contract_v2 import validate_response_rows  # noqa: E402
from seg_eval.aggregation.deterministic_aggregation import (  # noqa: E402
    weighted_mean_applicable, combine_local_and_arc, exchange_weighted_mean,
    DialogueScoreInputs, sensitivity_grid, ranking_is_stable,
)

GROUND_TRUTH = REPO_ROOT / "data/gold/tutor_variant_injected_errors.json"

# Pedagogical dimensions for the P2 dissociation test: what "teaching well" means, as distinct
# from "being factually right". Chosen from the rubric's own groupings before results were seen.
PEDAGOGICAL_DIMENSIONS = [
    "scaffolding_quality", "clarity_cognitive_load", "actionability_moving_forward",
    "solution_control", "student_level_calibration",
]

RHO = 0.8  # local-vs-ARC weight, live-derived; provisional, same value used in 08_FIRST_REAL_AGGREGATE_RESULTS.md

def _arm(tag: str) -> dict:
    return {
        "local": (f"data/processed/evaluation_packets/v35_{tag}_eval_20260829/kc_segment_local",
                   f"data/processed/judge_prompts/v35_{tag}_kc_segment_local_20260829",
                   f"data/processed/judge_responses/v35_{tag}_kc_segment_local_20260829"),
        "arc": (f"data/processed/evaluation_packets/v35_{tag}_eval_20260829/topic_rollup_arc",
                 f"data/processed/judge_prompts/v35_{tag}_topic_rollup_arc_20260829",
                 f"data/processed/judge_responses/v35_{tag}_topic_rollup_arc_20260829"),
    }


VARIANTS = {
    "T1_strong": _arm("tv_a"),
    "T2_answer_dumping": _arm("tv_b"),
    "T3_subtly_wrong": _arm("tv_c"),
}


MACRO_RESPONSES = {
    "T1_strong": "data/processed/judge_responses/v35_tv_a_macro_20260829",
    "T2_answer_dumping": "data/processed/judge_responses/v35_tv_b_macro_20260829",
    "T3_subtly_wrong": "data/processed/judge_responses/v35_tv_c_macro_20260829",
}

# Opaque dialogue ids: tv_a = T1_strong, tv_b = T2_answer_dumping, tv_c = T3_subtly_wrong.
# The mapping lives here and in the ground-truth file, never in an id the judge can read.
#
# SUPERSEDED RUN, NOT ANALYZED: an earlier build used descriptive dialogue ids
# (dm3_tutor_answer_dumping / dm4_tutor_subtly_wrong). Because dialogue_id is embedded in
# segment_id and every exchange id, those strings appeared inside each judge prompt --
# "subtly_wrong" 13x and "answer_dumping" 5x per prompt -- disclosing the experimental condition
# to a judge that was supposed to score blind. Those responses still exist on disk under
# v35_dm3_*/v35_dm4_* and must never be reported as results. T1 is also rebuilt here under an
# opaque id so that no arm is distinguishable by its label, and so all three arms share one
# segmentation run, GPU, and session (borderline segment boundaries were observed to differ
# between runs of byte-identical input -- 2 of 35 groupings -- so cross-run comparison is avoided).


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_macro_score(variant: str) -> float | None:
    """Mean of the macro dimension scores for one variant, or None when the macro response is
    absent or contract-invalid. Nulls are excluded, never coerced to 0."""
    rel = MACRO_RESPONSES.get(variant)
    if not rel:
        return None
    rows = load_jsonl(REPO_ROOT / rel / "macro_judge_responses.jsonl")
    if not rows:
        return None
    payload = None
    raw = rows[0].get("raw_response_text")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    scores = payload.get("dimension_scores") or {}
    vals = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def collect_family(packet_dir: str, prompt_dir: str, response_dir: str) -> tuple[list[dict], dict]:
    packets = load_jsonl(REPO_ROOT / packet_dir / "segment_evaluation_packets.jsonl")
    prompts = load_jsonl(REPO_ROOT / prompt_dir / "segment_judge_prompts.jsonl")
    responses = load_jsonl(REPO_ROOT / response_dir / "judge_responses.jsonl")
    if not responses:
        return [], {"attempted": len(packets), "valid": 0, "coverage": None}
    valid_rows, error_rows, _ = validate_response_rows(
        prompt_rows=prompts, packet_rows=packets, response_rows=responses)
    packets_by_id = {p["segment_id"]: p for p in packets}
    out = []
    for row in valid_rows:
        pkt = packets_by_id.get(row["segment_id"], {})
        out.append({
            "segment_id": row["segment_id"],
            "judge_output": row["judge_output"],
            "member_exchange_ids": pkt.get("member_exchange_ids") or [],
        })
    return out, {
        "attempted": len(packets), "valid": len(valid_rows),
        "coverage": len(valid_rows) / len(packets) if packets else None,
    }


def dimension_means(rows: list[dict]) -> dict[str, float | None]:
    """Mean of each dimension over units where the judge scored it (null/not-applicable excluded,
    never coerced to 0)."""
    out: dict[str, float | None] = {}
    for dim in PARTIAL_DIMENSIONS:
        vals = []
        for r in rows:
            v = (r["judge_output"].get("dimension_scores") or {}).get(dim)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        out[dim] = (sum(vals) / len(vals)) if vals else None
    return out


def trust_flag_rates(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for flag in TRUST_FLAGS:
        fired, total = 0, 0
        for r in rows:
            v = (r["judge_output"].get("trust_flags") or {}).get(flag)
            if v is None:
                continue
            total += 1
            fired += int(v) == 1
        out[flag] = {"fired": fired, "scored": total,
                      "rate": (fired / total) if total else None}
    return out


def dialogue_scores(local_rows: list[dict], arc_rows: list[dict]) -> dict:
    """Aggregate to dialogue level using the frozen deterministic aggregation."""
    def unit_score(r, key):
        jo = r["judge_output"]
        v = jo.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def mean_of(rows, key):
        vals = [unit_score(r, key) for r in rows]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    d_local_trust = mean_of(local_rows, "trust_adjusted_score")
    d_arc_trust = mean_of(arc_rows, "trust_adjusted_score")
    d_local_micro = mean_of(local_rows, "micro_score_raw")
    d_arc_micro = mean_of(arc_rows, "micro_score_raw")

    combined_trust = (combine_local_and_arc(d_local_trust, d_arc_trust, RHO).value
                       if d_local_trust is not None else None)
    combined_micro = (combine_local_and_arc(d_local_micro, d_arc_micro, RHO).value
                       if d_local_micro is not None else None)
    return {
        "d_local_trust_adjusted": d_local_trust,
        "d_arc_trust_adjusted": d_arc_trust,
        "d_micro_trust_adjusted": combined_trust,
        "d_local_micro_raw": d_local_micro,
        "d_arc_micro_raw": d_arc_micro,
        "d_micro_raw": combined_micro,
        "rho": RHO,
    }


def injected_error_detection(rows: list[dict], injected_exchange_ids: set[str]) -> dict:
    """Per-unit: did any trust flag fire on a unit that actually contains an injected error?

    Recall is computed over units that CONTAIN an injected error (the positive class the baseline
    corpus never had). Specificity is computed over units that contain none.
    """
    tp = fn = fp = tn = 0
    missed, caught = [], []
    for r in rows:
        short_ids = {m.split("::")[-1] for m in r["member_exchange_ids"]}
        has_injection = bool(short_ids & injected_exchange_ids)
        flags = r["judge_output"].get("trust_flags") or {}
        any_fired = any(int(v) == 1 for v in flags.values() if v is not None)
        if has_injection and any_fired:
            tp += 1
            caught.append(r["segment_id"])
        elif has_injection and not any_fired:
            fn += 1
            missed.append(r["segment_id"])
        elif not has_injection and any_fired:
            fp += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall_on_injected_units": recall,
        "precision": precision,
        "specificity": (tn / (tn + fp)) if (tn + fp) else None,
        "units_with_injection": tp + fn,
        "caught_units": caught, "missed_units": missed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data/gold/tutor_variant_ranking"))
    args = ap.parse_args()

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    injected_ids = {e["exchange_id"] for e in gt["injected_errors"]}

    results: dict[str, dict] = {}
    for variant, fams in VARIANTS.items():
        local_rows, local_cov = collect_family(*fams["local"])
        arc_rows, arc_cov = collect_family(*fams["arc"])
        if not local_rows and not arc_rows:
            print(f"NOTE: no responses for {variant} -- skipping")
            continue
        entry = {
            "coverage": {"local": local_cov, "arc": arc_cov},
            "dimension_means_local": dimension_means(local_rows),
            "dimension_means_arc": dimension_means(arc_rows),
            "trust_flag_rates_local": trust_flag_rates(local_rows),
            "trust_flag_rates_arc": trust_flag_rates(arc_rows),
            "dialogue_scores": dialogue_scores(local_rows, arc_rows),
        }
        if variant == "T3_subtly_wrong":
            entry["injected_error_detection_local"] = injected_error_detection(local_rows, injected_ids)
            entry["injected_error_detection_arc"] = injected_error_detection(arc_rows, injected_ids)
        results[variant] = entry

    # ---------------- prediction tests ----------------
    verdicts: dict[str, dict] = {}

    def trust_score(v):
        return (results.get(v, {}).get("dialogue_scores") or {}).get("d_micro_trust_adjusted")

    # P1 ranking
    have_all = all(trust_score(v) is not None for v in VARIANTS)
    if have_all:
        order = sorted(VARIANTS, key=lambda v: trust_score(v), reverse=True)
        predicted = ["T1_strong", "T2_answer_dumping", "T3_subtly_wrong"]
        verdicts["P1_ranking"] = {
            "predicted": predicted, "observed": order,
            "scores": {v: trust_score(v) for v in VARIANTS},
            "verdict": "PASS" if order == predicted else "FAIL",
        }
    else:
        verdicts["P1_ranking"] = {"verdict": "UNDETERMINED", "reason": "missing trust-adjusted scores"}

    # P2 dissociation
    if "T1_strong" in results and "T3_subtly_wrong" in results:
        ped = {}
        for dim in PEDAGOGICAL_DIMENSIONS:
            a = results["T1_strong"]["dimension_means_local"].get(dim)
            b = results["T3_subtly_wrong"]["dimension_means_local"].get(dim)
            if a is None:
                a = results["T1_strong"]["dimension_means_arc"].get(dim)
                b = results["T3_subtly_wrong"]["dimension_means_arc"].get(dim)
            ped[dim] = {"T1": a, "T3": b, "delta": (b - a) if (a is not None and b is not None) else None}
        deltas = [v["delta"] for v in ped.values() if v["delta"] is not None]
        mean_ped_delta = (sum(deltas) / len(deltas)) if deltas else None
        t1_trust = trust_score("T1_strong")
        t3_trust = trust_score("T3_subtly_wrong")
        trust_sep = (t1_trust - t3_trust) if (t1_trust is not None and t3_trust is not None) else None
        # dissociation holds when pedagogy barely moves but trust-adjusted score separates
        verdict = "UNDETERMINED"
        if mean_ped_delta is not None and trust_sep is not None:
            preserved = abs(mean_ped_delta) < 0.10
            separated = trust_sep > 0.02
            verdict = "PASS" if (preserved and separated) else "FAIL"
        verdicts["P2_dissociation"] = {
            "pedagogical_dimensions": ped,
            "mean_pedagogical_delta_T3_minus_T1": mean_ped_delta,
            "trust_adjusted_separation_T1_minus_T3": trust_sep,
            "criterion": "|mean pedagogical delta| < 0.10 AND trust-adjusted separation > 0.02",
            "verdict": verdict,
        }

    # P3 false-positive control: T2 must not trip trust flags
    if "T2_answer_dumping" in results:
        fired = 0
        for fam in ("trust_flag_rates_local", "trust_flag_rates_arc"):
            for flag, st in results["T2_answer_dumping"][fam].items():
                fired += st["fired"]
        verdicts["P3_T2_false_positive_control"] = {
            "t2_total_trust_flags_fired": fired,
            "verdict": "PASS" if fired == 0 else "FAIL",
            "note": "T2 introduces no factual errors; any fired trust flag means the judge conflates weak pedagogy with false statements.",
        }

    # P4 recall on injected errors
    if "T3_subtly_wrong" in results:
        det = results["T3_subtly_wrong"]["injected_error_detection_local"]
        r = det["recall_on_injected_units"]
        verdicts["P4_injected_error_recall"] = {
            "local": det,
            "arc": results["T3_subtly_wrong"]["injected_error_detection_arc"],
            "verdict": "UNDETERMINED" if r is None else ("PASS" if r > 0 else "FAIL"),
        }

    # P5 ranking stability across alpha.
    #
    # final_tutor_score falls back to D_micro whenever the macro score is None, so with no macro
    # run the alpha term drops out entirely and every alpha yields an identical ordering. That
    # would make ranking_is_stable trivially True -- a vacuous PASS that looks like evidence and
    # is not. So the macro score is required, and its absence is reported as VACUOUS rather than
    # allowed to masquerade as a passing result.
    if have_all:
        macro_by_variant = {v: load_macro_score(v) for v in VARIANTS}
        if any(m is None for m in macro_by_variant.values()):
            verdicts["P5_ranking_stability"] = {
                "verdict": "VACUOUS",
                "macro_scores": macro_by_variant,
                "reason": (
                    "One or more variants has no macro score. final_tutor_score falls back to "
                    "D_micro when M is None, so alpha cannot change any ordering and a 'stable' "
                    "result would carry no information. Run the macro judge on every variant to "
                    "make this prediction testable."
                ),
            }
        else:
            inputs = [
                DialogueScoreInputs(
                    dialogue_id=v,
                    d_segment=results[v]["dialogue_scores"]["d_local_trust_adjusted"],
                    d_arc=results[v]["dialogue_scores"]["d_arc_trust_adjusted"],
                    m=macro_by_variant[v],
                )
                for v in VARIANTS
            ]
            alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            grid = sensitivity_grid(inputs, rho=RHO, alpha_values=alphas)
            stable = ranking_is_stable(grid)
            verdicts["P5_ranking_stability"] = {
                "alpha_values": alphas,
                "macro_scores": macro_by_variant,
                "stable": stable,
                "verdict": "PASS" if stable else "FAIL",
                "grid": {str(k): v for k, v in grid.items()},
            }

    report = {
        "experiment": "tutor_variant_ranking_v1",
        "preregistration": "local_audits/evaluation_completion_20260824T093426Z/22_TUTOR_VARIANT_PREREGISTRATION.md",
        "injected_error_count": len(injected_ids),
        "variants": results,
        "prediction_verdicts": verdicts,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tutor_variant_ranking_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v.get("verdict") for k, v in verdicts.items()}, indent=2))
    for v in results:
        ds = results[v]["dialogue_scores"]
        print(f"{v:20s} trust_adjusted={ds['d_micro_trust_adjusted']} micro_raw={ds['d_micro_raw']}")
    print(f"WROTE {out_dir}")


if __name__ == "__main__":
    main()
