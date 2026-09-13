"""Recompute the Option B final tutor score from frozen Selene artifacts.

This script performs no model calls. It verifies that the segment, topic,
factuality, and macro rows used for the development-condition final-score
readout are Selene-1-Llama-3.3-70B rows, then computes:

    D_micro = rho * D_segment + (1 - rho) * D_topic
    T       = alpha * D_micro + (1 - alpha) * M

where M is adaptability alone. The micro scores use the publication basis for
the development conditions: all ten unit dimensions, equal weights, beta=0, and
the deterministic trust cap at segment scope.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from compute_routed_tutor_scores import (  # noqa: E402
    ARMS,
    FACT_DIR,
    load_factuality,
    load_family,
    n_exchanges,
)
from seg_eval.aggregation.deterministic_aggregation import (  # noqa: E402
    DialogueScoreInputs,
    apply_deterministic_trust_cap,
    combine_local_and_arc,
    exchange_weighted_mean,
    final_tutor_score,
    ranking_is_stable,
    sensitivity_grid,
    weighted_mean_applicable,
)
from seg_eval.aggregation.scope_routing import (  # noqa: E402
    CURRENT_ROUTING,
    arc_dimensions,
    local_dimensions,
)


EXPECTED_MODEL = "Selene-1-Llama-3.3-70B"
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
DIAGNOSTIC_ALPHA = 1.0
BETA_DEVELOPMENT = 0.0
BETA_REPLICATION = 0.2

CONDITIONS = [
    ("T1_strong", "REF"),
    ("T2_answer_dumping", "PED-DEG"),
    ("T3_subtly_wrong", "FACT-CORR"),
    ("T4_macro_degraded", "DLG-DEG"),
    ("T5_expert_register", "EXP-REG"),
]

# Values expected from the frozen v1 macro pass after Round 7b construct
# validation. The script still reads the raw macro rows and fails if they differ.
EXPECTED_ADAPTABILITY = {
    "REF": 1.0,
    "PED-DEG": 0.5,
    "FACT-CORR": 1.0,
    "DLG-DEG": 1.0,
    "EXP-REG": 0.0,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def model_leaf(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def assert_selene_rows(path: Path, label: str) -> dict:
    rows = load_jsonl(path)
    models = sorted({row.get("model_path") for row in rows if row.get("model_path")})
    bad = [model for model in models if model_leaf(model) != EXPECTED_MODEL]
    if bad or not models:
        raise SystemExit(
            f"{label} is not a verified {EXPECTED_MODEL} response file: "
            f"{rel(path)} has model_path values {models!r}"
        )
    return {"label": label, "path": rel(path), "rows": len(rows), "model_path_values": models}


def local_response_path(tag: str) -> Path:
    return REPO / f"data/processed/judge_responses/v35_{tag}_kc_segment_local_selene/judge_responses.jsonl"


def topic_response_path(tag: str) -> Path:
    return REPO / f"data/processed/judge_responses/v35_{tag}_topic_rollup_arc_selene/judge_responses.jsonl"


def factuality_response_path(tag: str) -> Path:
    return REPO / "data/processed/judge_responses" / FACT_DIR[tag] / "grounded_factuality_responses.jsonl"


def macro_response_path(tag: str, stamp: str) -> Path:
    return REPO / f"data/processed/judge_responses/v35_{tag}_macro_{stamp}_selene/macro_judge_responses.jsonl"


def load_adaptability(tag: str, stamp: str, label: str) -> tuple[float, dict]:
    path = macro_response_path(tag, stamp)
    check = assert_selene_rows(path, f"{label} macro/adaptability")
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise SystemExit(f"{label} macro pass should contain one dialogue row, found {len(rows)}")
    try:
        payload = json.loads(rows[0]["raw_response_text"])
    except Exception as exc:  # pragma: no cover - defensive for audit scripts
        raise SystemExit(f"{label} macro row is not parseable JSON: {exc}") from exc
    score = (payload.get("dimension_scores") or {}).get("adaptability")
    if not isinstance(score, (int, float)):
        raise SystemExit(f"{label} macro row has no numeric adaptability score")
    value = float(score)
    expected = EXPECTED_ADAPTABILITY[label]
    if abs(value - expected) > 1e-12:
        raise SystemExit(f"{label} adaptability is {value}, expected frozen value {expected}")
    return value, check


def all_ten_unit_scores(rows: list[dict], factuality: dict | None, cap_segments: bool) -> tuple[list[tuple[float, int]], int]:
    out: list[tuple[float, int]] = []
    capped = 0
    for row in rows:
        judge_output = row["judge_output"]
        score = weighted_mean_applicable(
            judge_output.get("dimension_scores") or {},
            judge_output.get("dimension_applicability") or {},
        )
        if score is None:
            continue
        if cap_segments:
            verdict, material = (None, 0)
            if factuality is not None and row["segment_id"] in factuality:
                verdict, material = factuality[row["segment_id"]]
            cap = apply_deterministic_trust_cap(score, verdict, material)
            capped += int(cap.cap_applied and cap.capped_score < score)
            score = cap.capped_score
        n = max(1, len(row.get("member_exchange_ids") or []))
        out.append((score, n))
    return out, capped


def read_replication_d_micro(rho: float) -> dict:
    path = REPO / "data/processed/publication_final/beta_sensitivity.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if abs(float(row["beta"]) - BETA_REPLICATION) <= 1e-12:
                d_seg_clean = float(row["d_segment_clean"])
                d_seg_corrupt = float(row["d_segment_corrupted"])
                d_topic_clean = float(row["d_topic_clean"])
                d_topic_corrupt = float(row["d_topic_corrupted"])
                clean = combine_local_and_arc(d_seg_clean, d_topic_clean, rho).value
                corrupt = combine_local_and_arc(d_seg_corrupt, d_topic_corrupt, rho).value
                return {
                    "source": rel(path),
                    "beta": BETA_REPLICATION,
                    "rho": rho,
                    "D_micro_clean": clean,
                    "D_micro_corrupted": corrupt,
                    "corrupted_minus_clean": corrupt - clean,
                    "T_status": "not_reported_no_parseable_dialogue_level_M",
                }
    raise SystemExit(f"Could not find beta={BETA_REPLICATION} in {rel(path)}")


def order_for_alpha(rows: list[dict], alpha: float) -> list[str]:
    scored = [(row["condition"], row[f"T_alpha_{alpha:.1f}"]) for row in rows]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [condition for condition, _ in scored]


def main() -> None:
    out_dir = REPO / "data/processed/publication_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_dims = local_dimensions(CURRENT_ROUTING)
    topic_dims = arc_dimensions(CURRENT_ROUTING)
    rho = len(local_dims) / (len(local_dims) + len(topic_dims))

    checks: list[dict] = []
    rows: list[dict] = []
    for arm, label in CONDITIONS:
        tag, stamp = ARMS[arm]
        checks.append(assert_selene_rows(local_response_path(tag), f"{label} segment"))
        checks.append(assert_selene_rows(topic_response_path(tag), f"{label} topic"))
        checks.append(assert_selene_rows(factuality_response_path(tag), f"{label} factuality"))

        adaptability, macro_check = load_adaptability(tag, stamp, label)
        checks.append(macro_check)

        local_rows = load_family(tag, stamp, "local", None)
        topic_rows = load_family(tag, stamp, "arc", "asbuilt")
        factuality, factuality_status = load_factuality(tag, stamp)
        if not local_rows or not topic_rows or factuality is None:
            raise SystemExit(f"{label} is missing segment, topic, or factuality data")

        local_scores, units_capped = all_ten_unit_scores(local_rows, factuality, cap_segments=True)
        topic_scores, _ = all_ten_unit_scores(topic_rows, factuality=None, cap_segments=False)
        d_segment = exchange_weighted_mean(local_scores)
        d_topic = exchange_weighted_mean(topic_scores)
        if d_segment is None or d_topic is None:
            raise SystemExit(f"{label} has no scorable segment or topic units")
        d_micro = combine_local_and_arc(d_segment, d_topic, rho).value

        row = {
            "arm": arm,
            "condition": label,
            "tag": tag,
            "stamp": stamp,
            "n_segment_units": len(local_rows),
            "n_topic_units": len(topic_rows),
            "n_segment_exchanges": n_exchanges(local_rows),
            "n_topic_exchanges": n_exchanges(topic_rows),
            "factuality_accepted": factuality_status["accepted"],
            "factuality_rejected": factuality_status["rejected"],
            "factuality_total": factuality_status["total"],
            "units_capped": units_capped,
            "D_segment": d_segment,
            "D_topic": d_topic,
            "D_micro": d_micro,
            "M_adaptability": adaptability,
        }
        for alpha in ALPHA_GRID + [DIAGNOSTIC_ALPHA]:
            row[f"T_alpha_{alpha:.1f}"] = final_tutor_score(d_micro, adaptability, alpha).value
        rows.append(row)

    inputs = [
        DialogueScoreInputs(
            dialogue_id=row["condition"],
            d_segment=row["D_segment"],
            d_arc=row["D_topic"],
            m=row["M_adaptability"],
        )
        for row in rows
    ]
    grid = sensitivity_grid(inputs, rho=rho, alpha_values=ALPHA_GRID)
    grid_with_endpoint = sensitivity_grid(inputs, rho=rho, alpha_values=ALPHA_GRID + [DIAGNOSTIC_ALPHA])
    orders = {f"{alpha:.1f}": order_for_alpha(rows, alpha) for alpha in ALPHA_GRID + [DIAGNOSTIC_ALPHA]}

    csv_path = out_dir / "option_b_final_tutor_score.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "option_b_final_tutor_score.json"
    payload = {
        "policy": "single_judge_selene_no_new_model_calls",
        "model": EXPECTED_MODEL,
        "basis": {
            "development_beta": BETA_DEVELOPMENT,
            "unit_score": "equal-weight mean of all ten applicable unit dimensions",
            "segment_postprocessing": "deterministic grounded-factuality trust cap",
            "topic_postprocessing": "no criterion blend and no trust cap",
            "macro_M": "adaptability alone",
            "topic_rollup": "asbuilt consecutive-only topic rollup",
        },
        "rho": {
            "value": rho,
            "routing_version": CURRENT_ROUTING,
            "segment_dimensions": local_dims,
            "topic_dimensions": topic_dims,
        },
        "alpha": {
            "reported_grid": ALPHA_GRID,
            "diagnostic_endpoint": DIAGNOSTIC_ALPHA,
            "stable_on_reported_grid": ranking_is_stable(grid),
            "stable_with_micro_only_endpoint": ranking_is_stable(grid_with_endpoint),
            "orders": orders,
        },
        "condition_scores": rows,
        "replication": read_replication_d_micro(rho),
        "source_checks": checks,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Option B final tutor score")
    print(f"rho={rho:.1f} from {len(local_dims)} segment dimensions and {len(topic_dims)} topic dimensions")
    print(f"wrote {rel(csv_path)}")
    print(f"wrote {rel(json_path)}")
    print()
    print("condition,D_segment,D_topic,D_micro,M,T_0.5,T_0.8,T_0.9,T_1.0")
    for row in rows:
        print(
            f"{row['condition']},{row['D_segment']:.4f},{row['D_topic']:.4f},"
            f"{row['D_micro']:.4f},{row['M_adaptability']:.4f},"
            f"{row['T_alpha_0.5']:.4f},{row['T_alpha_0.8']:.4f},"
            f"{row['T_alpha_0.9']:.4f},{row['T_alpha_1.0']:.4f}"
        )
    print()
    print(f"reported alpha grid stable: {ranking_is_stable(grid)}")
    print(f"stable including alpha=1.0 endpoint: {ranking_is_stable(grid_with_endpoint)}")
    for alpha in ALPHA_GRID + [DIAGNOSTIC_ALPHA]:
        print(f"alpha={alpha:.1f}: {' > '.join(orders[f'{alpha:.1f}'])}")
    repl = payload["replication"]
    print()
    print(
        "replication D_micro at beta=0.2: clean={D_micro_clean:.4f}, "
        "corrupted={D_micro_corrupted:.4f}, corrupted-clean={corrupted_minus_clean:.4f}; "
        "T not reported because dialogue-level M is missing".format(**repl)
    )


if __name__ == "__main__":
    main()
