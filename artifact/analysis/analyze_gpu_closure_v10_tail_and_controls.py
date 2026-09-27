from __future__ import annotations

"""Tail-risk and frozen-base calibration audit for GPU closure v10.

The script is deliberately read-only with respect to ``results_nlp_ata``.  It
reads the frozen manifest and completed JSON artifacts, computes action-
conditional and served-population tails, and writes new analysis artifacts to
``nlp_ata_experiments/analysis``.  When immutable base-logit caches are present,
it also evaluates two post-outcome diagnostic controls on MultiEURLEX:

* frozen base + a global threshold fitted on CERT-FIT;
* frozen base + per-label thresholds fitted on CERT-FIT.

Those controls are diagnostic rather than confirmatory because deployment
outcomes had already been opened before this analysis was specified.
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ata_nlp.closure_v10 import (  # noqa: E402
    fit_global_threshold_by_macro_f1,
    load_json,
    task_metrics,
)


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "gpu_closure_v10.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "gpu_closure_v10.draft.json"
HARM_THRESHOLD_POINTS = -0.5
MATERIAL_BENEFIT_POINTS = 0.5


def result_path(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    root = PROJECT_ROOT / config["paths"]["result_root"]
    if "history_train" in job["phase"]:
        return root / "history" / f"{job['job_id']}.json"
    return root / "jobs" / job["suite_id"] / f"{job['job_id']}.json"


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cvar_lower(values: Sequence[float], fraction: float = 0.2) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return statistics.fmean(ordered[:count])


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "minimum": None,
            "q05": None,
            "median": None,
            "q95": None,
            "maximum": None,
            "lower_cvar20": None,
        }
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "minimum": min(clean),
        "q05": quantile(clean, 0.05),
        "median": quantile(clean, 0.5),
        "q95": quantile(clean, 0.95),
        "maximum": max(clean),
        "lower_cvar20": cvar_lower(clean, 0.2),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    value = quantile(values, probability)
    if value is None:
        raise ValueError("A percentile requires at least one value")
    return value


def clustered_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int | None]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            clusters[str(row["origin_id"])].append(float(row[field]))
    if not clusters:
        return {"clusters": 0, "draws": 0, "estimate": None, "ci95_low": None, "ci95_high": None}
    origin_ids = sorted(clusters)
    estimate = statistics.fmean(value for origin in origin_ids for value in clusters[origin])
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected = [rng.choice(origin_ids) for _ in origin_ids]
        values = [value for origin in selected for value in clusters[origin]]
        samples.append(statistics.fmean(values))
    return {
        "clusters": len(origin_ids),
        "draws": draws,
        "estimate": estimate,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "resampling_unit": "origin_with_all_nested_seeds_carried_together",
    }


def cell_from_artifact(job: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any] | None:
    phase = job["phase"]
    if phase == "posthoc_candidate_train_and_eval":
        policy = artifact["policy"]
        authorized = bool(policy["authorized"])
        action_gain = float(policy["raw_T_gain_points"])
        served_gain = float(policy["served_T_gain_points"])
        return {
            "suite_id": job["suite_id"],
            "dataset_id": job["dataset_id"],
            "action": job["action"],
            "origin_id": job["origin_id"],
            "seed": int(job["seed"]),
            "backbone": job["backbone"],
            "phase": phase,
            "evidence_role": job["evidence_role"],
            "variant_axis": job.get("variant", {}).get("axis", "reference"),
            "variant_value": job.get("variant", {}).get("value", "reference"),
            "cert_gain_points": float(policy["cert_gain_points"]),
            "raw_gain_points": action_gain,
            "action_gain_points": action_gain,
            "served_gain_points": served_gain,
            "authorized": authorized,
        }
    if phase == "prospective_deploy_after_group_lock":
        policy = artifact["policy"]
        authorized = bool(policy["authorized_from_cert_gate"])
        action_gain = float(policy["calibrated_candidate_gain_points"])
        served_gain = float(policy["served_gain_points"])
        return {
            "suite_id": job["suite_id"],
            "dataset_id": job["dataset_id"],
            "action": job["action"],
            "origin_id": job["origin_id"],
            "seed": int(job["seed"]),
            "backbone": job["backbone"],
            "phase": phase,
            "evidence_role": job["evidence_role"],
            "variant_axis": "prospective_primary_policy",
            "variant_value": "scalar_plus_threshold",
            "cert_gain_points": None,
            "raw_gain_points": float(policy["raw_candidate_gain_points"]),
            "action_gain_points": action_gain,
            "served_gain_points": served_gain,
            "authorized": authorized,
        }
    return None


def tail_summary(rows: Sequence[Mapping[str, Any]], *, bootstrap_draws: int, seed: int) -> dict[str, Any]:
    served = [float(row["served_gain_points"]) for row in rows]
    deployed = [row for row in rows if bool(row["authorized"])]
    abstained = [row for row in rows if not bool(row["authorized"])]
    deployed_gains = [float(row["action_gain_points"]) for row in deployed]
    raw = [float(row["raw_gain_points"]) for row in rows]
    regret = [max(0.0, float(row["action_gain_points"])) - float(row["served_gain_points"]) for row in rows]
    paired_selection_lift = [float(row["served_gain_points"]) - float(row["raw_gain_points"]) for row in rows]
    return {
        "cells": len(rows),
        "coverage": len(deployed) / len(rows) if rows else None,
        "deployments": len(deployed),
        "abstentions": len(abstained),
        "raw_candidate_population": summarize_values(raw),
        "served_population_including_zero_for_abstention": summarize_values(served),
        "deployed_action_conditional": summarize_values(deployed_gains),
        "negative_deployments_below_zero": sum(value < 0.0 for value in deployed_gains),
        "mild_negative_deployments_minus_0_5_to_zero": sum(HARM_THRESHOLD_POINTS <= value < 0.0 for value in deployed_gains),
        "harmful_deployments_below_minus_0_5": sum(value < HARM_THRESHOLD_POINTS for value in deployed_gains),
        "missed_positive_actions_among_abstentions": sum(float(row["action_gain_points"]) > 0.0 for row in abstained),
        "missed_material_benefit_above_plus_0_5": sum(float(row["action_gain_points"]) > MATERIAL_BENEFIT_POINTS for row in abstained),
        "regret_against_cellwise_oracle_adapt_or_base": summarize_values(regret),
        "paired_selection_lift_served_minus_raw": summarize_values(paired_selection_lift),
        "origin_clustered_bootstrap": {
            "raw_gain": clustered_bootstrap(rows, "raw_gain_points", draws=bootstrap_draws, seed=seed + 1),
            "served_gain": clustered_bootstrap(rows, "served_gain_points", draws=bootstrap_draws, seed=seed + 2),
            "selection_lift": clustered_bootstrap(
                [dict(row, selection_lift=value) for row, value in zip(rows, paired_selection_lift)],
                "selection_lift",
                draws=bootstrap_draws,
                seed=seed + 3,
            ),
        },
    }


def group_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    output: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(str(row[key]) for key in keys)].append(row)
    return dict(output)


def load_cache_state(root: Path, key: str):
    import torch

    path = root / "immutable_base_cache" / f"{key}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def fit_per_label_thresholds(logits, labels) -> list[float]:
    from sklearn.metrics import f1_score
    import torch

    probabilities = torch.sigmoid(logits.float()).cpu().numpy()
    gold = labels.cpu().numpy()
    thresholds: list[float] = []
    for label_index in range(probabilities.shape[1]):
        best: tuple[float, float, float] | None = None
        for grid_index in range(19):
            threshold = 0.05 * (grid_index + 1)
            predicted = (probabilities[:, label_index] >= threshold).astype(int)
            score = float(f1_score(gold[:, label_index], predicted, zero_division=0))
            candidate = (score, -abs(threshold - 0.5), threshold)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        thresholds.append(float(best[2]))
    return thresholds


def macro_f1(logits, labels, *, threshold: float | Sequence[float]) -> float:
    return float(
        task_metrics(
            logits,
            labels,
            task_type="multi_label_sequence_classification",
            threshold=threshold,
        )["macro_f1"]
    )


def base_only_calibration_controls(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    result_root = PROJECT_ROOT / config["paths"]["result_root"]
    prospective_candidates = [
        job for job in manifest["jobs"] if job["phase"] == "prospective_candidate_train_cert_fit_cert_gate"
    ]
    prospective_deployments = [
        job for job in manifest["jobs"] if job["phase"] == "prospective_deploy_after_group_lock"
    ]
    candidate_by_cell: dict[tuple[str, int], Mapping[str, Any]] = {}
    for job in prospective_candidates:
        key = (str(job["origin_id"]), int(job["seed"]))
        candidate_by_cell.setdefault(key, artifacts[job["job_id"]])
    deploy_by_cell: dict[tuple[str, int], Mapping[str, Any]] = {}
    for job in prospective_deployments:
        key = (str(job["origin_id"]), int(job["seed"]))
        deploy_by_cell.setdefault(key, artifacts[job["job_id"]])

    controls: dict[str, list[dict[str, Any]]] = {"global_threshold": [], "per_label_thresholds": []}
    for cell_key in sorted(candidate_by_cell):
        candidate_artifact = candidate_by_cell[cell_key]
        deploy_artifact = deploy_by_cell[cell_key]
        fit_key = candidate_artifact["inference"]["cert_fit"]["base"]["cache_key"]
        gate_key = candidate_artifact["inference"]["cert_gate"]["base"]["cache_key"]
        deploy_key = deploy_artifact["runtime"]["base"]["cache_key"]
        fit_state = load_cache_state(result_root, fit_key)
        gate_state = load_cache_state(result_root, gate_key)
        deploy_state = load_cache_state(result_root, deploy_key)
        native_gate = macro_f1(gate_state["logits"], gate_state["labels"], threshold=0.5)
        native_deploy = macro_f1(deploy_state["logits"], deploy_state["labels"], threshold=0.5)

        global_fit = fit_global_threshold_by_macro_f1(
            fit_state["logits"],
            fit_state["labels"],
            task_type="multi_label_sequence_classification",
        )
        per_label = fit_per_label_thresholds(fit_state["logits"], fit_state["labels"])
        for name, threshold in (("global_threshold", float(global_fit["threshold"])), ("per_label_thresholds", per_label)):
            gate_score = macro_f1(gate_state["logits"], gate_state["labels"], threshold=threshold)
            deploy_score = macro_f1(deploy_state["logits"], deploy_state["labels"], threshold=threshold)
            authorized = gate_score > native_gate
            action_gain = 100.0 * (deploy_score - native_deploy)
            controls[name].append(
                {
                    "suite_id": "S5_postoutcome_frozen_base_calibration_control",
                    "dataset_id": "multieurlex_en_level1_temporal_external",
                    "action": f"frozen_base_plus_{name}",
                    "origin_id": cell_key[0],
                    "seed": cell_key[1],
                    "backbone": "roberta-base",
                    "phase": "postoutcome_diagnostic_control",
                    "evidence_role": "postoutcome_diagnostic_not_confirmatory",
                    "variant_axis": "base_only_calibration",
                    "variant_value": name,
                    "cert_gain_points": 100.0 * (gate_score - native_gate),
                    "raw_gain_points": 0.0,
                    "action_gain_points": action_gain,
                    "served_gain_points": action_gain if authorized else 0.0,
                    "authorized": authorized,
                    "threshold": threshold,
                }
            )
    return {
        name: {
            "cells": rows,
            "summary": tail_summary(rows, bootstrap_draws=bootstrap_draws, seed=seed + index * 101),
        }
        for index, (name, rows) in enumerate(controls.items())
    }


def flatten_summary_rows(group_name: str, key: tuple[str, ...], summary: Mapping[str, Any]) -> dict[str, Any]:
    conditional = summary["deployed_action_conditional"]
    population = summary["served_population_including_zero_for_abstention"]
    regret = summary["regret_against_cellwise_oracle_adapt_or_base"]
    return {
        "grouping": group_name,
        "group_key": "::".join(key),
        "cells": summary["cells"],
        "coverage": summary["coverage"],
        "deployments": summary["deployments"],
        "served_mean": population["mean"],
        "deployed_min": conditional["minimum"],
        "deployed_q05": conditional["q05"],
        "deployed_cvar20": conditional["lower_cvar20"],
        "negative_deployments": summary["negative_deployments_below_zero"],
        "mild_negative_deployments": summary["mild_negative_deployments_minus_0_5_to_zero"],
        "harmful_deployments": summary["harmful_deployments_below_minus_0_5"],
        "missed_positive": summary["missed_positive_actions_among_abstentions"],
        "missed_above_plus_0_5": summary["missed_material_benefit_above_plus_0_5"],
        "worst_regret": regret["maximum"],
        "mean_regret": regret["mean"],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["grouping", "group_key"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# GPU Closure v10: tail-risk and key-control audit",
        "",
        "All gains are macro-F1 percentage points. The dataset is the independent generalization unit; origin and seed are nested repeated measurements. Bootstrap intervals resample origins and carry all nested seeds together.",
        "",
        "## Tail-risk table",
        "",
        "| Group | Cells | Coverage | Served mean | Deployed min | Deployed q05 | Lower CVaR20 | Mild negative | Harmful < -0.5 | Missed > +0.5 | Worst regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["flat_tail_table"]:
        lines.append(
            "| {group_key} | {cells} | {coverage:.1%} | {served_mean} | {deployed_min} | {deployed_q05} | {deployed_cvar20} | {mild_negative_deployments} | {harmful_deployments} | {missed_above_plus_0_5} | {worst_regret} |".format(
                group_key=row["group_key"],
                cells=row["cells"],
                coverage=float(row["coverage"]),
                served_mean=fmt(row["served_mean"]),
                deployed_min=fmt(row["deployed_min"]),
                deployed_q05=fmt(row["deployed_q05"]),
                deployed_cvar20=fmt(row["deployed_cvar20"]),
                mild_negative_deployments=row["mild_negative_deployments"],
                harmful_deployments=row["harmful_deployments"],
                missed_above_plus_0_5=row["missed_above_plus_0_5"],
                worst_regret=fmt(row["worst_regret"]),
            )
        )
    lines += [
        "",
        "## Frozen-base calibration controls",
        "",
        "These controls were specified after MultiEURLEX deployment outcomes were opened. They diagnose attribution; they are not a second prospective confirmation.",
        "",
    ]
    controls = payload.get("base_only_calibration_controls")
    if not controls:
        lines.append("Immutable base-logit caches were unavailable; no control result was fabricated.")
    else:
        lines += [
            "| Control | Coverage | Served mean | Deployed min | Harmful |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, block in controls.items():
            summary = block["summary"]
            lines.append(
                f"| {name} | {summary['coverage']:.1%} | {fmt(summary['served_population_including_zero_for_abstention']['mean'])} | {fmt(summary['deployed_action_conditional']['minimum'])} | {summary['harmful_deployments_below_minus_0_5']} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- A zero harmful count at the preregistered -0.5 pp threshold does not imply that every deployment is non-negative; inspect the mild-negative column and the deployed minimum.",
        "- Cell-level bootstrap intervals describe repeated origins and seeds. They do not turn one dataset into multiple independent external replications.",
        "- S1--S4 and the frozen-base controls are post-outcome diagnostics. S5 remains the only newly locked prospective dataset in v10.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only tail and calibration-control audit for GPU closure v10.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--skip-base-controls", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    manifest = load_json(args.manifest)
    artifacts: dict[str, Mapping[str, Any]] = {}
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        path = result_path(config, job)
        if not path.exists():
            if "history_train" not in job["phase"]:
                missing.append(job["job_id"])
            continue
        artifact = load_json(path)
        artifacts[job["job_id"]] = artifact
        cell = cell_from_artifact(job, artifact)
        if cell is not None:
            rows.append(cell)
    if missing:
        raise FileNotFoundError(f"Refusing partial tail analysis; {len(missing)} non-history artifacts are missing")

    group_specs = {
        "suite": ("suite_id",),
        "suite_dataset": ("suite_id", "dataset_id"),
        "suite_dataset_action": ("suite_id", "dataset_id", "action"),
    }
    grouped: dict[str, dict[str, Any]] = {}
    flat: list[dict[str, Any]] = []
    for group_index, (name, keys) in enumerate(group_specs.items()):
        blocks = group_rows(rows, keys)
        grouped[name] = {}
        for key_index, (key, values) in enumerate(sorted(blocks.items())):
            summary = tail_summary(
                values,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed + group_index * 10_000 + key_index * 17,
            )
            grouped[name]["::".join(key)] = summary
            if name == "suite_dataset":
                flat.append(flatten_summary_rows(name, key, summary))

    s4_rows = [row for row in rows if row["suite_id"] == "S4_budget_and_geometry_sensitivity"]
    s4_groups = group_rows(s4_rows, ("dataset_id", "variant_axis", "variant_value"))
    s4_variant = {
        "::".join(key): tail_summary(values, bootstrap_draws=args.bootstrap_draws, seed=args.seed + 50_000 + index)
        for index, (key, values) in enumerate(sorted(s4_groups.items()))
    }

    controls = None
    control_error = None
    if not args.skip_base_controls:
        try:
            controls = base_only_calibration_controls(
                config,
                manifest,
                artifacts,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed + 70_000,
            )
        except (FileNotFoundError, KeyError, ImportError) as exc:
            control_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "schema_version": "ata-nlp-gpu-closure-tail-control-audit-10.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_read_only_analysis",
        "cell_count": len(rows),
        "harm_threshold_points": HARM_THRESHOLD_POINTS,
        "material_benefit_threshold_points": MATERIAL_BENEFIT_POINTS,
        "independent_generalization_unit": "dataset",
        "nested_repeated_units": ["origin", "seed", "deployment_bin"],
        "grouped_tail_summaries": grouped,
        "s4_variant_tail_summaries": s4_variant,
        "base_only_calibration_controls": controls,
        "base_only_control_error": control_error,
        "flat_tail_table": flat,
        "interpretation_constraints": [
            "A harmful count uses the preregistered -0.5 pp threshold and is not equivalent to a count of all negative deployments.",
            "Origin-clustered intervals are descriptive within a dataset; the dataset remains the independent generalization unit.",
            "S1-S4 and frozen-base calibration controls are post-outcome diagnostics; S5 is the only new prospective dataset in v10.",
        ],
    }

    if args.output_prefix is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_prefix = PACKAGE_ROOT / "analysis" / f"gpu_closure_v10_tail_control_{stamp}"
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, flat)
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path), "cells": len(rows), "base_controls": controls is not None}, indent=2))


if __name__ == "__main__":
    main()
