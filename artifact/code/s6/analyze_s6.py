from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = PROJECT_ROOT / "artifacts" / "v12_completed_evidence_20260926"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "v12_select_certify_20260926"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def lower_cvar(values: Sequence[float], fraction: float = 0.20) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    return statistics.mean(ordered[:count])


def safe_mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def origin_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, draws: int, confidence: float, seed: int
) -> dict[str, float]:
    by_origin: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_origin[str(row["origin_id"])].append(float(row["system_gain_points"]))
    origins = sorted(by_origin)
    if not origins:
        return {"estimate": 0.0, "lcb": 0.0, "ucb": 0.0}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = [rng.choice(origins) for _ in origins]
        values = [value for origin in sampled for value in by_origin[origin]]
        estimates.append(statistics.mean(values))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": statistics.mean(float(row["system_gain_points"]) for row in rows),
        "lcb": float(quantile(estimates, alpha)),
        "ucb": float(quantile(estimates, 1.0 - alpha)),
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean, y_mean = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_scale == 0.0 or y_scale == 0.0:
        return None
    return numerator / (x_scale * y_scale)


def stable_rank(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()


def summarize_policy(
    *,
    dataset_id: str,
    policy: str,
    family: str,
    cells: Sequence[Mapping[str, Any]],
    deployed: Mapping[str, str | None],
    draws: int,
    confidence: float,
    bootstrap_seed: int,
    harm_threshold: float,
    severe_threshold: float,
    benefit_threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    served_cell_gains: list[float] = []
    served_bin_gains: list[float] = []
    missed_nominee = 0
    missed_any = 0
    worst_regret = 0.0
    for cell in cells:
        decision_id = str(cell["decision_id"])
        action = deployed.get(decision_id)
        selected = action is not None
        action_gains = cell["action_gains"]
        action_bin_gains = cell["action_bin_gains"]
        candidate_gain = float(action_gains[action]) if selected else 0.0
        system_gain = candidate_gain if selected else 0.0
        bins = [float(value) for value in action_bin_gains[action]] if selected else [0.0] * 4
        if selected:
            served_cell_gains.append(candidate_gain)
            served_bin_gains.extend(bins)
        nominee_gain = float(cell["nominee_T_gain_points"])
        best_gain = max(0.0, max(float(value) for value in action_gains.values()))
        if not selected and nominee_gain > benefit_threshold:
            missed_nominee += 1
        if not selected and best_gain > benefit_threshold:
            missed_any += 1
        worst_regret = max(worst_regret, best_gain - system_gain)
        rows.append({
            "decision_id": decision_id,
            "origin_id": cell["origin_id"],
            "seed": int(cell["seed"]),
            "selected": selected,
            "served_action": action or "calibrated_frozen_base",
            "system_gain_points": system_gain,
        })

    deployed_cells = len(served_cell_gains)
    harmful = sum(value < -harm_threshold for value in served_cell_gains)
    severe = sum(value < -severe_threshold for value in served_cell_gains)
    interval = origin_cluster_bootstrap(
        rows, draws=draws, confidence=confidence, seed=bootstrap_seed
    )
    return {
        "dataset_id": dataset_id,
        "policy": policy,
        "policy_family": family,
        "available_cells": len(cells),
        "deployed_cells": deployed_cells,
        "coverage": deployed_cells / max(len(cells), 1),
        "system_mean_gain_points": statistics.mean(row["system_gain_points"] for row in rows),
        "served_mean_gain_points": safe_mean(served_cell_gains),
        "harmful_action_count": harmful,
        "harmful_action_rate": harmful / max(deployed_cells, 1),
        "severe_harm_count": severe,
        "worst_served_cell_gain_points": min(served_cell_gains) if served_cell_gains else None,
        "served_cell_gain_q05_points": quantile(served_cell_gains, 0.05),
        "lower_cvar20_cell_gain_points": lower_cvar(served_cell_gains),
        "worst_served_bin_gain_points": min(served_bin_gains) if served_bin_gains else None,
        "lower_cvar20_bin_gain_points": lower_cvar(served_bin_gains),
        "missed_beneficial_nominee_count": missed_nominee,
        "missed_any_beneficial_action_count": missed_any,
        "worst_regret_vs_best_counterfactual_points": worst_regret,
        "origin_cluster_bootstrap_system_gain_95": interval,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete frozen v12 select-then-certify experiment.")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    package = args.evidence_root / "nlp_ata_experiments"
    result_root = args.evidence_root / "results_nlp_ata" / "v12_select_certify"
    config_path = package / "configs" / "prospective_select_certify_v12.draft.json"
    action_path = package / "locks" / "prospective_select_certify_v12.actions.lock.json"
    barrier_path = package / "locks" / "prospective_select_certify_v12.pre_T.lock.json"
    execution_path = package / "locks" / "prospective_select_certify_v12.execution.lock.json"
    config, action_lock, barrier = load_json(config_path), load_json(action_path), load_json(barrier_path)
    execution = load_json(execution_path)

    deployment_paths = sorted((result_root / "deployment").glob("*.json"))
    certification_paths = sorted((result_root / "certification").glob("*.json"))
    history_paths = sorted((result_root / "history").glob("*.json"))
    failure_paths = sorted((result_root / "failures").rglob("*.json")) if (result_root / "failures").exists() else []
    if (len(history_paths), len(certification_paths), len(deployment_paths), len(failure_paths)) != (48, 288, 288, 0):
        raise RuntimeError(
            "Incomplete v12 evidence: "
            f"history={len(history_paths)}, certification={len(certification_paths)}, "
            f"deployment={len(deployment_paths)}, failures={len(failure_paths)}"
        )
    if action_lock.get("decision_count") != 48 or len(action_lock.get("decisions", {})) != 48:
        raise RuntimeError("The frozen action lock does not contain all 48 decisions.")
    action_sha, barrier_sha = file_sha256(action_path), file_sha256(barrier_path)
    if barrier.get("action_lock_sha256") != action_sha:
        raise RuntimeError("The pre-T barrier is not bound to the downloaded action lock.")

    deployment: dict[tuple[str, str], dict[str, Any]] = {}
    action_rows: list[dict[str, Any]] = []
    for path in deployment_paths:
        artifact = load_json(path)
        if artifact.get("stage") != "T_opened_after_all_48_actions_locked":
            raise RuntimeError(f"Invalid deployment stage: {path}")
        if artifact.get("action_lock_sha256") != action_sha or artifact.get("global_barrier_sha256") != barrier_sha:
            raise RuntimeError(f"Deployment artifact is not bound to the frozen locks: {path}")
        decision_id = str(artifact["decision_id"])
        action = str(artifact["audited_action"])
        key = (decision_id, action)
        if key in deployment:
            raise RuntimeError(f"Duplicate deployment artifact: {key}")
        deployment[key] = artifact

    datasets = [row["dataset_id"] for row in config["datasets"]]
    actions = list(config["runtime"]["candidates"])
    capacity_order = list(config["primary_policy"]["nomination_tie_break_order_low_to_high_capacity"])
    capacity_rank = {action: index for index, action in enumerate(capacity_order)}
    salt = str(config["global_rules"]["matched_coverage_salt"])
    cells_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_rows: list[dict[str, Any]] = []

    for decision_id, decision in sorted(action_lock["decisions"].items()):
        dataset_id = str(decision["dataset_id"])
        if dataset_id not in datasets:
            raise RuntimeError(f"Unknown dataset in action lock: {dataset_id}")
        action_gains: dict[str, float] = {}
        action_bin_gains: dict[str, list[float]] = {}
        primary_markers = 0
        for action in actions:
            artifact = deployment[(decision_id, action)]
            action_gains[action] = float(artifact["outcomes"]["counterfactual_candidate_gain_points"])
            bin_map = artifact["outcomes"]["counterfactual_candidate_bin_gain_points"]
            action_bin_gains[action] = [float(bin_map[str(index)]) for index in range(4)]
            primary_markers += int(bool(artifact["is_primary_served_action"]))
            c_only = decision["all_six_C_only_artifacts"][action]
            action_rows.append({
                "decision_id": decision_id,
                "dataset_id": dataset_id,
                "origin_id": decision["origin_id"],
                "seed": int(decision["seed"]),
                "action": action,
                "is_nominee": action == decision["nominee_action"],
                "authorized_primary": bool(decision["authorized"] and action == decision["nominee_action"]),
                "select_gain_points": float(c_only["select_gain_points"]),
                "gate_gain_points": float(c_only["gate_gain_points"]),
                "gate_gain_lcb_points": float(c_only["gate_gain_lcb_points"]),
                "T_gain_points": action_gains[action],
                **{f"T_bin{index}_gain_points": action_bin_gains[action][index] for index in range(4)},
            })
        expected_markers = 1 if decision["authorized"] else 0
        if primary_markers != expected_markers:
            raise RuntimeError(f"Primary served marker mismatch for {decision_id}")
        nominee = str(decision["nominee_action"])
        best_action = max(actions, key=lambda name: action_gains[name])
        cell = {
            "decision_id": decision_id,
            "dataset_id": dataset_id,
            "origin_id": str(decision["origin_id"]),
            "seed": int(decision["seed"]),
            "nominee_action": nominee,
            "authorized": bool(decision["authorized"]),
            "select_gain_points": float(decision["select_gain_points"]),
            "gate_gain_points": float(decision["gate_gain_points"]),
            "gate_gain_lcb_points": float(decision["gate_gain_lcb_points"]),
            "nominee_T_gain_points": action_gains[nominee],
            "best_T_action": best_action,
            "best_T_gain_points": action_gains[best_action],
            "worst_T_gain_points": min(action_gains.values()),
            "action_gains": action_gains,
            "action_bin_gains": action_bin_gains,
        }
        cells_by_dataset[dataset_id].append(cell)
        decision_rows.append({key: value for key, value in cell.items() if key not in {"action_gains", "action_bin_gains"}})

    for dataset_id in datasets:
        if len(cells_by_dataset[dataset_id]) != 24:
            raise RuntimeError(f"Expected 24 decisions for {dataset_id}; observed {len(cells_by_dataset[dataset_id])}")

    harm = float(config["global_rules"]["harm_threshold_points"])
    severe = float(config["global_rules"]["severe_harm_threshold_points"])
    benefit = float(config["global_rules"]["benefit_threshold_points"])
    draws = int(config["global_rules"]["bootstrap_draws"])
    confidence = float(config["global_rules"]["bootstrap_confidence"])
    summaries: list[dict[str, Any]] = []
    matched_names = (
        "hash_random_matched_to_primary_coverage",
        "C_select_gain_rank_matched_to_primary_coverage",
        "parameter_count_rank_matched_to_primary_coverage",
    )

    for dataset_index, dataset_id in enumerate(datasets):
        cells = sorted(cells_by_dataset[dataset_id], key=lambda row: row["decision_id"])
        primary = {
            cell["decision_id"]: cell["nominee_action"] if cell["authorized"] else None
            for cell in cells
        }
        common = dict(
            dataset_id=dataset_id,
            cells=cells,
            draws=draws,
            confidence=confidence,
            harm_threshold=harm,
            severe_threshold=severe,
            benefit_threshold=benefit,
        )
        summaries.append(summarize_policy(
            policy="split_select_then_certify", family="primary", deployed=primary,
            bootstrap_seed=1200 + dataset_index, **common,
        ))
        summaries.append(summarize_policy(
            policy="calibrated_frozen_base", family="control", deployed={cell["decision_id"]: None for cell in cells},
            bootstrap_seed=1300 + dataset_index, **common,
        ))
        summaries.append(summarize_policy(
            policy="always_deploy_C_select_nominee", family="control", deployed={cell["decision_id"]: cell["nominee_action"] for cell in cells},
            bootstrap_seed=1400 + dataset_index, **common,
        ))
        for action_index, action in enumerate(actions):
            summaries.append(summarize_policy(
                policy=f"always_adapt::{action}", family="always_adapt",
                deployed={cell["decision_id"]: action for cell in cells},
                bootstrap_seed=2000 + 20 * dataset_index + action_index, **common,
            ))
            summaries.append(summarize_policy(
                policy=f"single_action_gain_lcb_gate::{action}", family="single_action_gate",
                deployed={
                    cell["decision_id"]: action
                    if float(action_lock["decisions"][cell["decision_id"]]["all_six_C_only_artifacts"][action]["gate_gain_lcb_points"]) > 0.0
                    else None
                    for cell in cells
                },
                bootstrap_seed=3000 + 20 * dataset_index + action_index, **common,
            ))

        target = sum(cell["authorized"] for cell in cells)
        matched_orders = {
            matched_names[0]: sorted(cells, key=lambda row: (stable_rank(row["decision_id"], salt), row["decision_id"])),
            matched_names[1]: sorted(cells, key=lambda row: (-float(row["select_gain_points"]), stable_rank(row["decision_id"], salt))),
            matched_names[2]: sorted(cells, key=lambda row: (capacity_rank[row["nominee_action"]], stable_rank(row["decision_id"], salt))),
        }
        for control_index, (name, ordered) in enumerate(matched_orders.items()):
            selected = {row["decision_id"] for row in ordered[:target]}
            summaries.append(summarize_policy(
                policy=name,
                family="matched_coverage",
                deployed={cell["decision_id"]: cell["nominee_action"] if cell["decision_id"] in selected else None for cell in cells},
                bootstrap_seed=4000 + 20 * dataset_index + control_index, **common,
            ))

    transport: list[dict[str, Any]] = []
    for dataset_id in datasets:
        cells = cells_by_dataset[dataset_id]
        authorized = [cell for cell in cells if cell["authorized"]]
        transport.append({
            "dataset_id": dataset_id,
            "decisions": len(cells),
            "authorized": len(authorized),
            "nominee_action_counts": dict(Counter(cell["nominee_action"] for cell in cells)),
            "authorized_action_counts": dict(Counter(cell["nominee_action"] for cell in authorized)),
            "authorized_positive_T": sum(cell["nominee_T_gain_points"] > 0.0 for cell in authorized),
            "authorized_harmful_T": sum(cell["nominee_T_gain_points"] < -harm for cell in authorized),
            "gate_gain_vs_T_gain_pearson": pearson(
                [float(cell["gate_gain_points"]) for cell in cells],
                [float(cell["nominee_T_gain_points"]) for cell in cells],
            ),
            "gate_lcb_vs_T_gain_pearson": pearson(
                [float(cell["gate_gain_lcb_points"]) for cell in cells],
                [float(cell["nominee_T_gain_points"]) for cell in cells],
            ),
        })

    verdicts: list[dict[str, Any]] = []
    full_tune_comparisons: list[dict[str, Any]] = []
    for dataset_id in datasets:
        primary = next(row for row in summaries if row["dataset_id"] == dataset_id and row["policy_family"] == "primary")
        matched = [row for row in summaries if row["dataset_id"] == dataset_id and row["policy_family"] == "matched_coverage"]
        comparisons = []
        for control in matched:
            comparisons.append({
                "control": control["policy"],
                "cell_worst_better": primary["worst_served_cell_gain_points"] > control["worst_served_cell_gain_points"],
                "cell_cvar20_better": primary["lower_cvar20_cell_gain_points"] > control["lower_cvar20_cell_gain_points"],
                "bin_worst_better": primary["worst_served_bin_gain_points"] > control["worst_served_bin_gain_points"],
                "bin_cvar20_better": primary["lower_cvar20_bin_gain_points"] > control["lower_cvar20_bin_gain_points"],
            })
        core_checks = {
            "coverage_at_least_25pct": primary["coverage"] >= float(config["success_rule"]["minimum_useful_coverage"]),
            "worst_served_cell_at_least_minus_1pp": primary["worst_served_cell_gain_points"] is not None and primary["worst_served_cell_gain_points"] >= float(config["success_rule"]["target_worst_served_gain_points"]),
            "origin_cluster_gain_lcb_positive": primary["origin_cluster_bootstrap_system_gain_95"]["lcb"] > float(config["success_rule"]["origin_cluster_bootstrap_system_gain_LCB_must_exceed_points"]),
        }
        strict_cell_tail = all(row["cell_worst_better"] and row["cell_cvar20_better"] for row in comparisons)
        worst_cell_tail = all(row["cell_worst_better"] for row in comparisons)
        any_registered_tail = all(
            any(row[key] for key in ("cell_worst_better", "cell_cvar20_better", "bin_worst_better", "bin_cvar20_better"))
            for row in comparisons
        )
        verdicts.append({
            "dataset_id": dataset_id,
            "core_checks": core_checks,
            "core_pass": all(core_checks.values()),
            "matched_control_comparisons": comparisons,
            "strict_cell_tail_dominance": strict_cell_tail,
            "worst_cell_tail_dominance": worst_cell_tail,
            "any_registered_tail_improvement_against_each_control": any_registered_tail,
            "strict_dataset_pass": all(core_checks.values()) and strict_cell_tail,
            "worst_cell_only_dataset_pass": all(core_checks.values()) and worst_cell_tail,
            "any_registered_tail_dataset_pass": all(core_checks.values()) and any_registered_tail,
        })

        full_tune = next(
            row for row in summaries
            if row["dataset_id"] == dataset_id and row["policy"] == "single_action_gain_lcb_gate::full_tune"
        )
        fields = (
            "coverage", "system_mean_gain_points", "served_mean_gain_points",
            "worst_served_cell_gain_points", "lower_cvar20_cell_gain_points",
        )
        equal = all(abs(float(primary[field]) - float(full_tune[field])) < 1e-12 for field in fields)
        full_tune_nonworse = all(float(full_tune[field]) >= float(primary[field]) for field in fields)
        full_tune_strict = any(float(full_tune[field]) > float(primary[field]) + 1e-12 for field in fields)
        full_tune_comparisons.append({
            "dataset_id": dataset_id,
            "primary": {field: primary[field] for field in fields},
            "single_action_full_tune_gate": {field: full_tune[field] for field in fields},
            "identical_on_registered_comparison_fields": equal,
            "full_tune_gate_dominates_primary": full_tune_nonworse and full_tune_strict,
        })

    strict_pass_count = sum(row["strict_dataset_pass"] for row in verdicts)
    any_tail_pass_count = sum(row["any_registered_tail_dataset_pass"] for row in verdicts)
    strict_operationalization = "GO" if strict_pass_count == 2 else "NO_GO"
    any_tail_operationalization = "GO" if any_tail_pass_count == 2 else "NO_GO"
    # The preregistration named worst gain, q05 and CVaR endpoints, but did not
    # state whether matched-control improvement meant AND/OR or cell/bin tails.
    # T has now been opened, so this analysis must expose the ambiguity rather
    # than choose the favorable interpretation retrospectively.
    preregistered_label = (
        strict_operationalization
        if strict_operationalization == any_tail_operationalization
        else "INDETERMINATE_DUE_TO_UNDERSPECIFIED_TAIL_DOMINANCE"
    )
    portfolio_increment = (
        "NOT_SUPPORTED_OVER_SINGLE_ACTION_FULL_TUNE_GATE"
        if all(row["identical_on_registered_comparison_fields"] or row["full_tune_gate_dominates_primary"] for row in full_tune_comparisons)
        else "NOT_RESOLVED"
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "ata-nlp-select-certify-analysis-12.0",
        "analysis_status": "complete_post_T_locked_evidence_audit",
        "evidence_root": str(args.evidence_root.resolve()),
        "input_hashes": {
            "config": file_sha256(config_path),
            "execution_lock": file_sha256(execution_path),
            "action_lock": action_sha,
            "pre_T_barrier": barrier_sha,
        },
        "integrity": {
            "history": len(history_paths),
            "certification": len(certification_paths),
            "deployment": len(deployment_paths),
            "failures": len(failure_paths),
            "action_decisions": action_lock["decision_count"],
            "execution_status": execution.get("status"),
            "action_status": action_lock.get("status"),
            "barrier_status": barrier.get("status"),
        },
        "matched_control_definition": (
            "Each matched control serves the frozen C_select nominee in exactly the primary policy's dataset-specific "
            "number of decision cells. Cells are ranked using only the frozen hash, nominee C_select gain, or the "
            "registered low-to-high nominee capacity order; deployment outcomes never enter selection."
        ),
        "policy_summaries": summaries,
        "transport_diagnostics": transport,
        "success_rule_audit": verdicts,
        "success_rule_interpretations": {
            "strict_cell_worst_AND_cvar": strict_operationalization,
            "any_registered_cell_or_bin_tail_metric": any_tail_operationalization,
            "preregistered_label": preregistered_label,
            "reason": "The frozen text did not define the logical operator or aggregation unit for matched-tail improvement.",
        },
        "full_tune_gate_comparison": full_tune_comparisons,
        "portfolio_algorithm_increment": portfolio_increment,
    }
    (args.output_root / "v12_complete_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary_csv: list[dict[str, Any]] = []
    for row in summaries:
        flat = {key: value for key, value in row.items() if key != "origin_cluster_bootstrap_system_gain_95"}
        interval = row["origin_cluster_bootstrap_system_gain_95"]
        flat.update({
            "system_gain_bootstrap_lcb": interval["lcb"],
            "system_gain_bootstrap_ucb": interval["ucb"],
        })
        summary_csv.append(flat)
    write_csv(args.output_root / "v12_policy_summary.csv", summary_csv)
    write_csv(args.output_root / "v12_decision_level.csv", decision_rows)
    write_csv(args.output_root / "v12_action_level.csv", action_rows)

    lines = [
        "# V12 frozen select-then-certify audit",
        "",
        f"Preregistered success label: **{preregistered_label}**.",
        f"Portfolio increment: **{portfolio_increment}**.",
        "",
        "All 48 history jobs, 288 certification jobs, and 288 deployment audits are present; no failure artifact exists. "
        "Every deployment artifact is cryptographically bound to the downloaded action lock and pre-T barrier.",
        "",
        "## Primary policy",
        "",
        "| Dataset | Coverage | System mean | Served mean | Harmful | Worst cell | Cell CVaR20 | Worst bin | Origin LCB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in datasets:
        row = next(value for value in summaries if value["dataset_id"] == dataset_id and value["policy_family"] == "primary")
        lines.append(
            f"| {dataset_id} | {100 * row['coverage']:.1f}% | {fmt(row['system_mean_gain_points'])} | "
            f"{fmt(row['served_mean_gain_points'])} | {row['harmful_action_count']} | "
            f"{fmt(row['worst_served_cell_gain_points'])} | {fmt(row['lower_cvar20_cell_gain_points'])} | "
            f"{fmt(row['worst_served_bin_gain_points'])} | "
            f"{fmt(row['origin_cluster_bootstrap_system_gain_95']['lcb'])} |"
        )
    lines.extend(["", "## Success-rule audit", ""])
    for verdict in verdicts:
        lines.append(f"### {verdict['dataset_id']}")
        lines.append("")
        for key, value in verdict["core_checks"].items():
            lines.append(f"- {'PASS' if value else 'FAIL'}: `{key}`")
        lines.append(f"- {'PASS' if verdict['strict_cell_tail_dominance'] else 'FAIL'}: `strict_cell_worst_AND_cvar_dominance`")
        lines.append(f"- {'PASS' if verdict['any_registered_tail_improvement_against_each_control'] else 'FAIL'}: `any_registered_tail_metric_against_each_control`")
        lines.append("")
    lines.extend([
        "## Fixed full-tune gate comparator",
        "",
        "| Dataset | Relation to portfolio | Portfolio coverage | Full-tune coverage | Portfolio system gain | Full-tune system gain | Portfolio worst | Full-tune worst |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for comparison in full_tune_comparisons:
        primary = comparison["primary"]
        full_tune = comparison["single_action_full_tune_gate"]
        relation = "identical" if comparison["identical_on_registered_comparison_fields"] else "full-tune dominates" if comparison["full_tune_gate_dominates_primary"] else "mixed"
        lines.append(
            f"| {comparison['dataset_id']} | {relation} | {100 * primary['coverage']:.1f}% | {100 * full_tune['coverage']:.1f}% | "
            f"{fmt(primary['system_mean_gain_points'])} | {fmt(full_tune['system_mean_gain_points'])} | "
            f"{fmt(primary['worst_served_cell_gain_points'])} | {fmt(full_tune['worst_served_cell_gain_points'])} |"
        )
    lines.extend([
        "## Interpretation boundary",
        "",
        "Matched controls preserve dataset-specific coverage and serve the already frozen C_select nominee; only the "
        "choice of decision cells changes. The parameter-count control uses the registered low-to-high capacity order. "
        "This operationalizes the preregistered control names without consulting T. The report presents cell-level and "
        "bin-level tails separately and does not treat seeds or bins as independent datasets. The frozen success text "
        "did not specify whether matched-tail improvement required all endpoints or any endpoint, nor whether the unit "
        "was a served cell or a temporal bin. Because T is open, the report shows both operationalizations and refuses "
        "to select one retrospectively.",
        "",
    ])
    (args.output_root / "V12_COMPLETE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output_root": str(args.output_root),
        "preregistered_label": preregistered_label,
        "portfolio_increment": portfolio_increment,
        "datasets": verdicts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
