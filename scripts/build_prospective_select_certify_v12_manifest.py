from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def make_job(**fields: Any) -> dict[str, Any]:
    row = dict(fields)
    identity = {key: value for key, value in row.items() if key != "dependencies"}
    row["job_id"] = digest(identity)[:16]
    row.setdefault("dependencies", [])
    row["gpu_required"] = True
    row["result_overwrite"] = False
    return row


def build(config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(value) for value in config["global_rules"]["seeds"]]
    origins = [str(row["origin_id"]) for row in config["rolling_origins"]]
    actions = list(config["runtime"]["candidates"])
    jobs: list[dict[str, Any]] = []
    history_ids: dict[tuple[str, str, int], str] = {}
    candidate_ids: dict[tuple[str, str, int], dict[str, str]] = {}

    for dataset in config["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        backbone = str(dataset["backbone"])
        for origin_id in origins:
            for seed in seeds:
                history = make_job(
                    phase="v12_history_train",
                    dataset_id=dataset_id,
                    origin_id=origin_id,
                    seed=seed,
                    backbone=backbone,
                    evidence_role="prospective_untouched_history",
                )
                history_ids[(dataset_id, origin_id, seed)] = history["job_id"]
                jobs.append(history)

        for origin_id in origins:
            for seed in seeds:
                key = (dataset_id, origin_id, seed)
                candidate_ids[key] = {}
                for action in actions:
                    candidate = make_job(
                        phase="v12_candidate_train_calibrate_select_certify",
                        dataset_id=dataset_id,
                        origin_id=origin_id,
                        seed=seed,
                        backbone=backbone,
                        action=action,
                        must_not_load_deploy=True,
                        evidence_role="C_cal_C_select_C_gate_only",
                        dependencies=[history_ids[key]],
                    )
                    candidate_ids[key][action] = candidate["job_id"]
                    jobs.append(candidate)

        for origin_id in origins:
            for seed in seeds:
                key = (dataset_id, origin_id, seed)
                for action in actions:
                    deploy = make_job(
                        phase="v12_counterfactual_deploy_audit_after_global_barrier",
                        dataset_id=dataset_id,
                        origin_id=origin_id,
                        seed=seed,
                        backbone=backbone,
                        action=action,
                        deploy_may_be_loaded_only_in_this_phase=True,
                        evidence_role="T_counterfactual_audit_with_one_prelocked_served_action",
                        dependencies=[
                            candidate_ids[key][action],
                            f"action_lock::{dataset_id}::{origin_id}::{seed}",
                            "global_pre_T_barrier::v12",
                        ],
                    )
                    jobs.append(deploy)

    ids = [row["job_id"] for row in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate v12 job IDs.")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v12 split-select-certify matrix without running it.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_json(args.config)
    jobs = build(config)
    phase_counts: dict[str, int] = {}
    for row in jobs:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
    expected = len(config["datasets"]) * len(config["rolling_origins"]) * len(config["global_rules"]["seeds"]) * (
        1 + 2 * len(config["runtime"]["candidates"])
    )
    if len(jobs) != expected:
        raise AssertionError(f"Expected {expected} jobs, generated {len(jobs)}")
    payload = {
        "schema_version": "ata-nlp-select-certify-manifest-12.0-draft",
        "status": "draft_non_runnable_until_data_revision_and_execution_locks_exist",
        "config_path": str(args.config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "job_count": len(jobs),
        "phase_counts": phase_counts,
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        old = load_json(args.output)
        if old.get("status") != payload["status"]:
            raise FileExistsError(f"Refusing to overwrite a non-draft manifest: {args.output}")
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"job_count": len(jobs), "phase_counts": phase_counts}, indent=2))


if __name__ == "__main__":
    main()
