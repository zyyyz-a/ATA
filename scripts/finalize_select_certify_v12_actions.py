from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ata_nlp.select_certify_v12 import build_action_lock


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"
DEFAULT_EXECUTION = PACKAGE_ROOT / "locks" / "prospective_select_certify_v12.execution.lock.json"


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


def write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"Refusing to overwrite a different frozen artifact: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialize all 48 v12 actions before opening any deployment labels.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION)
    args = parser.parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    execution = load_json(args.execution_lock)
    if execution.get("status") != "frozen_before_v12_model_or_deployment_outcomes":
        raise PermissionError("A valid v12 execution lock is required.")
    for label, path in (("config", args.config), ("manifest", args.manifest)):
        if execution.get("bound_files", {}).get(label) != file_sha256(path):
            raise ValueError(f"Frozen {label} changed after sealing.")

    result_root = PROJECT_ROOT / config["paths"]["result_root"]
    deploy_root = result_root / "deployment"
    if deploy_root.exists() and any(deploy_root.glob("*.json")):
        raise PermissionError("Deployment output already exists; refusing retrospective action serialization.")

    jobs = [
        row for row in manifest["jobs"]
        if row["phase"] == "v12_candidate_train_calibrate_select_certify"
    ]
    candidates = []
    for job in jobs:
        path = result_root / "certification" / f"{job['job_id']}.json"
        if not path.exists():
            raise FileNotFoundError(f"C-only candidate is incomplete; T remains sealed: {path}")
        artifact = load_json(path)
        if artifact.get("stage") != "C_cal_C_select_C_gate_complete_T_unopened":
            raise PermissionError(f"Invalid C-only stage: {path}")
        if artifact.get("T_opened") is not False or artifact.get("deploy_role_loaded") is not False:
            raise PermissionError(f"Certification artifact touched T: {path}")
        features = artifact["C_only_features"]
        candidates.append({
            "dataset_id": job["dataset_id"],
            "origin_id": job["origin_id"],
            "seed": int(job["seed"]),
            "action": job["action"],
            "candidate_job_id": job["job_id"],
            "certification_sha256": file_sha256(path),
            "select_gain_points": float(features["select_gain_points"]),
            "gate_gain_points": float(features["gate_gain_points"]),
            "gate_gain_lcb_points": float(features["gate_gain_interval"]["lcb"]),
        })

    lock = build_action_lock(
        candidates,
        dataset_ids=[row["dataset_id"] for row in config["datasets"]],
        origin_ids=[row["origin_id"] for row in config["rolling_origins"]],
        seeds=config["global_rules"]["seeds"],
        actions=list(config["runtime"]["candidates"]),
        tie_break_order=config["primary_policy"]["nomination_tie_break_order_low_to_high_capacity"],
        config_sha256=file_sha256(args.config),
    )
    action_path = PROJECT_ROOT / config["paths"]["action_lock"]
    write_once(action_path, lock)
    barrier = {
        "schema_version": "ata-nlp-select-certify-pre-T-barrier-12.0",
        "status": "all_48_actions_locked_before_any_v12_T_outcome",
        "test_outcomes_used": False,
        "decision_count": lock["decision_count"],
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": file_sha256(args.manifest),
        "execution_lock_sha256": file_sha256(args.execution_lock),
        "action_lock_sha256": file_sha256(action_path),
    }
    barrier_path = PROJECT_ROOT / config["paths"]["global_barrier"]
    write_once(barrier_path, barrier)
    print(json.dumps({
        "action_lock": str(action_path),
        "barrier": str(barrier_path),
        "decisions": lock["decision_count"],
        "authorized": sum(row["authorized"] for row in lock["decisions"].values()),
    }, indent=2))


if __name__ == "__main__":
    main()

