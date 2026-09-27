from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"
DEFAULT_REVISIONS = PACKAGE_ROOT / "locks" / "prospective_select_certify_v12.revisions.lock.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "locks" / "prospective_select_certify_v12.execution.lock.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal v12 after source materialization and before all model outcomes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--revision-lock", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config, manifest, revisions = load_json(args.config), load_json(args.manifest), load_json(args.revision_lock)
    if revisions.get("status") != "resolved_before_v12_model_or_deployment_outcomes":
        raise PermissionError("The v12 revision lock is absent or invalid.")
    result_root = PROJECT_ROOT / config["paths"]["result_root"]
    if result_root.exists() and any(result_root.rglob("*.json")):
        raise PermissionError("v12 result JSON already exists; refusing to create a retrospective execution lock.")

    data_hashes = {}
    for spec in config["datasets"]:
        prepared = PROJECT_ROOT / config["paths"]["prepared_root"] / spec["dataset_id"] / "post_quarantine_v1" / "manifest.json"
        if not prepared.exists():
            raise FileNotFoundError(f"Prepared v12 dataset is missing: {prepared}")
        artifact = load_json(prepared)
        expected_revision = revisions["datasets"][spec["dataset_id"]]["resolved_revision"]
        if artifact.get("resolved_revision") != expected_revision:
            raise ValueError(f"Prepared revision mismatch for {spec['dataset_id']}")
        if artifact.get("status") != "prepared_without_model_inference_or_metric_outcomes":
            raise PermissionError(f"Prepared data has an invalid status: {prepared}")
        if artifact.get("model_outcomes_generated") is not False:
            raise PermissionError(f"Prepared data is not outcome-free: {prepared}")
        if artifact.get("test_firewall") is not True or artifact.get("deploy_files_may_only_be_loaded_after_complete_v12_action_lock") is not True:
            raise PermissionError(f"Prepared data lacks the v12 deployment firewall: {prepared}")
        file_hashes = artifact.get("file_sha256", {})
        if len(file_hashes) != len(config["rolling_origins"]) * 6:
            raise ValueError(f"Prepared data has an incomplete role-file inventory: {prepared}")
        for relative, expected_hash in sorted(file_hashes.items()):
            role_file = prepared.parent / relative
            if not role_file.is_file():
                raise FileNotFoundError(f"Prepared role file is missing: {role_file}")
            if sha256(role_file) != expected_hash:
                raise ValueError(f"Prepared role-file hash mismatch: {role_file}")
        data_hashes[spec["dataset_id"]] = sha256(prepared)

    code_paths = {
        "runner": PACKAGE_ROOT / "scripts" / "run_select_certify_v12_job.py",
        "shared_runner": PACKAGE_ROOT / "scripts" / "run_prospective_v11_job.py",
        "finalizer": PACKAGE_ROOT / "scripts" / "finalize_select_certify_v12_actions.py",
        "materializer": PACKAGE_ROOT / "scripts" / "materialize_select_certify_v12.py",
        "selector": PACKAGE_ROOT / "src" / "ata_nlp" / "select_certify_v12.py",
        "closure_runtime": PACKAGE_ROOT / "src" / "ata_nlp" / "closure_v10.py",
        "base_runtime": PACKAGE_ROOT / "src" / "ata_nlp" / "runtime.py",
        "adapters": PACKAGE_ROOT / "src" / "ata_nlp" / "adapters.py",
    }
    bound = {
        "config": sha256(args.config),
        "manifest": sha256(args.manifest),
        "revision_lock": sha256(args.revision_lock),
        **{key: sha256(path) for key, path in code_paths.items()},
    }
    payload = {
        "schema_version": "ata-nlp-select-certify-execution-lock-12.0",
        "status": "frozen_before_v12_model_or_deployment_outcomes",
        "protocol_id": config["protocol_id"],
        "job_count": manifest["job_count"],
        "bound_files": bound,
        "bound_code_paths": {key: str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for key, path in code_paths.items()},
        "data_manifest_sha256": data_hashes,
        "test_outcomes_used": False,
        "primary_policy": config["primary_policy"],
        "success_rule": config["success_rule"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Refusing to replace a different v12 execution lock: {args.output}")
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": payload["status"], "jobs": payload["job_count"], "datasets": list(data_hashes)}, indent=2))


if __name__ == "__main__":
    main()
