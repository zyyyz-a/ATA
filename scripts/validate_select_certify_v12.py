from __future__ import annotations

import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("job_count") != 624:
        errors.append("v12 manifest must contain exactly 624 GPU jobs")
    if len(config.get("datasets", [])) != 2:
        errors.append("v12 requires two untouched datasets")
    if len(config.get("rolling_origins", [])) != 8:
        errors.append("v12 requires eight rolling origins")
    if len(config.get("runtime", {}).get("candidates", {})) != 6:
        errors.append("v12 requires six registered adaptation actions")
    roles = ("history_end", "adapt_end", "cert_cal_end", "cert_select_end", "cert_gate_end", "deploy_end")
    for origin in config.get("rolling_origins", []):
        values = [float(origin[key]) for key in roles]
        if values != sorted(values) or len(values) != len(set(values)):
            errors.append(f"non-strict chronology at {origin.get('origin_id')}")
    jobs = manifest.get("jobs", [])
    ids = [row["job_id"] for row in jobs]
    if len(ids) != len(set(ids)):
        errors.append("duplicate job IDs")
    for row in jobs:
        if row["phase"] == "v12_candidate_train_calibrate_select_certify" and row.get("must_not_load_deploy") is not True:
            errors.append(f"candidate job lacks deployment firewall: {row['job_id']}")
        if row["phase"] == "v12_counterfactual_deploy_audit_after_global_barrier":
            ordinary = [dep for dep in row["dependencies"] if not str(dep).startswith(("action_lock::", "global_pre_T_barrier::"))]
            if len(ordinary) != 1:
                errors.append(f"counterfactual deployment must bind exactly one candidate: {row['job_id']}")
    if errors:
        raise RuntimeError("\n".join(errors))
    print(json.dumps({
        "status": "pass",
        "datasets": [row["dataset_id"] for row in config["datasets"]],
        "origins": len(config["rolling_origins"]),
        "candidate_actions": len(config["runtime"]["candidates"]),
        "gpu_jobs": manifest["job_count"],
        "primary_policy": config["primary_policy"]["name"],
    }, indent=2))


if __name__ == "__main__":
    main()
