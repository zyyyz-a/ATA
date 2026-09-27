from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"
RUNNER = PACKAGE_ROOT / "scripts" / "run_select_certify_v12_job.py"


def result_path(config, job):
    root = PROJECT_ROOT / config["paths"]["result_root"]
    folder = {
        "v12_history_train": "history",
        "v12_candidate_train_calibrate_select_certify": "certification",
        "v12_counterfactual_deploy_audit_after_global_barrier": "deployment",
    }[job["phase"]]
    return root / folder / f"{job['job_id']}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a restart-safe v12 phase on one visible GPU.")
    parser.add_argument(
        "--phase", required=True,
        choices=("v12_history_train", "v12_candidate_train_calibrate_select_certify", "v12_counterfactual_deploy_audit_after_global_barrier"),
    )
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = [
        row for row in manifest["jobs"]
        if row["phase"] == args.phase and (not args.dataset_id or row["dataset_id"] in set(args.dataset_id))
    ]
    pending = [row for row in selected if not result_path(config, row).exists()]
    print(json.dumps({"phase": args.phase, "selected": len(selected), "complete": len(selected) - len(pending), "pending": len(pending)}, indent=2))
    if args.dry_run:
        return
    failure_root = PROJECT_ROOT / config["paths"]["failure_root"]
    failure_root.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(pending, start=1):
        started = time.time()
        command = [sys.executable, str(RUNNER), "--job-id", job["job_id"]]
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode == 0:
            print(f"[{index}/{len(pending)}] complete {job['job_id']}")
            continue
        failure = failure_root / f"{job['job_id']}.json"
        payload = {
            "schema_version": "ata-nlp-select-certify-failure-12.0",
            "job": job,
            "return_code": completed.returncode,
            "elapsed_seconds": time.time() - started,
            "result_exists": result_path(config, job).exists(),
        }
        if not failure.exists():
            failure.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{index}/{len(pending)}] failed {job['job_id']} (retained; queue continues)")
        if args.stop_on_error:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
