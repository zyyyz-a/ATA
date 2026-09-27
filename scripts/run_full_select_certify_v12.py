from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"
ORCHESTRATOR = PACKAGE_ROOT / "scripts" / "orchestrate_select_certify_v12.py"
FINALIZER = PACKAGE_ROOT / "scripts" / "finalize_select_certify_v12_actions.py"

PHASES = (
    "v12_history_train",
    "v12_candidate_train_calibrate_select_certify",
    "v12_counterfactual_deploy_audit_after_global_barrier",
)
FOLDERS = {
    PHASES[0]: "history",
    PHASES[1]: "certification",
    PHASES[2]: "deployment",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def wait_for_existing_orchestrator(pid: int) -> None:
    proc = Path("/proc") / str(pid)
    last_report = 0.0
    while proc.exists():
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            break
        if "orchestrate_select_certify_v12.py" not in command:
            break
        now = time.time()
        if now - last_report >= 300:
            print(json.dumps({"waiting_for_existing_history_pid": pid, "command": command}), flush=True)
            last_report = now
        time.sleep(30)


def phase_jobs(manifest: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    return [row for row in manifest["jobs"] if row["phase"] == phase]


def result_path(config: dict[str, Any], job: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["result_root"] / FOLDERS[job["phase"]] / f"{job['job_id']}.json"


def assert_phase_complete(config: dict[str, Any], manifest: dict[str, Any], phase: str) -> None:
    jobs = phase_jobs(manifest, phase)
    missing = [row["job_id"] for row in jobs if not result_path(config, row).is_file()]
    print(json.dumps({
        "phase": phase,
        "expected": len(jobs),
        "complete": len(jobs) - len(missing),
        "missing": len(missing),
    }), flush=True)
    if missing:
        raise RuntimeError(
            f"Refusing to cross the v12 phase barrier; {len(missing)} results are missing "
            f"from {phase}. First missing IDs: {missing[:10]}"
        )


def run_phase(config: dict[str, Any], manifest: dict[str, Any], phase: str) -> None:
    subprocess.run(
        [sys.executable, str(ORCHESTRATOR), "--phase", phase],
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert_phase_complete(config, manifest, phase)


def ensure_action_lock(config: dict[str, Any]) -> None:
    action = PROJECT_ROOT / config["paths"]["action_lock"]
    barrier = PROJECT_ROOT / config["paths"]["global_barrier"]
    if action.is_file() and barrier.is_file():
        print(json.dumps({"action_lock": str(action), "global_barrier": str(barrier), "status": "already_present"}), flush=True)
        return
    if action.exists() or barrier.exists():
        raise RuntimeError("Only one of the v12 action lock and global barrier exists; refusing an ambiguous resume.")
    subprocess.run([sys.executable, str(FINALIZER)], cwd=PROJECT_ROOT, check=True)
    if not action.is_file() or not barrier.is_file():
        raise RuntimeError("The v12 finalizer returned without both frozen pre-T artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restart-safe, barrier-preserving full v12 GPU pipeline.")
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    config, manifest = load_json(CONFIG), load_json(MANIFEST)

    if args.wait_for_pid is not None:
        wait_for_existing_orchestrator(args.wait_for_pid)

    run_phase(config, manifest, PHASES[0])
    run_phase(config, manifest, PHASES[1])
    ensure_action_lock(config)
    run_phase(config, manifest, PHASES[2])
    print(json.dumps({
        "status": "v12_complete",
        "history": len(phase_jobs(manifest, PHASES[0])),
        "certification": len(phase_jobs(manifest, PHASES[1])),
        "deployment_audit": len(phase_jobs(manifest, PHASES[2])),
        "gpu_required": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
