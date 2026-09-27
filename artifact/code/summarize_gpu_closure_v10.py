from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ata_nlp.closure_v10 import load_json, write_json_once


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "gpu_closure_v10.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "gpu_closure_v10.draft.json"


def result_path(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    root = PROJECT_ROOT / config["paths"]["result_root"]
    if "history_train" in job["phase"]:
        return root / "history" / f"{job['job_id']}.json"
    return root / "jobs" / job["suite_id"] / f"{job['job_id']}.json"


def interval(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> None:
    parser = argparse.ArgumentParser(description="Completeness-first aggregation of v10 GPU closure artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, manifest = load_json(args.config), load_json(args.manifest)
    complete, missing = [], []
    phase_complete = Counter()
    for job in manifest["jobs"]:
        path = result_path(config, job)
        if path.exists():
            complete.append((job, load_json(path), path))
            phase_complete[job["phase"]] += 1
        else:
            missing.append(job)
    aggregate: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for job, artifact, _ in complete:
        key = (job["suite_id"], job["dataset_id"], str(job.get("action", "history")))
        if job["phase"] == "posthoc_candidate_train_and_eval":
            for field in ("cert_gain_points", "raw_T_gain_points", "served_T_gain_points"):
                aggregate[key][field].append(float(artifact["policy"][field]))
            aggregate[key]["authorized"].append(float(bool(artifact["policy"]["authorized"])))
            aggregate[key]["harmful_served_action"].append(float(bool(artifact["policy"]["harmful_served_action"])))
            aggregate[key]["train_seconds"].append(float(artifact["training"]["elapsed_seconds"]))
            aggregate[key]["peak_gpu_memory_bytes"].append(float(artifact["training"]["peak_gpu_memory_bytes"]))
        elif job["phase"] == "prospective_candidate_train_cert_fit_cert_gate":
            aggregate[key]["cert_gate_gain_points"].append(float(artifact["policy"]["cert_gate_gain_points"]))
            aggregate[key]["authorized"].append(float(bool(artifact["policy"]["authorized"])))
        elif job["phase"] == "prospective_deploy_after_group_lock":
            policy = artifact["policy"]
            for field in ("raw_candidate_gain_points", "calibrated_candidate_gain_points", "served_gain_points"):
                aggregate[key][field].append(float(policy[field]))
            aggregate[key]["authorized"].append(float(bool(policy["authorized_from_cert_gate"])))
            aggregate[key]["harmful_served_action"].append(float(bool(policy["harmful_served_action"])))
        elif job["phase"] == "posthoc_inference":
            for intervention, row in artifact["interventions"].items():
                aggregate[(job["suite_id"], job["dataset_id"], intervention)]["deploy_gain_points"].append(float(row["deploy"]["gain_points_vs_base"]))
    groups = {
        "::".join(key): {metric: interval(values) for metric, values in sorted(metrics.items())}
        for key, metrics in sorted(aggregate.items())
    }
    expected_phase = Counter(row["phase"] for row in manifest["jobs"])
    failures_root = PROJECT_ROOT / config["paths"]["failure_root"]
    failure_files = list(failures_root.rglob("*.json")) if failures_root.exists() else []
    payload = {
        "schema_version": "ata-nlp-gpu-closure-summary-10.0",
        "status": "complete" if not missing else "incomplete_no_missing_cells_hidden",
        "job_count_expected": len(manifest["jobs"]),
        "job_count_complete": len(complete),
        "job_count_missing": len(missing),
        "phase_completeness": {
            phase: {"expected": expected_phase[phase], "complete": phase_complete[phase], "missing": expected_phase[phase] - phase_complete[phase]}
            for phase in expected_phase
        },
        "missing_job_ids": [row["job_id"] for row in missing],
        "failure_artifact_count": len(failure_files),
        "aggregates_cell_level_descriptive_not_independent_dataset_inference": groups,
        "statistical_warning": "Origins and seeds are nested repeated measurements; dataset is the independent generalization unit.",
    }
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = PACKAGE_ROOT / "analysis" / f"gpu_closure_v10_summary_{stamp}.json"
    write_json_once(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "complete": len(complete), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
