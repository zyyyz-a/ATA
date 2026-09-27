from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def verify_required_files(root: Path) -> None:
    required = [
        "configs/prospective_select_certify_v12.draft.json",
        "manifests/prospective_select_certify_v12.draft.json",
        "locks/prospective_select_certify_v12.actions.lock.json",
        "locks/prospective_select_certify_v12.pre_T.lock.json",
        "artifact/analysis/s6/s6_policy_summary.csv",
        "artifact/analysis/sensor_gate/gate1_dataset_summaries.csv",
        "artifact/analysis/sensor_gate/measurement_floor_report.json",
        "artifact/analysis/sensor_gate/measurement_floor_trajectories.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    require(not missing, f"Missing release files: {missing}")


def verify_barrier(root: Path) -> dict[str, str]:
    action_path = root / "locks/prospective_select_certify_v12.actions.lock.json"
    barrier_path = root / "locks/prospective_select_certify_v12.pre_T.lock.json"
    barrier = load_json(barrier_path)
    action_digest = sha256(action_path)
    require(
        barrier.get("action_lock_sha256") == action_digest,
        "The pre-deployment barrier is not bound to the published action lock.",
    )
    return {"action_lock_sha256": action_digest, "barrier_sha256": sha256(barrier_path)}


def verify_policy_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "artifact/analysis/s6/s6_policy_summary.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy_family"] == "primary"]
    require(len(rows) == 2, f"Expected two primary policy rows, found {len(rows)}")

    expected = {
        "arxiv_temporal_primary_category_v12": (24, 19, 1.547791978891988),
        "patclass_temporal_ipc_section_v12": (24, 7, 0.0685350082516023),
    }
    summary: list[dict[str, Any]] = []
    for row in rows:
        dataset = row["dataset_id"]
        require(dataset in expected, f"Unexpected primary dataset: {dataset}")
        available, deployed, worst = expected[dataset]
        require(int(row["available_cells"]) == available, f"Cell count mismatch for {dataset}")
        require(int(row["deployed_cells"]) == deployed, f"Deployment count mismatch for {dataset}")
        require(int(row["harmful_action_count"]) == 0, f"Unexpected harmful cell for {dataset}")
        require(
            math.isclose(float(row["worst_served_cell_gain_points"]), worst, abs_tol=1e-12),
            f"Worst served cell mismatch for {dataset}",
        )
        summary.append(
            {
                "dataset": dataset,
                "deployed": deployed,
                "available": available,
                "coverage": float(row["coverage"]),
                "system_gain_points": float(row["system_mean_gain_points"]),
                "worst_served_cell_gain_points": worst,
            }
        )
    return summary


def verify_measurement_floor(root: Path) -> dict[str, Any]:
    report = load_json(root / "artifact/analysis/sensor_gate/measurement_floor_report.json")
    require(report.get("resolution") == 32, "Unexpected primary sensor resolution")
    require(
        math.isclose(float(report.get("target_split_half_reliability")), 0.70),
        "Unexpected reliability target",
    )

    path = root / "artifact/analysis/sensor_gate/measurement_floor_trajectories.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 96, f"Expected 96 trajectory-sensor rows, found {len(rows)}")
    finite = [row for row in rows if row["finite_requirement"].lower() == "true"]
    require(len(finite) == 43, f"Expected 43 finite requirements, found {len(finite)}")
    required = [float(row["required_total_block_n_for_0_70"]) for row in finite]
    multipliers = [float(row["required_budget_multiplier"]) for row in finite]
    median_required = statistics.median(required)
    median_multiplier = statistics.median(multipliers)
    require(round(median_required) == 1987, "Median required block size changed")
    require(round(median_multiplier, 1) == 25.7, "Median budget multiplier changed")
    return {
        "rows": len(rows),
        "no_finite_estimate": len(rows) - len(finite),
        "finite_estimate": len(finite),
        "median_required_records": median_required,
        "median_budget_multiplier": median_multiplier,
    }


def scan_public_tree(root: Path) -> list[str]:
    patterns = {
        "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
        "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "Windows home path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
        "Linux root path": re.compile(r"(?<![A-Za-z0-9_])/root/"),
        "server hostname": re.compile(r"podtcp|compshare", re.IGNORECASE),
    }
    findings: list[str] = []
    text_suffixes = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        # This module necessarily contains the signatures it searches for.
        # Scanning it would report the audit rules themselves as disclosures.
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(root)}")
    return findings


def run_checks(root: Path = ROOT) -> dict[str, Any]:
    verify_required_files(root)
    findings = scan_public_tree(root)
    require(not findings, "Sensitive-pattern scan failed:\n" + "\n".join(findings))
    return {
        "barrier": verify_barrier(root),
        "prospective_policy": verify_policy_rows(root),
        "measurement_floor": verify_measurement_floor(root),
        "sensitive_pattern_findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the public ATA audit release.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args()
    result = run_checks(args.root.resolve())
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("ATA release checks passed.")
        for row in result["prospective_policy"]:
            print(
                f"  {row['dataset']}: {row['deployed']}/{row['available']} served, "
                f"system gain {row['system_gain_points']:+.3f} pp"
            )
        floor = result["measurement_floor"]
        print(
            "  measurement floor: "
            f"{floor['no_finite_estimate']}/{floor['rows']} without a finite estimate; "
            f"median finite requirement {floor['median_required_records']:.1f} records"
        )


if __name__ == "__main__":
    main()
