from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REQUIRED_FIELDS = {
    "record_hash",
    "label",
    "deploy_bin",
    "base_logits",
    "served_logits",
}
ROLES = ("cert_select", "cert_gate", "deploy")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    action, remainder = stem.split("__", 1)
    for role in ROLES:
        marker = f"{role}__"
        if remainder.startswith(marker):
            return action, role, remainder[len(marker):]
    raise ValueError(f"Unrecognized sidecar name: {path.name}")


def load_and_audit(path: Path) -> dict:
    action, role, job_id = parse_name(path)
    rows = []
    schema_union: set[str] = set()
    schema_intersection: set[str] | None = None
    finite = True
    compatible_logits = True
    labels_are_int = True
    hashes = []
    labels = []
    bins = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Non-object at {path}:{line_number}")
            keys = set(row)
            schema_union |= keys
            schema_intersection = keys if schema_intersection is None else schema_intersection & keys
            base = row.get("base_logits", [])
            served = row.get("served_logits", [])
            compatible_logits &= bool(base) and len(base) == len(served)
            finite &= all(math.isfinite(float(value)) for value in [*base, *served])
            labels_are_int &= isinstance(row.get("label"), int)
            hashes.append(str(row.get("record_hash", "")))
            labels.append(row.get("label"))
            bins.append(row.get("deploy_bin"))
            rows.append(row)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "action": action,
        "role": role,
        "job_id": job_id,
        "rows": len(rows),
        "schema_union": sorted(schema_union),
        "schema_intersection": sorted(schema_intersection or set()),
        "required_fields_present_every_row": REQUIRED_FIELDS <= (schema_intersection or set()),
        "timestamp_present_every_row": "timestamp" in (schema_intersection or set()),
        "record_hash_unique": len(set(hashes)) == len(hashes),
        "record_hash_sequence": hashes,
        "label_sequence": labels,
        "deploy_bin_values": sorted(set(bins), key=str),
        "finite_logits": finite,
        "compatible_logits": compatible_logits,
        "labels_are_int": labels_are_int,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("artifacts/v12_completed_evidence_20260926/results_nlp_ata/v12_select_certify/predictions_hashed"),
    )
    parser.add_argument("--output", type=Path, default=Path("analysis/fpcc_gate_20260926/gate0_inventory.json"))
    parser.add_argument(
        "--timestamp-root",
        type=Path,
        default=Path("analysis/fpcc_gate_20260926/fpcc_gate_v12_timestamps_20260926"),
    )
    args = parser.parse_args()

    paths = sorted(args.prediction_root.rglob("*.jsonl"))
    audits = []
    errors = []
    grouped = defaultdict(dict)
    counts = Counter()
    for path in paths:
        try:
            audit = load_and_audit(path)
        except Exception as error:  # retain every failure in the audit artifact
            errors.append({"path": str(path), "error": repr(error)})
            continue
        relative = path.relative_to(args.prediction_root)
        dataset, origin, seed = relative.parts[:3]
        audit.update({"dataset": dataset, "origin": origin, "seed": seed})
        key = (dataset, origin, seed, audit["action"], audit["role"])
        grouped[key] = audit
        counts[(dataset, audit["action"], audit["role"])] += 1
        audits.append(audit)

    datasets = sorted({row["dataset"] for row in audits})
    origins = sorted({row["origin"] for row in audits})
    seeds = sorted({row["seed"] for row in audits})
    actions = sorted({row["action"] for row in audits})

    expected = {
        (dataset, origin, seed, action, role)
        for dataset in datasets
        for origin in origins
        for seed in seeds
        for action in actions
        for role in ROLES
    }
    missing = sorted(expected - set(grouped))

    # Identity/order must not depend on action or seed. This is stronger than
    # merely checking row counts and establishes an immutable ordinal timeline.
    order_mismatches = []
    label_mismatches = []
    for dataset in datasets:
        for origin in origins:
            for role in ROLES:
                candidates = [
                    value for key, value in grouped.items()
                    if key[0] == dataset and key[1] == origin and key[4] == role
                ]
                if not candidates:
                    continue
                reference = candidates[0]
                for candidate in candidates[1:]:
                    if candidate["record_hash_sequence"] != reference["record_hash_sequence"]:
                        order_mismatches.append([reference["path"], candidate["path"]])
                    if candidate["label_sequence"] != reference["label_sequence"]:
                        label_mismatches.append([reference["path"], candidate["path"]])

    full_tune_keys = {
        key for key in grouped if key[3] == "full_tune"
    }
    expected_full_tune = {
        (dataset, origin, seed, "full_tune", role)
        for dataset in datasets
        for origin in origins
        for seed in seeds
        for role in ROLES
    }

    common_valid = all(
        row["required_fields_present_every_row"]
        and row["record_hash_unique"]
        and row["finite_logits"]
        and row["compatible_logits"]
        and row["labels_are_int"]
        for row in audits
    )
    direct_timestamps_present = bool(audits) and all(row["timestamp_present_every_row"] for row in audits)
    timestamp_join_errors = []
    timestamp_files_verified = 0
    timestamp_sequences = {}
    for dataset in datasets:
        for origin in origins:
            for role in ROLES:
                path = args.timestamp_root / dataset / origin / f"{role}_timestamps.jsonl"
                if not path.exists():
                    timestamp_join_errors.append({"path": str(path), "error": "missing timestamp sidecar"})
                    continue
                rows = []
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        rows.append(json.loads(line))
                hashes = [str(row.get("record_hash", "")) for row in rows]
                labels = [row.get("label") for row in rows]
                bins = [row.get("deploy_bin") for row in rows]
                timestamps = [str(row.get("timestamp", "")) for row in rows]
                try:
                    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in timestamps]
                except Exception as error:
                    timestamp_join_errors.append({"path": str(path), "error": f"timestamp parse: {error!r}"})
                    continue
                if any(right < left for left, right in zip(parsed, parsed[1:])):
                    timestamp_join_errors.append({"path": str(path), "error": "timestamps are not monotone"})
                    continue
                candidates = [
                    value for key, value in grouped.items()
                    if key[0] == dataset and key[1] == origin and key[4] == role
                ]
                for candidate in candidates:
                    if hashes != candidate["record_hash_sequence"]:
                        timestamp_join_errors.append({"path": str(path), "sidecar": candidate["path"], "error": "record hash sequence mismatch"})
                    if labels != candidate["label_sequence"]:
                        timestamp_join_errors.append({"path": str(path), "sidecar": candidate["path"], "error": "label sequence mismatch"})
                    candidate_bins = candidate["deploy_bin_values"]
                    if sorted(set(bins), key=str) != candidate_bins:
                        timestamp_join_errors.append({"path": str(path), "sidecar": candidate["path"], "error": "deploy-bin value mismatch"})
                timestamp_files_verified += 1
                timestamp_sequences[(dataset, origin, role)] = {
                    "path": str(path),
                    "rows": len(rows),
                    "first_timestamp": timestamps[0] if timestamps else None,
                    "last_timestamp": timestamps[-1] if timestamps else None,
                    "monotone": True,
                }
    joined_timestamps_verified = (
        timestamp_files_verified == len(datasets) * len(origins) * len(ROLES)
        and not timestamp_join_errors
    )
    timestamps_present = direct_timestamps_present or joined_timestamps_verified
    ordinal_gate1_ready = (
        not errors
        and not (expected_full_tune - full_tune_keys)
        and not order_mismatches
        and not label_mismatches
        and common_valid
    )

    compact_files = []
    for row in audits:
        compact_files.append({key: value for key, value in row.items() if key not in {"record_hash_sequence", "label_sequence"}})
    report = {
        "schema_version": "fpcc-gate0-inventory-1.0",
        "prediction_root": str(args.prediction_root),
        "scope": {
            "datasets": datasets,
            "origins": origins,
            "seeds": seeds,
            "actions": actions,
            "roles": list(ROLES),
            "sidecar_files": len(audits),
            "rows": sum(row["rows"] for row in audits),
        },
        "integrity": {
            "parse_errors": errors,
            "missing_grid_cells": [list(value) for value in missing],
            "missing_full_tune_cells": [list(value) for value in sorted(expected_full_tune - full_tune_keys)],
            "order_mismatches": order_mismatches,
            "label_mismatches": label_mismatches,
            "all_logits_finite_and_shape_compatible": common_valid,
            "timestamp_join_errors": timestamp_join_errors,
            "timestamp_files_verified": timestamp_files_verified,
            "direct_timestamps_present": direct_timestamps_present,
            "joined_timestamps_verified": joined_timestamps_verified,
        },
        "decision": {
            "strict_gate0_timestamp_requirement_pass": ordinal_gate1_ready and timestamps_present,
            "ordered_gate1_split_half_ready": ordinal_gate1_ready,
            "gate2_calendar_time_ready": ordinal_gate1_ready and timestamps_present,
            "reason": (
                "All prediction rows join exactly to monotone privacy-preserving timestamp sidecars."
                if ordinal_gate1_ready and joined_timestamps_verified else
                "All full-tune logits, labels, and stable chronological row identities are present, but timestamps remain unresolved."
                if ordinal_gate1_ready and not timestamps_present else "See integrity fields."
            ),
        },
        "timestamp_sequences": {"|".join(key): value for key, value in sorted(timestamp_sequences.items())},
        "counts_by_dataset_action_role": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "files": compact_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "integrity": report["integrity"], "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
