from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ata_nlp.prospective_v11 import (
    deterministic_stratified_sample,
    file_sha256,
    load_json,
    normalized_text_hash,
    parse_timestamp,
    simhash64,
    stable_rank,
    write_json_once,
)


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
IPC_RE = re.compile(r"[A-H]", re.IGNORECASE)


def synthetic_id(dataset_id: str, values: Iterable[Any]) -> str:
    payload = dataset_id + "\0" + "\0".join(map(str, values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def arxiv_leaf(categories: Any) -> str:
    if not isinstance(categories, (list, tuple)) or not categories:
        raise ValueError("arXiv categories must be a non-empty source-ordered sequence.")
    primary = str(categories[0]).strip()
    leaf = primary.split("->")[-1].strip()
    if not leaf:
        raise ValueError(f"Invalid arXiv primary category: {categories!r}")
    return leaf


def patent_date(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 8:
        raise ValueError(f"Invalid patent date: {value!r}")
    parsed = datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def ipc_section(value: Any) -> str:
    match = IPC_RE.search(str(value).upper())
    if match is None:
        raise ValueError(f"No IPC section A-H in {value!r}")
    return match.group(0).upper()


def arxiv_rows(spec: dict[str, Any], revision: str):
    from datasets import load_dataset

    loaded = load_dataset(spec["hf_id"], spec.get("hf_config", "default"), revision=revision)
    rows = []
    scanned = invalid = 0
    for split in spec["hf_splits"]:
        for raw in loaded[split]:
            scanned += 1
            try:
                source_id = str(raw[spec["id_field"]]).strip()
                title = str(raw.get("title", "") or "").strip()
                abstract = str(raw.get("abstract", "") or "").strip()
                text = " </s> ".join(value for value in (title, abstract) if value)
                timestamp = parse_timestamp(raw[spec["timestamp_field"]]).isoformat()
                category = arxiv_leaf(raw[spec["category_field"]])
                if not source_id or not text:
                    raise ValueError("empty identity or text")
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                invalid += 1
                continue
            rows.append({
                "record_id": synthetic_id(spec["dataset_id"], [source_id]),
                "timestamp": timestamp,
                "text": text,
                "source_label": category,
                "group_keys": [f"arxiv:{source_id}"],
            })
    return rows, {"source_records_scanned": scanned, "invalid_records": invalid, "source_files": "datasets_revision_snapshot"}


def patent_rows(spec: dict[str, Any], revision: str):
    from huggingface_hub import hf_hub_download

    # Patent descriptions and claims can exceed Python's conservative 128 KiB
    # CSV-field default even though v12 retains only title and abstract.
    csv.field_size_limit(sys.maxsize)
    rows = []
    scanned = invalid = non_english = 0
    files = {}
    for year in spec["source_years"]:
        filename = spec["source_file_pattern"].format(year=year)
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                source = Path(hf_hub_download(
                    repo_id=spec["hf_id"], repo_type="dataset", revision=revision, filename=filename
                ))
                break
            except Exception as error:
                last_error = error
                if attempt == 8:
                    raise
                delay = min(60, 5 * attempt)
                print(
                    f"Transient download failure for {filename} "
                    f"(attempt {attempt}/8); retrying in {delay}s: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        else:  # pragma: no cover - defensive; the loop either breaks or raises.
            raise RuntimeError(f"Failed to download {filename}") from last_error
        files[filename] = {"local_path": str(source), "sha256": file_sha256(source), "bytes": source.stat().st_size}
        with source.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=spec.get("csv_separator", ";"))
            for raw in reader:
                scanned += 1
                normalized = {str(key).strip().lstrip("\ufeff"): value for key, value in raw.items() if key is not None}
                if str(normalized.get("lang", "")).strip().lower() != spec["language_filter"]:
                    non_english += 1
                    continue
                try:
                    source_id = str(normalized[spec["id_field"]]).strip()
                    title = str(normalized.get("title", "") or "").strip()
                    abstract = str(normalized.get("abstract", "") or "").strip()
                    text = " </s> ".join(value for value in (title, abstract) if value)
                    timestamp = patent_date(normalized[spec["timestamp_field"]])
                    label = ipc_section(normalized[spec["category_field"]])
                    if not source_id or not text:
                        raise ValueError("empty identity or text")
                except (KeyError, TypeError, ValueError):
                    invalid += 1
                    continue
                patent_number = str(normalized.get("patent_number", "") or "").strip()
                groups = [f"ucid:{source_id}"]
                if patent_number:
                    groups.append(f"patent:{patent_number}")
                rows.append({
                    "record_id": synthetic_id(spec["dataset_id"], [source_id]),
                    "timestamp": timestamp,
                    "text": text,
                    "source_label": label,
                    "group_keys": groups,
                })
    return rows, {
        "source_records_scanned": scanned,
        "invalid_records": invalid,
        "non_english_records": non_english,
        "source_files": files,
        "download_bytes": sum(value["bytes"] for value in files.values()),
    }


def deduplicate(rows: list[dict[str, Any]]):
    seen_ids = set()
    seen_text = set()
    output = []
    duplicate_ids = duplicate_texts = 0
    for row in sorted(rows, key=lambda value: (parse_timestamp(value["timestamp"]), value["record_id"])):
        if row["record_id"] in seen_ids:
            duplicate_ids += 1
            continue
        text_hash = normalized_text_hash(row["text"])
        if text_hash in seen_text:
            duplicate_texts += 1
            continue
        value = dict(row)
        value["text_hash"] = text_hash
        value["simhash64"] = f"{simhash64(row['text']):016x}"
        seen_ids.add(value["record_id"])
        seen_text.add(text_hash)
        output.append(value)
    return output, {"duplicate_ids_removed": duplicate_ids, "exact_text_duplicates_removed": duplicate_texts}


def freeze_labels(rows: list[dict[str, Any]], spec: dict[str, Any]):
    if spec["dataset_id"].startswith("arxiv_"):
        cutoff = max(1, int(0.25 * len(rows)))
        counts = Counter(row["source_label"] for row in rows[:cutoff])
        labels = [label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(spec["num_labels"])]]
    else:
        labels = list(spec["label_names"])
    if len(labels) != int(spec["num_labels"]):
        raise RuntimeError(f"Could not freeze {spec['num_labels']} labels for {spec['dataset_id']}")
    label_to_id = {label: index for index, label in enumerate(labels)}
    task_rows = []
    for row in rows:
        if row["source_label"] not in label_to_id:
            continue
        value = dict(row)
        source_label = value.pop("source_label")
        value["label"] = label_to_id[source_label]
        task_rows.append(value)
    return task_rows, labels


def safe_cut(rows: list[dict[str, Any]], target: int, lower: int, upper: int) -> int:
    boundaries = [
        index for index in range(max(1, lower + 1), min(len(rows), upper + 1))
        if rows[index - 1]["timestamp"] < rows[index]["timestamp"]
    ]
    if not boundaries:
        raise RuntimeError("No timestamp-safe boundary near a v12 chronological cut.")
    return min(boundaries, key=lambda value: (abs(value - target), value))


def cap_rows(rows: list[dict[str, Any]], maximum: int, role: str, origin_id: str):
    if len(rows) <= maximum:
        return rows
    chosen = sorted(rows, key=lambda row: stable_rank(row["record_id"], role, origin_id, salt="ata-v12-role-cap"))[:maximum]
    return sorted(chosen, key=lambda row: (row["timestamp"], row["record_id"]))


def assign_origin(rows, origin, caps, deploy_bins):
    endpoint_keys = ("history_end", "adapt_end", "cert_cal_end", "cert_select_end", "cert_gate_end", "deploy_end")
    names = ("history", "adapt", "cert_cal", "cert_select", "cert_gate", "deploy")
    cuts = []
    previous = 0
    for index, key in enumerate(endpoint_keys):
        target = int(len(rows) * float(origin[key]))
        cut = safe_cut(rows, target, previous, len(rows) - (len(endpoint_keys) - index - 1))
        if cut <= previous:
            raise RuntimeError(f"Non-increasing v12 cut at {origin['origin_id']} {key}")
        cuts.append(cut)
        previous = cut
    raw = {}
    start = 0
    for name, end in zip(names, cuts):
        raw[name] = rows[start:end]
        start = end

    group_roles = defaultdict(set)
    for role, role_rows in raw.items():
        for row in role_rows:
            for group in row["group_keys"]:
                group_roles[group].add(role)
    crossing = {group for group, roles in group_roles.items() if len(roles) > 1}
    quarantined = {
        row["record_id"] for role_rows in raw.values() for row in role_rows
        if any(group in crossing for group in row["group_keys"])
    }
    clean = {role: [row for row in role_rows if row["record_id"] not in quarantined] for role, role_rows in raw.items()}

    bands = defaultdict(list)
    for role, role_rows in clean.items():
        for row in role_rows:
            value = int(row["simhash64"], 16)
            for band in range(4):
                bands[(band, (value >> (16 * band)) & 0xFFFF)].append((role, row["record_id"], value))
    near = set()
    checked = set()
    for candidates in bands.values():
        if len({role for role, _, _ in candidates}) < 2 or len(candidates) > 200:
            continue
        for index, (left_role, left_id, left_value) in enumerate(candidates):
            for right_role, right_id, right_value in candidates[index + 1:]:
                if left_role == right_role:
                    continue
                pair = tuple(sorted((left_id, right_id)))
                if pair in checked:
                    continue
                checked.add(pair)
                if (left_value ^ right_value).bit_count() <= 3:
                    near.update(pair)
    clean = {role: [row for row in role_rows if row["record_id"] not in near] for role, role_rows in clean.items()}
    clean = {role: cap_rows(role_rows, int(caps[role]), role, origin["origin_id"]) for role, role_rows in clean.items()}
    for role, role_rows in clean.items():
        for index, row in enumerate(role_rows):
            row["deploy_bin"] = min(deploy_bins - 1, index * deploy_bins // max(len(role_rows), 1)) if role == "deploy" else -1
    return clean, {
        "raw_counts": {role: len(value) for role, value in raw.items()},
        "retained_counts": {role: len(value) for role, value in clean.items()},
        "group_crossing_records_quarantined": len(quarantined),
        "near_duplicate_records_quarantined": len(near),
        "cuts": cuts,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            public = {key: value for key, value in row.items() if key not in {"group_keys", "text_hash", "simhash64"}}
            handle.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize and audit both untouched v12 streams without model inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-id", action="append", default=[])
    args = parser.parse_args()
    config = load_json(args.config)
    from huggingface_hub import HfApi

    selected = [row for row in config["datasets"] if not args.dataset_id or row["dataset_id"] in set(args.dataset_id)]
    summary = []
    for spec in selected:
        revision = str(HfApi().dataset_info(spec["hf_id"], revision=spec["requested_revision"]).sha)
        root = PROJECT_ROOT / config["paths"]["prepared_root"] / spec["dataset_id"] / "post_quarantine_v1"
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite an existing v12 prepared stream: {root}")
        rows, source = arxiv_rows(spec, revision) if spec["dataset_id"].startswith("arxiv_") else patent_rows(spec, revision)
        if len(rows) < int(spec["minimum_source_records"]):
            raise RuntimeError(f"Too few usable source records for {spec['dataset_id']}: {len(rows)}")
        rows, duplicate_audit = deduplicate(rows)
        rows = deterministic_stratified_sample(
            rows, maximum=int(spec["maximum_retained_records"]), timestamp_field="timestamp", id_field="record_id"
        )
        rows, labels = freeze_labels(rows, spec)
        if len(rows) < int(spec["minimum_task_records"]):
            raise RuntimeError(f"Too few task records for {spec['dataset_id']}: {len(rows)}")
        months = {row["timestamp"][:7] for row in rows}
        if len(months) < int(spec["minimum_distinct_months"]):
            raise RuntimeError(f"Too few distinct months for {spec['dataset_id']}: {len(months)}")
        hashes = {}
        audits = {}
        for origin in config["rolling_origins"]:
            roles, audit = assign_origin(rows, origin, config["role_caps_per_origin"], int(config["global_rules"]["deploy_bins"]))
            for role, minimum in config["minimum_role_records_per_origin"].items():
                if len(roles[role]) < int(minimum):
                    raise RuntimeError(
                        f"Underpowered v12 role {spec['dataset_id']} {origin['origin_id']} {role}: {len(roles[role])} < {minimum}"
                    )
            audits[origin["origin_id"]] = audit
            for role, role_rows in roles.items():
                path = root / origin["origin_id"] / f"{role}.jsonl"
                write_jsonl(path, role_rows)
                hashes[f"{origin['origin_id']}/{role}.jsonl"] = file_sha256(path)
        manifest = {
            "schema_version": "ata-nlp-select-certify-data-12.0",
            "status": "prepared_without_model_inference_or_metric_outcomes",
            "dataset_id": spec["dataset_id"],
            "hf_id": spec["hf_id"],
            "requested_revision": spec["requested_revision"],
            "resolved_revision": revision,
            "license": spec["license"],
            "task_type": spec["task_type"],
            "label_names": labels,
            "num_labels": len(labels),
            "max_length": spec["max_length"],
            "source": source,
            "retained_task_records": len(rows),
            "distinct_months": len(months),
            "duplicate_audit": duplicate_audit,
            "origin_audits": audits,
            "file_sha256": hashes,
            "test_firewall": True,
            "deploy_files_may_only_be_loaded_after_complete_v12_action_lock": True,
            "model_outcomes_generated": False,
        }
        write_json_once(root / "manifest.json", manifest)
        summary.append({"dataset_id": spec["dataset_id"], "revision": revision, "records": len(rows), "months": len(months)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
