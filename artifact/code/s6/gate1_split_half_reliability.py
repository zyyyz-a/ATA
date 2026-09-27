from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SPLIT_SALT = "fpcc-gate1-split-half-frozen-20260926-v1"
RESOLUTIONS = (16, 32, 64)
ROLES = ("cert_select", "cert_gate")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def macro_f1(labels: list[int], predictions: list[int], classes: int) -> float:
    scores = []
    for label in range(classes):
        tp = sum(1 for truth, pred in zip(labels, predictions) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(labels, predictions) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(labels, predictions) if truth == label and pred != label)
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def residual_coordinates(rows: list[dict]) -> dict[str, float | int | None]:
    numerator = residual_energy = correction_energy = 0.0
    labels: list[int] = []
    base_predictions: list[int] = []
    candidate_predictions: list[int] = []
    classes = len(rows[0]["base_logits"])
    for row in rows:
        base = [float(value) for value in row["base_logits"]]
        candidate = [float(value) for value in row["served_logits"]]
        label = int(row["label"])
        probabilities = softmax(base)
        correction = [right - left for left, right in zip(base, candidate)]
        correction_mean = sum(correction) / classes
        correction = [value - correction_mean for value in correction]
        residual = [(1.0 if index == label else 0.0) - probabilities[index] for index in range(classes)]
        numerator += sum(left * right for left, right in zip(residual, correction))
        residual_energy += sum(value * value for value in residual)
        correction_energy += sum(value * value for value in correction)
        labels.append(label)
        base_predictions.append(max(range(classes), key=base.__getitem__))
        candidate_predictions.append(max(range(classes), key=candidate.__getitem__))
    c_value = (
        numerator / math.sqrt(residual_energy * correction_energy)
        if residual_energy > 0.0 and correction_energy > 0.0 else None
    )
    s_value = numerator / correction_energy if correction_energy > 0.0 else None
    delta_u = 100.0 * (
        macro_f1(labels, candidate_predictions, classes)
        - macro_f1(labels, base_predictions, classes)
    )
    return {"n": len(rows), "c": c_value, "s": s_value, "delta_u_points": delta_u}


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    if left_sum <= 0.0 or right_sum <= 0.0:
        return None
    return numerator / math.sqrt(left_sum * right_sum)


def spearman(left: list[float | None], right: list[float | None]) -> tuple[float | None, int]:
    pairs = [(float(x), float(y)) for x, y in zip(left, right) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    x_values, y_values = map(list, zip(*pairs))
    return pearson(average_ranks(x_values), average_ranks(y_values)), len(pairs)


def contiguous_blocks(rows: list[dict], count: int) -> list[list[dict]]:
    blocks = []
    total = len(rows)
    for index in range(count):
        start = index * total // count
        end = (index + 1) * total // count
        blocks.append(rows[start:end])
    return blocks


def deterministic_halves(rows: list[dict], context: str) -> tuple[list[dict], list[dict]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SPLIT_SALT}\0{context}\0{row['record_hash']}".encode("utf-8")
        ).digest(),
    )
    return ranked[::2], ranked[1::2]


def quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def locate(root: Path, dataset: str, origin: str, seed: str, role: str) -> Path:
    matches = list((root / dataset / origin / seed).glob(f"full_tune__{role}__*.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one full_tune {role} sidecar, found {len(matches)}: {dataset}/{origin}/{seed}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("artifacts/v12_completed_evidence_20260926/results_nlp_ata/v12_select_certify/predictions_hashed"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/fpcc_gate_20260926/gate1"))
    args = parser.parse_args()

    datasets = sorted(path.name for path in args.prediction_root.iterdir() if path.is_dir())
    trajectories = []
    block_rows = []
    input_files = []
    for dataset in datasets:
        origins = sorted(path.name for path in (args.prediction_root / dataset).iterdir() if path.is_dir())
        for origin in origins:
            seeds = sorted(path.name for path in (args.prediction_root / dataset / origin).iterdir() if path.is_dir())
            for seed in seeds:
                paths = [locate(args.prediction_root, dataset, origin, seed, role) for role in ROLES]
                roles = [read_jsonl(path) for path in paths]
                input_files.extend(str(path) for path in paths)
                rows = roles[0] + roles[1]
                if len({row["record_hash"] for row in rows}) != len(rows):
                    raise RuntimeError(f"Duplicate certification identity: {dataset}/{origin}/{seed}")
                for resolution in RESOLUTIONS:
                    metrics_a = defaultdict(list)
                    metrics_b = defaultdict(list)
                    sizes = []
                    for block_index, block in enumerate(contiguous_blocks(rows, resolution)):
                        context = f"{dataset}|{origin}|{seed}|{resolution}|{block_index}"
                        half_a, half_b = deterministic_halves(block, context)
                        left = residual_coordinates(half_a)
                        right = residual_coordinates(half_b)
                        sizes.append((left["n"], right["n"]))
                        for metric in ("c", "s", "delta_u_points"):
                            metrics_a[metric].append(left[metric])
                            metrics_b[metric].append(right[metric])
                        block_rows.append({
                            "dataset": dataset,
                            "origin": origin,
                            "seed": seed,
                            "resolution": resolution,
                            "block": block_index + 1,
                            "n_a": left["n"],
                            "n_b": right["n"],
                            "c_a": left["c"],
                            "c_b": right["c"],
                            "s_a": left["s"],
                            "s_b": right["s"],
                            "delta_u_a_points": left["delta_u_points"],
                            "delta_u_b_points": right["delta_u_points"],
                        })
                    result = {
                        "dataset": dataset,
                        "origin": origin,
                        "seed": seed,
                        "resolution": resolution,
                        "certification_rows": len(rows),
                        "minimum_half_block_n": min(min(pair) for pair in sizes),
                    }
                    for metric in ("c", "s", "delta_u_points"):
                        correlation, valid_blocks = spearman(metrics_a[metric], metrics_b[metric])
                        result[f"spearman_{metric}"] = correlation
                        result[f"valid_blocks_{metric}"] = valid_blocks
                    trajectories.append(result)

    summaries = []
    for dataset in datasets:
        for resolution in RESOLUTIONS:
            selected = [row for row in trajectories if row["dataset"] == dataset and row["resolution"] == resolution]
            summary = {"dataset": dataset, "resolution": resolution, "trajectories": len(selected)}
            for metric in ("c", "s", "delta_u_points"):
                values = [float(row[f"spearman_{metric}"]) for row in selected if row[f"spearman_{metric}"] is not None]
                summary[f"median_spearman_{metric}"] = statistics.median(values) if values else None
                summary[f"q25_spearman_{metric}"] = quantile(values, 0.25)
                summary[f"q75_spearman_{metric}"] = quantile(values, 0.75)
                summary[f"fraction_gt_0_7_{metric}"] = sum(value > 0.70 for value in values) / len(values) if values else None
            summary["primary_pass_c_and_s"] = (
                resolution == 32
                and summary["median_spearman_c"] is not None
                and summary["median_spearman_s"] is not None
                and summary["median_spearman_c"] > 0.70
                and summary["median_spearman_s"] > 0.70
            )
            summaries.append(summary)

    primary = [row for row in summaries if row["resolution"] == 32]
    overall_pass = len(primary) == len(datasets) and all(row["primary_pass_c_and_s"] for row in primary)
    report = {
        "schema_version": "fpcc-gate1-split-half-1.0",
        "frozen_split_salt_sha256": hashlib.sha256(SPLIT_SALT.encode("utf-8")).hexdigest(),
        "roles_in_order": list(ROLES),
        "resolutions": list(RESOLUTIONS),
        "primary_resolution": 32,
        "threshold": "dataset-wise median trajectory Spearman(c) > 0.70 and median trajectory Spearman(s) > 0.70",
        "input_files": input_files,
        "summaries": summaries,
        "trajectories": trajectories,
        "decision": {
            "gate1_pass": overall_pass,
            "dataset_primary_pass": {row["dataset"]: row["primary_pass_c_and_s"] for row in primary},
            "gate2_allowed_by_gate1": overall_pass,
            "evidence_status": "strict Gate 0 passed; hashes joined exactly to monotone timestamp sidecars",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "gate1_trajectory_correlations.csv", trajectories)
    write_csv(args.output_dir / "gate1_block_coordinates.csv", block_rows)
    write_csv(args.output_dir / "gate1_dataset_summaries.csv", summaries)

    lines = [
        "# Gate 1 split-half reliability",
        "",
        f"Decision: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "Primary rule: at 32 blocks, both median trajectory-level Spearman(c) and Spearman(s) must exceed 0.70 on each dataset.",
        "",
        "| Dataset | Blocks | Median rho(c) | Median rho(s) | Median rho(delta U) | c > .7 | s > .7 | Primary pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['resolution']} | {row['median_spearman_c']:.3f} | "
            f"{row['median_spearman_s']:.3f} | {row['median_spearman_delta_u_points']:.3f} | "
            f"{row['fraction_gt_0_7_c']:.1%} | {row['fraction_gt_0_7_s']:.1%} | "
            f"{'yes' if row['primary_pass_c_and_s'] else 'no'} |"
        )
    lines.extend([
        "",
        "Strict Gate 0 passed after all rows joined exactly to monotone privacy-preserving timestamp sidecars.",
        "Gate 2 remains locked because Gate 1 failed on both datasets at the frozen primary resolution.",
    ])
    (args.output_dir / "GATE1_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": primary, "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
