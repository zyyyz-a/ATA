from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


ROLES = ("cert_select", "cert_gate")
RESOLUTION = 32
TARGET_RELIABILITY = 0.70


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def locate(root: Path, dataset: str, origin: str, seed: str, role: str) -> Path:
    matches = list((root / dataset / origin / seed).glob(f"full_tune__{role}__*.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one full_tune {role} sidecar, found {len(matches)}: "
            f"{dataset}/{origin}/{seed}"
        )
    return matches[0]


def contiguous_blocks(rows: list[dict], count: int) -> list[list[dict]]:
    total = len(rows)
    return [
        rows[index * total // count : (index + 1) * total // count]
        for index in range(count)
    ]


def additive_terms(rows: list[dict]) -> np.ndarray:
    terms: list[list[float]] = []
    for row in rows:
        base = np.asarray(row["base_logits"], dtype=float)
        candidate = np.asarray(row["served_logits"], dtype=float)
        label = int(row["label"])
        residual = -softmax(base)
        residual[label] += 1.0
        correction = candidate - base
        correction = correction - np.mean(correction)
        terms.append(
            [
                float(np.dot(residual, correction)),
                float(np.dot(residual, residual)),
                float(np.dot(correction, correction)),
            ]
        )
    return np.asarray(terms, dtype=float)


def coordinate_and_delta_variance(terms: np.ndarray, metric: str) -> tuple[float, float]:
    n = terms.shape[0]
    means = np.mean(terms, axis=0)
    a_mean, r_mean, d_mean = means
    covariance = np.cov(terms, rowvar=False, ddof=1)
    if metric == "c":
        if r_mean <= 0.0 or d_mean <= 0.0:
            return math.nan, math.nan
        value = a_mean / math.sqrt(r_mean * d_mean)
        gradient = np.asarray(
            [
                1.0 / math.sqrt(r_mean * d_mean),
                -0.5 * value / r_mean,
                -0.5 * value / d_mean,
            ]
        )
    elif metric == "s":
        if d_mean <= 0.0:
            return math.nan, math.nan
        value = a_mean / d_mean
        gradient = np.asarray([1.0 / d_mean, 0.0, -a_mean / (d_mean**2)])
    else:
        raise ValueError(metric)
    sampling_variance = float(gradient @ covariance @ gradient / n)
    return float(value), max(0.0, sampling_variance)


def quantile(values: list[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    return float(np.quantile(np.asarray(finite), probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "artifacts/v12_completed_evidence_20260926/results_nlp_ata/"
            "v12_select_certify/predictions_hashed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/fpcc_gate_20260926/gate1_measurement_floor"),
    )
    args = parser.parse_args()

    trajectory_rows: list[dict] = []
    block_rows: list[dict] = []
    datasets = sorted(path.name for path in args.prediction_root.iterdir() if path.is_dir())
    for dataset in datasets:
        origins = sorted(path.name for path in (args.prediction_root / dataset).iterdir() if path.is_dir())
        for origin in origins:
            seeds = sorted(
                path.name
                for path in (args.prediction_root / dataset / origin).iterdir()
                if path.is_dir()
            )
            for seed in seeds:
                rows: list[dict] = []
                for role in ROLES:
                    rows.extend(read_jsonl(locate(args.prediction_root, dataset, origin, seed, role)))
                blocks = contiguous_blocks(rows, RESOLUTION)
                for metric in ("c", "s"):
                    estimates: list[float] = []
                    sampling_variances: list[float] = []
                    block_sizes: list[int] = []
                    for block_index, block in enumerate(blocks, start=1):
                        terms = additive_terms(block)
                        estimate, sampling_variance = coordinate_and_delta_variance(terms, metric)
                        estimates.append(estimate)
                        sampling_variances.append(sampling_variance)
                        block_sizes.append(len(block))
                        block_rows.append(
                            {
                                "dataset": dataset,
                                "origin": origin,
                                "seed": seed,
                                "metric": metric,
                                "block": block_index,
                                "n_full_block": len(block),
                                "estimate": estimate,
                                "delta_sampling_variance_full_block": sampling_variance,
                            }
                        )

                    observed_variance = float(np.var(np.asarray(estimates), ddof=1))
                    mean_noise_full = float(np.mean(np.asarray(sampling_variances)))
                    signal_variance = observed_variance - mean_noise_full
                    noise_constant = float(
                        np.mean(
                            np.asarray(sampling_variances) * np.asarray(block_sizes, dtype=float)
                        )
                    )
                    mean_full_n = float(np.mean(np.asarray(block_sizes, dtype=float)))
                    if signal_variance > 0.0 and noise_constant > 0.0:
                        required_half_n = (
                            noise_constant
                            * TARGET_RELIABILITY
                            / ((1.0 - TARGET_RELIABILITY) * signal_variance)
                        )
                        required_total_n = 2.0 * required_half_n
                        multiplier = required_total_n / mean_full_n
                        modeled_current_split_half = signal_variance / (
                            signal_variance + noise_constant / (mean_full_n / 2.0)
                        )
                    else:
                        required_total_n = math.inf
                        multiplier = math.inf
                        modeled_current_split_half = 0.0
                    trajectory_rows.append(
                        {
                            "dataset": dataset,
                            "origin": origin,
                            "seed": seed,
                            "metric": metric,
                            "blocks": RESOLUTION,
                            "mean_full_block_n": mean_full_n,
                            "observed_between_block_variance": observed_variance,
                            "mean_delta_sampling_variance_full_block": mean_noise_full,
                            "noise_corrected_signal_variance": signal_variance,
                            "modeled_current_split_half_reliability": modeled_current_split_half,
                            "required_total_block_n_for_0_70": required_total_n,
                            "required_budget_multiplier": multiplier,
                            "finite_requirement": math.isfinite(required_total_n),
                        }
                    )

    summaries: list[dict] = []
    for dataset in datasets:
        for metric in ("c", "s"):
            selected = [
                row
                for row in trajectory_rows
                if row["dataset"] == dataset and row["metric"] == metric
            ]
            finite_n = [
                float(row["required_total_block_n_for_0_70"])
                for row in selected
                if row["finite_requirement"]
            ]
            finite_multiplier = [
                float(row["required_budget_multiplier"])
                for row in selected
                if row["finite_requirement"]
            ]
            summaries.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "trajectories": len(selected),
                    "nonpositive_signal_variance": sum(
                        float(row["noise_corrected_signal_variance"]) <= 0.0
                        for row in selected
                    ),
                    "finite_requirement_trajectories": len(finite_n),
                    "median_current_full_block_n": statistics.median(
                        float(row["mean_full_block_n"]) for row in selected
                    ),
                    "median_required_total_block_n_among_finite": (
                        statistics.median(finite_n) if finite_n else None
                    ),
                    "q25_required_total_block_n_among_finite": quantile(finite_n, 0.25),
                    "q75_required_total_block_n_among_finite": quantile(finite_n, 0.75),
                    "median_required_multiplier_among_finite": (
                        statistics.median(finite_multiplier) if finite_multiplier else None
                    ),
                    "q25_required_multiplier_among_finite": quantile(finite_multiplier, 0.25),
                    "q75_required_multiplier_among_finite": quantile(finite_multiplier, 0.75),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "measurement_floor_trajectories.csv", trajectory_rows)
    write_csv(args.output_dir / "measurement_floor_blocks.csv", block_rows)
    write_csv(args.output_dir / "measurement_floor_summary.csv", summaries)
    report = {
        "schema_version": "gate1-measurement-floor-delta-1.0",
        "resolution": RESOLUTION,
        "target_split_half_reliability": TARGET_RELIABILITY,
        "model": (
            "Classical-test attenuation approximation. Per-record additive terms are propagated "
            "through c and s with a first-order delta method. Noise-corrected between-block "
            "variance is observed variance minus mean sampling variance. Required block size "
            "counts both halves. Nonpositive corrected signal variance has no finite plug-in "
            "sample-size estimate."
        ),
        "summaries": summaries,
        "interpretation_boundary": (
            "This is a measurement-budget diagnostic, not a prospective power guarantee. "
            "The approximation assumes independent records within blocks, locally stable "
            "per-record variance, and parallel half-block measurements."
        ),
    }
    (args.output_dir / "measurement_floor_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
