from __future__ import annotations

"""Build the COLING v11 quantitative evidence figure from the sealed v10 summary.

The figure deliberately distinguishes raw candidate utility, authorized/served
utility, and post-hoc decision-coordinate diagnostics. Intervals are
descriptive 95% cell-level intervals; origins and seeds remain nested repeated
measurements rather than independent datasets.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SUMMARY = PROJECT_ROOT / "nlp_ata_experiments" / "analysis" / "gpu_closure_v10_summary_20260918T094049Z.json"
TAIL_AUDIT = PROJECT_ROOT / "nlp_ata_experiments" / "analysis" / "gpu_closure_v10_tail_control_final_20260919_v2.json"
OUT_STEM = HERE / "ata_v11_quantitative_evidence"
SOURCE_CSV = HERE / "ata_v11_quantitative_evidence_source.csv"


def require_font(name: str) -> None:
    try:
        font_manager.findfont(name, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(f"Required figure font is unavailable: {name}") from exc


require_font("Comic Sans MS")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Comic Sans MS", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


NAVY = "#104E8B"
BLUE = "#376B9E"
MID_BLUE = "#5F89B1"
PALE_BLUE = "#AFC3D8"
CYAN = "#C5E9E3"
LIGHT = "#D7E1EB"
PALE_RED = "#F2DADA"
ROSE = "#E5B5B5"
RED = "#B22222"
ORANGE = "#D89090"
GREY = "#7A8792"
DARK = "#24323D"
GREEN = "#4D8B6A"


def load_groups() -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return payload["aggregates_cell_level_descriptive_not_independent_dataset_inference"]


def load_tail_audit() -> Mapping[str, Any]:
    return json.loads(TAIL_AUDIT.read_text(encoding="utf-8"))


def matching(groups: Mapping[str, Any], *, suite: str, action: str | None = None) -> list[Mapping[str, Any]]:
    rows = []
    for key, value in groups.items():
        parts = key.split("::")
        if parts[0] != suite:
            continue
        if action is not None and parts[-1] != action:
            continue
        rows.append(value)
    return rows


def pool_metric(rows: Iterable[Mapping[str, Any]], metric: str) -> dict[str, float]:
    cells = [row[metric] for row in rows if metric in row and row[metric]["n"]]
    if not cells:
        raise KeyError(metric)
    n_total = sum(int(cell["n"]) for cell in cells)
    mean = sum(int(cell["n"]) * float(cell["mean"]) for cell in cells) / n_total
    if n_total > 1:
        ss_within = sum((int(cell["n"]) - 1) * float(cell["sd"]) ** 2 for cell in cells)
        ss_between = sum(int(cell["n"]) * (float(cell["mean"]) - mean) ** 2 for cell in cells)
        sd = math.sqrt((ss_within + ss_between) / (n_total - 1))
    else:
        sd = 0.0
    half = 1.96 * sd / math.sqrt(n_total) if n_total > 1 else 0.0
    return {"n": n_total, "mean": mean, "low": mean - half, "high": mean + half, "sd": sd}


def direct_metric(group: Mapping[str, Any], metric: str) -> dict[str, float]:
    cell = group[metric]
    return {
        "n": int(cell["n"]),
        "mean": float(cell["mean"]),
        "low": float(cell["ci95_low"]),
        "high": float(cell["ci95_high"]),
        "sd": float(cell["sd"]),
    }


def tail_summary_metric(summary: Mapping[str, Any]) -> dict[str, float]:
    cell = summary["served_population_including_zero_for_abstention"]
    n = int(cell["n"])
    mean = float(cell["mean"])
    sd = float(cell["sd"])
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "low": mean - half, "high": mean + half, "sd": sd}


def error(metric: Mapping[str, float]) -> np.ndarray:
    return np.asarray([[metric["mean"] - metric["low"]], [metric["high"] - metric["mean"]]])


def style_axis(ax: plt.Axes) -> None:
    ax.axhline(0, color="#65727C", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.grid(axis="y", color="#D7E1EB", linewidth=0.7, alpha=0.8, zorder=0)
    ax.tick_params(length=3, width=0.8, color="#65727C")
    ax.set_axisbelow(True)


def plot() -> None:
    groups = load_groups()
    tail_audit = load_tail_audit()
    source_rows: list[dict[str, Any]] = []

    fig = plt.figure(figsize=(7.205, 6.0), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], hspace=0.59, wspace=0.43)

    # a — the protocol-level effect: raw candidates versus what the locked policy served.
    ax = fig.add_subplot(grid[0, 0])
    suite_specs = [
        ("S2", "S2_unified_adapter_baselines", "raw_T_gain_points", "served_T_gain_points"),
        ("S3", "S3_cross_backbone_controls", "raw_T_gain_points", "served_T_gain_points"),
        ("S4", "S4_budget_and_geometry_sensitivity", "raw_T_gain_points", "served_T_gain_points"),
        ("S5", "S5_multieurlex_fourth_untouched_external", "raw_candidate_gain_points", "served_gain_points"),
    ]
    positions = np.arange(len(suite_specs), dtype=float)
    width = 0.33
    for offset, stage, color in ((-width / 2, "Raw candidate", GREY), (width / 2, "Served policy", NAVY)):
        for index, (short, suite, raw_field, served_field) in enumerate(suite_specs):
            metric = pool_metric(matching(groups, suite=suite), raw_field if stage == "Raw candidate" else served_field)
            ax.bar(positions[index] + offset, metric["mean"], width=width, color=color, edgecolor="white", linewidth=0.6, zorder=2)
            ax.errorbar(positions[index] + offset, metric["mean"], yerr=error(metric), color=DARK, capsize=2.8, linewidth=0.9, zorder=3)
            source_rows.append({"panel": "a", "group": short, "series": stage, **metric})
    ax.set_xticks(positions, [spec[0] for spec in suite_specs])
    ax.set_ylabel("Macro-F1 gain (pp)")
    ax.set_title("a  Authorization changes service", loc="left", fontweight="bold")
    ax.add_patch(Rectangle((0.04, 0.875), 0.035, 0.045, transform=ax.transAxes, facecolor=GREY, edgecolor="none", linewidth=0, clip_on=False))
    ax.text(0.09, 0.90, "Raw candidate", transform=ax.transAxes, color=DARK, fontsize=11, va="center")
    ax.add_patch(Rectangle((0.04, 0.715), 0.035, 0.045, transform=ax.transAxes, facecolor=NAVY, edgecolor="none", linewidth=0, clip_on=False))
    ax.text(0.09, 0.74, "Served policy", transform=ax.transAxes, color=DARK, fontsize=11, va="center")
    style_axis(ax)
    ax.grid(False)

    # b — equal-information action comparison; capacity is not a safety guarantee.
    ax = fig.add_subplot(grid[0, 1])
    action_specs = [
        ("Full tune", "full_tune"),
        ("Vector scale", "vector_logit_scaling"),
        ("Bottleneck", "bottleneck"),
        ("Head only", "classifier_head_only"),
        ("LoRA", "lora"),
    ]
    y = np.arange(len(action_specs))[::-1]
    for position, (display, action) in zip(y, action_specs):
        rows = matching(groups, suite="S2_unified_adapter_baselines", action=action)
        raw = pool_metric(rows, "raw_T_gain_points")
        served = pool_metric(rows, "served_T_gain_points")
        ax.plot([raw["mean"], served["mean"]], [position, position], color=PALE_BLUE, linewidth=2.1, zorder=1)
        ax.errorbar(raw["mean"], position, xerr=error(raw), fmt="o", color=GREY, markersize=5.3, capsize=2.5, zorder=3)
        ax.errorbar(served["mean"], position, xerr=error(served), fmt="o", color=NAVY, markersize=5.3, capsize=2.5, zorder=3)
        source_rows.extend([
            {"panel": "b", "group": display, "series": "Raw candidate", **raw},
            {"panel": "b", "group": display, "series": "Served policy", **served},
        ])
    ax.axvline(0, color="#65727C", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y, [display for display, _ in action_specs])
    ax.set_xlabel("Macro-F1 gain (pp)")
    ax.set_title("b  Capacity does not prevent tail loss", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D7E1EB", linewidth=0.7, alpha=0.8)

    # c — Jigsaw intervention: threshold movement, not temperature, reverses the sign.
    ax = fig.add_subplot(grid[1, 0])
    s1 = "S1_jigsaw_intervention_and_identity_audit"
    intervention_specs = [
        ("Raw", "raw_alpha_1"),
        ("Temperature", "cert_temperature"),
        ("Scalar", "cert_scalar_unbounded"),
        ("Clipped scalar", "cert_scalar_clipped_0_1"),
        ("Half scalar", "cert_scalar_shrunk_50_percent"),
        ("Threshold", "cert_macro_f1_threshold"),
        ("Scalar + threshold", "cert_scalar_plus_macro_f1_threshold"),
    ]
    metrics = [direct_metric(next(iter(matching(groups, suite=s1, action=key))), "deploy_gain_points") for _, key in intervention_specs]
    y = np.arange(len(metrics))[::-1]
    colors = [RED if metric["mean"] < 0 else GREEN for metric in metrics]
    ax.barh(y, [metric["mean"] for metric in metrics], color=colors, edgecolor="white", linewidth=0.6, zorder=2)
    for position, (display, _), metric in zip(y, intervention_specs, metrics):
        ax.errorbar(metric["mean"], position, xerr=error(metric), color=DARK, capsize=2.8, linewidth=0.9, zorder=3)
        source_rows.append({"panel": "c", "group": display, "series": "Deploy gain", **metric})
    ax.set_yticks(y, [label for label, _ in intervention_specs])
    ax.set_xlabel("Deploy gain (pp)")
    ax.set_title("c  Thresholding reverses Jigsaw failure", loc="left", fontweight="bold")
    ax.axvline(0, color="#65727C", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.grid(axis="x", color="#D7E1EB", linewidth=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # d — prospective MultiEURLEX actions plus post-outcome frozen-base attribution controls.
    ax = fig.add_subplot(grid[1, 1])
    s5 = "S5_multieurlex_fourth_untouched_external"
    action_specs_s5 = [
        ("Full tune", "full_tune"),
        ("LoRA", "lora"),
        ("Bottleneck", "bottleneck"),
        ("Head only", "classifier_head_only"),
        ("Vector scale", "vector_logit_scaling"),
        ("ATA residual", "ata_output_residual"),
    ]
    base_control_specs = [
        ("Base global τ†", "global_threshold", GREEN),
        ("Base label τ†", "per_label_thresholds", ORANGE),
    ]
    all_labels = [display for display, _ in action_specs_s5] + [display for display, _, _ in base_control_specs]
    y = np.arange(len(all_labels), dtype=float)[::-1]
    stage_specs = [
        ("Raw", "raw_candidate_gain_points", GREY, 0.18),
        ("Authorize", "cert_gate_gain_points", MID_BLUE, 0.0),
        ("Deploy", "served_gain_points", NAVY, -0.18),
    ]
    for stage, field, color, offset in stage_specs:
        for position, (display, action) in zip(y[: len(action_specs_s5)], action_specs_s5):
            group = next(iter(matching(groups, suite=s5, action=action)))
            metric = direct_metric(group, field)
            ax.errorbar(metric["mean"], position + offset, xerr=error(metric), fmt="o", color=color, markersize=4.8, capsize=2.3, linewidth=0.9, zorder=3)
            source_rows.append({"panel": "d", "group": display, "series": stage, **metric})

    controls = tail_audit["base_only_calibration_controls"]
    for position, (display, key, color) in zip(y[-len(base_control_specs):], base_control_specs):
        metric = tail_summary_metric(controls[key]["summary"])
        ax.errorbar(metric["mean"], position, xerr=error(metric), fmt="D", color=color, markersize=5.2, capsize=2.5, linewidth=1.0, zorder=4)
        source_rows.append({"panel": "d", "group": display, "series": "Post-outcome base-only", **metric})
    ax.axhline(1.5, color=PALE_BLUE, linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
    ax.set_yticks(y, all_labels)
    ax.set_ylim(-0.55, float(y[0]) + 0.55)
    ax.set_xlabel("Macro-F1 gain (pp)")
    ax.set_title("d  Calibration explains prospective gain", loc="left", fontweight="bold")
    ax.axvline(0, color="#65727C", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.grid(axis="x", color="#D7E1EB", linewidth=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.012,
        "Points are means; bars are descriptive 95% cell-level intervals. † Post-outcome frozen-base attribution controls.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#4D5963",
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.145)

    with SOURCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["panel", "group", "series", "n", "mean", "low", "high", "sd"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
