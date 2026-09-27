from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
SCRIPTS = PACKAGE_ROOT / "scripts"
for value in (SRC, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_prospective_v11_job as shared

from ata_nlp.closure_v10 import (
    build_candidate_v10,
    count_parameters,
    evaluate_logits,
    fit_residual_scalar_by_macro_f1,
    fit_temperature_by_loss,
    hashed_record_id,
    load_window_dataset,
    make_loader,
    require_cuda,
    seed_everything,
    task_metrics,
)
from ata_nlp.prospective_v11 import (
    block_bootstrap_lcb,
    file_sha256,
    load_json,
    residual_coordinates,
    temporal_block_indices,
    write_json_once,
)


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_select_certify_v12.draft.json"
DEFAULT_REVISIONS = PACKAGE_ROOT / "locks" / "prospective_select_certify_v12.revisions.lock.json"
DEFAULT_EXECUTION = PACKAGE_ROOT / "locks" / "prospective_select_certify_v12.execution.lock.json"


def verify_context(args):
    config, manifest = load_json(args.config), load_json(args.manifest)
    revisions, execution = load_json(args.revision_lock), load_json(args.execution_lock)
    if revisions.get("status") != "resolved_before_v12_model_or_deployment_outcomes":
        raise PermissionError("Missing frozen v12 revision lock.")
    if execution.get("status") != "frozen_before_v12_model_or_deployment_outcomes":
        raise PermissionError("Missing frozen v12 execution lock.")
    for key, path in (
        ("config", args.config),
        ("manifest", args.manifest),
        ("revision_lock", args.revision_lock),
        ("runner", Path(__file__)),
    ):
        if execution["bound_files"].get(key) != file_sha256(path):
            raise ValueError(f"Frozen {key} differs from the current file.")
    for key, relative in execution.get("bound_code_paths", {}).items():
        path = PROJECT_ROOT / relative
        if execution["bound_files"].get(key) != file_sha256(path):
            raise ValueError(f"Frozen code dependency {key} differs from the current file.")
    return config, manifest, revisions, execution


def prepared_context(config, execution, job):
    root = PROJECT_ROOT / config["paths"]["prepared_root"] / job["dataset_id"] / "post_quarantine_v1"
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != execution["data_manifest_sha256"][job["dataset_id"]]:
        raise ValueError("Prepared-data manifest changed after sealing.")
    prepared_manifest = load_json(manifest_path)
    origin = root / job["origin_id"]
    phase_roles = {
        "v12_history_train": ("history",),
        "v12_candidate_train_calibrate_select_certify": ("adapt", "cert_cal", "cert_select", "cert_gate"),
        "v12_counterfactual_deploy_audit_after_global_barrier": ("deploy",),
    }
    for role in phase_roles[job["phase"]]:
        path = origin / f"{role}.jsonl"
        key = f"{job['origin_id']}/{role}.jsonl"
        if file_sha256(path) != prepared_manifest["file_sha256"][key]:
            raise ValueError(f"Prepared role changed after sealing: {path}")
    return origin, prepared_manifest


def history_dependency(config, manifest, job):
    ordinary = [value for value in job["dependencies"] if not str(value).startswith(("action_lock::", "global_pre_T_barrier::"))]
    for dependency in ordinary:
        parent = shared.job_by_id(manifest, dependency)
        if parent["phase"] == "v12_history_train":
            checkpoint, result = shared.history_paths(config, parent)
            artifact = load_json(result)
            if artifact["checkpoint_digest"] != shared.checkpoint_digest(checkpoint):
                raise ValueError("History checkpoint changed after serialization.")
            return checkpoint
    raise FileNotFoundError("History dependency is missing.")


def calibrated_pair(base_value, candidate_value, base_bias, candidate_bias, alpha):
    candidate_logits = base_value.logits + float(alpha) * (candidate_value.logits - base_value.logits)
    base_cal = evaluate_logits(
        shared.apply_bias(base_value.logits, base_bias),
        base_value.labels,
        base_value.deploy_bins,
        task_type="sequence_classification",
    )
    candidate_cal = evaluate_logits(
        shared.apply_bias(candidate_logits, candidate_bias),
        base_value.labels,
        base_value.deploy_bins,
        task_type="sequence_classification",
    )
    return base_cal, candidate_cal


def write_prediction_sidecar(
    config: Mapping[str, Any], job: Mapping[str, Any], role: str, prepared: Path, base_value, candidate_value
) -> tuple[str, str]:
    destination = (
        PROJECT_ROOT / config["paths"]["prediction_root"] / job["dataset_id"] / job["origin_id"]
        / f"seed{job['seed']}" / f"{job.get('action', 'portfolio')}__{role}__{job['job_id']}.jsonl"
    )
    rows = []
    with (prepared / f"{role}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if len(rows) != int(base_value.logits.shape[0]) or len(rows) != int(candidate_value.logits.shape[0]):
        raise ValueError(f"Prediction/identity row mismatch for {role}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows):
                payload = {
                    "record_hash": hashed_record_id(
                        row["record_id"], salt=f"ata-v12:{job['dataset_id']}:{job['origin_id']}:{role}"
                    ),
                    "label": int(base_value.labels[index]),
                    "deploy_bin": int(base_value.deploy_bins[index]),
                    "base_logits": [float(value) for value in base_value.logits[index].tolist()],
                    "served_logits": [float(value) for value in candidate_value.logits[index].tolist()],
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(destination)
    return str(destination), file_sha256(destination)


def run_candidate(config, manifest, execution, args, job, spec, prepared, device) -> None:
    import torch
    from transformers import get_linear_schedule_with_warmup
    from ata_nlp.runtime import train_steps

    checkpoint, output = shared.candidate_paths(config, job)
    if output.exists():
        return
    history = history_dependency(config, manifest, job)
    tokenizer = shared.load_tokenizer(history)
    base = shared.load_base(history).to(device)
    candidate = build_candidate_v10(
        shared.load_base(history),
        action=job["action"],
        task_type="sequence_classification",
        rank=int(config["runtime"]["candidates"][job["action"]].get("rank", 4)),
    ).to(device)
    windows = ("adapt", "cert_cal", "cert_select", "cert_gate")
    data = load_window_dataset(
        prepared, tokenizer, windows=windows,
        task_type="sequence_classification", max_length=int(spec["max_length"]),
    )
    runtime = config["runtime"]["candidates"][job["action"]]
    loader = make_loader(data["adapt"], batch_size=int(runtime["micro_batch_size"]), shuffle=True, seed=int(job["seed"]))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=float(runtime["learning_rate"]), weight_decay=float(runtime["weight_decay"]),
    )
    steps = int(config["runtime"]["candidate_steps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(runtime["warmup_steps"]), steps)
    training = train_steps(
        candidate, loader, optimizer=optimizer, device=device, steps=steps,
        precision=config["runtime"]["mixed_precision"],
        gradient_accumulation_steps=int(runtime["gradient_accumulation_steps"]), scheduler=scheduler,
    )
    digest = shared.serialize_candidate(candidate, checkpoint, job["action"], tokenizer)

    evaluations = {}
    runtimes = {}
    for role in ("cert_cal", "cert_select", "cert_gate"):
        role_loader = make_loader(
            data[role], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]),
            shuffle=False, seed=int(job["seed"]),
        )
        base_value, base_runtime = shared.timed_evaluate(base, role_loader, device=device)
        candidate_value, candidate_runtime = shared.timed_evaluate(candidate, role_loader, device=device)
        evaluations[role] = (base_value, candidate_value)
        runtimes[role] = {"base": base_runtime, "candidate": candidate_runtime}

    base_cal_fit, candidate_cal_fit = evaluations["cert_cal"]
    scalar = fit_residual_scalar_by_macro_f1(
        base_cal_fit.logits,
        candidate_cal_fit.logits - base_cal_fit.logits,
        base_cal_fit.labels,
        task_type="sequence_classification", lower=0.0, upper=1.5, steps=151,
    )
    alpha = float(scalar["alpha"])
    fitted_candidate = evaluate_logits(
        base_cal_fit.logits + alpha * (candidate_cal_fit.logits - base_cal_fit.logits),
        base_cal_fit.labels, base_cal_fit.deploy_bins, task_type="sequence_classification",
    )
    base_temperature = fit_temperature_by_loss(
        base_cal_fit.logits, base_cal_fit.labels, task_type="sequence_classification"
    )
    base_bias = shared.fit_class_bias(base_cal_fit.logits, base_cal_fit.labels)
    candidate_bias = shared.fit_class_bias(fitted_candidate.logits, fitted_candidate.labels)

    base_select, candidate_select = calibrated_pair(
        *evaluations["cert_select"], base_bias, candidate_bias, alpha
    )
    base_gate, candidate_gate = calibrated_pair(
        *evaluations["cert_gate"], base_bias, candidate_bias, alpha
    )
    select_gain = 100.0 * (
        candidate_select.metrics["macro_f1"] - base_select.metrics["macro_f1"]
    )
    gate_gain = 100.0 * (candidate_gate.metrics["macro_f1"] - base_gate.metrics["macro_f1"])
    block_gains = shared.metric_gain_by_blocks(
        base_gate, candidate_gate, int(config["global_rules"]["certification_blocks"])
    )
    gate_interval = block_bootstrap_lcb(
        block_gains,
        draws=int(config["global_rules"]["bootstrap_draws"]),
        confidence=float(config["global_rules"]["bootstrap_confidence"]),
        seed=int(job["job_id"], 16) % (2**31),
    )
    coordinate_blocks = []
    raw_base_gate, raw_candidate_gate = evaluations["cert_gate"]
    for indices in temporal_block_indices(len(raw_base_gate.labels), int(config["global_rules"]["certification_blocks"])):
        selected = torch.as_tensor(indices, dtype=torch.long)
        coordinate_blocks.append(
            residual_coordinates(
                raw_base_gate.logits[selected], candidate_gate.logits[selected], raw_base_gate.labels[selected]
            )
        )
    select_sidecar = write_prediction_sidecar(
        config, job, "cert_select", prepared, base_select, candidate_select
    )
    gate_sidecar = write_prediction_sidecar(config, job, "cert_gate", prepared, base_gate, candidate_gate)
    write_json_once(output, {
        "schema_version": "ata-nlp-select-certify-certification-12.0",
        "stage": "C_cal_C_select_C_gate_complete_T_unopened",
        "job": job,
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_digest": digest,
        "training": training.to_dict(),
        "parameters": count_parameters(candidate),
        "cert_cal": {
            "base_raw": shared.evaluate_report(base_cal_fit),
            "candidate_raw": shared.evaluate_report(candidate_cal_fit),
            "candidate_scaled": shared.evaluate_report(fitted_candidate),
        },
        "cert_select": {
            "base_calibrated": shared.evaluate_report(base_select),
            "candidate_calibrated": shared.evaluate_report(candidate_select),
        },
        "cert_gate": {
            "base_calibrated": shared.evaluate_report(base_gate),
            "candidate_calibrated": shared.evaluate_report(candidate_gate),
        },
        "calibration": {
            "residual_scalar": scalar,
            "base_temperature_diagnostic": base_temperature,
            "base_class_bias": base_bias,
            "candidate_class_bias": candidate_bias,
        },
        "C_only_features": {
            "select_gain_points": select_gain,
            "gate_gain_points": gate_gain,
            "gate_block_gain_points": block_gains,
            "gate_gain_interval": gate_interval,
            "residual_coordinate_blocks": coordinate_blocks,
        },
        "prediction_sidecars": {
            "cert_select": {"path": select_sidecar[0], "sha256": select_sidecar[1]},
            "cert_gate": {"path": gate_sidecar[0], "sha256": gate_sidecar[1]},
        },
        "T_opened": False,
        "deploy_role_loaded": False,
        "runtime": runtimes,
        "execution_lock_sha256": file_sha256(args.execution_lock),
        "gpu": torch.cuda.get_device_name(device),
    })


def run_deploy(config, manifest, args, job, spec, prepared, device) -> None:
    import torch

    output = shared.deploy_result(config, job)
    if output.exists():
        return
    action_path = PROJECT_ROOT / config["paths"]["action_lock"]
    barrier_path = PROJECT_ROOT / config["paths"]["global_barrier"]
    action_lock, barrier = load_json(action_path), load_json(barrier_path)
    if barrier.get("status") != "all_48_actions_locked_before_any_v12_T_outcome":
        raise PermissionError("The complete v12 pre-T barrier is absent.")
    if barrier.get("action_lock_sha256") != file_sha256(action_path):
        raise ValueError("The action lock differs from the global barrier.")
    decision_id = f"{job['dataset_id']}::{job['origin_id']}::{int(job['seed'])}"
    decision = action_lock["decisions"][decision_id]
    nominee_id = decision["nominee_candidate_job_id"]
    ordinary_dependencies = [
        value for value in job["dependencies"]
        if not str(value).startswith(("action_lock::", "global_pre_T_barrier::"))
    ]
    if len(ordinary_dependencies) != 1:
        raise ValueError("A v12 counterfactual deployment job must bind exactly one candidate.")
    candidate_job = shared.job_by_id(manifest, ordinary_dependencies[0])
    if candidate_job["action"] != job["action"]:
        raise ValueError("Counterfactual deployment action/candidate mismatch.")
    cert_path = PROJECT_ROOT / config["paths"]["result_root"] / "certification" / f"{candidate_job['job_id']}.json"
    evidence = decision["all_six_C_only_artifacts"][job["action"]]
    if evidence["certification_sha256"] != file_sha256(cert_path):
        raise ValueError("The nominee certification artifact changed after locking.")
    cert = load_json(cert_path)
    history = history_dependency(config, manifest, candidate_job)
    tokenizer = shared.load_tokenizer(history)
    base = shared.load_base(history).to(device)

    # The deployment file is referenced only after the complete action lock and
    # barrier have been verified above.
    data = load_window_dataset(
        prepared, tokenizer, windows=("deploy",),
        task_type="sequence_classification", max_length=int(spec["max_length"]),
    )
    loader = make_loader(
        data["deploy"], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]),
        shuffle=False, seed=int(job["seed"]),
    )
    base_raw, base_runtime = shared.timed_evaluate(base, loader, device=device)
    base_cal = evaluate_logits(
        shared.apply_bias(base_raw.logits, cert["calibration"]["base_class_bias"]),
        base_raw.labels, base_raw.deploy_bins, task_type="sequence_classification",
    )
    checkpoint, _ = shared.candidate_paths(config, candidate_job)
    if cert["candidate_checkpoint_digest"] != shared.checkpoint_digest(checkpoint):
        raise ValueError("Counterfactual candidate checkpoint changed after certification.")
    candidate = shared.restore_candidate(history, checkpoint, job["action"]).to(device)
    candidate_raw, candidate_runtime = shared.timed_evaluate(candidate, loader, device=device)
    _, candidate_cal = calibrated_pair(
        base_raw,
        candidate_raw,
        cert["calibration"]["base_class_bias"],
        cert["calibration"]["candidate_class_bias"],
        float(cert["calibration"]["residual_scalar"]["alpha"]),
    )
    is_primary_served_action = bool(
        decision["authorized"] and decision["nominee_action"] == job["action"]
    )
    served = candidate_cal if is_primary_served_action else base_cal
    counterfactual_gain = 100.0 * (
        candidate_cal.metrics["macro_f1"] - base_cal.metrics["macro_f1"]
    )
    primary_gain = 100.0 * (served.metrics["macro_f1"] - base_cal.metrics["macro_f1"])
    counterfactual_bin_gains = {}
    primary_bin_gains = {}
    for bin_id in range(int(config["global_rules"]["deploy_bins"])):
        indices = [index for index, value in enumerate(base_cal.deploy_bins) if value == bin_id]
        if not indices:
            continue
        selected = torch.as_tensor(indices, dtype=torch.long)
        base_score = task_metrics(
            base_cal.logits[selected], base_cal.labels[selected], task_type="sequence_classification"
        )["macro_f1"]
        candidate_score = task_metrics(
            candidate_cal.logits[selected], candidate_cal.labels[selected], task_type="sequence_classification"
        )["macro_f1"]
        served_score = task_metrics(
            served.logits[selected], served.labels[selected], task_type="sequence_classification"
        )["macro_f1"]
        counterfactual_bin_gains[str(bin_id)] = 100.0 * (candidate_score - base_score)
        primary_bin_gains[str(bin_id)] = 100.0 * (served_score - base_score)
    deploy_sidecar = write_prediction_sidecar(config, job, "deploy", prepared, base_cal, candidate_cal)
    write_json_once(output, {
        "schema_version": "ata-nlp-select-certify-deployment-12.0",
        "stage": "T_opened_after_all_48_actions_locked",
        "job": job,
        "decision_id": decision_id,
        "nominee_candidate_job_id": nominee_id,
        "nominee_action": decision["nominee_action"],
        "authorized": bool(decision["authorized"]),
        "audited_candidate_job_id": candidate_job["job_id"],
        "audited_action": job["action"],
        "is_primary_served_action": is_primary_served_action,
        "served_action": decision["serialized_action"],
        "base_T_calibrated": shared.evaluate_report(base_cal),
        "served_T_calibrated": shared.evaluate_report(served),
        "counterfactual_candidate_T_calibrated": shared.evaluate_report(candidate_cal),
        "outcomes": {
            "primary_system_gain_points": primary_gain,
            "primary_system_bin_gain_points": primary_bin_gains,
            "counterfactual_candidate_gain_points": counterfactual_gain,
            "counterfactual_candidate_bin_gain_points": counterfactual_bin_gains,
        },
        "prediction_sidecar": {"path": deploy_sidecar[0], "sha256": deploy_sidecar[1]},
        "action_lock_sha256": file_sha256(action_path),
        "global_barrier_sha256": file_sha256(barrier_path),
        "runtime": {"base": base_runtime, "candidate": candidate_runtime},
        "gpu": torch.cuda.get_device_name(device),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one immutable v12 GPU job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--revision-lock", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION)
    args = parser.parse_args()
    config, manifest, revisions, execution = verify_context(args)
    job = shared.job_by_id(manifest, args.job_id)
    spec = shared.dataset_spec(config, job["dataset_id"])
    prepared, data_manifest = prepared_context(config, execution, job)
    device = require_cuda()
    seed_everything(int(job["seed"]))
    if job["phase"] == "v12_history_train":
        shared.run_history(config, revisions, execution, args, job, spec, prepared, data_manifest, device)
    elif job["phase"] == "v12_candidate_train_calibrate_select_certify":
        run_candidate(config, manifest, execution, args, job, spec, prepared, device)
    elif job["phase"] == "v12_counterfactual_deploy_audit_after_global_barrier":
        run_deploy(config, manifest, args, job, spec, prepared, device)
    else:
        raise ValueError(job["phase"])


if __name__ == "__main__":
    main()
