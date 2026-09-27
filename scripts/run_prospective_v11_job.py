from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ata_nlp.closure_v10 import (
    build_candidate_v10,
    count_parameters,
    evaluate_logits,
    evaluate_model,
    fit_residual_scalar_by_macro_f1,
    fit_temperature_by_loss,
    load_window_dataset,
    make_loader,
    require_cuda,
    seed_everything,
    task_metrics,
)
from ata_nlp.prospective_v11 import (
    block_bootstrap_lcb,
    content_hash,
    file_sha256,
    load_json,
    residual_coordinates,
    temporal_block_indices,
    write_json_once,
)


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_closure_v11.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "prospective_closure_v11.draft.json"
DEFAULT_REVISIONS = PACKAGE_ROOT / "locks" / "prospective_closure_v11.revisions.lock.json"
DEFAULT_EXECUTION = PACKAGE_ROOT / "locks" / "prospective_closure_v11.execution.lock.json"


def slug(value: str) -> str:
    return value.replace("/", "--")


def job_by_id(manifest: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in manifest["jobs"] if row["job_id"] == job_id]
    if len(matches) != 1:
        raise KeyError(job_id)
    return matches[0]


def dataset_spec(config: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in config["datasets"] if row["dataset_id"] == dataset_id]
    if len(matches) != 1:
        raise KeyError(dataset_id)
    return matches[0]


def checkpoint_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(directory)
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def history_paths(config: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[Path, Path]:
    checkpoint = (
        PROJECT_ROOT / config["paths"]["checkpoint_root"] / "history" / job["dataset_id"]
        / slug(job["backbone"]) / job["origin_id"] / f"seed{job['seed']}__{job['job_id']}"
    )
    result = PROJECT_ROOT / config["paths"]["result_root"] / "history" / f"{job['job_id']}.json"
    return checkpoint, result


def candidate_paths(config: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[Path, Path]:
    checkpoint = (
        PROJECT_ROOT / config["paths"]["checkpoint_root"] / "candidates" / job["dataset_id"]
        / job["action"] / job["origin_id"] / f"seed{job['seed']}__{job['job_id']}"
    )
    result = PROJECT_ROOT / config["paths"]["result_root"] / "certification" / f"{job['job_id']}.json"
    return checkpoint, result


def deploy_result(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["result_root"] / "deployment" / f"{job['job_id']}.json"


def prepared_context(config: Mapping[str, Any], execution: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = PROJECT_ROOT / config["paths"]["prepared_root"] / job["dataset_id"] / "post_quarantine_v1"
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != execution["data_manifest_sha256"][job["dataset_id"]]:
        raise ValueError("Prepared-data manifest changed after sealing.")
    manifest = load_json(manifest_path)
    origin = root / job["origin_id"]
    phase_roles = {
        "prospective_history_train": ("history",),
        "prospective_candidate_train_certify": ("adapt", "cert_fit", "cert_gate"),
        "prospective_deploy_after_global_barrier": ("deploy",),
    }
    for role in phase_roles[job["phase"]]:
        path = origin / f"{role}.jsonl"
        key = f"{job['origin_id']}/{role}.jsonl"
        if file_sha256(path) != manifest["file_sha256"][key]:
            raise ValueError(f"Prepared role changed after sealing: {path}")
    return origin, manifest


def verify_context(args) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, manifest = load_json(args.config), load_json(args.manifest)
    revisions, execution = load_json(args.revision_lock), load_json(args.execution_lock)
    if revisions.get("status") != "resolved_before_v11_model_outcomes":
        raise PermissionError("Missing frozen v11 revision lock.")
    if execution.get("status") != "frozen_before_v11_model_outcomes":
        raise PermissionError("Missing frozen v11 execution lock.")
    for key, path in (("config", args.config), ("manifest", args.manifest), ("revision_lock", args.revision_lock), ("runner", Path(__file__))):
        if execution["bound_files"].get(key) != file_sha256(path):
            raise ValueError(f"Frozen {key} differs from the current file.")
    return config, manifest, revisions, execution


def load_tokenizer(path: str | Path, revision: str | None = None):
    from transformers import AutoTokenizer
    kwargs = {"use_fast": True}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(path, **kwargs)


def load_base(path: Path):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(path)


def history_dependency(config: Mapping[str, Any], manifest: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    dependencies = [value for value in job["dependencies"] if not str(value).startswith(("group_lock::", "wave_barrier::"))]
    for dependency in dependencies:
        parent = job_by_id(manifest, dependency)
        if parent["phase"] == "prospective_history_train":
            checkpoint, result = history_paths(config, parent)
            artifact = load_json(result)
            if artifact["checkpoint_digest"] != checkpoint_digest(checkpoint):
                raise ValueError("History checkpoint changed after result serialization.")
            return checkpoint
    raise FileNotFoundError("History dependency is missing.")


def timed_evaluate(model, loader, *, device):
    import torch
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    value = evaluate_model(model, loader, device=device, task_type="sequence_classification")
    torch.cuda.synchronize(device)
    return value, {
        "elapsed_seconds": time.perf_counter() - start,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def evaluate_report(value) -> dict[str, Any]:
    return value.compact()


def metric_gain_by_blocks(base, candidate, blocks: int) -> list[float]:
    import torch
    gains = []
    for indices in temporal_block_indices(len(base.labels), blocks):
        selected = torch.as_tensor(indices, dtype=torch.long)
        base_metric = task_metrics(base.logits[selected], base.labels[selected], task_type="sequence_classification")["macro_f1"]
        candidate_metric = task_metrics(candidate.logits[selected], candidate.labels[selected], task_type="sequence_classification")["macro_f1"]
        gains.append(100.0 * (candidate_metric - base_metric))
    return gains


def fit_class_bias(logits, labels) -> dict[str, Any]:
    import torch
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    bias = torch.zeros(logits.shape[-1], requires_grad=True)
    optimizer = torch.optim.LBFGS([bias], lr=0.25, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits + bias, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    centered = (bias.detach() - bias.detach().mean()).tolist()
    return {"bias": [float(value) for value in centered], "fit_cross_entropy": float(torch.nn.functional.cross_entropy(logits + torch.tensor(centered), labels))}


def apply_bias(logits, fit: Mapping[str, Any]):
    import torch
    return logits + torch.as_tensor(fit["bias"], dtype=logits.dtype)


def state_stability(base_fit, base_gate, candidate_fit, candidate_gate) -> float:
    import torch
    fit_probs = torch.softmax(base_fit.logits.float(), -1).mean(0)
    gate_probs = torch.softmax(base_gate.logits.float(), -1).mean(0)
    midpoint = 0.5 * (fit_probs + gate_probs)
    js = 0.5 * (
        torch.sum(fit_probs * (fit_probs.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()))
        + torch.sum(gate_probs * (gate_probs.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()))
    )
    fit_norm = (candidate_fit.logits - base_fit.logits).norm(dim=-1).mean()
    gate_norm = (candidate_gate.logits - base_gate.logits).norm(dim=-1).mean()
    norm_shift = (gate_norm - fit_norm).abs() / fit_norm.clamp_min(1e-8)
    return float((js + norm_shift).item())


def serialize_candidate(candidate, checkpoint: Path, action: str, tokenizer) -> str:
    import torch
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    if action == "full_tune":
        candidate.save_pretrained(checkpoint, safe_serialization=True)
        kind = "full_pretrained"
    else:
        state = {name: parameter.detach().cpu() for name, parameter in candidate.named_parameters() if parameter.requires_grad}
        if not state:
            raise RuntimeError("Candidate has no trainable parameters.")
        torch.save(state, checkpoint / "trainable_state.pt")
        kind = "trainable_state"
    tokenizer.save_pretrained(checkpoint)
    (checkpoint / "v11_checkpoint_meta.json").write_text(json.dumps({"action": action, "kind": kind}, indent=2) + "\n", encoding="utf-8")
    return checkpoint_digest(checkpoint)


def restore_candidate(history: Path, checkpoint: Path, action: str):
    import torch
    from transformers import AutoModelForSequenceClassification
    if action == "full_tune":
        return AutoModelForSequenceClassification.from_pretrained(checkpoint)
    candidate = build_candidate_v10(load_base(history), action=action, task_type="sequence_classification", rank=4)
    state = torch.load(checkpoint / "trainable_state.pt", map_location="cpu", weights_only=True)
    incompatible = candidate.load_state_dict(state, strict=False)
    expected = {name for name, parameter in candidate.named_parameters() if parameter.requires_grad}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) & expected:
        raise ValueError(f"Candidate restore mismatch: {incompatible}")
    return candidate


def run_history(config, revisions, execution, args, job, spec, prepared, data_manifest, device) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from ata_nlp.runtime import train_steps
    checkpoint, output = history_paths(config, job)
    if output.exists():
        return
    revision = revisions["models"][job["backbone"]]
    tokenizer = load_tokenizer(job["backbone"], revision)
    names = list(data_manifest["label_names"])
    model = AutoModelForSequenceClassification.from_pretrained(
        job["backbone"], revision=revision, num_labels=len(names),
        label2id={name: index for index, name in enumerate(names)},
        id2label={index: name for index, name in enumerate(names)}, ignore_mismatched_sizes=True,
    ).to(device)
    dataset = load_window_dataset(prepared, tokenizer, windows=("history",), task_type="sequence_classification", max_length=int(spec["max_length"]))
    runtime = config["runtime"]["history"]
    loader = make_loader(dataset["history"], batch_size=int(runtime["micro_batch_size"]), shuffle=True, seed=int(job["seed"]))
    steps = int(runtime["epochs"]) * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime["learning_rate"]), weight_decay=float(runtime["weight_decay"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, round(steps * float(runtime["warmup_ratio"])), steps)
    training = train_steps(model, loader, optimizer=optimizer, device=device, steps=steps, precision=runtime["mixed_precision"], gradient_accumulation_steps=int(runtime["gradient_accumulation_steps"]), scheduler=scheduler)
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    write_json_once(output, {
        "schema_version": "ata-nlp-prospective-history-11.0", "stage": "history_complete_A_C_T_unopened",
        "job": job, "checkpoint": str(checkpoint), "checkpoint_digest": checkpoint_digest(checkpoint),
        "training": training.to_dict(), "parameters": count_parameters(model), "model_revision": revision,
        "execution_lock_sha256": file_sha256(args.execution_lock), "gpu": torch.cuda.get_device_name(device),
    })


def run_candidate(config, manifest, execution, args, job, spec, prepared, device) -> None:
    import torch
    from transformers import get_linear_schedule_with_warmup
    from ata_nlp.runtime import train_steps
    checkpoint, output = candidate_paths(config, job)
    if output.exists():
        return
    history = history_dependency(config, manifest, job)
    tokenizer = load_tokenizer(history)
    # Keep a physically separate frozen evaluator.  Head-only and full tuning
    # otherwise mutate the same object and silently contaminate the baseline.
    base = load_base(history).to(device)
    candidate_backbone = load_base(history)
    candidate = build_candidate_v10(candidate_backbone, action=job["action"], task_type="sequence_classification", rank=int(config["runtime"]["candidates"][job["action"]].get("rank", 4))).to(device)
    data = load_window_dataset(prepared, tokenizer, windows=("adapt", "cert_fit", "cert_gate"), task_type="sequence_classification", max_length=int(spec["max_length"]))
    runtime = config["runtime"]["candidates"][job["action"]]
    loader = make_loader(data["adapt"], batch_size=int(runtime["micro_batch_size"]), shuffle=True, seed=int(job["seed"]))
    optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=float(runtime["learning_rate"]), weight_decay=float(runtime["weight_decay"]))
    steps = int(config["runtime"]["candidate_steps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(runtime["warmup_steps"]), steps)
    training = train_steps(candidate, loader, optimizer=optimizer, device=device, steps=steps, precision=config["runtime"]["mixed_precision"], gradient_accumulation_steps=int(runtime["gradient_accumulation_steps"]), scheduler=scheduler)
    digest = serialize_candidate(candidate, checkpoint, job["action"], tokenizer)
    evaluations = {}
    runtimes = {}
    for role in ("cert_fit", "cert_gate"):
        role_loader = make_loader(data[role], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]), shuffle=False, seed=int(job["seed"]))
        base_value, base_runtime = timed_evaluate(base, role_loader, device=device)
        candidate_value, candidate_runtime = timed_evaluate(candidate, role_loader, device=device)
        evaluations[role] = (base_value, candidate_value)
        runtimes[role] = {"base": base_runtime, "candidate": candidate_runtime}
    base_fit, candidate_fit = evaluations["cert_fit"]
    base_gate, candidate_gate = evaluations["cert_gate"]
    scalar = fit_residual_scalar_by_macro_f1(base_fit.logits, candidate_fit.logits - base_fit.logits, base_fit.labels, task_type="sequence_classification", lower=0.0, upper=1.5, steps=151)
    alpha = float(scalar["alpha"])
    fit_scaled = evaluate_logits(base_fit.logits + alpha * (candidate_fit.logits - base_fit.logits), base_fit.labels, base_fit.deploy_bins, task_type="sequence_classification")
    gate_scaled = evaluate_logits(base_gate.logits + alpha * (candidate_gate.logits - base_gate.logits), base_gate.labels, base_gate.deploy_bins, task_type="sequence_classification")
    base_temperature = fit_temperature_by_loss(base_fit.logits, base_fit.labels, task_type="sequence_classification")
    base_bias = fit_class_bias(base_fit.logits, base_fit.labels)
    candidate_bias = fit_class_bias(fit_scaled.logits, fit_scaled.labels)
    base_gate_cal = evaluate_logits(apply_bias(base_gate.logits, base_bias), base_gate.labels, base_gate.deploy_bins, task_type="sequence_classification")
    gate_cal = evaluate_logits(apply_bias(gate_scaled.logits, candidate_bias), gate_scaled.labels, gate_scaled.deploy_bins, task_type="sequence_classification")
    block_gains = metric_gain_by_blocks(base_gate_cal, gate_cal, int(config["global_rules"]["certification_blocks"]))
    interval = block_bootstrap_lcb(block_gains, draws=int(config["global_rules"]["bootstrap_draws"]), confidence=float(config["global_rules"]["bootstrap_confidence"]), seed=int(job["job_id"], 16) % (2**31))
    coordinate_blocks = []
    for indices in temporal_block_indices(len(base_gate.labels), int(config["global_rules"]["certification_blocks"])):
        import torch as _torch
        selected = _torch.as_tensor(indices, dtype=_torch.long)
        coordinate_blocks.append(residual_coordinates(base_gate.logits[selected], gate_scaled.logits[selected], base_gate.labels[selected]))
    c_interval = block_bootstrap_lcb([row["c"] for row in coordinate_blocks], draws=int(config["global_rules"]["bootstrap_draws"]), confidence=float(config["global_rules"]["bootstrap_confidence"]), seed=(int(job["job_id"], 16) + 1) % (2**31))
    s_interval = block_bootstrap_lcb([row["s"] for row in coordinate_blocks], draws=int(config["global_rules"]["bootstrap_draws"]), confidence=float(config["global_rules"]["bootstrap_confidence"]), seed=(int(job["job_id"], 16) + 2) % (2**31))
    slope = 0.0 if len(block_gains) < 2 else (block_gains[-1] - block_gains[0]) / (len(block_gains) - 1)
    if interval["lcb"] <= 0:
        lease_bins = 0
    elif slope >= 0:
        lease_bins = int(config["global_rules"]["deploy_bins"])
    else:
        lease_bins = min(int(config["global_rules"]["deploy_bins"]), max(1, int(block_gains[-1] / max(-slope, 1e-12))))
    gate_gain = 100.0 * (gate_cal.metrics["macro_f1"] - base_gate_cal.metrics["macro_f1"])
    write_json_once(output, {
        "schema_version": "ata-nlp-prospective-certification-11.0", "stage": "C_complete_T_unopened",
        "job": job, "candidate_checkpoint": str(checkpoint), "candidate_checkpoint_digest": digest,
        "training": training.to_dict(), "parameters": count_parameters(candidate),
        "cert_fit": {"base": evaluate_report(base_fit), "candidate_raw": evaluate_report(candidate_fit), "candidate_scaled": evaluate_report(fit_scaled)},
        "cert_gate": {"base_calibrated": evaluate_report(base_gate_cal), "candidate_calibrated": evaluate_report(gate_cal)},
        "calibration": {"residual_scalar": scalar, "base_temperature": base_temperature, "base_class_bias": base_bias, "candidate_class_bias": candidate_bias},
        "C_only_features": {
            "gate_gain_points": gate_gain, "block_gain_points": block_gains, "gain_interval": interval,
            "state_stability_score": state_stability(base_fit, base_gate, candidate_fit, candidate_gate),
            "lease_bins": lease_bins, "block_gain_slope": slope,
            "rtc": {"coordinate_blocks": coordinate_blocks, "c_interval": c_interval, "s_interval": s_interval,
                    "point_decision": bool(interval["lcb"] > 0 and c_interval["lcb"] > 0 and s_interval["lcb"] > alpha / 2),
                    "development_only": bool(job.get("rtc_development_probe"))},
        },
        "T_opened": False, "deploy_role_loaded": False, "runtime": runtimes,
        "execution_lock_sha256": file_sha256(args.execution_lock), "gpu": torch.cuda.get_device_name(device),
    })


def run_deploy(config, manifest, execution, args, job, spec, prepared, device) -> None:
    import torch
    output = deploy_result(config, job)
    if output.exists():
        return
    barrier_path = PACKAGE_ROOT / "locks" / f"prospective_closure_v11.{job['wave']}.pre_T.lock.json"
    barrier = load_json(barrier_path)
    if barrier.get("status") != "wave_actions_locked_before_any_wave_T" or barrier.get("wave") != job["wave"]:
        raise PermissionError("The required v11 wave barrier is absent.")
    group_path = PACKAGE_ROOT / "locks" / "prospective_closure_v11_groups" / job["wave"] / job["dataset_id"] / f"{job['action']}.lock.json"
    if barrier["group_locks"].get(str(group_path.relative_to(PACKAGE_ROOT)).replace("\\", "/")) != file_sha256(group_path):
        raise ValueError("Group lock differs from the wave barrier.")
    group = load_json(group_path)
    cell = group["cells"][job["job_id"]]
    candidate_job = job_by_id(manifest, cell["candidate_job_id"])
    checkpoint, cert_path = candidate_paths(config, candidate_job)
    if file_sha256(cert_path) != cell["certification_sha256"] or checkpoint_digest(checkpoint) != cell["candidate_checkpoint_digest"]:
        raise ValueError("Certification/candidate changed after locking.")
    cert = load_json(cert_path)
    history = history_dependency(config, manifest, candidate_job)
    tokenizer = load_tokenizer(history)
    base = load_base(history).to(device)
    candidate = restore_candidate(history, checkpoint, job["action"]).to(device)
    # First deploy-file reference occurs only after all barrier checks above.
    data = load_window_dataset(prepared, tokenizer, windows=("deploy",), task_type="sequence_classification", max_length=int(spec["max_length"]))
    loader = make_loader(data["deploy"], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]), shuffle=False, seed=int(job["seed"]))
    base_t, base_runtime = timed_evaluate(base, loader, device=device)
    candidate_t, candidate_runtime = timed_evaluate(candidate, loader, device=device)
    alpha = float(cert["calibration"]["residual_scalar"]["alpha"])
    candidate_scaled_logits = base_t.logits + alpha * (candidate_t.logits - base_t.logits)
    base_temp = evaluate_logits(base_t.logits / float(cert["calibration"]["base_temperature"]["temperature"]), base_t.labels, base_t.deploy_bins, task_type="sequence_classification")
    base_cal = evaluate_logits(apply_bias(base_t.logits, cert["calibration"]["base_class_bias"]), base_t.labels, base_t.deploy_bins, task_type="sequence_classification")
    candidate_cal = evaluate_logits(apply_bias(candidate_scaled_logits, cert["calibration"]["candidate_class_bias"]), base_t.labels, base_t.deploy_bins, task_type="sequence_classification")
    raw_gain = 100.0 * (candidate_t.metrics["macro_f1"] - base_t.metrics["macro_f1"])
    calibrated_gain = 100.0 * (candidate_cal.metrics["macro_f1"] - base_cal.metrics["macro_f1"])
    bin_gains = {}
    for bin_id in range(int(config["global_rules"]["deploy_bins"])):
        indices = [index for index, value in enumerate(base_t.deploy_bins) if value == bin_id]
        if not indices:
            continue
        selected = torch.as_tensor(indices, dtype=torch.long)
        b = task_metrics(base_cal.logits[selected], base_cal.labels[selected], task_type="sequence_classification")["macro_f1"]
        c = task_metrics(candidate_cal.logits[selected], candidate_cal.labels[selected], task_type="sequence_classification")["macro_f1"]
        bin_gains[str(bin_id)] = 100.0 * (c - b)
    write_json_once(output, {
        "schema_version": "ata-nlp-prospective-deployment-11.0", "stage": "T_opened_after_wave_barrier",
        "job": job, "candidate_job_id": candidate_job["job_id"], "wave_barrier_sha256": file_sha256(barrier_path),
        "group_lock_sha256": file_sha256(group_path), "base_T_raw": evaluate_report(base_t),
        "base_T_temperature_calibrated": evaluate_report(base_temp), "base_T_calibrated": evaluate_report(base_cal), "candidate_T_raw": evaluate_report(candidate_t),
        "candidate_T_calibrated": evaluate_report(candidate_cal),
        "outcomes": {"raw_gain_points": raw_gain, "calibrated_gain_points": calibrated_gain, "bin_gain_points": bin_gains},
        "locked_actions": cell["actions"], "runtime": {"base": base_runtime, "candidate": candidate_runtime},
        "gpu": torch.cuda.get_device_name(device),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one immutable v11 GPU job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--revision-lock", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION)
    args = parser.parse_args()
    config, manifest, revisions, execution = verify_context(args)
    job = job_by_id(manifest, args.job_id)
    spec = dataset_spec(config, job["dataset_id"])
    prepared, data_manifest = prepared_context(config, execution, job)
    device = require_cuda()
    seed_everything(int(job["seed"]))
    if job["phase"] == "prospective_history_train":
        run_history(config, revisions, execution, args, job, spec, prepared, data_manifest, device)
    elif job["phase"] == "prospective_candidate_train_certify":
        run_candidate(config, manifest, execution, args, job, spec, prepared, device)
    elif job["phase"] == "prospective_deploy_after_global_barrier":
        run_deploy(config, manifest, execution, args, job, spec, prepared, device)
    else:
        raise ValueError(job["phase"])


if __name__ == "__main__":
    main()
