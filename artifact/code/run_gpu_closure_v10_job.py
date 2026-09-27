from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ata_nlp.closure_v10 import (
    IDENTITY_COLUMNS,
    build_candidate_v10,
    classification_calibration,
    content_hash,
    count_parameters,
    evaluate_logits,
    evaluate_model,
    file_sha256,
    fit_global_threshold_by_macro_f1,
    fit_residual_scalar_by_macro_f1,
    fit_temperature_by_loss,
    hashed_record_id,
    load_json,
    load_window_dataset,
    make_loader,
    read_jsonl,
    require_cuda,
    resolve_existing_path,
    seed_everything,
    subgroup_binary_metrics,
    task_metrics,
    task_predictions,
    write_json_once,
)


DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "gpu_closure_v10.draft.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "gpu_closure_v10.draft.json"
DEFAULT_REVISIONS = PACKAGE_ROOT / "locks" / "gpu_closure_v10_backbone_and_dataset_revisions.json"
DEFAULT_EXECUTION_LOCK = PACKAGE_ROOT / "locks" / "gpu_closure_v10.execution.lock.json"
DEFAULT_BARRIER = PACKAGE_ROOT / "locks" / "gpu_closure_v10_multieurlex_all_actions_pre_T.lock.json"
IDENTITY_SALT = "ata-v10-jigsaw-identity-sidecar-20260915"


def slug(value: str) -> str:
    return value.replace("/", "--")


def job_by_id(manifest: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in manifest["jobs"] if row["job_id"] == job_id]
    if len(matches) != 1:
        raise KeyError(f"Job was not found exactly once: {job_id}")
    return matches[0]


def suite_by_id(config: Mapping[str, Any], suite_id: str) -> dict[str, Any]:
    return next(dict(row) for row in config["suites"] if row["suite_id"] == suite_id)


def checkpoint_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Checkpoint is empty: {directory}")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def verify_context(args) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    revisions = load_json(args.revision_lock)
    execution = load_json(args.execution_lock)
    if revisions.get("status") != "resolved_before_v10_model_outcomes":
        raise PermissionError("The immutable revision lock is missing or invalid.")
    if execution.get("status") != "frozen_before_v10_model_outcomes":
        raise PermissionError("The v10 execution protocol has not been frozen.")
    registered = execution["bound_files"]
    checks = {
        "config": args.config,
        "manifest": args.manifest,
        "revision_lock": args.revision_lock,
        "runner": Path(__file__),
    }
    for name, path in checks.items():
        if registered.get(name) != file_sha256(path):
            raise ValueError(f"Frozen {name} differs from the current file.")
    return config, manifest, revisions, execution


def dataset_spec(config: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    if job["suite_id"] == "S5_multieurlex_fourth_untouched_external":
        source = suite_by_id(config, job["suite_id"])["source"]
        return {
            "task_type": source["task_type"],
            "num_labels": int(source["num_labels"]),
            "max_length": int(source["max_length"]),
        }
    binding = config["existing_dataset_bindings"][job["dataset_id"]]
    maximums = {
        "spatiotemporal_news_location_external": 256,
        "swiss_judgment_de_temporal_external": 256,
        "jigsaw_civil_comments_temporal_external": 256,
    }
    return {
        "task_type": binding["task_type"],
        "num_labels": int(binding["num_labels"]),
        "max_length": maximums[job["dataset_id"]],
    }


def prepared_context(config: Mapping[str, Any], job: Mapping[str, Any], execution: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if job["suite_id"] == "S5_multieurlex_fourth_untouched_external":
        root = PROJECT_ROOT / "data_nlp_ata_v10_multieurlex" / job["dataset_id"] / "post_quarantine_v1"
    else:
        binding = config["existing_dataset_bindings"][job["dataset_id"]]
        geometry = str(job.get("variant", {}).get("temporal_geometry", "reference"))
        if geometry not in {"reference", "v7_v9_original"}:
            root = PROJECT_ROOT / "data_nlp_ata_v10_geometry" / job["dataset_id"] / geometry
        else:
            root = resolve_existing_path(PROJECT_ROOT, binding["prepared_root_candidates"])
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() and job["suite_id"] != "S5_multieurlex_fourth_untouched_external":
        geometry_manifest = PROJECT_ROOT / "data_nlp_ata_v10_geometry" / "manifest.json"
        manifest_path = geometry_manifest if geometry_manifest.exists() else root.parent / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    if job["suite_id"] == "S5_multieurlex_fourth_untouched_external":
        expected = execution["data_manifest_hashes"]["multieurlex"]
    elif str(job.get("variant", {}).get("temporal_geometry", "reference")) not in {"reference", "v7_v9_original"}:
        expected = execution["data_manifest_hashes"]["geometry"]
    else:
        expected = execution["existing_input_locks"][job["dataset_id"]]["prepared_manifest_sha256"]
    if file_sha256(manifest_path) != expected:
        raise ValueError(f"Prepared-data manifest changed after execution sealing: {manifest_path}")
    origin = root / str(job["origin_id"])
    if not origin.is_dir():
        raise FileNotFoundError(origin)
    return origin, manifest


def required_windows(job: Mapping[str, Any]) -> tuple[str, ...]:
    phase = job["phase"]
    if "history_train" in phase:
        return ("history",)
    if phase == "posthoc_inference":
        return ("cert", "deploy")
    if phase == "posthoc_candidate_train_and_eval":
        return ("adapt", "cert", "deploy")
    if phase == "prospective_candidate_train_cert_fit_cert_gate":
        return ("adapt", "cert_fit", "cert_gate")
    if phase == "prospective_deploy_after_group_lock":
        return ("deploy",)
    raise ValueError(phase)


def verify_role_files(config: Mapping[str, Any], job: Mapping[str, Any], prepared_dir: Path, prepared_manifest: Mapping[str, Any], execution: Mapping[str, Any]) -> None:
    geometry = str(job.get("variant", {}).get("temporal_geometry", "reference"))
    for window in required_windows(job):
        path = prepared_dir / f"{window}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        if job["suite_id"] == "S5_multieurlex_fourth_untouched_external":
            expected = prepared_manifest["file_sha256"][f"{job['origin_id']}/{window}.jsonl"]
        elif geometry not in {"reference", "v7_v9_original"}:
            expected = prepared_manifest["datasets"][job["dataset_id"]]["variants"][geometry]["origins"][job["origin_id"]]["file_sha256"][window]
        else:
            expected = execution["existing_input_locks"][job["dataset_id"]]["prepared_role_file_sha256"][f"{job['origin_id']}/{window}.jsonl"]
        if file_sha256(path) != expected:
            raise ValueError(f"Prepared role file changed after sealing: {path}")


def generated_history_paths(config: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[Path, Path]:
    root = PROJECT_ROOT / config["paths"]["checkpoint_root"] / "history"
    checkpoint = root / job["suite_id"] / job["dataset_id"] / slug(job["backbone"]) / job["origin_id"] / f"seed{job['seed']}__{job['job_id']}"
    result = PROJECT_ROOT / config["paths"]["result_root"] / "history" / f"{job['job_id']}.json"
    return checkpoint, result


def candidate_paths(config: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[Path, Path]:
    checkpoint = (
        PROJECT_ROOT / config["paths"]["checkpoint_root"] / "candidates" / job["suite_id"]
        / job["dataset_id"] / job["action"] / slug(job["backbone"]) / job["origin_id"]
        / f"seed{job['seed']}__{job['job_id']}"
    )
    result = PROJECT_ROOT / config["paths"]["result_root"] / "jobs" / job["suite_id"] / f"{job['job_id']}.json"
    return checkpoint, result


def resolve_existing_history(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    binding = config["existing_dataset_bindings"][job["dataset_id"]]
    root = resolve_existing_path(PROJECT_ROOT, binding["history_root_candidates"])
    candidates = [
        root / job["origin_id"] / f"seed{job['seed']}",
        root / job["origin_id"] / f"seed{job['seed']}__history",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    matches = list((root / job["origin_id"]).glob(f"seed{job['seed']}*")) if (root / job["origin_id"]).is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Existing HISTORY checkpoint unresolved: {candidates}; matches={matches}")


def dependency_history(config: Mapping[str, Any], manifest: Mapping[str, Any], job: Mapping[str, Any], execution: Mapping[str, Any] | None = None) -> Path:
    history_ids = [value for value in job.get("dependencies", []) if not str(value).startswith("group_lock::")]
    for dependency in history_ids:
        parent = job_by_id(manifest, dependency)
        if "history_train" in parent["phase"]:
            checkpoint, result = generated_history_paths(config, parent)
            if not result.exists() or not checkpoint.is_dir():
                raise FileNotFoundError(f"Required HISTORY dependency is incomplete: {dependency}")
            artifact = load_json(result)
            if artifact["checkpoint_digest"] != checkpoint_digest(checkpoint):
                raise ValueError("HISTORY checkpoint changed after its result was written.")
            return checkpoint
    checkpoint = resolve_existing_history(config, job)
    if execution is not None:
        expected = execution["existing_input_locks"][job["dataset_id"]]["history_checkpoint_digests"][f"{job['origin_id']}/seed{job['seed']}"]["digest"]
        if checkpoint_digest(checkpoint) != expected:
            raise ValueError("Existing HISTORY checkpoint changed after execution sealing.")
    return checkpoint


def load_tokenizer(checkpoint_or_id: str | Path, revision: str | None = None):
    from transformers import AutoTokenizer

    kwargs = {"use_fast": True}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(checkpoint_or_id, **kwargs)


def new_base_model(backbone: str, revision: str, spec: Mapping[str, Any], label_names: Sequence[str]):
    from transformers import AutoModelForSequenceClassification

    label2id = {str(name): index for index, name in enumerate(label_names)}
    id2label = {index: str(name) for index, name in enumerate(label_names)}
    kwargs: dict[str, Any] = {
        "revision": revision,
        "num_labels": int(spec["num_labels"]),
        "label2id": label2id,
        "id2label": id2label,
        "ignore_mismatched_sizes": True,
    }
    if spec["task_type"] == "multi_label_sequence_classification":
        kwargs["problem_type"] = "multi_label_classification"
    return AutoModelForSequenceClassification.from_pretrained(backbone, **kwargs)


def load_history_model(checkpoint: Path, spec: Mapping[str, Any]):
    from transformers import AutoModelForSequenceClassification

    kwargs = {}
    if spec["task_type"] == "multi_label_sequence_classification":
        kwargs["problem_type"] = "multi_label_classification"
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, **kwargs)


def label_names(prepared_manifest: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    if prepared_manifest.get("label_names"):
        names = [str(value) for value in prepared_manifest["label_names"]]
    elif prepared_manifest.get("label_mapping"):
        mapping = prepared_manifest["label_mapping"]
        names = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    else:
        names = [str(index) for index in range(int(spec["num_labels"]))]
    if len(names) != int(spec["num_labels"]):
        raise ValueError(f"Label-map size {len(names)} differs from num_labels={spec['num_labels']}")
    return names


def training_runtime(config: Mapping[str, Any], action: str) -> dict[str, Any]:
    runtime = dict(config["runtime"]["candidates"][action])
    runtime["effective_batch_size"] = int(runtime["micro_batch_size"]) * int(runtime["gradient_accumulation_steps"])
    runtime["mixed_precision"] = config["runtime"]["mixed_precision"]
    return runtime


def timed_evaluate(model, loader, *, device, task_type: str, threshold: float | Sequence[float] = 0.5):
    import torch

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = evaluate_model(model, loader, device=device, task_type=task_type, threshold=threshold)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return value, {
        "elapsed_seconds": elapsed,
        "examples_per_second": value.logits.shape[0] / max(elapsed, 1e-9),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def cached_base_evaluate(config, model, loader, *, device, task_type: str, history_digest: str, role_path: Path):
    import torch

    identity = {
        "history_checkpoint_digest": history_digest,
        "prepared_role_sha256": file_sha256(role_path),
        "task_type": task_type,
        "runner_sha256": file_sha256(Path(__file__)),
    }
    key = content_hash(identity)
    root = PROJECT_ROOT / config["paths"]["result_root"] / "immutable_base_cache"
    tensor_path, meta_path = root / f"{key}.pt", root / f"{key}.json"
    if tensor_path.exists() and meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("cache_identity") != identity or meta.get("tensor_sha256") != file_sha256(tensor_path):
            raise ValueError("Immutable base-evaluation cache failed its hash check.")
        state = torch.load(tensor_path, map_location="cpu", weights_only=True)
        evaluation = evaluate_logits(state["logits"], state["labels"], state["deploy_bins"].tolist(), task_type=task_type)
        return evaluation, {"cache_hit": True, "cache_key": key, **meta["original_runtime"]}
    if tensor_path.exists() or meta_path.exists():
        raise FileExistsError(f"Partial immutable cache exists for {key}")
    evaluation, runtime = timed_evaluate(model, loader, device=device, task_type=task_type)
    root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "logits": evaluation.logits, "labels": evaluation.labels,
        "deploy_bins": torch.as_tensor(evaluation.deploy_bins, dtype=torch.long),
    }, tensor_path)
    write_json_once(meta_path, {
        "schema_version": "ata-nlp-base-evaluation-cache-10.0",
        "cache_identity": identity,
        "tensor_sha256": file_sha256(tensor_path),
        "original_runtime": runtime,
    })
    return evaluation, {"cache_hit": False, "cache_key": key, **runtime}


def metric_by_bin(evaluation, *, task_type: str, threshold: float | Sequence[float] = 0.5) -> dict[str, float]:
    import torch

    output = {}
    bins = sorted({value for value in evaluation.deploy_bins if value >= 0})
    for value in bins:
        indices = [index for index, current in enumerate(evaluation.deploy_bins) if current == value]
        selected = torch.as_tensor(indices, dtype=torch.long)
        output[str(value)] = task_metrics(
            evaluation.logits[selected], evaluation.labels[selected], task_type=task_type, threshold=threshold
        )["macro_f1"]
    return output


def evaluation_report(evaluation, *, task_type: str) -> dict[str, Any]:
    return {
        **evaluation.compact(),
        "calibration": classification_calibration(evaluation.logits, evaluation.labels, task_type=task_type),
    }


def serialize_candidate(candidate, checkpoint: Path, *, action: str, tokenizer) -> dict[str, Any]:
    import torch

    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite candidate checkpoint: {checkpoint}")
    checkpoint.mkdir(parents=True)
    if action == "full_tune":
        candidate.save_pretrained(checkpoint, safe_serialization=True)
        kind = "full_pretrained"
    else:
        trainable = {name: parameter.detach().cpu() for name, parameter in candidate.named_parameters() if parameter.requires_grad}
        if not trainable:
            raise RuntimeError("No trainable parameters were found.")
        torch.save(trainable, checkpoint / "trainable_state.pt")
        kind = "trainable_state"
    tokenizer.save_pretrained(checkpoint)
    metadata = {"schema_version": "ata-nlp-v10-checkpoint-meta-1.0", "serialization_kind": kind, "action": action}
    (checkpoint / "v10_checkpoint_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["checkpoint_digest"] = checkpoint_digest(checkpoint)
    return metadata


def restore_candidate(history_checkpoint: Path, checkpoint: Path, *, action: str, task_type: str, rank: int):
    import torch
    from transformers import AutoModelForSequenceClassification

    if action == "full_tune":
        kwargs = {"problem_type": "multi_label_classification"} if task_type == "multi_label_sequence_classification" else {}
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, **kwargs)
    base = load_history_model(history_checkpoint, {"task_type": task_type})
    candidate = build_candidate_v10(base, action=action, task_type=task_type, rank=rank)
    state = torch.load(checkpoint / "trainable_state.pt", map_location="cpu", weights_only=True)
    expected = {name for name, parameter in candidate.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(f"Candidate trainable-state mismatch: missing={sorted(expected-set(state))[:5]}, extra={sorted(set(state)-expected)[:5]}")
    incompatible = candidate.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) & expected:
        raise ValueError(f"Candidate restore failed: {incompatible}")
    return candidate


def combined_evaluation(base, candidate, *, authorized: bool, task_type: str, threshold=0.5):
    logits = candidate.logits if authorized else base.logits
    return evaluate_logits(logits, base.labels, base.deploy_bins, task_type=task_type, threshold=threshold)


def mixed_bin_evaluation(base, candidate, *, selected_bins: set[int], task_type: str, threshold=0.5):
    import torch

    mask = torch.as_tensor([int(value) in selected_bins for value in base.deploy_bins], dtype=torch.bool).unsqueeze(-1)
    logits = torch.where(mask, candidate.logits, base.logits)
    return evaluate_logits(logits, base.labels, base.deploy_bins, task_type=task_type, threshold=threshold)


def posthoc_policy_audit(config, cert_base, cert_candidate, deploy_base, deploy_candidate, *, task_type: str) -> dict[str, Any]:
    from ata_nlp.certification import blocked_risk_certificate

    legacy = load_json(PACKAGE_ROOT / "configs" / "protocol_v1.json")["certificates"]["blocked_risk"]
    task_positive = cert_candidate.metrics["macro_f1"] > cert_base.metrics["macro_f1"]
    static_mean = statistics_mean(cert_candidate.losses) < statistics_mean(cert_base.losses)
    blocked = blocked_risk_certificate(
        cert_base.losses, cert_candidate.losses,
        block_size=int(legacy["block_size"]), gap_size=int(legacy["gap_size"]),
        harm_threshold=float(legacy["harm_threshold_ce"]), win_probability_target=float(legacy["win_probability_target"]),
        harm_probability_cap=float(legacy["harm_probability_cap"]), delta=float(legacy["delta"]),
        beta_upper=float(legacy["beta_upper"]), min_blocks=int(legacy["min_blocks"]),
    )
    bins = sorted({int(value) for value in deploy_base.deploy_bins if int(value) >= 0})
    selections = {
        "no_adapt": set(),
        "always_adapt": set(bins),
        "static_mean_loss": set(bins) if static_mean else set(),
        "task_positive": set(bins) if task_positive else set(),
        "task_positive_and_blocked_risk": set(bins) if task_positive and blocked.deploy else set(),
        "finite_lease_2_bins": set(bins[:2]) if task_positive else set(),
    }
    harm = float(config["global_rules"]["harm_threshold_points"])
    base_bins = metric_by_bin(deploy_base, task_type=task_type)
    candidate_bins = metric_by_bin(deploy_candidate, task_type=task_type)
    bin_gain = {value: 100.0 * (candidate_bins[str(value)] - base_bins[str(value)]) for value in bins}
    output = {}
    for name, selected in selections.items():
        served = mixed_bin_evaluation(deploy_base, deploy_candidate, selected_bins=selected, task_type=task_type)
        gain = 100.0 * (served.metrics["macro_f1"] - deploy_base.metrics["macro_f1"])
        selected_observations = sum(int(value) in selected for value in deploy_base.deploy_bins)
        selected_gains = [bin_gain[value] for value in selected]
        output[name] = {
            "selected_bins": sorted(selected),
            "coverage_bins_fraction": len(selected) / max(len(bins), 1),
            "coverage_observation_fraction": selected_observations / max(len(deploy_base.deploy_bins), 1),
            "served_macro_f1": served.metrics["macro_f1"],
            "served_gain_points": gain,
            "worst_selected_bin_gain_points": min(selected_gains) if selected_gains else None,
            "harmful_selected_bins": sorted(value for value in selected if bin_gain[value] < -harm),
            "harmful_served_policy": bool(selected and gain < -harm),
        }
    output["certificate_decisions"] = {
        "static_mean_loss": bool(static_mean),
        "task_positive": bool(task_positive),
        "blocked_risk": bool(blocked.deploy),
        "blocked_risk_details": blocked.to_dict(),
    }
    return output


def statistics_mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / max(len(values), 1))


def fit_calibration(base, candidate, *, task_type: str) -> dict[str, Any]:
    correction = candidate.logits - base.logits
    scalar = fit_residual_scalar_by_macro_f1(base.logits, correction, base.labels, task_type=task_type, lower=0.0, upper=1.0)
    scaled = base.logits + scalar["alpha"] * correction
    raw_threshold = fit_global_threshold_by_macro_f1(candidate.logits, base.labels, task_type=task_type)
    threshold = fit_global_threshold_by_macro_f1(scaled, base.labels, task_type=task_type)
    return {"scalar_clipped_0_1": scalar, "raw_threshold": raw_threshold, "threshold_after_scalar": threshold}


def verify_identity_alignment(prepared_rows, identity_rows) -> list[str]:
    hashes = [hashed_record_id(row["record_id"], salt=IDENTITY_SALT) for row in prepared_rows]
    if [row["record_id_hash"] for row in identity_rows] != hashes:
        raise ValueError("Jigsaw identity sidecar order/hash differs from its prepared stream.")
    return hashes


def harmful_identity_bins(base_logits, intervention_logits, labels, bins, identity_rows, *, threshold: float, harm_points: float) -> dict[str, Any]:
    _, base_probabilities = task_predictions(base_logits, task_type="sequence_classification", threshold=0.5)
    _, intervention_probabilities = task_predictions(intervention_logits, task_type="sequence_classification", threshold=threshold)
    output: dict[str, Any] = {}
    for bin_id in sorted({int(value) for value in bins if int(value) >= 0}):
        indices = [index for index, value in enumerate(bins) if int(value) == bin_id]
        sliced_identity = [identity_rows[index] for index in indices]
        sliced_labels = [int(labels[index]) for index in indices]
        base_group = subgroup_binary_metrics(sliced_labels, base_probabilities[indices, 1].tolist(), sliced_identity, decision_threshold=0.5)
        candidate_group = subgroup_binary_metrics(sliced_labels, intervention_probabilities[indices, 1].tolist(), sliced_identity, decision_threshold=threshold)
        for column in IDENTITY_COLUMNS:
            base_row = base_group["groups"][column]
            candidate_row = candidate_group["groups"][column]
            if not base_row.get("available") or not candidate_row.get("available"):
                continue
            delta = 100.0 * (float(candidate_row["macro_f1"]) - float(base_row["macro_f1"]))
            row = output.setdefault(column, {"bin_gain_points": {}, "harmful_bins": []})
            row["bin_gain_points"][str(bin_id)] = delta
            if delta < -float(harm_points):
                row["harmful_bins"].append(bin_id)
    return output


def run_history(config, manifest, revisions, execution, execution_lock_path, job, spec, prepared_dir, prepared_manifest, device):
    import torch
    from transformers import get_linear_schedule_with_warmup

    checkpoint, output = generated_history_paths(config, job)
    if output.exists():
        print(json.dumps({"job_id": job["job_id"], "status": "already_complete", "result": str(output)}, indent=2)); return
    revision = revisions["models"][job["backbone"]]["resolved_revision"]
    names = label_names(prepared_manifest, spec)
    tokenizer = load_tokenizer(job["backbone"], revision)
    model = new_base_model(job["backbone"], revision, spec, names).to(device)
    data = load_window_dataset(prepared_dir, tokenizer, windows=("history",), task_type=spec["task_type"], max_length=spec["max_length"])
    history_runtime = config["runtime"]["history"]
    loader = make_loader(data["history"], batch_size=int(history_runtime["micro_batch_size"]), shuffle=True, seed=job["seed"])
    epochs = int(job.get("variant", {}).get("history_epochs", history_runtime["epochs"]))
    steps = epochs * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(history_runtime["learning_rate"]), weight_decay=float(history_runtime["weight_decay"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=round(float(history_runtime["warmup_ratio"]) * steps), num_training_steps=steps)
    from ata_nlp.runtime import train_steps
    training = train_steps(model, loader, optimizer=optimizer, device=device, steps=steps, precision=history_runtime["mixed_precision"], gradient_accumulation_steps=int(history_runtime["gradient_accumulation_steps"]), scheduler=scheduler)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    artifact = {
        "schema_version": "ata-nlp-gpu-closure-history-10.0", "stage": "history_complete_no_A_C_or_T_opened",
        "job": job, "checkpoint": str(checkpoint), "checkpoint_digest": checkpoint_digest(checkpoint),
        "training": training.to_dict(), "parameters": count_parameters(model), "model_revision": revision,
        "prepared_manifest_hash": content_hash(prepared_manifest), "execution_lock_sha256": file_sha256(execution_lock_path),
        "gpu": torch.cuda.get_device_name(device), "software": {"torch": torch.__version__},
    }
    write_json_once(output, artifact)
    print(json.dumps({"job_id": job["job_id"], "status": "complete", "result": str(output)}, indent=2))


def train_candidate(config, manifest, execution, job, spec, prepared_dir, prepared_manifest, device):
    import torch
    from transformers import get_linear_schedule_with_warmup
    from ata_nlp.runtime import train_steps

    checkpoint, output = candidate_paths(config, job)
    if output.exists():
        print(json.dumps({"job_id": job["job_id"], "status": "already_complete", "result": str(output)}, indent=2)); return
    history = dependency_history(config, manifest, job, execution)
    tokenizer = load_tokenizer(history)
    candidate = build_candidate_v10(
        load_history_model(history, spec), action=job["action"], task_type=spec["task_type"],
        rank=int(job.get("variant", {}).get("rank", 4)),
    ).to(device)
    runtime = training_runtime(config, job["action"])
    if job["action"] == "full_tune" and hasattr(candidate, "gradient_checkpointing_enable"):
        candidate.gradient_checkpointing_enable(); candidate.config.use_cache = False
    adapt = load_window_dataset(prepared_dir, tokenizer, windows=("adapt",), task_type=spec["task_type"], max_length=spec["max_length"])
    loader = make_loader(adapt["adapt"], batch_size=runtime["micro_batch_size"], shuffle=True, seed=job["seed"])
    parameters_before = count_parameters(candidate)
    steps = int(job.get("variant", {}).get("optimizer_steps", 256))
    optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=runtime["learning_rate"], weight_decay=runtime["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(runtime["warmup_steps"], steps), num_training_steps=steps)
    training = train_steps(candidate, loader, optimizer=optimizer, device=device, steps=steps, precision=runtime["mixed_precision"], gradient_accumulation_steps=runtime["gradient_accumulation_steps"], scheduler=scheduler)
    checkpoint_meta = serialize_candidate(candidate, checkpoint, action=job["action"], tokenizer=tokenizer)
    del adapt, loader, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()

    base = load_history_model(history, spec).to(device)
    candidate = restore_candidate(history, checkpoint, action=job["action"], task_type=spec["task_type"], rank=int(job.get("variant", {}).get("rank", 4))).to(device)
    prospective = job["phase"] == "prospective_candidate_train_cert_fit_cert_gate"
    history_digest = checkpoint_digest(history)
    windows = ("cert_fit", "cert_gate") if prospective else ("cert", "deploy")
    data = load_window_dataset(prepared_dir, tokenizer, windows=windows, task_type=spec["task_type"], max_length=spec["max_length"])
    evaluations: dict[str, Any] = {}
    inference: dict[str, Any] = {}
    for window in windows:
        loader = make_loader(data[window], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]), shuffle=False, seed=job["seed"])
        base_eval, base_runtime = cached_base_evaluate(
            config, base, loader, device=device, task_type=spec["task_type"], history_digest=history_digest,
            role_path=prepared_dir / f"{window}.jsonl",
        )
        candidate_eval, candidate_runtime = timed_evaluate(candidate, loader, device=device, task_type=spec["task_type"])
        evaluations[f"base_{window}"] = base_eval
        evaluations[f"candidate_{window}"] = candidate_eval
        inference[window] = {"base": base_runtime, "candidate": candidate_runtime}

    if prospective:
        calibration = fit_calibration(evaluations["base_cert_fit"], evaluations["candidate_cert_fit"], task_type=spec["task_type"])
        alpha = calibration["scalar_clipped_0_1"]["alpha"]
        threshold = calibration["threshold_after_scalar"]["threshold"]
        gate_base = evaluations["base_cert_gate"]
        gate_candidate = evaluations["candidate_cert_gate"]
        calibrated_logits = gate_base.logits + alpha * (gate_candidate.logits - gate_base.logits)
        calibrated_gate = evaluate_logits(calibrated_logits, gate_base.labels, gate_base.deploy_bins, task_type=spec["task_type"], threshold=threshold)
        gate_ablation = {
            "raw_alpha_1": evaluation_report(gate_candidate, task_type=spec["task_type"]),
            "cert_fit_scalar_clipped_0_1": evaluation_report(evaluate_logits(calibrated_logits, gate_base.labels, gate_base.deploy_bins, task_type=spec["task_type"]), task_type=spec["task_type"]),
            "cert_fit_macro_f1_threshold": evaluation_report(evaluate_logits(gate_candidate.logits, gate_base.labels, gate_base.deploy_bins, task_type=spec["task_type"], threshold=calibration["raw_threshold"]["threshold"]), task_type=spec["task_type"]),
            "cert_fit_scalar_plus_threshold": evaluation_report(calibrated_gate, task_type=spec["task_type"]),
        }
        authorized = calibrated_gate.metrics["macro_f1"] > gate_base.metrics["macro_f1"]
        result_windows = {
            "base_cert_fit": evaluation_report(evaluations["base_cert_fit"], task_type=spec["task_type"]), "candidate_cert_fit": evaluation_report(evaluations["candidate_cert_fit"], task_type=spec["task_type"]),
            "base_cert_gate": evaluation_report(gate_base, task_type=spec["task_type"]), "candidate_cert_gate_raw": evaluation_report(gate_candidate, task_type=spec["task_type"]),
            "candidate_cert_gate_calibrated": evaluation_report(calibrated_gate, task_type=spec["task_type"]),
            "cert_gate_calibration_ablation": gate_ablation,
        }
        policy = {
            "primary": "task_positive_on_cert_gate_after_cert_fit_scalar_and_threshold",
            "authorized": bool(authorized),
            "cert_gate_gain_points": 100.0 * (calibrated_gate.metrics["macro_f1"] - gate_base.metrics["macro_f1"]),
            "calibration": calibration,
            "T_opened": False,
        }
    else:
        cert_base, cert_candidate = evaluations["base_cert"], evaluations["candidate_cert"]
        deploy_base, deploy_candidate = evaluations["base_deploy"], evaluations["candidate_deploy"]
        authorized = cert_candidate.metrics["macro_f1"] > cert_base.metrics["macro_f1"]
        served = combined_evaluation(deploy_base, deploy_candidate, authorized=authorized, task_type=spec["task_type"])
        all_policies = posthoc_policy_audit(
            config, cert_base, cert_candidate, deploy_base, deploy_candidate, task_type=spec["task_type"]
        )
        result_windows = {name: evaluation_report(value, task_type=spec["task_type"]) for name, value in evaluations.items()}
        policy = {
            "primary": "task_positive_on_cert", "authorized": bool(authorized),
            "cert_gain_points": 100.0 * (cert_candidate.metrics["macro_f1"] - cert_base.metrics["macro_f1"]),
            "raw_T_gain_points": 100.0 * (deploy_candidate.metrics["macro_f1"] - deploy_base.metrics["macro_f1"]),
            "served_T_gain_points": 100.0 * (served.metrics["macro_f1"] - deploy_base.metrics["macro_f1"]),
            "harmful_served_action": bool(authorized and 100.0 * (deploy_candidate.metrics["macro_f1"] - deploy_base.metrics["macro_f1"]) < -float(config["global_rules"]["harm_threshold_points"])),
            "base_T_by_bin": metric_by_bin(deploy_base, task_type=spec["task_type"]),
            "candidate_T_by_bin": metric_by_bin(deploy_candidate, task_type=spec["task_type"]),
            "all_certification_policies": all_policies,
            "T_opened": True,
        }
    artifact = {
        "schema_version": "ata-nlp-gpu-closure-candidate-10.0", "stage": job["phase"], "job": job,
        "history_checkpoint": str(history), "history_checkpoint_digest": checkpoint_digest(history),
        "candidate_checkpoint": str(checkpoint), "candidate_checkpoint_digest": checkpoint_meta["checkpoint_digest"],
        "candidate_serialized_before_evaluation": True, "parameters": parameters_before,
        "runtime": runtime, "training": training.to_dict(), "inference": inference,
        "windows": result_windows, "policy": policy, "prepared_manifest_hash": content_hash(prepared_manifest),
        "gpu": torch.cuda.get_device_name(device), "software": {"torch": torch.__version__},
    }
    write_json_once(output, artifact)
    print(json.dumps({"job_id": job["job_id"], "status": "complete", "result": str(output), "authorized": bool(authorized)}, indent=2))


def portable_source_checkpoint(path: str) -> Path:
    original = Path(path)
    if original.is_dir():
        return original
    normalized = path.replace("\\", "/")
    marker = "/CS-MoE-Research/"
    if marker in normalized:
        relocated = PROJECT_ROOT / normalized.split(marker, 1)[1]
        if relocated.is_dir():
            return relocated
    raise FileNotFoundError(f"Existing checkpoint could not be relocated: {path}")


def run_s1(config, execution, job, spec, prepared_dir, prepared_manifest, device):
    import torch

    _, output = candidate_paths(config, job)
    if output.exists():
        print(json.dumps({"job_id": job["job_id"], "status": "already_complete", "result": str(output)}, indent=2)); return
    source_cert_path = PROJECT_ROOT / job["source_certification_artifact"]
    source_cert = load_json(source_cert_path)
    history = resolve_existing_history(config, job)
    expected_history = execution["existing_input_locks"][job["dataset_id"]]["history_checkpoint_digests"][f"{job['origin_id']}/seed{job['seed']}"]["digest"]
    if checkpoint_digest(history) != expected_history:
        raise ValueError("Existing Jigsaw HISTORY checkpoint changed after execution sealing.")
    checkpoint = portable_source_checkpoint(job["source_candidate_checkpoint"])
    source_lock = execution["s1_source_locks"][job["job_id"]]
    if file_sha256(source_cert_path) != source_lock["certification_artifact_sha256"]:
        raise ValueError("The v9 Jigsaw certification artifact changed after execution sealing.")
    if checkpoint_digest(checkpoint) != source_lock["candidate_checkpoint_digest"]:
        raise ValueError("The v9 Jigsaw candidate changed after execution sealing.")
    if source_cert["candidate_checkpoint_digest"] != checkpoint_digest(checkpoint):
        raise ValueError("Existing Jigsaw candidate differs from the v9 certification artifact.")
    tokenizer = load_tokenizer(history)
    base = load_history_model(history, spec).to(device)
    candidate = restore_candidate(history, checkpoint, action="ata_output_residual", task_type=spec["task_type"], rank=4).to(device)
    data = load_window_dataset(prepared_dir, tokenizer, windows=("cert", "deploy"), task_type=spec["task_type"], max_length=spec["max_length"])
    values = {}
    runtimes = {}
    for window in ("cert", "deploy"):
        loader = make_loader(data[window], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]), shuffle=False, seed=job["seed"])
        values[f"base_{window}"], runtimes[f"base_{window}"] = cached_base_evaluate(
            config, base, loader, device=device, task_type=spec["task_type"], history_digest=checkpoint_digest(history),
            role_path=prepared_dir / f"{window}.jsonl",
        )
        values[f"candidate_{window}"], runtimes[f"candidate_{window}"] = timed_evaluate(candidate, loader, device=device, task_type=spec["task_type"])
    base_c, candidate_c = values["base_cert"], values["candidate_cert"]
    base_t, candidate_t = values["base_deploy"], values["candidate_deploy"]
    correction_c = candidate_c.logits - base_c.logits
    correction_t = candidate_t.logits - base_t.logits
    unbounded = fit_residual_scalar_by_macro_f1(base_c.logits, correction_c, base_c.labels, task_type=spec["task_type"], lower=-2.0, upper=4.0, steps=301)
    clipped = fit_residual_scalar_by_macro_f1(base_c.logits, correction_c, base_c.labels, task_type=spec["task_type"], lower=0.0, upper=1.0, steps=201)
    raw_threshold = fit_global_threshold_by_macro_f1(candidate_c.logits, base_c.labels, task_type=spec["task_type"])
    scaled_c = base_c.logits + clipped["alpha"] * correction_c
    scaled_threshold = fit_global_threshold_by_macro_f1(scaled_c, base_c.labels, task_type=spec["task_type"])
    temperature = fit_temperature_by_loss(candidate_c.logits, base_c.labels, task_type=spec["task_type"])
    interventions = {
        "raw_alpha_1": (candidate_c.logits, candidate_t.logits, 0.5),
        "cert_scalar_unbounded": (base_c.logits + unbounded["alpha"] * correction_c, base_t.logits + unbounded["alpha"] * correction_t, 0.5),
        "cert_scalar_clipped_0_1": (base_c.logits + clipped["alpha"] * correction_c, base_t.logits + clipped["alpha"] * correction_t, 0.5),
        "cert_scalar_shrunk_50_percent": (base_c.logits + 0.5 * clipped["alpha"] * correction_c, base_t.logits + 0.5 * clipped["alpha"] * correction_t, 0.5),
        "cert_macro_f1_threshold": (candidate_c.logits, candidate_t.logits, raw_threshold["threshold"]),
        "cert_temperature": (candidate_c.logits / temperature["temperature"], candidate_t.logits / temperature["temperature"], 0.5),
        "cert_scalar_plus_macro_f1_threshold": (base_c.logits + clipped["alpha"] * correction_c, base_t.logits + clipped["alpha"] * correction_t, scaled_threshold["threshold"]),
    }
    identity_root = PROJECT_ROOT / "data_nlp_ata_v10_identity" / job["dataset_id"] / job["origin_id"]
    identity_c = read_jsonl(identity_root / "cert_identity.jsonl")
    identity_t = read_jsonl(identity_root / "deploy_identity.jsonl")
    verify_identity_alignment(read_jsonl(prepared_dir / "cert.jsonl"), identity_c)
    expected_hashes = verify_identity_alignment(read_jsonl(prepared_dir / "deploy.jsonl"), identity_t)
    output_interventions = {}
    prediction_root = PROJECT_ROOT / config["paths"]["prediction_root"] / job["suite_id"]
    prediction_path = prediction_root / f"{job['job_id']}.jsonl"
    if prediction_path.exists():
        raise FileExistsError(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    probability_rows = {}
    for name, (logits_c, logits_t, threshold) in interventions.items():
        evaluation_c = evaluate_logits(logits_c, base_c.labels, base_c.deploy_bins, task_type=spec["task_type"], threshold=threshold)
        evaluation = evaluate_logits(logits_t, base_t.labels, base_t.deploy_bins, task_type=spec["task_type"], threshold=threshold)
        _, probabilities_c = task_predictions(logits_c, task_type=spec["task_type"], threshold=threshold)
        _, probabilities = task_predictions(logits_t, task_type=spec["task_type"], threshold=threshold)
        group_c = subgroup_binary_metrics(base_c.labels.tolist(), probabilities_c[:, 1].tolist(), identity_c, decision_threshold=float(threshold))
        group_t = subgroup_binary_metrics(base_t.labels.tolist(), probabilities[:, 1].tolist(), identity_t, decision_threshold=float(threshold))
        output_interventions[name] = {
            "cert": {**evaluation_report(evaluation_c, task_type=spec["task_type"]), "gain_points_vs_base": 100.0 * (evaluation_c.metrics["macro_f1"] - base_c.metrics["macro_f1"]), "identity_audit": group_c},
            "deploy": {**evaluation_report(evaluation, task_type=spec["task_type"]), "gain_points_vs_base": 100.0 * (evaluation.metrics["macro_f1"] - base_t.metrics["macro_f1"]), "identity_audit": group_t},
            "threshold": float(threshold), "by_bin": metric_by_bin(evaluation, task_type=spec["task_type"], threshold=threshold),
            "harmful_identity_bins": harmful_identity_bins(
                base_t.logits, logits_t, base_t.labels.tolist(), base_t.deploy_bins, identity_t,
                threshold=float(threshold), harm_points=float(config["global_rules"]["harm_threshold_points"]),
            ),
        }
        probability_rows[name] = probabilities[:, 1].tolist()
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, record_hash in enumerate(expected_hashes):
            handle.write(json.dumps({
                "record_id_hash": record_hash, "label": int(base_t.labels[index]), "deploy_bin": int(base_t.deploy_bins[index]),
                "positive_probability": {name: float(values[index]) for name, values in probability_rows.items()},
            }, sort_keys=True) + "\n")
    artifact = {
        "schema_version": "ata-nlp-jigsaw-intervention-identity-10.0", "stage": "posthoc_T_already_opened_not_confirmatory",
        "job": job, "source_certification_sha256": file_sha256(source_cert_path), "source_checkpoint_digest": checkpoint_digest(checkpoint),
        "base_C": evaluation_report(base_c, task_type=spec["task_type"]), "candidate_C": evaluation_report(candidate_c, task_type=spec["task_type"]), "base_T": evaluation_report(base_t, task_type=spec["task_type"]), "candidate_T": evaluation_report(candidate_t, task_type=spec["task_type"]),
        "fitted_on_C": {"scalar_unbounded": unbounded, "scalar_clipped_0_1": clipped, "raw_threshold": raw_threshold, "scaled_threshold": scaled_threshold, "temperature": temperature},
        "interventions": output_interventions, "prediction_file": str(prediction_path), "prediction_sha256": file_sha256(prediction_path),
        "runtime": runtimes, "must_not_claim": "prospective validation of any intervention", "gpu": torch.cuda.get_device_name(device),
    }
    write_json_once(output, artifact)
    print(json.dumps({"job_id": job["job_id"], "status": "complete", "result": str(output)}, indent=2))


def run_deploy(config, manifest, execution, execution_lock_path, job, spec, prepared_dir, prepared_manifest, device, barrier_path: Path):
    import torch

    _, output = candidate_paths(config, job)
    if output.exists():
        print(json.dumps({"job_id": job["job_id"], "status": "already_complete", "result": str(output)}, indent=2)); return
    barrier = load_json(barrier_path)
    if barrier.get("status") != "all_multieurlex_actions_locked_before_any_T":
        raise PermissionError("The all-action pre-T barrier is absent or invalid.")
    if barrier.get("execution_lock_sha256") != file_sha256(execution_lock_path):
        raise ValueError("The pre-T barrier belongs to another execution lock.")
    group_path = PACKAGE_ROOT / "locks" / "gpu_closure_v10_multieurlex_groups" / f"{job['action']}.lock.json"
    group_key = str(group_path.relative_to(PACKAGE_ROOT)).replace("\\", "/")
    if group_key not in barrier["group_locks"]:
        raise PermissionError("The required action group was not included in the all-action barrier.")
    if file_sha256(group_path) != barrier["group_locks"][group_key]:
        raise ValueError("The action group lock changed after the all-action barrier was created.")
    group = load_json(group_path)
    if group.get("status") != "locked_before_any_MultiEURLEX_T" or group.get("action") != job["action"]:
        raise PermissionError("The action group lock has an invalid status or action.")
    if group.get("test_outcomes_used") is not False or group.get("all_cert_fit_and_cert_gate_complete") is not True:
        raise PermissionError("The action group lock does not establish a complete pre-T decision.")
    cell = group["cells"][job["job_id"]]
    candidate_job = job_by_id(manifest, cell["candidate_job_id"])
    checkpoint, cert_result = candidate_paths(config, candidate_job)
    if file_sha256(cert_result) != cell["certification_result_sha256"] or checkpoint_digest(checkpoint) != cell["candidate_checkpoint_digest"]:
        raise ValueError("Candidate/certification artifacts changed after the group lock.")
    cert = load_json(cert_result)
    history = dependency_history(config, manifest, candidate_job, execution)
    tokenizer = load_tokenizer(history)
    base = load_history_model(history, spec).to(device)
    candidate = restore_candidate(history, checkpoint, action=job["action"], task_type=spec["task_type"], rank=int(candidate_job.get("variant", {}).get("rank", 4))).to(device)
    # TEST FIREWALL: deploy.jsonl is first referenced only after the all-action barrier checks above.
    data = load_window_dataset(prepared_dir, tokenizer, windows=("deploy",), task_type=spec["task_type"], max_length=spec["max_length"])
    loader = make_loader(data["deploy"], batch_size=int(config["runtime"]["evaluation_micro_batch_size"]), shuffle=False, seed=job["seed"])
    base_t, base_runtime = cached_base_evaluate(
        config, base, loader, device=device, task_type=spec["task_type"], history_digest=checkpoint_digest(history),
        role_path=prepared_dir / "deploy.jsonl",
    )
    candidate_t, candidate_runtime = timed_evaluate(candidate, loader, device=device, task_type=spec["task_type"])
    calibration = cert["policy"]["calibration"]
    alpha = calibration["scalar_clipped_0_1"]["alpha"]
    threshold = calibration["threshold_after_scalar"]["threshold"]
    calibrated_logits = base_t.logits + alpha * (candidate_t.logits - base_t.logits)
    calibrated_t = evaluate_logits(calibrated_logits, base_t.labels, base_t.deploy_bins, task_type=spec["task_type"], threshold=threshold)
    calibration_ablation = {
        "raw_alpha_1": evaluation_report(candidate_t, task_type=spec["task_type"]),
        "cert_fit_scalar_clipped_0_1": evaluation_report(evaluate_logits(calibrated_logits, base_t.labels, base_t.deploy_bins, task_type=spec["task_type"]), task_type=spec["task_type"]),
        "cert_fit_macro_f1_threshold": evaluation_report(evaluate_logits(candidate_t.logits, base_t.labels, base_t.deploy_bins, task_type=spec["task_type"], threshold=calibration["raw_threshold"]["threshold"]), task_type=spec["task_type"]),
        "cert_fit_scalar_plus_threshold": evaluation_report(calibrated_t, task_type=spec["task_type"]),
    }
    authorized = bool(cell["authorized"])
    served = calibrated_t if authorized else base_t
    raw_gain = 100.0 * (candidate_t.metrics["macro_f1"] - base_t.metrics["macro_f1"])
    calibrated_gain = 100.0 * (calibrated_t.metrics["macro_f1"] - base_t.metrics["macro_f1"])
    artifact = {
        "schema_version": "ata-nlp-multieurlex-deployment-10.0", "stage": "prospective_T_opened_after_all_action_barrier",
        "job": job, "candidate_job_id": candidate_job["job_id"], "all_action_barrier_sha256": file_sha256(barrier_path),
        "group_lock_sha256": file_sha256(group_path), "base_T": evaluation_report(base_t, task_type=spec["task_type"]), "candidate_T_raw": evaluation_report(candidate_t, task_type=spec["task_type"]),
        "candidate_T_calibrated": evaluation_report(calibrated_t, task_type=spec["task_type"]), "policy": {
            "authorized_from_cert_gate": authorized, "raw_candidate_gain_points": raw_gain, "calibrated_candidate_gain_points": calibrated_gain,
            "served_gain_points": calibrated_gain if authorized else 0.0,
            "harmful_served_action": bool(authorized and calibrated_gain < -float(config["global_rules"]["harm_threshold_points"])),
            "base_by_bin": metric_by_bin(base_t, task_type=spec["task_type"]),
            "candidate_calibrated_by_bin": metric_by_bin(calibrated_t, task_type=spec["task_type"], threshold=threshold),
        },
        "fixed_calibration_ablation_not_used_for_T_selection": calibration_ablation,
        "runtime": {"base": base_runtime, "candidate": candidate_runtime}, "gpu": torch.cuda.get_device_name(device),
    }
    write_json_once(output, artifact)
    print(json.dumps({"job_id": job["job_id"], "status": "complete", "result": str(output), "authorized": authorized}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exactly one immutable v10 GPU closure job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--revision-lock", type=Path, default=DEFAULT_REVISIONS)
    parser.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION_LOCK)
    parser.add_argument("--all-action-barrier", type=Path, default=DEFAULT_BARRIER)
    args = parser.parse_args()
    config, manifest, revisions, execution = verify_context(args)
    job = job_by_id(manifest, args.job_id)
    device = require_cuda()
    seed_everything(int(job["seed"]))
    spec = dataset_spec(config, job)
    prepared_dir, prepared_manifest = prepared_context(config, job, execution)
    verify_role_files(config, job, prepared_dir, prepared_manifest, execution)
    if "history_train" in job["phase"]:
        run_history(config, manifest, revisions, execution, args.execution_lock, job, spec, prepared_dir, prepared_manifest, device)
    elif job["phase"] == "posthoc_inference":
        run_s1(config, execution, job, spec, prepared_dir, prepared_manifest, device)
    elif job["phase"] in {"posthoc_candidate_train_and_eval", "prospective_candidate_train_cert_fit_cert_gate"}:
        train_candidate(config, manifest, execution, job, spec, prepared_dir, prepared_manifest, device)
    elif job["phase"] == "prospective_deploy_after_group_lock":
        run_deploy(config, manifest, execution, args.execution_lock, job, spec, prepared_dir, prepared_manifest, device, args.all_action_barrier)
    else:
        raise ValueError(f"Unsupported phase: {job['phase']}")


if __name__ == "__main__":
    main()
