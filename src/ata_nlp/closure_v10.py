from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_COLUMNS = (
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.setdefault("artifact_content_hash", content_hash(value))
    serialized = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"Refusing to overwrite a different result: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def resolve_existing_path(project_root: Path, candidates: Sequence[str | Path]) -> Path:
    attempted: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = project_root / path
        attempted.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the registered paths exists: {attempted}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Non-object JSONL row in {path}")
            rows.append(value)
    return rows


def hashed_record_id(value: Any, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()[:24]


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Every formal v10 model or inference job requires a CUDA GPU.")
    return torch.device("cuda")


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_loss(logits, labels, *, task_type: str, reduction: str = "mean"):
    import torch

    if task_type == "multi_label_sequence_classification":
        values = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels.float(), reduction="none"
        ).mean(dim=-1)
        return values if reduction == "none" else values.mean()
    values = torch.nn.functional.cross_entropy(logits.float(), labels.long(), reduction="none")
    return values if reduction == "none" else values.mean()


def task_predictions(logits, *, task_type: str, threshold: float | Sequence[float] = 0.5):
    import torch

    if task_type == "multi_label_sequence_classification":
        probabilities = torch.sigmoid(logits.float())
        threshold_tensor = torch.as_tensor(threshold, dtype=probabilities.dtype, device=probabilities.device)
        return (probabilities >= threshold_tensor).long(), probabilities
    probabilities = torch.softmax(logits.float(), dim=-1)
    if logits.shape[-1] == 2 and threshold != 0.5:
        threshold_value = float(threshold) if not isinstance(threshold, Sequence) else float(threshold[0])
        prediction = (probabilities[:, 1] >= threshold_value).long()
    else:
        prediction = probabilities.argmax(dim=-1)
    return prediction, probabilities


def task_metrics(logits, labels, *, task_type: str, threshold: float | Sequence[float] = 0.5) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    prediction, _ = task_predictions(logits, task_type=task_type, threshold=threshold)
    gold = labels.detach().cpu().numpy()
    predicted = prediction.detach().cpu().numpy()
    if task_type == "multi_label_sequence_classification":
        per_label = f1_score(gold, predicted, average=None, zero_division=0)
        return {
            "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(gold, predicted, average="micro", zero_division=0)),
            "per_label_f1": [float(value) for value in per_label],
            "worst_label_f1": float(per_label.min()) if len(per_label) else 0.0,
            "exact_match_accuracy": float(accuracy_score(gold, predicted)),
        }
    per_class = f1_score(gold, predicted, average=None, labels=list(range(logits.shape[-1])), zero_division=0)
    return {
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "per_class_f1": [float(value) for value in per_class],
        "worst_class_f1": float(per_class.min()) if len(per_class) else 0.0,
    }


def classification_calibration(logits, labels, *, task_type: str, bins: int = 15) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss

    _, probabilities = task_predictions(logits, task_type=task_type)
    probs = probabilities.detach().cpu().numpy()
    gold = labels.detach().cpu().numpy()
    if task_type == "multi_label_sequence_classification":
        confidence = np.maximum(probs, 1.0 - probs).reshape(-1)
        correctness = ((probs >= 0.5) == gold).astype(float).reshape(-1)
        brier = float(np.mean((probs - gold) ** 2))
        nll = float(-(gold * np.log(np.clip(probs, 1e-12, 1.0)) + (1 - gold) * np.log(np.clip(1 - probs, 1e-12, 1.0))).mean())
    else:
        prediction = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        correctness = (prediction == gold).astype(float)
        one_hot = np.eye(probs.shape[1], dtype=float)[gold.astype(int)]
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        nll = float(log_loss(gold, probs, labels=list(range(probs.shape[1]))))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return {"nll": nll, "brier": brier, "ece_15": float(ece)}


def load_window_dataset(
    prepared_dir: str | Path,
    tokenizer,
    *,
    windows: Sequence[str],
    task_type: str,
    max_length: int,
):
    from datasets import Dataset, DatasetDict

    prepared_dir = Path(prepared_dir)
    datasets: dict[str, Any] = {}
    for window in windows:
        rows = read_jsonl(prepared_dir / f"{window}.jsonl")
        normalized: list[dict[str, Any]] = []
        for row in rows:
            labels = row.get("labels") if task_type == "multi_label_sequence_classification" else row.get("label")
            if labels is None:
                raise KeyError(f"Missing labels for {task_type} in {window}")
            if task_type == "multi_label_sequence_classification":
                labels = [float(value) for value in labels]
            normalized.append({
                "text": str(row["text"]),
                "labels": labels,
                "deploy_bin": int(row.get("deploy_bin", -1) if row.get("deploy_bin") is not None else -1),
            })
        dataset = Dataset.from_list(normalized)

        def encode(batch):
            encoded = tokenizer(
                batch["text"], truncation=True, padding="max_length", max_length=int(max_length)
            )
            encoded["labels"] = batch["labels"]
            encoded["deploy_bin"] = batch["deploy_bin"]
            return encoded

        datasets[window] = dataset.map(encode, batched=True, remove_columns=dataset.column_names)
    return DatasetDict(datasets)


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int):
    import torch
    from transformers import default_data_collator

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=default_data_collator,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


@dataclass
class V10Evaluation:
    logits: Any
    labels: Any
    deploy_bins: list[int]
    losses: list[float]
    metrics: dict[str, Any]
    base_logits: Any | None = None
    corrections: Any | None = None

    def compact(self) -> dict[str, Any]:
        from collections import Counter

        return {
            "observations": int(self.logits.shape[0]),
            "mean_loss": float(sum(self.losses) / max(len(self.losses), 1)),
            "metrics": self.metrics,
            "deploy_bin_counts": dict(sorted(Counter(self.deploy_bins).items())),
        }


def evaluate_model(model, loader, *, device, task_type: str, threshold: float | Sequence[float] = 0.5) -> V10Evaluation:
    import torch

    model.eval()
    logits_rows, label_rows, bin_rows = [], [], []
    base_rows, correction_rows = [], []
    with torch.no_grad():
        for batch in loader:
            bins = batch.pop("deploy_bin")
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            base = outputs.get("base_logits") if isinstance(outputs, dict) else getattr(outputs, "base_logits", None)
            correction = outputs.get("correction") if isinstance(outputs, dict) else getattr(outputs, "correction", None)
            logits_rows.append(logits.detach().float().cpu())
            label_rows.append(inputs["labels"].detach().cpu())
            bin_rows.extend(int(value) for value in bins.tolist())
            if base is not None:
                base_rows.append(base.detach().float().cpu())
            if correction is not None:
                correction_rows.append(correction.detach().float().cpu())
    logits = torch.cat(logits_rows)
    labels = torch.cat(label_rows)
    losses = task_loss(logits, labels, task_type=task_type, reduction="none").tolist()
    return V10Evaluation(
        logits=logits,
        labels=labels,
        deploy_bins=bin_rows,
        losses=[float(value) for value in losses],
        metrics=task_metrics(logits, labels, task_type=task_type, threshold=threshold),
        base_logits=torch.cat(base_rows) if base_rows else None,
        corrections=torch.cat(correction_rows) if correction_rows else None,
    )


def evaluate_logits(logits, labels, deploy_bins: Sequence[int], *, task_type: str, threshold: float | Sequence[float] = 0.5) -> V10Evaluation:
    logits = logits.detach().float().cpu()
    labels = labels.detach().cpu()
    losses = task_loss(logits, labels, task_type=task_type, reduction="none").tolist()
    return V10Evaluation(
        logits=logits,
        labels=labels,
        deploy_bins=[int(value) for value in deploy_bins],
        losses=[float(value) for value in losses],
        metrics=task_metrics(logits, labels, task_type=task_type, threshold=threshold),
    )


def fit_residual_scalar_by_macro_f1(
    base_logits,
    correction,
    labels,
    *,
    task_type: str,
    lower: float = 0.0,
    upper: float = 2.0,
    steps: int = 201,
) -> dict[str, float]:
    best: tuple[float, float, float, float] | None = None
    for index in range(int(steps)):
        alpha = lower + (upper - lower) * index / max(steps - 1, 1)
        logits = base_logits + correction * alpha
        metrics = task_metrics(logits, labels, task_type=task_type)
        loss = float(task_loss(logits, labels, task_type=task_type).item())
        candidate = (float(metrics["macro_f1"]), -loss, -abs(alpha), float(alpha))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return {"alpha": best[3], "macro_f1": best[0], "mean_loss": -best[1], "grid_lower": lower, "grid_upper": upper, "grid_steps": int(steps)}


def fit_global_threshold_by_macro_f1(logits, labels, *, task_type: str) -> dict[str, float]:
    if task_type not in {"sequence_classification", "multi_label_sequence_classification"}:
        raise ValueError(f"Threshold calibration is unsupported for task type {task_type}")
    if task_type == "sequence_classification" and logits.shape[-1] != 2:
        return {"threshold": 0.5, "macro_f1": task_metrics(logits, labels, task_type=task_type)["macro_f1"], "not_applicable_multiclass": True}
    best: tuple[float, float, float] | None = None
    for index in range(19):
        threshold = 0.05 * (index + 1)
        macro = task_metrics(logits, labels, task_type=task_type, threshold=threshold)["macro_f1"]
        candidate = (float(macro), -abs(threshold - 0.5), float(threshold))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return {"threshold": best[2], "macro_f1": best[0], "grid_lower": 0.05, "grid_upper": 0.95, "grid_step": 0.05}


def fit_temperature_by_loss(logits, labels, *, task_type: str) -> dict[str, float]:
    candidates = [math.exp(math.log(0.25) + index * (math.log(4.0) - math.log(0.25)) / 80) for index in range(81)]
    best: tuple[float, float] | None = None
    for temperature in candidates:
        loss = float(task_loss(logits / temperature, labels, task_type=task_type).item())
        candidate = (loss, float(temperature))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {"temperature": best[1], "mean_loss": best[0], "grid_lower": 0.25, "grid_upper": 4.0, "grid_steps": 81}


def subgroup_binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    identity_rows: Sequence[Mapping[str, float | None]],
    *,
    identity_threshold: float = 0.5,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    y = np.asarray(labels, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    prediction = (score >= float(decision_threshold)).astype(int)
    if len(identity_rows) != len(y):
        raise ValueError("Identity sidecar and prediction rows have different lengths.")

    def safe_auc(gold, values) -> float | None:
        return float(roc_auc_score(gold, values)) if len(np.unique(gold)) == 2 else None

    output: dict[str, Any] = {}
    for column in IDENTITY_COLUMNS:
        values = np.asarray([
            np.nan if row.get(column) is None else float(row[column]) for row in identity_rows
        ])
        subgroup = np.isfinite(values) & (values >= float(identity_threshold))
        if not subgroup.any():
            output[column] = {"observations": 0, "available": False}
            continue
        gold = y[subgroup]
        pred = prediction[subgroup]
        tp = int(((gold == 1) & (pred == 1)).sum())
        tn = int(((gold == 0) & (pred == 0)).sum())
        fp = int(((gold == 0) & (pred == 1)).sum())
        fn = int(((gold == 1) & (pred == 0)).sum())
        bpsn = (subgroup & (y == 0)) | (~subgroup & (y == 1))
        bnsp = (subgroup & (y == 1)) | (~subgroup & (y == 0))
        output[column] = {
            "observations": int(subgroup.sum()),
            "available": True,
            "positive_labels": int((gold == 1).sum()),
            "negative_labels": int((gold == 0).sum()),
            "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
            "fpr": float(fp / max(fp + tn, 1)),
            "fnr": float(fn / max(fn + tp, 1)),
            "subgroup_auc": safe_auc(gold, score[subgroup]),
            "bpsn_auc": safe_auc(y[bpsn], score[bpsn]),
            "bnsp_auc": safe_auc(y[bnsp], score[bnsp]),
        }
    available_f1 = [row["macro_f1"] for row in output.values() if row.get("available")]
    return {
        "identity_threshold": float(identity_threshold),
        "decision_threshold": float(decision_threshold),
        "groups": output,
        "worst_group_macro_f1": min(available_f1) if available_f1 else None,
    }


def build_candidate_v10(base_model, *, action: str, task_type: str, rank: int = 4):
    import torch
    from .adapters_v3 import build_candidate_v3

    if task_type != "multi_label_sequence_classification" and action in {"ata_output_residual", "lora", "bottleneck", "full_tune"}:
        configs = {
            "ata_output_residual": {"rank": int(rank), "activation": "gelu"},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["query", "value"]},
            "bottleneck": {"width": 64, "dropout": 0.0},
            "full_tune": {},
        }
        method = "output_residual" if action == "ata_output_residual" else action
        return build_candidate_v3(base_model, method=method, task_type=task_type, method_config=configs[action])

    nn = torch.nn
    functional = torch.nn.functional
    num_labels = int(base_model.config.num_labels)
    hidden_size = int(base_model.config.hidden_size)

    class OutputResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model
            self.down = nn.Linear(hidden_size, int(rank), bias=False)
            self.up = nn.Linear(int(rank), num_labels, bias=True)
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

        def forward(self, labels=None, **inputs):
            self.base_model.eval()
            with torch.no_grad():
                base = self.base_model(**inputs, output_hidden_states=True, return_dict=True)
            correction = self.up(functional.gelu(self.down(base.hidden_states[-1][:, 0].detach())))
            logits = base.logits.detach() + correction
            loss = task_loss(logits, labels, task_type=task_type) if labels is not None else None
            return {"loss": loss, "logits": logits, "base_logits": base.logits.detach(), "correction": correction}

    class VectorScaling(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model
            self.scale = nn.Parameter(torch.ones(num_labels))
            self.bias = nn.Parameter(torch.zeros(num_labels))
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

        def forward(self, labels=None, **inputs):
            self.base_model.eval()
            with torch.no_grad():
                base = self.base_model(**inputs, return_dict=True)
            logits = base.logits.detach() * self.scale + self.bias
            loss = task_loss(logits, labels, task_type=task_type) if labels is not None else None
            return {"loss": loss, "logits": logits, "base_logits": base.logits.detach(), "correction": logits - base.logits.detach()}

    if action == "ata_output_residual":
        return OutputResidual()
    if action == "vector_logit_scaling":
        return VectorScaling()
    if action == "full_tune":
        for parameter in base_model.parameters():
            parameter.requires_grad_(True)
        return base_model
    if action == "classifier_head_only":
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        classifier = getattr(base_model, "classifier", None)
        if classifier is None:
            raise TypeError("The registered backbone has no classifier head.")
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)
        return base_model
    if action in {"lora", "bottleneck"}:
        base_model.config.problem_type = "multi_label_classification"
        if action == "lora":
            return build_candidate_v3(
                base_model,
                method="lora",
                task_type="sequence_classification",
                method_config={"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["query", "value"]},
            )
        # Multi-label bottleneck uses the same representation but BCE loss.
        class Bottleneck(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = base_model
                self.down = nn.Linear(hidden_size, 64)
                self.up = nn.Linear(64, hidden_size)
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)
                for parameter in self.base_model.parameters():
                    parameter.requires_grad_(False)

            def forward(self, labels=None, **inputs):
                self.base_model.eval()
                with torch.no_grad():
                    base = self.base_model(**inputs, output_hidden_states=True, return_dict=True)
                hidden = base.hidden_states[-1].detach()
                adapted = hidden + self.up(functional.gelu(self.down(hidden)))
                classifier = self.base_model.classifier
                logits = classifier(adapted) if str(getattr(self.base_model.config, "model_type", "")) in {"roberta", "xlm-roberta"} else classifier(adapted[:, 0])
                loss = task_loss(logits, labels, task_type=task_type) if labels is not None else None
                return {"loss": loss, "logits": logits, "base_logits": base.logits.detach()}
        return Bottleneck()
    if action == "vector_logit_scaling":
        return VectorScaling()
    raise ValueError(f"Unsupported v10 action: {action}")


def count_parameters(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "trainable_fraction": float(trainable / max(total, 1))}
