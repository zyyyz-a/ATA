from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+", re.IGNORECASE)


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


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json_once(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.setdefault("artifact_content_hash", content_hash(value))
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite a different artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def normalized_text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def stable_rank(*values: Any, salt: str = "ata-v11") -> str:
    return hashlib.sha256((salt + "\0" + "\0".join(map(str, values))).encode("utf-8")).hexdigest()


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite timestamp: {value!r}")
        # Public temporal corpora mix Unix seconds with millisecond,
        # microsecond, and nanosecond counters.  Twelve-digit millisecond
        # values occur in early Amazon records, so a 1e12 cutoff is unsafe.
        # Repeatedly reduce only magnitudes that cannot be contemporary Unix
        # seconds; this keeps ordinary second counters unchanged.
        while abs(number) >= 1e11:
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def month_key(value: Any) -> str:
    stamp = parse_timestamp(value)
    return f"{stamp.year:04d}-{stamp.month:02d}"


def deterministic_stratified_sample(
    rows: Iterable[dict[str, Any]], *, maximum: int, timestamp_field: str, id_field: str
) -> list[dict[str, Any]]:
    """Keep the lowest stable hashes within each month, then globally trim.

    The routine is outcome independent and makes source iteration order irrelevant.
    """
    pools: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    per_month_soft_cap = max(500, math.ceil(maximum / 24))
    for row in rows:
        key = month_key(row[timestamp_field])
        rank = stable_rank(row[id_field], key, salt="ata-v11-source-sample")
        bucket = pools[key]
        bucket.append((rank, row))
        if len(bucket) > per_month_soft_cap * 2:
            bucket.sort(key=lambda item: item[0])
            del bucket[per_month_soft_cap:]
    candidates = [item for bucket in pools.values() for item in sorted(bucket, key=lambda item: item[0])[:per_month_soft_cap]]
    candidates.sort(key=lambda item: item[0])
    retained = [row for _, row in candidates[:maximum]]
    retained.sort(key=lambda row: (parse_timestamp(row[timestamp_field]), str(row[id_field])))
    return retained


def simhash64(text: str) -> int:
    tokens = TOKEN_RE.findall(normalize_text(text))
    if len(tokens) < 3:
        tokens = tokens or [""]
        shingles = tokens
    else:
        shingles = [" ".join(tokens[index:index + 3]) for index in range(len(tokens) - 2)]
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    output = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            output |= 1 << bit
    return output


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def temporal_block_indices(length: int, blocks: int) -> list[list[int]]:
    if length < blocks:
        blocks = max(1, length)
    return [
        list(range(start, end))
        for block in range(blocks)
        for start, end in [(round(block * length / blocks), round((block + 1) * length / blocks))]
        if end > start
    ]


def block_bootstrap_lcb(values: Sequence[float], *, draws: int, confidence: float, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lcb": float("-inf"), "ucb": float("inf")}
    rng = random.Random(int(seed))
    simulations = []
    for _ in range(int(draws)):
        simulations.append(sum(rng.choice(values) for _ in values) / len(values))
    simulations.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    lower = simulations[min(len(simulations) - 1, max(0, int(alpha * len(simulations))))]
    upper = simulations[min(len(simulations) - 1, max(0, int((1.0 - alpha) * len(simulations)) - 1))]
    return {"mean": float(sum(values) / len(values)), "lcb": float(lower), "ucb": float(upper)}


def residual_coordinates(base_logits, candidate_logits, labels) -> dict[str, float]:
    import torch

    probabilities = torch.softmax(base_logits.float(), dim=-1)
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=base_logits.shape[-1]).float()
    residual = one_hot - probabilities
    correction = candidate_logits.float() - base_logits.float()
    correction = correction - correction.mean(dim=-1, keepdim=True)
    numerator = (residual * correction).sum()
    residual_norm = residual.square().sum().sqrt()
    correction_energy = correction.square().sum()
    correction_norm = correction_energy.sqrt()
    c = numerator / (residual_norm * correction_norm).clamp_min(1e-12)
    s = numerator / correction_energy.clamp_min(1e-12)
    return {
        "c": float(c.item()),
        "s": float(s.item()),
        "residual_energy": float(residual.square().sum().item()),
        "correction_energy": float(correction_energy.item()),
    }


def lower_cvar(values: Sequence[float], fraction: float = 0.2) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    count = max(1, math.ceil(len(ordered) * float(fraction)))
    return sum(ordered[:count]) / count


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = float(probability) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def matched_ids(rows: Sequence[Mapping[str, Any]], *, count: int, score_field: str | None, salt: str) -> set[str]:
    if count <= 0:
        return set()
    if score_field is None:
        ordered = sorted(rows, key=lambda row: stable_rank(row["cell_id"], salt=salt))
    else:
        ordered = sorted(rows, key=lambda row: (-float(row[score_field]), str(row["cell_id"])))
    return {str(row["cell_id"]) for row in ordered[:count]}
