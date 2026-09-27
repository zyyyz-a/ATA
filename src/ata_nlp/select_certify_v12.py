from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("A confidence interval requires at least one chronological block.")
    position = min(1.0, max(0.0, float(probability))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def chronological_block_bootstrap(
    values: Sequence[float], *, draws: int, confidence: float, seed: int
) -> dict[str, float]:
    observed = [float(value) for value in values]
    if not observed:
        raise ValueError("No chronological block values were supplied.")
    rng = random.Random(int(seed))
    estimates = [
        sum(rng.choice(observed) for _ in observed) / len(observed)
        for _ in range(int(draws))
    ]
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": sum(observed) / len(observed),
        "lcb": quantile(estimates, alpha),
        "ucb": quantile(estimates, 1.0 - alpha),
    }


def choose_nominee(
    candidates: Sequence[Mapping[str, Any]], *, tie_break_order: Sequence[str]
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("No candidates were supplied for nomination.")
    order = {str(action): index for index, action in enumerate(tie_break_order)}
    actions = [str(row["action"]) for row in candidates]
    if len(actions) != len(set(actions)):
        raise ValueError(f"Duplicate actions in one nomination group: {actions}")
    unknown = sorted(set(actions) - set(order))
    if unknown:
        raise ValueError(f"Unregistered actions in nomination group: {unknown}")
    return min(
        candidates,
        key=lambda row: (
            -float(row["select_gain_points"]),
            order[str(row["action"])],
        ),
    )


def authorize_nominee(nominee: Mapping[str, Any]) -> bool:
    """Apply the frozen one-shot gate; no alternate action is considered."""
    return float(nominee["gate_gain_lcb_points"]) > 0.0


def build_action_lock(
    candidates: Sequence[Mapping[str, Any]],
    *,
    dataset_ids: Sequence[str],
    origin_ids: Sequence[str],
    seeds: Sequence[int],
    actions: Sequence[str],
    tie_break_order: Sequence[str],
    config_sha256: str,
) -> dict[str, Any]:
    """Construct the complete pre-T decision ledger from C-only artifacts.

    Each dataset/origin/seed group must contain every registered action.  The
    C_select winner is the only action evaluated by the authorization rule on
    C_gate.  Rejection serializes the calibrated frozen base, never the runner-up.
    """
    by_group: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (str(row["dataset_id"]), str(row["origin_id"]), int(row["seed"]))
        by_group[key].append(row)

    expected_groups = {
        (str(dataset_id), str(origin_id), int(seed))
        for dataset_id in dataset_ids
        for origin_id in origin_ids
        for seed in seeds
    }
    if set(by_group) != expected_groups:
        missing = sorted(expected_groups - set(by_group))
        extra = sorted(set(by_group) - expected_groups)
        raise ValueError(f"Incomplete C-only matrix; missing={missing[:5]}, extra={extra[:5]}")

    registered_actions = set(map(str, actions))
    decisions: dict[str, dict[str, Any]] = {}
    for dataset_id, origin_id, seed in sorted(expected_groups):
        group = by_group[(dataset_id, origin_id, seed)]
        observed_actions = {str(row["action"]) for row in group}
        if observed_actions != registered_actions or len(group) != len(registered_actions):
            raise ValueError(
                f"Incomplete action set for {dataset_id}/{origin_id}/{seed}: "
                f"expected={sorted(registered_actions)}, observed={sorted(observed_actions)}"
            )
        nominee = choose_nominee(group, tie_break_order=tie_break_order)
        authorized = authorize_nominee(nominee)
        decision_id = f"{dataset_id}::{origin_id}::{seed}"
        evidence = {
            str(row["action"]): {
                "candidate_job_id": str(row["candidate_job_id"]),
                "certification_sha256": str(row["certification_sha256"]),
                "select_gain_points": float(row["select_gain_points"]),
                "gate_gain_points": float(row["gate_gain_points"]),
                "gate_gain_lcb_points": float(row["gate_gain_lcb_points"]),
            }
            for row in sorted(group, key=lambda value: str(value["action"]))
        }
        decisions[decision_id] = {
            "dataset_id": dataset_id,
            "origin_id": origin_id,
            "seed": seed,
            "nominee_action": str(nominee["action"]),
            "nominee_candidate_job_id": str(nominee["candidate_job_id"]),
            "select_gain_points": float(nominee["select_gain_points"]),
            "gate_gain_points": float(nominee["gate_gain_points"]),
            "gate_gain_lcb_points": float(nominee["gate_gain_lcb_points"]),
            "authorized": authorized,
            "serialized_action": str(nominee["action"]) if authorized else "calibrated_frozen_base",
            "runner_up_fallback_attempted": False,
            "all_six_C_only_artifacts": evidence,
        }

    payload = {
        "schema_version": "ata-nlp-select-certify-action-lock-12.0",
        "status": "all_actions_serialized_before_any_v12_T_outcome",
        "policy": "C_cal_calibrate__C_select_top1__C_gate_single_LCB__no_fallback",
        "config_sha256": str(config_sha256),
        "test_outcomes_used": False,
        "decision_count": len(decisions),
        "decisions": decisions,
    }
    payload["action_lock_content_sha256"] = content_hash(payload)
    return payload

