"""Weight rebalancer — adjusts policy weights based on accumulated influence records."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.learning.influence import InfluenceRecord
from src.policies.profiles import PolicyProfile


class WeightAdjustment(BaseModel):
    bare_key: str
    old_weight: float
    new_weight: float
    delta: float
    net_influence: float  # positive_count - negative_count
    clamped: bool


def rebalance_weights(
    influence_records: list[InfluenceRecord],
    policy: PolicyProfile,
    original_weights: dict[str, float],
    gain: float = 0.05,
    floor_factor: float = 0.5,
    ceiling_factor: float = 2.0,
) -> list[WeightAdjustment]:
    """Compute per-primitive net influence and return weight adjustments.

    Positive influence (good moves) increases weight; negative (bad moves) decreases.
    Adjustments are clamped between floor_factor and ceiling_factor of the original weight.
    """
    positive: dict[str, float] = {}
    negative: dict[str, float] = {}

    for record in influence_records:
        for entry in record.top_primitives:
            key = entry.bare_key
            if record.is_good:
                positive[key] = positive.get(key, 0.0) + 1.0
            if record.is_bad:
                negative[key] = negative.get(key, 0.0) + 1.0

    all_keys = set(positive) | set(negative)
    adjustments: list[WeightAdjustment] = []

    for key in sorted(all_keys):
        if key not in policy.weight_map:
            continue

        net = positive.get(key, 0.0) - negative.get(key, 0.0)
        old_w = policy.weight_map[key]
        orig_w = original_weights.get(key, old_w)
        floor = orig_w * floor_factor
        ceiling = orig_w * ceiling_factor

        raw_new = old_w + gain * net
        new_w = max(floor, min(ceiling, raw_new))
        clamped = abs(new_w - raw_new) > 1e-9

        adjustments.append(WeightAdjustment(
            bare_key=key,
            old_weight=old_w,
            new_weight=round(new_w, 4),
            delta=round(new_w - old_w, 4),
            net_influence=net,
            clamped=clamped,
        ))
        policy.weight_map[key] = new_w

    return adjustments


def save_learned_policy(policy: PolicyProfile, policy_name: str, run_id: str) -> Path:
    """Write adjusted weights to configs/policies/learned/{policy_name}_learned_{run_id}.yaml."""
    out_dir = Path("configs") / "policies" / "learned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{policy_name}_learned_{run_id}.yaml"

    data = policy.model_dump()
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return out_path
