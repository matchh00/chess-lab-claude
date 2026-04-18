from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PolicyProfile(BaseModel):
    policy_id: str
    name: str
    description: str
    weight_map: dict[str, float] = Field(default_factory=dict)
    group_weights: dict[str, float] = Field(default_factory=dict)
    text_priority_style: str = "moderate"
    version: str = "1.0"

    def get_weight(self, primitive_id: str, category: str) -> float:
        """Look up weight: primitive-specific → category group → default 1.0."""
        if primitive_id in self.weight_map:
            return self.weight_map[primitive_id]
        if category in self.group_weights:
            return self.group_weights[category]
        return 1.0

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


def load_policy(policy_name: str, configs_dir: str = "configs") -> PolicyProfile:
    path = Path(configs_dir) / "policies" / f"{policy_name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return PolicyProfile(**data)


def load_all_policies(configs_dir: str = "configs") -> dict[str, PolicyProfile]:
    policies: dict[str, PolicyProfile] = {}
    policy_dir = Path(configs_dir) / "policies"
    for path in sorted(policy_dir.glob("*.yaml")):
        policy = load_policy(path.stem, configs_dir)
        policies[policy.policy_id] = policy
    return policies
