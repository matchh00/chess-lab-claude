from __future__ import annotations

import os

CURRENT_VERSION = "v1.1"

VERSIONS: dict[str, str] = {
    "v1.0": "configs/llm/prompts/v1.0.md",
    "v1.1": "configs/llm/prompts/v1.1.md",
    "v1.2": "configs/llm/prompts/v1.2.md",
}


def get_version() -> str:
    return CURRENT_VERSION


def get_template_path(version: str = CURRENT_VERSION) -> str:
    if version not in VERSIONS:
        raise ValueError(f"Unknown prompt version: {version!r}. Available: {list(VERSIONS)}")
    return VERSIONS[version]
