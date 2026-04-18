from __future__ import annotations

from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env from project root regardless of working directory
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1000
_TEMPERATURE = 0.2


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the Anthropic API and return the raw response text."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text
