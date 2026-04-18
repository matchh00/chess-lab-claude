from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class LLMDecisionResponse(BaseModel):
    selected_move: str
    candidate_rank: int
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


def parse_response(raw_text: str) -> LLMDecisionResponse:
    """Parse LLM response text into LLMDecisionResponse.

    Handles responses wrapped in markdown code blocks (```json ... ```) or bare JSON.
    Raises ValueError if parsing fails.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find first JSON object in the text
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        text = brace_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nRaw text: {raw_text!r}") from e

    return LLMDecisionResponse.model_validate(data)
