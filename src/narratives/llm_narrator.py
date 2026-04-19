"""Interpreter LLM call — synthesises primitive observations into a strategic briefing."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.llm.prompt_version import get_interpreter_template_path
from src.narratives.token_budget import count_tokens
from src.policies.profiles import PolicyProfile
from src.storage.models import InterpreterNarrative, PrimitiveValue, WeightedPrimitiveState

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)

_INTERPRETER_MODEL = "claude-sonnet-4-6"
_INTERPRETER_MAX_TOKENS = 400
_INTERPRETER_TEMPERATURE = 0.3
_INTERPRETER_VERSION = "interpreter_v1.0"


def _tone_label(weighted_score: float) -> str:
    if weighted_score >= 1.4:
        return "critical"
    if weighted_score >= 0.8:
        return "high"
    if weighted_score >= 0.3:
        return "moderate"
    return "low"


def _game_phase(move_count: int) -> str:
    if move_count <= 10:
        return "opening"
    if move_count <= 25:
        return "middlegame"
    return "endgame"


def _build_interpreter_user_prompt(
    weighted_states: list[WeightedPrimitiveState],
    primitives: list[PrimitiveValue],
    policy: PolicyProfile,
    move_history: list[str],
    move_count: int,
) -> str:
    prim_map: dict[str, PrimitiveValue] = {p.primitive_id: p for p in primitives}
    sorted_states = sorted(weighted_states, key=lambda s: s.weighted_score, reverse=True)

    observations: list[str] = []
    for state in sorted_states:
        prim = prim_map.get(state.primitive_id)
        if prim and prim.text_render and prim.text_render.strip():
            tone = _tone_label(state.weighted_score)
            observations.append(f"- [{tone}] {prim.text_render}")

    phase = _game_phase(move_count)
    last_3 = move_history[-3:] if move_history else []
    recent = ", ".join(last_3) if last_3 else "none"

    parts: list[str] = [
        f"## Game Phase: {phase}",
        "",
        f"## Recent moves: {recent}",
        "",
        f"## Strategic Policy: {policy.name}",
        policy.plain_language or policy.description,
        "",
        "## Position Observations (ordered by strategic importance)",
    ]
    parts.extend(observations)
    parts += ["", "Write the strategic briefing."]
    return "\n".join(parts)


def _call_interpreter_llm(system_prompt: str, user_prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=_INTERPRETER_MODEL,
        max_tokens=_INTERPRETER_MAX_TOKENS,
        temperature=_INTERPRETER_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def _parse_interpreter_response(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    sections = ("SITUATION", "ALERT", "PRIORITY", "DIRECTIVE")
    boundaries = "|".join(sections)
    for key in sections:
        pattern = rf"{key}:\s*(.+?)(?=\n(?:{boundaries}):|$)"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            result[key.lower()] = value if value else "unavailable"
        else:
            logger.warning("Interpreter parse: missing section %s", key)
            result[key.lower()] = "unavailable"
    return result


def build_interpreter_narrative(
    weighted_states: list[WeightedPrimitiveState],
    primitives: list[PrimitiveValue],
    policy: PolicyProfile,
    move_history: list[str],
    move_count: int,
    llm_call_fn: Callable[[str, str], str] | None = None,
) -> InterpreterNarrative:
    """Make one LLM call that writes a strategic briefing from primitive observations.

    Returns an InterpreterNarrative. On any failure, affected fields are set to
    "unavailable" and the exception is logged — never raised.
    """
    template_path = get_interpreter_template_path(_INTERPRETER_VERSION)
    with open(template_path) as f:
        system_prompt = f.read().strip()

    user_prompt = _build_interpreter_user_prompt(
        weighted_states, primitives, policy, move_history, move_count
    )

    caller = llm_call_fn or _call_interpreter_llm
    try:
        raw_output = caller(system_prompt, user_prompt)
    except Exception as exc:
        logger.error("Interpreter LLM call failed: %s", exc)
        raw_output = ""

    sections: dict[str, str] = {}
    if raw_output:
        try:
            sections = _parse_interpreter_response(raw_output)
        except Exception as exc:
            logger.error("Interpreter response parse failed: %s", exc)

    token_count = count_tokens(raw_output) if raw_output else 0

    return InterpreterNarrative(
        situation=sections.get("situation", "unavailable"),
        alert=sections.get("alert", "unavailable"),
        priority=sections.get("priority", "unavailable"),
        directive=sections.get("directive", "unavailable"),
        raw_output=raw_output,
        token_count=token_count,
        prompt_version=_INTERPRETER_VERSION,
    )
