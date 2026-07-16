from __future__ import annotations

import logging

from src.llm.prompt_version import get_template_path
from src.narratives.token_budget import count_tokens
from src.storage.models import BoardSnapshot, CandidateMoveRecord, DecisionPromptRecord

logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA = """{
  "selected_move": "<uci string>",
  "candidate_rank": <integer>,
  "reason": "<1-2 sentence explanation>",
  "confidence": <float 0.0-1.0>
}"""


def _build_candidate_block(candidates: list[CandidateMoveRecord]) -> str:
    lines: list[str] = ["## Candidate Moves (in presentation order)"]
    sorted_candidates = sorted(candidates, key=lambda c: c.presentation_index)
    for c in sorted_candidates:
        narrative = c.candidate_narrative.strip() if c.candidate_narrative else ""
        risk_str = ", ".join(c.risk_flags) if c.risk_flags else "none"
        line = f"[{c.presentation_index}] {c.san} ({c.uci})"
        if narrative:
            line += f" — {narrative}"
        line += f" Risk: {risk_str}."
        lines.append(line)
    return "\n".join(lines)


def build_prompt(
    board_snapshot: BoardSnapshot,
    position_narrative: str,
    policy_summary: str,
    candidates: list[CandidateMoveRecord],
    version: str = "v1.1",
    self_model_block: str = "",
) -> DecisionPromptRecord:
    template_path = get_template_path(version)
    with open(template_path) as f:
        system_prompt = f.read().strip()

    candidate_block = _build_candidate_block(candidates)

    user_parts = [
        f"## Position",
        f"FEN: {board_snapshot.fen}",
        f"Turn: {board_snapshot.turn}",
        f"Move history (last 5): {', '.join(board_snapshot.move_history[-5:]) or 'none'}",
        "",
        f"## Position Narrative",
        position_narrative or "(no narrative available)",
        "",
        f"## Policy Guidance",
        policy_summary or "(no policy guidance available)",
        "",
    ]
    if self_model_block:
        user_parts.extend([self_model_block, ""])
    user_parts.extend([
        candidate_block,
        "",
        "Respond with JSON only.",
    ])
    user_prompt = "\n".join(user_parts)

    token_count = count_tokens(system_prompt + "\n" + user_prompt)
    logger.debug("Prompt token count: %d (version=%s)", token_count, version)

    return DecisionPromptRecord(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        position_summary=position_narrative,
        policy_summary=policy_summary,
        candidate_block=candidate_block,
        self_model_block=self_model_block,
        response_schema=_RESPONSE_SCHEMA,
        prompt_version=version,
        token_count=token_count,
    )
