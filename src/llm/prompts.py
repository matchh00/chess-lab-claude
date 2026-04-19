from __future__ import annotations

import logging
import re

from src.llm.prompt_version import get_template_path
from src.narratives.token_budget import count_tokens
from src.storage.models import (
    BoardSnapshot,
    CandidateMoveRecord,
    DecisionPromptRecord,
    InterpreterNarrative,
)

logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA = """{
  "selected_move": "<uci string>",
  "candidate_rank": <integer>,
  "reason": "<1-2 sentence explanation>",
  "confidence": <float 0.0-1.0>
}"""

# ── risk flag → prose ─────────────────────────────────────────────────────────

_RISK_PROSE: dict[str, str] = {
    "hangs_piece": "leaves a piece exposed",
    "exposes_king": "weakens king safety",
    "loses_material": "risks material loss",
    "king_under_pressure": "increases pressure on your king",
}

# ── primitive concept → natural phrase (for v1.2 candidate block) ─────────────

_CONCEPT_PROSE: dict[str, str] = {
    "center occupancy": "central control",
    "center pawn presence": "pawn center",
    "center attacks": "central pressure",
    "minor pieces developed": "piece development",
    "bishop pair": "bishop pair",
    "pawn shield quality": "king cover",
    "bishop activity": "bishop activity",
    "rook on open file": "rook activity on an open file",
    "knight outpost": "strong knight placement",
    "material difference": "material advantage",
    "hanging own pieces": "piece safety",
    "hanging opponent pieces": "targeting an opponent piece",
    "immediate threat": "a direct threat",
    "enemy attackers near king": "reducing pressure on the king",
    "open lines toward king": "controlling key lines",
    "castled status": "castling",
    "undeveloped back rank minors": "developing a back-rank piece",
    "passed pawns": "a passed pawn",
    "rook count difference": "rook balance",
    "queen presence": "queen activity",
    "tempo gaining candidate": "a tempo gain",
    "forcing moves count": "forcing options",
    "checks available": "checking possibilities",
    "captures available": "a capture opportunity",
    "pawn islands": "pawn structure",
    "isolated pawns": "pawn structure",
    "doubled pawns": "pawn structure",
    "backward pawns": "backward pawn improvement",
    "threatened major pieces": "major piece safety",
    "queen overextension risk": "queen safety",
}


def _concept_phrase(primitive_name: str) -> str:
    return _CONCEPT_PROSE.get(primitive_name.strip(), primitive_name.strip())


def _extract_improving_concepts(delta_summary: str) -> list[str]:
    """Pull improving primitive names from delta summary, stripping all numbers."""
    if not delta_summary or "No significant" in delta_summary:
        return []
    pos_match = re.search(r"\+:\s*(.+?)(?:\s*\||-:|$)", delta_summary)
    if not pos_match:
        return []
    pos_text = pos_match.group(1)
    pos_text = re.sub(r"\s*\([^)]*\)", "", pos_text)
    names = [n.strip() for n in pos_text.split(",") if n.strip()]
    return names


# ── v1.1 candidate block (unchanged from before) ─────────────────────────────

def _build_v11_candidate_block(candidates: list[CandidateMoveRecord]) -> str:
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


# ── v1.2 candidate block (tone vocabulary, no numbers, no source labels) ──────

def _build_v12_candidate_block(candidates: list[CandidateMoveRecord]) -> str:
    pos_deltas = [
        c.score_delta for c in candidates
        if c.score_delta is not None and c.score_delta > 0
    ]
    max_delta = max(pos_deltas) if pos_deltas else 0.0
    third = max_delta / 3 if max_delta > 0 else 0.0

    def _tone(delta: float | None) -> str:
        if delta is None or delta <= 0 or max_delta == 0:
            return ""
        if delta >= third * 2:
            return "Strong"
        if delta >= third:
            return "Solid"
        return "Modest"

    sorted_candidates = sorted(candidates, key=lambda c: c.presentation_index)
    lines: list[str] = ["## Candidate Moves (in presentation order)"]

    for c in sorted_candidates:
        tone = _tone(c.score_delta)
        concepts = _extract_improving_concepts(c.primitive_delta_summary or "")
        phrases = [_concept_phrase(n) for n in concepts[:2]]

        if tone and phrases:
            desc = f"{tone} improvement in {' and '.join(phrases)}"
        elif tone:
            desc = f"{tone} overall improvement"
        elif phrases:
            desc = f"Improves {' and '.join(phrases)}"
        else:
            desc = "Positionally neutral"

        if c.risk_flags:
            risk_parts = [_RISK_PROSE.get(f, f.replace("_", " ")) for f in c.risk_flags]
            risk_str = "; ".join(risk_parts) + "."
        else:
            risk_str = "No apparent risk."

        lines.append(f"{c.san} ({c.uci}) — {desc}. {risk_str}")

    return "\n".join(lines)


# ── public build_prompt ───────────────────────────────────────────────────────

def build_prompt(
    board_snapshot: BoardSnapshot,
    position_narrative: str,
    policy_summary: str,
    candidates: list[CandidateMoveRecord],
    version: str = "v1.1",
    interpreter_narrative: InterpreterNarrative | None = None,
    policy_plain_language: str = "",
) -> DecisionPromptRecord:
    template_path = get_template_path(version)
    with open(template_path) as f:
        system_prompt = f.read().strip()

    if version == "v1.2":
        candidate_block = _build_v12_candidate_block(candidates)
        move_history_san = board_snapshot.move_history[-5:]

        interp_section: list[str] = []
        if interpreter_narrative:
            interp_section = [
                "## Strategic Assessment",
                f"SITUATION: {interpreter_narrative.situation}",
                f"ALERT: {interpreter_narrative.alert}",
                f"PRIORITY: {interpreter_narrative.priority}",
                f"DIRECTIVE: {interpreter_narrative.directive}",
            ]
        else:
            interp_section = ["## Strategic Assessment", "(not available)"]

        user_parts = [
            "## Recent Moves (last 5)",
            ", ".join(move_history_san) if move_history_san else "none",
            "",
        ] + interp_section + [
            "",
            "## Strategic Reminder",
            policy_plain_language or policy_summary or "(no policy guidance)",
            "",
            candidate_block,
            "",
            "Respond with JSON only.",
        ]
    else:
        candidate_block = _build_v11_candidate_block(candidates)
        user_parts = [
            "## Position",
            f"FEN: {board_snapshot.fen}",
            f"Turn: {board_snapshot.turn}",
            f"Move history (last 5): {', '.join(board_snapshot.move_history[-5:]) or 'none'}",
            "",
            "## Position Narrative",
            position_narrative or "(no narrative available)",
            "",
            "## Policy Guidance",
            policy_summary or "(no policy guidance available)",
            "",
            candidate_block,
            "",
            "Respond with JSON only.",
        ]

    user_prompt = "\n".join(user_parts)
    token_count = count_tokens(system_prompt + "\n" + user_prompt)
    logger.debug("Prompt token count: %d (version=%s)", token_count, version)

    return DecisionPromptRecord(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        position_summary=position_narrative,
        policy_summary=policy_summary,
        candidate_block=candidate_block,
        response_schema=_RESPONSE_SCHEMA,
        prompt_version=version,
        token_count=token_count,
    )
