from __future__ import annotations

from src.self_model.models import MoveReflection, SelfModelState, SelfPattern

DEFAULT_HISTORY_MOVES = 6

# Directives are generated deterministically from detected pattern kinds so
# condition C stays reproducible — no free-form generation in the loop.
_DIRECTIVES: dict[str, str] = {
    "overconfidence": (
        "Lower your confidence and double-check the opponent's best reply "
        "before committing to a move you feel sure about."
    ),
    "calibration_gap": (
        "Report high confidence only when you can name the concrete refutation "
        "of the opponent's strongest response."
    ),
    "category_bias": (
        "When the flagged signal dominates, deliberately weigh the opposing "
        "signals (king safety, material risk) before deciding."
    ),
    "cost_streak": (
        "Break the pattern: prefer the solid, low-risk candidate over the "
        "line of play you have been following."
    ),
}


def _outcome_phrase(reflection: MoveReflection) -> str:
    if reflection.centipawn_loss is None:
        return "outcome unmeasured"
    label = f" ({reflection.blunder_label})" if reflection.blunder_label else ""
    return f"lost {reflection.centipawn_loss:.0f}cp{label}"


def _history_lines(state: SelfModelState, max_moves: int) -> list[str]:
    lines: list[str] = []
    for r in state.reflections[-max_moves:]:
        line = f"- ply {r.ply_index}: {r.san} — confidence {r.confidence:.2f} → {_outcome_phrase(r)}."
        if r.fallback_used:
            line += " (fallback move — my own choice failed validation)"
        lines.append(line)
    return lines


def render_history_block(state: SelfModelState, max_moves: int = DEFAULT_HISTORY_MOVES) -> str:
    """Condition B: factual record of recent decisions and outcomes, no interpretation."""
    if state.moves_observed == 0:
        return ""
    lines = ["## Decision History (your own recent moves this game)"]
    lines.extend(_history_lines(state, max_moves))
    return "\n".join(lines)


def render_self_model_block(state: SelfModelState, max_moves: int = DEFAULT_HISTORY_MOVES) -> str:
    """Condition C: history plus the derived self-model — calibration, biases, directives."""
    if state.moves_observed == 0:
        return ""

    lines = ["## Self-Model (your own decision record and known biases)"]
    lines.append("### Recent decisions")
    lines.extend(_history_lines(state, max_moves))

    lines.append("### Judgment summary")
    conf = f"{state.mean_confidence:.2f}" if state.mean_confidence is not None else "n/a"
    cpl = f"{state.mean_cpl:.0f}cp" if state.mean_cpl is not None else "n/a"
    lines.append(
        f"- {state.moves_observed} decisions observed; mean confidence {conf}; "
        f"mean centipawn loss {cpl}; {state.blunder_count} blunder(s)."
    )

    if state.patterns:
        lines.append("### Known biases")
        for pattern in state.patterns:
            lines.append(f"- {pattern.text}")
        lines.append("### Directives")
        for directive in _directives(state.patterns):
            lines.append(f"- {directive}")
    else:
        lines.append("- No recurring bias detected yet. Keep confidence tied to concrete analysis.")

    return "\n".join(lines)


def _directives(patterns: list[SelfPattern]) -> list[str]:
    seen: list[str] = []
    for pattern in patterns:
        directive = _DIRECTIVES[pattern.kind]
        if directive not in seen:
            seen.append(directive)
    return seen
