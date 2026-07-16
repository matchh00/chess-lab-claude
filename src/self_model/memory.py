from __future__ import annotations

import logging
from typing import Literal, Optional

from src.self_model.models import CategoryBias, MoveReflection, SelfModelState, SelfPattern
from src.storage.models import DecisionRecord, WeightedPrimitiveState

logger = logging.getLogger(__name__)

# Thresholds are aligned with src.analytics.move_metrics: a move losing more
# than MISTAKE_THRESHOLD (100cp) counts as costly for self-assessment.
HIGH_CONFIDENCE = 0.7
COSTLY_CPL = 100.0
CLEAN_CPL = 50.0

# A category bias is only reported once it has enough support and the
# conditional mean CPL is clearly worse than the overall mean.
MIN_CATEGORY_DECISIONS = 3
CATEGORY_BIAS_RATIO = 1.5

COST_STREAK_LENGTH = 3
MIN_OVERCONFIDENT_MOVES = 2

DOMINANT_CATEGORY_COUNT = 2


def dominant_categories(
    weighted_states: list[WeightedPrimitiveState],
    top_n: int = DOMINANT_CATEGORY_COUNT,
) -> list[str]:
    """Categories of the strongest weighted signals at decision time, deduplicated."""
    ranked = sorted(weighted_states, key=lambda s: s.weighted_score, reverse=True)
    seen: list[str] = []
    for state in ranked:
        if state.category not in seen:
            seen.append(state.category)
        if len(seen) >= top_n:
            break
    return seen


class SelfModelMemory:
    """Accumulates MoveReflections and derives the lab's model of its own judgment.

    scope="game" resets between games (continuity within one game only);
    scope="run" persists across all games in a run, letting detected biases
    carry over to later games. Running the same condition with the memory
    disabled (mode "off") is the ablation test for causal force.
    """

    def __init__(self, scope: Literal["game", "run"] = "game") -> None:
        self.scope = scope
        self._reflections: list[MoveReflection] = []

    def start_game(self) -> None:
        if self.scope == "game":
            self._reflections = []

    def observe(self, reflection: MoveReflection) -> None:
        self._reflections.append(reflection)

    def observe_move(
        self,
        game_id: str,
        ply_index: int,
        san: str,
        decision: DecisionRecord,
        centipawn_loss: Optional[float],
        blunder_label: Optional[str],
        weighted_states: list[WeightedPrimitiveState],
    ) -> MoveReflection:
        reflection = MoveReflection(
            game_id=game_id,
            ply_index=ply_index,
            san=san,
            uci=decision.selected_uci,
            confidence=decision.confidence,
            centipawn_loss=centipawn_loss,
            blunder_label=blunder_label,
            reasoning_summary=decision.reasoning_summary,
            dominant_categories=dominant_categories(weighted_states),
            fallback_used=decision.fallback_used,
        )
        self.observe(reflection)
        return reflection

    @property
    def reflections(self) -> list[MoveReflection]:
        return list(self._reflections)

    def state(self) -> SelfModelState:
        refs = self._reflections
        scored = [r for r in refs if r.centipawn_loss is not None]
        confident = [r for r in refs if not r.fallback_used]

        mean_confidence = (
            sum(r.confidence for r in confident) / len(confident) if confident else None
        )
        mean_cpl = (
            sum(r.centipawn_loss for r in scored) / len(scored) if scored else None
        )
        blunder_count = sum(1 for r in refs if r.blunder_label == "blunder")

        costly = [r for r in scored if r.centipawn_loss >= COSTLY_CPL]
        clean = [r for r in scored if r.centipawn_loss < CLEAN_CPL]
        high_conf_costly = [r for r in costly if r.confidence >= HIGH_CONFIDENCE]

        mean_conf_costly = (
            sum(r.confidence for r in costly) / len(costly) if costly else None
        )
        mean_conf_clean = (
            sum(r.confidence for r in clean) / len(clean) if clean else None
        )

        biases = self._category_biases(scored)
        patterns = self._detect_patterns(
            scored=scored,
            high_conf_costly=high_conf_costly,
            mean_cpl=mean_cpl,
            mean_conf_costly=mean_conf_costly,
            mean_conf_clean=mean_conf_clean,
            biases=biases,
        )

        return SelfModelState(
            scope=self.scope,
            moves_observed=len(refs),
            mean_confidence=mean_confidence,
            mean_cpl=mean_cpl,
            blunder_count=blunder_count,
            high_confidence_costly_moves=len(high_conf_costly),
            mean_confidence_on_costly=mean_conf_costly,
            mean_confidence_on_clean=mean_conf_clean,
            category_biases=biases,
            patterns=patterns,
            reflections=list(refs),
        )

    @staticmethod
    def _category_biases(scored: list[MoveReflection]) -> list[CategoryBias]:
        by_category: dict[str, list[MoveReflection]] = {}
        for r in scored:
            for category in r.dominant_categories:
                by_category.setdefault(category, []).append(r)

        biases = [
            CategoryBias(
                category=category,
                decision_count=len(group),
                mean_cpl=sum(r.centipawn_loss for r in group) / len(group),
                blunder_count=sum(1 for r in group if r.blunder_label == "blunder"),
            )
            for category, group in by_category.items()
        ]
        biases.sort(key=lambda b: (b.mean_cpl or 0.0), reverse=True)
        return biases

    @staticmethod
    def _detect_patterns(
        scored: list[MoveReflection],
        high_conf_costly: list[MoveReflection],
        mean_cpl: Optional[float],
        mean_conf_costly: Optional[float],
        mean_conf_clean: Optional[float],
        biases: list[CategoryBias],
    ) -> list[SelfPattern]:
        patterns: list[SelfPattern] = []

        if len(high_conf_costly) >= MIN_OVERCONFIDENT_MOVES:
            sans = ", ".join(r.san for r in high_conf_costly[-3:])
            patterns.append(SelfPattern(
                kind="overconfidence",
                text=(
                    f"{len(high_conf_costly)} of my high-confidence choices "
                    f"(≥{HIGH_CONFIDENCE:.1f}) lost {COSTLY_CPL:.0f}+ centipawns "
                    f"(recently: {sans}). My confidence is not tracking move quality."
                ),
            ))

        if (
            mean_conf_costly is not None
            and mean_conf_clean is not None
            and mean_conf_costly >= mean_conf_clean
        ):
            patterns.append(SelfPattern(
                kind="calibration_gap",
                text=(
                    f"My average confidence on costly moves ({mean_conf_costly:.2f}) "
                    f"is not lower than on clean moves ({mean_conf_clean:.2f}). "
                    "Confidence alone is not separating good moves from bad ones."
                ),
            ))

        if mean_cpl is not None and mean_cpl > 0:
            for bias in biases:
                if (
                    bias.decision_count >= MIN_CATEGORY_DECISIONS
                    and bias.mean_cpl is not None
                    and bias.mean_cpl >= CATEGORY_BIAS_RATIO * mean_cpl
                ):
                    patterns.append(SelfPattern(
                        kind="category_bias",
                        text=(
                            f"When '{bias.category}' signals dominate my reading, "
                            f"I average {bias.mean_cpl:.0f}cp loss over "
                            f"{bias.decision_count} decisions vs {mean_cpl:.0f}cp overall. "
                            f"That signal is overpowering my judgment."
                        ),
                    ))

        if len(scored) >= COST_STREAK_LENGTH:
            recent = scored[-COST_STREAK_LENGTH:]
            if all(r.centipawn_loss >= COSTLY_CPL for r in recent):
                patterns.append(SelfPattern(
                    kind="cost_streak",
                    text=(
                        f"My last {COST_STREAK_LENGTH} moves each lost "
                        f"{COSTLY_CPL:.0f}+ centipawns. My current line of reasoning "
                        "is failing repeatedly."
                    ),
                ))

        return patterns
