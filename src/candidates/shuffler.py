from __future__ import annotations

import random

from src.storage.models import CandidateMoveRecord


def shuffle_candidates(
    candidates: list[CandidateMoveRecord],
    rng: random.Random | None = None,
) -> list[CandidateMoveRecord]:
    """Randomize presentation order and assign presentation_index.

    internal_rank is preserved unchanged. The returned list is in shuffled
    (presentation) order, with presentation_index reflecting that order.
    """
    if rng is None:
        rng = random.Random()

    indices = list(range(len(candidates)))
    rng.shuffle(indices)

    shuffled: list[CandidateMoveRecord] = []
    for presentation_idx, original_idx in enumerate(indices):
        c = candidates[original_idx]
        shuffled.append(c.model_copy(update={"presentation_index": presentation_idx}))

    return shuffled
