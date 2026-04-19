from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

# v1.2 two-call token targets
INTERPRETER_MAX_TOKENS: int = 400
DECISION_MAX_TOKENS: int = 1000
COMBINED_TARGET_TOKENS: int = 1400


@dataclass
class BudgetResult:
    text: str
    word_count: int
    token_count: int
    was_truncated: bool
    budget_type: str  # "words" or "tokens"


def count_words(text: str) -> int:
    return len(text.split()) if text.strip() else 0


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def enforce_word_limit(text: str, limit: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= limit:
        return text, False
    return " ".join(words[:limit]), True


def check_budget(text: str, limit_words: int, budget_type: str = "words") -> BudgetResult:
    truncated_text, was_truncated = enforce_word_limit(text, limit_words)
    return BudgetResult(
        text=truncated_text,
        word_count=count_words(truncated_text),
        token_count=count_tokens(truncated_text),
        was_truncated=was_truncated,
        budget_type=budget_type,
    )
