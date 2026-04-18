from __future__ import annotations

from src.narratives.token_budget import count_words, enforce_word_limit


class GameNarrative:
    """Rolling game narrative that stays under a word budget."""

    def __init__(self, max_words: int = 200) -> None:
        self.max_words = max_words
        self._entries: list[dict] = []  # {"ply": int, "san": str, "summary": str}

    def add(self, ply: int, san: str, summary: str) -> None:
        self._entries.append({"ply": ply, "san": san, "summary": summary})
        self._prune()

    def _prune(self) -> None:
        while len(self._entries) > 1 and count_words(self.render()) > self.max_words:
            self._entries.pop(0)

    def render(self) -> str:
        parts: list[str] = []
        for entry in self._entries:
            parts.append(f"Ply {entry['ply']} ({entry['san']}): {entry['summary']}")
        text = " | ".join(parts)
        truncated, _ = enforce_word_limit(text, self.max_words)
        return truncated

    def to_dict(self) -> dict:
        return {
            "max_words": self.max_words,
            "entries": list(self._entries),
            "rendered": self.render(),
            "word_count": count_words(self.render()),
        }
