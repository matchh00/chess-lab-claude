from __future__ import annotations

from typing import Callable, Iterator, Optional

from src.primitives.definitions import PrimitiveDefinition


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._primitives: dict[str, PrimitiveDefinition] = {}

    def add(self, defn: PrimitiveDefinition, fn: Optional[Callable] = None) -> None:
        if fn is not None:
            defn.extraction_fn = fn
        self._primitives[defn.primitive_id] = defn

    def get(self, primitive_id: str) -> PrimitiveDefinition:
        if primitive_id not in self._primitives:
            raise KeyError(f"Primitive not found: {primitive_id}")
        return self._primitives[primitive_id]

    def enable(self, primitive_id: str) -> None:
        self.get(primitive_id).enabled = True

    def disable(self, primitive_id: str) -> None:
        self.get(primitive_id).enabled = False

    def version(self, primitive_id: str) -> str:
        return self.get(primitive_id).version

    def all_enabled(self) -> list[PrimitiveDefinition]:
        return [d for d in self._primitives.values() if d.enabled]

    def all(self) -> list[PrimitiveDefinition]:
        return list(self._primitives.values())

    def by_category(self, category: str) -> list[PrimitiveDefinition]:
        return [d for d in self._primitives.values() if d.category == category and d.enabled]

    def __iter__(self) -> Iterator[PrimitiveDefinition]:
        return iter(self.all_enabled())

    def __len__(self) -> int:
        return len(self._primitives)
