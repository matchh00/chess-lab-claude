from __future__ import annotations

from src.storage.models import PrimitiveValue


def render_primitive_text(pv: PrimitiveValue) -> str:
    return pv.text_render


def render_primitives_summary(primitives: list[PrimitiveValue], top_n: int = 10) -> str:
    """Render top-N primitives sorted by normalized_value * confidence descending."""
    scored = sorted(primitives, key=lambda p: p.normalized_value * p.confidence, reverse=True)
    lines = [f"  [{p.category}] {p.text_render} (conf={p.confidence:.2f})" for p in scored[:top_n]]
    return "\n".join(lines)


def render_primitives_by_category(primitives: list[PrimitiveValue]) -> str:
    from collections import defaultdict
    by_cat: dict[str, list[PrimitiveValue]] = defaultdict(list)
    for p in primitives:
        by_cat[p.category].append(p)

    sections = []
    for cat in sorted(by_cat):
        header = f"[{cat.upper()}]"
        lines = [f"  {p.text_render}" for p in by_cat[cat]]
        sections.append(header + "\n" + "\n".join(lines))
    return "\n\n".join(sections)


def render_full_extraction(primitives: list[PrimitiveValue]) -> str:
    """Full text dump of all extracted primitives for inspection/debugging."""
    lines = []
    for p in primitives:
        lines.append(
            f"{p.primitive_id:<45} val={p.value!r:<10} norm={p.normalized_value:.3f}"
            f"  conf={p.confidence:.2f}  | {p.text_render}"
        )
    return "\n".join(lines)
