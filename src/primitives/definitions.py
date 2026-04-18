from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field


ValueType = Literal["bool", "int", "float", "category", "text"]
SideScope = Literal["self", "opponent", "global"]


class PrimitiveDefinition(BaseModel):
    primitive_id: str
    name: str
    category: str
    description: str
    value_type: ValueType
    side_scope: SideScope
    render_template: str
    version: str = "1.0"
    enabled: bool = True
    default_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Not serialized — set after construction via registry
    extraction_fn: Optional[Any] = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def render(self, value: Any) -> str:
        """Fill render_template with the extracted value."""
        try:
            return self.render_template.format(value=value)
        except (KeyError, ValueError):
            return f"{self.name}: {value}"
