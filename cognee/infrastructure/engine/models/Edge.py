from typing import Any

from pydantic import BaseModel


class Edge(BaseModel):
    """
    Represents edge metadata for relationships between DataPoints.

    This class is used to define edge properties like weight when creating
    relationships between DataPoints using tuple syntax:

    Example:
        # Single weight (backward compatible)
        has_items: (Edge(weight=0.5), list[Item])

        # Multiple weights
        has_items: (Edge(weights={"strength": 0.8, "confidence": 0.9, "importance": 0.7}), list[Item])

        # Mixed usage
        has_items: (Edge(weight=0.5, weights={"confidence": 0.9}), list[Item])

        # With edge_text for rich embedding representation
        contains: (Edge(relationship_type="contains", edge_text="This chunk mentions Alice: Alice works at Acme."), Entity)
    """

    weight: float | None = None
    weights: dict[str, float] | None = None
    relationship_type: str | None = None
    properties: dict[str, Any] | None = None
    edge_text: str | None = None
