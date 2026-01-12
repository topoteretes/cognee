from pydantic import BaseModel, field_validator
from typing import Optional, Any, Dict


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
        contains: (Edge(relationship_type="contains", edge_text="relationship_name: contains; entity_description: Alice"), Entity)
    """

    weight: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    relationship_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    edge_text: Optional[str] = None

    @field_validator("edge_text", mode="before")
    @classmethod
    def ensure_edge_text(cls, v, info):
        """Auto-populate edge_text from relationship_type if not explicitly provided."""
        if v is None and info.data.get("relationship_type"):
            return info.data["relationship_type"]
        return v
