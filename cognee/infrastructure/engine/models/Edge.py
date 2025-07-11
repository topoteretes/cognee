from pydantic import BaseModel
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
    """

    weight: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    relationship_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
