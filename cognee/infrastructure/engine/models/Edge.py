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

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge metadata to dictionary for storage in graph database."""
        result = {}

        # Add single weight if present
        if self.weight is not None:
            result["weight"] = self.weight

        # Add multiple weights if present
        if self.weights is not None:
            result["weights"] = self.weights
            # Also add individual weights as separate fields for easier querying
            for weight_name, weight_value in self.weights.items():
                result[f"weight_{weight_name}"] = weight_value

        if self.relationship_type is not None:
            result["relationship_type"] = self.relationship_type

        if self.properties is not None:
            result.update(self.properties)

        return result

    def get_weight(self, weight_name: Optional[str] = None) -> Optional[float]:
        """
        Get a specific weight value.

        Args:
            weight_name: Name of the weight to retrieve. If None, returns the default weight.

        Returns:
            The weight value or None if not found.
        """
        if weight_name is None:
            return self.weight

        if self.weights and weight_name in self.weights:
            return self.weights[weight_name]

        return None

    def get_all_weights(self) -> Dict[str, float]:
        """
        Get all weights as a dictionary.

        Returns:
            Dictionary containing all weights. Single weight is included as 'default' if present.
        """
        result = {}

        if self.weight is not None:
            result["default"] = self.weight

        if self.weights is not None:
            result.update(self.weights)

        return result
