from pydantic import BaseModel
from typing import Optional, Any, Dict


class Edge(BaseModel):
    """
    Represents edge metadata for relationships between DataPoints.
    
    This class is used to define edge properties like weight when creating
    relationships between DataPoints using tuple syntax:
    
    Example:
        has_items: (Edge(weight=0.5), list[Item])
    """
    
    weight: Optional[float] = None
    relationship_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge metadata to dictionary for storage in graph database."""
        result = {}
        if self.weight is not None:
            result["weight"] = self.weight
        if self.relationship_type is not None:
            result["relationship_type"] = self.relationship_type
        if self.properties is not None:
            result.update(self.properties)
        return result 