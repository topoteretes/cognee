from typing import Any, Generic, Optional

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import TypeVar

from cognee.infrastructure.engine.models.DataPoint import DataPoint

Source = TypeVar("Source", bound=DataPoint, default=DataPoint)
Target = TypeVar("Target", bound=DataPoint, default=DataPoint)
RelationshipType = TypeVar("RelationshipType", default=Optional[str])


class Edge(BaseModel, Generic[Source, Target, RelationshipType]):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Optional[Source] = None
    target: Optional[Target] = None
    relationship_type: RelationshipType = None

    weight: float | None = None
    weights: dict[str, float] | None = None
    properties: dict[str, Any] | None = None
    edge_text: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_edge_instance(cls, value):
        """Let a bare ``Edge`` revalidate against a parametrized annotation."""
        if isinstance(value, Edge):
            return dict(value.__dict__)
        return value

    def normalize(
        self,
        owner: DataPoint,
        field_name: str,
        target: DataPoint | None = None,
    ) -> "Edge":
        """Return an Edge with source, target and relationship_type filled from context.

        Source falls back to ``owner``, name to ``self.relationship_type or field_name``.
        When both ``self.target`` and the ``target`` argument are set, the argument
        wins: it is the tuple form's target at the point of use.
        """
        resolved_source = self.source if self.source is not None else owner
        resolved_target = target if target is not None else self.target
        resolved_name = self.relationship_type or field_name

        if resolved_target is None:
            raise ValueError("Edge.normalize requires a target: set Edge.target or pass target=...")

        if (
            self.source is resolved_source
            and self.target is resolved_target
            and self.relationship_type == resolved_name
        ):
            return self

        return self.model_copy(
            update={
                "source": resolved_source,
                "target": resolved_target,
                "relationship_type": resolved_name,
            }
        )

    def to_properties(self) -> dict[str, Any]:
        """Edge metadata for storage, excluding source, target and relationship_type."""
        data = self.model_dump(
            exclude_none=True,
            exclude={"source", "target", "relationship_type"},
        )
        if self.weights is not None:
            for weight_name, weight_value in self.weights.items():
                data[f"weight_{weight_name}"] = weight_value
        return data
