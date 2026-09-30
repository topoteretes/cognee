from typing import Any, Generic, Optional, cast

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, TypeVar

from cognee.infrastructure.engine.models.DataPoint import DataPoint

Source = TypeVar("Source", bound=DataPoint, default=DataPoint)
Target = TypeVar("Target", bound=DataPoint, default=DataPoint)
RelationshipType = TypeVar("RelationshipType", default=str | None)


class Edge(BaseModel, Generic[Source, Target, RelationshipType]):
    """One relationship between DataPoints: endpoints plus metadata.

    As a type, ``Edge[Source, Target, RelationshipType]`` declares a typed edge field
    on a custom graph model: ``friends_with: list[Edge[Person, Person]]`` asks the LLM
    for flat relationship rows that cognee resolves into real edges. The third
    parameter controls naming — omitted: the field name; ``Literal["a", "b"]``: the
    LLM picks one; ``str``: free-form.

    As a value, an Edge may be written partially; ``fill_endpoints`` completes it from
    the field it is declared on. The relationship name falls back to the field name and
    the source falls back to the declaring node — on a parametrized edge the declaring
    node must match the declared ``Source``, so set ``source=`` explicitly for edges
    declared on a container. A missing target is always an error.

    ``to_properties`` is the storage property bag: every set metadata field, endpoints
    excluded.

    The tuple spelling is the older form, still supported:

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

    source: Source | None = None
    target: Target | None = None
    relationship_type: RelationshipType = cast(Any, None)

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

    def fill_endpoints(
        self,
        owner: DataPoint,
        field_name: str,
        target: DataPoint | None = None,
    ) -> Self:
        """Fill in source, target and relationship_type from the declaring field.

        An ``Edge`` written on a model usually carries only metadata; what makes it
        storable is how it is addressed, and that comes from where it sits. Those three
        fields are exactly the ones ``to_properties`` leaves out.

        Source falls back to ``owner``, name to ``self.relationship_type or field_name``.
        When both ``self.target`` and the ``target`` argument are set, the argument
        wins: it is the tuple form's target at the point of use.

        Returns a filled copy, or ``self`` when nothing was left to fill in.
        """
        resolved_source = self.source if self.source is not None else owner
        resolved_target = target if target is not None else self.target
        resolved_name = self.relationship_type or field_name

        if resolved_target is None:
            raise ValueError(
                "Edge.fill_endpoints requires a target: set Edge.target or pass target=..."
            )

        if self.source is None:
            self._check_owner_is_declared_source(owner, field_name)

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

    def _check_owner_is_declared_source(self, owner: Any, field_name: str) -> None:
        """An omitted source falls back to ``owner``; hold the fallback to the generics.

        On an owner-declared edge the owner *is* the declared ``Source``, so the
        fallback passes. On a root container it silently made the container the
        subject, contradicting the declared type — that now raises, like a missing
        target. Matching is by ``isinstance``, or by class name against the owner's
        MRO: string endpoints (``Edge["Person", "Person"]``) and ``copy_model`` copies
        carry the class name without the class object. An unparametrized Edge (the
        tuple form and other legacy spellings) records no generics and keeps the
        permissive fallback.
        """
        metadata = getattr(type(self), "__pydantic_generic_metadata__", None) or {}
        args = metadata.get("args") or ()
        if not args:
            return

        declared = args[0]
        if isinstance(declared, type):
            if isinstance(owner, declared):
                return
            declared_name = declared.__name__
        else:
            declared_name = getattr(declared, "__forward_arg__", declared)

        if isinstance(declared_name, str) and declared_name in {
            base.__name__ for base in type(owner).__mro__
        }:
            return

        raise ValueError(
            f"Edge.fill_endpoints: {field_name!r} left source unset on a "
            f"{type(owner).__name__}, which is not the declared source type "
            f"({declared!r}). An omitted source falls back to the node the field is "
            f"declared on; set source= explicitly for edges declared on a container."
        )

    def to_properties(self) -> dict[str, Any]:
        """Edge metadata for storage, excluding source and target.

        ``relationship_type`` stays in the bag: stored-edge readers (visualization,
        edge-to-text, the global context index) still look it up there. Migrating them
        to the first-class ``relationship_name`` is tracked separately.
        """
        data = self.model_dump(
            exclude_none=True,
            exclude={"source", "target"},
        )
        if self.weights is not None:
            # Flattened to scalar weight_<name> properties so graph backends can
            # filter on them without JSON support.
            for weight_name, weight_value in self.weights.items():
                data[f"weight_{weight_name}"] = weight_value
        return data
