"""Pydantic-native graph export: get your memory back as typed objects.

``cognee.export(..., format="pydantic")`` returns a :class:`GraphSnapshot` — a
Pydantic model whose ``nodes`` are rehydrated into the same DataPoint
subclasses that produced them (``Entity``, ``DocumentChunk``, your custom
models, ...). Because DataPoint *is* a Pydantic model, the full round trip is
native:

    snapshot = await cognee.export("main_dataset", format="pydantic")
    alice = snapshot.find(name="Alice")[0]       # a real Entity instance
    blob = snapshot.model_dump_json()            # lossless serialization
    again = GraphSnapshot.model_validate_json(blob)  # typed objects restored

Rehydration resolves a node's stored ``type`` against all loaded DataPoint
subclasses (discovered via ``__subclasses__``, so importing your custom model
module is enough to get your own classes back). Unknown types degrade
gracefully to dynamically created DataPoint subclasses with ``extra="allow"``
so no property is lost.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, create_model, field_validator

from cognee.infrastructure.engine import DataPoint

# Edge properties that are internal bookkeeping rather than knowledge content.
_SKIP_EDGE_KEYS = ("source_node_id", "target_node_id", "relationship_name")

# Base DataPoint fields are never relation targets for link_relations().
_BASE_FIELDS = frozenset(DataPoint.model_fields.keys())


def datapoint_registry() -> Dict[str, Type[DataPoint]]:
    """All currently loaded DataPoint subclasses, keyed by class name.

    Also keyed by ``module.ClassName`` so identically named classes from
    different modules stay individually addressable.
    """
    registry: Dict[str, Type[DataPoint]] = {}

    def _walk(cls: Type[DataPoint]) -> None:
        for subclass in cls.__subclasses__():
            # Dynamic fallback models are never authoritative for a type name.
            if issubclass(subclass, _DynamicDataPoint) or subclass is _DynamicDataPoint:
                continue
            registry.setdefault(subclass.__name__, subclass)
            registry[f"{subclass.__module__}.{subclass.__name__}"] = subclass
            _walk(subclass)

    _walk(DataPoint)
    return registry


class _DynamicDataPoint(DataPoint):
    """Base for on-the-fly models of unknown node types: keep every property."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


_dynamic_models: Dict[str, Type[DataPoint]] = {}


def _dynamic_model(type_name: str) -> Type[DataPoint]:
    if type_name not in _dynamic_models:
        _dynamic_models[type_name] = create_model(type_name, __base__=_DynamicDataPoint)
    return _dynamic_models[type_name]


def _clean_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    props = dict(properties)
    # Graph stores may serialize dict-valued fields as JSON strings.
    for key in ("metadata", "belongs_to_set"):
        value = props.get(key)
        if isinstance(value, str):
            try:
                props[key] = json.loads(value)
            except (ValueError, TypeError):
                props.pop(key)
    return props


def rehydrate_node(
    properties: Dict[str, Any], registry: Optional[Dict[str, Type[DataPoint]]] = None
) -> DataPoint:
    """Turn stored node properties back into a typed DataPoint instance.

    Resolution order: the registered class for the node's ``type`` -> a
    dynamically created DataPoint subclass with the same name (``extra="allow"``)
    -> the dynamic class fed only DataPoint-safe base fields.
    """
    props = _clean_properties(properties)
    type_name = str(props.get("type") or "DataPoint")
    registry = registry if registry is not None else datapoint_registry()

    known = registry.get(type_name)
    if known is not None:
        try:
            return known(**props)
        except Exception:  # noqa: BLE001 — fall back to a dynamic model
            pass

    try:
        # Created lazily: only when the registered class is absent or fails.
        return _dynamic_model(type_name)(**props)
    except Exception:  # noqa: BLE001 — final fallback below
        pass

    base_safe = {key: value for key, value in props.items() if key in _BASE_FIELDS}
    return _dynamic_model(type_name)(**base_safe)


class GraphEdge(BaseModel):
    """A typed graph relationship between two node ids."""

    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphSnapshot(BaseModel):
    """A dataset's knowledge graph as typed Pydantic objects.

    ``nodes`` hold real DataPoint subclass instances (``SerializeAsAny`` keeps
    every subclass field through ``model_dump_json``; the before-validator
    rehydrates them on ``model_validate_json``), so the snapshot is both an
    in-memory object graph and a lossless file format.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str = ""
    dataset_id: str = ""
    nodes: List[SerializeAsAny[DataPoint]] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _rehydrate_nodes(cls, value):
        if not isinstance(value, list):
            return value
        registry = datapoint_registry()
        return [
            rehydrate_node(item, registry) if isinstance(item, dict) else item for item in value
        ]

    def nodes_of_type(self, node_type: Union[str, Type[DataPoint]]) -> List[DataPoint]:
        if isinstance(node_type, type):
            return [node for node in self.nodes if isinstance(node, node_type)]
        return [node for node in self.nodes if node.type == node_type]

    def find(
        self, node_type: Union[str, Type[DataPoint], None] = None, **field_filters: Any
    ) -> List[DataPoint]:
        """Find nodes by type and/or exact field values: ``find(Entity, name="Alice")``."""
        nodes = self.nodes_of_type(node_type) if node_type is not None else list(self.nodes)
        for field_name, expected in field_filters.items():
            nodes = [node for node in nodes if getattr(node, field_name, None) == expected]
        return nodes

    def link_relations(self) -> "GraphSnapshot":
        """Re-attach edges as object references on declared relation fields.

        Nested DataPoints are flattened into edges at write time; this walks
        the edges back and, where the relationship name matches a field the
        source node's class declares (e.g. ``Entity.is_a``), sets the target
        instance on it — turning the two lists into a traversable object graph.
        """
        by_id: Dict[str, DataPoint] = {str(node.id): node for node in self.nodes}
        for edge in self.edges:
            source = by_id.get(edge.source_id)
            target = by_id.get(edge.target_id)
            if source is None or target is None:
                continue
            field_name = edge.relationship
            if field_name in _BASE_FIELDS or field_name not in type(source).model_fields:
                continue
            current = getattr(source, field_name, None)
            if current is None:
                object.__setattr__(source, field_name, target)
            elif isinstance(current, list) and target not in current:
                current.append(target)
        return self

    def save(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "GraphSnapshot":
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def __repr__(self):
        return (
            f"GraphSnapshot(dataset={self.dataset_name!r}, "
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )


def build_snapshot(
    nodes,
    edges,
    dataset_name: str = "",
    dataset_id: str = "",
    link_relations: bool = False,
) -> GraphSnapshot:
    """Build a GraphSnapshot from ``get_graph_data()`` shapes.

    Nodes are ``(node_id, properties)`` tuples; edges are ``(source_id,
    target_id, relationship_name, properties)`` tuples.
    """
    snapshot = GraphSnapshot(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        nodes=[{**(properties or {}), "id": str(node_id)} for node_id, properties in nodes],
        edges=[
            GraphEdge(
                source_id=str(source),
                target_id=str(target),
                relationship=str(relationship),
                properties={
                    key: value
                    for key, value in (properties or {}).items()
                    if key not in _SKIP_EDGE_KEYS and value is not None
                },
            )
            for source, target, relationship, properties in edges
        ],
    )
    if link_relations:
        snapshot.link_relations()
    return snapshot
