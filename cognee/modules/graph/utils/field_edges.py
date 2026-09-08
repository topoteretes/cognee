"""Reading a model's fields: which of them declare relationships.

A field either points at other DataPoints or holds a plain value. Every part of the
graph walk asks that question, and this module is the only place it is answered.

These are free functions rather than ``DataPoint`` methods for two reasons: the model
class should not know the graph walk exists, and not every node the walk visits is a
``DataPoint``. ``copy_model`` mints plain ``BaseModel`` subclasses that keep the
original class name, and those get walked too — a chunk rebuilt from an export is one,
and it still holds real ``DataPoint`` children whose edges have to be emitted.
"""

from typing import Any

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint, Edge


def get_edges_from_fields(owner: BaseModel) -> list[tuple[str, Edge]]:
    """Return ``(field_name, edge)`` for every field that expands into graph edges.

    Targets stay raw: a transparent container is still a container here.
    """
    return [
        (field_name, edge)
        for field_name, field_value in owner
        if field_name != "metadata"
        for edge in _edges_in_field(owner, field_name, field_value)
    ]


def get_fields_without_edges(owner: BaseModel) -> list[tuple[str, Any]]:
    """Return ``(field_name, value)`` for fields that do not expand into edges."""
    return [
        (field_name, field_value)
        for field_name, field_value in owner
        if field_name != "metadata" and not _edges_in_field(owner, field_name, field_value)
    ]


def _edges_in_field(owner: BaseModel, field_name: str, value: Any) -> list[Edge]:
    """Every edge one field declares. Empty means the field is a stored value."""
    items = value if isinstance(value, list) else [value]
    edges: list[Edge] = []
    for item in items:
        if isinstance(item, DataPoint):
            edges.append(
                Edge.model_construct(source=owner, target=item, relationship_type=field_name)
            )
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Edge):
            edge_metadata, targets = item
            if isinstance(targets, DataPoint):
                edges.append(edge_metadata.fill_endpoints(owner, field_name, target=targets))
                continue
            if isinstance(targets, list) and targets and isinstance(targets[0], DataPoint):
                for inner in targets:
                    edges.append(edge_metadata.fill_endpoints(owner, field_name, target=inner))
            continue
        if isinstance(item, Edge) and item.target is not None:
            edges.append(item.fill_endpoints(owner, field_name))
    return edges
