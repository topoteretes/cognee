"""Reading a model's fields: which of them declare relationships.

A field either points at other DataPoints or holds a plain value. Every part of the
graph walk asks that question, and this module is the only place it is answered.

These are free functions rather than ``DataPoint`` methods for two reasons: the model
class should not know the graph walk exists, and not every node the walk visits is a
``DataPoint``. ``copy_model`` mints plain ``BaseModel`` subclasses that keep the
original class name, and those get walked too — a chunk rebuilt from an export is one,
and it still holds real ``DataPoint`` children whose edges have to be emitted.

``_iter_edges_in_field`` is the single decider; ``split_field_edges`` drains it once
per field and answers both halves of the question in the same pass, so no ``Edge`` is
ever built twice or built only to be discarded.
"""

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint, Edge

# Values of these types can never declare an edge. Most fields on a node are DataPoint
# infrastructure scalars, so this check keeps the per-field generator machinery off the
# hot path entirely. Conservative on purpose: anything not listed goes through the real
# classifier.
_NEVER_EDGES = (type(None), str, int, float, bool, dict)


def split_field_edges(owner: BaseModel) -> tuple[list[tuple[str, Edge]], list[tuple[str, Any]]]:
    """One pass over the fields: ``(edges, plain_fields)``.

    ``edges`` is ``(field_name, edge)`` for every edge a field expands into; targets
    stay raw — a transparent container is still a container here. ``plain_fields`` is
    ``(field_name, value)`` for the fields that expanded into none.
    """
    edges: list[tuple[str, Edge]] = []
    plain_fields: list[tuple[str, Any]] = []
    for field_name, value in owner:
        if field_name == "metadata":
            continue
        if isinstance(value, _NEVER_EDGES):
            plain_fields.append((field_name, value))
            continue
        found = len(edges)
        edges.extend((field_name, edge) for edge in _iter_edges_in_field(owner, field_name, value))
        if len(edges) == found:
            plain_fields.append((field_name, value))
    return edges, plain_fields


def get_edges_from_fields(owner: BaseModel) -> list[tuple[str, Edge]]:
    """The edge half of ``split_field_edges`` alone."""
    return split_field_edges(owner)[0]


def get_fields_without_edges(owner: BaseModel) -> list[tuple[str, Any]]:
    """The plain-value half of ``split_field_edges`` alone."""
    return split_field_edges(owner)[1]


def _iter_edges_in_field(owner: BaseModel, field_name: str, value: Any) -> Iterator[Edge]:
    """Every edge one field declares. Nothing yielded means the field is a stored value."""
    items = value if isinstance(value, list) else [value]
    for item in items:
        if isinstance(item, DataPoint):
            yield Edge.model_construct(source=owner, target=item, relationship_type=field_name)
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], Edge):
            edge_metadata, targets = item
            if isinstance(targets, DataPoint):
                yield edge_metadata.fill_endpoints(owner, field_name, target=targets)
                continue
            if isinstance(targets, list) and targets and isinstance(targets[0], DataPoint):
                # Checked per item: a loosely-typed field (e.g. ``tuple | None``) can
                # hold a mixed list, and a non-DataPoint target cannot become an edge.
                for inner in targets:
                    if isinstance(inner, DataPoint):
                        yield edge_metadata.fill_endpoints(owner, field_name, target=inner)
            continue
        if isinstance(item, Edge) and item.target is not None:
            yield item.fill_endpoints(owner, field_name)
