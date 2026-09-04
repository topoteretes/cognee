"""Reading a DataPoint's fields: which of them declare relationships.

A field either points at other DataPoints or holds a plain value. Every part of the
graph walk asks that question, and this module is the only place it is answered.
"""

from typing import Any, Iterator, List, Optional, Tuple

from cognee.infrastructure.engine import DataPoint, Edge

# One relationship declaration: the edge metadata (if any) and everything it points at.
EdgeTargets = Tuple[Optional[Edge], List[DataPoint]]


def _as_edge_targets(value: Any) -> Optional[EdgeTargets]:
    """Read one relationship declaration, or None when ``value`` is a plain property.

    Accepts a bare DataPoint or an ``(Edge, DataPoint | list[DataPoint])`` tuple —
    the two forms a field, or an item inside a list field, may take.
    """
    if isinstance(value, DataPoint):
        return (None, [value])

    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Edge):
        edge_metadata, targets = value
        if isinstance(targets, DataPoint):
            return (edge_metadata, [targets])
        if isinstance(targets, list) and targets and isinstance(targets[0], DataPoint):
            return (edge_metadata, targets)

    return None


def extract_relationships(field_value: Any) -> List[EdgeTargets]:
    """Every relationship declared by one field. Empty list means it is a property."""
    items = field_value if isinstance(field_value, list) else [field_value]
    return [pair for item in items if (pair := _as_edge_targets(item)) is not None]


def iter_fields(
    data_point: DataPoint,
    skip: Tuple[str, ...] = (),
) -> Iterator[Tuple[str, Any, List[EdgeTargets]]]:
    """Every field of a node, paired with the relationships it declares.

    An empty ``edge_targets`` means the field is a scalar property. Deciding that here
    is what keeps the walk and the container resolution from drifting on what counts as
    a relationship, and it is decided before any container is resolved: a field that
    points at an empty container is still a relationship, not a property.
    """
    for field_name, field_value in data_point:
        if field_name == "metadata" or field_name in skip:
            continue
        yield field_name, field_value, extract_relationships(field_value)


def iter_targets(edge_targets: List[EdgeTargets]) -> Iterator[DataPoint]:
    """Every target a field points at, dropping the edge metadata."""
    for _edge_metadata, targets in edge_targets:
        yield from targets
