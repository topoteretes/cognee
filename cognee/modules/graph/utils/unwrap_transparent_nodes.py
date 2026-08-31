"""Transparent containers: nodes the author marked as structure, not content.

``metadata["transparent"]`` states that a DataPoint groups other DataPoints rather
than being one. The rule is single: wherever such a node appears, it is replaced by
its DataPoint children. Nothing here stores anything - resolution happens before the
walk decides what to write.
"""

from typing import Any, List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger()


# (class qualname, field name) pairs already warned about. Bounded by the number
# of declared fields on transparent classes, so it cannot grow without limit.
_WARNED_DROPPED_FIELDS: set = set()


def is_transparent(data_point: DataPoint) -> bool:
    """True when the author marked this node as a container rather than content."""
    return bool((getattr(data_point, "metadata", None) or {}).get("transparent"))


def _warn_dropped_field(data_point: DataPoint, field_name: str, value: Any) -> None:
    """Warn once per (class, field) that a transparent node is dropping real data.

    Silent for a field inherited from ``DataPoint`` (every node carries those) and for
    an empty value — an optional relationship left ``None`` lost nothing. Array-likes
    raise on truth testing; treat those as carrying.
    """
    if field_name in DataPoint.model_fields:
        return

    try:
        carries = bool(value)
    except Exception:
        carries = True

    key = (type(data_point).__qualname__, field_name)
    if not carries or key in _WARNED_DROPPED_FIELDS:
        return
    _WARNED_DROPPED_FIELDS.add(key)

    logger.warning(
        "%s is marked transparent but carries data in %r; a transparent node is never "
        "stored, so that value is dropped. If the field is worth searching for, the class "
        "is not a container - remove metadata['transparent'].",
        type(data_point).__qualname__,
        field_name,
    )


def _warn_foreign_source_edge(data_point: DataPoint, field_name: str) -> None:
    """Warn once per (class, field) that a container is skipping a foreign-source edge."""
    key = (type(data_point).__qualname__, field_name)
    if key in _WARNED_DROPPED_FIELDS:
        return
    _WARNED_DROPPED_FIELDS.add(key)

    logger.warning(
        "%s is marked transparent but %r declares a relationship whose source is not "
        "this container; a transparent node is never stored, so that edge is skipped "
        "rather than half-hoisted. Put the edge on a node that owns it, or remove "
        "metadata['transparent'].",
        type(data_point).__qualname__,
        field_name,
    )


def unwrap_transparent(
    data_point: DataPoint,
    _active: Optional[frozenset] = None,
) -> List[DataPoint]:
    """Replace a transparent node with its DataPoint children, recursively.

    A non-transparent node resolves to ``[data_point]`` — the same object — so callers
    may apply this unconditionally. Order is field-declaration order, then list order
    within a field.
    """
    if not is_transparent(data_point):
        return [data_point]

    # Cycle guard keyed on OBJECT IDENTITY, not node id: identity_fields let two
    # distinct wrapper instances share a node id, and treating the second as a cycle
    # would silently drop its children. Every object on the active path is held alive
    # by a caller frame, so id() cannot be reused mid-walk.
    if _active and id(data_point) in _active:
        return []
    _active = (_active or frozenset()) | {id(data_point)}

    properties, _excluded, declared = data_point.graph_fields()
    for field_name, value in properties.items():
        _warn_dropped_field(data_point, field_name, value)

    resolved: List[DataPoint] = []
    for field_name, edge in declared:
        if field_name == "belongs_to_set":
            continue
        if edge.source is not data_point:
            _warn_foreign_source_edge(data_point, field_name)
            continue
        resolved.extend(unwrap_transparent(edge.target, _active))
    return resolved
