"""Transparent containers: nodes the author marked as structure, not content.

``metadata["transparent"]`` states that a DataPoint groups other DataPoints rather
than being one. The rule is single: wherever such a node appears, it is replaced by
its DataPoint children. Nothing here stores anything - resolution happens before the
walk decides what to write.
"""

from typing import Any, List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.extract_field_relationships import (
    EdgeTargets,
    iter_targets,
    iter_fields,
)
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

    resolved: List[DataPoint] = []
    seen = set()

    # ``belongs_to_set`` is a relationship, but a wrapper's NodeSets are not its
    # children - inheriting them would mint ``Parent --field--> NodeSet`` edges.
    for field_name, field_value, edge_targets in iter_fields(data_point, ("belongs_to_set",)):
        if not edge_targets:
            _warn_dropped_field(data_point, field_name, field_value)
            continue

        for target in iter_targets(edge_targets):
            for node in unwrap_transparent(target, _active):
                if str(node.id) not in seen:
                    seen.add(str(node.id))
                    resolved.append(node)

    return resolved


def unwrap_transparent_targets(edge_targets: List[EdgeTargets]) -> List[EdgeTargets]:
    """Replace every transparent target with its children, in place of the container.

    Returns the very same list object when nothing in the field is transparent, so a
    graph using no containers behaves exactly as it does without this function.

    A declaration whose targets all resolve away is kept as ``(edge, [])`` rather than
    dropped, so the result stays one-for-one with what was declared.
    """
    if not any(is_transparent(target) for target in iter_targets(edge_targets)):
        return edge_targets

    return [
        (edge_metadata, [node for t in targets for node in unwrap_transparent(t)])
        for edge_metadata, targets in edge_targets
    ]
