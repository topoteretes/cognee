"""Transparent containers: nodes the author marked as structure, not content.

``metadata["transparent"]`` states that a DataPoint groups other DataPoints rather
than being one. The rule is single: wherever such a node appears, it is replaced by
its DataPoint children. Nothing here stores anything - resolution happens before the
walk decides what to write.
"""

from typing import Any, List

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.field_edges import split_field_edges
from cognee.shared.logging_utils import get_logger, warn_once

logger = get_logger()


def is_transparent(data_point: DataPoint) -> bool:
    """True when the author marked this node as a container rather than content."""
    return bool((getattr(data_point, "metadata", None) or {}).get("transparent"))


def _warn_once(kind: str, data_point: DataPoint, field_name: str, message: str) -> None:
    """Warn once per (kind, class, field); later occurrences log at DEBUG.

    The kind is part of the key: one field can deserve two different warnings.
    """
    qualname = type(data_point).__qualname__
    warn_once(logger, f"{kind}:{qualname}.{field_name}", message, qualname, field_name)


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
        logger.debug("Ignoring exception in _warn_dropped_field", exc_info=True)
        carries = True

    if carries:
        _warn_once(
            "dropped_field",
            data_point,
            field_name,
            "%s is marked transparent but carries data in %r; a transparent node is never "
            "stored, so that value is dropped. If the field is worth searching for, the class "
            "is not a container - remove metadata['transparent'].",
        )


def _warn_foreign_source_edge(data_point: DataPoint, field_name: str) -> None:
    """Warn once per (class, field) that a container is skipping a foreign-source edge."""
    _warn_once(
        "foreign_source_edge",
        data_point,
        field_name,
        "%s is marked transparent but %r declares a relationship whose source is not "
        "this container; a transparent node is never stored, so that edge is skipped "
        "rather than half-hoisted. Put the edge on a node that owns it, or remove "
        "metadata['transparent'].",
    )


def warn_transparent_source_edge(source_node: DataPoint, field_name: str) -> None:
    """Warn once per (class, field) that an edge names a transparent node as its source.

    Used by the storage walk: a transparent node is never stored, so an edge from it
    would dangle. The walk skips the edge; unwrapping the source instead would be
    ambiguous (which child becomes the subject?).
    """
    _warn_once(
        "transparent_source_edge",
        source_node,
        field_name,
        "%s is marked transparent but appears as the source of %r; a transparent node "
        "is never stored, so the edge would point from nothing and is skipped. Set the "
        "edge's source to a stored node, or remove metadata['transparent'].",
    )


def unwrap_transparent(data_point: DataPoint) -> List[DataPoint]:
    """Replace a transparent node with its DataPoint children, recursively.

    A non-transparent node resolves to ``[data_point]`` — the same object — so callers
    may apply this unconditionally. Order is field-declaration order, then list order
    within a field; each node appears once (deduplicated by node id, first reach wins),
    matching what storage would write.
    """
    if not is_transparent(data_point):
        return [data_point]

    resolved: List[DataPoint] = []
    _resolve_children(
        data_point, active=frozenset(), expanded=set(), seen_ids=set(), resolved=resolved
    )
    return resolved


def _resolve_children(
    container: DataPoint,
    active: frozenset,
    expanded: set,
    seen_ids: set,
    resolved: List[DataPoint],
) -> None:
    """Append ``container``'s resolved children to ``resolved``, each node once.

    Three guards, keyed differently on purpose:

    - ``active``: OBJECT identities on the current path — the true-cycle guard. Keyed
      on ``id()``, not node id: identity_fields let two distinct wrapper instances
      share a node id, and treating the second as a cycle would silently drop its
      children.
    - ``expanded``: object identities of containers already fully resolved. The same
      object reached through two fields (a diamond) yields the same children, so
      re-walking it is pure duplicated work — and exponential on stacked diamonds.
    - ``seen_ids``: NODE ids already appended. Two instances sharing a node id are one
      stored node, so the result carries the first.

    Every object involved is reachable from the tree being walked, so no ``id()`` can
    be recycled mid-walk.
    """
    if id(container) in active or id(container) in expanded:
        return
    active = active | {id(container)}

    field_edges, plain_fields = split_field_edges(container)

    for field_name, value in plain_fields:
        _warn_dropped_field(container, field_name, value)

    for field_name, edge in field_edges:
        if field_name == "belongs_to_set":
            continue
        if edge.source is not container:
            _warn_foreign_source_edge(container, field_name)
            continue
        target = edge.target
        if is_transparent(target):
            _resolve_children(target, active, expanded, seen_ids, resolved)
            continue
        node_id = str(target.id)
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            resolved.append(target)

    expanded.add(id(container))
