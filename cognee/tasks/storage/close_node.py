"""Bi-temporal helper: close a graph node by setting its valid_to timestamp.

When a fact changes (e.g., "Alice works at X" → "Alice works at Y"), call
``close_node(old_node_id)`` to stamp the old node as no longer valid before
writing the replacement. This lets search queries filter out stale facts by
checking ``valid_to is None or valid_to > now``.
"""

from datetime import datetime, timezone
from uuid import UUID


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


async def close_node(node_id: UUID | str) -> None:
    """Set valid_to = now on *node_id* in the graph, marking it as superseded.

    No-op if the node does not exist or the graph backend does not support
    partial property updates (fails silently — the graph is not corrupted).
    """
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    try:
        await graph_engine.update_node(str(node_id), {"valid_to": _now_ms()})
    except Exception:
        pass


def is_valid(node, at_ms: int | None = None) -> bool:
    """Return True if *node* is valid at the given ms-epoch timestamp (default: now).

    Works for any object with a ``valid_to`` attribute (DataPoint instances,
    plain dicts, or graph node dicts from get_neighbours).
    """
    if at_ms is None:
        at_ms = _now_ms()
    valid_to = getattr(node, "valid_to", None) if not isinstance(node, dict) else node.get("valid_to")
    return valid_to is None or int(valid_to) > at_ms
