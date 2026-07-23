"""Bi-temporal helper: close a graph node by stamping its ``valid_to`` timestamp.

When a fact changes (e.g. "Alice works at X" -> "Alice works at Y"), call
``close_node(old_node_id)`` to mark the old node as no longer valid — instead of
deleting it — and then write the replacement. Consumers filter stale facts with
``is_valid(node)``, which stays true while ``valid_to`` is ``None`` (never closed) or
still lies in the future.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


async def close_node(node_id: UUID | str, at_ms: int | None = None) -> bool:
    """Stamp ``valid_to`` on *node_id*, marking it superseded as of *at_ms* (default now).

    Returns ``True`` if the node existed and was updated, ``False`` otherwise (e.g. the
    node is not in the graph). Backends that do not implement partial node updates yet
    log a warning and return ``False`` rather than failing silently.

    Last-write-wins: closing an already-closed node overwrites ``valid_to`` (a later or
    an earlier timestamp), so guard with ``is_valid`` first if you need it to be
    idempotent.
    """
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    valid_to = at_ms if at_ms is not None else _now_ms()
    try:
        return await graph_engine.update_node(str(node_id), {"valid_to": valid_to})
    except NotImplementedError:
        logger.warning(
            "close_node: the configured graph backend does not support partial node "
            "updates; valid_to was not set for node %s",
            node_id,
        )
        return False


def is_valid(node, at_ms: int | None = None) -> bool:
    """Return ``True`` if *node* is still valid at *at_ms* (ms epoch, default now).

    Works for any object carrying a ``valid_to`` attribute (``DataPoint`` instances) or
    a ``valid_to`` key (plain graph-node dicts). A node is valid when ``valid_to`` is
    ``None`` (never closed) or lies strictly in the future.
    """
    if at_ms is None:
        at_ms = _now_ms()
    valid_to = node.get("valid_to") if isinstance(node, dict) else getattr(node, "valid_to", None)
    return valid_to is None or int(valid_to) > at_ms
