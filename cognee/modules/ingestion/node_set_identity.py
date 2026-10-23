"""Node-set scope as part of a data item's dedup identity.

``node_set`` is the scoping mechanism *inside* a dataset: a caller that stores
the same bytes under two different node sets (two end users of one cognee
user, two projects) means two different things that happen to share content.
Dedup therefore matches a stored row only when its node set is the same set of
tags as the one being ingested — the same content under another node set is a
new data item with its own graph documents, searchable and deletable in its own
scope. Before this, the second add was skipped as already completed and the
first row's tags were the only ones the content ever carried.

A row's scope is the node set ingestion stores on it: the call-level
``node_set`` joined with the item's own (``union_node_sets``). Two names are
one tag when they resolve to one ``NodeSet`` id — the same rule the union and
the graph use — so ``"Notion:A B"`` and ``"notion:a_b"`` are the same scope.

``Data.node_set`` is a JSON-encoded list of names. These helpers are the only
place that reads it back for comparison, so the encoding is a detail of this
module.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from cognee.modules.engine.models.node_set import NodeSet

# ``identify*`` callers that do not know the node set keep the old content-only
# match. ``None`` means "no node set" there, so a distinct sentinel is needed.
UNSCOPED: Any = object()


def decode_node_set(node_set: Any) -> list[str] | None:
    """The names a node set holds, in stored order; ``None`` when there are none.

    Accepts the raw caller argument (a list of names), the stored JSON string,
    or ``None``. Anything unparseable counts as no node set.
    """
    if node_set is None or node_set is UNSCOPED:
        return None
    if isinstance(node_set, str):
        try:
            node_set = json.loads(node_set)
        except (TypeError, ValueError):
            return None
    if not isinstance(node_set, list | tuple):
        return None
    names = [name for name in node_set if isinstance(name, str) and name.strip()]
    return names or None


def node_set_key(node_set: Any) -> frozenset[UUID] | None:
    """The identity of a node-set scope: the ``NodeSet`` ids its names resolve to.

    ``None`` is the scope of rows stored without a node set. Order and
    duplicate spellings do not matter, so this is what two scopes are compared
    (and grouped) by.
    """
    names = decode_node_set(node_set)
    if names is None:
        return None
    return frozenset(NodeSet.id_for(name) for name in names)


def node_set_matches(stored: Any, requested: Any) -> bool:
    """Whether a stored row's node set is the same scope as the requested one.

    ``requested is UNSCOPED`` matches any row (content-only dedup, the
    behaviour of callers that do not know the node set).
    """
    if requested is UNSCOPED:
        return True
    return node_set_key(stored) == node_set_key(requested)


def union_node_sets(
    call_node_set: list[str] | None, item_node_set: list[str] | None
) -> list[str] | None:
    """Combine the call-level node_set with a DataItem's own node_set.

    Call-first, order-preserving, deduplicated on the NodeSet id key, so two
    spellings that map to one graph node keep only the first. With no
    item-level node_set this returns ``call_node_set`` untouched (not even
    deduped), exactly as before this field existed. Both sides are already
    validated by ``ingest_data`` as lists of names.
    """
    if not item_node_set:
        return call_node_set

    seen: set[UUID] = set()
    combined: list[str] = []
    for name in (call_node_set or []) + item_node_set:
        key = NodeSet.id_for(name)
        if key not in seen:
            seen.add(key)
            combined.append(name)
    return combined or None
