"""Node-set scope as part of a data item's dedup identity.

``node_set`` is the scoping mechanism *inside* a dataset: a caller that stores
the same bytes under two different node sets (two end users of one cognee
user, two projects) means two different things that happen to share content.
Dedup therefore matches a stored row only when its node set is the same set of
tags as the one being ingested — the same content under another node set is a
new data item with its own graph documents, searchable and deletable in its own
scope. Before this, the second add was skipped as already completed and the
first row's tags were the only ones the content ever carried.

``Data.node_set`` is a JSON-encoded list of strings. These helpers are the only
place that reads it back for comparison, so the encoding is a detail of this
module.
"""

from __future__ import annotations

import json
from typing import Any

# ``identify*`` callers that do not know the node set keep the old content-only
# match. ``None`` means "no node set" there, so a distinct sentinel is needed.
UNSCOPED: Any = object()


def normalize_node_set(node_set: Any) -> list[str] | None:
    """Canonical form of a node set: sorted unique tag names, ``None`` when empty.

    Accepts the raw caller argument, a stored JSON string, or a list; anything
    unparseable counts as no node set.
    """
    if node_set is None or node_set is UNSCOPED:
        return None
    if isinstance(node_set, str):
        try:
            node_set = json.loads(node_set)
        except (TypeError, ValueError):
            return None
    if not isinstance(node_set, list | tuple | set | frozenset):
        return None
    tags = sorted({str(tag) for tag in node_set if tag is not None and str(tag) != ""})
    return tags or None


def encode_node_set(node_set: Any) -> str | None:
    """The value ``Data.node_set`` stores: the normalized list as JSON, or ``None``."""
    normalized = normalize_node_set(node_set)
    return json.dumps(normalized) if normalized is not None else None


def node_set_matches(stored: Any, requested: Any) -> bool:
    """Whether a stored row's node set is the same scope as the requested one.

    ``requested is UNSCOPED`` matches any row (content-only dedup, the
    behaviour of callers that do not know the node set).
    """
    if requested is UNSCOPED:
        return True
    return normalize_node_set(stored) == normalize_node_set(requested)
