"""Marker base class for graph nodes that must never be surfaced in retrieval output.

Internal nodes carry per-user or system state (e.g. user preferences) inside the
shared graph. They are filtered at the graph read chokepoints — projection for
search, triplet embedding, contradiction detection, natural-language search
(schema and result rows), and the provenance/schema-inventory views — by
checking the ``is_internal`` property on the raw node dictionary, never by
matching type names. The marker is a class default, so constructing the node is
what sets it; no writer has to remember to pass it.

Known exception: ``SearchType.CYPHER`` executes user-written Cypher verbatim
(gated by ``ALLOW_CYPHER_QUERY``) and applies no filter — it is a raw
escape hatch that can reach internal nodes.
"""

from cognee.infrastructure.engine.models.DataPoint import DataPoint

INTERNAL_PROPERTY = "is_internal"


class InternalDataPoint(DataPoint):
    """A node that must never be surfaced in retrieval output."""

    is_internal: bool = True


def is_internal_node(properties) -> bool:
    """True for a raw graph-node dict that must not reach a user. Tolerates None."""
    return bool((properties or {}).get(INTERNAL_PROPERTY))
