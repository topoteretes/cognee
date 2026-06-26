"""Graph provenance markers.

A graph "stores its provenance in the graph" when its graph-level metadata
advertises the graph-provenance delete mode. Marking happens lazily for empty
graphs only, via ``set_graph_metadata``. On backends that implement provenance
(e.g. Ladybug + LanceDB) a fresh empty graph gets marked and reads/deletes go
through the graph instead of the relational ledger; on backends without
provenance support ``set_graph_metadata`` / ``get_graph_metadata`` raise
``UnsupportedProvenanceCapability``, so nothing is marked and old/unmarked
graphs keep using the relational ledger.
"""

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
)


async def stores_provenance_in_graph(graph_engine) -> bool:
    """Return True when the graph's metadata marks it as carrying its own provenance.

    A backend without provenance support (``UnsupportedProvenanceCapability``)
    simply does not store provenance in the graph -> fall back to the
    relational-ledger path. Any *other* error reading metadata is allowed to
    propagate: a destructive op must fail closed rather than silently route a
    graph that holds its provenance in-graph (and so has no ledger rows) to the
    legacy path and skip graph cleanup.

    Both marker fields must match: ``delete_mode`` and ``provenance_version``.
    A partial or older marker does not count.
    """
    try:
        metadata = await graph_engine.get_graph_metadata()
    except UnsupportedProvenanceCapability:
        return False

    return (
        metadata.get(GRAPH_DELETE_MODE_KEY) == GRAPH_DELETE_MODE_GRAPH_PROVENANCE
        and metadata.get(GRAPH_PROVENANCE_VERSION_KEY) == GRAPH_PROVENANCE_VERSION
    )


async def mark_graph_provenance_if_empty(graph_engine) -> bool:
    """Mark an empty graph to store provenance in-graph; return whether it does.

    - Already marked -> True (idempotent, no re-mark).
    - Non-empty (existing/old graph) -> False; it stays on the ledger path.
    - Empty graph -> attempt to mark via ``set_graph_metadata``. On
      ``UnsupportedProvenanceCapability`` (pre-Part-1 backend) -> False. Only a
      successful mark returns True.
    """
    if await stores_provenance_in_graph(graph_engine):
        return True

    if not await graph_engine.is_empty():
        return False

    try:
        await graph_engine.set_graph_metadata(
            {
                GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
                GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
            }
        )
    except UnsupportedProvenanceCapability:
        return False

    return True
