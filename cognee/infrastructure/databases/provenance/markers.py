"""Graph-native provenance markers.

A graph is "graph-native" when its graph-level metadata advertises the
graph-native delete mode. Marking happens lazily for empty graphs only, via
``set_graph_metadata``. On pre-Part-1 backends ``set_graph_metadata`` and
``get_graph_metadata`` raise ``UnsupportedProvenanceCapability``, so nothing is
ever marked and the whole graph-native path stays inert — old/unmarked graphs
keep using the relational ledger.
"""

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_NATIVE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
)


async def is_graph_native_graph(graph_engine) -> bool:
    """Return True when the graph metadata marks it as graph-native.

    Fail-safe: any exception (including ``UnsupportedProvenanceCapability`` from
    a backend that does not implement provenance) is treated as "not
    graph-native" so callers fall back to the relational-ledger delete path.
    """
    try:
        metadata = await graph_engine.get_graph_metadata()
    except Exception:
        return False

    return metadata.get(GRAPH_DELETE_MODE_KEY) == GRAPH_DELETE_MODE_GRAPH_NATIVE


async def ensure_graph_native_for_new_graph(graph_engine) -> bool:
    """Mark an empty graph as graph-native, returning whether it is graph-native.

    - Already graph-native -> True (idempotent, no re-mark).
    - Non-empty (existing/old graph) -> False; it stays on the ledger path.
    - Empty graph -> attempt to mark via ``set_graph_metadata``. On
      ``UnsupportedProvenanceCapability`` (pre-Part-1 backend) -> False. Only a
      successful mark returns True.
    """
    if await is_graph_native_graph(graph_engine):
        return True

    if not await graph_engine.is_empty():
        return False

    try:
        await graph_engine.set_graph_metadata(
            {
                GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
                GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_NATIVE,
            }
        )
    except UnsupportedProvenanceCapability:
        return False

    return True
