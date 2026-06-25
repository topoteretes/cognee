"""Graph-native provenance markers.

A graph is "graph-native" when its graph-level metadata advertises the
graph-native delete mode. Marking happens lazily for empty graphs only, via
``set_graph_metadata``. On backends that implement provenance (e.g. Ladybug +
LanceDB) a fresh empty graph gets marked and the graph-native path becomes live;
on backends without provenance support ``set_graph_metadata`` /
``get_graph_metadata`` raise ``UnsupportedProvenanceCapability``, so nothing is
marked and old/unmarked graphs keep using the relational ledger.
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

    A backend without provenance support (``UnsupportedProvenanceCapability``) is
    simply "not graph-native" -> fall back to the relational-ledger path. Any
    *other* error reading metadata is allowed to propagate: a destructive op must
    fail closed rather than silently route a real graph-native graph (which has
    no ledger rows) to the legacy path and skip graph cleanup.

    Both marker fields must match: ``delete_mode`` and ``provenance_version``.
    A partial or older marker is not treated as graph-native.
    """
    try:
        metadata = await graph_engine.get_graph_metadata()
    except UnsupportedProvenanceCapability:
        return False

    return (
        metadata.get(GRAPH_DELETE_MODE_KEY) == GRAPH_DELETE_MODE_GRAPH_NATIVE
        and metadata.get(GRAPH_PROVENANCE_VERSION_KEY) == GRAPH_PROVENANCE_VERSION
    )


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
