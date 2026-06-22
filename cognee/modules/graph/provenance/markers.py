"""Graph-native marker: the routing authority for delete/rollback (Part 2).

A graph is "graph-native" when it carries a marker node stamped with the
current ``provenance_version`` and ``delete_mode = "graph_native"``. Delete and
rollback consult this marker to decide whether to take the graph-native path
(provenance lives on the graph) or fall back to the relational nodes/edges
ledger (old graphs, retired in Part 3).

The marker is a single well-known node read/written through the existing
``GraphDBInterface`` node CRUD (``get_node`` / ``add_node`` / ``is_empty``), so
it needs none of the Part 1 storage primitives — only graphs newly created
under Part 2 get marked, and every pre-existing graph stays on the ledger path
until migrated.
"""

from uuid import NAMESPACE_OID, uuid5

from cognee.modules.graph.provenance.constants import (
    DELETE_MODE_GRAPH_NATIVE,
    DELETE_MODE_KEY,
    PROVENANCE_VERSION,
    PROVENANCE_VERSION_KEY,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("graph_native_marker")

# Well-known id of the marker node. Stable across graphs so a single lookup
# answers "is this graph graph-native?".
GRAPH_NATIVE_MARKER_NODE_ID = str(uuid5(NAMESPACE_OID, "cognee:graph-native-marker:v1"))

# Node ``type`` for the marker so it is trivially distinguishable and never
# mistaken for a content node by traversal/search.
GRAPH_NATIVE_MARKER_NODE_TYPE = "GraphNativeProvenanceMarker"


def _marker_properties() -> dict:
    return {
        "id": GRAPH_NATIVE_MARKER_NODE_ID,
        "type": GRAPH_NATIVE_MARKER_NODE_TYPE,
        PROVENANCE_VERSION_KEY: PROVENANCE_VERSION,
        DELETE_MODE_KEY: DELETE_MODE_GRAPH_NATIVE,
    }


async def is_graph_native_graph(graph_engine) -> bool:
    """Return True when ``graph_engine`` carries the graph-native marker node.

    This is the routing authority: a True result means delete/rollback may use
    graph provenance; False means the graph predates Part 2 and must stay on the
    relational ledger path. Any error reading the marker is treated as "not
    graph-native" so routing fails safe onto the ledger.
    """
    try:
        marker = await graph_engine.get_node(GRAPH_NATIVE_MARKER_NODE_ID)
    except Exception as error:  # noqa: BLE001 - fail safe onto the ledger path
        logger.debug("Graph-native marker lookup failed; treating as ledger graph: %s", error)
        return False

    if not marker:
        return False

    return marker.get(DELETE_MODE_KEY) == DELETE_MODE_GRAPH_NATIVE


async def ensure_graph_native_for_new_graph(graph_engine) -> bool:
    """Mark a brand-new (empty) graph as graph-native; leave existing graphs alone.

    Must be called BEFORE the first nodes are written, so an empty graph can be
    distinguished from a populated pre-Part-2 one. Returns True if the graph is
    graph-native after this call (already marked, or marked just now), False if
    it is a pre-existing unmarked graph that must stay on the ledger path.

    Idempotent: re-marking an already-marked graph is a no-op.

    A graph is only ever marked when the backend actually implements the
    graph-native provenance primitives (``supports_graph_native_provenance()``).
    Until Part 1 lands those on the real adapters, this returns False on the
    default stack, so add/cognify keep writing the relational ledger and nothing
    routes through the graph-native path — Part 2 stays inert in production while
    being fully exercised against the Part 0 fakes.
    """
    if await is_graph_native_graph(graph_engine):
        return True

    if not graph_engine.supports_graph_native_provenance():
        return False

    try:
        is_empty = await graph_engine.is_empty()
    except Exception as error:  # noqa: BLE001 - fail safe onto the ledger path
        logger.debug("Graph emptiness check failed; treating as ledger graph: %s", error)
        return False

    if not is_empty:
        # Pre-existing graph with content but no marker → an old graph. It keeps
        # the relational ledger path until Part 3 migrates it.
        return False

    await graph_engine.add_node(GRAPH_NATIVE_MARKER_NODE_ID, _marker_properties())
    logger.info("Marked new graph as graph-native (delete_mode=%s).", DELETE_MODE_GRAPH_NATIVE)
    return True


__all__ = [
    "GRAPH_NATIVE_MARKER_NODE_ID",
    "GRAPH_NATIVE_MARKER_NODE_TYPE",
    "is_graph_native_graph",
    "ensure_graph_native_for_new_graph",
]
