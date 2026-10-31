"""A graph with nodes and no edges is a legitimate graph, not an empty one.

The Ladybug adapter used to fabricate one ``(id, id, "SELF")`` self-loop per
node whenever a graph had no edges. Nothing asked it to: it existed only so
``CogneeGraph``'s "empty graph" guards -- which tested the EDGE list -- would
not raise on a graph that merely had no relationships yet. The fakes were
never filtered out of the retrieval path, so an edgeless graph answered
queries from ``X --SELF--> X`` triplets.

These tests pin both halves of the fix: the adapter reports edges honestly,
and the projection accepts a graph that has none.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.exceptions import EntityNotFoundError

NODES = [
    ("node-a", {"name": "Alice", "type": "Entity"}),
    ("node-b", {"name": "Berlin", "type": "Entity"}),
]

PROJECTION = {
    "node_properties_to_project": ["name", "type"],
    "edge_properties_to_project": ["relationship_type"],
}


def _adapter(nodes, edges) -> MagicMock:
    """An adapter stub with no get_id_filtered_graph_data, so the full-graph path runs."""
    adapter = MagicMock(spec=["get_graph_data"])
    adapter.get_graph_data = AsyncMock(return_value=(nodes, edges))
    return adapter


@pytest.mark.asyncio
async def test_graph_with_nodes_and_no_edges_projects_without_raising():
    """The case the SELF placeholders existed to hide."""
    graph = CogneeGraph()

    await graph.project_graph_from_db(_adapter(NODES, []), **PROJECTION)

    assert len(graph.nodes) == len(NODES)
    assert graph.edges == []


@pytest.mark.asyncio
async def test_graph_with_no_nodes_still_raises():
    """Emptiness is decided by nodes, so a genuinely empty graph must still raise."""
    graph = CogneeGraph()

    with pytest.raises(EntityNotFoundError):
        await graph.project_graph_from_db(_adapter([], []), **PROJECTION)


@pytest.mark.asyncio
async def test_real_edges_are_still_projected():
    """The permissive guard must not stop real relationships from being projected."""
    graph = CogneeGraph()
    edges = [("node-a", "node-b", "lives_in", {"relationship_type": "lives_in"})]

    await graph.project_graph_from_db(_adapter(NODES, edges), **PROJECTION)

    assert len(graph.edges) == 1
    assert graph.edges[0].attributes["relationship_type"] == "lives_in"


def test_ladybug_adapter_no_longer_fabricates_self_edges():
    """The adapter must not invent edges; source-level pin so the hack cannot return."""
    from pathlib import Path

    import cognee.infrastructure.databases.graph.ladybug.adapter as ladybug_adapter

    source = Path(ladybug_adapter.__file__).read_text()

    assert '"SELF"' not in source, (
        "Ladybug must not fabricate (id, id, 'SELF') edges: they reach retrieval "
        "unfiltered and are indistinguishable from real relationships."
    )
