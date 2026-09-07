import pytest

pytest.importorskip("neo4j")

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import BASE_LABEL, Neo4jAdapter


def _make_adapter() -> Neo4jAdapter:
    return Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )


@pytest.mark.asyncio
async def test_get_edges_reports_true_edge_direction_for_incoming_edges():
    """get_edges walks the relationship undirected, so an edge stored as
    (other) -[:reports_to]-> (queried node) surfaces with n = the queried node.
    The contract is (source_id, target_id, ...) in TRUE edge direction (the
    consolidation pipeline renders 'neighbor —relationship→ node'), so the
    source must come from startNode(r), not from which endpoint the traversal
    happened to bind first."""
    adapter = _make_adapter()
    adapter.query = _FakeQuery(
        [
            {
                # stored edge: node-b -reports_to-> node-a; MATCH bound n=node-a
                "n": {"id": "node-a"},
                "m": {"id": "node-b"},
                "r": ({"id": "node-b"}, "reports_to", {"id": "node-a"}),
                "source_id": "node-b",
            },
            {
                # stored edge: node-a -works_on-> node-c; n=node-a again
                "n": {"id": "node-a"},
                "m": {"id": "node-c"},
                "r": ({"id": "node-a"}, "works_on", {"id": "node-c"}),
                "source_id": "node-a",
            },
        ]
    )

    edges = await adapter.get_edges("node-a")

    assert edges == [
        ("node-b", "node-a", {"relationship_name": "reports_to"}),
        ("node-a", "node-c", {"relationship_name": "works_on"}),
    ]


@pytest.mark.asyncio
async def test_get_edges_query_projects_start_node_id():
    """The query must project startNode(r).id: the serialized relationship
    triple carries the true edge endpoints, which the plain n/r/m projection
    cannot recover once the undirected match has bound n to the queried node."""
    adapter = _make_adapter()
    adapter.query = _FakeQuery([])

    await adapter.get_edges("node-a")

    query = adapter.query.await_args.args[0]
    assert "startNode(r).id AS source_id" in query
    assert "MATCH (n:" in query


class _FakeQuery:
    """Async callable standing in for adapter.query, recording its args."""

    def __init__(self, rows):
        self._rows = rows
        self.await_args = None

    async def __call__(self, query, params=None):
        import unittest.mock

        self.await_args = unittest.mock.call(query, params)
        return self._rows


def _note_base_label():
    """BASE_LABEL must stay part of the match so unlabeled nodes are skipped."""
    assert BASE_LABEL == "__Node__"
