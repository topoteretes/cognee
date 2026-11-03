"""``stream_graph_data``: the interface default, and Neptune's paging without a live Neptune.

Adapters with a store to run against are covered by
cognee/tests/integration/infrastructure/graph/test_stream_graph_data_contract.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph import graph_db_interface
from cognee.infrastructure.databases.graph.graph_db_interface import (
    DEFAULT_GRAPH_STREAM_BATCH_SIZE,
    GraphDBInterface,
)
from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB

NODES = [(f"n{index}", {"name": f"n{index}", "type": "T"}) for index in range(5)]
EDGES = [("n0", "n1", "calls", {}), ("n0", "n1", "imports", {}), ("n1", "n2", "calls", {})]


class _GraphDataOnlyAdapter:
    """An adapter with no native stream: exercises the inherited default.

    Not a GraphDBInterface subclass, like the double in test_top_degree_node_ids.py.
    """

    def __init__(self, nodes, edges):
        self.get_graph_data = AsyncMock(return_value=(nodes, edges))

    def stream_graph_data(self, *args, **kwargs):
        return GraphDBInterface.stream_graph_data(self, *args, **kwargs)


async def _pages(adapter, **kwargs):
    return [page async for page in adapter.stream_graph_data(**kwargs)]


def test_the_default_is_inherited_not_abstract():
    assert "stream_graph_data" not in GraphDBInterface.__abstractmethods__
    assert DEFAULT_GRAPH_STREAM_BATCH_SIZE == 2000


@pytest.mark.asyncio
async def test_default_slices_get_graph_data_nodes_first():
    adapter = _GraphDataOnlyAdapter(NODES, EDGES)

    pages = await _pages(adapter, batch_size=2)

    assert pages == [
        (NODES[0:2], []),
        (NODES[2:4], []),
        (NODES[4:5], []),
        ([], EDGES[0:2]),
        ([], EDGES[2:3]),
    ]
    adapter.get_graph_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_of_an_empty_graph_yields_nothing():
    assert await _pages(_GraphDataOnlyAdapter([], []), batch_size=2) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -3])
async def test_default_rejects_a_non_positive_batch_size_before_reading(batch_size):
    adapter = _GraphDataOnlyAdapter(NODES, EDGES)

    with pytest.raises(ValueError, match="batch_size"):
        await _pages(adapter, batch_size=batch_size)
    adapter.get_graph_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_warns_once_per_adapter_type(monkeypatch):
    warnings = []
    monkeypatch.setattr(graph_db_interface, "_warned_graph_stream_fallbacks", set())
    monkeypatch.setattr(
        graph_db_interface.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )

    await _pages(_GraphDataOnlyAdapter(NODES, EDGES))
    await _pages(_GraphDataOnlyAdapter(NODES, EDGES))

    assert len(warnings) == 1
    assert "_GraphDataOnlyAdapter" in warnings[0]


# --- Neptune -----------------------------------------------------------------


class _FakeNeptune:
    """Answers NeptuneGraphDB's page queries from an in-memory graph.

    Checks the query shapes the adapter builds (keyset on ``id(n)``, ordered,
    limited) and applies them, so the paging loop runs for real.
    """

    def __init__(self, nodes, edges):
        self.nodes = sorted(nodes)
        self.edges = edges
        self.queries = []

    async def __call__(self, query, params):
        self.queries.append((query, dict(params)))
        if query.lstrip().startswith("UNWIND $source_ids"):
            assert "WHERE id(source) = source_node_id" in query
            sources = set(params["source_ids"])
            return [
                {
                    "source_id": source,
                    "target_id": target,
                    "relationship_name": name,
                    "properties": props,
                }
                for source, target, name, props in self.edges
                if source in sources
            ]
        assert "MATCH (n:COGNEE_NODE)" in query
        assert "ORDER BY id(n) LIMIT $batch_size" in query
        rows = self.nodes
        if "WHERE id(n) > $last_id" in query:
            rows = [row for row in rows if row[0] > params["last_id"]]
        rows = rows[: params["batch_size"]]
        if "properties(n)" in query:
            return [{"node_id": node_id, "properties": props} for node_id, props in rows]
        return [{"node_id": node_id} for node_id, _props in rows]


def _neptune(nodes, edges):
    fake = _FakeNeptune(nodes, edges)
    stub = SimpleNamespace(_GRAPH_NODE_LABEL="COGNEE_NODE", query=fake)
    for name in (
        "_stream_node_page_query",
        "_stream_source_page_query",
        "_stream_edge_lookup_query",
        "_stream_query",
    ):
        setattr(stub, name, getattr(NeptuneGraphDB, name).__get__(stub))
    return stub, fake


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 2, 10])
async def test_neptune_pages_nodes_then_source_edges(batch_size):
    stub, _fake = _neptune(NODES, EDGES)

    pages = [page async for page in NeptuneGraphDB.stream_graph_data(stub, batch_size)]

    nodes = [node for page_nodes, _ in pages for node in page_nodes]
    edges = [edge for _, page_edges in pages for edge in page_edges]
    assert nodes == NODES
    assert sorted(edges) == sorted(EDGES)
    phases = ["nodes" if page_nodes else "edges" for page_nodes, _ in pages]
    assert phases == sorted(phases, key=lambda phase: phase != "nodes")
    for page_nodes, page_edges in pages:
        assert len(page_nodes) <= batch_size
        assert len({edge[0] for edge in page_edges}) <= batch_size


@pytest.mark.asyncio
async def test_neptune_first_page_has_no_keyset_and_later_pages_do():
    stub, fake = _neptune(NODES, [])

    [page async for page in NeptuneGraphDB.stream_graph_data(stub, 2)]

    node_queries = [query for query, _ in fake.queries if "properties(n)" in query]
    assert "WHERE" not in node_queries[0]
    assert all("WHERE id(n) > $last_id" in query for query in node_queries[1:])


@pytest.mark.asyncio
async def test_neptune_empty_graph_reads_no_edges():
    stub, fake = _neptune([], EDGES)

    assert [page async for page in NeptuneGraphDB.stream_graph_data(stub, 2)] == []
    assert len(fake.queries) == 1


@pytest.mark.asyncio
async def test_neptune_wraps_a_failed_page_like_get_graph_data():
    stub = SimpleNamespace(
        _GRAPH_NODE_LABEL="COGNEE_NODE", query=AsyncMock(side_effect=Exception("boom"))
    )
    for name in ("_stream_node_page_query", "_stream_query"):
        setattr(stub, name, getattr(NeptuneGraphDB, name).__get__(stub))

    with pytest.raises(RuntimeError, match="Failed to get graph data"):
        [page async for page in NeptuneGraphDB.stream_graph_data(stub, 2)]
