"""``stream_graph_data`` returns exactly ``get_graph_data``, in bounded pages, nodes first.

Runs against every in-tree adapter it can reach: Ladybug in-process and in
subprocess mode and Turso always (local files); Postgres and Neo4j when
``.env`` configures them (see conftest). Neptune has no live instance in CI;
its paging is unit-tested in test_stream_graph_data.py.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.engine import DataPoint
from cognee.tests.integration.infrastructure.graph.conftest import (
    _make_neo4j_adapter,
    _make_postgres_adapter,
)

pytestmark = pytest.mark.asyncio


class _StreamNode(DataPoint):
    """A node with nested JSON-valued properties, so the per-adapter merge is exercised."""

    name: str
    tags: list[str] = []
    details: dict = {}
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture(params=["ladybug", "ladybug_subprocess", "turso", "postgres", "neo4j"])
async def stream_adapter(request, tmp_path):
    if request.param == "ladybug":
        module = pytest.importorskip("cognee.infrastructure.databases.graph.ladybug.adapter")
        adapter = module.LadybugAdapter(str(tmp_path / "graph_db"))
    elif request.param == "ladybug_subprocess":
        module = pytest.importorskip("cognee.infrastructure.databases.graph.ladybug.adapter")
        adapter = module.LadybugAdapter.create_subprocess(db_path=str(tmp_path / "graph_db"))
    elif request.param == "turso":
        pytest.importorskip("turso", reason="pyturso not installed")
        from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

        adapter = TursoAdapter(database_path=str(tmp_path / "graph.db"))
        await adapter.initialize()
    elif request.param == "postgres":
        adapter = await _make_postgres_adapter()
    else:
        adapter = await _make_neo4j_adapter()

    try:
        yield request.param, adapter
    finally:
        await adapter.close()


async def _seed(adapter) -> None:
    """Seven nodes: a hub with four outgoing edges (three to one target, differing
    only by relationship name), a short cycle, and one node with no edges."""
    nodes = {
        key: _StreamNode(
            id=uuid4(),
            name=f"node-{key}",
            tags=[key, "stream"],
            details={"key": key, "nested": {"depth": len(key)}},
        )
        for key in ("hub", "b", "c", "d", "e", "f", "isolated")
    }
    await adapter.add_nodes(list(nodes.values()))

    def edge(source, target, relationship):
        source_id, target_id = str(nodes[source].id), str(nodes[target].id)
        return (
            source_id,
            target_id,
            relationship,
            {
                "source_node_id": source_id,
                "target_node_id": target_id,
                "relationship_name": relationship,
                "weight": 1,
            },
        )

    await adapter.add_edges(
        [
            edge("hub", "b", "calls"),
            edge("hub", "b", "imports"),
            edge("hub", "b", "mentions"),
            edge("hub", "c", "calls"),
            edge("b", "c", "calls"),
            edge("c", "d", "calls"),
            edge("d", "hub", "calls"),
            edge("e", "f", "contains"),
        ]
    )


def _canonical(items) -> list[str]:
    return sorted(json.dumps(item, sort_keys=True, default=str) for item in items)


async def _pages(adapter, batch_size):
    return [page async for page in adapter.stream_graph_data(batch_size=batch_size)]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 1000])
async def test_stream_matches_get_graph_data_in_bounded_pages(stream_adapter, batch_size):
    provider, adapter = stream_adapter
    await _seed(adapter)
    expected_nodes, expected_edges = await adapter.get_graph_data()
    assert len(expected_nodes) == 7
    assert len(expected_edges) == 8

    pages = await _pages(adapter, batch_size)

    streamed_nodes = [node for nodes, _ in pages for node in nodes]
    streamed_edges = [edge for _, edges in pages for edge in edges]
    assert _canonical(streamed_nodes) == _canonical(expected_nodes)
    assert _canonical(streamed_edges) == _canonical(expected_edges)

    # Each node and edge once.
    assert len({str(node_id) for node_id, _ in streamed_nodes}) == len(streamed_nodes)
    assert len({tuple(map(str, edge[:3])) for edge in streamed_edges}) == len(streamed_edges)

    # Nodes first, and a page is one or the other.
    phases = ["nodes" if nodes else "edges" for nodes, _ in pages]
    assert phases == sorted(phases, key=lambda phase: phase != "nodes")
    for nodes, edges in pages:
        assert bool(nodes) != bool(edges)

    # Bounded: nodes per page; edges by source node or by row, per adapter.
    for nodes, edges in pages:
        assert len(nodes) <= batch_size
        if provider in ("postgres", "turso"):
            assert len(edges) <= batch_size
        else:
            assert len({str(edge[0]) for edge in edges}) <= batch_size


async def test_stream_of_an_empty_graph_yields_nothing(stream_adapter):
    _provider, adapter = stream_adapter

    assert await adapter.get_graph_data() == ([], [])
    assert await _pages(adapter, 2) == []


@pytest.mark.parametrize("batch_size", [0, -1])
async def test_stream_rejects_a_non_positive_batch_size(stream_adapter, batch_size):
    _provider, adapter = stream_adapter
    await _seed(adapter)

    with pytest.raises(ValueError, match="batch_size"):
        await _pages(adapter, batch_size)
