import json
import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.visualization.subgraph_data import fetch_visualization_graph_data


def _chain_graph(node_count: int = 20):
    nodes = [(str(i), {"type": "Entity", "name": f"N{i}"}) for i in range(node_count)]
    edges = [(str(i), str(i + 1), "related_to", {}) for i in range(node_count - 1)]
    return nodes, edges


def _story_node_count(html: str) -> int:
    match = re.search(r"var nodes\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if not match:
        return 0
    return len(json.loads(match.group(1)))


@asynccontextmanager
async def _noop_db_context(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_visualize_graph_default_uses_subgraph(tmp_path):
    full_graph = _chain_graph(20)
    subgraph = (full_graph[0][8:12], full_graph[1][8:11])

    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)
    engine.query = AsyncMock(return_value=[])

    destination = tmp_path / "subgraph.html"

    with (
        patch("cognee.api.v1.visualize.visualize.get_graph_engine", AsyncMock(return_value=engine)),
        patch(
            "cognee.api.v1.visualize.visualize.set_database_global_context_variables",
            _noop_db_context,
        ),
        patch(
            "cognee.modules.visualization.session_events.collect_session_events",
            AsyncMock(return_value=[]),
        ),
        patch(
            "cognee.modules.visualization.session_events.get_latest_session_seed_node_ids",
            AsyncMock(return_value=["10"]),
        ),
    ):
        from cognee.api.v1.visualize.visualize import visualize_graph

        html = await visualize_graph(
            str(destination),
            dataset=None,
            user=MagicMock(),
        )

    assert "<html" in html
    assert "var nodes =" in html
    assert engine.get_neighborhood.await_count == 1
    assert engine.get_graph_data.await_count == 0
    assert _story_node_count(html) < _story_node_count(
        await _render_full_graph_html(full_graph, tmp_path)
    )


@pytest.mark.asyncio
async def test_visualize_graph_full_true_renders_entire_graph(tmp_path):
    full_graph = _chain_graph(20)
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock()

    destination = tmp_path / "full.html"

    with (
        patch("cognee.api.v1.visualize.visualize.get_graph_engine", AsyncMock(return_value=engine)),
        patch(
            "cognee.api.v1.visualize.visualize.set_database_global_context_variables",
            _noop_db_context,
        ),
        patch(
            "cognee.modules.visualization.session_events.collect_session_events",
            AsyncMock(return_value=[]),
        ),
    ):
        from cognee.api.v1.visualize.visualize import visualize_graph

        html = await visualize_graph(
            str(destination),
            dataset=None,
            user=MagicMock(),
            full=True,
        )

    assert engine.get_graph_data.await_count == 1
    assert engine.get_neighborhood.await_count == 0
    assert _story_node_count(html) == 20


@pytest.mark.asyncio
async def test_fetch_visualization_graph_data_seed_node_ids_direct():
    engine = MagicMock()
    subgraph = _chain_graph(4)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)

    graph_data, meta = await fetch_visualization_graph_data(
        engine,
        seed_node_ids=["1"],
        neighborhood_depth=1,
        max_nodes=500,
    )
    assert len(graph_data[0]) == 4
    assert meta.seed_ids == ["1"]


async def _render_full_graph_html(full_graph, tmp_path):
    from cognee.modules.visualization.cognee_network_visualization import (
        cognee_network_visualization,
    )

    return await cognee_network_visualization(full_graph, str(tmp_path / "reference.html"))
