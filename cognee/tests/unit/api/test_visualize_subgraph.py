"""End-to-end tests: visualize_graph renders exactly the expected subgraph.

Only the I/O boundaries (graph engine, DB context, session events) are mocked;
the real seed-resolution, truncation, and HTML renderer all run, and we assert
on the node/edge payload actually embedded in the produced HTML.
"""

import json
import re
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# `cognee.api.v1.__init__` rebinds the name `visualize` on the v1 package to the
# visualize_graph *function*, which shadows the submodule under attribute access.
# So both a dotted-string patch target ("cognee.api.v1.visualize.visualize.<name>")
# and `import cognee.api.v1.visualize.visualize` traverse that function and fail
# once the suite has imported the v1 package. A full-name `from`-import resolves
# the submodule by its module path (bypassing the shadowed attribute); we then
# grab the module object from sys.modules and patch on it (patch.object), which
# is import-order independent.
from cognee.api.v1.visualize.visualize import visualize_graph as _visualize_graph  # noqa: F401

visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]


def _chain_graph(node_count: int = 20):
    nodes = [(str(i), {"type": "Entity", "name": f"N{i}"}) for i in range(node_count)]
    edges = [(str(i), str(i + 1), "related_to", {}) for i in range(node_count - 1)]
    return nodes, edges


def _rendered_ids_and_edges(html: str):
    nodes_match = re.search(r"var nodes\s*=\s*(\[.*?\]);", html, re.DOTALL)
    links_match = re.search(r"var links\s*=\s*(\[.*?\]);", html, re.DOTALL)
    assert nodes_match and links_match, "nodes/links payload missing from HTML"
    node_ids = {str(n["id"]) for n in json.loads(nodes_match.group(1))}
    edge_pairs = {
        (str(link["source"]), str(link["target"])) for link in json.loads(links_match.group(1))
    }
    return node_ids, edge_pairs


@asynccontextmanager
async def _noop_db_context(*_args, **_kwargs):
    yield


def _patches(engine):
    return (
        patch.object(visualize_module, "get_graph_engine", AsyncMock(return_value=engine)),
        patch.object(visualize_module, "set_database_global_context_variables", _noop_db_context),
        patch(
            "cognee.modules.visualization.session_events.collect_session_events",
            AsyncMock(return_value=[]),
        ),
    )


async def _visualize(engine, tmp_path, **kwargs):
    ctx_a, ctx_b, ctx_c = _patches(engine)
    with ctx_a, ctx_b, ctx_c:
        return await visualize_module.visualize_graph(
            str(tmp_path / "out.html"), dataset=None, user=MagicMock(), **kwargs
        )


@pytest.mark.asyncio
async def test_query_default_renders_expected_subgraph(tmp_path):
    full_graph = _chain_graph(20)
    # Neighborhood around node "10": nodes 9-11, edges 9-10 and 10-11.
    subgraph = ([full_graph[0][i] for i in (9, 10, 11)], [full_graph[1][9], full_graph[1][10]])
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)

    with patch(
        "cognee.modules.visualization.subgraph_data.resolve_seeds_from_query",
        AsyncMock(return_value=["10"]),
    ):
        html = await _visualize(engine, tmp_path, query="what is N10?")

    node_ids, edge_pairs = _rendered_ids_and_edges(html)
    assert node_ids == {"9", "10", "11"}
    assert edge_pairs == {("9", "10"), ("10", "11")}
    engine.get_neighborhood.assert_awaited_once()
    engine.get_graph_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_true_renders_entire_graph(tmp_path):
    full_graph = _chain_graph(20)
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock()

    html = await _visualize(engine, tmp_path, full=True)

    node_ids, _ = _rendered_ids_and_edges(html)
    assert node_ids == {str(i) for i in range(20)}
    engine.get_graph_data.assert_awaited_once()
    engine.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_seed_ids_render_subgraph(tmp_path):
    full_graph = _chain_graph(20)
    subgraph = ([full_graph[0][i] for i in (9, 10, 11)], [full_graph[1][9], full_graph[1][10]])
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=subgraph)
    engine.get_graph_data = AsyncMock()

    html = await _visualize(engine, tmp_path, seed_node_ids=["10"], neighborhood_depth=1)

    node_ids, _ = _rendered_ids_and_edges(html)
    assert node_ids == {"9", "10", "11"}
    engine.get_neighborhood.assert_awaited_once_with(node_ids=["10"], depth=1)
    engine.get_graph_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_result_provenance_seeds_subgraph(tmp_path):
    full_graph = _chain_graph(20)
    subgraph = ([full_graph[0][i] for i in (9, 10, 11)], [full_graph[1][9], full_graph[1][10]])
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=subgraph)

    recall_result = [SimpleNamespace(used_graph_element_ids={"node_ids": ["10"]})]
    html = await _visualize(engine, tmp_path, recall_result=recall_result, neighborhood_depth=1)

    node_ids, _ = _rendered_ids_and_edges(html)
    assert node_ids == {"9", "10", "11"}
    engine.get_neighborhood.assert_awaited_once_with(node_ids=["10"], depth=1)


@pytest.mark.asyncio
async def test_no_seed_falls_back_to_degree(tmp_path):
    full_graph = _chain_graph(20)
    subgraph = ([full_graph[0][i] for i in (9, 10, 11)], [full_graph[1][9], full_graph[1][10]])
    engine = MagicMock()
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock(return_value=subgraph)

    html = await _visualize(engine, tmp_path)

    node_ids, _ = _rendered_ids_and_edges(html)
    assert node_ids == {"9", "10", "11"}
    engine.get_graph_data.assert_awaited_once()  # degree fallback loads the graph
    engine.get_neighborhood.assert_awaited_once()
