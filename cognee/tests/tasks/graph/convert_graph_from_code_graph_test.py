import asyncio

import pytest

from cognee.shared.CodeGraphEntities import Repository
from cognee.tasks.graph.convert_graph_from_code_graph import (
    convert_graph_from_code_graph,
)
from cognee.tests.tasks.graph.code_graph_test_data_generation import (
    code_graph_test_data_generation,
)


def test_convert_graph_from_code_graph():
    repo = Repository(path="test/repo/path")
    nodes, edges = code_graph_test_data_generation()
    repo_out, nodes_out, edges_out = asyncio.run(
        convert_graph_from_code_graph(repo, nodes, edges)
    )

    assert repo == repo_out, f"{repo = } != {repo_out = }"

    for node_in, node_out in zip(nodes, nodes_out):
        assert node_in == node_out, f"{node_in = } != {node_out = }"

    for edge_in, edge_out in zip(edges, edges_out):
        assert edge_in == edge_out, f"{edge_in = } != {edge_out = }"
