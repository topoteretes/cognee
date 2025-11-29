import pytest
import asyncio
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

# 导入 CogneeGraph 相关的元素
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge

from cognee.tasks.memify.propagate_importance_weights import propagate_importance_weights

class MockNode(Node):

    def __init__(self, node_id: str, importance_weight: Optional[float] = None):
        super().__init__(node_id, dimension=1)
        if importance_weight is not None:
            self.attributes["importance_weight"] = importance_weight

@pytest.fixture
def mock_memory_fragment() -> CogneeGraph:

    node_a = MockNode("N_A", importance_weight=1.0)
    node_b = MockNode("N_B", importance_weight=0.2)

    node_x = MockNode("N_X")
    node_y = MockNode("N_Y")
    node_z = MockNode("N_Z")

    graph = CogneeGraph(directed=False)

    for node in [node_a, node_b, node_x, node_y, node_z]:
        graph.add_node(node)

    edge_ax = Edge(node_a, node_x, directed=False)
    graph.add_edge(edge_ax)

    edge_ay = Edge(node_a, node_y, directed=False)
    graph.add_edge(edge_ay)

    edge_by = Edge(node_b, node_y, directed=False)
    graph.add_edge(edge_by)

    return graph

@pytest.mark.asyncio
async def test_weight_propagation_and_fusion(mock_memory_fragment: CogneeGraph):
    data = [mock_memory_fragment]
    updated_data = await propagate_importance_weights(data)

    updated_graph: CogneeGraph = updated_data[0]

    nodes = {node.id: node for node in updated_graph.nodes.values()}

    n_a = nodes['N_A']
    assert abs(n_a.attributes["importance_weight"] - 1.0) < 1e-4

    n_b = nodes['N_B']
    assert abs(n_b.attributes["importance_weight"] - 0.2) < 1e-4

    n_x = nodes['N_X']
    assert abs(n_x.attributes["importance_weight"] - 1.0) < 1e-4

    n_y = nodes['N_Y']
    assert abs(n_y.attributes["importance_weight"] - 0.6) < 1e-4

    assert "importance_weight" not in nodes['N_Z'].attributes

    edges = updated_graph.get_edges()
    edge_map = {(e.node1.id, e.node2.id): e for e in edges if e.node1.id < e.node2.id}

    edge_ax = edge_map[('N_A', 'N_X')]
    assert abs(edge_ax.attributes["importance_weight"] - 1.0) < 1e-4

    edge_ay = edge_map[('N_A', 'N_Y')]
    assert abs(edge_ay.attributes["importance_weight"] - 0.8) < 1e-4

    edge_by = edge_map[('N_B', 'N_Y')]
    assert abs(edge_by.attributes["importance_weight"] - 0.4) < 1e-4