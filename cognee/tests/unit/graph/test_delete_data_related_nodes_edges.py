import pytest
from cognee.graph.helpers import (
    delete_data_nodes,
    delete_data_edges,
)

# Minimal test for data-related node deletion
def test_delete_data_nodes():
    # Example seed data
    nodes = [{"id": 1, "data": "X"}, {"id": 2, "data": "Y"}]
    
    # Call the helper
    remaining_nodes = delete_data_nodes(nodes, data="X")
    
    # Assert nodes with data X are removed
    assert all(node["data"] != "X" for node in remaining_nodes)

# Minimal test for data-related edge deletion
def test_delete_data_edges():
    # Example seed data
    edges = [{"id": 1, "data": "X"}, {"id": 2, "data": "Y"}]
    
    # Call the helper
    remaining_edges = delete_data_edges(edges, data="X")
    
    # Assert edges with data X are removed
    assert all(edge["data"] != "X" for edge in remaining_edges)
