import pytest
from cognee.graph.helpers import (
    delete_dataset_nodes,
    delete_dataset_edges,
)

# Minimal test for dataset-related node deletion
def test_delete_dataset_nodes():
    # Example seed data
    nodes = [{"id": 1, "dataset": "A"}, {"id": 2, "dataset": "B"}]
    
    # Call the helper
    remaining_nodes = delete_dataset_nodes(nodes, dataset="A")
    
    # Assert nodes from dataset A are removed
    assert all(node["dataset"] != "A" for node in remaining_nodes)

# Minimal test for dataset-related edge deletion
def test_delete_dataset_edges():
    # Example seed data
    edges = [{"id": 1, "dataset": "A"}, {"id": 2, "dataset": "B"}]
    
    # Call the helper
    remaining_edges = delete_dataset_edges(edges, dataset="A")
    
    # Assert edges from dataset A are removed
    assert all(edge["dataset"] != "A" for edge in remaining_edges)
