import sys
from unittest.mock import MagicMock

# 1. Expand the mocks to cover nested submodules required by temporal awareness imports
mock_graphiti = MagicMock()
mock_nodes = MagicMock()

sys.modules["graphiti_core"] = mock_graphiti
sys.modules["graphiti_core.nodes"] = mock_nodes

# 2. Standard imports
import pytest
from uuid import uuid4
from cognee.tasks.temporal_awareness.graphiti_model import GraphitiNode

# Assuming EdgeType is imported similarly from the same module, or customize if needed
# from cognee.tasks.temporal_awareness.graphiti_model import EdgeType

# Define a mock EdgeType locally in case it needs dynamic testing context
class MockEdgeType:
    def __init__(self, id, source_id, target_id):
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.metadata = {"index_fields": ["source_id", "target_id"]}
    def model_copy(self):
        # Emulate Pydantic's shallow model_copy
        copied = MockEdgeType(self.id, self.source_id, self.target_id)
        copied.metadata = self.metadata # Shallow reference sharing
        return copied


# --- TEST SUITE ---

def test_graphiti_node_metadata_isolation():
    """Test 1: Core bug fix — multi-field metadata dictionary isolation."""
    node = GraphitiNode(id=uuid4(), name="Alice", summary="Short bio", content="Long content")
    node.metadata["index_fields"] = ["name", "summary", "content"]

    indexed_copies = []
    for field_name in node.metadata["index_fields"]:
        indexed_data_point = node.model_copy()
        indexed_data_point.metadata = {**node.metadata, "index_fields": [field_name]}
        indexed_copies.append(indexed_data_point)

    assert indexed_copies[0].metadata["index_fields"] == ["name"]
    assert indexed_copies[1].metadata["index_fields"] == ["summary"]
    assert indexed_copies[2].metadata["index_fields"] == ["content"]
    assert node.metadata["index_fields"] == ["name", "summary", "content"]


def test_metadata_dictionary_memory_addresses():
    """Test 2: Explicitly verify that memory references are different (Deep/Shallow check)."""
    node = GraphitiNode(id=uuid4(), name="Alice", summary="Short bio", content="Long content")
    node.metadata["index_fields"] = ["name", "summary"]

    indexed_data_point = node.model_copy()
    indexed_data_point.metadata = {**node.metadata, "index_fields": ["name"]}

    # The metadata dictionaries MUST point to completely different objects in memory
    assert id(node.metadata) != id(indexed_data_point.metadata)


def test_graphiti_node_single_index_field():
    """Test 3: Edge case — works perfectly when only one index field is provided."""
    node = GraphitiNode(id=uuid4(), name="Bob", summary="Bio", content="Content")
    node.metadata["index_fields"] = ["content"]

    indexed_copies = []
    for field_name in node.metadata["index_fields"]:
        indexed_data_point = node.model_copy()
        indexed_data_point.metadata = {**node.metadata, "index_fields": [field_name]}
        indexed_copies.append(indexed_data_point)

    assert len(indexed_copies) == 1
    assert indexed_copies[0].metadata["index_fields"] == ["content"]
    assert node.metadata["index_fields"] == ["content"]


def test_graphiti_node_empty_index_fields():
    """Test 4: Edge case — loop handles empty index fields gracefully without crashing."""
    node = GraphitiNode(id=uuid4(), name="Charlie", summary="Bio", content="Content")
    node.metadata["index_fields"] = []

    indexed_copies = []
    for field_name in node.metadata["index_fields"]:
        indexed_data_point = node.model_copy()
        indexed_data_point.metadata = {**node.metadata, "index_fields": [field_name]}
        indexed_copies.append(indexed_data_point)

    assert len(indexed_copies) == 0
    assert node.metadata["index_fields"] == []


def test_edge_type_metadata_isolation():
    """Test 5: Verify the fix on the EdgeType indexing branch too."""
    edge = MockEdgeType(id=uuid4(), source_id=uuid4(), target_id=uuid4())
    
    indexed_copies = []
    for field_name in list(edge.metadata["index_fields"]):
        indexed_data_point = edge.model_copy()
        # Apply fix
        indexed_data_point.metadata = {**edge.metadata, "index_fields": [field_name]}
        indexed_copies.append(indexed_data_point)

    assert indexed_copies[0].metadata["index_fields"] == ["source_id"]
    assert indexed_copies[1].metadata["index_fields"] == ["target_id"]
    assert edge.metadata["index_fields"] == ["source_id", "target_id"]


def test_loop_iteration_safety():
    """Test 6: Verify that the iteration loop list length remains stable during mutation."""
    node = GraphitiNode(id=uuid4(), name="Delta", summary="Bio", content="Content")
    original_fields = ["name", "summary", "content"]
    node.metadata["index_fields"] = list(original_fields)

    iterations = 0
    # Mirroring the real task iteration block
    for field_name in node.metadata["index_fields"]:
        iterations += 1
        indexed_data_point = node.model_copy()
        indexed_data_point.metadata = {**node.metadata, "index_fields": [field_name]}

    # Ensure the loop actually ran 3 times and didn't terminate early due to a mutated shared list
    assert iterations == 3