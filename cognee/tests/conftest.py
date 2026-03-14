"""Shared pytest fixtures for cognee test suite."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from cognee.shared.data_models import KnowledgeGraph, Node, Edge


@pytest.fixture
def temp_data_dir(tmp_path):
    """Provides a temporary data directory for test isolation."""
    return tmp_path


@pytest.fixture
def mock_llm_client():
    """Provides a mocked LLM client that returns deterministic responses."""
    mock = AsyncMock()
    mock.acreate_structured_output = AsyncMock(return_value="Generated answer")
    return mock


@pytest.fixture
def sample_knowledge_graph():
    """Pre-built KnowledgeGraph with test nodes and edges."""
    node1_kw = {"id": "1", "name": "Alice", "type": "Person", "description": "Test node A"}
    if "label" in Node.model_fields:
        node1_kw["label"] = "Alice"
    node1 = Node(**node1_kw)

    node2_kw = {"id": "2", "name": "Bob", "type": "Person", "description": "Test node B"}
    if "label" in Node.model_fields:
        node2_kw["label"] = "Bob"
    node2 = Node(**node2_kw)

    edge = Edge(source_node_id="1", target_node_id="2", relationship_name="KNOWS")

    kg_kw = {"nodes": [node1, node2], "edges": [edge]}
    if "summary" in KnowledgeGraph.model_fields:
        kg_kw["summary"] = "Test graph with two persons."
        kg_kw["description"] = "Minimal knowledge graph for testing."
    return KnowledgeGraph(**kg_kw)


@pytest.fixture
def sample_dataset():
    """Minimal dataset-shaped object for testing."""
    from uuid import uuid4

    return SimpleNamespace(
        id=uuid4(),
        name="test_dataset",
    )


@pytest.fixture(autouse=True)
def cleanup_graph_engine_cache():
    """Clears graph engine cache after each test to ensure isolation."""
    yield
    try:
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            _create_graph_engine,
        )

        _create_graph_engine.cache_clear()
    except Exception:
        pass
