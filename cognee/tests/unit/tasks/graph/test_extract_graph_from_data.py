import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.shared.data_models import KnowledgeGraph, Node, Edge as KGEdge
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs

egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")


def _mock_resolver():
    resolver = MagicMock()
    resolver.get_subgraph.return_value = ([], [], None)
    return resolver


def _make_chunk(text="chunk text"):
    chunk = MagicMock()
    chunk.text = text
    chunk.contains = None
    chunk.belongs_to_set = []
    return chunk


def _two_node_graph():
    return KnowledgeGraph(
        nodes=[
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Bob", type="Person", description="desc"),
        ],
        edges=[KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )


@pytest.mark.asyncio
@patch.object(egd_module, "retrieve_existing_edges", new_callable=AsyncMock)
async def test_integration_does_not_write_to_db(mock_retrieve):
    mock_retrieve.return_value = {}
    chunk = _make_chunk()
    graph = _two_node_graph()
    context = {}

    # If add_data_points were called it would fail (not mocked), proving it is not called.
    result = await integrate_chunk_graphs(
        [chunk], [graph], KnowledgeGraph, _mock_resolver(), context
    )

    assert result == [chunk]


@pytest.mark.asyncio
@patch.object(egd_module, "retrieve_existing_edges", new_callable=AsyncMock)
async def test_chunk_contains_entities_after_integration(mock_retrieve):
    mock_retrieve.return_value = {}
    chunk = _make_chunk()
    graph = _two_node_graph()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, _mock_resolver(), {})

    assert chunk.contains is not None and len(chunk.contains) > 0
    _, entity = chunk.contains[0]
    assert entity.name in ("alice", "bob")


@pytest.mark.asyncio
@patch.object(egd_module, "retrieve_existing_edges", new_callable=AsyncMock)
async def test_entity_relations_populated_after_integration(mock_retrieve):
    mock_retrieve.return_value = {}
    chunk = _make_chunk()
    graph = _two_node_graph()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, _mock_resolver(), {})

    entities = [e for _, e in chunk.contains]
    alice = next((e for e in entities if e.name == "alice"), None)
    assert alice is not None
    assert len(alice.relations) == 1
    _, target = alice.relations[0]
    assert target.name == "bob"


@pytest.mark.asyncio
@patch.object(egd_module, "retrieve_existing_edges", new_callable=AsyncMock)
async def test_cache_entity_embeddings_hook_called(mock_retrieve):
    mock_retrieve.return_value = {}
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Alice", type="Person", description="desc")],
        edges=[],
    )
    hook = MagicMock(return_value=None)

    await integrate_chunk_graphs(
        [chunk],
        [graph],
        KnowledgeGraph,
        _mock_resolver(),
        {},
        cache_entity_embeddings=hook,
    )

    hook.assert_called_once()
    entity_nodes_arg = hook.call_args.args[0]
    assert len(entity_nodes_arg) > 0


@pytest.mark.asyncio
async def test_non_knowledge_graph_model_unchanged():
    from pydantic import BaseModel

    class CustomModel(BaseModel):
        pass

    chunk = _make_chunk()
    custom_graph = CustomModel()

    result = await integrate_chunk_graphs(
        [chunk], [custom_graph], CustomModel, _mock_resolver(), {}
    )

    assert chunk.contains == custom_graph
    assert result == [chunk]
