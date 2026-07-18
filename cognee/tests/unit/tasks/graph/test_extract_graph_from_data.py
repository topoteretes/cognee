import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.shared.data_models import KnowledgeGraph, Node, Edge as KGEdge
from cognee.tasks.graph.extract_graph_from_data import (
    extract_graph_from_data,
    integrate_chunk_graphs,
)

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
async def test_integration_forwards_pipeline_context_for_existing_edge_provenance(mock_retrieve):
    mock_retrieve.return_value = {}
    chunk = _make_chunk()
    graph = _two_node_graph()
    ctx = MagicMock()

    await integrate_chunk_graphs(
        [chunk],
        [graph],
        KnowledgeGraph,
        _mock_resolver(),
        ctx=ctx,
    )

    mock_retrieve.assert_awaited_once_with([chunk], [graph], ctx=ctx)


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


class _KGSubclass(KnowledgeGraph):
    pass


@pytest.mark.asyncio
@patch.object(egd_module, "retrieve_existing_edges", new_callable=AsyncMock)
async def test_integrate_chunk_graphs_accepts_knowledge_graph_subclass(mock_retrieve):
    mock_retrieve.return_value = {}
    chunk = _make_chunk()
    graph = _KGSubclass(
        nodes=[
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Bob", type="Person", description="desc"),
        ],
        edges=[KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )

    await integrate_chunk_graphs([chunk], [graph], _KGSubclass, _mock_resolver(), {})

    # Subclass should take the integration path, not the contains-passthrough path.
    assert chunk.contains is not None and len(chunk.contains) > 0
    _, entity = chunk.contains[0]
    assert entity.name in ("alice", "bob")


@pytest.mark.asyncio
@patch.object(egd_module, "integrate_chunk_graphs", new_callable=AsyncMock)
async def test_extract_graph_from_data_filters_edges_for_subclass(mock_integrate):
    chunk = _make_chunk()
    graph = _KGSubclass(
        nodes=[Node(id="n1", name="Alice", type="Person", description="desc")],
        edges=[
            KGEdge(source_node_id="n1", target_node_id="n1", relationship_name="self"),
            KGEdge(source_node_id="n1", target_node_id="missing", relationship_name="dangling"),
        ],
    )
    mock_integrate.side_effect = lambda *a, **kw: a[0]

    async def fake_calc(chunks, graph_model, custom_prompt, **kwargs):
        return [graph]

    config = {"ontology_config": {"ontology_resolver": _mock_resolver()}}

    await extract_graph_from_data(
        [chunk],
        _KGSubclass,
        config=config,
        calculate_chunk_graphs=fake_calc,
    )

    # Edge with missing target should be filtered out for the subclass.
    assert len(graph.edges) == 1
    assert graph.edges[0].target_node_id == "n1"


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


@pytest.mark.asyncio
async def test_all_dlt_chunks_short_circuits_llm_extraction():
    from cognee.modules.data.processing.document_types import DltRowDocument

    chunks = [_make_chunk(f"row {i}") for i in range(3)]
    for chunk in chunks:
        chunk.is_part_of = MagicMock(spec=DltRowDocument)

    fake_calc = MagicMock()

    result = await extract_graph_from_data(
        chunks,
        KnowledgeGraph,
        calculate_chunk_graphs=fake_calc,
    )

    assert result == chunks
    fake_calc.assert_not_called()


@pytest.mark.asyncio
@patch.object(egd_module, "integrate_chunk_graphs", new_callable=AsyncMock)
async def test_dlt_chunks_partitioned_from_llm_extraction(mock_integrate):
    from cognee.modules.data.processing.document_types import DltRowDocument

    dlt_chunk = _make_chunk("dlt row")
    dlt_chunk.is_part_of = MagicMock(spec=DltRowDocument)
    normal_chunks = [_make_chunk("regular a"), _make_chunk("regular b")]
    mock_integrate.side_effect = lambda *a, **kw: a[0]

    received = []

    async def fake_calc(chunks, graph_model, custom_prompt, **kwargs):
        received.extend(chunks)
        return [_two_node_graph() for _ in chunks]

    config = {"ontology_config": {"ontology_resolver": _mock_resolver()}}

    await extract_graph_from_data(
        [dlt_chunk, *normal_chunks],
        KnowledgeGraph,
        config=config,
        calculate_chunk_graphs=fake_calc,
    )

    # Only the non-DLT chunks reach LLM extraction, in original order.
    assert received == normal_chunks
