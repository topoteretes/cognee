import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.observability import capture
from cognee.modules.observability.capture import KIND_EXTRACTION_CHUNK_GRAPH
from cognee.shared.data_models import Edge, KnowledgeGraph, Node

cascade_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data_v2")


@pytest.mark.asyncio
@patch.object(cascade_module, "integrate_chunk_graphs", new_callable=AsyncMock)
@patch.object(cascade_module, "extract_edge_triplets", new_callable=AsyncMock)
@patch.object(
    cascade_module,
    "extract_content_nodes_and_relationship_names",
    new_callable=AsyncMock,
)
@patch.object(cascade_module, "extract_nodes", new_callable=AsyncMock)
async def test_extract_graph_from_data_captures_the_raw_chunk_graph(
    mock_extract_nodes,
    mock_extract_content_nodes_and_relationship_names,
    mock_extract_edge_triplets,
    mock_integrate_chunk_graphs,
    fake_capture_sink,
):
    """Eval capture (SDK-529) on the cascade route: the helper is a coroutine and
    must be awaited — un-awaited it captured nothing and left a never-awaited
    coroutine behind — so with capture on the raw per-chunk graph is recorded
    before ``integrate_chunk_graphs`` mutates it."""
    chunk = MagicMock(text="Alice founded Acme.")
    chunk.id = uuid4()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="alice", name="Alice", type="Person", description="founder"),
            Node(id="acme", name="Acme", type="Organization", description="company"),
        ],
        edges=[Edge(source_node_id="alice", target_node_id="acme", relationship_name="founded")],
    )
    mock_extract_nodes.return_value = ["node"]
    mock_extract_content_nodes_and_relationship_names.return_value = (["node"], ["relationship"])
    mock_extract_edge_triplets.return_value = graph
    mock_integrate_chunk_graphs.return_value = [chunk]

    await cascade_module.extract_graph_from_data(
        [chunk], ontology_resolver=MagicMock(), ctx=MagicMock()
    )
    await capture.drain()

    [record] = [r for r in fake_capture_sink.records if r["kind"] == KIND_EXTRACTION_CHUNK_GRAPH]
    assert record["stage"] == "extract_graph_from_data"
    assert record["payload"]["chunk_id"] == str(chunk.id)
    assert record["payload"]["chunk_size_chars"] == len(chunk.text)
    assert [node["name"] for node in record["payload"]["graph"]["nodes"]] == ["Alice", "Acme"]
    mock_integrate_chunk_graphs.assert_awaited_once()


@pytest.mark.asyncio
@patch.object(cascade_module, "integrate_chunk_graphs", new_callable=AsyncMock)
@patch.object(cascade_module, "extract_edge_triplets", new_callable=AsyncMock)
@patch.object(
    cascade_module,
    "extract_content_nodes_and_relationship_names",
    new_callable=AsyncMock,
)
@patch.object(cascade_module, "extract_nodes", new_callable=AsyncMock)
async def test_extract_graph_from_data_passes_ontology_resolver(
    mock_extract_nodes,
    mock_extract_content_nodes_and_relationship_names,
    mock_extract_edge_triplets,
    mock_integrate_chunk_graphs,
):
    chunk = MagicMock(text="chunk text")
    ontology_resolver = MagicMock()
    ctx = MagicMock()

    mock_extract_nodes.return_value = ["node"]
    mock_extract_content_nodes_and_relationship_names.return_value = (
        ["node"],
        ["relationship"],
    )
    mock_extract_edge_triplets.return_value = MagicMock(name="chunk_graph")
    mock_integrate_chunk_graphs.return_value = [chunk]

    result = await cascade_module.extract_graph_from_data(
        [chunk], ontology_resolver=ontology_resolver, ctx=ctx
    )

    assert result == [chunk]
    mock_integrate_chunk_graphs.assert_awaited_once_with(
        data_chunks=[chunk],
        chunk_graphs=[mock_extract_edge_triplets.return_value],
        graph_model=KnowledgeGraph,
        ontology_resolver=ontology_resolver,
        ctx=ctx,
    )
