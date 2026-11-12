"""index_chunks_for_search and the embedding skip in add_data_points (SDK-397)."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.storage.add_data_points import SEARCH_INDEXED_IDS, add_data_points
from cognee.tasks.storage.index_chunks_for_search import index_chunks_for_search

adp_module = sys.modules["cognee.tasks.storage.add_data_points"]
early_module = sys.modules["cognee.tasks.storage.index_chunks_for_search"]


class SimplePoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _chunk(document: Document, text: str, index: int) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        chunk_size=len(text.split()),
        chunk_index=index,
        cut_type="paragraph_end",
        is_part_of=document,
    )


def _document() -> Document:
    return Document(
        name="notes.txt",
        raw_data_location="/tmp/notes.txt",
        external_metadata=None,
        mime_type="text/plain",
    )


def _make_unified_mock():
    graph_engine = AsyncMock()
    vector_engine = MagicMock()
    unified = AsyncMock()
    unified.graph = graph_engine
    unified.vector = vector_engine
    unified.has_capability = MagicMock(return_value=False)
    graph_engine.is_empty = AsyncMock(return_value=False)
    graph_engine.get_graph_metadata = AsyncMock(return_value={})
    return unified, graph_engine, vector_engine


@pytest.mark.asyncio
@patch.object(early_module, "add_data_points", new_callable=AsyncMock)
async def test_stores_the_chunks_and_records_what_it_embedded(mock_add):
    document = _document()
    chunks = [_chunk(document, "Alice met the rabbit.", 0), _chunk(document, "The queen.", 1)]
    ctx = PipelineContext(pipeline_run_id=uuid4())

    result = await index_chunks_for_search(chunks, ctx=ctx)

    assert result is chunks, "the extraction task must receive the very same chunk objects"
    mock_add.assert_awaited_once()
    stored, kwargs = mock_add.await_args.args[0], mock_add.await_args.kwargs
    assert stored == chunks and kwargs["ctx"] is ctx
    assert kwargs["index_edges"] is False, "structural edges are indexed by the final write"
    assert ctx.extras[SEARCH_INDEXED_IDS] == {
        str(chunks[0].id),
        str(chunks[1].id),
        str(document.id),
    }


@pytest.mark.asyncio
@patch.object(early_module, "add_data_points", new_callable=AsyncMock)
async def test_accumulates_across_batches_and_tolerates_no_context(mock_add):
    document = _document()
    first = [_chunk(document, "one", 0)]
    second = [_chunk(document, "two", 1)]
    ctx = PipelineContext()

    await index_chunks_for_search(first, ctx=ctx)
    await index_chunks_for_search(second, ctx=ctx)
    assert {str(first[0].id), str(second[0].id)} <= ctx.extras[SEARCH_INDEXED_IDS]

    # No context: the write still happens, nothing is recorded, nothing raises.
    assert await index_chunks_for_search(first) is first
    assert mock_add.await_count == 3


@pytest.mark.asyncio
@patch.object(early_module, "add_data_points", new_callable=AsyncMock)
async def test_an_empty_batch_writes_nothing(mock_add):
    ctx = PipelineContext()

    assert await index_chunks_for_search([], ctx=ctx) == []

    mock_add.assert_not_awaited()
    assert SEARCH_INDEXED_IDS not in ctx.extras


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_skips_embedding_already_indexed_nodes(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    """The final write upserts every graph row but embeds only the new nodes."""
    early = SimplePoint(text="embedded by index_chunks_for_search")
    late = SimplePoint(text="an entity extracted afterwards")
    mock_get_graph.side_effect = [([early, late], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, _ = _make_unified_mock()
    mock_get_unified.return_value = unified
    ctx = PipelineContext(extras={SEARCH_INDEXED_IDS: {str(early.id)}})

    await add_data_points([early], ctx=ctx)

    written = graph_engine.add_nodes.await_args.args[0]
    assert {node.id for node in written} == {early.id, late.id}
    embedded = mock_index_nodes.await_args.args[0]
    assert [node.id for node in embedded] == [late.id]


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_skips_the_record_on_hybrid_backends_too(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    """Hybrid backends embed inside the graph write: the recorded nodes go in
    without vectors, the rest with them."""
    from cognee.infrastructure.databases.unified.capabilities import EngineCapability

    early = SimplePoint(text="embedded by index_chunks_for_search")
    late = SimplePoint(text="an entity extracted afterwards")
    mock_get_graph.side_effect = [([early, late], [])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, _ = _make_unified_mock()
    unified.has_capability = MagicMock(
        side_effect=lambda capability: capability == EngineCapability.HYBRID_WRITE
    )
    mock_get_unified.return_value = unified
    ctx = PipelineContext(extras={SEARCH_INDEXED_IDS: {str(early.id)}})

    await add_data_points([early], ctx=ctx)

    plain = graph_engine.add_nodes.await_args.args[0]
    with_vectors = graph_engine.add_nodes_with_vectors.await_args.args[0]
    assert [node.id for node in plain] == [early.id]
    assert [node.id for node in with_vectors] == [late.id]
    mock_index_nodes.assert_not_awaited()


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_can_leave_edge_indexing_to_a_later_write(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    first = SimplePoint(text="a chunk")
    second = SimplePoint(text="its document")
    edge = (str(first.id), str(second.id), "is_part_of", {})
    mock_get_graph.side_effect = [([first, second], [edge])]
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, graph_engine, _ = _make_unified_mock()
    mock_get_unified.return_value = unified

    await add_data_points([first], ctx=PipelineContext(), index_edges=False)

    graph_engine.add_edges.assert_awaited_once(), "the edges are still written"
    mock_index_nodes.assert_awaited_once()
    mock_index_edges.assert_not_awaited()


@pytest.mark.asyncio
@patch.object(adp_module, "index_graph_edges")
@patch.object(adp_module, "index_data_points")
@patch.object(adp_module, "get_unified_engine")
@patch.object(adp_module, "deduplicate_nodes_and_edges")
@patch.object(adp_module, "get_graph_from_model")
async def test_add_data_points_embeds_everything_without_a_record(
    mock_get_graph, mock_dedup, mock_get_unified, mock_index_nodes, mock_index_edges
):
    point = SimplePoint(text="nothing was embedded earlier")
    mock_get_graph.return_value = ([point], [])
    mock_dedup.side_effect = lambda n, e: (n, e)
    unified, _, _ = _make_unified_mock()
    mock_get_unified.return_value = unified

    await add_data_points([point], ctx=PipelineContext())
    await add_data_points([point])

    for call in mock_index_nodes.await_args_list:
        assert [node.id for node in call.args[0]] == [point.id]
