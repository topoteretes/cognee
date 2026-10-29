"""find_indexed_chunks asks the vector store which chunks already exist (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.tasks.storage.add_data_points import SEARCH_INDEXED_IDS
from cognee.tasks.storage.find_indexed_chunks import LOOKUP_BATCH_SIZE, find_indexed_chunks

module = sys.modules["cognee.tasks.storage.find_indexed_chunks"]


def _chunks(count):
    document = TextDocument(
        name="n.txt", raw_data_location="/n", external_metadata=None, mime_type="text/plain"
    )
    return [
        DocumentChunk(
            text=f"chunk {i}",
            chunk_size=2,
            chunk_index=i,
            cut_type="paragraph_end",
            is_part_of=document,
        )
        for i in range(count)
    ]


def _engine(found_ids):
    engine = SimpleNamespace(
        retrieve=AsyncMock(
            side_effect=lambda collection, ids: [
                SimpleNamespace(id=chunk_id) for chunk_id in ids if str(chunk_id) in found_ids
            ]
        )
    )
    return engine


@pytest.mark.asyncio
async def test_records_the_chunks_the_store_holds_and_their_documents():
    chunks = _chunks(3)
    found = {str(chunks[0].id), str(chunks[2].id)}
    engine = _engine(found)
    ctx = PipelineContext()
    with patch.object(module, "get_vector_engine_async", AsyncMock(return_value=engine)):
        result = await find_indexed_chunks(chunks, ctx=ctx)

    assert result is chunks
    assert ctx.extras[SEARCH_INDEXED_IDS] == found | {str(chunks[0].is_part_of.id)}
    engine.retrieve.assert_awaited_once()
    assert engine.retrieve.await_args.args[0] == "DocumentChunk_text"


@pytest.mark.asyncio
async def test_a_store_that_cannot_answer_fails_before_extraction():
    """Extraction's final write goes to the same store; failing here spends no tokens."""
    chunks = _chunks(2)
    engine = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("store unreachable")))
    ctx = PipelineContext()
    with (
        patch.object(module, "get_vector_engine_async", AsyncMock(return_value=engine)),
        pytest.raises(RuntimeError, match="store unreachable"),
    ):
        await find_indexed_chunks(chunks, ctx=ctx)
    assert SEARCH_INDEXED_IDS not in ctx.extras


@pytest.mark.asyncio
async def test_without_a_context_or_chunks_nothing_is_looked_up():
    chunks = _chunks(1)
    with patch.object(module, "get_vector_engine_async", AsyncMock()) as get_engine:
        assert await find_indexed_chunks(chunks) is chunks
        assert await find_indexed_chunks([], ctx=PipelineContext()) == []
    get_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_ids_are_looked_up_in_bounded_batches():
    chunks = _chunks(LOOKUP_BATCH_SIZE + 1)
    engine = _engine({str(chunk.id) for chunk in chunks})
    with patch.object(module, "get_vector_engine_async", AsyncMock(return_value=engine)):
        await find_indexed_chunks(chunks, ctx=PipelineContext())
    assert engine.retrieve.await_count == 2


# --- chunks an earlier indexing run cut with other settings -------------------


def _item_ctx(dataset_id, stamped):
    data_item = SimpleNamespace(
        id=uuid4(),
        pipeline_status={
            name: {str(dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}
            for name in stamped
        },
    )
    return PipelineContext(data_item=data_item, dataset=SimpleNamespace(id=dataset_id))


def _graph(document_id, stored_nodes):
    connections = [
        ({"id": node["id"]}, {"relationship_name": "is_part_of"}, {"id": str(document_id)})
        for node in stored_nodes
    ]
    return SimpleNamespace(
        get_connections=AsyncMock(return_value=connections),
        get_nodes=AsyncMock(return_value=stored_nodes),
    )


def _cut(chunks, tokens):
    for chunk in chunks:
        chunk.max_chunk_tokens = tokens
        chunk.chunker_id = "text"
    return chunks


async def _run(chunks, ctx, graph, found_ids=()):
    deleted = AsyncMock()
    with (
        patch.object(
            module, "get_vector_engine_async", AsyncMock(return_value=_engine(set(found_ids)))
        ),
        patch.object(module, "get_graph_engine", AsyncMock(return_value=graph)),
        patch.object(module, "stores_provenance_in_graph", AsyncMock(return_value=True)),
        patch.object(module, "delete_chunks_incremental", deleted),
    ):
        await find_indexed_chunks(chunks, ctx=ctx)
    return deleted


def _stored(tokens, count=2):
    return [
        {
            "id": str(uuid4()),
            "type": "DocumentChunk",
            "max_chunk_tokens": tokens,
            "chunker_id": "text",
        }
        for _ in range(count)
    ]


@pytest.mark.asyncio
async def test_a_retry_with_another_chunk_size_deletes_the_earlier_cut():
    chunks = _cut(_chunks(2), tokens=256)
    dataset_id = uuid4()
    ctx = _item_ctx(dataset_id, stamped=["indexing_pipeline"])
    stale = _stored(tokens=64)

    deleted = await _run(chunks, ctx, _graph(chunks[0].is_part_of.id, stale))

    deleted.assert_awaited_once()
    ids, called_dataset, called_item = deleted.await_args.args
    assert sorted(ids) == sorted(node["id"] for node in stale)
    assert (called_dataset, called_item) == (dataset_id, ctx.data_item.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stamped", "found", "stored_tokens"),
    [
        (["indexing_pipeline", "cognify_pipeline"], False, 64),  # extracted before: not ours
        ([], False, 64),  # no earlier indexing run: nothing of it is stored
        (["indexing_pipeline"], True, 256),  # existing chunks with the same settings
        (["indexing_pipeline"], False, 256),  # same settings: never stale
        (["indexing_pipeline"], False, None),  # settings unrecorded: never taken for stale
    ],
)
async def test_nothing_else_is_deleted(stamped, found, stored_tokens):
    chunks = _cut(_chunks(2), tokens=256)
    ctx = _item_ctx(uuid4(), stamped=stamped)
    found_ids = {str(chunks[0].id)} if found else set()

    deleted = await _run(
        chunks, ctx, _graph(chunks[0].is_part_of.id, _stored(stored_tokens)), found_ids
    )

    deleted.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_document_is_checked_once_per_item():
    chunks = _cut(_chunks(2), tokens=256)
    ctx = _item_ctx(uuid4(), stamped=["indexing_pipeline"])
    graph = _graph(chunks[0].is_part_of.id, _stored(tokens=64))

    await _run(chunks[:1], ctx, graph)
    await _run(chunks[1:], ctx, graph)

    graph.get_connections.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_retry_retires_the_old_cut_even_when_a_chunk_id_is_reused():
    """A short paragraph can keep its content ID across a chunk-size change.

    Retire before looking up vectors: the shared ID can itself be deleted,
    and must then be embedded again rather than recorded as already indexed.
    """
    chunks = _cut(_chunks(2), tokens=256)
    ctx = _item_ctx(uuid4(), stamped=["indexing_pipeline"])
    stale = _stored(tokens=64)
    stale[0]["id"] = str(chunks[0].id)
    found = {stale[0]["id"]}
    engine = _engine(found)
    graph = _graph(chunks[0].is_part_of.id, stale)

    async def delete(ids, *_args):
        found.difference_update(ids)

    with (
        patch.object(module, "get_vector_engine_async", AsyncMock(return_value=engine)),
        patch.object(module, "get_graph_engine", AsyncMock(return_value=graph)),
        patch.object(module, "stores_provenance_in_graph", AsyncMock(return_value=True)),
        patch.object(module, "delete_chunks_incremental", AsyncMock(side_effect=delete)) as deleted,
    ):
        await find_indexed_chunks(chunks, ctx=ctx)

    deleted.assert_awaited_once()
    assert set(deleted.await_args.args[0]) == {node["id"] for node in stale}
    assert not ctx.extras.get(SEARCH_INDEXED_IDS)


@pytest.mark.asyncio
async def test_a_failed_check_fails_the_run():
    """Continuing would leave the old cut next to the new one, so chunk searches
    would return the passage twice, once without entities."""
    chunks = _cut(_chunks(1), tokens=256)
    ctx = _item_ctx(uuid4(), stamped=["indexing_pipeline"])
    graph = SimpleNamespace(get_connections=AsyncMock(side_effect=RuntimeError("graph down")))

    with pytest.raises(RuntimeError, match="graph down"):
        await _run(chunks, ctx, graph)
