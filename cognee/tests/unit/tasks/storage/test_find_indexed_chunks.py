"""find_indexed_chunks asks the vector store which chunks already exist (SDK-397)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.pipelines.models import PipelineContext
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
async def test_a_store_that_cannot_answer_records_nothing():
    chunks = _chunks(2)
    engine = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("table missing")))
    ctx = PipelineContext()
    with patch.object(module, "get_vector_engine_async", AsyncMock(return_value=engine)):
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
