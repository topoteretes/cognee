"""update() re-extracts edited chunks with the extractor cognify() would use (SDK-893)."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.chunking.chunk_policy import ChunkPlan
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.shared.data_models import KnowledgeGraph

incremental = importlib.import_module("cognee.api.v1.update.incremental")


def _document():
    return TextDocument(
        id=uuid4(),
        name="brief",
        raw_data_location="/tmp/brief.md",
        mime_type="text/markdown",
        external_metadata=None,
    )


def _fresh(document, n=3):
    return [
        DocumentChunk(
            id=uuid4(),
            text=f"Changed paragraph {i}",
            chunk_size=3,
            chunk_index=i,
            cut_type="paragraph_end",
            is_part_of=document,
            max_chunk_tokens=512,
        )
        for i in range(n)
    ]


def _write(monkeypatch, document, fresh, extractor):
    config = SimpleNamespace(
        chunks_per_batch=2, triplet_embedding=False, contradiction_detection=False
    )
    llm_extract = AsyncMock(return_value=[])
    gliner_extract = AsyncMock(return_value=[])
    prepare_schema = AsyncMock()
    monkeypatch.setattr(incremental, "get_cognify_config", lambda: config)
    monkeypatch.setattr(incremental, "_resolve_extraction_config", lambda: None)
    monkeypatch.setattr(incremental, "extract_graph_and_summarize", llm_extract)
    monkeypatch.setattr(incremental, "extract_graph_and_summarize_with_gliner", gliner_extract)
    monkeypatch.setattr(incremental, "prepare_gliner_schema", prepare_schema)
    monkeypatch.setattr(incremental, "ensure_extractor_runtime", AsyncMock())
    monkeypatch.setattr(incremental, "add_data_points", AsyncMock())
    monkeypatch.setattr(incremental, "publish_updated_data", AsyncMock())
    coroutine = incremental._write_and_publish(
        {
            "staged": object(),
            "document": document,
            "stored_chunks": [],
            "plan": ChunkPlan(fresh=fresh),
        },
        document.id,
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
        None,
        KnowledgeGraph,
        None,
        uuid4(),
        extractor=extractor,
    )
    return coroutine, llm_extract, gliner_extract, prepare_schema


@pytest.mark.asyncio
async def test_default_extractor_is_the_llm_path(monkeypatch):
    document = _document()
    coroutine, llm_extract, gliner_extract, prepare_schema = _write(
        monkeypatch, document, _fresh(document), incremental.LLM_EXTRACTOR
    )
    await coroutine

    assert llm_extract.await_count == 2  # 3 chunks, batches of 2
    gliner_extract.assert_not_awaited()
    prepare_schema.assert_not_awaited()


@pytest.mark.asyncio
async def test_gliner_extractor_prepares_the_schema_once_and_never_calls_the_llm(monkeypatch):
    document = _document()
    fresh = _fresh(document)
    coroutine, llm_extract, gliner_extract, prepare_schema = _write(
        monkeypatch, document, fresh, incremental.GLINER_DEMO_EXTRACTOR
    )
    await coroutine

    llm_extract.assert_not_awaited()
    prepare_schema.assert_awaited_once()
    documents, _schema, max_chunk_size = prepare_schema.await_args.args
    assert documents == [document]
    assert max_chunk_size == 512  # the budget the edited chunks were cut against
    assert prepare_schema.await_args.kwargs["chunker"] is incremental.TextChunker
    assert gliner_extract.await_count == 2
    batches = [call.args[0] for call in gliner_extract.await_args_list]
    assert [len(b) for b in batches] == [2, 1]
    assert all(chunk in fresh for batch in batches for chunk in batch)
    # one stats object across the batches, like the pipeline task
    stats = {id(call.args[1]) for call in gliner_extract.await_args_list}
    assert len(stats) == 1


@pytest.mark.asyncio
async def test_gliner_extractor_with_no_fresh_chunks_prepares_nothing(monkeypatch):
    document = _document()
    coroutine, llm_extract, gliner_extract, prepare_schema = _write(
        monkeypatch, document, [], incremental.GLINER_DEMO_EXTRACTOR
    )
    await coroutine

    prepare_schema.assert_not_awaited()
    gliner_extract.assert_not_awaited()
    llm_extract.assert_not_awaited()
