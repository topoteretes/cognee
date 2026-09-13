"""Incremental writes must remain searchable by the document's node sets."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.chunking.chunk_policy import ChunkPlan
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.tasks.documents.classify_documents import update_node_set

incremental = importlib.import_module("cognee.api.v1.update.incremental")


def _document(tagged=True):
    document = TextDocument(
        id=uuid4(),
        title="brief.md",
        name="brief",
        raw_data_location="/tmp/brief.md",
        mime_type="text/markdown",
        external_metadata='{"node_set":["workflow:meeting-actions","kind:brief"]}'
        if tagged
        else "{}",
    )
    update_node_set(document)
    return document


@pytest.mark.asyncio
@pytest.mark.parametrize("tagged", [True, False])
async def test_fresh_chunks_reach_extraction_with_document_membership(monkeypatch, tagged):
    document = _document(tagged)
    fresh = [
        DocumentChunk(
            id=uuid4(),
            text=f"Changed action {i}",
            chunk_size=3,
            chunk_index=i,
            cut_type="paragraph_end",
            is_part_of=document,
        )
        for i in range(2)
    ]
    observed = []

    async def extract(batch, **kwargs):
        observed.extend(batch)
        for chunk in batch:
            assert chunk.belongs_to_set == document.belongs_to_set
            assert chunk.source_node_set == document.source_node_set
        return []

    config = SimpleNamespace(
        chunks_per_batch=1, triplet_embedding=False, contradiction_detection=False
    )
    publish = AsyncMock()
    monkeypatch.setattr(incremental, "get_cognify_config", lambda: config)
    monkeypatch.setattr(incremental, "_resolve_extraction_config", lambda: None)
    monkeypatch.setattr(incremental, "extract_graph_and_summarize", extract)
    monkeypatch.setattr(incremental, "add_data_points", AsyncMock())
    monkeypatch.setattr(incremental, "publish_updated_data", publish)
    result = await incremental._write_and_publish(
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
        object,
        None,
        uuid4(),
    )
    assert observed == fresh
    assert result["added_chunks"] == 2
    publish.assert_awaited_once()


@pytest.mark.parametrize("tagged", [True, False])
def test_repositioned_chunks_keep_document_membership(tagged):
    document = _document(tagged)
    chunk_id = uuid4()
    chunk = incremental._rehydrate_chunk(
        document,
        {"id": str(chunk_id), "text": "existing action", "chunk_size": 2},
        7,
    )
    assert chunk.belongs_to_set == document.belongs_to_set
    assert chunk.source_node_set == document.source_node_set
    assert chunk.id == chunk_id
    assert chunk.chunk_index == 7
