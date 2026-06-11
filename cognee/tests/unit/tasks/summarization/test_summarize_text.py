from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import DltRowDocument, TextDocument
from cognee.tasks.summarization.summarize_text import summarize_text


def _document():
    return TextDocument(
        name="notes.txt",
        raw_data_location="/tmp/notes.txt",
        external_metadata="",
        mime_type="text/plain",
    )


def _chunk(text="Chunk text", document=None, belongs_to_set=None):
    return DocumentChunk(
        text=text,
        chunk_size=len(text.split()),
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=document or _document(),
        contains=[],
        belongs_to_set=belongs_to_set,
    )


@pytest.mark.asyncio
async def test_summarize_text_sets_source_chunk_reference_fields():
    chunk = _chunk(belongs_to_set=["KEEP"])

    with patch(
        "cognee.tasks.summarization.summarize_text.extract_summary",
        new=AsyncMock(return_value=SimpleNamespace(summary="Short summary")),
    ):
        summaries = await summarize_text([chunk], summarization_model=object)

    assert len(summaries) == 1
    assert summaries[0].source_chunk_id == str(chunk.id)
    assert summaries[0].belongs_to_set == ["KEEP"]
    assert summaries[0].made_from == chunk


@pytest.mark.asyncio
async def test_summarize_text_leaves_dlt_row_chunks_unchanged():
    document = DltRowDocument(
        name="row",
        raw_data_location="/tmp/row.txt",
        external_metadata="",
        mime_type="application/x-dlt-row",
    )
    chunk = _chunk("Structured row", document=document)

    result = await summarize_text([chunk], summarization_model=object)

    assert result == [chunk]
