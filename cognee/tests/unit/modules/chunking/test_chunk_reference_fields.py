"""Unit tests asserting chunkers populate the reference scalar fields.

The References (Evidence) feature relies on every produced ``DocumentChunk``
carrying flat ``document_id`` / ``document_name`` scalars (basename only, never
an absolute path). These tests run the real chunkers over an in-memory text
generator (no LLM, no network) and assert the fields are set, and that the
format helper renders the 1-based number from ``chunk_index + 1``.
"""

from uuid import uuid4

import pytest

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap
from cognee.modules.data.processing.document_types import Document
from cognee.modules.retrieval.utils.references import format_chunk_references


@pytest.fixture(params=["TextChunker", "TextChunkerWithOverlap"])
def chunker_class(request):
    return TextChunker if request.param == "TextChunker" else TextChunkerWithOverlap


def _make_text_generator(*texts):
    async def gen():
        for text in texts:
            yield text

    return gen


async def _collect(chunker):
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_chunk_sets_document_id_and_name_from_document_name(chunker_class):
    """document_id is the document id; document_name uses document.name when present."""
    doc_id = uuid4()
    document = Document(
        id=doc_id,
        name="annual_report.pdf",
        raw_data_location="/abs/path/to/annual_report.pdf",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunker = chunker_class(document, _make_text_generator("Hello world."), max_chunk_size=512)
    chunks = await _collect(chunker)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == str(doc_id)
        assert chunk.document_name == "annual_report.pdf"


@pytest.mark.asyncio
async def test_chunk_document_name_falls_back_to_basename(chunker_class):
    """When document.name is falsy, document_name uses basename(raw_data_location)."""
    document = Document(
        id=uuid4(),
        name="",  # empty -> fall back to basename
        raw_data_location="/abs/path/to/source_file.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunker = chunker_class(
        document, _make_text_generator("Some content here."), max_chunk_size=512
    )
    chunks = await _collect(chunker)

    assert len(chunks) >= 1
    for chunk in chunks:
        # basename only, never the absolute path
        assert chunk.document_name == "source_file.txt"
        assert "/" not in chunk.document_name


@pytest.mark.asyncio
async def test_format_helper_renders_one_based_number_from_real_chunk(chunker_class):
    """A real chunk's payload-shaped dict renders chunk_index + 1 in the Evidence block."""
    document = Document(
        id=uuid4(),
        name="report.pdf",
        raw_data_location="/p/report.pdf",
        external_metadata=None,
        mime_type="text/plain",
    )
    chunker = chunker_class(document, _make_text_generator("First chunk text."), max_chunk_size=512)
    chunks = await _collect(chunker)
    chunk = chunks[0]

    payload = {
        "document_name": chunk.document_name,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
    }
    result = format_chunk_references([payload])

    assert f"- chunk {chunk.chunk_index + 1} of document report.pdf:" in result
