"""Regression tests for LangchainChunker.

LangchainChunker was left targeting the pre-unification chunker API
(``max_chunk_tokens``, a 4-arg base ``Chunker.__init__``, and
``word_count``/``token_count`` fields ``DocumentChunk`` does not define), so
it could not be instantiated at all — neither positionally nor through the
standard ``Document.read`` call path
(``chunker_cls(self, max_chunk_size=..., get_text=...)``). These tests pin
the class to the current API.

The langchain import is guarded so test collection stays safe in
environments without the ``langchain`` extra installed (the previous fix,
#2966, was reverted in #3167; its regression test imported langchain
unconditionally).
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("langchain_text_splitters")

from cognee.modules.chunking.chunk_id import chunk_content_hash
from cognee.modules.chunking.LangchainChunker import LangchainChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document


class _WordCountTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _make_document() -> Document:
    return Document(
        id=uuid.uuid4(),
        name="test-document",
        raw_data_location="/tmp/test-document.txt",
        external_metadata=None,
        mime_type="text/plain",
    )


def _mock_vector_engine():
    engine = MagicMock()
    engine.embedding_engine.tokenizer = _WordCountTokenizer()
    return engine


def test_constructs_positionally_like_base_chunker():
    document = _make_document()

    async def get_text():
        yield "hello world"

    chunker = LangchainChunker(document, get_text, 512)

    assert chunker.max_chunk_size == 512
    assert chunker.document is document


def test_constructs_via_document_read_call_shape():
    """Every Document.read() builds its chunker with
    chunker_cls(self, max_chunk_size=..., get_text=...)."""
    document = _make_document()

    async def get_text():
        yield "hello world"

    chunker = LangchainChunker(document, max_chunk_size=512, get_text=get_text)

    assert chunker.max_chunk_size == 512


@pytest.mark.asyncio
async def test_read_yields_valid_document_chunks():
    document = _make_document()
    text = "one two three four five. " * 40

    async def get_text():
        yield text

    chunker = LangchainChunker(document, max_chunk_size=512, get_text=get_text)

    with patch(
        "cognee.modules.chunking.LangchainChunker.get_vector_engine_async",
        new=AsyncMock(return_value=_mock_vector_engine()),
    ):
        chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) > 0
    for index, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.text
        assert chunk.chunk_size > 0
        assert chunk.chunk_index == index
        assert chunk.is_part_of == document
        assert chunk.document_id == str(document.id)


@pytest.mark.asyncio
async def test_read_raises_for_chunks_over_max_chunk_size():
    document = _make_document()

    async def get_text():
        yield "word " * 50

    # splitter chunk_size is large enough that the split chunk exceeds
    # max_chunk_size, which must raise instead of yielding oversized chunks
    chunker = LangchainChunker(document, max_chunk_size=3, get_text=get_text, chunk_size=1000)

    with (
        patch(
            "cognee.modules.chunking.LangchainChunker.get_vector_engine_async",
            new=AsyncMock(return_value=_mock_vector_engine()),
        ),
        pytest.raises(ValueError, match="larger than the maximum"),
    ):
        [chunk async for chunk in chunker.read()]


async def _read_chunks(document: Document, text: str) -> list[DocumentChunk]:
    async def get_text():
        yield text

    chunker = LangchainChunker(
        document, max_chunk_size=512, get_text=get_text, chunk_size=8, chunk_overlap=0
    )
    with patch(
        "cognee.modules.chunking.LangchainChunker.get_vector_engine_async",
        new=AsyncMock(return_value=_mock_vector_engine()),
    ):
        return [chunk async for chunk in chunker.read()]


@pytest.mark.asyncio
async def test_identical_text_in_different_documents_gets_distinct_chunk_ids():
    """Chunk ids were derived from the text alone, so two documents sharing a
    passage produced one chunk id and add_data_points kept only one node."""
    text = "Confidential. Do not distribute outside the company."

    chunks_a = await _read_chunks(_make_document(), text)
    chunks_b = await _read_chunks(_make_document(), text)

    assert len(chunks_a) == len(chunks_b) == 1
    assert chunks_a[0].id != chunks_b[0].id


@pytest.mark.asyncio
async def test_repeated_text_in_one_document_gets_distinct_chunk_ids():
    footer = "Confidential. Do not distribute outside the company."
    chunks = await _read_chunks(_make_document(), f"{footer}\n\nPage one body.\n\n{footer}")

    assert [chunk.text for chunk in chunks] == [footer, "Page one body.", footer]
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert chunks[0].content_hash == chunks[2].content_hash == chunk_content_hash(footer)


@pytest.mark.asyncio
async def test_chunk_ids_are_stable_for_the_same_document_and_text():
    document = _make_document()
    text = "Alpha beta gamma.\n\nDelta epsilon zeta."

    first = await _read_chunks(document, text)
    second = await _read_chunks(document, text)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
