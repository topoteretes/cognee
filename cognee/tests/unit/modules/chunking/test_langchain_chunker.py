"""Regression tests for LangchainChunker.

LangchainChunker was left referencing the pre-unification chunker API
(``max_chunk_tokens`` + a 4-arg base ``Chunker.__init__`` + ``word_count``/
``token_count`` fields on ``DocumentChunk``). After the chunker API was
unified on ``max_chunk_size`` those names no longer exist, so the class could
not even be instantiated through the standard ``Document.read`` call path
(``chunker_cls(self, max_chunk_size=..., get_text=...)``), and its ``read``
emitted fields ``DocumentChunk`` does not define while omitting the required
``chunk_size``. These tests pin the class to the current API.
"""

import sys
import uuid

import pytest

from cognee.modules.chunking.LangchainChunker import LangchainChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument

langchain_chunker_module = sys.modules.get("cognee.modules.chunking.LangchainChunker")


class _WordCountTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _MockEmbeddingEngine:
    tokenizer = _WordCountTokenizer()


class _MockVectorEngine:
    embedding_engine = _MockEmbeddingEngine()


def _mock_get_vector_engine():
    return _MockVectorEngine()


def _make_document() -> TextDocument:
    return TextDocument(
        id=uuid.uuid4(),
        name="t.txt",
        raw_data_location="t.txt",
        external_metadata="",
        mime_type="text/plain",
    )


def test_langchain_chunker_instantiates_via_document_call_path():
    """The standard Document.read call path uses the ``max_chunk_size`` kwarg."""

    async def get_text():
        yield "hello world"

    # Must not raise: this is exactly how every Document.read instantiates a chunker.
    chunker = LangchainChunker(_make_document(), max_chunk_size=128, get_text=get_text)
    assert chunker.max_chunk_size == 128


@pytest.mark.asyncio
async def test_langchain_chunker_read_yields_valid_document_chunks(monkeypatch):
    monkeypatch.setattr(langchain_chunker_module, "get_vector_engine", _mock_get_vector_engine)

    async def get_text():
        yield "The quick brown fox jumps over the lazy dog. " * 20

    chunker = LangchainChunker(_make_document(), max_chunk_size=4096, get_text=get_text)

    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) > 0
    for chunk in chunks:
        # chunk_size is a required field on DocumentChunk; the old code never
        # set it (it passed non-existent word_count/token_count instead), so
        # construction raised a ValidationError before this fix.
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(chunk.chunk_size, int)
        assert chunk.chunk_size > 0

    # Chunk indices are monotonically increasing.
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
