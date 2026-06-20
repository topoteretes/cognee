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

pytest.importorskip("langchain_text_splitters")

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


@pytest.mark.asyncio
async def test_langchain_chunker_honors_small_max_chunk_size(monkeypatch):
    """A small max_chunk_size must shrink the splitter, not just the size check.

    Every Document.read call passes only max_chunk_size and lets chunk_size
    default. If the splitter ignores max_chunk_size it emits ~1024-word chunks
    that then trip the ``token_count <= max_chunk_size`` guard in read() and
    raise ValueError. Before the fix this test raised; after, the splitter is
    bounded by the caller's budget.
    """
    monkeypatch.setattr(langchain_chunker_module, "get_vector_engine", _mock_get_vector_engine)

    async def get_text():
        # ~500 words, far above the small budget below.
        yield "word " * 500

    max_chunk_size = 32
    chunker = LangchainChunker(_make_document(), max_chunk_size=max_chunk_size, get_text=get_text)

    # The splitter itself is bounded by the unified budget, not the 1024 default.
    assert chunker.splitter._chunk_size <= max_chunk_size

    # read() must not raise, and every emitted chunk must respect the budget
    # (the mock tokenizer counts words, so chunk_size == word count here).
    chunks = [chunk async for chunk in chunker.read()]
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunk_size <= max_chunk_size
