"""Unit tests for LangchainChunker."""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from cognee.modules.chunking.LangchainChunker import LangchainChunker
from cognee.modules.data.processing.document_types import Document


@pytest.fixture
def make_document():
    """Factory for Document instances."""

    def _factory():
        return Document(
            id=uuid4(),
            name="test_document",
            raw_data_location="/test/path",
            external_metadata=None,
            mime_type="text/plain",
        )

    return _factory


@pytest.fixture
def make_text_generator():
    """Factory for async text generators."""

    def _factory(*texts):
        async def gen():
            for text in texts:
                yield text

        return gen

    return _factory


@pytest.fixture
def mock_tokenizer():
    """Mock get_vector_engine with a tokenizer that returns 5 tokens per chunk."""
    tokenizer = MagicMock()
    tokenizer.count_tokens.return_value = 5
    engine = MagicMock()
    engine.embedding_engine.tokenizer = tokenizer
    with patch(
        "cognee.modules.chunking.LangchainChunker.get_vector_engine", return_value=engine
    ):
        yield tokenizer


async def collect_chunks(chunker):
    """Consume async generator and return list of chunks."""
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_empty_input_produces_no_chunks(make_document, make_text_generator, mock_tokenizer):
    """Empty string input should yield no chunks."""
    document = make_document()
    get_text = make_text_generator("")
    chunker = LangchainChunker(document, get_text, max_chunk_tokens=100)
    chunks = await collect_chunks(chunker)
    assert chunks == [], "Empty input should produce no chunks"


@pytest.mark.asyncio
async def test_basic_chunking_yields_chunks(make_document, make_text_generator, mock_tokenizer):
    """Non-empty text input should produce at least one chunk with correct attributes."""
    document = make_document()
    text = "Hello world. This is a test document. It has multiple sentences."
    get_text = make_text_generator(text)
    chunker = LangchainChunker(document, get_text, max_chunk_tokens=100, chunk_size=100)
    chunks = await collect_chunks(chunker)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.text, "Each chunk should have non-empty text"
        assert chunk.cut_type == "missing", "LangchainChunker always sets cut_type to 'missing'"
        assert chunk.chunk_size == len(chunk.text.split()), "chunk_size should equal word count"


@pytest.mark.asyncio
async def test_chunk_indices_are_sequential(make_document, make_text_generator, mock_tokenizer):
    """Chunk indices should start at 0 and increment sequentially."""
    document = make_document()
    text = " ".join([f"word{i}" for i in range(100)])
    get_text = make_text_generator(text)
    chunker = LangchainChunker(
        document, get_text, max_chunk_tokens=100, chunk_size=5, chunk_overlap=0
    )
    chunks = await collect_chunks(chunker)
    assert len(chunks) > 1, "Small chunk_size should produce multiple chunks"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks))), (
        "Chunk indices must be sequential starting from 0"
    )


@pytest.mark.asyncio
async def test_exceeding_max_chunk_tokens_raises_value_error(
    make_document, make_text_generator
):
    """When token count exceeds max_chunk_tokens, ValueError should be raised."""
    document = make_document()
    text = "This is a test sentence with several words."
    get_text = make_text_generator(text)
    chunker = LangchainChunker(document, get_text, max_chunk_tokens=3, chunk_size=100)

    tokenizer = MagicMock()
    tokenizer.count_tokens.return_value = 10  # exceeds max_chunk_tokens=3
    engine = MagicMock()
    engine.embedding_engine.tokenizer = tokenizer

    with patch(
        "cognee.modules.chunking.LangchainChunker.get_vector_engine", return_value=engine
    ):
        with pytest.raises(ValueError, match="tokens is larger than the maximum"):
            await collect_chunks(chunker)


@pytest.mark.asyncio
async def test_chunk_ids_are_deterministic(make_document, make_text_generator, mock_tokenizer):
    """Running chunker twice on identical input should produce identical chunk IDs."""
    document = make_document()
    text = "Deterministic chunking test sentence with enough words to form a chunk."

    get_text1 = make_text_generator(text)
    chunker1 = LangchainChunker(document, get_text1, max_chunk_tokens=100, chunk_size=100)
    chunks1 = await collect_chunks(chunker1)

    get_text2 = make_text_generator(text)
    chunker2 = LangchainChunker(document, get_text2, max_chunk_tokens=100, chunk_size=100)
    chunks2 = await collect_chunks(chunker2)

    assert len(chunks1) == len(chunks2), "Same input should produce same number of chunks"
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.id == c2.id, "Chunk IDs should be deterministic for identical content"


@pytest.mark.asyncio
async def test_chunk_ids_are_unique_within_a_run(make_document, make_text_generator, mock_tokenizer):
    """Different chunks within the same run should have different IDs."""
    document = make_document()
    text = " ".join([f"unique_word_{i}" for i in range(20)])
    get_text = make_text_generator(text)
    chunker = LangchainChunker(
        document, get_text, max_chunk_tokens=100, chunk_size=5, chunk_overlap=0
    )
    chunks = await collect_chunks(chunker)
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids), "All chunk IDs must be unique within a run"


@pytest.mark.asyncio
async def test_unicode_input_is_handled(make_document, make_text_generator, mock_tokenizer):
    """Unicode and emoji text should be chunked without errors."""
    document = make_document()
    text = "Japanese: 日本語テスト. Emoji: 🚀🎯. Arabic: مرحبا. Greek: γεια σας."
    get_text = make_text_generator(text)
    chunker = LangchainChunker(document, get_text, max_chunk_tokens=100, chunk_size=100)
    chunks = await collect_chunks(chunker)
    assert len(chunks) >= 1, "Unicode input should produce at least one chunk"
    all_text = " ".join(c.text for c in chunks)
    assert "日本語" in all_text or "Arabic" in all_text, "Unicode content should be preserved"


@pytest.mark.asyncio
async def test_multiple_text_segments_are_all_processed(
    make_document, make_text_generator, mock_tokenizer
):
    """All text segments yielded by get_text() should contribute to output chunks."""
    document = make_document()
    segment1 = "First segment of text with several words for testing."
    segment2 = "Second segment of text with several words for testing."
    get_text = make_text_generator(segment1, segment2)
    chunker = LangchainChunker(document, get_text, max_chunk_tokens=100, chunk_size=100)
    chunks = await collect_chunks(chunker)
    all_text = " ".join(c.text for c in chunks)
    assert "First segment" in all_text, "First text segment should appear in chunks"
    assert "Second segment" in all_text, "Second text segment should appear in chunks"


@pytest.mark.asyncio
async def test_chunk_is_part_of_document(make_document, make_text_generator, mock_tokenizer):
    """Each chunk's is_part_of should reference the original document."""
    document = make_document()
    text = "Some text to chunk for document reference testing."
    get_text = make_text_generator(text)
    chunker = LangchainChunker(document, get_text, max_chunk_tokens=100, chunk_size=100)
    chunks = await collect_chunks(chunker)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.is_part_of.id == document.id, (
            "Each chunk's is_part_of should reference the source document"
        )
