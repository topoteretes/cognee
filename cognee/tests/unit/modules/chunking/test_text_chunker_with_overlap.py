"""Unit tests for TextChunkerWithOverlap overlap behavior."""

import sys
import pytest
from uuid import uuid4
from unittest.mock import patch

from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap
from cognee.modules.data.processing.document_types import Document
from cognee.tasks.chunks import chunk_by_paragraph


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
def make_controlled_chunk_data():
    """Factory for controlled chunk_data generators."""

    def _factory(*sentences, chunk_size_per_sentence=10):
        def _chunk_data(text):
            return [
                {
                    "text": sentence,
                    "chunk_size": chunk_size_per_sentence,
                    "cut_type": "sentence",
                    "chunk_id": uuid4(),
                }
                for sentence in sentences
            ]

        return _chunk_data

    return _factory


@pytest.mark.asyncio
async def test_half_overlap_preserves_content_across_chunks(
    make_text_generator, make_controlled_chunk_data
):
    """With 50% overlap, consecutive chunks should share half their content."""
    s1 = "one"
    s2 = "two"
    s3 = "three"
    s4 = "four"
    text = "dummy"
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    get_chunk_data = make_controlled_chunk_data(s1, s2, s3, s4, chunk_size_per_sentence=10)
    chunker = TextChunkerWithOverlap(
        document,
        get_text,
        max_chunk_size=20,
        chunk_overlap_ratio=0.5,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 3, "Should produce exactly 3 chunks (s1+s2, s2+s3, s3+s4)"
    assert [c.chunk_index for c in chunks] == [0, 1, 2], "Chunk indices should be [0, 1, 2]"
    assert "one" in chunks[0].text and "two" in chunks[0].text, "Chunk 0 should contain s1 and s2"
    assert "two" in chunks[1].text and "three" in chunks[1].text, (
        "Chunk 1 should contain s2 (overlap) and s3"
    )
    assert "three" in chunks[2].text and "four" in chunks[2].text, (
        "Chunk 2 should contain s3 (overlap) and s4"
    )


@pytest.mark.asyncio
async def test_zero_overlap_produces_no_duplicate_content(
    make_text_generator, make_controlled_chunk_data
):
    """With 0% overlap, no content should appear in multiple chunks."""
    s1 = "one"
    s2 = "two"
    s3 = "three"
    s4 = "four"
    text = "dummy"
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    get_chunk_data = make_controlled_chunk_data(s1, s2, s3, s4, chunk_size_per_sentence=10)
    chunker = TextChunkerWithOverlap(
        document,
        get_text,
        max_chunk_size=20,
        chunk_overlap_ratio=0.0,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 2, "Should produce exactly 2 chunks (s1+s2, s3+s4)"
    assert "one" in chunks[0].text and "two" in chunks[0].text, (
        "First chunk should contain s1 and s2"
    )
    assert "three" in chunks[1].text and "four" in chunks[1].text, (
        "Second chunk should contain s3 and s4"
    )
    assert "two" not in chunks[1].text and "three" not in chunks[0].text, (
        "No overlap: end of chunk 0 should not appear in chunk 1"
    )


@pytest.mark.asyncio
async def test_small_overlap_ratio_creates_minimal_overlap(
    make_text_generator, make_controlled_chunk_data
):
    """With 25% overlap ratio, chunks should have minimal overlap."""
    s1 = "alpha"
    s2 = "beta"
    s3 = "gamma"
    s4 = "delta"
    s5 = "epsilon"
    text = "dummy"
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    get_chunk_data = make_controlled_chunk_data(s1, s2, s3, s4, s5, chunk_size_per_sentence=10)
    chunker = TextChunkerWithOverlap(
        document,
        get_text,
        max_chunk_size=40,
        chunk_overlap_ratio=0.25,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 2, "Should produce exactly 2 chunks"
    assert [c.chunk_index for c in chunks] == [0, 1], "Chunk indices should be [0, 1]"
    assert all(token in chunks[0].text for token in [s1, s2, s3, s4]), (
        "Chunk 0 should contain s1 through s4"
    )
    assert s4 in chunks[1].text and s5 in chunks[1].text, (
        "Chunk 1 should contain overlap s4 and new content s5"
    )


@pytest.mark.asyncio
async def test_high_overlap_ratio_creates_significant_overlap(
    make_text_generator, make_controlled_chunk_data
):
    """With 75% overlap ratio, consecutive chunks should share most content."""
    s1 = "red"
    s2 = "blue"
    s3 = "green"
    s4 = "yellow"
    s5 = "purple"
    text = "dummy"
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    get_chunk_data = make_controlled_chunk_data(s1, s2, s3, s4, s5, chunk_size_per_sentence=5)
    chunker = TextChunkerWithOverlap(
        document,
        get_text,
        max_chunk_size=20,
        chunk_overlap_ratio=0.75,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 2, "Should produce exactly 2 chunks with 75% overlap"
    assert [c.chunk_index for c in chunks] == [0, 1], "Chunk indices should be [0, 1]"
    assert all(token in chunks[0].text for token in [s1, s2, s3, s4]), (
        "Chunk 0 should contain s1, s2, s3, s4"
    )
    assert all(token in chunks[1].text for token in [s2, s3, s4, s5]), (
        "Chunk 1 should contain s2, s3, s4 (overlap) and s5"
    )


@pytest.mark.asyncio
async def test_single_chunk_no_dangling_overlap(make_text_generator, make_controlled_chunk_data):
    """Text that fits in one chunk should produce exactly one chunk, no overlap artifact."""
    s1 = "alpha"
    s2 = "beta"
    text = "dummy"
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    get_chunk_data = make_controlled_chunk_data(s1, s2, chunk_size_per_sentence=10)
    chunker = TextChunkerWithOverlap(
        document,
        get_text,
        max_chunk_size=20,
        chunk_overlap_ratio=0.5,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 1, (
        "Should produce exactly 1 chunk when content fits within max_chunk_size"
    )
    assert chunks[0].chunk_index == 0, "Single chunk should have index 0"
    assert "alpha" in chunks[0].text and "beta" in chunks[0].text, (
        "Single chunk should contain all content"
    )


@pytest.mark.asyncio
async def test_paragraph_chunking_with_overlap(make_text_generator):
    """Test that chunk_by_paragraph integration produces 25% overlap between chunks."""

    def mock_get_embedding_engine():
        class MockEngine:
            tokenizer = None

        return MockEngine()

    chunk_by_sentence_module = sys.modules.get("cognee.tasks.chunks.chunk_by_sentence")

    max_chunk_size = 20
    overlap_ratio = 0.25  # 5 token overlap
    paragraph_max_size = int(0.5 * overlap_ratio * max_chunk_size)  # = 2

    text = (
        "A0 A1. A2 A3. A4 A5. A6 A7. A8 A9. "  # 10 tokens (0-9)
        "B0 B1. B2 B3. B4 B5. B6 B7. B8 B9. "  # 10 tokens (10-19)
        "C0 C1. C2 C3. C4 C5. C6 C7. C8 C9. "  # 10 tokens (20-29)
        "D0 D1. D2 D3. D4 D5. D6 D7. D8 D9. "  # 10 tokens (30-39)
        "E0 E1. E2 E3. E4 E5. E6 E7. E8 E9."  # 10 tokens (40-49)
    )

    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )

    get_text = make_text_generator(text)

    def get_chunk_data(text_input):
        return chunk_by_paragraph(
            text_input, max_chunk_size=paragraph_max_size, batch_paragraphs=True
        )

    with patch.object(
        chunk_by_sentence_module, "get_embedding_engine", side_effect=mock_get_embedding_engine
    ):
        chunker = TextChunkerWithOverlap(
            document,
            get_text,
            max_chunk_size=max_chunk_size,
            chunk_overlap_ratio=overlap_ratio,
            get_chunk_data=get_chunk_data,
        )
        chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 3, f"Should produce exactly 3 chunks, got {len(chunks)}"

    assert chunks[0].chunk_index == 0, "First chunk should have index 0"
    assert chunks[1].chunk_index == 1, "Second chunk should have index 1"
    assert chunks[2].chunk_index == 2, "Third chunk should have index 2"

    assert "A0" in chunks[0].text, "Chunk 0 should start with A0"
    assert "A9" in chunks[0].text, "Chunk 0 should contain A9"
    assert "B0" in chunks[0].text, "Chunk 0 should contain B0"
    assert "B9" in chunks[0].text, "Chunk 0 should contain up to B9 (20 tokens)"

    assert "B" in chunks[1].text, "Chunk 1 should have overlap from B section"
    assert "C" in chunks[1].text, "Chunk 1 should contain C section"
    assert "D" in chunks[1].text, "Chunk 1 should contain D section"

    assert "D" in chunks[2].text, "Chunk 2 should have overlap from D section"
    assert "E0" in chunks[2].text, "Chunk 2 should contain E0"
    assert "E9" in chunks[2].text, "Chunk 2 should end with E9"

    chunk_0_end_words = chunks[0].text.split()[-4:]
    chunk_1_words = chunks[1].text.split()
    overlap_0_1 = any(word in chunk_1_words for word in chunk_0_end_words)
    assert overlap_0_1, (
        f"No overlap detected between chunks 0 and 1. "
        f"Chunk 0 ends with: {chunk_0_end_words}, "
        f"Chunk 1 starts with: {chunk_1_words[:6]}"
    )

    chunk_1_end_words = chunks[1].text.split()[-4:]
    chunk_2_words = chunks[2].text.split()
    overlap_1_2 = any(word in chunk_2_words for word in chunk_1_end_words)
    assert overlap_1_2, (
        f"No overlap detected between chunks 1 and 2. "
        f"Chunk 1 ends with: {chunk_1_end_words}, "
        f"Chunk 2 starts with: {chunk_2_words[:6]}"
    )
