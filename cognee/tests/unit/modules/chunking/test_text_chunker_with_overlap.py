"""Unit tests for TextChunkerWithOverlap overlap behavior."""

import pytest
from uuid import uuid4

from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap
from cognee.modules.data.processing.document_types import Document


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

    assert len(chunks) == 3, "Should produce exactly 3 chunks (s1+s2, overlap s2, s3+s4)"
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
    assert "three" not in chunks[0].text and "one" not in chunks[1].text, (
        "No overlap between chunks"
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
        max_chunk_size=30,
        chunk_overlap_ratio=0.25,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 2, "Should produce exactly 2 chunks"
    assert [c.chunk_index for c in chunks] == [0, 1], "Chunk indices should be [0, 1]"
    assert "alpha" in chunks[0].text and "beta" in chunks[0].text and "gamma" in chunks[0].text, (
        "Chunk 0 should contain s1, s2, s3"
    )
    assert chunks[1].text == "delta", (
        "Chunk 1 should contain s4 only (overlap too small for full sentence)"
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
    text = "dummy"
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    get_chunk_data = make_controlled_chunk_data(s1, s2, s3, s4, chunk_size_per_sentence=5)
    chunker = TextChunkerWithOverlap(
        document,
        get_text,
        max_chunk_size=15,
        chunk_overlap_ratio=0.75,
        get_chunk_data=get_chunk_data,
    )
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 2, "Should produce exactly 2 chunks with 75% overlap"
    assert [c.chunk_index for c in chunks] == [0, 1], "Chunk indices should be [0, 1]"
    assert "red" in chunks[0].text and "blue" in chunks[0].text and "green" in chunks[0].text, (
        "Chunk 0 should contain s1, s2, s3"
    )
    assert "blue" in chunks[1].text and "green" in chunks[1].text and "yellow" in chunks[1].text, (
        "Chunk 1 should contain s2, s3, s4 (75% overlap)"
    )
