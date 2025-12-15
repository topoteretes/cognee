"""Unit tests for TextChunker and TextChunkerWithOverlap behavioral equivalence."""

import pytest
from uuid import uuid4

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap
from cognee.modules.data.processing.document_types import Document


@pytest.fixture(params=["TextChunker", "TextChunkerWithOverlap"])
def chunker_class(request):
    """Parametrize tests to run against both implementations."""
    return TextChunker if request.param == "TextChunker" else TextChunkerWithOverlap


@pytest.fixture
def make_text_generator():
    """Factory for async text generators."""

    def _factory(*texts):
        async def gen():
            for text in texts:
                yield text

        return gen

    return _factory


async def collect_chunks(chunker):
    """Consume async generator and return list of chunks."""
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_empty_input_produces_no_chunks(chunker_class, make_text_generator):
    """Empty input should yield no chunks."""
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator("")
    chunker = chunker_class(document, get_text, max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 0, "Empty input should produce no chunks"


@pytest.mark.asyncio
async def test_whitespace_only_input_emits_single_chunk(chunker_class, make_text_generator):
    """Whitespace-only input should produce exactly one chunk with unchanged text."""
    whitespace_text = "   \n\t   \r\n   "
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(whitespace_text)
    chunker = chunker_class(document, get_text, max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 1, "Whitespace-only input should produce exactly one chunk"
    assert chunks[0].text == whitespace_text, "Chunk text should equal input (whitespace preserved)"
    assert chunks[0].chunk_index == 0, "First chunk should have index 0"


@pytest.mark.asyncio
async def test_single_paragraph_below_limit_emits_one_chunk(chunker_class, make_text_generator):
    """Single paragraph below limit should emit exactly one chunk."""
    text = "This is a short paragraph."
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    chunker = chunker_class(document, get_text, max_chunk_size=512)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 1, "Single short paragraph should produce exactly one chunk"
    assert chunks[0].text == text, "Chunk text should match input"
    assert chunks[0].chunk_index == 0, "First chunk should have index 0"
    assert chunks[0].chunk_size > 0, "Chunk should have positive size"


@pytest.mark.asyncio
async def test_oversized_paragraph_gets_emitted_as_a_single_chunk(
    chunker_class, make_text_generator
):
    """Oversized paragraph from chunk_by_paragraph should be emitted as single chunk."""
    text = ("A" * 1500) + ". Next sentence."
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    chunker = chunker_class(document, get_text, max_chunk_size=50)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2, "Should produce 2 chunks (oversized paragraph + next sentence)"
    assert chunks[0].chunk_size > 50, "First chunk should be oversized"
    assert chunks[0].chunk_index == 0, "First chunk should have index 0"
    assert chunks[1].chunk_index == 1, "Second chunk should have index 1"


@pytest.mark.asyncio
async def test_overflow_on_next_paragraph_emits_separate_chunk(chunker_class, make_text_generator):
    """First paragraph near limit plus small paragraph should produce two separate chunks."""
    first_para = " ".join(["word"] * 5)
    second_para = "Short text."
    text = first_para + " " + second_para
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    chunker = chunker_class(document, get_text, max_chunk_size=10)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 2, "Should produce 2 chunks due to overflow"
    assert chunks[0].text.strip() == first_para, "First chunk should contain only first paragraph"
    assert chunks[1].text.strip() == second_para, (
        "Second chunk should contain only second paragraph"
    )
    assert chunks[0].chunk_index == 0, "First chunk should have index 0"
    assert chunks[1].chunk_index == 1, "Second chunk should have index 1"


@pytest.mark.asyncio
async def test_small_paragraphs_batch_correctly(chunker_class, make_text_generator):
    """Multiple small paragraphs should batch together with joiner spaces counted."""
    paragraphs = [" ".join(["word"] * 12) for _ in range(40)]
    text = " ".join(paragraphs)
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text = make_text_generator(text)
    chunker = chunker_class(document, get_text, max_chunk_size=49)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 20, (
        "Should batch paragraphs (2 per chunk: 12 words × 2 tokens = 24, 24 + 1 joiner + 24 = 49)"
    )
    assert all(c.chunk_index == i for i, c in enumerate(chunks)), (
        "Chunk indices should be sequential"
    )
    all_text = " ".join(chunk.text.strip() for chunk in chunks)
    expected_text = " ".join(paragraphs)
    assert all_text == expected_text, "All paragraph text should be preserved with correct spacing"


@pytest.mark.asyncio
async def test_alternating_large_and_small_paragraphs_dont_batch(
    chunker_class, make_text_generator
):
    """Alternating near-max and small paragraphs should each become separate chunks."""
    large1 = "word" * 15 + "."
    small1 = "Short."
    large2 = "word" * 15 + "."
    small2 = "Tiny."
    text = large1 + " " + small1 + " " + large2 + " " + small2
    document = Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    max_chunk_size = 10
    get_text = make_text_generator(text)
    chunker = chunker_class(document, get_text, max_chunk_size=max_chunk_size)
    chunks = await collect_chunks(chunker)

    assert len(chunks) == 4, "Should produce multiple chunks"
    assert all(c.chunk_index == i for i, c in enumerate(chunks)), (
        "Chunk indices should be sequential"
    )
    assert chunks[0].chunk_size > max_chunk_size, (
        "First chunk should be oversized (large paragraph)"
    )
    assert chunks[1].chunk_size <= max_chunk_size, "Second chunk should be small (small paragraph)"
    assert chunks[2].chunk_size > max_chunk_size, (
        "Third chunk should be oversized (large paragraph)"
    )
    assert chunks[3].chunk_size <= max_chunk_size, "Fourth chunk should be small (small paragraph)"


@pytest.mark.asyncio
async def test_chunk_indices_and_ids_are_deterministic(chunker_class, make_text_generator):
    """Running chunker twice on identical input should produce identical indices and IDs."""
    sentence1 = "one " * 4 + ". "
    sentence2 = "two " * 4 + ". "
    sentence3 = "one " * 4 + ". "
    sentence4 = "two " * 4 + ". "
    text = sentence1 + sentence2 + sentence3 + sentence4
    doc_id = uuid4()
    max_chunk_size = 20

    document1 = Document(
        id=doc_id,
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text1 = make_text_generator(text)
    chunker1 = chunker_class(document1, get_text1, max_chunk_size=max_chunk_size)
    chunks1 = await collect_chunks(chunker1)

    document2 = Document(
        id=doc_id,
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )
    get_text2 = make_text_generator(text)
    chunker2 = chunker_class(document2, get_text2, max_chunk_size=max_chunk_size)
    chunks2 = await collect_chunks(chunker2)

    assert len(chunks1) == 2, "Should produce exactly 2 chunks (4 sentences, 2 per chunk)"
    assert len(chunks2) == 2, "Should produce exactly 2 chunks (4 sentences, 2 per chunk)"
    assert [c.chunk_index for c in chunks1] == [0, 1], "First run indices should be [0, 1]"
    assert [c.chunk_index for c in chunks2] == [0, 1], "Second run indices should be [0, 1]"
    assert chunks1[0].id == chunks2[0].id, "First chunk ID should be deterministic"
    assert chunks1[1].id == chunks2[1].id, "Second chunk ID should be deterministic"
    assert chunks1[0].id != chunks1[1].id, "Chunk IDs should be unique within a run"
