"""Regression test: chunkers must propagate the document's importance_weight.

A ``DocumentChunk`` carries the ``importance_weight`` that later flows into every
``Entity``/``EntityType`` extracted from it (node weighting in the graph).
``TextChunker`` forwards ``self.document.importance_weight`` into each chunk, but
``TextChunkerWithOverlap`` and ``CsvChunker`` omitted it, so every chunk they
produced silently fell back to the ``DocumentChunk`` default (0.5) regardless of
the document's real weight. These chunkers now forward it like ``TextChunker``.
"""

from uuid import uuid4

import pytest

from cognee.modules.chunking.CsvChunker import CsvChunker
from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap
from cognee.modules.data.processing.document_types import Document


def _make_text_generator(*texts):
    async def gen():
        for text in texts:
            yield text

    return gen


def _document(importance_weight):
    return Document(
        id=uuid4(),
        name="weighted_document",
        raw_data_location="/test/path.csv",
        external_metadata=None,
        mime_type="text/plain",
        importance_weight=importance_weight,
    )


@pytest.mark.asyncio
async def test_overlap_chunker_propagates_importance_weight():
    document = _document(0.9)

    def controlled_chunk_data(_text):
        return [
            {"text": "alpha", "chunk_size": 5, "cut_type": "sentence", "chunk_id": uuid4()},
            {"text": "beta", "chunk_size": 4, "cut_type": "sentence", "chunk_id": uuid4()},
        ]

    chunker = TextChunkerWithOverlap(
        document,
        _make_text_generator("dummy"),
        max_chunk_size=20,
        get_chunk_data=controlled_chunk_data,
    )

    chunks = [chunk async for chunk in chunker.read()]

    assert chunks  # sanity: the chunker produced output
    assert all(chunk.importance_weight == 0.9 for chunk in chunks)


@pytest.mark.asyncio
async def test_csv_chunker_propagates_importance_weight():
    document = _document(0.9)
    chunker = CsvChunker(
        document,
        _make_text_generator("col_a,col_b\n1,2\n3,4\n"),
        max_chunk_size=4096,
    )

    chunks = [chunk async for chunk in chunker.read()]

    assert chunks
    assert all(chunk.importance_weight == 0.9 for chunk in chunks)
