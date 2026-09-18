"""Tests for DefaultChunkEngine.chunk_by_sentence output shape.

When a sentence exceeded chunk_size, chunk_by_sentence called chunk_data_exact
(which returns a ``(chunks, numbered_chunks)`` tuple) and `extend`ed the sentence
list with the whole tuple. That spliced in the chunk list and the numbered-chunk
list as two elements, so an over-long sentence yielded nested lists instead of
plain chunk strings. Every returned chunk must be a string.
"""

from __future__ import annotations

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy


def _engine() -> DefaultChunkEngine:
    return DefaultChunkEngine(ChunkStrategy.SENTENCE, chunk_size=10, chunk_overlap=2)


def test_long_sentence_produces_only_string_chunks():
    text = "This sentence is definitely much longer than ten characters. Short one."
    chunks, numbered = _engine().chunk_by_sentence([text], chunk_size=10, chunk_overlap=2)

    assert all(isinstance(chunk, str) for chunk in chunks), (
        f"expected only string chunks, got types {[type(c).__name__ for c in chunks]}"
    )
    # The short sentence survives intact as its own chunk.
    assert "Short one." in chunks
    # numbered entries pair an index with the same string chunk.
    assert all(isinstance(entry[1], str) and entry[0] == i + 1 for i, entry in enumerate(numbered))


def test_short_sentences_are_kept_whole():
    text = "One. Two. Three."
    chunks, _ = _engine().chunk_by_sentence([text], chunk_size=100, chunk_overlap=2)
    assert chunks == ["One.", "Two.", "Three."]
