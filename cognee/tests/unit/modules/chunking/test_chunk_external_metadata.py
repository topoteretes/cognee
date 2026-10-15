"""Document ``external_metadata`` reaches every chunk as JSON text.

``test_text_chunker.py`` covers the two text chunkers; this file covers the
helper's contract and the model-level derivation every construction site relies
on. Because ``DocumentChunk`` fills the field itself, one chunker is exercised
end to end as a smoke test rather than all five: they share the code path.
"""

from uuid import uuid4

import pytest

from cognee.modules.chunking.external_metadata import (
    decode_external_metadata,
    normalize_external_metadata,
)
from cognee.modules.chunking.JsonListChunker import JsonListChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document

STORED = '{"created_at": "2024-01-15"}'


def _document(metadata=STORED) -> Document:
    return Document(
        id=uuid4(),
        name="doc.txt",
        raw_data_location="/tmp/doc.txt",
        external_metadata=metadata,
        mime_type="text/plain",
    )


def _text(*texts):
    async def gen():
        for text in texts:
            yield text

    return gen


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("{}", None),
        ({}, None),
        ("not json", None),
        ('["a"]', None),
        ("42", None),
        (STORED, STORED),
        ({"created_at": "2024-01-15"}, STORED),
        ('{\n  "a": 1\n}', '{"a": 1}'),
    ],
)
def test_normalize_external_metadata(value, expected):
    """Ingest guard: anything that is not a non-empty JSON object becomes None."""
    assert normalize_external_metadata(value) == expected


def test_normalize_never_raises_on_unserialisable_values():
    assert normalize_external_metadata({"when": object()}) is not None  # default=str
    assert normalize_external_metadata({("tuple", "key"): 1}) is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("{oops", None),
        ('["a"]', None),
        (STORED, {"created_at": "2024-01-15"}),
        ({"created_at": "2024-01-15"}, {"created_at": "2024-01-15"}),
    ],
)
def test_decode_external_metadata(value, expected):
    assert decode_external_metadata(value) == expected


def _chunk(document, **overrides) -> DocumentChunk:
    fields = {
        "text": "t",
        "chunk_size": 1,
        "chunk_index": 0,
        "cut_type": "paragraph_end",
        "is_part_of": document,
        "contains": [],
    }
    fields.update(overrides)
    return DocumentChunk(**fields)


def test_chunk_derives_external_metadata_from_its_document():
    """Any construction site that passes is_part_of gets the copy for free."""
    assert _chunk(_document(STORED)).external_metadata == STORED
    assert _chunk(_document(None)).external_metadata is None
    assert _chunk(_document("{oops")).external_metadata is None


def test_chunk_keeps_an_explicit_external_metadata_value():
    """Rehydrate copies the stored node's value; the document must not override it."""
    chunk = _chunk(_document(STORED), external_metadata='{"stored": 1}')
    assert chunk.external_metadata == '{"stored": 1}'


def test_chunk_external_metadata_is_not_embedded():
    assert _chunk(_document(STORED)).metadata["index_fields"] == ["text"]


@pytest.mark.asyncio
async def test_a_chunker_stamps_every_chunk_it_cuts():
    """Smoke test over a real chunker; the validator covers the other four."""
    chunker = JsonListChunker(_document(), _text('[{"a": 1}, {"b": 2}]'), max_chunk_size=512)
    chunks = [chunk async for chunk in chunker.read()]

    assert len(chunks) == 2
    assert all(chunk.external_metadata == STORED for chunk in chunks)
