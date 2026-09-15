"""Document ``external_metadata`` reaches every chunk as JSON text.

``test_text_chunker.py`` covers the two text chunkers; this file covers the
helper's contract, the model-level derivation every construction site relies
on, and the remaining chunkers (CSV, JSON list, langchain).
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.tokenizer.TikToken.adapter import TikTokenTokenizer
from cognee.modules.chunking.CsvChunker import CsvChunker
from cognee.modules.chunking.external_metadata import (
    document_external_metadata,
    normalize_external_metadata,
    parse_external_metadata,
)
from cognee.modules.chunking.JsonListChunker import JsonListChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document

_chunk_by_row = importlib.import_module("cognee.tasks.chunks.chunk_by_row")

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


async def _collect(chunker):
    return [chunk async for chunk in chunker.read()]


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
    assert normalize_external_metadata(value) == expected


def test_normalize_never_raises_on_unserialisable_values():
    assert normalize_external_metadata({"when": object()}) is not None  # default=str
    assert normalize_external_metadata({("tuple", "key"): 1}) is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("{}", None),
        ("{oops", None),
        ('["a"]', None),
        (STORED, {"created_at": "2024-01-15"}),
        ({"created_at": "2024-01-15"}, {"created_at": "2024-01-15"}),
    ],
)
def test_parse_external_metadata(value, expected):
    assert parse_external_metadata(value) == expected


def test_document_helper_reads_the_document_field():
    assert document_external_metadata(_document(STORED)) == STORED
    assert document_external_metadata(_document(None)) is None
    assert document_external_metadata(SimpleNamespace()) is None


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
async def test_json_list_chunker_stamps_every_item():
    chunks = await _collect(
        JsonListChunker(_document(), _text('[{"a": 1}, {"b": 2}]'), max_chunk_size=512)
    )

    assert len(chunks) == 2
    assert all(chunk.external_metadata == STORED for chunk in chunks)


@pytest.mark.asyncio
async def test_csv_chunker_stamps_every_row():
    engine = SimpleNamespace(
        tokenizer=TikTokenTokenizer(model="text-embedding-3-large", max_completion_tokens=8191)
    )
    with patch.object(_chunk_by_row, "get_embedding_engine", return_value=engine):
        chunks = await _collect(
            CsvChunker(_document(), _text("name: John\n\nname: Jane"), max_chunk_size=512)
        )

    assert len(chunks) == 2
    assert all(chunk.external_metadata == STORED for chunk in chunks)


@pytest.mark.asyncio
async def test_langchain_chunker_stamps_every_chunk():
    pytest.importorskip("langchain_text_splitters")
    from cognee.modules.chunking.LangchainChunker import LangchainChunker

    engine = SimpleNamespace(
        embedding_engine=SimpleNamespace(
            tokenizer=SimpleNamespace(count_tokens=lambda text: len(text.split()))
        )
    )
    chunker = LangchainChunker(_document(), max_chunk_size=512, get_text=_text("one two three"))
    with patch(
        "cognee.modules.chunking.LangchainChunker.get_vector_engine_async",
        new=AsyncMock(return_value=engine),
    ):
        chunks = await _collect(chunker)

    assert chunks
    assert all(chunk.external_metadata == STORED for chunk in chunks)
