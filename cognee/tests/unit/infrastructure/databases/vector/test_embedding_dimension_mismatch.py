"""Changing the embedding model under an existing collection says so.

A collection's vector column is a fixed-size list decided at creation, so a
changed EMBEDDING_MODEL breaks every later write and query. LanceDB reports
that as an Arrow cast failure ("Vector column 'vector' has variable length
vectors") that names neither the model nor the dataset, and reads as data
corruption rather than the config change it is. The path this covers is the
common one: ingest keylessly (local embedder, 384), configure a key, ingest
again (OpenAI, 3072).
"""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.vector.exceptions import EmbeddingDimensionMismatchError
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

STORED_DIMENSIONS = 384
CONFIGURED_DIMENSIONS = 3072


def adapter_with(dimensions: int, model: str | None = "openai/text-embedding-3-large"):
    """A bare adapter: the check reads only the engine, never a connection."""
    instance = LanceDBAdapter.__new__(LanceDBAdapter)
    instance.embedding_engine = SimpleNamespace(
        get_vector_size=lambda: dimensions,
        model=model,
    )
    return instance


class FakeCollection:
    """Enough of a LanceDB table to answer `await collection.schema()`."""

    def __init__(self, vector_type):
        self._vector_type = vector_type

    async def schema(self):
        return SimpleNamespace(field=lambda _name: SimpleNamespace(type=self._vector_type))


def fixed_size_vector(list_size: int):
    return SimpleNamespace(list_size=list_size)


@pytest.mark.asyncio
async def test_mismatch_names_both_widths_the_collection_and_the_model():
    adapter = adapter_with(CONFIGURED_DIMENSIONS)
    collection = FakeCollection(fixed_size_vector(STORED_DIMENSIONS))

    with pytest.raises(EmbeddingDimensionMismatchError) as error:
        await adapter._assert_embedding_dimensions(collection, "DocumentChunk_text")

    message = error.value.message
    assert "DocumentChunk_text" in message
    assert str(STORED_DIMENSIONS) in message
    assert str(CONFIGURED_DIMENSIONS) in message
    assert "openai/text-embedding-3-large" in message
    # The way out matters more than the diagnosis: re-embed, or go back.
    assert "memory_only=True" in error.value.remediation
    assert "EMBEDDING_MODEL" in error.value.remediation


@pytest.mark.asyncio
async def test_matching_dimensions_pass():
    adapter = adapter_with(STORED_DIMENSIONS)
    collection = FakeCollection(fixed_size_vector(STORED_DIMENSIONS))

    assert await adapter._assert_embedding_dimensions(collection, "DocumentChunk_text") is None


@pytest.mark.asyncio
async def test_an_unknown_model_still_reports_the_widths():
    adapter = adapter_with(CONFIGURED_DIMENSIONS, model=None)
    collection = FakeCollection(fixed_size_vector(STORED_DIMENSIONS))

    with pytest.raises(EmbeddingDimensionMismatchError) as error:
        await adapter._assert_embedding_dimensions(collection, "Entity_name")

    assert "the configured embedding model produces" in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collection",
    [
        # A variable-size list column: no list_size to compare against.
        FakeCollection(SimpleNamespace()),
        # A layout this cannot read at all.
        SimpleNamespace(),
    ],
    ids=["no_list_size", "no_schema"],
)
async def test_an_unreadable_schema_never_fails_the_operation(collection):
    """The check is diagnostic. It must not turn a working setup into an error."""
    adapter = adapter_with(CONFIGURED_DIMENSIONS)

    assert await adapter._assert_embedding_dimensions(collection, "Entity_name") is None
