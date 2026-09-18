"""PGVector reports a changed embedding model the same way LanceDB does.

A ``vector(N)`` column fixes N at creation, so switching EMBEDDING_MODEL breaks
writes and queries with a Postgres error that names neither the model nor the
dataset. The width is read from the catalog rather than from the adapter's
reflection cache, which nothing in production invalidates — a stale entry would
block the re-index this error tells the user to run.
"""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.vector.exceptions import EmbeddingDimensionMismatchError
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

STORED_DIMENSIONS = 384
CONFIGURED_DIMENSIONS = 3072


class FakeSession:
    """Answers the one catalog query the check makes."""

    def __init__(self, declared_type, recorder: dict):
        self._declared_type = declared_type
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, statement, parameters=None):
        self._recorder["sql"] = str(statement)
        self._recorder["parameters"] = parameters
        return SimpleNamespace(scalar_one_or_none=lambda: self._declared_type)


def adapter_with(declared_type, *, schema: str = "", dimensions=CONFIGURED_DIMENSIONS):
    recorder: dict = {}
    instance = PGVectorAdapter.__new__(PGVectorAdapter)
    instance.schema = schema
    instance.embedding_engine = SimpleNamespace(
        get_vector_size=lambda: dimensions,
        model="openai/text-embedding-3-large",
    )
    instance.get_async_session = lambda: FakeSession(declared_type, recorder)
    return instance, recorder


@pytest.mark.asyncio
async def test_mismatch_names_both_widths_and_the_collection():
    adapter, _ = adapter_with(f"vector({STORED_DIMENSIONS})")

    with pytest.raises(EmbeddingDimensionMismatchError) as error:
        await adapter._assert_embedding_dimensions("DocumentChunk_text")

    message = error.value.message
    assert "DocumentChunk_text" in message
    assert str(STORED_DIMENSIONS) in message
    assert str(CONFIGURED_DIMENSIONS) in message
    assert "memory_only=True" in error.value.remediation


@pytest.mark.asyncio
async def test_matching_dimensions_pass():
    adapter, _ = adapter_with(f"vector({CONFIGURED_DIMENSIONS})")

    assert await adapter._assert_embedding_dimensions("DocumentChunk_text") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "declared_type",
    [None, "vector", "jsonb", "", "vector(abc)"],
    ids=["no_such_column", "unconstrained", "wrong_type", "empty", "unparseable"],
)
async def test_an_unreadable_column_never_fails_the_operation(declared_type):
    """The check is diagnostic. It must not turn a working setup into an error."""
    adapter, _ = adapter_with(declared_type)

    assert await adapter._assert_embedding_dimensions("Entity_name") is None


@pytest.mark.asyncio
async def test_a_failing_catalog_query_never_fails_the_operation():
    adapter, _ = adapter_with(f"vector({STORED_DIMENSIONS})")

    def explode():
        raise RuntimeError("connection gone")

    adapter.get_async_session = explode

    assert await adapter._assert_embedding_dimensions("Entity_name") is None


@pytest.mark.asyncio
async def test_the_collection_is_quoted_and_schema_qualified():
    """Cognee's collection names are mixed-case, so an unquoted name would not resolve."""
    adapter, recorder = adapter_with(f"vector({CONFIGURED_DIMENSIONS})", schema="dataset_42")

    await adapter._assert_embedding_dimensions("DocumentChunk_text")

    assert recorder["parameters"] == {"qualified_name": '"dataset_42"."DocumentChunk_text"'}
    # Passed to to_regclass as a bind parameter, never interpolated into the SQL.
    assert "DocumentChunk_text" not in recorder["sql"]


@pytest.mark.asyncio
async def test_an_unpinned_schema_uses_the_bare_quoted_name():
    adapter, recorder = adapter_with(f"vector({CONFIGURED_DIMENSIONS})")

    await adapter._assert_embedding_dimensions("DocumentChunk_text")

    assert recorder["parameters"] == {"qualified_name": '"DocumentChunk_text"'}
