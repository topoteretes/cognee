"""LanceDB write failures from a vector-dimension mismatch surface a clear error.

A collection's ``vector`` column is a fixed-size list, so an upsert at a
different embedding width fails inside the Arrow/Lance writer with a message
that names neither the collection nor the cause (issue #4364 / #4313). These
tests pin the adapter's diagnosis: a schema-confirmed mismatch raises
``VectorDimensionMismatchError`` (preserving the original failure as its cause),
while a same-width failure keeps the original exception.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.exceptions import CogneeApiError

try:
    from cognee.infrastructure.databases.vector.exceptions import (
        VectorDimensionMismatchError,
    )
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_vector_size(self):
        return self.dimension

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1] * self.dimension for _ in texts]


def _point(text: str) -> IndexSchema:
    return IndexSchema(id=str(uuid4()), text=text, name="")


def _adapter(url: str, dimension: int) -> LanceDBAdapter:
    return LanceDBAdapter(url=url, api_key=None, embedding_engine=_FakeEmbeddingEngine(dimension))


class _FailingMergeInsert:
    def __init__(self, error: Exception):
        self._error = error

    def when_matched_update_all(self):
        return self

    def when_not_matched_insert_all(self):
        return self

    async def execute(self, _records):
        raise self._error


class _WriteFailingCollection:
    """Wrap a real table so only ``merge_insert`` fails, as a mid-write error."""

    def __init__(self, collection, error: Exception):
        self._collection = collection
        self._error = error

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def merge_insert(self, _key):
        return _FailingMergeInsert(self._error)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_dimension_mismatch_write_raises_clear_error(tmp_path):
    collection = "Entity_name"
    url = str(tmp_path / "db")

    builder = _adapter(url, 384)
    await builder.create_collection(collection, IndexSchema)
    await builder.create_data_points(collection, [_point("built at 384")])

    with pytest.raises(VectorDimensionMismatchError) as raised:
        await _adapter(url, 512).create_data_points(collection, [_point("written at 512")])

    error = raised.value
    assert isinstance(error, CogneeApiError)
    assert error.status_code == 409
    assert error.stored_dimensions == 384
    assert error.incoming_dimensions == 512
    assert "Entity_name" in str(error)
    assert error.remediation
    # The opaque LanceDB failure is preserved, not swallowed.
    assert error.__cause__ is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_same_width_write_succeeds(tmp_path):
    collection = "Entity_name"
    url = str(tmp_path / "db")

    adapter = _adapter(url, 384)
    await adapter.create_collection(collection, IndexSchema)
    await adapter.create_data_points(collection, [_point("first")])
    await adapter.create_data_points(collection, [_point("second")])

    rows = await (await adapter.get_collection(collection)).count_rows()
    assert rows == 2


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_same_width_write_failure_keeps_original_error(tmp_path, monkeypatch):
    collection = "Entity_name"
    url = str(tmp_path / "db")

    adapter = _adapter(url, 384)
    await adapter.create_collection(collection, IndexSchema)

    real_get_collection = adapter.get_collection

    async def _get_collection(name):
        return _WriteFailingCollection(
            await real_get_collection(name), RuntimeError("disk on fire")
        )

    monkeypatch.setattr(adapter, "get_collection", _get_collection)

    with pytest.raises(RuntimeError, match="disk on fire"):
        await adapter.create_data_points(collection, [_point("boom")])
