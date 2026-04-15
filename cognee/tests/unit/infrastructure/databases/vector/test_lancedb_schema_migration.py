"""Tests for LanceDB schema migration when the payload schema evolves."""

from __future__ import annotations

import pyarrow as pa
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, Mock
from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        LanceDBAdapter,
        IndexSchema,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_point(id: str, text: str) -> IndexSchema:
    return IndexSchema(id=id, text=text)


class _InheritedFieldsPoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


async def _seed(adapter, name, points):
    """Create collection then insert — mirrors production flow."""
    await adapter.create_collection(name, type(points[0]))
    await adapter.create_data_points(name, points)


async def _strip_payload_field(adapter, name: str) -> str:
    """Remove the last non-essential field from the payload struct,
    simulating a table created before the model gained that field."""
    table = await (await adapter.get_collection(name)).to_arrow()
    pi = table.schema.get_field_index("payload")
    pt = table.schema.field(pi).type

    removable = None
    for i in range(pt.num_fields):
        if pt.field(i).name not in ("id", "text"):
            removable = i
    assert removable is not None

    removed = pt.field(removable).name
    keep = [j for j in range(pt.num_fields) if j != removable]
    fields = [pt.field(j) for j in keep]
    col = table.column("payload").combine_chunks()
    arrays = [col.field(pt.field(j).name) for j in keep]

    narrowed = table.set_column(
        pi,
        pa.field("payload", pa.struct(fields)),
        pa.StructArray.from_arrays(arrays, fields=fields),
    )
    conn = await adapter.get_connection()
    await conn.drop_table(name)
    await conn.create_table(name, data=narrowed)
    return removed


def _payload_fields(schema: pa.Schema) -> list[str]:
    pt = schema.field(schema.get_field_index("payload")).type
    return [pt.field(i).name for i in range(pt.num_fields)]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_preserves_data(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"

    old = [_make_point(str(uuid4()), "one"), _make_point(str(uuid4()), "two")]
    await _seed(adapter, col, old)

    removed = await _strip_payload_field(adapter, col)
    assert removed not in _payload_fields(
        (await (await adapter.get_collection(col)).to_arrow()).schema
    )

    new = [_make_point(str(uuid4()), "three")]
    await adapter.create_data_points(col, new)

    results = await adapter.retrieve(col, [p.id for p in old + new])
    assert {str(r.id) for r in results} == {p.id for p in old + new}
    assert removed in _payload_fields((await (await adapter.get_collection(col)).to_arrow()).schema)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_no_migration_needed(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    id1, id2 = str(uuid4()), str(uuid4())

    await _seed(adapter, col, [_make_point(id1, "first")])
    await adapter.create_data_points(col, [_make_point(id2, "second")])

    assert len(await adapter.retrieve(col, [id1, id2])) == 2


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_dedup_by_id(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    sid = str(uuid4())

    await _seed(adapter, col, [_make_point(sid, "original")])
    await _strip_payload_field(adapter, col)
    await adapter.create_data_points(col, [_make_point(sid, "updated")])

    results = await adapter.retrieve(col, [sid])
    assert len(results) == 1
    assert results[0].payload["text"] == "updated"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_uses_pydantic_defaults(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"

    old_point = _make_point(str(uuid4()), "check defaults")
    await _seed(adapter, col, [old_point])

    removed = await _strip_payload_field(adapter, col)
    await adapter.create_data_points(col, [_make_point(str(uuid4()), "new")])

    result = (await adapter.retrieve(col, [old_point.id]))[0]
    if removed == "feedback_weight":
        assert result.payload["feedback_weight"] == 0.5


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_non_schema_errors_propagate(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    await _seed(adapter, col, [_make_point(str(uuid4()), "seed")])

    async def _explode(_):
        raise RuntimeError("network timeout")

    adapter.embed_data = _explode

    with pytest.raises(RuntimeError, match="network timeout"):
        await adapter.create_data_points(col, [_make_point(str(uuid4()), "boom")])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_payload_preserves_inherited_datapoint_fields(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "InheritedFields_text"
    point = _InheritedFieldsPoint(
        id=str(uuid4()),
        text="payload parity",
        belongs_to_set=["alpha", "beta"],
        source_task="unit-test",
        feedback_weight=0.75,
    )

    await _seed(adapter, col, [point])

    result = (await adapter.retrieve(col, [point.id]))[0]

    assert result.payload["id"] == str(point.id)
    assert result.payload["text"] == point.text
    assert result.payload["type"] == "_InheritedFieldsPoint"
    assert result.payload["belongs_to_set"] == ["alpha", "beta"]
    assert result.payload["source_task"] == "unit-test"
    assert result.payload["feedback_weight"] == 0.75
    assert result.payload["created_at"] == point.created_at
    assert result.payload["updated_at"] == point.updated_at


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_coalesces_null_payload_values_to_defaults(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    legacy_point = _make_point(str(uuid4()), "legacy row").model_dump()
    legacy_point["feedback_weight"] = None

    old_collection = AsyncMock()
    old_collection.to_arrow.return_value = Mock(
        to_pylist=Mock(
            return_value=[
                {
                    "id": str(uuid4()),
                    "vector": [0.1, 0.2, 0.3],
                    "payload": legacy_point,
                }
            ]
        )
    )

    new_collection = AsyncMock()
    connection = AsyncMock()
    connection.open_table.return_value = new_collection

    adapter.get_connection = AsyncMock(return_value=connection)

    await adapter._migrate_collection_schema(
        collection_name="Test_text",
        old_collection=old_collection,
        payload_schema=IndexSchema,
        new_lance_data_points=[],
    )

    assert new_collection.add.await_count == 1
    migrated_rows = new_collection.add.await_args.args[0]
    assert len(migrated_rows) == 1
    assert migrated_rows[0].payload.feedback_weight == 0.5


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_create_data_points_repairs_null_non_nullable_lance_error(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )

    merge_builder = Mock()
    merge_builder.when_matched_update_all.return_value = merge_builder
    merge_builder.when_not_matched_insert_all.return_value = merge_builder
    merge_builder.execute = AsyncMock(
        side_effect=RuntimeError(
            "lance error: Invalid user input: The field `feedback_weight` "
            "contained null values even though the field is marked non-null in the schema"
        )
    )

    collection = Mock()
    collection.merge_insert.return_value = merge_builder

    adapter.has_collection = AsyncMock(return_value=True)
    adapter.get_collection = AsyncMock(return_value=collection)
    adapter.embed_data = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    adapter._migrate_collection_schema = AsyncMock()

    await adapter.create_data_points("Test_text", [_make_point(str(uuid4()), "repair me")])

    assert adapter._migrate_collection_schema.await_count == 1
