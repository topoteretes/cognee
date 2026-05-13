"""Tests for LanceDB schema migration when the payload schema evolves."""

from __future__ import annotations

import pyarrow as pa
import pytest
from uuid import uuid4
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


def _make_point(id: str, text: str, name: str = "") -> IndexSchema:
    return IndexSchema(id=id, text=text, name=name)


class _InheritedFieldsPoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class _ImpossibleDefaultPoint(DataPoint):
    text: str
    required_pair: tuple[int, int]
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


async def _strip_payload_named_field(adapter, name: str, field_name: str):
    """Remove a specific payload field by name."""
    table = await (await adapter.get_collection(name)).to_arrow()
    pi = table.schema.get_field_index("payload")
    pt = table.schema.field(pi).type

    target_index = None
    for i in range(pt.num_fields):
        if pt.field(i).name == field_name:
            target_index = i
            break
    assert target_index is not None

    keep = [j for j in range(pt.num_fields) if j != target_index]
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
    if removed == "name":
        assert result.payload["name"] == ""


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
async def test_migration_aborts_without_safe_default_and_preserves_rows(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "ImpossibleDefault_text"

    old_point = _ImpossibleDefaultPoint(
        id=str(uuid4()),
        text="old",
        required_pair=(1, 2),
    )
    await _seed(adapter, col, [old_point])
    await _strip_payload_named_field(adapter, col, "required_pair")

    with pytest.raises(RuntimeError, match="Add an explicit default value"):
        await adapter.create_data_points(
            col,
            [
                _ImpossibleDefaultPoint(
                    id=str(uuid4()),
                    text="new",
                    required_pair=(3, 4),
                )
            ],
        )

    # Migration abort should preserve the original table and rows.
    result = await adapter.retrieve(col, [old_point.id])
    assert len(result) == 1
    assert result[0].payload["text"] == "old"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_null_in_non_null_field_triggers_migration(tmp_path):
    # Regression for #2702: the lance-file writer raises
    # "contained null values even though the field is marked non-null" when an
    # old-schema row is upserted against a newer schema that requires a field
    # to be non-null. The message does not match "not found in target schema",
    # so the auto-migration path was bypassed and the RuntimeError escaped.
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    await _seed(adapter, col, [_make_point(str(uuid4()), "seed")])

    # Wrap get_collection so we see every collection instance the adapter
    # opens — the one used inside create_data_points is fetched after our
    # hook runs, so we patch merge_insert on each one we hand back.
    original_get_collection = adapter.get_collection
    called = {"count": 0}

    async def wrapped_get_collection(name):
        collection = await original_get_collection(name)
        original_merge_insert = collection.merge_insert

        def patched_merge_insert(key):
            builder = original_merge_insert(key)
            original_execute = builder.execute

            async def fail_once(data):
                called["count"] += 1
                if called["count"] == 1:
                    raise RuntimeError(
                        "lance error: Invalid user input: The field `feedback_weight` "
                        "contained null values even though the field is marked non-null "
                        "in the schema"
                    )
                return await original_execute(data)

            builder.execute = fail_once
            return builder

        collection.merge_insert = patched_merge_insert
        return collection

    adapter.get_collection = wrapped_get_collection

    new_id = str(uuid4())
    await adapter.create_data_points(col, [_make_point(new_id, "after-drift")])

    # Migration path ran (rebuilt the table via drop/create), so restore the
    # real get_collection before reading back.
    adapter.get_collection = original_get_collection

    results = await adapter.retrieve(col, [new_id])
    assert len(results) == 1
    assert results[0].payload["text"] == "after-drift"
    assert called["count"] >= 1


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
