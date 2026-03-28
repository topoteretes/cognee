"""Tests for LanceDB schema migration when the payload schema evolves.

Simulates a table created with schema V1 that is later written to with
schema V2 (which has an extra column).  The adapter should transparently
migrate the table while preserving every historical row.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from uuid import uuid4

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        LanceDBAdapter,
        IndexSchema,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    """Deterministic embedding engine for testing."""

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_point(id: str, text: str) -> IndexSchema:
    return IndexSchema(id=id, text=text)


async def _seed_collection(adapter, collection_name, points):
    """Seed a collection following the production flow: create_collection first,
    then create_data_points (avoids the known lock re-entry in create_data_points)."""
    payload_schema = type(points[0])
    await adapter.create_collection(collection_name, payload_schema)
    await adapter.create_data_points(collection_name, points)


async def _narrow_payload_schema(adapter, collection_name: str):
    """Remove a non-essential field from the payload struct in an existing
    LanceDB table, simulating schema drift (table created before the model
    gained that field).

    Returns the name of the field that was removed.
    """
    collection = await adapter.get_collection(collection_name)
    arrow_table = await collection.to_arrow()
    payload_idx = arrow_table.schema.get_field_index("payload")
    payload_struct = arrow_table.schema.field(payload_idx).type

    # Pick the last removable field (anything other than id/text which
    # IndexSchema requires).
    removable = None
    for i in range(payload_struct.num_fields):
        name = payload_struct.field(i).name
        if name not in ("id", "text"):
            removable = i

    assert removable is not None, "No removable field found in payload struct"

    removed_name = payload_struct.field(removable).name

    keep_fields = [
        payload_struct.field(j) for j in range(payload_struct.num_fields) if j != removable
    ]
    payload_col = arrow_table.column("payload").combine_chunks()
    keep_arrays = [
        payload_col.field(payload_struct.field(j).name)
        for j in range(payload_struct.num_fields)
        if j != removable
    ]
    narrowed_payload = pa.StructArray.from_arrays(keep_arrays, fields=keep_fields)
    narrowed_table = arrow_table.set_column(
        payload_idx,
        pa.field("payload", pa.struct(keep_fields)),
        narrowed_payload,
    )

    connection = await adapter.get_connection()
    await connection.drop_table(collection_name)
    await connection.create_table(collection_name, data=narrowed_table)

    return removed_name


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_schema_migration_preserves_existing_data(tmp_path):
    """Insert data, artificially narrow the table schema, then insert again.

    The second insert should trigger an automatic migration and all rows
    from the first insert must still be retrievable.
    """
    db_path = str(tmp_path / "test_migration.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestMigration_text"

    # Phase 1: seed the table
    old_points = [
        _make_point(id=str(uuid4()), text="historical record one"),
        _make_point(id=str(uuid4()), text="historical record two"),
    ]
    await _seed_collection(adapter, collection_name, old_points)

    retrieved = await adapter.retrieve(collection_name, [old_points[0].id])
    assert len(retrieved) == 1
    assert retrieved[0].payload["text"] == "historical record one"

    # Phase 2: simulate schema drift
    removed_field = await _narrow_payload_schema(adapter, collection_name)

    # Confirm the field is gone
    reopened = await adapter.get_collection(collection_name)
    schema_after = (await reopened.to_arrow()).schema
    payload_struct_after = schema_after.field(schema_after.get_field_index("payload")).type
    field_names_after = [
        payload_struct_after.field(k).name for k in range(payload_struct_after.num_fields)
    ]
    assert removed_field not in field_names_after, "Test setup: field should be gone"

    # Phase 3: insert new data — triggers migration
    new_points = [_make_point(id=str(uuid4()), text="new record after migration")]
    await adapter.create_data_points(collection_name, new_points)

    # Phase 4: all data must have survived
    all_ids = [p.id for p in old_points] + [p.id for p in new_points]
    all_results = await adapter.retrieve(collection_name, all_ids)
    retrieved_ids = {str(r.id) for r in all_results}

    for point in old_points:
        assert point.id in retrieved_ids, f"Historical point {point.id} lost during migration"
    for point in new_points:
        assert point.id in retrieved_ids, f"New point {point.id} not found after migration"

    # Verify the migrated table has the full schema again
    final_collection = await adapter.get_collection(collection_name)
    final_schema = (await final_collection.to_arrow()).schema
    payload_struct_final = final_schema.field(final_schema.get_field_index("payload")).type
    final_field_names = [
        payload_struct_final.field(k).name for k in range(payload_struct_final.num_fields)
    ]
    assert removed_field in final_field_names, (
        f"Migrated table should include the '{removed_field}' field"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_no_migration_when_schema_matches(tmp_path):
    """When the schema hasn't changed, data is upserted normally."""
    db_path = str(tmp_path / "test_no_migration.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestNoMigration_text"

    first_id = str(uuid4())
    second_id = str(uuid4())

    points_1 = [_make_point(id=first_id, text="first")]
    await _seed_collection(adapter, collection_name, points_1)

    await adapter.create_data_points(collection_name, [_make_point(id=second_id, text="second")])

    results = await adapter.retrieve(collection_name, [first_id, second_id])
    assert len(results) == 2


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_migration_deduplicates_by_id(tmp_path):
    """When new data overlaps with existing ids, the new data should win
    after migration (same merge_insert semantics)."""
    db_path = str(tmp_path / "test_dedup.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestDedup_text"
    shared_id = str(uuid4())

    await _seed_collection(
        adapter, collection_name, [_make_point(id=shared_id, text="original text")]
    )

    await _narrow_payload_schema(adapter, collection_name)

    # Insert with same id but updated text — triggers migration
    await adapter.create_data_points(
        collection_name, [_make_point(id=shared_id, text="updated text")]
    )

    results = await adapter.retrieve(collection_name, [shared_id])
    assert len(results) == 1
    assert results[0].payload["text"] == "updated text"
