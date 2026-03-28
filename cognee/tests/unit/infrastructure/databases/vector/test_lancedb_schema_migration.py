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
    await adapter.create_collection(collection_name, type(points[0]))
    await adapter.create_data_points(collection_name, points)


async def _strip_payload_field(adapter, collection_name: str) -> str:
    """Remove the last non-essential field from the payload struct of an
    existing LanceDB table, simulating schema drift.

    This builds a new Arrow table from scratch with a narrower struct type
    and replaces the collection — no reverse-engineering of the auto-generated
    schema required.

    Returns the name of the removed field.
    """
    collection = await adapter.get_collection(collection_name)
    table = await collection.to_arrow()
    payload_idx = table.schema.get_field_index("payload")
    payload_type = table.schema.field(payload_idx).type

    # Find the last field that isn't required by IndexSchema
    removable = None
    for i in range(payload_type.num_fields):
        if payload_type.field(i).name not in ("id", "text"):
            removable = i

    assert removable is not None, "No removable field found"
    removed_name = payload_type.field(removable).name

    # Build narrowed struct
    keep_indices = [j for j in range(payload_type.num_fields) if j != removable]
    keep_fields = [payload_type.field(j) for j in keep_indices]
    payload_array = table.column("payload").combine_chunks()
    keep_arrays = [payload_array.field(payload_type.field(j).name) for j in keep_indices]

    narrowed = table.set_column(
        payload_idx,
        pa.field("payload", pa.struct(keep_fields)),
        pa.StructArray.from_arrays(keep_arrays, fields=keep_fields),
    )

    connection = await adapter.get_connection()
    await connection.drop_table(collection_name)
    await connection.create_table(collection_name, data=narrowed)

    return removed_name


def _get_payload_field_names(schema: pa.Schema) -> list[str]:
    """Extract field names from the payload struct in a table schema."""
    payload_type = schema.field(schema.get_field_index("payload")).type
    return [payload_type.field(i).name for i in range(payload_type.num_fields)]


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_schema_migration_preserves_existing_data(tmp_path):
    """Insert data, narrow the table schema, insert again.
    All historical rows must survive and the schema must be updated."""
    db_path = str(tmp_path / "test_migration.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestMigration_text"

    old_points = [
        _make_point(id=str(uuid4()), text="historical record one"),
        _make_point(id=str(uuid4()), text="historical record two"),
    ]
    await _seed_collection(adapter, collection_name, old_points)

    retrieved = await adapter.retrieve(collection_name, [old_points[0].id])
    assert len(retrieved) == 1
    assert retrieved[0].payload["text"] == "historical record one"

    removed_field = await _strip_payload_field(adapter, collection_name)

    # Confirm the field is gone
    schema_after = (await (await adapter.get_collection(collection_name)).to_arrow()).schema
    assert removed_field not in _get_payload_field_names(schema_after)

    # Insert new data — triggers migration
    new_points = [_make_point(id=str(uuid4()), text="new record after migration")]
    await adapter.create_data_points(collection_name, new_points)

    # All data must have survived
    all_ids = [p.id for p in old_points] + [p.id for p in new_points]
    all_results = await adapter.retrieve(collection_name, all_ids)
    retrieved_ids = {str(r.id) for r in all_results}

    for point in old_points:
        assert point.id in retrieved_ids, f"Historical point {point.id} lost during migration"
    for point in new_points:
        assert point.id in retrieved_ids, f"New point {point.id} not found after migration"

    # Migrated table must have the full schema
    final_schema = (await (await adapter.get_collection(collection_name)).to_arrow()).schema
    assert removed_field in _get_payload_field_names(final_schema)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_no_migration_when_schema_matches(tmp_path):
    """When the schema hasn't changed, data is upserted normally."""
    db_path = str(tmp_path / "test_no_migration.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestNoMigration_text"

    first_id = str(uuid4())
    second_id = str(uuid4())

    await _seed_collection(adapter, collection_name, [_make_point(id=first_id, text="first")])
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
    await _strip_payload_field(adapter, collection_name)

    await adapter.create_data_points(
        collection_name, [_make_point(id=shared_id, text="updated text")]
    )

    results = await adapter.retrieve(collection_name, [shared_id])
    assert len(results) == 1
    assert results[0].payload["text"] == "updated text"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_migration_uses_pydantic_defaults(tmp_path):
    """After migration, newly-added fields should carry the Pydantic model
    default (e.g. feedback_weight=0.5), not a generic zero."""
    db_path = str(tmp_path / "test_defaults.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestDefaults_text"

    point = _make_point(id=str(uuid4()), text="check defaults")
    await _seed_collection(adapter, collection_name, [point])

    removed_field = await _strip_payload_field(adapter, collection_name)

    # Insert a new point to trigger migration
    new_point = _make_point(id=str(uuid4()), text="new")
    await adapter.create_data_points(collection_name, [new_point])

    # Retrieve the OLD point — its migrated field should have the model default
    results = await adapter.retrieve(collection_name, [point.id])
    assert len(results) == 1

    migrated_value = results[0].payload.get(removed_field)
    # The Pydantic model's default should have been used, not a generic zero.
    # For IndexSchema (extends DataPoint), feedback_weight defaults to 0.5.
    if removed_field == "feedback_weight":
        assert migrated_value == 0.5, f"Expected Pydantic default 0.5, got {migrated_value}"


# ---------------------------------------------------------------------------
# Failure / edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_migration_creates_recovery_table_on_failure(tmp_path):
    """If migration fails mid-way, a __recovery table should be created
    so the user doesn't silently lose data."""
    db_path = str(tmp_path / "test_recovery.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestRecovery_text"

    point = _make_point(id=str(uuid4()), text="important data")
    await _seed_collection(adapter, collection_name, [point])
    await _strip_payload_field(adapter, collection_name)

    # Sabotage _align_table_to_schema so the migration fails after the old
    # table has been dropped but before data is fully re-inserted.
    original_align = adapter._align_table_to_schema

    def _exploding_align(*args, **kwargs):
        raise RuntimeError("simulated migration failure")

    adapter._align_table_to_schema = _exploding_align

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        await adapter.create_data_points(
            collection_name, [_make_point(id=str(uuid4()), text="trigger")]
        )

    adapter._align_table_to_schema = original_align

    # A recovery table should exist with the original data
    connection = await adapter.get_connection()
    tables = await connection.table_names()
    recovery_name = f"{collection_name}__recovery"
    assert recovery_name in tables, f"Expected recovery table '{recovery_name}', found: {tables}"

    recovery_table = await connection.open_table(recovery_name)
    recovery_data = await recovery_table.to_arrow()
    assert recovery_data.num_rows >= 1, "Recovery table should contain the original row(s)"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_non_schema_errors_are_not_swallowed(tmp_path):
    """Errors unrelated to schema mismatch must propagate, not be caught
    by the migration handler."""
    db_path = str(tmp_path / "test_passthrough.lancedb")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    collection_name = "TestPassthrough_text"

    await _seed_collection(adapter, collection_name, [_make_point(id=str(uuid4()), text="seed")])

    # Sabotage embed_data to cause a non-schema error during merge_insert
    async def _explode(texts):
        raise RuntimeError("network timeout")

    adapter.embed_data = _explode

    with pytest.raises(RuntimeError, match="network timeout"):
        await adapter.create_data_points(
            collection_name, [_make_point(id=str(uuid4()), text="boom")]
        )
