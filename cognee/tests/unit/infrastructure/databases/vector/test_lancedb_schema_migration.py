"""Tests for LanceDB schema migration when the payload schema evolves."""

from __future__ import annotations

import hashlib
import importlib
import os
import tracemalloc
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

import pyarrow as pa
import pytest

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.exceptions import StoredVectorWidthMismatchError
from cognee.infrastructure.engine import DataPoint
from cognee_db_workers.lancedb_schema_evolution import (
    MIGRATING_COLUMN,
    OLD_PAYLOAD_COLUMN,
    STATE_KEY,
    SchemaEvolutionError,
)

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    model = "fake"

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return EmbeddingResult([[0.1, 0.2, 0.3] for _ in texts], tokens=0)


def _make_point(id: str, text: str, name: str = "") -> IndexSchema:
    return IndexSchema(id=id, text=text, name=name)


class _InheritedFieldsPoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class _Tier(str, Enum):
    FREE = "free"
    PAID = "paid"


class _TypedDefaultsPoint(DataPoint):
    """Defaults pyarrow cannot build from their string form, and one required Enum
    (whose inferred default is an Enum member)."""

    text: str
    seen_at: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tier: _Tier
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
async def test_migration_backfills_source_chunk_id_default(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"

    old_point = _make_point(str(uuid4()), "old schema")
    await _seed(adapter, col, [old_point])
    await _strip_payload_named_field(adapter, col, "source_chunk_id")

    await adapter.create_data_points(col, [_make_point(str(uuid4()), "new schema")])

    result = (await adapter.retrieve(col, [old_point.id]))[0]
    assert result.payload["source_chunk_id"] is None


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


# ---------------------------------------------------------------------------
# In-place evolution: the vectors are never read, nothing is dropped.
# ---------------------------------------------------------------------------


class _WideEmbeddingEngine:
    """3,072 dimensions, like the production default: the width that made the old
    rewrite swap; it also tells a cheap migration from one that touched the vectors."""

    dimensions = 3072

    def get_vector_size(self):
        return self.dimensions

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return EmbeddingResult([[0.0] * self.dimensions for _ in texts], tokens=0)


def _data_files(db_path: str, collection: str) -> dict[str, str]:
    """sha256 of every data file of the table, by file name."""
    data_dir = os.path.join(db_path, f"{collection}.lance", "data")
    return {
        name: hashlib.sha256(open(os.path.join(data_dir, name), "rb").read()).hexdigest()
        for name in os.listdir(data_dir)
    }


async def _payload_type(adapter, collection):
    schema = await (await adapter.get_collection(collection)).schema()
    return schema.field("payload").type


async def _rewrite_payload(adapter, collection, new_struct: pa.StructType, build):
    """Replace the table's payload struct with one built by ``build(old_struct_array)``,
    simulating a table written by an older or differently typed model."""
    table = await (await adapter.get_collection(collection)).to_arrow()
    pi = table.schema.get_field_index("payload")
    col = table.column("payload").combine_chunks()
    rewritten = table.set_column(pi, pa.field("payload", new_struct), build(col))
    conn = await adapter.get_connection()
    await conn.drop_table(collection)
    await conn.create_table(collection, data=rewritten)


def _struct_without(struct: pa.StructType, *names) -> tuple[pa.StructType, list[int]]:
    keep = [i for i in range(struct.num_fields) if struct.field(i).name not in names]
    return pa.struct([struct.field(i) for i in keep]), keep


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_proactive_migration_reads_schemas_only(tmp_path, monkeypatch):
    """A version bump over a compatible store never loads a table's rows."""
    import lancedb

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    await _seed(adapter, "Entity_name", [_make_point(str(uuid4()), "alice")])
    await _seed(adapter, "DocumentChunk_text", [_make_point(str(uuid4()), "a chunk")])

    async def never(self, *args, **kwargs):
        raise AssertionError("the schema check must not read the table")

    monkeypatch.setattr(lancedb.AsyncTable, "to_arrow", never)
    monkeypatch.setattr(lancedb.AsyncTable, "to_pandas", never, raising=False)

    result = await adapter.run_migrations()

    assert sorted(result["checked_collections"]) == ["DocumentChunk_text", "Entity_name"]
    assert result["migrated_collections"] == []


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_leaves_the_vector_data_files_untouched(tmp_path, monkeypatch):
    """An added field is backfilled without the vectors leaving the engine: every data
    file written before the migration is byte-identical after it, the table is never
    read into Python (``to_arrow`` is off), and no Python memory scales with the
    vectors (the old path turned every float into an object)."""
    import lancedb

    db_path = str(tmp_path / "db")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_WideEmbeddingEngine())
    col = "EdgeType_relationship_name"
    rows = 2000
    ids = [str(uuid4()) for _ in range(rows)]
    await _seed(adapter, col, [_make_point(id_, f"edge {i}") for i, id_ in enumerate(ids)])
    await _strip_payload_named_field(adapter, col, "external_metadata")
    before = _data_files(db_path, col)
    vector_bytes = rows * _WideEmbeddingEngine.dimensions * 4

    async def never(self, *args, **kwargs):
        raise AssertionError("the migration must not read the table into Python")

    monkeypatch.setattr(lancedb.AsyncTable, "to_arrow", never)
    tracemalloc.start()
    try:
        result = await adapter.run_migrations()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    monkeypatch.undo()

    assert result["migrated_collections"] == [col]
    after = _data_files(db_path, col)
    assert all(after[name] == digest for name, digest in before.items())
    assert peak < vector_bytes / 4, f"{peak} bytes of Python memory for {vector_bytes} of vectors"
    assert "external_metadata" in _payload_fields(
        await (await adapter.get_collection(col)).schema()
    )
    # The rows and vectors are intact: the first id still searches to itself.
    [row] = await adapter.retrieve(col, [ids[0]])
    assert row.payload["text"] == "edge 0" and row.payload["external_metadata"] is None
    # And a second pass has nothing to do.
    assert (await adapter.run_migrations())["migrated_collections"] == []


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_never_drops_the_table(tmp_path, monkeypatch):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    await _seed(adapter, col, [_make_point(str(uuid4()), "kept")])
    await _strip_payload_field(adapter, col)

    connection = await adapter.get_connection()

    async def no_drop(*args, **kwargs):
        raise AssertionError("the migration must not drop the table")

    monkeypatch.setattr(type(connection), "drop_table", no_drop)
    assert (await adapter.run_migrations())["migrated_collections"] == [col]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_interrupted_migration_after_the_first_step_is_computed_again(tmp_path):
    """A crash after the new column was added (before the swap) leaves ``payload`` as
    it was next to ``__payload_migrating``; the table reads as before, and the next
    run drops the leftover and completes."""
    from cognee_db_workers.lancedb_compaction import open_as_lance

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "survivor")
    await _seed(adapter, col, [point])
    await _strip_payload_named_field(adapter, col, "external_metadata")

    dataset = await open_as_lance(await adapter.get_collection(col))
    dataset.add_columns({MIGRATING_COLUMN: "payload"})
    assert MIGRATING_COLUMN in (await (await adapter.get_collection(col)).schema()).names
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["text"] == "survivor"

    assert (await adapter.run_migrations())["migrated_collections"] == [col]
    schema = await (await adapter.get_collection(col)).schema()
    assert MIGRATING_COLUMN not in schema.names and OLD_PAYLOAD_COLUMN not in schema.names
    assert "external_metadata" in _payload_fields(schema)
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["text"] == "survivor"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_interrupted_migration_after_the_swap_keeps_the_new_payload(tmp_path):
    """A crash after the swap leaves the new struct as ``payload`` next to
    ``__payload_old``: the table already reads in the new shape, and the next run
    drops the old column without computing the payload again."""
    from cognee_db_workers.lancedb_compaction import open_as_lance
    from cognee_db_workers.lancedb_schema_evolution import evolve_payload_schema

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "survivor")
    await _seed(adapter, col, [point])
    await _strip_payload_named_field(adapter, col, "external_metadata")
    target, required = adapter._target_payload_struct(IndexSchema)

    dataset = await open_as_lance(await adapter.get_collection(col))
    real_drop = dataset.drop_columns

    def crash_at_the_drop(names):
        raise OSError("simulated crash before the last step")

    dataset.drop_columns = crash_at_the_drop
    with pytest.raises(OSError, match="simulated crash"):
        evolve_payload_schema(
            dataset,
            target,
            adapter._get_payload_defaults(IndexSchema),
            required=required,
        )
    dataset.drop_columns = real_drop

    # Every column is present: payload is already the new struct.
    schema = await (await adapter.get_collection(col)).schema()
    assert OLD_PAYLOAD_COLUMN in schema.names and MIGRATING_COLUMN not in schema.names
    assert "external_metadata" in _payload_fields(schema)
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["text"] == "survivor" and row.payload["external_metadata"] is None

    computed = []
    real_add = type(dataset).add_columns
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            type(dataset), "add_columns", lambda *a, **k: computed.append(a) or real_add(*a, **k)
        )
        assert (await adapter.run_migrations())["migrated_collections"] == [col]
    assert computed == []
    schema = await (await adapter.get_collection(col)).schema()
    assert OLD_PAYLOAD_COLUMN not in schema.names
    assert "external_metadata" in _payload_fields(schema)
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["text"] == "survivor"
    assert (await adapter.run_migrations())["migrated_collections"] == []


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_payload_column_is_present_at_every_committed_version(tmp_path):
    """No version of the table lacks ``payload``: a reader during a migration never
    sees the intermediate schema."""
    from cognee_db_workers.lancedb_compaction import open_as_lance

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    await _seed(adapter, col, [_make_point(str(uuid4()), "row")])
    await _strip_payload_named_field(adapter, col, "external_metadata")
    assert (await adapter.run_migrations())["migrated_collections"] == [col]

    dataset = await open_as_lance(await adapter.get_collection(col))
    versions = [v["version"] for v in dataset.versions()]
    assert len(versions) >= 4
    for version in versions:
        assert "payload" in dataset.checkout_version(version).schema.names, version


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_drops_a_field_the_model_no_longer_has(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "with extra")
    await _seed(adapter, col, [point])
    struct = await _payload_type(adapter, col)
    wider = pa.struct(
        [struct.field(i) for i in range(struct.num_fields)] + [pa.field("legacy_flag", pa.bool_())]
    )
    await _rewrite_payload(
        adapter,
        col,
        wider,
        lambda old: pa.StructArray.from_arrays(
            old.flatten() + [pa.array([True] * len(old))], fields=list(wider)
        ),
    )
    assert "legacy_flag" in _payload_fields(await (await adapter.get_collection(col)).schema())

    assert (await adapter.run_migrations())["migrated_collections"] == [col]
    fields = _payload_fields(await (await adapter.get_collection(col)).schema())
    assert "legacy_flag" not in fields
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["text"] == "with extra"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_casts_a_field_whose_type_changed(tmp_path):
    """``importance_weight`` stored as an integer is cast to the model's float."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "typed")
    await _seed(adapter, col, [point])
    struct = await _payload_type(adapter, col)
    index = struct.get_field_index("importance_weight")
    retyped = pa.struct(
        [
            pa.field("importance_weight", pa.int64()) if i == index else struct.field(i)
            for i in range(struct.num_fields)
        ]
    )

    def build(old):
        children = old.flatten()
        children[index] = pa.array([2] * len(old), pa.int64())
        return pa.StructArray.from_arrays(children, fields=list(retyped))

    await _rewrite_payload(adapter, col, retyped, build)

    assert (await adapter.run_migrations())["migrated_collections"] == [col]
    assert pa.types.is_float64((await _payload_type(adapter, col)).field(index).type)
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["importance_weight"] == 2.0


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_fixes_a_nullability_only_difference(tmp_path):
    """``text`` stored nullable (no nulls in it) becomes non-null as the model says."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "non-null")
    await _seed(adapter, col, [point])
    struct = await _payload_type(adapter, col)
    index = struct.get_field_index("text")
    assert not struct.field(index).nullable
    relaxed = pa.struct(
        [
            pa.field("text", pa.string(), nullable=True) if i == index else struct.field(i)
            for i in range(struct.num_fields)
        ]
    )
    await _rewrite_payload(
        adapter,
        col,
        relaxed,
        lambda old: pa.StructArray.from_arrays(old.flatten(), fields=list(relaxed)),
    )
    assert not adapter._is_payload_schema_compatible(await _payload_type(adapter, col), IndexSchema)

    assert (await adapter.run_migrations())["migrated_collections"] == [col]
    assert adapter._is_payload_schema_compatible(await _payload_type(adapter, col), IndexSchema)
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["text"] == "non-null"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_fills_nulls_of_a_now_required_field_from_its_default(tmp_path):
    """Nulls in a field the model declares non-null take the model's default."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "weighted")
    await _seed(adapter, col, [point])
    struct = await _payload_type(adapter, col)
    index = struct.get_field_index("feedback_weight")
    assert not struct.field(index).nullable
    relaxed = pa.struct(
        [
            pa.field("feedback_weight", pa.float64(), nullable=True)
            if i == index
            else struct.field(i)
            for i in range(struct.num_fields)
        ]
    )

    def build(old):
        children = old.flatten()
        children[index] = pa.nulls(len(old), pa.float64())
        return pa.StructArray.from_arrays(children, fields=list(relaxed))

    await _rewrite_payload(adapter, col, relaxed, build)

    assert (await adapter.run_migrations())["migrated_collections"] == [col]
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["feedback_weight"] == IndexSchema.model_fields["feedback_weight"].default


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_aborts_on_an_impossible_cast_and_changes_nothing(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "uncastable")
    await _seed(adapter, col, [point])
    struct = await _payload_type(adapter, col)
    index = struct.get_field_index("chunk_index")
    retyped = pa.struct(
        [
            pa.field("chunk_index", pa.string()) if i == index else struct.field(i)
            for i in range(struct.num_fields)
        ]
    )

    def build(old):
        children = old.flatten()
        children[index] = pa.array(["not a number"] * len(old))
        return pa.StructArray.from_arrays(children, fields=list(retyped))

    await _rewrite_payload(adapter, col, retyped, build)
    before = await _payload_type(adapter, col)

    with pytest.raises(SchemaEvolutionError, match="cannot be cast"):
        await adapter._migrate_collection_schema(col, IndexSchema, [])
    assert await _payload_type(adapter, col) == before
    [row] = await adapter.retrieve(col, [point.id])
    assert row.payload["chunk_index"] == "not a number"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_truth_centroid_table_migrates_against_its_own_model(tmp_path):
    """The one non-IndexSchema collection of a dataset store keeps every field."""
    from cognee.modules.truth_subspace.constants import TRUTH_CENTROID_COLLECTION
    from cognee.modules.truth_subspace.models import TruthCentroidPayload

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    centroid_id = uuid4()
    payload = TruthCentroidPayload(
        dataset_id="ds", slot=1, count=3, truth_epoch=2, updated_at=5, centroid=[0.1, 0.2, 0.3]
    )
    await adapter.upsert_raw_vectors(
        TRUTH_CENTROID_COLLECTION,
        [{"id": centroid_id, "vector": [0.1, 0.2, 0.3], "payload": payload.model_dump()}],
        payload_schema=TruthCentroidPayload,
    )

    result = await adapter.run_migrations()

    assert TRUTH_CENTROID_COLLECTION in result["checked_collections"]
    assert result["migrated_collections"] == []
    [row] = await adapter.retrieve(TRUTH_CENTROID_COLLECTION, [str(centroid_id)])
    assert row.payload["slot"] == 1 and row.payload["centroid"] == [0.1, 0.2, 0.3]
    assert "text" not in row.payload


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_refuses_a_vector_width_change(tmp_path):
    """A different embedding width is a re-embedding, not a schema migration."""
    narrow = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    point = _make_point(str(uuid4()), "three wide")
    await _seed(narrow, col, [point])
    await _strip_payload_named_field(narrow, col, "external_metadata")

    class _FiveWide(_FakeEmbeddingEngine):
        def get_vector_size(self):
            return 5

    wide = LanceDBAdapter(url=str(tmp_path / "db"), api_key=None, embedding_engine=_FiveWide())
    before = await _payload_type(wide, col)
    with pytest.raises(StoredVectorWidthMismatchError) as raised:
        await wide.run_migrations()
    assert "EMBEDDING_MODEL" in raised.value.remediation
    assert await _payload_type(wide, col) == before


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_runs_inside_the_subprocess_worker(tmp_path, monkeypatch):
    """Subprocess mode is the default: the evolution runs on the worker's handle, and the
    reactive path (a write that finds the drift) uses it too."""
    from cognee.infrastructure.databases.vector.config import VectorConfig

    db_path = str(tmp_path / "db")
    seeded = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    col = "Test_text"
    old = _make_point(str(uuid4()), "before")
    await _seed(seeded, col, [old])
    await _strip_payload_named_field(seeded, col, "external_metadata")
    await seeded.close()
    before = _data_files(db_path, col)

    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    config = VectorConfig(
        vector_db_provider="lancedb",
        vector_db_url=db_path,
        vector_db_name="worker_migration",
        vector_db_subprocess_enabled=True,
    )
    adapter = factory.create_vector_engine(**config.to_dict())
    try:
        assert adapter._subprocess_mode
        new = _make_point(str(uuid4()), "after")
        await adapter.create_data_points(col, [new])

        after = _data_files(db_path, col)
        assert all(after[name] == digest for name, digest in before.items())
        rows = {str(r.id): r.payload for r in await adapter.retrieve(col, [old.id, new.id])}
        assert rows[old.id]["text"] == "before" and rows[old.id]["external_metadata"] is None
        assert rows[new.id]["text"] == "after"
        # The worker's relaxed nullability is what the check compares against,
        # so a second pass finds the table compatible.
        assert (await adapter.run_migrations())["migrated_collections"] == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_subprocess_migration_refuses_a_required_field_without_default(tmp_path, monkeypatch):
    """The worker stores every field nullable, but which fields must hold a value is
    the model's rule: the same data the in-process path refuses is refused here, and
    nothing is written."""
    from cognee.infrastructure.databases.vector.config import VectorConfig

    db_path = str(tmp_path / "db")
    seeded = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    col = "ImpossibleDefault_text"
    old = _ImpossibleDefaultPoint(id=str(uuid4()), text="old", required_pair=(1, 2))
    await _seed(seeded, col, [old])
    await _strip_payload_named_field(seeded, col, "required_pair")
    await seeded.close()
    before = _data_files(db_path, col)

    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    config = VectorConfig(
        vector_db_provider="lancedb",
        vector_db_url=db_path,
        vector_db_name="worker_required",
        vector_db_subprocess_enabled=True,
    )
    adapter = factory.create_vector_engine(**config.to_dict())
    try:
        assert adapter._subprocess_mode
        with pytest.raises(RuntimeError, match="Add an explicit default value"):
            await adapter.create_data_points(
                col, [_ImpossibleDefaultPoint(id=str(uuid4()), text="new", required_pair=(3, 4))]
            )
        assert _data_files(db_path, col) == before
        schema = await (await adapter.get_collection(col)).schema()
        assert "required_pair" not in _payload_fields(schema)
        [row] = await adapter.retrieve(col, [old.id])
        assert row.payload["text"] == "old"
    finally:
        await adapter.close()


async def _interrupt(adapter, col, step: str):
    """Run the evolution of ``col`` up to ``step`` (``"swap"`` or ``"drop"``) and crash
    there, leaving the table as an interrupted migration would."""
    from cognee_db_workers.lancedb_compaction import open_as_lance
    from cognee_db_workers.lancedb_schema_evolution import evolve_payload_schema

    target, required = adapter._target_payload_struct(IndexSchema)
    dataset = await open_as_lance(await adapter.get_collection(col))

    def crash(*_args, **_kwargs):
        raise OSError("simulated crash")

    setattr(dataset, "alter_columns" if step == "swap" else "drop_columns", crash)
    with pytest.raises(OSError, match="simulated crash"):
        evolve_payload_schema(
            dataset, target, adapter._get_payload_defaults(IndexSchema), required=required
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
@pytest.mark.parametrize("step", ["swap", "drop"])
async def test_writes_land_while_a_migration_is_interrupted(tmp_path, step):
    """A write carries only id, vector and payload. The leftover column of either
    interrupted state is nullable, so merge_insert takes the write instead of refusing
    a non-nullable column the source does not provide."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "Test_text"
    old = _make_point(str(uuid4()), "before")
    await _seed(adapter, col, [old])
    await _strip_payload_named_field(adapter, col, "external_metadata")
    await _interrupt(adapter, col, step)

    schema = await (await adapter.get_collection(col)).schema()
    leftover = MIGRATING_COLUMN if step == "swap" else OLD_PAYLOAD_COLUMN
    assert leftover in schema.names and schema.field(leftover).nullable

    new = _make_point(str(uuid4()), "during")
    await adapter.create_data_points(col, [new])
    rows = {str(r.id): r.payload for r in await adapter.retrieve(col, [old.id, new.id])}
    assert rows[old.id]["text"] == "before" and rows[new.id]["text"] == "during"
    # Before the swap the write finds the old payload, hits the drift error and
    # finishes the migration reactively; after it the payload is already the new
    # struct, the write lands, and the leftover waits for run_migrations.
    schema = await (await adapter.get_collection(col)).schema()
    finished_by_write = leftover not in schema.names
    assert finished_by_write == (step == "swap")

    migrated = (await adapter.run_migrations())["migrated_collections"]
    assert migrated == ([] if finished_by_write else [col])
    schema = await (await adapter.get_collection(col)).schema()
    assert leftover not in schema.names and "external_metadata" in _payload_fields(schema)
    rows = {str(r.id): r.payload for r in await adapter.retrieve(col, [old.id, new.id])}
    assert rows[old.id]["text"] == "before" and rows[new.id]["text"] == "during"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_fills_datetime_and_enum_defaults(tmp_path):
    """A datetime default is built as a timestamp, not from its string form, and a
    required Enum field takes its inferred member's value."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "TypedDefaults_text"
    old = _TypedDefaultsPoint(id=str(uuid4()), text="old", tier=_Tier.PAID)
    await _seed(adapter, col, [old])
    await _strip_payload_named_field(adapter, col, "seen_at")
    await _strip_payload_named_field(adapter, col, "tier")

    stats = await adapter._migrate_collection_schema(col, _TypedDefaultsPoint, [])
    assert sorted(stats["added"]) == ["seen_at", "tier"]
    [row] = await adapter.retrieve(col, [old.id])
    assert row.payload["tier"] == _Tier.FREE.value
    assert (
        row.payload["seen_at"].replace(tzinfo=timezone.utc)
        == _TypedDefaultsPoint.model_fields["seen_at"].default
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_migration_aborts_on_a_default_of_the_wrong_type(tmp_path):
    """A default pyarrow cannot build as the field's type is a SchemaEvolutionError with
    the field named, not a bare Arrow error, and the table is unchanged."""
    from cognee_db_workers.lancedb_compaction import open_as_lance
    from cognee_db_workers.lancedb_schema_evolution import evolve_payload_schema

    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    col = "TypedDefaults_text"
    old = _TypedDefaultsPoint(id=str(uuid4()), text="old", tier=_Tier.PAID)
    await _seed(adapter, col, [old])
    await _strip_payload_named_field(adapter, col, "seen_at")
    target, required = adapter._target_payload_struct(_TypedDefaultsPoint)
    dataset = await open_as_lance(await adapter.get_collection(col))
    version = dataset.version

    with pytest.raises(SchemaEvolutionError, match="default 'not a time' of field 'seen_at'"):
        evolve_payload_schema(dataset, target, {"seen_at": "not a time"}, required=required)
    assert dataset.version == version


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_run_migrations_visits_every_table_before_raising(tmp_path):
    """Tables whose data cannot be migrated do not leave the others on the old schema:
    they are migrated, then one error names each table that failed (a single failure
    is raised as itself)."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    bad = ["Bad_text", "Worse_text"]
    for name in bad:
        await _seed(adapter, name, [_make_point(str(uuid4()), "uncastable")])
        struct = await _payload_type(adapter, name)
        index = struct.get_field_index("chunk_index")
        retyped = pa.struct(
            [
                pa.field("chunk_index", pa.string()) if i == index else struct.field(i)
                for i in range(struct.num_fields)
            ]
        )

        def build(old, index=index, retyped=retyped):
            children = old.flatten()
            children[index] = pa.array(["not a number"] * len(old))
            return pa.StructArray.from_arrays(children, fields=list(retyped))

        await _rewrite_payload(adapter, name, retyped, build)
    good = "Test_text"
    await _seed(adapter, good, [_make_point(str(uuid4()), "row")])
    await _strip_payload_named_field(adapter, good, "external_metadata")

    with pytest.raises(RuntimeError, match="cannot be cast") as info:
        await adapter.run_migrations()
    assert all(f"'{name}'" in str(info.value) for name in bad)
    assert isinstance(info.value.__cause__, SchemaEvolutionError)
    assert "external_metadata" in _payload_fields(
        await (await adapter.get_collection(good)).schema()
    )
    for name in bad:
        assert (await _payload_type(adapter, name)).field("chunk_index").type == pa.string()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_run_migrations_skips_a_table_whose_payload_is_not_a_struct(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    conn = await adapter.get_connection()
    await conn.create_table(
        "Odd_text",
        data=pa.table(
            {
                "id": ["1"],
                "vector": pa.array([[0.1, 0.2, 0.3]], pa.list_(pa.float32(), 3)),
                "payload": ["not a struct"],
            }
        ),
    )
    result = await adapter.run_migrations()
    assert result["skipped_collections"] == ["Odd_text"]
    assert result["migrated_collections"] == []


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_defaults_cross_to_the_worker_as_plain_values(tmp_path, monkeypatch):
    """An Enum member or model instance in the defaults would make the worker import
    cognee when unpickled; the adapter sends plain values."""
    from cognee.infrastructure.databases.vector.config import VectorConfig
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import RemoteLanceDBTable

    db_path = str(tmp_path / "db")
    seeded = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    col = "TypedDefaults_text"
    old = _TypedDefaultsPoint(id=str(uuid4()), text="old", tier=_Tier.PAID)
    await _seed(seeded, col, [old])
    await _strip_payload_named_field(seeded, col, "tier")
    await seeded.close()

    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    adapter = factory.create_vector_engine(
        **VectorConfig(
            vector_db_provider="lancedb",
            vector_db_url=db_path,
            vector_db_name="worker_defaults",
            vector_db_subprocess_enabled=True,
        ).to_dict()
    )
    sent = []
    real = RemoteLanceDBTable.evolve_payload_schema

    async def recording(self, target, **kwargs):
        sent.append(kwargs["defaults"])
        return await real(self, target, **kwargs)

    monkeypatch.setattr(RemoteLanceDBTable, "evolve_payload_schema", recording)
    try:
        await adapter._migrate_collection_schema(col, _TypedDefaultsPoint, [])
        [defaults] = sent
        assert defaults["tier"] == "free" and not isinstance(defaults["tier"], Enum)
        assert all(type(v).__module__ != "cognee" for v in defaults.values())
        [row] = await adapter.retrieve(col, [old.id])
        assert row.payload["tier"] == "free"
    finally:
        await adapter.close()


def test_width_error_names_a_variable_length_vector_column():
    error = StoredVectorWidthMismatchError("Test_text", None, 1536)
    assert "variable-length" in error.message and "None" not in error.message
    assert "EMBEDDING_MODEL" not in str(error)


def _intruder(db_path: str, col: str, existing_id: str, appended_id: str):
    """A writer in another process: through its own connection, outside the adapter's
    lock, it updates the existing row's text and appends a row, both in the stored
    (old) payload shape, copied from the existing row's Arrow payload."""
    import lancedb
    import pyarrow.compute as pc

    def write():
        table = lancedb.connect(db_path).open_table(col)
        arrow = table.to_arrow()
        payload = arrow.filter(pc.equal(arrow["id"], existing_id)).column("payload")
        payload = payload.combine_chunks()
        struct = payload.type
        text_index = struct.get_field_index("text")

        def with_text(text):
            children = payload.flatten()
            children[text_index] = pa.array([text], type=struct.field(text_index).type)
            return pa.StructArray.from_arrays(children, fields=list(struct))

        source = pa.table(
            {
                "id": [existing_id, appended_id],
                "vector": pa.array([[0.5, 0.5, 0.5]] * 2, pa.list_(pa.float32(), 3)),
                "payload": pa.concat_arrays(
                    [with_text("updated meanwhile"), with_text("appended meanwhile")]
                ),
            }
        )
        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            source
        )

    return write


async def _migrate_with_window_writes(adapter, col, db_path, existing_id, crash_at=None):
    """Evolve ``col`` while another connection appends one row and updates another
    between the add-column commit and the swap; optionally crash at ``crash_at``
    (``"update_config"`` or ``"drop_columns"``) after the swap."""
    from cognee_db_workers.lancedb_compaction import open_as_lance
    from cognee_db_workers.lancedb_schema_evolution import evolve_payload_schema

    appended_id = str(uuid4())
    intrude = _intruder(db_path, col, existing_id, appended_id)

    target, required = adapter._target_payload_struct(IndexSchema)
    dataset = await open_as_lance(await adapter.get_collection(col))
    real_swap = dataset.alter_columns

    def swap_after_intruder(*args, **kwargs):
        intrude()
        return real_swap(*args, **kwargs)

    dataset.alter_columns = swap_after_intruder
    if crash_at:

        def crash(*_args, **_kwargs):
            raise OSError("simulated crash")

        setattr(dataset, crash_at, crash)
    run = lambda: evolve_payload_schema(
        dataset, target, adapter._get_payload_defaults(IndexSchema), required=required
    )
    if crash_at:
        with pytest.raises(OSError, match="simulated crash"):
            run()
        return None, existing_id, appended_id
    return run(), existing_id, appended_id


async def _assert_window_rows_survived(adapter, col, updated_id, appended_id):
    rows = {str(r.id): r.payload for r in await adapter.retrieve(col, [updated_id, appended_id])}
    assert rows[updated_id]["text"] == "updated meanwhile"
    assert rows[appended_id]["text"] == "appended meanwhile"
    assert "external_metadata" in rows[updated_id] and "external_metadata" in rows[appended_id]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_rows_written_by_another_process_during_the_migration_survive(tmp_path):
    """The adapter's lock covers this process only. A row appended, and one updated,
    between the add-column commit and the swap keep their payload: the swap moves
    it into the old column, and the repair rebuilds payload from there for exactly
    the fragments the window created."""
    db_path = str(tmp_path / "db")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    col = "Test_text"
    seeded = _make_point(str(uuid4()), "before")
    await _seed(adapter, col, [seeded])
    await _strip_payload_named_field(adapter, col, "external_metadata")

    stats, updated_id, appended_id = await _migrate_with_window_writes(
        adapter, col, db_path, seeded.id
    )
    assert stats["repaired"] == 2 and stats["rows"] == 2
    await _assert_window_rows_survived(adapter, col, updated_id, appended_id)
    schema = await (await adapter.get_collection(col)).schema()
    assert OLD_PAYLOAD_COLUMN not in schema.names
    from cognee_db_workers.lancedb_compaction import open_as_lance

    assert STATE_KEY not in (await open_as_lance(await adapter.get_collection(col))).config()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_interrupted_after_the_swap_repairs_the_window_rows_on_the_next_run(tmp_path):
    """The window's fragment ids were recorded with the swap, so a run interrupted
    before the repair finishes it exactly, after later writes have landed."""
    db_path = str(tmp_path / "db")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    col = "Test_text"
    seeded = _make_point(str(uuid4()), "before")
    await _seed(adapter, col, [seeded])
    await _strip_payload_named_field(adapter, col, "external_metadata")
    _, updated_id, appended_id = await _migrate_with_window_writes(
        adapter, col, db_path, seeded.id, crash_at="drop_columns"
    )

    # A write after the swap is in the new shape and must not be touched.
    later = _make_point(str(uuid4()), "after the swap")
    await adapter.create_data_points(col, [later])

    assert (await adapter.run_migrations())["migrated_collections"] == [col]
    await _assert_window_rows_survived(adapter, col, updated_id, appended_id)
    [row] = await adapter.retrieve(col, [later.id])
    assert row.payload["text"] == "after the swap"
    schema = await (await adapter.get_collection(col)).schema()
    assert OLD_PAYLOAD_COLUMN not in schema.names


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_interrupted_before_the_window_was_recorded_keeps_the_old_column(tmp_path):
    """Without the record the window rows cannot be told from later writes: the next
    run refuses, says where the payloads are, and the table stays usable."""
    db_path = str(tmp_path / "db")
    adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine())
    col = "Test_text"
    seeded = _make_point(str(uuid4()), "before")
    await _seed(adapter, col, [seeded])
    await _strip_payload_named_field(adapter, col, "external_metadata")
    _, _updated_id, appended_id = await _migrate_with_window_writes(
        adapter, col, db_path, seeded.id, crash_at="update_config"
    )

    with pytest.raises(SchemaEvolutionError, match=f"'{OLD_PAYLOAD_COLUMN}' is kept"):
        await adapter.run_migrations()
    schema = await (await adapter.get_collection(col)).schema()
    assert OLD_PAYLOAD_COLUMN in schema.names
    later = _make_point(str(uuid4()), "still writable")
    await adapter.create_data_points(col, [later])
    rows = {str(r.id): r.payload for r in await adapter.retrieve(col, [later.id, appended_id])}
    assert rows[later.id]["text"] == "still writable"
    assert rows[appended_id] is None  # the window row, until restored by hand
