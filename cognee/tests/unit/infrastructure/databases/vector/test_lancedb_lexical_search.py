"""LanceDB's lexical (BM25) index on a real LanceDB in a temporary directory.

``create_lexical_index`` builds a full-text index on the text column of an
index collection, ``index_lexical_data_points`` writes the rows the index owns
(no embedding, NULL vector) and ``lexical_search`` ranks rows by BM25 with
higher scores better. Both the default subprocess mode and in-process mode
are covered; the compaction pass that folds new rows into the index needs
the pylance release line lancedb bundles and is skipped elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    import lancedb

    from cognee.infrastructure.databases.vector.lancedb import LanceDBAdapter as adapter_module
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        LEXICAL_TEXT_COLUMN,
        LanceDBAdapter,
    )
    from cognee_db_workers.lancedb_compaction import lance_core_mismatch

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

pytestmark = pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")

MODES = ["local", "subprocess"]


class _FakeEmbeddingEngine:
    """Deterministic 3-d embeddings, so a stored vector can be checked."""

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[float(len(text) % 5), 1.0, 0.5] for text in texts]


class Chunk(DataPoint):
    """A field that is embedded and lexical: rows with a vector, ``owns_rows=False``."""

    text: str
    chunk_index: int | None = None
    metadata: dict = {"index_fields": ["text"], "lexical_fields": ["text"]}


class Cached(DataPoint):
    """A lexical-only field: rows without a vector, ``owns_rows=True`` (a cached document)."""

    name: str
    text: str
    document_id: str | None = None
    metadata: dict = {"index_fields": [], "lexical_fields": ["text"]}


def _cached(name: str, text: str, **fields) -> Cached:
    return Cached(id=uuid5(NAMESPACE_OID, name), name=name, text=text, **fields)


@contextlib.asynccontextmanager
async def _adapter(mode: str, tmp_path):
    db_path = str(tmp_path / "db")
    engine = _FakeEmbeddingEngine()
    if mode == "subprocess":
        adapter = LanceDBAdapter.create_subprocess(
            url=db_path, api_key=None, embedding_engine=engine
        )
    else:
        adapter = LanceDBAdapter(url=db_path, api_key=None, embedding_engine=engine)
    try:
        yield adapter, db_path
    finally:
        await adapter.close()


async def _index_stats(db_path: str, collection: str):
    """The index statistics lancedb reports, read through a plain connection."""
    connection = await lancedb.connect_async(db_path)
    table = await connection.open_table(collection)
    return await table.index_stats(f"{LEXICAL_TEXT_COLUMN}_idx")


async def _index_cached(adapter, points: list[Cached]) -> None:
    await adapter.create_lexical_index("Cached", "text")
    await adapter.index_lexical_data_points("Cached", "text", points, owns_rows=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_lexical_only_rows_are_written_without_a_vector_and_found(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        assert adapter.supports_lexical_search is True
        pelican = _cached("pelican", "the pelican harbour report", document_id="doc-1")
        await _index_cached(adapter, [pelican, _cached("zebra", "zebra crossing manual")])

        results = await adapter.lexical_search("Cached_text", "pelican", limit=5)

        assert [result.id for result in results] == [pelican.id]
        assert results[0].score > 0
        assert results[0].payload["text"] == "the pelican harbour report"
        assert results[0].payload["document_id"] == "doc-1"
        [row] = await adapter.retrieve("Cached_text", [str(pelican.id)], include_vector=True)
        assert row.payload["vector"] is None
        # Vector search over the same collection skips the rows without a vector.
        assert await adapter.search("Cached_text", query_vector=[1.0, 1.0, 0.5], limit=5) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_embedded_rows_are_found_and_keep_their_vectors(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        chunk = Chunk(text="otters eat shellfish", chunk_index=0)
        await adapter.create_vector_index("Chunk", "text")
        await adapter.index_data_points("Chunk", "text", [chunk])
        await adapter.create_lexical_index("Chunk", "text")
        await adapter.index_lexical_data_points("Chunk", "text", [chunk], owns_rows=False)

        results = await adapter.lexical_search("Chunk_text", "shellfish")

        assert [result.id for result in results] == [chunk.id]
        [row] = await adapter.retrieve("Chunk_text", [str(chunk.id)], include_vector=True)
        assert list(row.payload["vector"]) == [float(len(chunk.text) % 5), 1.0, 0.5]
        assert [r.id for r in await adapter.search("Chunk_text", query_text="x", limit=1)] == [
            chunk.id
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_rare_term_outranks_a_common_one(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        rare = _cached("rare", "harbour pelican sighting")
        await _index_cached(
            adapter,
            [
                _cached("common-1", "harbour tide tables"),
                rare,
                _cached("common-2", "harbour master office hours"),
            ],
        )

        results = await adapter.lexical_search("Cached_text", "pelican harbour", limit=10)

        assert results[0].id == rare.id
        assert len(results) == 3
        assert results[0].score > results[1].score
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
        assert len(await adapter.lexical_search("Cached_text", "harbour", limit=2)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_deleted_row_disappears(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        gone = _cached("gone", "walrus tusk inventory")
        kept = _cached("kept", "walrus colony census")
        await _index_cached(adapter, [gone, kept])
        assert len(await adapter.lexical_search("Cached_text", "walrus")) == 2

        await adapter.delete_data_points("Cached_text", [gone.id])

        assert [r.id for r in await adapter.lexical_search("Cached_text", "walrus")] == [kept.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_missing_collection_returns_nothing(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        assert await adapter.lexical_search("Nothing_text", "anything") == []


@pytest.mark.asyncio
async def test_a_query_with_no_tokens_returns_nothing_without_a_query(tmp_path, monkeypatch):
    async with _adapter("local", tmp_path) as (adapter, _):
        await _index_cached(adapter, [_cached("one", "the only row")])

        async def _unexpected(*_args, **_kwargs):
            raise AssertionError("no query should reach the store")

        monkeypatch.setattr(adapter, "get_collection", _unexpected)
        for query in ("", "   ", "the and of", "!!!"):
            assert await adapter.lexical_search("Cached_text", query) == []
        assert await adapter.lexical_search("Cached_text", "row", limit=0) == []


@pytest.mark.asyncio
async def test_reindexing_replaces_the_row_and_merges_its_tags(tmp_path):
    async with _adapter("local", tmp_path) as (adapter, _):
        first = _cached("doc", "pelican first draft", belongs_to_set=["set-a"])
        await _index_cached(adapter, [first])
        second = _cached("doc", "pelican final version", belongs_to_set=["set-b"])
        await adapter.index_lexical_data_points("Cached", "text", [second], owns_rows=True)

        table = await adapter.get_collection("Cached_text")
        assert await table.count_rows() == 1
        assert await adapter.lexical_search("Cached_text", "draft") == []
        [result] = await adapter.lexical_search("Cached_text", "final")
        assert result.id == first.id
        assert result.payload["belongs_to_set"] == ["set-a", "set-b"]


@pytest.mark.asyncio
async def test_a_duplicate_id_in_one_batch_keeps_the_last_text_and_both_tags(tmp_path):
    async with _adapter("local", tmp_path) as (adapter, _):
        await _index_cached(
            adapter,
            [
                _cached("doc", "pelican one", belongs_to_set=["set-a"]),
                _cached("doc", "pelican two", belongs_to_set=["set-b"]),
            ],
        )
        [result] = await adapter.lexical_search("Cached_text", "pelican")
        assert result.payload["text"] == "pelican two"
        assert result.payload["belongs_to_set"] == ["set-a", "set-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_create_lexical_index_is_idempotent(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        await adapter.create_lexical_index("Cached", "text")
        await adapter.create_lexical_index("Cached", "text")
        await adapter.index_lexical_data_points(
            "Cached", "text", [_cached("one", "a row")], owns_rows=True
        )

        table = await adapter.get_collection("Cached_text")
        indices = await table.list_indices()
        assert [(index.index_type, list(index.columns)) for index in indices] == [
            ("FTS", [LEXICAL_TEXT_COLUMN])
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_stale_handle_does_not_race_into_a_second_index(mode, tmp_path):
    """Regression: every data item of a caching run indexes the same collection
    at once. A writer whose table handle was opened before another writer
    committed the index must see that index, not create it again (which Lance
    refuses with a commit conflict and which failed the whole caching run)."""
    async with _adapter(mode, tmp_path) as (adapter, _):
        await adapter.create_collection("Cached_text", payload_schema=adapter_module.IndexSchema)
        stale = await adapter.get_collection("Cached_text")
        await adapter.create_lexical_index("Cached", "text")

        await adapter._ensure_lexical_index("Cached_text", stale)

        table = await adapter.get_collection("Cached_text")
        assert [index.index_type for index in await table.list_indices()] == ["FTS"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_concurrent_writers_of_a_fresh_collection_all_succeed(mode, tmp_path):
    """The caching run's shape: many items write the same new collection at once."""
    async with _adapter(mode, tmp_path) as (adapter, _):

        async def write(i):
            point = _cached(f"doc-{i}", f"document {i} about the white rabbit")
            await adapter.create_lexical_index("Cached", "text")
            await adapter.index_lexical_data_points("Cached", "text", [point], owns_rows=True)

        await asyncio.gather(*(write(i) for i in range(8)))

        assert len(await adapter.lexical_search("Cached_text", "rabbit", limit=20)) == 8
        table = await adapter.get_collection("Cached_text")
        assert [index.index_type for index in await table.list_indices()] == ["FTS"]


@pytest.mark.asyncio
async def test_a_commit_conflict_is_answered_by_checking_again(tmp_path):
    """A creator outside this adapter's lock (another adapter, another process)
    wins the commit: the conflict is not an error once the index exists."""
    async with _adapter("local", tmp_path) as (adapter, _):
        await adapter.create_collection("Cached_text", payload_schema=adapter_module.IndexSchema)
        index = SimpleNamespace(index_type="FTS", columns=[LEXICAL_TEXT_COLUMN])
        handles = [
            SimpleNamespace(
                list_indices=AsyncMock(return_value=[]),
                create_index=AsyncMock(
                    side_effect=RuntimeError(
                        "lance error: Retryable commit conflict for version 2: This "
                        "CreateIndex transaction was preempted by concurrent transaction "
                        "CreateIndex at version 2. Please retry."
                    )
                ),
            ),
            SimpleNamespace(list_indices=AsyncMock(return_value=[index]), create_index=AsyncMock()),
        ]
        stale = SimpleNamespace(list_indices=AsyncMock(return_value=[]))
        with patch.object(adapter, "get_collection", AsyncMock(side_effect=handles)):
            await adapter._ensure_lexical_index("Cached_text", stale)

        handles[0].create_index.assert_awaited_once()
        handles[1].create_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_index_errors_are_not_swallowed(tmp_path):
    async with _adapter("local", tmp_path) as (adapter, _):
        handle = SimpleNamespace(
            list_indices=AsyncMock(return_value=[]),
            create_index=AsyncMock(side_effect=RuntimeError("lance error: disk full")),
        )
        with (
            patch.object(adapter, "get_collection", AsyncMock(return_value=handle)),
            pytest.raises(RuntimeError, match="disk full"),
        ):
            await adapter._ensure_lexical_index("Cached_text", handle)
        handle.create_index.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_has_lexical_index_reports_the_tables_index(mode, tmp_path):
    async with _adapter(mode, tmp_path) as (adapter, _):
        assert await adapter.has_lexical_index("Cached_text") is False  # no table
        await adapter.create_collection("Cached_text", payload_schema=adapter_module.IndexSchema)
        assert await adapter.has_lexical_index("Cached_text") is False  # table, no index
        await adapter.create_lexical_index("Cached", "text")
        assert await adapter.has_lexical_index("Cached_text") is True


@pytest.mark.asyncio
async def test_rows_added_after_the_index_was_built_are_searched_at_once(tmp_path):
    """LanceDB scans the rows an index does not cover yet, so nothing has to run
    between a write and a search; the compaction pass folds them in later. The
    driver creates the index before the first batch, i.e. over an empty table,
    so every row is such a row until the first pass."""
    async with _adapter("local", tmp_path) as (adapter, db_path):
        await _index_cached(adapter, [_cached("old", "pelican report")])
        await adapter.index_lexical_data_points(
            "Cached", "text", [_cached("new", "pelican update")], owns_rows=True
        )

        stats = await _index_stats(db_path, "Cached_text")
        assert (stats.num_indexed_rows, stats.num_unindexed_rows) == (0, 2)
        assert len(await adapter.lexical_search("Cached_text", "pelican")) == 2


@pytest.mark.asyncio
async def test_english_mode_stems_the_query(tmp_path, monkeypatch):
    async with _adapter("local", tmp_path) as (adapter, _):
        monkeypatch.setattr(adapter, "_lexical_language", lambda: "english")
        await _index_cached(adapter, [_cached("one", "runners were running the race")])

        assert len(await adapter.lexical_search("Cached_text", "run")) == 1
        # English stop words are the index's own: a stop-word query finds nothing.
        assert await adapter.lexical_search("Cached_text", "were") == []


@pytest.mark.asyncio
async def test_simple_mode_does_not_stem(tmp_path):
    async with _adapter("local", tmp_path) as (adapter, _):
        await _index_cached(adapter, [_cached("one", "Runners were running the Race")])

        assert await adapter.lexical_search("Cached_text", "run") == []
        assert len(await adapter.lexical_search("Cached_text", "RUNNING race")) == 1


@pytest.mark.asyncio
async def test_update_payload_keeps_rows_without_a_vector(tmp_path):
    async with _adapter("local", tmp_path) as (adapter, _):
        point = _cached("doc", "pelican report")
        await _index_cached(adapter, [point])

        await adapter.update_payload("Cached_text", {str(point.id): {"chunk_index": 7}})

        [row] = await adapter.retrieve("Cached_text", [str(point.id)], include_vector=True)
        assert row.payload["chunk_index"] == 7
        assert row.payload["vector"] is None
        assert len(await adapter.lexical_search("Cached_text", "pelican")) == 1


@pytest.mark.asyncio
async def test_the_index_data_points_task_drives_the_lexical_index(tmp_path):
    """The task passes the data point type and field, as it does for the embedding."""
    from cognee.tasks.storage.index_data_points import index_data_points

    async with _adapter("local", tmp_path) as (adapter, _):
        cached = _cached("doc", "cached pelican document")
        chunk = Chunk(text="chunk about a pelican", chunk_index=0)
        await index_data_points([cached, chunk], vector_engine=adapter)

        assert [r.id for r in await adapter.lexical_search("Cached_text", "pelican")] == [cached.id]
        assert [r.id for r in await adapter.lexical_search("Chunk_text", "pelican")] == [chunk.id]
        [row] = await adapter.retrieve("Chunk_text", [str(chunk.id)], include_vector=True)
        assert row.payload["vector"] is not None


def test_index_ops_are_wired_end_to_end():
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import (
        RemoteLanceDBTable,
        RemoteQuery,
    )
    from cognee_db_workers.lancedb_protocol import (
        OP_TABLE_CREATE_INDEX,
        OP_TABLE_LIST_INDICES,
        OP_TABLE_REFRESH_INVERTED_INDICES,
    )
    from cognee_db_workers.lancedb_worker import DISPATCH

    for op in (OP_TABLE_CREATE_INDEX, OP_TABLE_LIST_INDICES, OP_TABLE_REFRESH_INVERTED_INDICES):
        assert op in DISPATCH
    for name in ("create_index", "list_indices", "refresh_inverted_indices"):
        assert hasattr(RemoteLanceDBTable, name)
    for name in ("nearest_to_text", "select", "limit"):
        assert hasattr(RemoteQuery, name)


@pytest.mark.asyncio
async def test_worker_creates_and_lists_the_index_on_the_real_table(tmp_path):
    from cognee_db_workers.harness import HandleRegistry, Request
    from cognee_db_workers.lancedb_protocol import OP_TABLE_CREATE_INDEX, OP_TABLE_LIST_INDICES
    from cognee_db_workers.lancedb_worker import _op_table_create_index, _op_table_list_indices

    connection = await lancedb.connect_async(str(tmp_path / "db"))
    rows = [{"id": "1", "vector": None, "payload": {"text": "pelican harbour"}}]
    table = await connection.create_table("WorkerTable", data=rows)
    registry = HandleRegistry()
    handle_id = registry.register(table)

    with pytest.raises(ValueError, match="only 'FTS'"):
        await _op_table_create_index(
            registry,
            Request(
                op=OP_TABLE_CREATE_INDEX, handle_id=handle_id, args=("payload.text", "IvfPq", {})
            ),
        )
    await _op_table_create_index(
        registry,
        Request(
            op=OP_TABLE_CREATE_INDEX,
            handle_id=handle_id,
            args=("payload.text", "FTS", {"stem": False, "with_position": False}),
            kwargs={"replace": None, "name": None, "wait_timeout": None, "train": True},
        ),
    )

    listed = await _op_table_list_indices(
        registry, Request(op=OP_TABLE_LIST_INDICES, handle_id=handle_id)
    )
    assert listed == [
        {"name": "payload.text_idx", "index_type": "FTS", "columns": ["payload.text"]}
    ]
    hits = await table.query().nearest_to_text("pelican", columns=["payload.text"]).to_list()
    assert [hit["id"] for hit in hits] == ["1"]


needs_matching_pylance = pytest.mark.skipif(
    HAS_LANCEDB and lance_core_mismatch() is not None,
    reason="pylance is not on the Lance core lancedb bundles; compaction is off",
)


@needs_matching_pylance
@pytest.mark.asyncio
async def test_compaction_folds_new_rows_into_the_index_and_bounds_its_segments(
    tmp_path, monkeypatch
):
    """Every compaction pass folds the rows written since the last one into the
    full-text index, and once the index has the configured number of segments
    the pass merges them. The index is built over the empty table, as the
    driver does it; Lance stops listing that empty first segment after the
    first pass, so the merge comes once two real segments exist rather than on
    a fixed pass number."""
    monkeypatch.setattr(adapter_module, "DEFAULT_RETENTION_SECONDS", 0)
    monkeypatch.setattr(adapter_module, "DEFAULT_MAX_INDEX_SEGMENTS", 2)
    async with _adapter("local", tmp_path) as (adapter, db_path):
        await _index_cached(adapter, [_cached("doc-0", "pelican report 0")])
        merged = 0
        for i in range(1, 5):
            await adapter.index_lexical_data_points(
                "Cached", "text", [_cached(f"doc-{i}", f"pelican report {i}")], owns_rows=True
            )
            stats = await adapter.compact()

            assert stats["Cached_text"]["inverted_indices"] == 1
            merged += stats["Cached_text"]["inverted_indices_merged"]
            after = await _index_stats(db_path, "Cached_text")
            assert after.num_unindexed_rows == 0, f"pass {i} left rows unindexed"
            assert after.num_indices <= 3, f"pass {i} left {after.num_indices} segments"
            found = await adapter.lexical_search("Cached_text", "pelican", limit=20)
            assert len(found) == i + 1, f"pass {i} lost rows from the index"
        assert merged >= 1, "the segments were never merged"


@needs_matching_pylance
@pytest.mark.asyncio
async def test_refresh_inverted_indices_leaves_a_table_without_one_alone(tmp_path):
    from cognee_db_workers.lancedb_compaction import refresh_inverted_indices

    connection = await lancedb.connect_async(str(tmp_path / "db"))
    table = await connection.create_table(
        "Plain", data=[{"id": "1", "vector": [0.1, 0.2, 0.3], "payload": {"text": "x"}}]
    )
    dataset = await table.to_lance()
    versions = len(dataset.versions())

    assert refresh_inverted_indices(dataset) == {
        "inverted_indices": 0,
        "inverted_indices_merged": 0,
    }
    assert len(dataset.versions()) == versions


@needs_matching_pylance
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_fully_quoted_query_is_searched_by_its_words(mode, tmp_path, monkeypatch):
    """The index stores no positions, so a phrase query raises once rows are indexed."""
    monkeypatch.setattr(adapter_module, "DEFAULT_RETENTION_SECONDS", 0)
    async with _adapter(mode, tmp_path) as (adapter, db_path):
        report = _cached("report", "the quarterly revenue report")
        await _index_cached(adapter, [report, _cached("memo", "a memo about pelicans")])
        await adapter.compact()
        assert (await _index_stats(db_path, "Cached_text")).num_unindexed_rows == 0

        results = await adapter.lexical_search("Cached_text", '"quarterly revenue"', limit=5)

        assert [result.id for result in results] == [report.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tokens_longer_than_forty_characters_are_found(mode, tmp_path):
    """LanceDB drops tokens over 40 characters by default, a SHA-256 among them."""
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    async with _adapter(mode, tmp_path) as (adapter, _):
        release = _cached("release", f"release checksum {sha}")
        await _index_cached(adapter, [release, _cached("other", "another file")])

        results = await adapter.lexical_search("Cached_text", sha, limit=5)

        assert [result.id for result in results] == [release.id]


@pytest.mark.asyncio
async def test_a_remote_store_still_refreshes_the_indexes_it_wrote(tmp_path, monkeypatch):
    """No compaction on object storage, but the written tables' indexes take their new rows."""
    async with _adapter("local", tmp_path) as (adapter, _):
        await _index_cached(adapter, [_cached("doc", "pelican report")])
        refreshed = []

        async def fake_refresh(collection, max_segments):
            refreshed.append(collection.name)
            return {"inverted_indices": 1, "inverted_indices_merged": 0}

        monkeypatch.setattr(adapter, "_refresh_lexical_indices", fake_refresh)
        monkeypatch.setattr(adapter, "_store_url", lambda: "s3://bucket/cognee")

        first = await adapter.compact()
        second = await adapter.compact()

        assert refreshed == ["Cached_text"]
        assert first == {
            "skipped": "remote_store",
            "Cached_text": {"inverted_indices": 1, "inverted_indices_merged": 0},
        }
        assert second == {"skipped": "remote_store"}, "nothing was written since"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_schema_migration_keeps_rows_without_a_vector_and_their_index(mode, tmp_path):
    """The next payload-schema change rebuilds every table; lexical rows have no vector."""
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import IndexSchema

    class IndexSchemaNext(IndexSchema):
        added_in_a_later_release: str | None = None

    async with _adapter(mode, tmp_path) as (adapter, _):
        pelican = _cached("pelican", "the pelican harbour report")
        await _index_cached(adapter, [pelican, _cached("zebra", "zebra crossing manual")])
        old_collection = await adapter.get_collection("Cached_text")

        await adapter._migrate_collection_schema("Cached_text", old_collection, IndexSchemaNext, [])

        assert await adapter.has_lexical_index("Cached_text")
        results = await adapter.lexical_search("Cached_text", "pelican", limit=5)
        assert [result.id for result in results] == [pelican.id]
        [row] = await adapter.retrieve("Cached_text", [str(pelican.id)], include_vector=True)
        assert row.payload["vector"] is None
