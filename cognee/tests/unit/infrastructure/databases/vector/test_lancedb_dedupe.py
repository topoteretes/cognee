"""Repair of LanceDB collections that store one id in several rows.

Writers that raced before the store-wide writer lock could each insert the same
new id; an upsert then updates every copy instead of collapsing them. These
tests seed such duplicates directly and check ``dedupe_collection`` keeps one
row per id without losing data.
"""

from __future__ import annotations

import hashlib

import lance
import pyarrow as pa
import pytest

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.engine import DataPoint

COLLECTION = "Note_text"


class _HashEmbeddingEngine:
    """Deterministic, picklable embeddings: same text, same vector."""

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([digest[0] / 255 + 0.01, digest[1] / 255 + 0.01, digest[2] / 255 + 0.01])
        return vectors


class Note(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


def _adapter(url: str) -> LanceDBAdapter:
    return LanceDBAdapter(url=url, api_key=None, embedding_engine=_HashEmbeddingEngine())


async def _rows(adapter: LanceDBAdapter) -> list[dict]:
    collection = await adapter.get_collection(COLLECTION)
    return await collection.query().with_row_id().to_list()


async def _add_copies(adapter: LanceDBAdapter, rows: list[dict]) -> None:
    """Append rows verbatim, bypassing merge_insert, as a racing writer would."""
    collection = await adapter.get_collection(COLLECTION)
    schema = await collection.schema()
    await collection.add(pa.Table.from_pylist(rows, schema=schema))


def _copy(row: dict, **payload_changes) -> dict:
    payload = {**row["payload"], **payload_changes}
    return {"id": row["id"], "vector": list(row["vector"]), "payload": payload}


@pytest.mark.asyncio
async def test_clean_and_missing_collections_report_no_duplicates(tmp_path):
    adapter = _adapter(str(tmp_path / "db"))
    assert await adapter.count_duplicate_ids("Missing_text") == 0
    assert (await adapter.dedupe_collection("Missing_text"))["rows_removed"] == 0

    await adapter.create_data_points(COLLECTION, [Note(text="a"), Note(text="b")])
    assert await adapter.count_duplicate_ids(COLLECTION) == 0
    collection = await adapter.get_collection(COLLECTION)
    version = await collection.version()
    report = await adapter.dedupe_collection(COLLECTION)
    assert report == {"duplicate_ids": 0, "rows_removed": 0, "ambiguous_ids": []}
    assert await (await adapter.get_collection(COLLECTION)).version() == version


@pytest.mark.asyncio
async def test_upsert_does_not_collapse_existing_duplicates(tmp_path):
    """The premise of the repair: merge_insert updates every copy of an id."""
    adapter = _adapter(str(tmp_path / "db"))
    note = Note(text="a")
    await adapter.create_data_points(COLLECTION, [note])
    await _add_copies(adapter, [_copy((await _rows(adapter))[0])])
    await adapter.create_data_points(COLLECTION, [note])
    assert await adapter.count_duplicate_ids(COLLECTION) == 1


@pytest.mark.asyncio
async def test_keeps_newest_copy_even_when_compaction_reordered_row_ids(tmp_path):
    """Highest ``_rowid`` is not the latest write once compaction ran.

    The older copy sits in a fragment with a deletion; compaction rewrites that
    fragment into a new, higher-numbered one, so the older copy ends up with
    the higher ``_rowid``. The newer ``updated_at`` must still win.
    """
    url = str(tmp_path / "db")
    adapter = _adapter(url)
    old = Note(text="old text", belongs_to_set=["dataset_a"])
    filler = Note(text="filler")
    await adapter.create_data_points(COLLECTION, [old, filler])
    old_row = next(row for row in await _rows(adapter) if row["id"] == str(old.id))

    newer = _copy(old_row, text="new text", updated_at=old_row["payload"]["updated_at"] + 1000)
    newer["payload"]["belongs_to_set"] = ["dataset_b"]
    newer["vector"] = [0.9, 0.1, 0.1]
    await _add_copies(adapter, [newer])
    await adapter.delete_data_points(COLLECTION, [filler.id])
    lance.dataset(f"{url}/{COLLECTION}.lance").optimize.compact_files(target_rows_per_fragment=1)

    copies = sorted(
        (row for row in await _rows(adapter) if row["id"] == str(old.id)),
        key=lambda row: row["_rowid"],
    )
    assert [row["payload"]["text"] for row in copies] == ["new text", "old text"]

    report = await adapter.dedupe_collection(COLLECTION)

    assert report == {"duplicate_ids": 1, "rows_removed": 1, "ambiguous_ids": []}
    (kept,) = await _rows(adapter)
    assert kept["payload"]["text"] == "new text"
    assert kept["vector"] == pytest.approx([0.9, 0.1, 0.1])
    assert kept["payload"]["belongs_to_set"] == ["dataset_b", "dataset_a"]


@pytest.mark.asyncio
async def test_repair_preserves_search_results_and_is_idempotent(tmp_path):
    adapter = _adapter(str(tmp_path / "db"))
    notes = [Note(text=f"note {index}", belongs_to_set=["main"]) for index in range(6)]
    await adapter.create_data_points(COLLECTION, notes)
    rows = await _rows(adapter)
    # Two extra copies of one id, one extra copy of another, all identical.
    await _add_copies(adapter, [_copy(rows[0]), _copy(rows[0]), _copy(rows[3])])
    assert await adapter.count_duplicate_ids(COLLECTION) == 3

    async def ranking():
        results = await adapter.search(COLLECTION, query_text="note 2", limit=20)
        unique = list(dict.fromkeys(str(result.id) for result in results))
        scores = {str(result.id): result.score for result in results}
        return unique, scores

    before_ids, before_scores = await ranking()

    report = await adapter.dedupe_collection(COLLECTION)

    assert report == {"duplicate_ids": 2, "rows_removed": 3, "ambiguous_ids": []}
    assert await adapter.count_duplicate_ids(COLLECTION) == 0
    after_ids, after_scores = await ranking()
    assert after_ids == before_ids
    assert after_scores == pytest.approx(before_scores)
    assert len(await _rows(adapter)) == len(notes)

    version = await (await adapter.get_collection(COLLECTION)).version()
    assert (await adapter.dedupe_collection(COLLECTION))["rows_removed"] == 0
    assert await (await adapter.get_collection(COLLECTION)).version() == version


@pytest.mark.asyncio
async def test_dry_run_reports_without_writing(tmp_path):
    adapter = _adapter(str(tmp_path / "db"))
    await adapter.create_data_points(COLLECTION, [Note(text="a")])
    await _add_copies(adapter, [_copy((await _rows(adapter))[0])])
    version = await (await adapter.get_collection(COLLECTION)).version()

    report = await adapter.dedupe_collection(COLLECTION, dry_run=True)

    assert report == {"duplicate_ids": 1, "rows_removed": 1, "ambiguous_ids": []}
    assert await (await adapter.get_collection(COLLECTION)).version() == version
    assert await adapter.count_duplicate_ids(COLLECTION) == 1


@pytest.mark.asyncio
async def test_differing_copies_without_a_newer_timestamp_are_reported(tmp_path):
    adapter = _adapter(str(tmp_path / "db"))
    await adapter.create_data_points(COLLECTION, [Note(text="it's quoted")])
    row = (await _rows(adapter))[0]
    await _add_copies(adapter, [_copy(row, text="same timestamp, other text")])

    report = await adapter.dedupe_collection(COLLECTION)

    assert report["ambiguous_ids"] == [row["id"]]
    assert report["rows_removed"] == 1
    assert len(await _rows(adapter)) == 1


@pytest.mark.asyncio
async def test_interrupted_repair_loses_nothing_and_resumes(tmp_path, monkeypatch):
    """A crash between the rewrite and the delete leaves identical copies only."""
    adapter = _adapter(str(tmp_path / "db"))
    note = Note(text="a", belongs_to_set=["dataset_a"])
    await adapter.create_data_points(COLLECTION, [note])
    row = (await _rows(adapter))[0]
    newer = _copy(row, text="b", updated_at=row["payload"]["updated_at"] + 1)
    newer["payload"]["belongs_to_set"] = ["dataset_b"]
    await _add_copies(adapter, [newer])

    real_get_collection = adapter.get_collection

    class _CrashOnDelete:
        def __init__(self, collection):
            self._collection = collection

        def __getattr__(self, name):
            return getattr(self._collection, name)

        async def delete(self, _predicate):
            raise RuntimeError("simulated crash")

    async def crashing_get_collection(name):
        return _CrashOnDelete(await real_get_collection(name))

    monkeypatch.setattr(adapter, "get_collection", crashing_get_collection)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await adapter.dedupe_collection(COLLECTION)
    monkeypatch.setattr(adapter, "get_collection", real_get_collection)

    copies = await _rows(adapter)
    assert len(copies) == 2
    for copy in copies:
        assert copy["payload"]["text"] == "b"
        assert copy["payload"]["belongs_to_set"] == ["dataset_b", "dataset_a"]

    report = await adapter.dedupe_collection(COLLECTION)
    assert report["rows_removed"] == 1
    (kept,) = await _rows(adapter)
    assert kept["payload"]["text"] == "b"


@pytest.mark.asyncio
async def test_opening_a_store_with_duplicates_flags_each_collection(tmp_path):
    url = str(tmp_path / "db")
    writer = _adapter(url)
    await writer.create_data_points(COLLECTION, [Note(text="a"), Note(text="b")])
    await writer.create_data_points("Clean_text", [Note(text="c")])
    await _add_copies(writer, [_copy(row) for row in await _rows(writer)])
    await writer.close()

    reader = _adapter(url)
    await reader.get_connection()
    assert reader._duplicate_check_task is not None
    assert await reader._duplicate_check_task == {COLLECTION: 2}
    await reader.close()
