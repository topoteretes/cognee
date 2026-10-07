"""LanceDB compaction: bounded, once per cognify run, fail-open.

Every ``merge_insert`` appends a fragment and leaves the superseded rows on
disk; LanceDB reclaims neither on its own (issue #4684). ``LanceDBAdapter.compact``
merges only small fragments, deletes a version only once its successor has
aged past the retention window, does a bounded number of tasks and version
deletions per run so a backlog drains over several runs, and never raises. These tests pin each of those
properties, plus the pylance/lancedb pairing the compaction depends on.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import BaseModel

try:
    from cognee.infrastructure.databases.vector.lancedb import LanceDBAdapter as adapter_module
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

pytestmark = pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Payload(BaseModel):
    slot: int
    label: str


def _data_files(db_path: str, collection_name: str) -> set:
    """On-disk data fragment files of a table (the thing that bloats)."""
    data_dir = Path(db_path) / f"{collection_name}.lance" / "data"
    return {p.name for p in data_dir.iterdir()} if data_dir.exists() else set()


def _referenced_files(db_path: str, collection_name: str) -> set:
    """Data files the LATEST version references (what a reader opening now gets)."""
    import lance

    dataset = lance.dataset(str(Path(db_path) / f"{collection_name}.lance"))
    return {df.path for fragment in dataset.get_fragments() for df in fragment.data_files()}


async def _fragment_count(adapter: LanceDBAdapter, collection_name: str) -> int:
    table = await adapter.get_collection(collection_name)
    return (await table.stats())["fragment_stats"]["num_fragments"]


async def _write_n_points(adapter: LanceDBAdapter, collection: str, n: int, start: int = 0) -> list:
    """One upsert per point: each one appends a fragment, like cognify's batches do."""
    ids = []
    for i in range(start, start + n):
        point_id = uuid4()
        ids.append(point_id)
        await adapter.upsert_raw_vectors(
            collection,
            [
                {
                    "id": point_id,
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"slot": i, "label": f"row-{i}"},
                }
            ],
            payload_schema=_Payload,
        )
    return ids


@pytest.fixture
def compaction_settings(monkeypatch):
    """Override the adapter's compaction constants (``DEFAULT_*``) for one test."""

    def configure(**settings):
        for key, value in settings.items():
            monkeypatch.setattr(adapter_module, f"DEFAULT_{key.upper()}", value)

    return configure


def _version_count(db_path: str, collection_name: str) -> int:
    import lance

    return len(lance.dataset(str(Path(db_path) / f"{collection_name}.lance")).versions())


def _adapter(tmp_path) -> tuple[LanceDBAdapter, str]:
    db_path = str(tmp_path / "db")
    return LanceDBAdapter(
        url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine()
    ), db_path


@pytest.mark.asyncio
async def test_compact_folds_fragments_and_keeps_every_row(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=0)
    adapter, db_path = _adapter(tmp_path)
    collection = "FoldTarget_label"
    ids = await _write_n_points(adapter, collection, 6)
    assert await _fragment_count(adapter, collection) == 6

    original = _data_files(db_path, collection)
    stats = await adapter.compact()

    assert stats[collection]["executed_tasks"] == 1
    assert await _fragment_count(adapter, collection) == 1
    table = await adapter.get_collection(collection)
    assert await table.count_rows() == 6
    assert len(await adapter.retrieve(collection, [str(i) for i in ids])) == 6
    # Retention 0: every superseded version and the files only it referenced
    # are gone in the same pass, not just unreferenced.
    assert stats[collection]["versions_pending"] == 0
    assert _data_files(db_path, collection).isdisjoint(original)
    assert _version_count(db_path, collection) == 1


@pytest.mark.asyncio
async def test_compact_is_a_cheap_no_op_on_a_compact_table(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=0)
    adapter, db_path = _adapter(tmp_path)
    collection = "NoopTarget_label"
    await _write_n_points(adapter, collection, 3)
    await adapter.compact()
    before = _data_files(db_path, collection)

    stats = await adapter.compact()

    assert stats[collection]["planned_tasks"] == 0
    assert stats[collection]["executed_tasks"] == 0
    assert stats[collection]["old_versions_removed"] == 0
    assert _data_files(db_path, collection) == before


@pytest.mark.asyncio
async def test_compact_leaves_fragments_at_target_size_alone(tmp_path, compaction_settings):
    """The property that makes it cheap on any disk: cold fragments are not rewritten."""
    compaction_settings(retention_seconds=0, target_rows_per_fragment=3)
    adapter, db_path = _adapter(tmp_path)
    collection = "ColdTarget_label"
    await _write_n_points(adapter, collection, 6)
    await adapter.compact()
    cold = _referenced_files(db_path, collection)
    assert len(cold) == 2  # two fragments of 3 rows

    await _write_n_points(adapter, collection, 2, start=6)
    await adapter.compact()

    assert cold <= _referenced_files(db_path, collection), "fragments at target size were rewritten"
    assert await _fragment_count(adapter, collection) == 3  # the two cold ones + one warm


@pytest.mark.asyncio
async def test_a_fragment_backlog_drains_over_several_passes(tmp_path, compaction_settings):
    """A bloated store is worked off a bit per cognify, not in one long stall."""
    compaction_settings(retention_seconds=0, target_rows_per_fragment=2, max_tasks_per_run=1)
    adapter, _ = _adapter(tmp_path)
    collection = "BacklogTarget_label"
    await _write_n_points(adapter, collection, 6)

    fragment_counts = []
    for _ in range(4):
        stats = await adapter.compact()
        fragment_counts.append(await _fragment_count(adapter, collection))
        assert stats[collection]["executed_tasks"] <= 1

    assert fragment_counts == [5, 4, 3, 3]
    table = await adapter.get_collection(collection)
    assert await table.count_rows() == 6


@pytest.mark.asyncio
async def test_a_version_backlog_drains_over_several_passes(tmp_path, compaction_settings):
    """Version cleanup is bounded too: the #4684 store had tens of thousands of
    versions, and deleting them all in one call outlived the worker's RPC timeout."""
    compaction_settings(retention_seconds=0, max_versions_per_run=2)
    adapter, db_path = _adapter(tmp_path)
    collection = "VersionBacklog_label"
    await _write_n_points(adapter, collection, 8)
    versions_before = _version_count(db_path, collection)
    assert versions_before >= 9

    first = await adapter.compact()

    assert first[collection]["old_versions_removed"] == 2
    assert first[collection]["versions_pending"] > 0

    passes = 1
    while (await adapter.compact())[collection]["versions_pending"] > 0:
        passes += 1
        assert passes < 20
    assert passes > 1
    assert _version_count(db_path, collection) == 1
    assert await (await adapter.get_collection(collection)).count_rows() == 8


@pytest.mark.asyncio
async def test_zero_budgets_drain_everything_in_one_pass(tmp_path, compaction_settings):
    compaction_settings(
        retention_seconds=0,
        target_rows_per_fragment=2,
        max_tasks_per_run=0,
        max_versions_per_run=0,
    )
    adapter, db_path = _adapter(tmp_path)
    collection = "UnlimitedTarget_label"
    await _write_n_points(adapter, collection, 6)

    stats = await adapter.compact()

    assert stats[collection]["executed_tasks"] == 3
    assert stats[collection]["versions_pending"] == 0
    assert await _fragment_count(adapter, collection) == 3
    assert _version_count(db_path, collection) == 1


@pytest.mark.asyncio
async def test_compaction_skips_remote_stores(compaction_settings):
    """On object storage every rewrite is network transfer; never do it implicitly."""
    compaction_settings()
    adapter = LanceDBAdapter(
        url="s3://bucket/cognee.lancedb", api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    assert await adapter.compact() == {"skipped": "remote_store"}


@pytest.mark.asyncio
async def test_default_retention_keeps_a_recent_reader_snapshot_readable(
    tmp_path, compaction_settings
):
    import lancedb

    compaction_settings()  # default retention window
    adapter, db_path = _adapter(tmp_path)
    collection = "SnapshotTarget_label"
    await _write_n_points(adapter, collection, 1)
    connection = await lancedb.connect_async(db_path)
    reader = await connection.open_table(collection)
    await reader.checkout(await reader.version())
    expected = (await reader.to_arrow()).to_pylist()
    await _write_n_points(adapter, collection, 2, start=1)

    await adapter.compact()

    # The merged fragment is new, the three it replaced are still on disk.
    assert len(_data_files(db_path, collection)) == 4
    assert (await reader.to_arrow()).to_pylist() == expected
    current = await adapter.get_collection(collection)
    assert await current.count_rows() == 3


@pytest.mark.asyncio
async def test_cleanup_waits_until_a_versions_successor_has_aged(
    tmp_path, compaction_settings, monkeypatch
):
    """The reader-safety rule: a version goes only once its SUCCESSOR is older than the window.

    Here the table idles past the window, a reader opens its latest version K,
    and compaction supersedes K. K's files must survive that pass and go in a
    later one. Lance is handed the versions to delete by number, so a delay
    between our clock read and the cleanup call cannot move the cut-off onto
    K: the cleanup is slowed down on purpose to prove it (with the earlier
    1 ms time margin, any delay above 1 ms deleted K's manifest, and a reader
    opening K after the pass failed).
    """
    import lancedb
    from lance.dataset import LanceDataset

    real_cleanup = LanceDataset.cleanup_old_versions

    def slow_cleanup(self, *args, **kwargs):
        import time

        time.sleep(0.05)  # the GIL handed to a busy event loop, for example
        return real_cleanup(self, *args, **kwargs)

    monkeypatch.setattr(LanceDataset, "cleanup_old_versions", slow_cleanup)
    compaction_settings(retention_seconds=1)
    adapter, db_path = _adapter(tmp_path)
    collection = "IdleTarget_label"
    await _write_n_points(adapter, collection, 3)
    await asyncio.sleep(1.5)  # idle: the latest version is now older than the window
    reader = await (await lancedb.connect_async(db_path)).open_table(collection)
    version_k = await reader.version()
    await reader.checkout(version_k)
    expected = (await reader.to_arrow()).to_pylist()
    before = _data_files(db_path, collection)

    first = await adapter.compact()

    assert first[collection]["executed_tasks"] == 1
    assert before <= _data_files(db_path, collection), "reader's files were deleted"
    assert (await reader.to_arrow()).to_pylist() == expected
    # A reader that resolved version K before the pass and opens it only now
    # (a fresh process, a second query of a long read) must still find it.
    late_reader = await (await lancedb.connect_async(db_path)).open_table(collection)
    await late_reader.checkout(version_k)
    assert (await late_reader.to_arrow()).to_pylist() == expected

    await asyncio.sleep(1.5)  # the superseding version has now aged past the window
    second = await adapter.compact()

    assert second[collection]["old_versions_removed"] >= 1
    assert _data_files(db_path, collection).isdisjoint(before)
    assert await (await adapter.get_collection(collection)).count_rows() == 3


@pytest.mark.asyncio
async def test_task_budget_is_shared_across_tables_and_rotates(tmp_path, compaction_settings):
    """`max_tasks_per_run` bounds one pass, not one table, and no table starves."""
    compaction_settings(retention_seconds=0, target_rows_per_fragment=2, max_tasks_per_run=4)
    adapter, _ = _adapter(tmp_path)
    tables = ["BudgetA_label", "BudgetB_label"]
    for name in tables:
        await _write_n_points(adapter, name, 6)  # 3 tasks each at a 2-row target

    first = await adapter.compact()
    executed = {name: first[name]["executed_tasks"] for name in tables}
    assert sum(executed.values()) == 4
    assert all(first[name]["planned_tasks"] == 3 for name in tables)
    starved = [name for name in tables if executed[name] < 3]
    assert len(starved) == 1
    # Its rewrite was cut short, so it stays first in line for the next pass.
    assert adapter._compaction_order(tables)[0] == starved[0]

    second = await adapter.compact()
    assert second[starved[0]]["executed_tasks"] == 3 - executed[starved[0]]
    for name in tables:
        assert await _fragment_count(adapter, name) == 3


@pytest.mark.asyncio
async def test_version_budget_is_shared_across_tables(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=0, max_versions_per_run=3)
    adapter, _ = _adapter(tmp_path)
    tables = ["VersionsA_label", "VersionsB_label"]
    for name in tables:
        await _write_n_points(adapter, name, 4)

    stats = await adapter.compact()

    assert sum(stats[name]["old_versions_removed"] for name in tables) == 3
    assert sum(stats[name]["versions_pending"] for name in tables) > 0


@pytest.mark.asyncio
async def test_tables_written_since_their_last_pass_go_first(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    names = ["OrderA_label", "OrderB_label", "OrderC_label"]
    for name in names:
        await _write_n_points(adapter, name, 1)
    await adapter.compact()
    assert adapter._compaction_dirty == set()

    await _write_n_points(adapter, "OrderC_label", 1)
    await adapter.delete_data_points("OrderB_label", [uuid4()])

    order = adapter._compaction_order(names)
    assert set(order[:2]) == {"OrderB_label", "OrderC_label"}
    assert order[2] == "OrderA_label"


@pytest.mark.asyncio
async def test_a_spent_budget_rotates_which_tables_go_first(tmp_path):
    """When the budget ends a pass early, the next pass starts elsewhere."""
    adapter, _ = _adapter(tmp_path)
    names = ["RotA", "RotB", "RotC"]
    starts = {adapter._compaction_order(names)[0] for _ in range(3)}
    assert starts == set(names)


@pytest.mark.asyncio
async def test_an_overlapping_pass_returns_instead_of_queueing(tmp_path, compaction_settings):
    compaction_settings()
    adapter, _ = _adapter(tmp_path)
    async with adapter._compaction_lock:
        assert await adapter.compact() == {"skipped": "in_progress"}


@pytest.mark.asyncio
async def test_fragment_rewrite_holds_the_write_lock_and_version_pruning_does_not(
    tmp_path, compaction_settings, monkeypatch
):
    """The rewrite commit conflicts with writers; pruning commits nothing and
    must not hold off upserts while it deletes a backlog of files."""
    adapter_module = importlib.import_module(
        "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
    )
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    await _write_n_points(adapter, "LockTarget_label", 3)
    lock_held = {}

    def recording(name, real):
        def wrapper(*args, **kwargs):
            lock_held[name] = adapter.VECTOR_DB_LOCK.locked()
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        adapter_module,
        "compact_fragments",
        recording("compact_fragments", adapter_module.compact_fragments),
    )
    monkeypatch.setattr(
        adapter_module,
        "prune_superseded_versions",
        recording("prune_superseded_versions", adapter_module.prune_superseded_versions),
    )

    await adapter.compact()

    assert lock_held == {"compact_fragments": True, "prune_superseded_versions": False}


@pytest.mark.asyncio
async def test_deletes_take_the_write_lock(tmp_path, monkeypatch):
    """A delete racing a compaction rewrite of the same fragments loses at commit."""
    adapter, _ = _adapter(tmp_path)
    collection = "DeleteLock_label"
    ids = await _write_n_points(adapter, collection, 2)
    real_get_collection = adapter.get_collection
    lock_held = []

    async def get_collection(name):
        table = await real_get_collection(name)
        real_delete = table.delete

        async def delete(predicate):
            lock_held.append(adapter.VECTOR_DB_LOCK.locked())
            return await real_delete(predicate)

        table.delete = delete
        return table

    monkeypatch.setattr(adapter, "get_collection", get_collection)

    await adapter.delete_data_points(collection, ids)

    assert lock_held == [True]


def _delete_right_after_open(adapter, db_path, collection, point_id, monkeypatch):
    """Commit a delete through another handle the first time the adapter opens ``collection``."""
    import lancedb

    real_get_collection = adapter.get_collection
    pending = [point_id]

    async def get_collection(name):
        table = await real_get_collection(name)
        if name == collection and pending:
            other = await (await lancedb.connect_async(db_path)).open_table(name)
            await other.delete(f"id = '{pending.pop()}'")
        return table

    monkeypatch.setattr(adapter, "get_collection", get_collection)


@pytest.mark.asyncio
async def test_a_delete_before_the_rewrite_takes_the_lock_does_not_fail_it(tmp_path, monkeypatch):
    """The rewrite plans against the version current under VECTOR_DB_LOCK, not the one
    the table handle was opened at."""
    adapter, db_path = _adapter(tmp_path)
    collection = "StaleHandle_label"
    ids = await _write_n_points(adapter, collection, 4)
    _delete_right_after_open(adapter, db_path, collection, ids[1], monkeypatch)

    stats = await adapter.compact()

    assert stats[collection]["executed_tasks"] == 1, stats
    assert len(await adapter.retrieve(collection, [str(i) for i in ids])) == 3


@pytest.mark.asyncio
async def test_compact_is_fail_open_per_collection(tmp_path, compaction_settings, monkeypatch):
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    await _write_n_points(adapter, "Healthy_label", 2)
    await _write_n_points(adapter, "Broken_label", 2)
    original = adapter._compact_collection

    async def flaky(name, *args):
        if name == "Broken_label":
            raise RuntimeError("boom")
        return await original(name, *args)

    monkeypatch.setattr(adapter, "_compact_collection", flaky)

    stats = await adapter.compact()

    assert "error" in stats["Broken_label"]
    assert stats["Healthy_label"]["executed_tasks"] == 1


@pytest.mark.asyncio
async def test_an_incompatible_pylance_turns_compaction_off(
    tmp_path, compaction_settings, monkeypatch
):
    """lancedb and pylance on different Lance cores: stop, warn once, never fail writes."""
    from cognee_db_workers.lancedb_compaction import PylanceIncompatibleError

    adapter_module = importlib.import_module(
        "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
    )
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    await _write_n_points(adapter, "PylanceA_label", 2)
    await _write_n_points(adapter, "PylanceB_label", 2)
    attempts = []

    async def incompatible(table):
        attempts.append(table)
        raise PylanceIncompatibleError("unsupported manifest version")

    monkeypatch.setattr(adapter_module, "open_as_lance", incompatible)

    first = await adapter.compact()

    assert len(attempts) == 1, "kept trying other tables after an incompatibility"
    assert [value for value in first.values() if "error" in value]
    assert await adapter.compact() == {"skipped": "pylance_incompatible"}
    await _write_n_points(adapter, "PylanceA_label", 1, start=2)  # writes still work


@pytest.mark.asyncio
async def test_proxy_sends_both_compaction_halves_without_a_deadline():
    """Neither half may be re-issued on a timeout while the first attempt still
    runs in the worker, so they go without one; their task/version caps bound them."""
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import RemoteLanceDBTable
    from cognee_db_workers.lancedb_protocol import (
        OP_TABLE_COMPACT_FRAGMENTS,
        OP_TABLE_PRUNE_VERSIONS,
    )

    session = Mock()
    session.call_async = AsyncMock(return_value=Mock(result={"executed_tasks": 1}))
    table = RemoteLanceDBTable(session, 17, "WorkerTarget_label")

    assert await table.compact_fragments(target_rows_per_fragment=20_000, max_tasks=4) == {
        "executed_tasks": 1
    }
    request = session.call_async.await_args.args[0]
    assert (request.op, request.handle_id) == (OP_TABLE_COMPACT_FRAGMENTS, 17)
    assert request.kwargs == {"target_rows_per_fragment": 20_000, "max_tasks": 4}
    assert session.call_async.await_args.kwargs == {"timeout": None}

    await table.prune_versions(retention_seconds=300, max_versions=1_000)
    request = session.call_async.await_args.args[0]
    assert request.op == OP_TABLE_PRUNE_VERSIONS
    assert request.kwargs == {"retention_seconds": 300, "max_versions": 1_000}
    assert session.call_async.await_args.kwargs == {"timeout": None}


@pytest.mark.asyncio
async def test_proxy_optimize_forwards_lancedbs_own_arguments():
    from datetime import timedelta

    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import RemoteLanceDBTable
    from cognee_db_workers.lancedb_protocol import OP_TABLE_OPTIMIZE

    session = Mock()
    session.call_async = AsyncMock(return_value=Mock(result=None))
    table = RemoteLanceDBTable(session, 3, "Optimize_label")

    await table.optimize(cleanup_older_than=timedelta(days=1), retrain=True)

    request = session.call_async.await_args.args[0]
    assert request.op == OP_TABLE_OPTIMIZE
    assert request.kwargs == {
        "cleanup_older_than": timedelta(days=1),
        "delete_unverified": False,
        "retrain": True,
    }


@pytest.mark.asyncio
async def test_worker_runs_both_compaction_halves_on_the_real_table(tmp_path):
    from datetime import timedelta

    import lancedb

    from cognee_db_workers.harness import HandleRegistry, Request
    from cognee_db_workers.lancedb_protocol import (
        OP_TABLE_COMPACT_FRAGMENTS,
        OP_TABLE_OPTIMIZE,
        OP_TABLE_PRUNE_VERSIONS,
    )
    from cognee_db_workers.lancedb_worker import (
        _op_table_compact_fragments,
        _op_table_optimize,
        _op_table_prune_versions,
    )

    connection = await lancedb.connect_async(str(tmp_path / "db"))
    rows = [{"id": str(i), "vector": [0.1, 0.2, 0.3], "payload": {"slot": i}} for i in range(5)]
    table = await connection.create_table("WorkerTable", data=rows[:1])
    for row in rows[1:]:
        await table.add([row])
    assert (await table.stats())["fragment_stats"]["num_fragments"] == 5
    registry = HandleRegistry()
    handle_id = registry.register(table)

    stats = await _op_table_compact_fragments(
        registry,
        Request(
            op=OP_TABLE_COMPACT_FRAGMENTS,
            handle_id=handle_id,
            kwargs={"target_rows_per_fragment": 20_000, "max_tasks": 0},
        ),
    )
    assert stats["executed_tasks"] == 1
    assert (await table.stats())["fragment_stats"]["num_fragments"] == 1

    pruned = await _op_table_prune_versions(
        registry,
        Request(
            op=OP_TABLE_PRUNE_VERSIONS,
            handle_id=handle_id,
            kwargs={"retention_seconds": 0, "max_versions": 0},
        ),
    )
    assert pruned["old_versions_removed"] >= 5
    assert pruned["versions_pending"] == 0
    assert await table.count_rows() == 5

    # lancedb's own optimize, with its own arguments, no stats returned.
    assert (
        await _op_table_optimize(
            registry,
            Request(
                op=OP_TABLE_OPTIMIZE,
                handle_id=handle_id,
                kwargs={"cleanup_older_than": timedelta(days=7)},
            ),
        )
        is None
    )
    assert (
        await _op_table_optimize(registry, Request(op=OP_TABLE_OPTIMIZE, handle_id=handle_id))
        is None
    )


@pytest.mark.asyncio
async def test_pylance_reads_the_tables_lancedb_writes(tmp_path, compaction_settings):
    """Guards the lancedb/pylance release-line pairing in pyproject.

    lancedb bundles its own Lance core; pylance must be built on the same one or
    ``to_lance()`` (which the compaction relies on) cannot decode the files
    lancedb wrote. A failure here means the two drifted apart.
    """
    from cognee_db_workers.lancedb_compaction import open_as_lance

    compaction_settings()
    adapter, _ = _adapter(tmp_path)
    collection = "PinPair_label"
    await _write_n_points(adapter, collection, 3)
    table = await adapter.get_collection(collection)

    dataset = await open_as_lance(table)

    assert dataset.to_table().num_rows == 3


def _engine_getter(engine):
    async def get_vector_engine_async():
        return engine

    return get_vector_engine_async


@pytest.mark.asyncio
async def test_compact_vector_store_never_fails_the_pipeline(monkeypatch):
    module = importlib.import_module("cognee.infrastructure.databases.vector.compact_vector_store")

    class Exploding:
        async def compact(self):
            raise RuntimeError("disk on fire")

    class Quiet:
        async def compact(self):
            return {"Entity_name": {"planned_tasks": 1, "executed_tasks": 1}}

    async def unavailable():
        raise RuntimeError("no vector engine")

    monkeypatch.setattr(module, "get_vector_engine_async", _engine_getter(Exploding()))
    assert await module.compact_vector_store() is None

    monkeypatch.setattr(module, "get_vector_engine_async", unavailable)
    assert await module.compact_vector_store() is None

    monkeypatch.setattr(module, "get_vector_engine_async", _engine_getter(Quiet()))
    assert await module.compact_vector_store() == {
        "Entity_name": {"planned_tasks": 1, "executed_tasks": 1}
    }


@pytest.mark.asyncio
async def test_compact_vector_store_uses_the_async_engine_getter(monkeypatch):
    """The deprecated sync getter warns on every run and fails under -W error."""
    import warnings

    module = importlib.import_module("cognee.infrastructure.databases.vector.compact_vector_store")
    assert not hasattr(module, "get_vector_engine")

    class Quiet:
        async def compact(self):
            return {}

    monkeypatch.setattr(module, "get_vector_engine_async", _engine_getter(Quiet()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert await module.compact_vector_store() == {}


@pytest.mark.asyncio
async def test_compact_runs_end_to_end_inside_the_subprocess_worker(
    tmp_path, monkeypatch, compaction_settings
):
    """Subprocess mode is the default: the whole compaction must work through the worker."""
    from cognee.infrastructure.databases.vector.config import VectorConfig

    compaction_settings(retention_seconds=0)
    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    db_path = str(tmp_path / "db")
    config = VectorConfig(
        vector_db_provider="lancedb",
        vector_db_url=db_path,
        vector_db_name="worker_compaction",
        vector_db_subprocess_enabled=True,
    )
    adapter = factory.create_vector_engine(**config.to_dict())
    try:
        assert adapter._subprocess_mode
        collection = "WorkerEndToEnd_label"
        ids = await _write_n_points(adapter, collection, 5)
        # The proxy exposes no ``stats``; the files on disk are the ground truth anyway.
        assert len(_data_files(db_path, collection)) == 5

        stats = await adapter.compact()

        assert stats[collection]["executed_tasks"] == 1
        assert stats[collection]["fragments_removed"] == 5
        assert stats[collection]["versions_pending"] == 0
        assert len(_data_files(db_path, collection)) == 1
        assert len(await adapter.retrieve(collection, [str(i) for i in ids])) == 5
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_compact_vector_store_lets_a_running_pass_finish_before_cancelling(monkeypatch):
    """A cancel -- even a repeated one -- must not leave the pass running against
    the adapter teardown that follows it."""
    module = importlib.import_module("cognee.infrastructure.databases.vector.compact_vector_store")
    started = asyncio.Event()
    finished = False

    class Slow:
        async def compact(self):
            nonlocal finished
            started.set()
            await asyncio.sleep(0.3)
            finished = True
            return {}

    monkeypatch.setattr(module, "get_vector_engine_async", _engine_getter(Slow()))
    task = asyncio.ensure_future(module.compact_vector_store())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()  # e.g. shutdown cancelling again while the first cancel waits
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished, "cancellation propagated while the compaction pass was still running"


class _FakeVersions:
    def __init__(self, stamps):
        self._stamps = stamps

    def versions(self):
        return [
            {"version": number, "timestamp": stamp}
            for number, stamp in enumerate(self._stamps, start=1)
        ]


@pytest.fixture
def belgrade_time(monkeypatch):
    """Local time in a zone with DST, so pylance's naive timestamps cross shifts."""
    import os
    import time

    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    monkeypatch.setenv("TZ", "Europe/Belgrade")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()
    assert os.environ.get("TZ") != "Europe/Belgrade"


def _pylance_stamp(epoch_seconds: float):
    """A timestamp built the way pylance's ``versions()`` builds it."""
    from datetime import datetime, timedelta

    return datetime.fromtimestamp(int(epoch_seconds)) + timedelta(microseconds=5)  # noqa: DTZ006


@pytest.mark.parametrize(
    "successor_epoch",
    [
        pytest.param(1_774_746_000 - 1, id="clocks_forward"),  # 2026-03-29 01:59:59 CET
        pytest.param(1_792_888_200 + 3_600, id="clocks_back_repeated_hour"),  # 02:30 CET
    ],
)
def test_version_age_survives_a_dst_change(belgrade_time, monkeypatch, successor_epoch):
    """A successor committed 10 s ago must not look an hour older across a DST shift.

    pylance's timestamps are naive local time with ``fold`` reset, so naive
    subtraction (or ``.timestamp()`` alone, in the repeated hour) reads them an
    hour too old -- and would delete a version a reader may still hold.
    """
    from cognee_db_workers import lancedb_compaction

    monkeypatch.setattr(lancedb_compaction.time, "time", lambda: successor_epoch + 10)
    dataset = _FakeVersions([_pylance_stamp(successor_epoch - 60), _pylance_stamp(successor_epoch)])

    assert lancedb_compaction.removable_versions(dataset, retention_seconds=300) == []
    assert lancedb_compaction.removable_versions(dataset, retention_seconds=5) == [1]


def test_lance_core_mapping_rejects_a_pair_on_different_cores(monkeypatch):
    import lance
    import lancedb

    from cognee_db_workers import lancedb_compaction

    assert lancedb_compaction.lance_core_mismatch() is None  # the installed, pinned pair

    monkeypatch.setattr(lance, "__version__", "12.1.0")  # newer core than lancedb's
    assert "pylance 12.1.0" in lancedb_compaction.lance_core_mismatch()

    monkeypatch.setattr(lance, "__version__", "12.0.3")
    monkeypatch.setattr(lancedb, "__version__", "0.40.0")  # no known bundled core
    assert "lancedb 0.40.0" in lancedb_compaction.lance_core_mismatch()


@pytest.mark.asyncio
async def test_a_mismatched_lance_core_turns_compaction_off(
    tmp_path, compaction_settings, monkeypatch
):
    adapter_module = importlib.import_module(
        "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter"
    )
    monkeypatch.setattr(
        adapter_module, "lance_core_mismatch", lambda: "pylance 12.1.0 is not on Lance 12.0.x"
    )
    compaction_settings(retention_seconds=0)
    adapter, db_path = _adapter(tmp_path)
    collection = "CoreMismatch_label"
    await _write_n_points(adapter, collection, 3)  # writes are unaffected

    assert await adapter.compact() == {"skipped": "lance_core_mismatch"}
    assert adapter._open_prune_task is None
    assert len(_data_files(db_path, collection)) == 3


async def _reopen(tmp_path) -> LanceDBAdapter:
    """A second adapter on the same store: a new process, or a re-created engine."""
    adapter, _ = _adapter(tmp_path)
    await adapter.get_connection()
    return adapter


@pytest.mark.asyncio
async def test_opening_a_quiet_store_reclaims_its_superseded_versions(
    tmp_path, compaction_settings
):
    """The files a pass supersedes go only in a later pass; a store that gets no
    more writes (searches only) must still get them back."""
    compaction_settings(retention_seconds=0)
    writer, db_path = _adapter(tmp_path)
    collection = "QuietStore_label"
    await _write_n_points(writer, collection, 6)
    await writer.close()
    assert _version_count(db_path, collection) >= 7

    reader = await _reopen(tmp_path)
    await asyncio.wait({reader._open_prune_task})

    assert reader._open_prune_task.result()[collection]["old_versions_removed"] >= 6
    assert _version_count(db_path, collection) == 1
    assert await (await reader.get_collection(collection)).count_rows() == 6
    await reader.close()


@pytest.mark.asyncio
async def test_the_open_prune_is_bounded_like_a_pass(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=0, max_versions_per_run=2)
    writer, db_path = _adapter(tmp_path)
    collection = "OpenBudget_label"
    await _write_n_points(writer, collection, 6)
    await writer.close()
    before = _version_count(db_path, collection)

    reader = await _reopen(tmp_path)
    await asyncio.wait({reader._open_prune_task})

    assert _version_count(db_path, collection) == before - 2
    await reader.close()


@pytest.mark.asyncio
async def test_compact_and_close_wait_for_a_running_open_prune(
    tmp_path, compaction_settings, monkeypatch
):
    """The two never overlap, and closing never tears the store down under it."""
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    started, finished = asyncio.Event(), []

    async def slow_prune_pass(options):
        started.set()
        await asyncio.sleep(0.2)
        finished.append(True)
        return {}

    monkeypatch.setattr(adapter, "_prune_pass", slow_prune_pass)
    await adapter.get_connection()
    await started.wait()

    stats = await adapter.compact()

    assert finished == [True]
    assert "skipped" not in stats

    second, _ = _adapter(tmp_path)
    monkeypatch.setattr(second, "_prune_pass", slow_prune_pass)
    started.clear()
    await second.get_connection()
    await started.wait()
    await second.close()
    assert finished == [True, True]
    await adapter.close()


@pytest.mark.asyncio
async def test_prune_never_drops_tables_under_a_running_open_prune(
    tmp_path, compaction_settings, monkeypatch
):
    compaction_settings(retention_seconds=0)
    writer, _ = _adapter(tmp_path)
    await _write_n_points(writer, "PruneRace_label", 2)
    await writer.close()

    adapter, _ = _adapter(tmp_path)
    started, events = asyncio.Event(), []

    async def slow_prune_pass(options):
        started.set()
        await asyncio.sleep(0.2)
        events.append("open_prune_done")
        return {}

    monkeypatch.setattr(adapter, "_prune_pass", slow_prune_pass)
    await adapter.get_connection()
    await started.wait()

    await adapter.prune()
    events.append("pruned")

    assert events == ["open_prune_done", "pruned"]
    assert not await adapter.has_collection("PruneRace_label")
    await adapter.close()


@pytest.mark.asyncio
async def test_prune_on_a_fresh_adapter_starts_no_open_prune(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=0)
    writer, _ = _adapter(tmp_path)
    await _write_n_points(writer, "PruneFresh_label", 2)
    await writer.close()

    adapter, _ = _adapter(tmp_path)
    await adapter.prune()

    assert adapter._open_prune_task is None
    await adapter.close()


@pytest.mark.asyncio
async def test_prune_waits_for_a_running_compaction_pass(tmp_path, compaction_settings):
    """A pass's version cleanup runs outside VECTOR_DB_LOCK; prune must not drop
    the tables under it."""
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    await _write_n_points(adapter, "PrunePass_label", 2)
    events = []

    async with adapter._compaction_lock:  # a pass in progress
        prune_task = asyncio.ensure_future(adapter.prune())
        await asyncio.sleep(0.1)
        assert not prune_task.done()
        events.append("pass_done")
    await prune_task
    events.append("pruned")

    assert events == ["pass_done", "pruned"]
    await adapter.close()


@pytest.mark.asyncio
async def test_versions_a_pass_left_pending_are_pruned_once_aged(tmp_path, compaction_settings):
    """An adapter that stays open (a server) reclaims them without another cognify."""
    compaction_settings(retention_seconds=1)
    adapter, db_path = _adapter(tmp_path)
    collection = "Followup_label"
    await _write_n_points(adapter, collection, 4)

    await adapter.compact()
    assert _version_count(db_path, collection) > 1  # still inside the retention window
    assert adapter._followup_prune_handle is not None

    await asyncio.sleep(2.3)
    await adapter._wait_for_open_prune()
    assert _version_count(db_path, collection) == 1
    assert _data_files(db_path, collection) == _referenced_files(db_path, collection)

    await adapter.compact()
    assert adapter._followup_prune_handle is not None
    await adapter.close()
    assert adapter._followup_prune_handle is None


@pytest.mark.asyncio
async def test_a_followup_prune_waits_for_a_running_prune(
    tmp_path, compaction_settings, monkeypatch
):
    """The follow-up takes the prune slot; whoever waits on the slot must still
    wait for the prune it replaced (both hold the compaction lock in turn)."""
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    started, events = asyncio.Event(), []

    async def slow_prune_pass(options):
        started.set()
        await asyncio.sleep(0.2)
        events.append("prune_done")
        return {}

    monkeypatch.setattr(adapter, "_prune_pass", slow_prune_pass)
    await adapter.get_connection()  # first-open prune starts
    await started.wait()
    first = adapter._open_prune_task

    adapter._start_followup_prune(adapter._compaction_options())
    assert adapter._open_prune_task is not first

    await adapter._wait_for_open_prune()

    assert first.done()
    assert events == ["prune_done", "prune_done"]
    await adapter.close()


@pytest.mark.asyncio
async def test_prunes_are_drained_by_wait_for_background_tasks(
    tmp_path, compaction_settings, monkeypatch
):
    """The server's shutdown drain (and scripts) must not cut a prune off."""
    from cognee.infrastructure.background_tasks import wait_for_background_tasks

    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    finished = []

    async def slow_prune_pass(options):
        await asyncio.sleep(0.2)
        finished.append(True)
        return {}

    monkeypatch.setattr(adapter, "_prune_pass", slow_prune_pass)
    await adapter.get_connection()

    assert await wait_for_background_tasks(timeout=5)
    assert finished == [True]
    await adapter.close()


@pytest.mark.asyncio
async def test_prune_cancels_a_pending_followup(tmp_path, compaction_settings):
    compaction_settings(retention_seconds=60)
    adapter, _ = _adapter(tmp_path)
    await _write_n_points(adapter, "FollowupPrune_label", 2)
    await adapter.compact()
    handle = adapter._followup_prune_handle
    assert handle is not None

    await adapter.prune()

    assert handle.cancelled()
    assert adapter._followup_prune_handle is None
    await adapter.close()
