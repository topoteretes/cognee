"""LanceDB compaction: bounded, once per pipeline run, fail-open.

Every ``merge_insert`` appends a fragment and leaves the superseded rows on
disk; LanceDB reclaims neither on its own (issue #4684). ``LanceDBAdapter.compact``
merges only small fragments, a bounded number of tasks per run, keeps old
files for a retention window, and never raises. These tests pin each of those
properties, plus the pylance/lancedb pin pair the compaction depends on.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import BaseModel

try:
    from cognee.infrastructure.databases.vector.config import get_vectordb_config
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
    """Override VECTOR_DB_COMPACTION_* for one test; the config cache is cleared both ways."""

    def configure(**settings):
        for key, value in settings.items():
            monkeypatch.setenv(f"VECTOR_DB_COMPACTION_{key.upper()}", str(value))
        get_vectordb_config.cache_clear()

    get_vectordb_config.cache_clear()
    yield configure
    get_vectordb_config.cache_clear()


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

    assert stats[collection]["executed_tasks"] >= 1
    assert await _fragment_count(adapter, collection) == 1
    table = await adapter.get_collection(collection)
    assert await table.count_rows() == 6
    assert len(await adapter.retrieve(collection, [str(i) for i in ids])) == 6
    # Retention 0: the superseded files are deleted, not just unreferenced. A
    # compaction commits a twin of the previous state microseconds before the
    # rewrite, and the cut-off's 1 ms margin may keep that twin for one pass,
    # so the original files are provably gone after the next write's pass.
    await _write_n_points(adapter, collection, 1, start=6)
    await adapter.compact()
    assert _data_files(db_path, collection).isdisjoint(original)


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
async def test_compact_executes_at_most_max_tasks_per_run(tmp_path, compaction_settings):
    """A bloated store drains over several runs instead of stalling one."""
    compaction_settings(retention_seconds=0, target_rows_per_fragment=2, max_tasks_per_run=1)
    adapter, _ = _adapter(tmp_path)
    collection = "BacklogTarget_label"
    await _write_n_points(adapter, collection, 6)

    stats = await adapter.compact()

    assert stats[collection]["planned_tasks"] == 3
    assert stats[collection]["executed_tasks"] == 1
    assert await _fragment_count(adapter, collection) == 5
    table = await adapter.get_collection(collection)
    assert await table.count_rows() == 6


@pytest.mark.asyncio
async def test_compaction_disabled_preserves_the_uncompacted_behaviour(
    tmp_path, compaction_settings
):
    compaction_settings(enabled="false")
    adapter, db_path = _adapter(tmp_path)
    collection = "DisabledTarget_label"
    await _write_n_points(adapter, collection, 4)

    assert await adapter.compact() == {"skipped": "disabled"}
    assert len(_data_files(db_path, collection)) == 4


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
async def test_cleanup_waits_until_a_versions_successor_has_aged(tmp_path, compaction_settings):
    """The reader-safety rule: a version goes only once its SUCCESSOR is older than the window.

    Lance's own cut-off ages a version by its commit time, which deletes an idle
    table's old-but-latest version from under a reader that opened it a moment
    ago. Here the table idles past the window, a reader opens it, compaction
    supersedes its version, and the files must survive until the next pass.
    """
    import asyncio

    import lancedb

    compaction_settings(retention_seconds=1)
    adapter, db_path = _adapter(tmp_path)
    collection = "IdleTarget_label"
    await _write_n_points(adapter, collection, 3)
    await asyncio.sleep(1.5)  # idle: the latest version is now older than the window
    reader = await (await lancedb.connect_async(db_path)).open_table(collection)
    await reader.checkout(await reader.version())
    expected = (await reader.to_arrow()).to_pylist()
    before = _data_files(db_path, collection)

    first = await adapter.compact()

    assert first[collection]["executed_tasks"] == 1
    assert before <= _data_files(db_path, collection), "reader's files were deleted"
    assert (await reader.to_arrow()).to_pylist() == expected

    await asyncio.sleep(1.5)  # the superseding version has now aged past the window
    second = await adapter.compact()

    assert second[collection]["old_versions_removed"] >= 1
    assert before.isdisjoint(_referenced_files(db_path, collection))
    # The twin version a compaction commits right before its rewrite can sit
    # within the cut-off's 1 ms margin; the pass after the next write removes it.
    await _write_n_points(adapter, collection, 1, start=3)
    await asyncio.sleep(1.5)
    await adapter.compact()
    assert _data_files(db_path, collection).isdisjoint(before)
    assert await (await adapter.get_collection(collection)).count_rows() == 4


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

    second = await adapter.compact()
    assert second[starved[0]]["executed_tasks"] == 2  # the pass started with the other table
    for name in tables:
        assert await _fragment_count(adapter, name) == 3


@pytest.mark.asyncio
async def test_compact_is_fail_open_per_collection(tmp_path, compaction_settings, monkeypatch):
    compaction_settings(retention_seconds=0)
    adapter, _ = _adapter(tmp_path)
    await _write_n_points(adapter, "Healthy_label", 2)
    await _write_n_points(adapter, "Broken_label", 2)
    original = adapter._compact_collection

    async def flaky(name, options):
        if name == "Broken_label":
            raise RuntimeError("boom")
        return await original(name, options)

    monkeypatch.setattr(adapter, "_compact_collection", flaky)

    stats = await adapter.compact()

    assert "error" in stats["Broken_label"]
    assert stats["Healthy_label"]["executed_tasks"] == 1


@pytest.mark.asyncio
async def test_compaction_options_reach_the_subprocess_worker():
    from cognee.infrastructure.databases.vector.lancedb.subprocess.proxy import RemoteLanceDBTable
    from cognee_db_workers.lancedb_protocol import OP_TABLE_OPTIMIZE

    session = Mock()
    session.call_async = AsyncMock(return_value=Mock(result={"executed_tasks": 1}))
    table = RemoteLanceDBTable(session, 17, "WorkerTarget_label")
    options = {"target_rows_per_fragment": 20_000, "retention_seconds": 300, "max_tasks": 4}

    assert await table.optimize(**options) == {"executed_tasks": 1}

    request = session.call_async.await_args.args[0]
    assert request.op == OP_TABLE_OPTIMIZE
    assert request.handle_id == 17
    assert request.kwargs == options

    # The plain form stays lancedb's own optimize (the id re-key migration uses it).
    await table.optimize()
    assert session.call_async.await_args.args[0].kwargs == {}


@pytest.mark.asyncio
async def test_worker_runs_bounded_compaction_on_the_real_table(tmp_path):
    import lancedb

    from cognee_db_workers.harness import HandleRegistry, Request
    from cognee_db_workers.lancedb_protocol import OP_TABLE_OPTIMIZE
    from cognee_db_workers.lancedb_worker import _op_table_optimize

    connection = await lancedb.connect_async(str(tmp_path / "db"))
    rows = [{"id": str(i), "vector": [0.1, 0.2, 0.3], "payload": {"slot": i}} for i in range(5)]
    table = await connection.create_table("WorkerTable", data=rows[:1])
    for row in rows[1:]:
        await table.add([row])
    assert (await table.stats())["fragment_stats"]["num_fragments"] == 5
    registry = HandleRegistry()
    handle_id = registry.register(table)

    stats = await _op_table_optimize(
        registry,
        Request(
            op=OP_TABLE_OPTIMIZE,
            handle_id=handle_id,
            kwargs={"target_rows_per_fragment": 20_000, "retention_seconds": 0, "max_tasks": 0},
        ),
    )

    assert stats["executed_tasks"] == 1
    assert (await table.stats())["fragment_stats"]["num_fragments"] == 1
    assert await table.count_rows() == 5
    # Legacy form: no kwargs -> lancedb's own optimize, no stats.
    assert (
        await _op_table_optimize(registry, Request(op=OP_TABLE_OPTIMIZE, handle_id=handle_id))
        is None
    )


@pytest.mark.asyncio
async def test_pylance_reads_the_tables_lancedb_writes(tmp_path, compaction_settings):
    """Guards the lancedb/pylance pin pair in pyproject.

    lancedb bundles its own Lance core; pylance must be built on the same one or
    ``to_lance()`` (which the compaction relies on) cannot decode the files
    lancedb wrote. A failure here means the two pins drifted apart.
    """
    compaction_settings()
    adapter, _ = _adapter(tmp_path)
    collection = "PinPair_label"
    await _write_n_points(adapter, collection, 3)
    table = await adapter.get_collection(collection)

    dataset = await table.to_lance()

    assert dataset.to_table().num_rows == 3


@pytest.mark.asyncio
async def test_compact_vector_store_never_fails_the_pipeline(monkeypatch):
    module = importlib.import_module("cognee.infrastructure.databases.vector.compact_vector_store")

    class Exploding:
        async def compact(self):
            raise RuntimeError("disk on fire")

    class Quiet:
        async def compact(self):
            return {"Entity_name": {"planned_tasks": 1, "executed_tasks": 1}}

    monkeypatch.setattr(module, "get_vector_engine", lambda: Exploding())
    assert await module.compact_vector_store() is None

    monkeypatch.setattr(module, "get_vector_engine", lambda: object())  # adapter without compact
    assert await module.compact_vector_store() is None

    monkeypatch.setattr(module, "get_vector_engine", lambda: Quiet())
    assert await module.compact_vector_store() == {
        "Entity_name": {"planned_tasks": 1, "executed_tasks": 1}
    }


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
        assert len(_data_files(db_path, collection)) == 1
        assert len(await adapter.retrieve(collection, [str(i) for i in ids])) == 5
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_compact_vector_store_lets_a_running_pass_finish_before_cancelling(monkeypatch):
    """A cancel must not leave the pass running against the rollback that follows it."""
    import asyncio

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

    monkeypatch.setattr(module, "get_vector_engine", lambda: Slow())
    task = asyncio.ensure_future(module.compact_vector_store())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished, "cancellation propagated while the compaction pass was still running"


def test_every_pipeline_run_ends_with_vector_maintenance():
    """Maintenance is not opt-in: run_tasks calls it unconditionally before completion."""
    import ast

    from cognee.modules.pipelines.operations import run_tasks as run_tasks_module

    source = Path(run_tasks_module.__file__).read_text()
    assert "vector_maintenance" not in source
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compact_vector_store"
    ]
    assert len(calls) == 1
    # The call is a plain statement in the success path, not guarded by a condition.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If):
            assert not any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "compact_vector_store"
                for sub in ast.walk(node)
            ), "compact_vector_store must not be gated by a condition"


@pytest.mark.asyncio
async def test_compaction_settings_keep_the_vector_factory_usable(tmp_path, monkeypatch):
    """The settings are read by the adapter, not passed through the factory."""
    from cognee.infrastructure.databases.vector.config import VectorConfig

    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    config = VectorConfig(
        vector_db_provider="lancedb",
        vector_db_url=str(tmp_path / "db"),
        vector_db_name="factory_compaction",
        vector_db_subprocess_enabled=False,
        vector_db_compaction_target_rows_per_fragment=2,
    )

    adapter = factory.create_vector_engine(**config.to_dict())
    try:
        await _write_n_points(adapter, "FactoryTarget_label", 1)
        table = await adapter.get_collection("FactoryTarget_label")
        assert await table.count_rows() == 1
    finally:
        await adapter.close()
