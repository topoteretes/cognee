import asyncio
import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

lock_module = importlib.import_module("cognee.infrastructure.locks.dataset_pipeline_lock")


@pytest.mark.asyncio
async def test_local_lock_serializes_callers_and_cleans_registry():
    dataset_id = uuid4()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first():
        async with lock_module._local_dataset_lock(dataset_id):
            first_entered.set()
            await release_first.wait()

    async def second():
        await first_entered.wait()
        async with lock_module._local_dataset_lock(dataset_id):
            second_entered.set()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_entered.wait()
    await asyncio.sleep(0)

    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert lock_module._local_locks == {}


@pytest.mark.asyncio
async def test_cancelled_local_waiter_does_not_leak_registry_entry():
    dataset_id = uuid4()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder():
        async with lock_module._local_dataset_lock(dataset_id):
            holder_entered.set()
            await release_holder.wait()

    async def waiter():
        async with lock_module._local_dataset_lock(dataset_id):
            pass

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()
    waiter_task = asyncio.create_task(waiter())

    while next(iter(lock_module._local_locks.values())).users != 2:
        await asyncio.sleep(0)
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    release_holder.set()
    await holder_task

    assert lock_module._local_locks == {}


@pytest.mark.asyncio
async def test_sqlite_advisory_lock_serializes_independent_callers(tmp_path):
    engine = SimpleNamespace(url=SimpleNamespace(database=str(tmp_path / "cognee.db")))
    dataset_id = uuid4()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first():
        async with lock_module._sqlite_advisory_lock(engine, dataset_id):
            first_entered.set()
            await release_first.wait()

    async def second():
        await first_entered.wait()
        async with lock_module._sqlite_advisory_lock(engine, dataset_id):
            second_entered.set()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_entered.wait()
    await asyncio.sleep(0.05)

    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert lock_module._sqlite_lock_path(engine, dataset_id).name.startswith(".cognee-pipeline-")


@pytest.mark.asyncio
async def test_cancelled_sqlite_waiter_does_not_orphan_os_lock(tmp_path):
    engine = SimpleNamespace(url=SimpleNamespace(database=str(tmp_path / "cognee.db")))
    dataset_id = uuid4()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder():
        async with lock_module._sqlite_advisory_lock(engine, dataset_id):
            holder_entered.set()
            await release_holder.wait()

    async def waiter():
        async with lock_module._sqlite_advisory_lock(engine, dataset_id):
            pass

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)

    waiter_task.cancel()
    release_holder.set()
    await holder_task
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    async with asyncio.timeout(1):
        async with lock_module._sqlite_advisory_lock(engine, dataset_id):
            pass


class _FakeConnection:
    def __init__(self):
        self.statements = []
        self.commits = 0

    async def execute(self, statement, parameters):
        self.statements.append((str(statement), parameters))

    async def commit(self):
        self.commits += 1


class _FakeConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


@pytest.mark.asyncio
async def test_postgres_advisory_lock_is_released_after_pipeline_error(monkeypatch):
    connection = _FakeConnection()
    engine = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        connect=lambda: _FakeConnectionContext(connection),
    )
    monkeypatch.setattr(
        lock_module,
        "get_relational_engine",
        lambda: SimpleNamespace(engine=engine),
    )
    dataset_id = UUID("12345678-1234-5678-1234-567812345678")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        async with lock_module._cross_worker_dataset_lock(dataset_id):
            raise RuntimeError("pipeline failed")

    expected_key = lock_module._postgres_lock_key(dataset_id)
    assert connection.statements == [
        ("SELECT pg_advisory_lock(:key)", {"key": expected_key}),
        ("SELECT pg_advisory_unlock(:key)", {"key": expected_key}),
    ]
    assert connection.commits == 2


def test_sqlite_lock_files_are_bounded_to_fixed_stripes(tmp_path):
    engine = SimpleNamespace(url=SimpleNamespace(database=str(tmp_path / "cognee.db")))
    paths = {
        lock_module._sqlite_lock_path(engine, UUID(int=dataset_number))
        for dataset_number in range(1_000)
    }

    assert len(paths) <= lock_module._SQLITE_LOCK_STRIPES
