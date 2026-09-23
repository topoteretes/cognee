"""The session cache on the Turso rewrite engine (``CACHE_BACKEND=turso``).

The CRUD suite already runs on Turso through its parametrized fixture; these tests
cover what is specific to the backend: proof that the rewrite engine executes the
SQL, the ``TURSO_*`` settings reaching the cache engine, and a round trip in
``mvcc`` mode where table creation needs an exclusive transaction while writes
run as ``BEGIN CONCURRENT``.
"""

import asyncio

import pytest
from sqlalchemy import text

pytest.importorskip("turso", reason="pyturso not installed")

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.turso import get_turso_config, turso_url


@pytest.fixture(params=["wal", "mvcc"])
def journal_mode(request, monkeypatch):
    monkeypatch.setenv("TURSO_JOURNAL_MODE", request.param)
    get_turso_config.cache_clear()
    yield request.param
    get_turso_config.cache_clear()


def _run(coro):
    return asyncio.run(coro)


def test_cache_runs_on_the_turso_engine_in_the_configured_mode(tmp_path, journal_mode):
    adapter = SqlCacheAdapter(turso_url(f"{tmp_path}/cache.db"))
    assert adapter.engine.dialect.driver == "cognee_turso"

    async def probe():
        await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
        entries = await adapter.get_all_qa_entries("u1", "s1")
        async with adapter.engine.connect() as connection:
            version = (await connection.execute(text("SELECT turso_version()"))).scalar()
            mode = (await connection.execute(text("PRAGMA journal_mode"))).scalar()
        await adapter.close()
        return entries, version, mode

    entries, version, mode = _run(probe())
    assert [entry.qa_id for entry in entries] == ["id1"]
    assert version  # only the Turso rewrite defines turso_version()
    assert mode == journal_mode


def test_cache_persists_across_reopen(tmp_path, journal_mode):
    url = turso_url(f"{tmp_path}/cache.db")

    async def write():
        adapter = SqlCacheAdapter(url)
        await adapter.create_qa_entry("u1", "s1", "Q1", "C", "A", qa_id="kept-1")
        await adapter.create_qa_entry("u1", "s1", "Q2", "C", "A", qa_id="kept-2")
        await adapter.close()

    async def read():
        adapter = SqlCacheAdapter(url)
        entries = await adapter.get_all_qa_entries("u1", "s1")
        await adapter.close()
        return entries

    _run(write())
    assert sorted(entry.qa_id for entry in _run(read())) == ["kept-1", "kept-2"]


@pytest.mark.parametrize("engine", ["sqlite", "turso"])
def test_adapters_write_the_same_cache_concurrently(tmp_path, journal_mode, engine):
    """Independent engines (connections) on one cache.db, as several processes would be.

    Runs on aiosqlite too so a failure that is SQLite's own behaviour (not Turso's)
    is visible as such. Schema creation happens once up front: racing ``CREATE
    TABLE`` from several deferred transactions is a snapshot conflict on any
    SQLite-family engine and is not what this test is about.
    """
    if engine == "turso":
        url = turso_url(f"{tmp_path}/cache.db")
    else:
        url = f"sqlite+aiosqlite:///{tmp_path}/cache.db"

    async def run():
        first = SqlCacheAdapter(url)
        await first.create_qa_entry("u0", "setup", "Q", "C", "A", qa_id="setup")
        adapters = [first, *(SqlCacheAdapter(url) for _ in range(3))]

        async def write(index, adapter):
            for n in range(10):
                await adapter.create_qa_entry(
                    "u1", f"s{index}", f"Q{n}", "C", "A", qa_id=f"w{index}-{n}"
                )

        await asyncio.gather(*(write(i, a) for i, a in enumerate(adapters)))
        counts = [len(await adapters[0].get_all_qa_entries("u1", f"s{i}")) for i in range(4)]
        for adapter in adapters:
            await adapter.close()
        return counts

    assert _run(run()) == [10, 10, 10, 10]
