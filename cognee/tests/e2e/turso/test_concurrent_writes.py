"""Concurrent writes against one Turso database from independent connections.

Runs in both journal modes the ``turso`` backends support, on the real engine:

* ``wal`` (default): writers serialize on the engine's write lock; a second writer
  waits up to ``busy_timeout`` and then fails with ``database is locked``.
* ``mvcc``: writes are ``BEGIN CONCURRENT`` transactions that commit in parallel;
  two transactions writing the same row fail eagerly with ``Write-write conflict``
  and cognee's write paths re-run the whole transaction.

Every test uses separate connections (separate SQLAlchemy engines with NullPool,
separate adapter instances, or separate raw ``turso`` connections), verifies the
committed data, and checks it survives a reopen. Requires pyturso.
"""

import asyncio
import contextlib
import threading

import pytest
from sqlalchemy import text

pytest.importorskip("turso", reason="pyturso not installed")

import turso

from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter as GraphAdapter
from cognee.infrastructure.databases.turso import (
    get_turso_config,
    is_retryable_conflict,
    retry_on_conflict,
)
from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import TursoVectorAdapter
from cognee.infrastructure.engine import DataPoint

WRITERS = 6
ROWS_PER_WRITER = 40


@pytest.fixture(params=["wal", "mvcc"])
def journal_mode(request, monkeypatch):
    """Point TursoConfig at the requested journal mode for the duration of a test."""
    monkeypatch.setenv("TURSO_JOURNAL_MODE", request.param)
    get_turso_config.cache_clear()
    yield request.param
    get_turso_config.cache_clear()


def _connect(path: str, mode: str):
    connection = turso.connect(path, isolation_level=None if mode == "mvcc" else "DEFERRED")
    connection.execute(f"PRAGMA journal_mode={mode}").fetchall()
    return connection


def _begin(mode: str) -> str:
    return "BEGIN CONCURRENT" if mode == "mvcc" else "BEGIN"


# --------------------------------------------------------------------------- #
# Graph adapter: N adapters (N engines, N connections) on one file
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_graph_adapters_write_concurrently_and_persist(tmp_path, journal_mode):
    path = str(tmp_path / "graph.db")
    adapters = [GraphAdapter(database_path=path) for _ in range(WRITERS)]
    await adapters[0].initialize()

    async with adapters[0].engine.connect() as connection:
        assert (await connection.execute(text("PRAGMA journal_mode"))).scalar() == journal_mode

    async def write(index: int, adapter: GraphAdapter) -> None:
        for batch in range(4):
            nodes = [
                (f"w{index}_n{batch * 10 + i}", {"name": f"node {i}", "type": "T"})
                for i in range(ROWS_PER_WRITER // 4)
            ]
            await adapter.add_nodes(nodes)
            await adapter.add_edges(
                [(nodes[i][0], nodes[i + 1][0], "NEXT", {}) for i in range(len(nodes) - 1)]
            )

    try:
        await asyncio.gather(*(write(i, adapter) for i, adapter in enumerate(adapters)))
        nodes, edges = await adapters[0].get_graph_data()
        assert len(nodes) == WRITERS * ROWS_PER_WRITER
        assert len(edges) == WRITERS * 4 * (ROWS_PER_WRITER // 4 - 1)
    finally:
        for adapter in adapters:
            await adapter.close()

    # Restart semantics: a fresh adapter on the same file sees every committed row.
    reopened = GraphAdapter(database_path=path)
    try:
        nodes, _ = await reopened.get_graph_data()
        assert len(nodes) == WRITERS * ROWS_PER_WRITER
    finally:
        await reopened.close()


# --------------------------------------------------------------------------- #
# Vector adapter: two adapter instances (two connections) on one file
# --------------------------------------------------------------------------- #
class _Embedding:
    def get_vector_size(self):
        return 3

    async def embed_text(self, texts):
        return [[1.0, float(len(text) % 7), 0.5] for text in texts]


class _Doc(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
async def test_vector_adapters_write_concurrently_and_persist(tmp_path, journal_mode):
    path = str(tmp_path / "vectors.db")
    adapters = [
        TursoVectorAdapter(url=path, api_key=None, embedding_engine=_Embedding())
        for _ in range(WRITERS)
    ]
    await adapters[0].create_collection("Doc_text")

    async def write(index: int, adapter: TursoVectorAdapter) -> None:
        for batch in range(4):
            docs = [
                _Doc(text=f"writer {index} batch {batch} doc {i}", belongs_to_set=[f"w{index}"])
                for i in range(ROWS_PER_WRITER // 4)
            ]
            await adapter.create_data_points("Doc_text", docs)

    try:
        await asyncio.gather(*(write(i, adapter) for i, adapter in enumerate(adapters)))
        rows = await adapters[0]._execute('SELECT count(*) FROM "Doc_text"', fetch=True)
        assert rows[0][0] == WRITERS * ROWS_PER_WRITER
        hits = await adapters[0].search("Doc_text", query_text="writer 1", limit=5)
        assert len(hits) == 5
    finally:
        for adapter in adapters:
            await adapter.close()

    reopened = TursoVectorAdapter(url=path, api_key=None, embedding_engine=_Embedding())
    try:
        rows = await reopened._execute('SELECT count(*) FROM "Doc_text"', fetch=True)
        assert rows[0][0] == WRITERS * ROWS_PER_WRITER
    finally:
        await reopened.close()


# --------------------------------------------------------------------------- #
# Raw connections: threads, conflicts and the documented failure modes
# --------------------------------------------------------------------------- #
def test_threads_with_independent_connections(tmp_path, journal_mode):
    path = str(tmp_path / "threads.db")
    setup = _connect(path, journal_mode)
    setup.execute("CREATE TABLE hits (writer INTEGER, n INTEGER)")
    setup.commit()
    setup.close()

    errors: list[str] = []

    def worker(index: int) -> None:
        connection = _connect(path, journal_mode)
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            for n in range(ROWS_PER_WRITER):
                connection.execute(_begin(journal_mode))
                connection.execute("INSERT INTO hits VALUES (?, ?)", (index, n))
                connection.execute("COMMIT")
        # Collected, not raised: the assertion below reports every worker's error.
        except turso.Error as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            connection.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(WRITERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    check = _connect(path, journal_mode)
    assert check.execute("SELECT count(*) FROM hits").fetchone()[0] == WRITERS * ROWS_PER_WRITER
    check.close()


def test_same_row_write_conflict_is_detected_and_retryable(tmp_path, journal_mode):
    """MVCC: eager ``Write-write conflict`` on the second writer, cleared by a retry.

    WAL: the second writer blocks for ``busy_timeout`` and fails with
    ``database is locked``; it succeeds once the first transaction commits.
    """
    path = str(tmp_path / "conflict.db")
    setup = _connect(path, journal_mode)
    setup.execute("CREATE TABLE acct (id TEXT PRIMARY KEY, bal INTEGER)")
    setup.execute("INSERT INTO acct VALUES ('a', 0)")
    setup.commit()
    setup.close()

    first = _connect(path, journal_mode)
    second = _connect(path, journal_mode)
    second.execute("PRAGMA busy_timeout=300")

    first.execute(_begin(journal_mode))
    first.execute("UPDATE acct SET bal = bal + 1 WHERE id = 'a'")

    second.execute(_begin(journal_mode))
    with pytest.raises(Exception) as error_info:
        second.execute("UPDATE acct SET bal = bal + 10 WHERE id = 'a'")
        second.execute("COMMIT")
    error = error_info.value
    assert is_retryable_conflict(error), error
    if journal_mode == "mvcc":
        assert "write-write conflict" in str(error).lower()
    else:
        assert "database is locked" in str(error).lower()
    # MVCC already aborted the transaction engine-side; WAL still holds one.
    with contextlib.suppress(Exception):
        second.execute("ROLLBACK")

    first.execute("COMMIT")

    async def retry_second_writer():
        attempts = {"n": 0}

        async def transaction():
            attempts["n"] += 1
            second.execute(_begin(journal_mode))
            try:
                second.execute("UPDATE acct SET bal = bal + 10 WHERE id = 'a'")
                second.execute("COMMIT")
            except Exception:
                with contextlib.suppress(Exception):
                    second.execute("ROLLBACK")
                raise

        await retry_on_conflict(transaction, attempts=5, base_delay=0)
        return attempts["n"]

    assert asyncio.run(retry_second_writer()) == 1
    assert first.execute("SELECT bal FROM acct WHERE id = 'a'").fetchone()[0] == 11
    first.close()
    second.close()
