"""``PGVectorAdapter._get_write_lock`` must hand every caller the same lock per collection.

The per-collection write lock serializes read-modify-write upserts across
threads and event loops, so a check-then-insert race that created two locks
for one collection would silently defeat it.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticLock

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter


def _bare_adapter() -> PGVectorAdapter:
    adapter = object.__new__(PGVectorAdapter)
    adapter._write_locks = {}
    adapter._write_locks_guard = threading.Lock()
    return adapter


def test_same_collection_gets_one_lock_across_threads():
    adapter = _bare_adapter()
    gate = threading.Barrier(16)

    def get():
        gate.wait()  # all threads race the first lookup together
        return adapter._get_write_lock("DocumentChunk_text")

    with ThreadPoolExecutor(max_workers=16) as pool:
        locks = list(pool.map(lambda _: get(), range(16)))

    assert all(lock is locks[0] for lock in locks)
    assert isinstance(locks[0], LoopAgnosticLock)
    assert len(adapter._write_locks) == 1


def test_different_collections_get_different_locks():
    adapter = _bare_adapter()
    assert adapter._get_write_lock("a") is not adapter._get_write_lock("b")
    assert adapter._get_write_lock("a") is adapter._get_write_lock("a")
