"""Unit tests for TursoAdapter's write-conflict handling.

Turso/libSQL is single-writer, so the adapter serializes its own writes with an
asyncio lock and retries transient busy errors with bounded backoff. These tests
cover the busy-error classification and the retry policy deterministically, without
a real driver or database (TursoAdapter is built via __new__ so __init__, which
imports the driver and opens an engine, is not run).
"""

import asyncio

import pytest
from sqlalchemy.exc import OperationalError

from cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter import (
    TursoAdapter,
    _is_busy_error,
)


def _make_adapter(max_retries: int = 3) -> TursoAdapter:
    """A TursoAdapter with just the write-lock machinery, no engine/driver."""
    adapter = object.__new__(TursoAdapter)
    adapter._write_lock = asyncio.Lock()
    adapter._write_max_retries = max_retries
    adapter._write_retry_base_delay = 0.0  # no real sleeping in tests
    return adapter


def _busy_error() -> OperationalError:
    return OperationalError("UPDATE ...", {}, Exception("database is locked"))


def test_is_busy_error_classifies_contention():
    assert _is_busy_error(Exception("database is locked"))
    assert _is_busy_error(Exception("SQLITE_BUSY"))
    assert _is_busy_error(Exception("write-write conflict"))


def test_is_busy_error_ignores_other_errors():
    assert not _is_busy_error(Exception("no such table: data"))
    assert not _is_busy_error(Exception("UNIQUE constraint failed"))


def test_run_write_retries_busy_then_succeeds():
    adapter = _make_adapter(max_retries=5)
    calls = {"n": 0}

    async def flaky(_self):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _busy_error()
        return "ok"

    result = asyncio.run(adapter._run_write(flaky))
    assert result == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_run_write_gives_up_after_max_retries():
    adapter = _make_adapter(max_retries=2)
    calls = {"n": 0}

    async def always_busy(_self):
        calls["n"] += 1
        raise _busy_error()

    with pytest.raises(OperationalError):
        asyncio.run(adapter._run_write(always_busy))
    assert calls["n"] == 3  # initial try + 2 retries


def test_run_write_does_not_retry_non_busy_errors():
    adapter = _make_adapter(max_retries=5)
    calls = {"n": 0}

    async def hard_failure(_self):
        calls["n"] += 1
        raise OperationalError("INSERT ...", {}, Exception("no such table: data"))

    with pytest.raises(OperationalError):
        asyncio.run(adapter._run_write(hard_failure))
    assert calls["n"] == 1  # non-retryable, tried exactly once


def test_run_write_serializes_concurrent_writers():
    """Under the lock, overlapping writers never run their critical sections at once."""
    adapter = _make_adapter()
    active = {"now": 0, "max": 0}

    async def writer(_self):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0)  # yield, so an unlocked version would interleave
        active["now"] -= 1

    async def scenario():
        await asyncio.gather(*[adapter._run_write(writer) for _ in range(10)])

    asyncio.run(scenario())
    assert active["max"] == 1  # never two critical sections in flight at once
