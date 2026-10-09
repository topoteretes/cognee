"""Shared contract tests for concrete CacheDBInterface adapters."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio


class _InMemoryRedisList:
    """Minimal async Redis list emulation for adapter contract tests."""

    def __init__(self):
        self.data: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}

    async def rpush(self, key: str, *values: str):
        self.data.setdefault(key, []).extend(values)

    async def lrange(self, key: str, start: int, end: int):
        values = self.data.get(key, [])
        start_index = start if start >= 0 else max(len(values) + start, 0)
        end_index = len(values) if end == -1 else end + 1
        return values[start_index:end_index]

    async def lindex(self, key: str, index: int):
        values = self.data.get(key, [])
        return values[index] if -len(values) <= index < len(values) else None

    async def llen(self, key: str):
        return len(self.data.get(key, []))

    async def lset(self, key: str, index: int, value: str):
        self.data[key][index] = value

    async def delete(self, key: str):
        self.ttls.pop(key, None)
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl

    async def ttl(self, key: str):
        if key not in self.data:
            return -2
        return self.ttls.get(key, -1)


@pytest_asyncio.fixture(params=["fs", "sql", "redis"])
async def cache_adapter(request, tmp_path):
    """Yield each cache adapter behind the same CacheDBInterface surface."""
    if request.param == "fs":
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": str(tmp_path / "fs")},
        ):
            from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                FSCacheAdapter,
            )

            adapter = FSCacheAdapter()
            yield adapter
            adapter.cache.close()
        return

    if request.param == "sql":
        from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter

        adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{Path(tmp_path) / 'cache.db'}")
        yield adapter
        await adapter.close()
        return

    redis_store = _InMemoryRedisList()
    patch_module = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_module}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_module}.aioredis.Redis", return_value=redis_store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        yield RedisAdapter(host="localhost", port=6379)


@pytest.mark.asyncio
async def test_qa_collection_methods_return_empty_lists_for_empty_sessions(cache_adapter):
    assert await cache_adapter.get_latest_qa_entries("user", "missing", last_n=0) == []
    assert await cache_adapter.get_latest_qa_entries("user", "missing", last_n=1) == []
    assert await cache_adapter.get_latest_qa_entries("user", "missing", last_n=5) == []
    assert await cache_adapter.get_all_qa_entries("user", "missing") == []
    assert await cache_adapter.get_qa_entries_by_ids("user", "missing", ["qa-1"]) == []
    assert await cache_adapter.get_qa_entries_by_ids("user", "missing", []) == []


@pytest.mark.asyncio
async def test_latest_qa_entries_respect_last_n_edges_and_chronological_order(cache_adapter):
    for index in range(1, 4):
        await cache_adapter.create_qa_entry(
            "user",
            "session",
            f"Question {index}",
            f"Context {index}",
            f"Answer {index}",
            qa_id=f"qa-{index}",
        )

    latest_zero = await cache_adapter.get_latest_qa_entries("user", "session", last_n=0)
    latest_one = await cache_adapter.get_latest_qa_entries("user", "session", last_n=1)
    latest_many = await cache_adapter.get_latest_qa_entries("user", "session", last_n=5)
    all_entries = await cache_adapter.get_all_qa_entries("user", "session")

    assert latest_zero == []
    assert [entry.qa_id for entry in latest_one] == ["qa-3"]
    assert [entry.qa_id for entry in latest_many] == ["qa-1", "qa-2", "qa-3"]
    assert [entry.qa_id for entry in all_entries] == ["qa-1", "qa-2", "qa-3"]


@pytest.mark.asyncio
async def test_qa_entries_by_ids_returns_empty_or_chronological_matches(cache_adapter):
    for index in range(1, 4):
        await cache_adapter.create_qa_entry(
            "user",
            "session",
            f"Question {index}",
            f"Context {index}",
            f"Answer {index}",
            qa_id=f"qa-{index}",
        )

    assert await cache_adapter.get_qa_entries_by_ids("user", "session", []) == []

    matches = await cache_adapter.get_qa_entries_by_ids(
        "user", "session", ["qa-3", "missing", "qa-1"]
    )
    assert [entry.qa_id for entry in matches] == ["qa-1", "qa-3"]


@pytest.mark.asyncio
async def test_agent_trace_collections_share_empty_and_last_n_contract(cache_adapter):
    assert await cache_adapter.get_agent_trace_session("user", "missing") == []
    assert await cache_adapter.get_agent_trace_session("user", "missing", last_n=1) == []
    assert await cache_adapter.get_agent_trace_feedback("user", "missing") == []

    for index in range(1, 4):
        await cache_adapter.append_agent_trace_step(
            "user",
            "trace-session",
            trace_id=f"trace-{index}",
            origin_function="contract_test",
            status="success",
            session_feedback=f"feedback-{index}",
        )

    latest_zero = await cache_adapter.get_agent_trace_session(
        "user", "trace-session", last_n=0
    )
    latest_one = await cache_adapter.get_agent_trace_session("user", "trace-session", last_n=1)
    latest_many = await cache_adapter.get_agent_trace_session("user", "trace-session", last_n=5)
    feedback_one = await cache_adapter.get_agent_trace_feedback(
        "user", "trace-session", last_n=1
    )

    assert latest_zero == []
    assert [entry.trace_id for entry in latest_one] == ["trace-3"]
    assert [entry.trace_id for entry in latest_many] == ["trace-1", "trace-2", "trace-3"]
    assert feedback_one == ["feedback-3"]


@pytest.mark.asyncio
async def test_session_context_collection_returns_empty_or_chronological_entries(cache_adapter):
    assert await cache_adapter.get_session_context_entries("user", "missing") == []

    await cache_adapter.create_session_context_entry(
        "user", "context-session", {"id": "ctx-1", "kind": "context", "text": "first"}
    )
    await cache_adapter.create_session_context_entry(
        "user", "context-session", {"id": "ctx-2", "kind": "feedback", "text": "second"}
    )

    entries = await cache_adapter.get_session_context_entries("user", "context-session")
    assert [entry["id"] for entry in entries] == ["ctx-1", "ctx-2"]
