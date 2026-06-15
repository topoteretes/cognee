"""Regression tests for the formalized KV methods on CacheDBInterface.

Covers:
- FsCacheAdapter get_value/set_value/delete_value round-trip (this was the live
  bug where graph snapshots silently no-op'd on fs: consumers fell back to a
  non-existent ``adapter._cache`` attribute — the real attribute is ``cache``).
- RedisAdapter KV methods producing byte-identical keys/values to the legacy
  ``adapter.async_redis`` duck-typed path.
- SessionManager graph-context consumers using the interface methods with
  verbatim legacy key strings.
"""

import tempfile

import pytest
from unittest.mock import MagicMock, patch

from cognee.infrastructure.session.session_manager import SessionManager


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fs_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ):
            from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                FSCacheAdapter,
            )

            inst = FSCacheAdapter()
            yield inst
            inst.cache.close()


class _InMemoryRedisKV:
    """Minimal in-memory Redis string-KV emulation (stores bytes, like real Redis)."""

    def __init__(self):
        self.kv: dict[str, bytes] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def set(self, key: str, value):
        self.kv[key] = value.encode("utf-8") if isinstance(value, str) else value

    async def get(self, key: str):
        return self.kv.get(key)

    async def delete(self, key: str):
        self.expire_calls = [call for call in self.expire_calls if call[0] != key]
        return 1 if self.kv.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        self.expire_calls.append((key, ttl))


@pytest.fixture
def redis_store():
    return _InMemoryRedisKV()


@pytest.fixture
def redis_adapter(redis_store):
    patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_mod}.aioredis.Redis", return_value=redis_store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        yield RedisAdapter(host="localhost", port=6379)


# --------------------------------------------------------------------------- #
# FsCacheAdapter KV round-trip (regression for the silent-no-op bug)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fs_kv_set_get_round_trip(fs_adapter):
    """set_value/get_value round-trips a raw string on the fs backend."""
    await fs_adapter.set_value("graph_knowledge:u1:s1", "graph snapshot")
    assert await fs_adapter.get_value("graph_knowledge:u1:s1") == "graph snapshot"


@pytest.mark.asyncio
async def test_fs_kv_get_returns_none_for_missing_key(fs_adapter):
    assert await fs_adapter.get_value("graph_knowledge:u1:missing") is None


@pytest.mark.asyncio
async def test_fs_kv_delete_value(fs_adapter):
    await fs_adapter.set_value("graph_knowledge:u1:s1", "snapshot")
    await fs_adapter.delete_value("graph_knowledge:u1:s1")
    assert await fs_adapter.get_value("graph_knowledge:u1:s1") is None
    # Deleting a missing key is a no-op, not an error.
    await fs_adapter.delete_value("graph_knowledge:u1:s1")


@pytest.mark.asyncio
async def test_fs_kv_set_with_ttl_overwrites_and_round_trips(fs_adapter):
    """ttl is passed through to diskcache expire; value stays readable before expiry."""
    await fs_adapter.set_value("k1", "v1", ttl=3600)
    assert await fs_adapter.get_value("k1") == "v1"
    await fs_adapter.set_value("k1", "v2", ttl=3600)
    assert await fs_adapter.get_value("k1") == "v2"


@pytest.mark.asyncio
async def test_fs_graph_context_round_trip_through_session_manager(fs_adapter):
    """SessionManager graph snapshots now actually persist on fs (previously no-op'd)."""
    session_manager = SessionManager(cache_engine=fs_adapter)

    await session_manager.set_graph_context(user_id="u1", session_id="s1", context="fs snapshot")

    assert await session_manager.get_graph_context(user_id="u1", session_id="s1") == "fs snapshot"
    # Stored under the verbatim legacy key in the diskcache (attribute is `cache`).
    assert fs_adapter.cache.get("graph_knowledge:u1:s1") == "fs snapshot"

    await session_manager.delete_session(user_id="u1", session_id="s1")
    assert await session_manager.get_graph_context(user_id="u1", session_id="s1") == ""


# --------------------------------------------------------------------------- #
# RedisAdapter KV byte-parity with the legacy async_redis duck-typed path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_redis_set_value_byte_identical_to_legacy_async_redis_set(redis_adapter, redis_store):
    """set_value stores exactly the key/value bytes the legacy async_redis path stored."""
    key = "graph_knowledge:u1:s1"
    value = "graph snapshot text"

    # Legacy duck-typed path (what session_manager used to call directly).
    await redis_adapter.async_redis.set(key, value)
    legacy_snapshot = dict(redis_store.kv)
    redis_store.kv.clear()

    # New interface method.
    await redis_adapter.set_value(key, value)

    assert redis_store.kv == legacy_snapshot
    assert list(redis_store.kv.keys()) == [key]
    assert redis_store.kv[key] == value.encode("utf-8")


@pytest.mark.asyncio
async def test_redis_get_value_decodes_bytes_like_legacy_consumers(redis_adapter, redis_store):
    """get_value returns str even though Redis hands back bytes (legacy decode parity)."""
    await redis_adapter.async_redis.set("k1", "legacy written")
    assert isinstance(redis_store.kv["k1"], bytes)

    result = await redis_adapter.get_value("k1")
    assert result == "legacy written"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_redis_get_value_returns_none_for_missing_key(redis_adapter):
    assert await redis_adapter.get_value("missing") is None


@pytest.mark.asyncio
async def test_redis_set_value_with_ttl_expires_like_legacy_path(redis_adapter, redis_store):
    """set_value(ttl=N) issues the same set + expire(key, N) sequence the legacy code did."""
    await redis_adapter.set_value("k1", "v1", ttl=604800)
    assert redis_store.kv["k1"] == b"v1"
    assert redis_store.expire_calls == [("k1", 604800)]


@pytest.mark.asyncio
async def test_redis_set_value_without_ttl_does_not_expire(redis_adapter, redis_store):
    await redis_adapter.set_value("k1", "v1")
    assert redis_store.expire_calls == []


@pytest.mark.asyncio
async def test_redis_delete_value_removes_legacy_written_key(redis_adapter, redis_store):
    """delete_value targets the byte-identical key written by the legacy path."""
    await redis_adapter.async_redis.set("graph_knowledge:u1:s1", "snapshot")
    await redis_adapter.delete_value("graph_knowledge:u1:s1")
    assert "graph_knowledge:u1:s1" not in redis_store.kv


@pytest.mark.asyncio
async def test_redis_graph_context_key_string_unchanged(redis_adapter, redis_store):
    """SessionManager writes the verbatim legacy key graph_knowledge:{user}:{session}."""
    session_manager = SessionManager(cache_engine=redis_adapter)

    await session_manager.set_graph_context(user_id="u1", session_id="s1", context="ctx")

    assert list(redis_store.kv.keys()) == ["graph_knowledge:u1:s1"]
    assert redis_store.kv["graph_knowledge:u1:s1"] == b"ctx"
    # TTL applied from adapter.session_ttl_seconds, exactly like the legacy path.
    assert redis_store.expire_calls == [("graph_knowledge:u1:s1", 604800)]

    assert await session_manager.get_graph_context(user_id="u1", session_id="s1") == "ctx"
