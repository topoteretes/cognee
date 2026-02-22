"""Unit tests for RedisAdapter CRUD operations."""

import pytest
from unittest.mock import MagicMock, patch

from cognee.infrastructure.databases.exceptions import SessionQAEntryValidationError


class _InMemoryRedisList:
    """Minimal in-memory Redis list emulation."""

    def __init__(self):
        self.data: dict[str, list[str]] = {}

    async def rpush(self, key: str, *vals: str):
        self.data.setdefault(key, []).extend(vals)

    async def lrange(self, key: str, start: int, end: int):
        lst = self.data.get(key, [])
        s = start if start >= 0 else len(lst) + start
        e = (end + 1) if end >= 0 else len(lst) + end + 1
        return lst[s:e]

    async def lindex(self, key: str, idx: int):
        lst = self.data.get(key, [])
        return lst[idx] if -len(lst) <= idx < len(lst) else None

    async def lset(self, key: str, idx: int, val: str):
        self.data[key][idx] = val

    async def delete(self, key: str):
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        pass

    async def ttl(self, key: str):
        """Return remaining TTL; -1 = no expiry, -2 = key missing. Mock returns -1."""
        return -1 if key in self.data else -2

    async def flushdb(self):
        self.data.clear()


@pytest.fixture
def adapter():
    store = _InMemoryRedisList()
    patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_mod}.aioredis.Redis", return_value=store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        yield RedisAdapter(host="localhost", port=6379)


@pytest.mark.asyncio
async def test_create_and_get(adapter):
    """Create a QA entry and retrieve it."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1 and entries[0]["qa_id"] == "id1"


@pytest.mark.asyncio
async def test_update(adapter):
    """Update a QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    ok = await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=5)
    assert ok and (await adapter.get_all_qa_entries("u1", "s1"))[0]["feedback_score"] == 5


@pytest.mark.asyncio
async def test_update_invalid_raises(adapter):
    """Raise SessionQAEntryValidationError when feedback_score is out of range or gets wrong format."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    with pytest.raises(SessionQAEntryValidationError):
        await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=10)

    with pytest.raises(SessionQAEntryValidationError):
        await adapter.update_qa_entry("u1", "s1", "id1", feedback_text=5)


@pytest.mark.asyncio
async def test_delete_feedback(adapter):
    """delete_feedback sets feedback_text and feedback_score to None."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.update_qa_entry("u1", "s1", "id1", feedback_text="good", feedback_score=5)
    ok = await adapter.delete_feedback("u1", "s1", "id1")
    assert ok
    entries = await adapter.get_all_qa_entries("u1", "s1")
    e = entries[0]
    assert e.get("feedback_text") is None and e.get("feedback_score") is None


@pytest.mark.asyncio
async def test_delete_entry(adapter):
    """Delete a single QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    ok = await adapter.delete_qa_entry("u1", "s1", "id1")
    assert ok and len(await adapter.get_all_qa_entries("u1", "s1")) == 1


@pytest.mark.asyncio
async def test_delete_session(adapter):
    """Delete the entire session and all its entries."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    ok = await adapter.delete_session("u1", "s1")
    assert ok and await adapter.get_all_qa_entries("u1", "s1") == []


@pytest.mark.asyncio
async def test_prune(adapter):
    """Flush all cached data."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.prune()
    assert await adapter.get_all_qa_entries("u1", "s1") == []


# Backward-compatibility tests (add_qa, get_latest_qa, get_all_qas) :TODO: Can be deleted after session manager integration into retrievers
@pytest.mark.asyncio
async def test_add_qa_backward_compat(adapter):
    """Legacy add_qa stores entry with auto-generated qa_id."""
    await adapter.add_qa("u1", "s1", "Q", "C", "A")
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    assert "qa_id" in entries[0]
    assert entries[0]["question"] == "Q" and entries[0]["answer"] == "A"


@pytest.mark.asyncio
async def test_get_all_qas_backward_compat(adapter):
    """Legacy get_all_qas returns same as get_all_qa_entries."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    via_legacy = await adapter.get_all_qas("u1", "s1")
    via_new = await adapter.get_all_qa_entries("u1", "s1")
    assert via_legacy == via_new
    assert len(via_legacy) == 2


@pytest.mark.asyncio
async def test_get_latest_qa_backward_compat(adapter):
    """Legacy get_latest_qa returns same as get_latest_qa_entries."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    await adapter.create_qa_entry("u1", "s1", "Q3", "C3", "A3", qa_id="id3")
    via_legacy = await adapter.get_latest_qa("u1", "s1", last_n=2)
    via_new = await adapter.get_latest_qa_entries("u1", "s1", last_n=2)
    assert via_legacy == via_new
    assert len(via_legacy) == 2
