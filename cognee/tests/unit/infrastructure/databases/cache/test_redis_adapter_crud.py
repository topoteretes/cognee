"""Unit tests for RedisAdapter CRUD operations."""

from datetime import datetime
from uuid import uuid4
import pytest
from unittest.mock import MagicMock, patch

from cognee.infrastructure.databases.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
)
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)


class _InMemoryRedisList:
    """Minimal in-memory Redis list emulation."""

    def __init__(self):
        self.data: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def rpush(self, key: str, *vals: str):
        self.data.setdefault(key, []).extend(vals)

    async def lrange(self, key: str, start: int, end: int):
        lst = self.data.get(key, [])
        s = start if start >= 0 else len(lst) + start
        e = (end + 1) if end >= 0 else len(lst) + end + 1
        return lst[s:e]

    async def llen(self, key: str):
        return len(self.data.get(key, []))

    async def lindex(self, key: str, idx: int):
        lst = self.data.get(key, [])
        return lst[idx] if -len(lst) <= idx < len(lst) else None

    async def lset(self, key: str, idx: int, val: str):
        self.data[key][idx] = val

    async def delete(self, key: str):
        self.ttls.pop(key, None)
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl
        self.expire_calls.append((key, ttl))

    async def ttl(self, key: str):
        """Return remaining TTL; -1 = no expiry, -2 = key missing."""
        if key not in self.data:
            return -2
        return self.ttls.get(key, -1)

    async def flushdb(self):
        self.data.clear()
        self.ttls.clear()
        self.expire_calls.clear()


@pytest.fixture
def redis_store():
    return _InMemoryRedisList()


@pytest.fixture
def adapter(redis_store):
    patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_mod}.aioredis.Redis", return_value=redis_store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        yield RedisAdapter(host="localhost", port=6379)


@pytest.mark.asyncio
async def test_create_and_get(adapter):
    """Create a QA entry and retrieve it."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1 and entries[0].qa_id == "id1"


@pytest.mark.asyncio
async def test_create_qa_entry_sets_session_ttl_when_enabled(adapter, redis_store):
    """Session keys receive TTL on create when Redis session TTL is enabled."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")

    session_key = "agent_sessions:u1:s1"
    assert await redis_store.ttl(session_key) == 604800
    assert redis_store.expire_calls[-1] == (session_key, 604800)


@pytest.mark.asyncio
async def test_create_qa_entry_does_not_set_session_ttl_when_disabled(redis_store):
    """Session keys do not receive TTL when disabled with 0 or None semantics."""
    patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_mod}.aioredis.Redis", return_value=redis_store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        adapter = RedisAdapter(host="localhost", port=6379, session_ttl_seconds=0)

    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    assert await redis_store.ttl("agent_sessions:u1:s1") == -1
    assert redis_store.expire_calls == []


@pytest.mark.asyncio
async def test_create_qa_entry_with_used_graph_element_ids_round_trip(adapter):
    """create_qa_entry with used_graph_element_ids stores and returns it."""
    used_ids = {"node_ids": ["n1"], "edge_ids": ["e1"]}
    await adapter.create_qa_entry(
        "u1", "s1", "Q", "C", "A", qa_id="id1", used_graph_element_ids=used_ids
    )
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    assert entries[0].used_graph_element_ids == used_ids


@pytest.mark.asyncio
async def test_create_qa_entry_invalid_used_graph_element_ids_raises(adapter):
    """create_qa_entry with invalid used_graph_element_ids (disallowed keys) raises."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    with pytest.raises(CacheConnectionError):
        await adapter.create_qa_entry(
            "u1",
            "s1",
            "Q2",
            "C2",
            "A2",
            qa_id="id2",
            used_graph_element_ids={"invalid_key": ["x"]},
        )


@pytest.mark.asyncio
async def test_append_agent_trace_step_and_get_session(adapter):
    """Append trace steps and retrieve them as a flat ordered session list."""
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        memory_query="trip preferences",
        memory_context="User likes quiet places",
        method_params={"city": "Tokyo"},
        method_return_value="Plan created",
        session_feedback="plan_trip succeeded.",
    )
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t2",
        origin_function="book_hotel",
        status="error",
        method_params={"area": "Shibuya"},
        error_message="No availability",
        session_feedback="book_hotel failed. Reason: No availability.",
    )

    entries = await adapter.get_agent_trace_session("u1", "s1")
    assert [entry.trace_id for entry in entries] == ["t1", "t2"]
    assert entries[0].origin_function == "plan_trip"
    assert entries[1].origin_function == "book_hotel"
    assert entries[1].error_message == "No availability"


@pytest.mark.asyncio
async def test_append_agent_trace_step_sets_trace_ttl_when_enabled(adapter, redis_store):
    """Trace keys receive TTL on append when Redis session TTL is enabled."""
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        session_feedback="plan_trip succeeded.",
    )

    trace_key = "agent_traces:u1:s1"
    assert await redis_store.ttl(trace_key) == 604800
    assert redis_store.expire_calls[-1] == (trace_key, 604800)


@pytest.mark.asyncio
async def test_get_agent_trace_feedback_returns_feedback_only(adapter):
    """Trace feedback helper returns ordered feedback strings only."""
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        session_feedback="plan_trip succeeded.",
    )
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t2",
        origin_function="book_hotel",
        status="error",
        error_message="No availability",
        session_feedback="book_hotel failed. Reason: No availability.",
    )

    feedback = await adapter.get_agent_trace_feedback("u1", "s1")
    assert feedback == [
        "plan_trip succeeded.",
        "book_hotel failed. Reason: No availability.",
    ]


@pytest.mark.asyncio
async def test_get_agent_trace_feedback_returns_only_last_n(adapter):
    """Trace feedback helper can return only the most recent feedback strings."""
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        session_feedback="plan_trip succeeded.",
    )
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t2",
        origin_function="book_hotel",
        status="error",
        session_feedback="book_hotel failed.",
    )

    feedback = await adapter.get_agent_trace_feedback("u1", "s1", last_n=1)
    assert feedback == ["book_hotel failed."]


@pytest.mark.asyncio
async def test_get_agent_trace_count_returns_number_of_steps(adapter):
    """Trace count helper returns the number of stored steps."""
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        session_feedback="plan_trip succeeded.",
    )
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t2",
        origin_function="book_hotel",
        status="error",
        session_feedback="book_hotel failed.",
    )

    assert await adapter.get_agent_trace_count("u1", "s1") == 2


@pytest.mark.asyncio
async def test_get_agent_trace_session_missing_returns_empty(adapter):
    """Missing trace sessions return an empty list."""
    assert await adapter.get_agent_trace_session("u1", "missing") == []
    assert await adapter.get_agent_trace_feedback("u1", "missing") == []


@pytest.mark.asyncio
async def test_append_agent_trace_step_sanitizes_non_json_safe_values(adapter):
    """Trace persistence sanitizes UUIDs, datetimes, and custom objects into JSON-safe values."""

    class _Obj:
        def __init__(self):
            self.id = "obj-1"

    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        method_params={
            "trip_id": uuid4(),
            "created_at": datetime(2026, 4, 14, 12, 0, 0),
            "obj": _Obj(),
        },
        method_return_value={"result_id": uuid4(), "owner": _Obj()},
        session_feedback="plan_trip succeeded.",
    )

    entries = await adapter.get_agent_trace_session("u1", "s1")
    assert isinstance(entries[0].method_params["trip_id"], str)
    assert isinstance(entries[0].method_params["created_at"], str)
    assert entries[0].method_params["obj"] == {"type": "_Obj", "id": "obj-1"}
    assert isinstance(entries[0].method_return_value["result_id"], str)
    assert entries[0].method_return_value["owner"] == {"type": "_Obj", "id": "obj-1"}


@pytest.mark.asyncio
async def test_append_agent_trace_step_rejects_blank_required_fields(adapter):
    """Trace persistence rejects blank required fields via model validation."""
    with pytest.raises(CacheConnectionError):
        await adapter.append_agent_trace_step(
            "u1",
            "s1",
            trace_id=" ",
            origin_function="plan_trip",
            status="success",
            session_feedback="plan_trip succeeded.",
        )

    with pytest.raises(CacheConnectionError):
        await adapter.append_agent_trace_step(
            "u1",
            "s1",
            trace_id="t1",
            origin_function=" ",
            status="success",
            session_feedback="plan_trip succeeded.",
        )


@pytest.mark.asyncio
async def test_get_agent_trace_session_does_not_refresh_ttl(adapter, redis_store):
    """Read-only trace access should not refresh TTL."""
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        session_feedback="plan_trip succeeded.",
    )
    redis_store.expire_calls.clear()

    entries = await adapter.get_agent_trace_session("u1", "s1")

    assert len(entries) == 1
    assert redis_store.expire_calls == []


@pytest.mark.asyncio
async def test_update(adapter):
    """Update a QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    ok = await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=5)
    assert ok and (await adapter.get_all_qa_entries("u1", "s1"))[0].feedback_score == 5


@pytest.mark.asyncio
async def test_update_refreshes_session_ttl(adapter, redis_store):
    """Session TTL is refreshed after QA updates."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    redis_store.expire_calls.clear()

    ok = await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=5)

    assert ok is True
    assert redis_store.expire_calls == [("agent_sessions:u1:s1", 604800)]


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
    assert e.feedback_text is None and e.feedback_score is None


@pytest.mark.asyncio
async def test_delete_feedback_refreshes_session_ttl(adapter, redis_store):
    """Clearing feedback should refresh the session TTL."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.update_qa_entry("u1", "s1", "id1", feedback_text="good", feedback_score=5)
    redis_store.expire_calls.clear()

    ok = await adapter.delete_feedback("u1", "s1", "id1")

    assert ok is True
    assert redis_store.expire_calls == [("agent_sessions:u1:s1", 604800)]


@pytest.mark.asyncio
async def test_update_memify_metadata_merges_existing_keys(adapter):
    """update_qa_entry merges memify_metadata keys instead of replacing the map."""
    await adapter.create_qa_entry(
        "u1",
        "s1",
        "Q",
        "C",
        "A",
        qa_id="id1",
        memify_metadata={"persist_sessions_in_knowledge_graph": True},
    )
    ok = await adapter.update_qa_entry(
        "u1",
        "s1",
        "id1",
        memify_metadata={MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: False},
    )
    assert ok
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert entries[0].memify_metadata == {
        "persist_sessions_in_knowledge_graph": True,
        MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: False,
    }


@pytest.mark.asyncio
async def test_delete_entry(adapter):
    """Delete a single QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    ok = await adapter.delete_qa_entry("u1", "s1", "id1")
    assert ok and len(await adapter.get_all_qa_entries("u1", "s1")) == 1


@pytest.mark.asyncio
async def test_delete_entry_reapplies_session_ttl_after_rewrite(adapter, redis_store):
    """Rewriting the session list during delete preserves the configured TTL."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    redis_store.expire_calls.clear()

    ok = await adapter.delete_qa_entry("u1", "s1", "id1")

    assert ok is True
    assert await redis_store.ttl("agent_sessions:u1:s1") == 604800
    assert redis_store.expire_calls == [("agent_sessions:u1:s1", 604800)]


@pytest.mark.asyncio
async def test_delete_session(adapter):
    """Delete the entire session, including QA and trace entries."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.append_agent_trace_step(
        "u1",
        "s1",
        trace_id="t1",
        origin_function="plan_trip",
        status="success",
        session_feedback="plan_trip succeeded.",
    )
    ok = await adapter.delete_session("u1", "s1")
    assert ok is True
    assert await adapter.get_all_qa_entries("u1", "s1") == []
    assert await adapter.get_agent_trace_session("u1", "s1") == []


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
    assert entries[0].qa_id is not None
    assert entries[0].question == "Q" and entries[0].answer == "A"


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
