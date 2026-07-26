"""Unit tests for SqlCacheAdapter CRUD operations (run on sqlite+aiosqlite, no server)."""

import uuid
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.cache.sql.tables import (
    cache_kv,
    cache_qa_entries,
    cache_session_context,
    cache_trace_entries,
    cache_usage_logs,
)
from cognee.infrastructure.databases.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
)
from cognee.infrastructure.databases.exceptions.exceptions import (
    SharedLadybugLockRequiresRedisError,
)
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)


def _make_adapter(tmp_path, **kwargs) -> SqlCacheAdapter:
    return SqlCacheAdapter(f"sqlite+aiosqlite:///{tmp_path}/cache.db", **kwargs)


@pytest_asyncio.fixture
async def adapter(tmp_path):
    inst = _make_adapter(tmp_path)
    yield inst
    await inst.close()


async def _fetch_expirations(inst, table, **filters):
    """Read raw expires_at values straight from the adapter's engine."""
    query = select(table.c.expires_at)
    for column_name, value in filters.items():
        query = query.where(table.c[column_name] == value)
    async with inst.engine.connect() as connection:
        result = await connection.execute(query)
        return [row[0] for row in result]


async def _backdate_expirations(inst, table, when, **filters):
    """Force expires_at via direct SQL on the adapter's engine."""
    statement = update(table).values(expires_at=when)
    for column_name, value in filters.items():
        statement = statement.where(table.c[column_name] == value)
    async with inst.engine.begin() as connection:
        await connection.execute(statement)


def _past():
    return datetime.now(timezone.utc) - timedelta(seconds=10)


# --------------------------------------------------------------------------- #
# QA entries
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_and_get(adapter):
    """Create a QA entry and retrieve it."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1 and entries[0].qa_id == "id1"
    assert entries[0].question == "Q" and entries[0].context == "C" and entries[0].answer == "A"


@pytest.mark.asyncio
async def test_create_qa_entry_generates_uuid4_qa_id_when_missing(adapter):
    """create_qa_entry without qa_id falls back to a generated uuid4."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A")
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1
    parsed = uuid.UUID(entries[0].qa_id)
    assert parsed.version == 4


@pytest.mark.asyncio
async def test_get_latest_qa_entries_chronological_ordering(adapter):
    """get_latest_qa_entries returns the most recent entries in chronological order."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    await adapter.create_qa_entry("u1", "s1", "Q3", "C3", "A3", qa_id="id3")

    latest = await adapter.get_latest_qa_entries("u1", "s1", last_n=2)
    assert [entry.qa_id for entry in latest] == ["id2", "id3"]

    all_entries = await adapter.get_all_qa_entries("u1", "s1")
    assert [entry.qa_id for entry in all_entries] == ["id1", "id2", "id3"]


@pytest.mark.asyncio
async def test_empty_session_returns_empty_list_for_all_last_n(adapter):
    """[] on empty for every last_n value — including last_n=1 (FS-style, not Redis quirk)."""
    assert await adapter.get_latest_qa_entries("u1", "missing", last_n=1) == []
    assert await adapter.get_latest_qa_entries("u1", "missing", last_n=5) == []
    assert await adapter.get_all_qa_entries("u1", "missing") == []


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
async def test_qa_entry_used_session_context_ids_round_trip(adapter):
    """used_session_context_ids round-trips on create and can be set via update.

    Guards the session-context guidance layer: SessionManager.add_qa always
    forwards this kwarg, so the SQL adapter must accept and persist it.
    """
    await adapter.create_qa_entry(
        "u1", "s1", "Q", "C", "A", qa_id="id1", used_session_context_ids=["ctx1", "ctx2"]
    )
    (entry,) = await adapter.get_all_qa_entries("u1", "s1")
    assert entry.used_session_context_ids == ["ctx1", "ctx2"]

    assert (
        await adapter.update_qa_entry("u1", "s1", "id1", used_session_context_ids=["ctx3"]) is True
    )
    (entry,) = await adapter.get_all_qa_entries("u1", "s1")
    assert entry.used_session_context_ids == ["ctx3"]


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
async def test_update(adapter):
    """Update a QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    ok = await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=5)
    assert ok and (await adapter.get_all_qa_entries("u1", "s1"))[0].feedback_score == 5


@pytest.mark.asyncio
async def test_update_none_preserves_every_field(adapter):
    """Passing None for a field on update preserves the existing value of every field."""
    used_ids = {"node_ids": ["n1"], "edge_ids": ["e1"]}
    await adapter.create_qa_entry(
        "u1",
        "s1",
        "Q",
        "C",
        "A",
        qa_id="id1",
        feedback_text="good",
        feedback_score=5,
        used_graph_element_ids=used_ids,
        memify_metadata={"persist_sessions_in_knowledge_graph": True},
    )

    ok = await adapter.update_qa_entry("u1", "s1", "id1")
    assert ok is True

    entry = (await adapter.get_all_qa_entries("u1", "s1"))[0]
    assert entry.question == "Q"
    assert entry.context == "C"
    assert entry.answer == "A"
    assert entry.qa_id == "id1"
    assert entry.feedback_text == "good"
    assert entry.feedback_score == 5
    assert entry.used_graph_element_ids == used_ids
    assert entry.memify_metadata == {"persist_sessions_in_knowledge_graph": True}


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
async def test_update_invalid_raises(adapter):
    """SessionQAEntryValidationError propagates unwrapped on invalid update payloads."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    with pytest.raises(SessionQAEntryValidationError):
        await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=10)

    with pytest.raises(SessionQAEntryValidationError):
        await adapter.update_qa_entry("u1", "s1", "id1", feedback_text=5)


@pytest.mark.asyncio
async def test_update_missing_qa_id_returns_false(adapter):
    """update_qa_entry returns False when qa_id does not exist."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    assert await adapter.update_qa_entry("u1", "s1", "missing", feedback_score=5) is False
    assert await adapter.delete_feedback("u1", "s1", "missing") is False


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
async def test_delete_entry(adapter):
    """Delete a single QA entry by qa_id (rowcount semantics)."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    ok = await adapter.delete_qa_entry("u1", "s1", "id1")
    assert ok and len(await adapter.get_all_qa_entries("u1", "s1")) == 1

    assert await adapter.delete_qa_entry("u1", "s1", "id1") is False
    assert await adapter.delete_qa_entry("u1", "s1", "missing") is False


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

    assert await adapter.delete_session("u1", "s1") is False


@pytest.mark.asyncio
async def test_delete_session_does_not_touch_other_sessions(adapter):
    """delete_session removes only the targeted user+session rows."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.create_qa_entry("u1", "s2", "Q", "C", "A", qa_id="id1")
    await adapter.create_qa_entry("u2", "s1", "Q", "C", "A", qa_id="id1")

    assert await adapter.delete_session("u1", "s1") is True
    assert await adapter.get_all_qa_entries("u1", "s1") == []
    assert len(await adapter.get_all_qa_entries("u1", "s2")) == 1
    assert len(await adapter.get_all_qa_entries("u2", "s1")) == 1


# --------------------------------------------------------------------------- #
# Agent traces
# --------------------------------------------------------------------------- #


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
async def test_get_agent_trace_session_last_n_returns_most_recent_in_order(adapter):
    """last_n returns only the most recent trace steps, oldest first."""
    for index in range(3):
        await adapter.append_agent_trace_step(
            "u1",
            "s1",
            trace_id=f"t{index}",
            origin_function="plan_trip",
            status="success",
            session_feedback=f"step {index}.",
        )

    entries = await adapter.get_agent_trace_session("u1", "s1", last_n=2)
    assert [entry.trace_id for entry in entries] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_get_agent_trace_feedback_returns_feedback_only(adapter):
    """Trace feedback helper delegates to get_agent_trace_session and keeps order."""
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

    assert await adapter.get_agent_trace_feedback("u1", "s1", last_n=1) == [
        "book_hotel failed. Reason: No availability."
    ]


@pytest.mark.asyncio
async def test_get_agent_trace_count_returns_number_of_steps(adapter):
    """Trace count helper returns the number of stored steps."""
    assert await adapter.get_agent_trace_count("u1", "s1") == 0
    for index in range(2):
        await adapter.append_agent_trace_step(
            "u1",
            "s1",
            trace_id=f"t{index}",
            origin_function="plan_trip",
            status="success",
            session_feedback="ok.",
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
    """Blank trace_id / origin_function raise CacheConnectionError (create-path contract)."""
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


# --------------------------------------------------------------------------- #
# TTL behavior
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_qa_entry_sets_session_ttl_when_enabled(adapter):
    """Writes stamp expires_at on every row of the session when TTL is enabled."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")

    expirations = await _fetch_expirations(adapter, cache_qa_entries, user_id="u1")
    assert len(expirations) == 1
    assert expirations[0] is not None


@pytest.mark.asyncio
async def test_writes_refresh_whole_session_ttl(adapter):
    """A new write slides the expiry of existing rows in the same session forward."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    near_future = datetime.now(timezone.utc) + timedelta(seconds=30)
    await _backdate_expirations(adapter, cache_qa_entries, near_future, qa_id="id1")

    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")

    expirations = await _fetch_expirations(adapter, cache_qa_entries, qa_id="id1")
    refreshed = expirations[0]
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=timezone.utc)
    assert refreshed > near_future + timedelta(seconds=60)


@pytest.mark.asyncio
async def test_reads_do_not_refresh_ttl(adapter):
    """Read-only access leaves expires_at untouched."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.append_agent_trace_step(
        "u1", "s1", trace_id="t1", origin_function="plan_trip", status="success"
    )
    qa_before = await _fetch_expirations(adapter, cache_qa_entries)
    trace_before = await _fetch_expirations(adapter, cache_trace_entries)

    await adapter.get_all_qa_entries("u1", "s1")
    await adapter.get_latest_qa_entries("u1", "s1", last_n=1)
    await adapter.get_agent_trace_session("u1", "s1")
    await adapter.get_agent_trace_count("u1", "s1")

    assert await _fetch_expirations(adapter, cache_qa_entries) == qa_before
    assert await _fetch_expirations(adapter, cache_trace_entries) == trace_before


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled_ttl", [0, None])
async def test_ttl_disabled_stores_rows_without_expiry(tmp_path, disabled_ttl):
    """session_ttl_seconds in (0, None) disables expiry entirely (expires_at stays NULL)."""
    inst = _make_adapter(tmp_path, session_ttl_seconds=disabled_ttl)
    try:
        await inst.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
        await inst.append_agent_trace_step(
            "u1", "s1", trace_id="t1", origin_function="plan_trip", status="success"
        )

        assert await _fetch_expirations(inst, cache_qa_entries) == [None]
        assert await _fetch_expirations(inst, cache_trace_entries) == [None]
        assert len(await inst.get_all_qa_entries("u1", "s1")) == 1
    finally:
        await inst.close()


@pytest.mark.asyncio
async def test_expired_rows_are_invisible(adapter):
    """Rows past expires_at are filtered out of every read and mutation path."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.append_agent_trace_step(
        "u1", "s1", trace_id="t1", origin_function="plan_trip", status="success"
    )

    past = _past()
    await _backdate_expirations(adapter, cache_qa_entries, past)
    await _backdate_expirations(adapter, cache_trace_entries, past)

    assert await adapter.get_all_qa_entries("u1", "s1") == []
    assert await adapter.get_latest_qa_entries("u1", "s1", last_n=1) == []
    assert await adapter.get_agent_trace_session("u1", "s1") == []
    assert await adapter.get_agent_trace_count("u1", "s1") == 0
    assert await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=5) is False
    assert await adapter.delete_qa_entry("u1", "s1", "id1") is False
    # Expired-only session counts as nonexistent.
    assert await adapter.delete_session("u1", "s1") is False


# --------------------------------------------------------------------------- #
# Usage logs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_log_usage_and_get_usage_logs(adapter):
    """Usage logs round-trip, most recent first, scoped per user."""
    await adapter.log_usage("u1", {"endpoint": "/add"})
    await adapter.log_usage("u1", {"endpoint": "/search"})
    await adapter.log_usage("u2", {"endpoint": "/cognify"})

    logs = await adapter.get_usage_logs("u1")
    assert [entry["endpoint"] for entry in logs] == ["/search", "/add"]
    assert await adapter.get_usage_logs("u1", limit=1) == [{"endpoint": "/search"}]
    assert await adapter.get_usage_logs("u3") == []


# --------------------------------------------------------------------------- #
# Key/value storage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_kv_round_trip_and_overwrite(adapter):
    """set_value/get_value/delete_value round-trip; set is an upsert."""
    key = "session_note:u1:s1"
    assert await adapter.get_value(key) is None

    await adapter.set_value(key, "snapshot-1")
    assert await adapter.get_value(key) == "snapshot-1"

    await adapter.set_value(key, "snapshot-2")
    assert await adapter.get_value(key) == "snapshot-2"

    await adapter.delete_value(key)
    assert await adapter.get_value(key) is None

    # Deleting a missing key is a no-op.
    await adapter.delete_value(key)


@pytest.mark.asyncio
async def test_kv_ttl_expiry(adapter):
    """A keyed value with ttl becomes invisible once expires_at passes."""
    key = "checkpoint:test"
    await adapter.set_value(key, "checkpoint", ttl=3600)
    assert await adapter.get_value(key) == "checkpoint"

    expirations = await _fetch_expirations(adapter, cache_kv, key=key)
    assert expirations == [expirations[0]] and expirations[0] is not None

    await _backdate_expirations(adapter, cache_kv, _past(), key=key)
    assert await adapter.get_value(key) is None


@pytest.mark.asyncio
async def test_kv_without_ttl_is_immortal(adapter):
    """ttl=None stores the value without expiry (expires_at stays NULL)."""
    key = "session_note:u1:s1"
    await adapter.set_value(key, "forever")

    assert await _fetch_expirations(adapter, cache_kv, key=key) == [None]
    assert await adapter.get_value(key) == "forever"


# --------------------------------------------------------------------------- #
# Maintenance: prune / close
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_prune_clears_only_the_four_cache_tables(adapter):
    """prune empties cache tables but never touches co-tenant tables in the same DB."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.append_agent_trace_step(
        "u1", "s1", trace_id="t1", origin_function="plan_trip", status="success"
    )
    await adapter.log_usage("u1", {"endpoint": "/add"})
    await adapter.set_value("session_note:u1:s1", "note")

    async with adapter.engine.begin() as connection:
        await connection.execute(text("CREATE TABLE co_tenant (id INTEGER PRIMARY KEY, v TEXT)"))
        await connection.execute(text("INSERT INTO co_tenant (v) VALUES ('keep me')"))

    await adapter.prune()

    assert await adapter.get_all_qa_entries("u1", "s1") == []
    assert await adapter.get_agent_trace_session("u1", "s1") == []
    assert await adapter.get_usage_logs("u1") == []
    assert await adapter.get_value("session_note:u1:s1") is None

    async with adapter.engine.connect() as connection:
        survivors = (await connection.execute(text("SELECT v FROM co_tenant"))).scalars().all()
    assert survivors == ["keep me"]

    for table in (cache_qa_entries, cache_trace_entries, cache_usage_logs, cache_kv):
        assert await _fetch_expirations(adapter, table) == []


@pytest.mark.asyncio
async def test_close_is_idempotent_and_allows_reinitialization(adapter):
    """close can be called repeatedly; a reused instance lazily re-initializes."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.close()
    await adapter.close()

    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert [entry.qa_id for entry in entries] == ["id1"]


# --------------------------------------------------------------------------- #
# Locks
# --------------------------------------------------------------------------- #


def test_acquire_lock_raises_on_sqlite(adapter):
    """Advisory locks need Postgres (or Redis); sqlite URLs must refuse loudly."""
    with pytest.raises(SharedLadybugLockRequiresRedisError):
        adapter.acquire_lock()


def test_release_lock_raises_on_sqlite(adapter):
    """release_lock mirrors acquire_lock's refusal on sqlite URLs."""
    with pytest.raises(SharedLadybugLockRequiresRedisError):
        adapter.release_lock()


# --------------------------------------------------------------------------- #
# Backward-compatibility shims (add_qa, get_latest_qa, get_all_qas)
# --------------------------------------------------------------------------- #


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


@pytest.mark.asyncio
async def test_get_qa_entries_by_ids_returns_matching_rows_in_chronological_order(adapter):
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    await adapter.create_qa_entry("u1", "s1", "Q3", "C3", "A3", qa_id="id3")

    entries = await adapter.get_qa_entries_by_ids("u1", "s1", ["id3", "missing", "id1"])

    assert [entry.qa_id for entry in entries] == ["id1", "id3"]


# --------------------------------------------------------------------------- #
# Session context (active guidance entries)
# --------------------------------------------------------------------------- #


def _ctx(entry_id, section="goals", content="x", kind="context"):
    return {"id": entry_id, "kind": kind, "section": section, "content": content}


@pytest.mark.asyncio
async def test_create_and_get_session_context_entries_preserve_order(adapter):
    """Entries round-trip as dicts in insertion order; both kinds are stored together."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1", content="first"))
    await adapter.create_session_context_entry(
        "u1", "s1", {"id": "f1", "kind": "feedback", "raw_text": "good"}
    )
    entries = await adapter.get_session_context_entries("u1", "s1")
    assert [(e["id"], e["kind"]) for e in entries] == [("c1", "context"), ("f1", "feedback")]
    assert entries[0]["content"] == "first"


@pytest.mark.asyncio
async def test_create_session_context_entry_without_id_is_stored(adapter):
    """An id-less payload is stored (parity with Redis/FS); it is just never updatable."""
    await adapter.create_session_context_entry("u1", "s1", {"kind": "context", "content": "x"})
    entries = await adapter.get_session_context_entries("u1", "s1")
    assert len(entries) == 1
    assert entries[0]["content"] == "x"
    assert "id" not in entries[0]  # payload is stored verbatim, no synthetic id injected


@pytest.mark.asyncio
async def test_get_session_context_entries_empty(adapter):
    """An unknown session returns an empty list, not an error."""
    assert await adapter.get_session_context_entries("u1", "nope") == []


@pytest.mark.asyncio
async def test_update_session_context_entry_shallow_merges(adapter):
    """update shallow-merges into the matching entry and returns True."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1", content="draft"))
    updated = await adapter.update_session_context_entry(
        "u1", "s1", "c1", {"content": "final", "rating": "helpful"}
    )
    assert updated is True
    (entry,) = await adapter.get_session_context_entries("u1", "s1")
    assert entry["content"] == "final"
    assert entry["rating"] == "helpful"
    assert entry["section"] == "goals"  # untouched fields preserved


@pytest.mark.asyncio
async def test_update_session_context_entry_missing_returns_false(adapter):
    """Updating an absent entry_id is a no-op that returns False."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1"))
    assert await adapter.update_session_context_entry("u1", "s1", "ghost", {"x": 1}) is False


@pytest.mark.asyncio
async def test_delete_session_context_returns_existence(adapter):
    """delete_session_context wipes the list; True only when live rows existed."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1"))
    assert await adapter.delete_session_context("u1", "s1") is True
    assert await adapter.get_session_context_entries("u1", "s1") == []
    assert await adapter.delete_session_context("u1", "s1") is False


@pytest.mark.asyncio
async def test_delete_session_also_clears_session_context(adapter):
    """delete_session drops session-context rows alongside QA and trace rows."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1"))
    assert await adapter.delete_session("u1", "s1") is True
    assert await adapter.get_session_context_entries("u1", "s1") == []


@pytest.mark.asyncio
async def test_session_context_isolated_per_session(adapter):
    """Context writes and deletes target only the given user+session."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1"))
    await adapter.create_session_context_entry("u1", "s2", _ctx("c2"))
    await adapter.create_session_context_entry("u2", "s1", _ctx("c3"))
    await adapter.delete_session_context("u1", "s1")
    assert await adapter.get_session_context_entries("u1", "s1") == []
    assert len(await adapter.get_session_context_entries("u1", "s2")) == 1
    assert len(await adapter.get_session_context_entries("u2", "s1")) == 1


@pytest.mark.asyncio
async def test_session_context_write_sets_and_refreshes_ttl(adapter):
    """Context writes stamp expires_at and slide the whole session forward."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1"))
    expirations = await _fetch_expirations(adapter, cache_session_context, user_id="u1")
    assert len(expirations) == 1 and expirations[0] is not None

    near_future = datetime.now(timezone.utc) + timedelta(seconds=30)
    await _backdate_expirations(adapter, cache_session_context, near_future, entry_id="c1")
    await adapter.create_session_context_entry("u1", "s1", _ctx("c2"))
    refreshed = (await _fetch_expirations(adapter, cache_session_context, entry_id="c1"))[0]
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=timezone.utc)
    assert refreshed > near_future + timedelta(seconds=60)


@pytest.mark.asyncio
async def test_session_context_reads_exclude_expired(adapter):
    """An entry past its expires_at is invisible to reads."""
    await adapter.create_session_context_entry("u1", "s1", _ctx("c1"))
    await _backdate_expirations(adapter, cache_session_context, _past(), entry_id="c1")
    assert await adapter.get_session_context_entries("u1", "s1") == []


@pytest.mark.asyncio
async def test_uuid_ids_write_and_read_as_string_keys(adapter):
    """uuid.UUID ids must key the same rows as their string form.

    Callers sometimes hold ``user.id`` as a ``uuid.UUID``; on Postgres the
    asyncpg bind cast then parses as ``text = uuid`` (42883) and on sqlite the
    driver rejects the bind. The StringKey column type stringifies every bind.
    """
    user_uuid, session_uuid = uuid4(), uuid4()
    await adapter.create_qa_entry(user_uuid, session_uuid, "Q", "C", "A", qa_id="id1")

    via_uuid = await adapter.get_all_qa_entries(user_uuid, session_uuid)
    via_str = await adapter.get_all_qa_entries(str(user_uuid), str(session_uuid))
    assert [entry.qa_id for entry in via_uuid] == ["id1"]
    assert [entry.qa_id for entry in via_str] == ["id1"]

    latest = await adapter.get_latest_qa_entries(user_uuid, session_uuid, last_n=5)
    assert [entry.qa_id for entry in latest] == ["id1"]


@pytest.mark.asyncio
async def test_uuid_ids_across_trace_context_and_usage(adapter):
    """UUID user ids work for traces, session context, and usage logs alike."""
    user_uuid, session_uuid = uuid4(), uuid4()

    await adapter.append_agent_trace_step(
        user_uuid, session_uuid, trace_id="t1", origin_function="f", status="ok"
    )
    traces = await adapter.get_agent_trace_session(str(user_uuid), str(session_uuid))
    assert [trace.trace_id for trace in traces] == ["t1"]

    await adapter.create_session_context_entry(user_uuid, session_uuid, _ctx("c1"))
    entries = await adapter.get_session_context_entries(str(user_uuid), str(session_uuid))
    assert [entry["id"] for entry in entries] == ["c1"]
    assert await adapter.update_session_context_entry(user_uuid, session_uuid, "c1", {"note": "n"})

    await adapter.log_usage(user_uuid, {"call": "search"})
    assert len(await adapter.get_usage_logs(str(user_uuid))) == 1

    assert await adapter.delete_session(user_uuid, session_uuid) is True
    assert await adapter.get_all_qa_entries(str(user_uuid), str(session_uuid)) == []
