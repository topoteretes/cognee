"""Unit tests for session invalidation on document/dataset deletion (COG-5947).

Exercises the real SessionManager over a filesystem cache adapter; only the
vector-index hooks and the relational session registry are patched out.
"""

import asyncio
import tempfile
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.session_persist_watermark import (
    get_persisted_qa_count,
    save_persisted_qa_count,
)
from cognee.modules.session_lifecycle.invalidate_sessions import (
    _invalidate_session_entries,
    invalidate_sessions_for_dataset,
    invalidate_sessions_for_deleted_data,
)

USER_ID = str(uuid4())
SESSION_ID = "test_session"


@pytest.fixture
def session_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
        from cognee.infrastructure.session.session_manager import SessionManager

        adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmpdir}/cache.db")
        with (
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vector",
                new=AsyncMock(),
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new=AsyncMock(),
            ),
        ):
            yield SessionManager(adapter)
        asyncio.run(adapter.close())


async def _seed_qa(session_manager, qa_id, used_node_ids=None, used_context_ids=None):
    await session_manager._cache.create_qa_entry(
        USER_ID,
        SESSION_ID,
        question=f"question {qa_id}",
        context="context",
        answer=f"answer {qa_id}",
        qa_id=qa_id,
        used_graph_element_ids={"node_ids": used_node_ids} if used_node_ids else None,
        used_session_context_ids=used_context_ids,
    )


async def _seed_context(session_manager, entry_dump):
    await session_manager._cache.create_session_context_entry(USER_ID, SESSION_ID, entry_dump)


@pytest.mark.asyncio
async def test_targeted_invalidation_removes_direct_and_transitive_entries(session_manager):
    """A turn using a deleted node is removed, along with the feedback entry
    referencing it, the lesson distilled from that feedback, and the later turn
    that consumed the lesson. Unrelated turns survive."""
    await _seed_qa(session_manager, "qa_direct", used_node_ids=["node_deleted", "node_other"])
    await _seed_qa(session_manager, "qa_unrelated", used_node_ids=["node_other"])
    await _seed_qa(session_manager, "qa_downstream", used_context_ids=["lesson_1"])
    await _seed_context(
        session_manager,
        {"id": "feedback_1", "kind": "feedback", "referenced_qa_ids": ["qa_direct"]},
    )
    await _seed_context(
        session_manager,
        {"id": "lesson_1", "kind": "context", "source_feedback_ids": ["feedback_1"]},
    )
    await _seed_context(
        session_manager,
        {"id": "lesson_unrelated", "kind": "context", "source_feedback_ids": ["feedback_x"]},
    )

    qa_deleted, context_deleted = await _invalidate_session_entries(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        deleted_node_ids={"node_deleted"},
        deleted_edge_ids=set(),
    )

    assert qa_deleted == 2
    assert context_deleted == 2
    surviving = await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID)
    assert [entry.qa_id for entry in surviving] == ["qa_unrelated"]
    surviving_context = await session_manager.get_session_context_entries(
        user_id=USER_ID, session_id=SESSION_ID
    )
    assert [entry["id"] for entry in surviving_context] == ["lesson_unrelated"]


@pytest.mark.asyncio
async def test_no_intersection_deletes_nothing(session_manager):
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_a"])

    qa_deleted, context_deleted = await _invalidate_session_entries(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        deleted_node_ids={"node_b"},
        deleted_edge_ids=set(),
    )

    assert (qa_deleted, context_deleted) == (0, 0)
    surviving = await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID)
    assert len(surviving) == 1


@pytest.mark.asyncio
async def test_edge_id_intersection_matches(session_manager):
    await session_manager._cache.create_qa_entry(
        USER_ID,
        SESSION_ID,
        question="q",
        context="c",
        answer="a",
        qa_id="qa_edge",
        used_graph_element_ids={"edge_ids": ["edge_deleted"]},
    )

    qa_deleted, _ = await _invalidate_session_entries(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        deleted_node_ids=set(),
        deleted_edge_ids={"edge_deleted"},
    )

    assert qa_deleted == 1
    assert await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID) == []


@pytest.mark.asyncio
async def test_watermark_clamped_after_targeted_delete(session_manager):
    """Deleting persisted turns must clamp the persist watermark, otherwise the
    next improve() treats it as stale and re-persists the whole session."""
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_deleted"])
    await _seed_qa(session_manager, "qa_2", used_node_ids=["node_deleted"])
    await _seed_qa(session_manager, "qa_3", used_node_ids=["node_other"])
    await save_persisted_qa_count(session_manager, USER_ID, SESSION_ID, 3)

    await _invalidate_session_entries(
        session_manager,
        user_id=USER_ID,
        session_id=SESSION_ID,
        deleted_node_ids={"node_deleted"},
        deleted_edge_ids=set(),
    )

    assert await get_persisted_qa_count(session_manager, USER_ID, SESSION_ID) == 1


@pytest.mark.asyncio
async def test_watermark_clamp_recounts_after_external_delete(session_manager):
    """The clamp writes a fresh post-delete recount, not the pre-delete
    snapshot — entries removed by a concurrent actor between the initial read
    and the clamp are reflected, so overlapping invalidations converge."""
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_deleted"])
    await _seed_qa(session_manager, "qa_2", used_node_ids=["node_other"])
    await _seed_qa(session_manager, "qa_3", used_node_ids=["node_other"])
    await save_persisted_qa_count(session_manager, USER_ID, SESSION_ID, 3)

    original_delete_qa = session_manager.delete_qa

    async def delete_qa_and_one_more(**kwargs):
        # Simulate a concurrent invalidation removing qa_3 mid-flight.
        deleted = await original_delete_qa(**kwargs)
        await original_delete_qa(user_id=USER_ID, qa_id="qa_3", session_id=SESSION_ID)
        return deleted

    with patch.object(session_manager, "delete_qa", side_effect=delete_qa_and_one_more):
        await _invalidate_session_entries(
            session_manager,
            user_id=USER_ID,
            session_id=SESSION_ID,
            deleted_node_ids={"node_deleted"},
            deleted_edge_ids=set(),
        )

    # Snapshot math would write 3 - 1 = 2; the fresh recount sees only qa_2.
    assert await get_persisted_qa_count(session_manager, USER_ID, SESSION_ID) == 1


@pytest.mark.asyncio
async def test_invalidate_sessions_for_dataset_deletes_attributed_sessions(session_manager):
    dataset_id = uuid4()
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_a"])

    with (
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.backend_access_control_enabled",
            return_value=False,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session._delete_session_rows",
            new=AsyncMock(return_value=set()),
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
            new=AsyncMock(return_value=[(USER_ID, SESSION_ID)]),
        ),
    ):
        result = await invalidate_sessions_for_dataset(dataset_id)

    assert result == {"sessions_considered": 1, "sessions_deleted": 1}
    assert await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID) == []


@pytest.mark.asyncio
async def test_invalidate_sessions_for_deleted_data_spans_sessions(session_manager):
    dataset_id = uuid4()
    other_session = "other_session"
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_deleted"])
    await session_manager._cache.create_qa_entry(
        USER_ID,
        other_session,
        question="q",
        context="c",
        answer="a",
        qa_id="qa_other",
        used_graph_element_ids={"node_ids": ["node_deleted"]},
    )

    with (
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
            new=AsyncMock(return_value=[(USER_ID, SESSION_ID), (USER_ID, other_session)]),
        ),
    ):
        result = await invalidate_sessions_for_deleted_data(dataset_id, {"node_deleted"}, set())

    assert result["sessions_considered"] == 2
    assert result["qa_entries_deleted"] == 2
    assert await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID) == []
    assert await session_manager.get_session(user_id=USER_ID, session_id=other_session) == []


@pytest.mark.asyncio
async def test_invalidate_sessions_for_deleted_data_scans_unattributed_user_sessions(
    session_manager,
):
    """Sessions with no dataset attribution are scanned for every user.
    Overlap with the dataset-attributed list is deduplicated."""
    dataset_id = uuid4()
    other_user = uuid4()
    unattributed_session = "default_session"
    await _seed_qa(session_manager, "qa_attributed", used_node_ids=["node_deleted"])
    await session_manager._cache.create_qa_entry(
        USER_ID,
        unattributed_session,
        question="q",
        context="c",
        answer="a",
        qa_id="qa_unattributed",
        used_graph_element_ids={"node_ids": ["node_deleted"]},
    )

    with (
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
            new=AsyncMock(return_value=[(USER_ID, SESSION_ID)]),
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_unattributed_sessions",
            new=AsyncMock(
                return_value=[
                    (USER_ID, unattributed_session),
                    (USER_ID, SESSION_ID),
                    (other_user, unattributed_session),
                ]
            ),
        ) as list_unattributed_mock,
    ):
        result = await invalidate_sessions_for_deleted_data(
            dataset_id, {"node_deleted"}, set(), user_id=uuid4()
        )

    list_unattributed_mock.assert_awaited_once_with()
    assert result["sessions_considered"] == 3
    assert result["qa_entries_deleted"] == 2
    assert await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID) == []
    assert await session_manager.get_session(user_id=USER_ID, session_id=unattributed_session) == []


@pytest.mark.asyncio
async def test_invalidate_sessions_for_deleted_data_lists_unattributed_without_user(
    session_manager,
):
    """Unattributed sessions are listed even when user_id is omitted."""
    with (
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_unattributed_sessions",
            new=AsyncMock(return_value=[]),
        ) as list_unattributed_mock,
    ):
        result = await invalidate_sessions_for_deleted_data(uuid4(), {"node_x"}, set())

    list_unattributed_mock.assert_awaited_once_with()
    assert result["sessions_considered"] == 0


@pytest.mark.asyncio
async def test_invalidate_sessions_for_deleted_data_noops_on_empty_ids(session_manager):
    with patch(
        "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
        new=AsyncMock(),
    ) as list_mock:
        result = await invalidate_sessions_for_deleted_data(uuid4(), set(), set())

    assert result["sessions_considered"] == 0
    list_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_delete_failure_is_nonfatal(session_manager):
    """A failing session delete is swallowed and the rest still runs."""
    dataset_id = uuid4()
    failing = AsyncMock(side_effect=RuntimeError("cache down"))

    with (
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.get_session_manager",
            return_value=session_manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.backend_access_control_enabled",
            return_value=False,
        ),
        patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
            new=AsyncMock(return_value=[(USER_ID, SESSION_ID)]),
        ),
        patch.object(session_manager, "delete_sessions", failing),
    ):
        result = await invalidate_sessions_for_dataset(dataset_id)

    failing.assert_awaited_once()
    assert result == {"sessions_considered": 1, "sessions_deleted": 0}


@pytest.mark.asyncio
async def test_data_delete_purges_unattributed_default_session_end_to_end(session_manager):
    """The COG-6292 leak, wired through the real listing queries: an unscoped
    search's turns live in the plain ``default_session`` row (``dataset_id``
    NULL, no dataset suffix), which dataset-scoped listing can never find. A
    data-level delete must discover that row via the real ``dataset_id IS
    NULL`` query — no listing helpers mocked — and remove the turns that used
    the deleted elements, so the session read path that feeds recall/search
    history has nothing left to replay."""
    from datetime import datetime, timezone
    from uuid import UUID

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.models import SessionRecord

    dataset_id = uuid4()
    user_uuid = UUID(USER_ID)
    unattributed_session = "default_session"

    # Session-cache state: one turn contaminated by the deleted node, one clean.
    await session_manager._cache.create_qa_entry(
        USER_ID,
        unattributed_session,
        question="What does the deleted note say?",
        context="deleted note context",
        answer="answer derived from the deleted note",
        qa_id="qa_deleted_fact",
        used_graph_element_ids={"node_ids": ["node_deleted"]},
    )
    await session_manager._cache.create_qa_entry(
        USER_ID,
        unattributed_session,
        question="Unrelated question",
        context="c",
        answer="unrelated answer",
        qa_id="qa_clean",
        used_graph_element_ids={"node_ids": ["node_other"]},
    )

    # Relational state: the suffix-less default-session row with no attribution.
    now = datetime.now(timezone.utc)
    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)
    async with engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id=unattributed_session,
                user_id=user_uuid,
                dataset_id=None,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        await session.commit()

    try:
        with patch(
            "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
            return_value=session_manager,
        ):
            result = await invalidate_sessions_for_deleted_data(
                dataset_id, {"node_deleted"}, set(), user_id=user_uuid
            )

        # The unattributed session was found through the real query and only
        # the contaminated turn was removed.
        assert result["sessions_considered"] >= 1
        assert result["qa_entries_deleted"] == 1
        surviving = await session_manager.get_session(
            user_id=USER_ID, session_id=unattributed_session
        )
        assert [entry.qa_id for entry in surviving] == ["qa_clean"]
    finally:
        async with engine.get_async_session() as session:
            row = await session.get(SessionRecord, (unattributed_session, user_uuid))
            if row:
                await session.delete(row)
            await session.commit()


@pytest.mark.asyncio
async def test_list_unattributed_sessions_returns_null_dataset_id_rows():
    """The listing query is dataset_id IS NULL, with no user filter."""
    from datetime import datetime, timezone

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.metrics import list_unattributed_sessions
    from cognee.modules.session_lifecycle.models import SessionRecord

    now = datetime.now(timezone.utc)
    user_a = uuid4()
    user_b = uuid4()
    dataset_id = uuid4()
    unattributed_a = f"default_session_unattr_{user_a}"
    unattributed_b = f"default_session_unattr_{user_b}"
    attributed = f"default_session_{dataset_id}"

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    async with engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id=unattributed_a,
                user_id=user_a,
                dataset_id=None,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        session.add(
            SessionRecord(
                session_id=unattributed_b,
                user_id=user_b,
                dataset_id=None,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        session.add(
            SessionRecord(
                session_id=attributed,
                user_id=user_a,
                dataset_id=dataset_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        await session.commit()

    try:
        listed = await list_unattributed_sessions()
        listed_pairs = {(row[0], row[1]) for row in listed}
        assert (user_a, unattributed_a) in listed_pairs
        assert (user_b, unattributed_b) in listed_pairs
        assert (user_a, attributed) not in listed_pairs
    finally:
        async with engine.get_async_session() as session:
            for session_id, user_id in (
                (unattributed_a, user_a),
                (unattributed_b, user_b),
                (attributed, user_a),
            ):
                row = await session.get(SessionRecord, (session_id, user_id))
                if row:
                    await session.delete(row)
            await session.commit()


@pytest.mark.asyncio
async def test_list_sessions_for_datasets_never_attributes_by_a_session_name_suffix():
    """A row counts by its dataset_id; a legacy row without one only as the dataset's exact
    default session. A client-chosen name ending in the dataset id attributes nothing."""
    from datetime import datetime, timezone

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.metrics import list_sessions_for_datasets
    from cognee.modules.session_lifecycle.models import SessionRecord

    now = datetime.now(timezone.utc)
    owner, other = uuid4(), uuid4()
    dataset_id, elsewhere = uuid4(), uuid4()
    rows = [
        ("chat", owner, dataset_id),  # attributed by its row
        (f"default_session_{dataset_id}", owner, None),  # legacy default session
        (f"notes_{dataset_id}", other, None),  # a client name ending in the id
        (f"default_session_{dataset_id}", other, elsewhere),  # attributed elsewhere
    ]

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)
    async with engine.get_async_session() as session:
        for session_id, user_id, row_dataset in rows:
            session.add(
                SessionRecord(
                    session_id=session_id,
                    user_id=user_id,
                    dataset_id=row_dataset,
                    status="running",
                    started_at=now,
                    last_activity_at=now,
                )
            )
        await session.commit()

    try:
        listed = set(await list_sessions_for_datasets([dataset_id]))
        assert listed == {(owner, "chat"), (owner, f"default_session_{dataset_id}")}
    finally:
        async with engine.get_async_session() as session:
            for session_id, user_id, _ in rows:
                row = await session.get(SessionRecord, (session_id, user_id))
                if row:
                    await session.delete(row)
            await session.commit()


@pytest.mark.asyncio
async def test_cleanup_bounds_parallel_sessions_and_finishes_them_before_strict_failure(
    monkeypatch,
):
    import asyncio
    import importlib

    module = importlib.import_module("cognee.modules.session_lifecycle.invalidate_sessions")
    sessions = [(uuid4(), f"session-{i}") for i in range(20)]
    manager = type("Manager", (), {"is_available": True})()
    monkeypatch.setattr(module, "get_session_manager", lambda: manager)
    monkeypatch.setattr(module, "list_sessions_for_dataset", AsyncMock(return_value=sessions))
    monkeypatch.setattr(module, "list_unattributed_sessions", AsyncMock(return_value=[]))
    active = peak = completed = 0

    async def clean(*args, session_id, **kwargs):
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            if session_id == "session-0":
                raise RuntimeError("cache write failed")
            return 1, 0
        finally:
            active -= 1
            completed += 1

    monkeypatch.setattr(module, "_invalidate_session_entries", clean)
    with pytest.raises(RuntimeError, match="cache write failed"):
        await module.invalidate_sessions_for_deleted_data(uuid4(), {"node"}, set(), strict=True)
    assert 1 < peak <= 8
    assert completed == len(sessions)
    assert active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("lesson_still_stored", [False, True])
async def test_strict_cleanup_counts_a_missing_entry_as_done_but_not_a_surviving_lesson(
    monkeypatch, lesson_still_stored
):
    """An entry a concurrent cleanup already removed is done; a lesson whose fail-open
    delete returned False but which a re-read still finds keeps the decision pending."""
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("cognee.modules.session_lifecycle.invalidate_sessions")
    turn = SimpleNamespace(
        qa_id="stale",
        used_graph_element_ids={"node_ids": ["conflict"], "edge_ids": []},
        used_session_context_ids=[],
    )
    lesson = {"id": "feedback-1", "kind": "feedback", "referenced_qa_ids": ["stale"]}
    manager = SimpleNamespace(
        is_available=True,
        get_session=AsyncMock(return_value=[turn]),
        get_session_context_entries=AsyncMock(
            side_effect=[[lesson], [lesson] if lesson_still_stored else []]
        ),
        delete_qa=AsyncMock(return_value=False),
        delete_session_context_entry=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(module, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        module, "list_sessions_for_dataset", AsyncMock(return_value=[(uuid4(), "s1")])
    )
    monkeypatch.setattr(module, "list_unattributed_sessions", AsyncMock(return_value=[]))
    if lesson_still_stored:
        with pytest.raises(RuntimeError, match="cached lesson"):
            await module.invalidate_sessions_for_deleted_data(
                uuid4(), {"conflict"}, set(), strict=True
            )
    else:
        totals = await module.invalidate_sessions_for_deleted_data(
            uuid4(), {"conflict"}, set(), strict=True
        )
        assert totals == {
            "sessions_considered": 1,
            "qa_entries_deleted": 0,
            "context_entries_deleted": 0,
        }
    manager.delete_qa.assert_awaited_once()
    manager.delete_session_context_entry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_read", ["initial", "re-read"])
async def test_strict_cleanup_fails_on_an_unreadable_session_context(
    session_manager, monkeypatch, failing_read
):
    """The manager's context reads are fail-open. Strict cleanup must not read an outage
    as a session without lessons, or as a lesson that is gone after a failed delete."""
    await _seed_qa(session_manager, "qa_stale", used_node_ids=["conflict"])
    await _seed_context(
        session_manager,
        {"id": "feedback_1", "kind": "feedback", "referenced_qa_ids": ["qa_stale"]},
    )
    cache = session_manager._cache
    read = cache.get_session_context_entries
    reads = 0

    async def flaky_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        if failing_read == "initial" or reads > 1:
            raise RuntimeError("cache down")
        return await read(*args, **kwargs)

    async def failing_delete(*args, **kwargs):
        raise RuntimeError("cache down")

    monkeypatch.setattr(cache, "get_session_context_entries", flaky_read)
    monkeypatch.setattr(cache, "delete_session_context_entry", failing_delete)
    cleanup = {
        "user_id": USER_ID,
        "session_id": SESSION_ID,
        "deleted_node_ids": {"conflict"},
        "deleted_edge_ids": set(),
    }
    with pytest.raises(RuntimeError, match="cache down"):
        await _invalidate_session_entries(session_manager, **cleanup, strict=True)
    if failing_read == "initial":
        # forget() and dataset deletes stay best-effort: the same outage is not raised.
        assert await _invalidate_session_entries(session_manager, **cleanup) == (1, 0)


@pytest.mark.asyncio
async def test_invalidate_sessions_for_dataset_cleans_session_rows_and_agent_connections(
    session_manager,
):
    from datetime import datetime, timezone
    from uuid import UUID

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.agents.registry import (
        clear_registered_agent_connections,
        list_registered_agent_connections,
        register_agent_connection,
    )
    from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord

    dataset_id = uuid4()
    user_uuid = UUID(USER_ID)
    now = datetime.now(timezone.utc)
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_a"])
    clear_registered_agent_connections()

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)
    async with engine.get_async_session() as db_session:
        db_session.add(
            SessionRecord(
                session_id=SESSION_ID,
                user_id=user_uuid,
                dataset_id=dataset_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        db_session.add(
            SessionModelUsage(
                session_id=SESSION_ID,
                user_id=user_uuid,
                model="gpt-4o-mini",
                kind="llm",
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.001,
                updated_at=now,
            )
        )
        await db_session.commit()

    try:
        with patch("cognee.modules.agents.registry._persist_agent_connection", new=AsyncMock()):
            await register_agent_connection(
                agent_session_name="claude_live_session",
                connection_type="claude_code",
                memory_mode="hybrid",
                user_id=user_uuid,
                session_id=SESSION_ID,
                datasets=[
                    {
                        "id": str(dataset_id),
                        "name": "demo_ds",
                        "role": "read_write",
                        "type": "dataset",
                    }
                ],
            )

        with (
            patch(
                "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
                return_value=session_manager,
            ),
            patch(
                "cognee.modules.session_lifecycle.delete_session.get_session_manager",
                return_value=session_manager,
            ),
            patch(
                "cognee.modules.session_lifecycle.delete_session.backend_access_control_enabled",
                return_value=False,
            ),
        ):
            result = await invalidate_sessions_for_dataset(dataset_id)

        assert result == {"sessions_considered": 1, "sessions_deleted": 1}
        assert await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID) == []
        async with engine.get_async_session() as db_session:
            assert await db_session.get(SessionRecord, (SESSION_ID, user_uuid)) is None
            assert (
                await db_session.get(SessionModelUsage, (SESSION_ID, user_uuid, "gpt-4o-mini"))
                is None
            )

        conns = list_registered_agent_connections()
        assert len(conns) == 1
        assert conns[0].datasets == []
        assert conns[0].status == "inactive"
    finally:
        clear_registered_agent_connections()
        async with engine.get_async_session() as db_session:
            usage = await db_session.get(SessionModelUsage, (SESSION_ID, user_uuid, "gpt-4o-mini"))
            if usage:
                await db_session.delete(usage)
            record = await db_session.get(SessionRecord, (SESSION_ID, user_uuid))
            if record:
                await db_session.delete(record)
            await db_session.commit()


@pytest.mark.asyncio
async def test_invalidate_sessions_for_dataset_waits_for_improve_and_turn_locks(session_manager):
    """Dataset session invalidation routes through delete_sessions, so an in-flight
    improve() claim or session_turn_lock finishes writing before the delete runs
    and cannot resurrect cache entries or relational rows afterwards (#6557)."""
    from datetime import datetime, timezone
    from uuid import UUID

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.locks import (
        acquire_improve_lock_many,
        release_improve_lock_many,
        session_turn_lock,
    )
    from cognee.infrastructure.locks.session_lock import improve_lock_keys
    from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord

    dataset_id = uuid4()
    user_uuid = UUID(USER_ID)
    now = datetime.now(timezone.utc)
    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    await _seed_qa(session_manager, "qa_initial", used_node_ids=["node_a"])
    async with engine.get_async_session() as db_session:
        db_session.add(
            SessionRecord(
                session_id=SESSION_ID,
                user_id=user_uuid,
                dataset_id=dataset_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        await db_session.commit()

    claim_keys = improve_lock_keys([SESSION_ID], user_uuid)
    try:
        with (
            patch(
                "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
                return_value=session_manager,
            ),
            patch(
                "cognee.modules.session_lifecycle.delete_session.get_session_manager",
                return_value=session_manager,
            ),
            patch(
                "cognee.modules.session_lifecycle.delete_session.backend_access_control_enabled",
                return_value=False,
            ),
        ):
            # 1) Hold the session's improve claim while dataset invalidation starts.
            await acquire_improve_lock_many(claim_keys)
            claimed = True
            try:
                invalidate_task = asyncio.create_task(invalidate_sessions_for_dataset(dataset_id))
                await asyncio.sleep(0.05)
                assert not invalidate_task.done()

                # Simulate an in-flight improve/sync writing cache + usage before releasing.
                await _seed_qa(session_manager, "qa_during_improve", used_node_ids=["node_a"])
                async with engine.get_async_session() as db_session:
                    db_session.add(
                        SessionModelUsage(
                            session_id=SESSION_ID,
                            user_id=user_uuid,
                            model="gpt-4o",
                            kind="llm",
                            tokens_in=20,
                            tokens_out=10,
                            cost_usd=0.01,
                            updated_at=now,
                        )
                    )
                    await db_session.commit()
            finally:
                if claimed:
                    await release_improve_lock_many(claim_keys)

            result = await asyncio.wait_for(invalidate_task, timeout=5)
            assert result == {"sessions_considered": 1, "sessions_deleted": 1}
            assert await session_manager.get_session(user_id=USER_ID, session_id=SESSION_ID) == []
            async with engine.get_async_session() as db_session:
                assert await db_session.get(SessionRecord, (SESSION_ID, user_uuid)) is None
                assert (
                    await db_session.get(SessionModelUsage, (SESSION_ID, user_uuid, "gpt-4o"))
                    is None
                )

            # 2) Hold the per-session turn lock while dataset invalidation starts.
            turn_session = f"{SESSION_ID}_turn"
            async with engine.get_async_session() as db_session:
                db_session.add(
                    SessionRecord(
                        session_id=turn_session,
                        user_id=user_uuid,
                        dataset_id=dataset_id,
                        status="running",
                        started_at=now,
                        last_activity_at=now,
                    )
                )
                await db_session.commit()

            async with session_turn_lock(str(user_uuid), turn_session):
                turn_invalidate = asyncio.create_task(invalidate_sessions_for_dataset(dataset_id))
                await asyncio.sleep(0.05)
                assert not turn_invalidate.done()
                await session_manager._cache.create_qa_entry(
                    USER_ID,
                    turn_session,
                    question="in-flight turn",
                    context="ctx",
                    answer="ans",
                    qa_id="qa_in_flight_turn",
                )

            turn_result = await asyncio.wait_for(turn_invalidate, timeout=5)
            assert turn_result == {"sessions_considered": 1, "sessions_deleted": 1}
            assert await session_manager.get_session(user_id=USER_ID, session_id=turn_session) == []
            async with engine.get_async_session() as db_session:
                assert await db_session.get(SessionRecord, (turn_session, user_uuid)) is None
    finally:
        async with engine.get_async_session() as db_session:
            usage = await db_session.get(SessionModelUsage, (SESSION_ID, user_uuid, "gpt-4o"))
            if usage:
                await db_session.delete(usage)
            for sid in (SESSION_ID, f"{SESSION_ID}_turn"):
                row = await db_session.get(SessionRecord, (sid, user_uuid))
                if row:
                    await db_session.delete(row)
            await db_session.commit()


@pytest.mark.asyncio
async def test_invalidate_sessions_for_dataset_skips_deleted_dataset_store_and_preserves_context(
    session_manager,
):
    """delete_sessions receives datasets_being_deleted=[dataset_id], so the deleted
    dataset's vector store is skipped while surviving readable datasets and the
    global store are cleaned without clobbering the caller's dataset ContextVars."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from uuid import UUID

    from cognee.context_global_variables import graph_db_config, vector_db_config
    from cognee.infrastructure.files.storage.config import file_storage_config

    dataset_id = uuid4()
    surviving_dataset_id = uuid4()
    user_uuid = UUID(USER_ID)
    await _seed_qa(session_manager, "qa_1", used_node_ids=["node_a"])

    entered_datasets: list[UUID] = []
    global_vector_configs_seen: list[dict | None] = []

    @asynccontextmanager
    async def fake_dataset_context(ds_id, uid):
        entered_datasets.append(ds_id)
        vector_db_config.set({"vector_db_name": f"vec_{ds_id}"})
        graph_db_config.set({"graph_database_name": f"graph_{ds_id}"})
        file_storage_config.set({"data_root_directory": f"root_{ds_id}"})
        yield

    async def record_global_detag(pairs):
        list(pairs)
        global_vector_configs_seen.append(vector_db_config.get())

    dataset_detag = AsyncMock()
    readable = [SimpleNamespace(id=dataset_id), SimpleNamespace(id=surviving_dataset_id)]
    caller_vec = {"vector_db_name": f"vec_{dataset_id}"}
    caller_graph = {"graph_database_name": f"graph_{dataset_id}"}
    caller_storage = {"data_root_directory": f"root_{dataset_id}"}

    from cognee.modules.session_lifecycle import delete_session as delete_session_mod

    vec_tok = vector_db_config.set(caller_vec)
    graph_tok = graph_db_config.set(caller_graph)
    storage_tok = file_storage_config.set(caller_storage)
    try:
        with (
            patch(
                "cognee.modules.session_lifecycle.invalidate_sessions.get_session_manager",
                return_value=session_manager,
            ),
            patch(
                "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
                new=AsyncMock(return_value=[(user_uuid, SESSION_ID)]),
            ),
            patch.object(delete_session_mod, "get_session_manager", return_value=session_manager),
            patch.object(delete_session_mod, "backend_access_control_enabled", return_value=True),
            patch.object(
                delete_session_mod,
                "get_user",
                new=AsyncMock(return_value=SimpleNamespace(id=user_uuid)),
            ),
            patch.object(
                delete_session_mod,
                "get_all_user_permission_datasets",
                new=AsyncMock(return_value=readable),
            ),
            patch.object(
                delete_session_mod,
                "_recorded_datasets",
                new=AsyncMock(return_value={(user_uuid, SESSION_ID): dataset_id}),
            ),
            patch.object(
                delete_session_mod,
                "_existing_datasets",
                new=AsyncMock(return_value={dataset_id}),
            ),
            patch.object(
                delete_session_mod,
                "set_database_global_context_variables",
                fake_dataset_context,
            ),
            patch.object(delete_session_mod, "delete_session_qa_vectors", dataset_detag),
            patch.object(
                delete_session_mod,
                "_delete_session_rows",
                new=AsyncMock(return_value={(user_uuid, SESSION_ID)}),
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                side_effect=record_global_detag,
            ),
        ):
            result = await invalidate_sessions_for_dataset(dataset_id)

        assert result == {"sessions_considered": 1, "sessions_deleted": 1}
        assert entered_datasets == [surviving_dataset_id]
        assert global_vector_configs_seen == [None]
        assert set(dataset_detag.await_args.args[0]) == {(USER_ID, SESSION_ID)}
        assert vector_db_config.get() == caller_vec
        assert graph_db_config.get() == caller_graph
        assert file_storage_config.get() == caller_storage
    finally:
        vector_db_config.reset(vec_tok)
        graph_db_config.reset(graph_tok)
        file_storage_config.reset(storage_tok)
