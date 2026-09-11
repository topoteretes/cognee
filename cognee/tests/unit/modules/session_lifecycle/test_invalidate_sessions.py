"""Unit tests for session invalidation on document/dataset deletion (COG-5947).

Exercises the real SessionManager over a filesystem cache adapter; only the
vector-index hooks and the relational session registry are patched out.
"""

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
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ):
            from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                FSCacheAdapter,
            )
            from cognee.infrastructure.session.session_manager import SessionManager

            adapter = FSCacheAdapter()
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
            adapter.cache.close()


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
            "cognee.modules.session_lifecycle.invalidate_sessions.list_sessions_for_dataset",
            new=AsyncMock(return_value=[(USER_ID, SESSION_ID)]),
        ),
        patch.object(session_manager, "delete_session", failing),
    ):
        result = await invalidate_sessions_for_dataset(dataset_id)

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
