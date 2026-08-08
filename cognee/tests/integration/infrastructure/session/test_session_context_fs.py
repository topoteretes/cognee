import tempfile
from unittest.mock import patch

import pytest

from cognee.infrastructure.session.session_manager import SessionManager


@pytest.fixture
def fs_adapter():
    """FsCacheAdapter with temp directory."""
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


@pytest.fixture
def session_manager(fs_adapter) -> SessionManager:
    """SessionManager wired to FsCacheAdapter."""
    return SessionManager(cache_engine=fs_adapter)


@pytest.mark.asyncio
async def test_add_qa_with_used_session_context_ids_round_trip(session_manager: SessionManager):
    """add_qa with used_session_context_ids stores and returns it via get_session."""
    qa_id = await session_manager.add_qa(
        user_id="u1",
        question="Q?",
        context="C",
        answer="A",
        session_id="s1",
        used_session_context_ids=["c1", "c2"],
    )
    assert qa_id is not None
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].used_session_context_ids == ["c1", "c2"]


@pytest.mark.asyncio
async def test_update_qa_with_used_session_context_ids(session_manager: SessionManager):
    """update_qa can set used_session_context_ids on an existing entry."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.update_qa(
        user_id="u1",
        qa_id=qa_id,
        used_session_context_ids=["c9"],
        session_id="s1",
    )
    assert ok
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0].used_session_context_ids == ["c9"]


@pytest.mark.asyncio
async def test_session_context_create_get_round_trip(session_manager: SessionManager):
    """create_session_context_entry then get_session_context_entries returns it."""
    entry = {"id": "c1", "kind": "context", "section": "rules", "content": "Use tabs."}
    ok = await session_manager.create_session_context_entry(
        user_id="u1", entry_dump=entry, session_id="s1"
    )
    assert ok
    entries = await session_manager.get_session_context_entries(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0]["id"] == "c1"
    assert entries[0]["kind"] == "context"
    assert entries[0]["content"] == "Use tabs."


@pytest.mark.asyncio
async def test_session_context_mixed_kinds(session_manager: SessionManager):
    """Both context and feedback kinds are stored in the same list and returned together."""
    await session_manager.create_session_context_entry(
        user_id="u1",
        entry_dump={"id": "c1", "kind": "context", "content": "x"},
        session_id="s1",
    )
    await session_manager.create_session_context_entry(
        user_id="u1",
        entry_dump={"id": "f1", "kind": "feedback", "raw_text": "thanks"},
        session_id="s1",
    )
    entries = await session_manager.get_session_context_entries(user_id="u1", session_id="s1")
    kinds = sorted(e["kind"] for e in entries)
    assert kinds == ["context", "feedback"]


@pytest.mark.asyncio
async def test_session_context_update(session_manager: SessionManager):
    """update_session_context_entry shallow-merges by entry id."""
    await session_manager.create_session_context_entry(
        user_id="u1",
        entry_dump={"id": "c1", "kind": "context", "helpful_count": 0},
        session_id="s1",
    )
    ok = await session_manager.update_session_context_entry(
        user_id="u1",
        entry_id="c1",
        merge={"helpful_count": 3, "last_served_at": "now"},
        session_id="s1",
    )
    assert ok
    entries = await session_manager.get_session_context_entries(user_id="u1", session_id="s1")
    assert entries[0]["helpful_count"] == 3
    assert entries[0]["last_served_at"] == "now"
    assert entries[0]["kind"] == "context"


@pytest.mark.asyncio
async def test_session_context_update_missing_id(session_manager: SessionManager):
    """update_session_context_entry returns False when entry id not found."""
    await session_manager.create_session_context_entry(
        user_id="u1", entry_dump={"id": "c1", "kind": "context"}, session_id="s1"
    )
    ok = await session_manager.update_session_context_entry(
        user_id="u1", entry_id="missing", merge={"x": 1}, session_id="s1"
    )
    assert ok is False


@pytest.mark.asyncio
async def test_session_context_delete(session_manager: SessionManager):
    """delete_session_context removes all context entries."""
    await session_manager.create_session_context_entry(
        user_id="u1", entry_dump={"id": "c1", "kind": "context"}, session_id="s1"
    )
    ok = await session_manager.delete_session_context(user_id="u1", session_id="s1")
    assert ok
    entries = await session_manager.get_session_context_entries(user_id="u1", session_id="s1")
    assert entries == []


@pytest.mark.asyncio
async def test_session_context_isolated_by_user_and_session(session_manager: SessionManager):
    """Session-context entries remain isolated by user_id and session_id."""
    await session_manager.create_session_context_entry(
        user_id="u1", entry_dump={"id": "a", "kind": "context"}, session_id="s1"
    )
    await session_manager.create_session_context_entry(
        user_id="u1", entry_dump={"id": "b", "kind": "context"}, session_id="s2"
    )
    await session_manager.create_session_context_entry(
        user_id="u2", entry_dump={"id": "c", "kind": "context"}, session_id="s1"
    )
    u1s1 = await session_manager.get_session_context_entries(user_id="u1", session_id="s1")
    u1s2 = await session_manager.get_session_context_entries(user_id="u1", session_id="s2")
    u2s1 = await session_manager.get_session_context_entries(user_id="u2", session_id="s1")
    assert [e["id"] for e in u1s1] == ["a"]
    assert [e["id"] for e in u1s2] == ["b"]
    assert [e["id"] for e in u2s1] == ["c"]


@pytest.mark.asyncio
async def test_delete_session_clears_context(session_manager: SessionManager):
    """delete_session also clears the session-context list."""
    await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    await session_manager.create_session_context_entry(
        user_id="u1", entry_dump={"id": "c1", "kind": "context"}, session_id="s1"
    )
    ok = await session_manager.delete_session(user_id="u1", session_id="s1")
    assert ok

    qa_entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert qa_entries == []
    context_entries = await session_manager.get_session_context_entries(
        user_id="u1", session_id="s1"
    )
    assert context_entries == []


@pytest.mark.asyncio
async def test_session_context_wrappers_fail_open_when_unavailable():
    """Wrappers return safe defaults when cache is unavailable instead of raising."""
    sm = SessionManager(cache_engine=None)
    assert (
        await sm.create_session_context_entry(
            user_id="u1", entry_dump={"id": "c1", "kind": "context"}, session_id="s1"
        )
        is False
    )
    assert await sm.get_session_context_entries(user_id="u1", session_id="s1") == []
    assert (
        await sm.update_session_context_entry(
            user_id="u1", entry_id="c1", merge={"x": 1}, session_id="s1"
        )
        is False
    )
    assert await sm.delete_session_context(user_id="u1", session_id="s1") is False
