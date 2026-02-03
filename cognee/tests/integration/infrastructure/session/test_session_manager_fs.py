import tempfile
import pytest
from unittest.mock import patch

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
def session_manager(fs_adapter):
    """SessionManager wired to FsCacheAdapter."""
    return SessionManager(cache_engine=fs_adapter)


@pytest.mark.asyncio
async def test_add_qa_and_get_session(session_manager):
    """Add QA via SessionManager and retrieve via get_session."""
    qa_id = await session_manager.add_qa("u1", "s1", "Q1?", "ctx1", "A1.")
    assert qa_id is not None

    entries = await session_manager.get_session("u1", "s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "Q1?"
    assert entries[0]["answer"] == "A1."
    assert entries[0]["qa_id"] == qa_id


@pytest.mark.asyncio
async def test_get_session_formatted(session_manager):
    """get_session with formatted=True returns prompt string."""
    await session_manager.add_qa("u1", "s1", "Q?", "C", "A")
    formatted = await session_manager.get_session("u1", "s1", formatted=True)
    assert isinstance(formatted, str)
    assert "Previous conversation" in formatted and "Q?" in formatted


@pytest.mark.asyncio
async def test_update_qa(session_manager):
    """update_qa updates entry via FsCacheAdapter."""
    qa_id = await session_manager.add_qa("u1", "s1", "Q", "C", "A")
    ok = await session_manager.update_qa("u1", "s1", qa_id, question="Q updated?")
    assert ok

    entries = await session_manager.get_session("u1", "s1")
    assert entries[0]["question"] == "Q updated?"


@pytest.mark.asyncio
async def test_add_feedback(session_manager):
    """add_feedback sets feedback on entry."""
    qa_id = await session_manager.add_qa("u1", "s1", "Q", "C", "A")
    ok = await session_manager.add_feedback("u1", "s1", qa_id, feedback_score=5)
    assert ok

    entries = await session_manager.get_session("u1", "s1")
    assert entries[0]["feedback_score"] == 5


@pytest.mark.asyncio
async def test_delete_feedback(session_manager):
    """delete_feedback clears feedback."""
    qa_id = await session_manager.add_qa(
        "u1", "s1", "Q", "C", "A", feedback_text="good", feedback_score=4
    )
    ok = await session_manager.delete_feedback("u1", "s1", qa_id)
    assert ok

    entries = await session_manager.get_session("u1", "s1")
    assert entries[0].get("feedback_score") is None
    assert entries[0].get("feedback_text") is None


@pytest.mark.asyncio
async def test_delete_qa(session_manager):
    """delete_qa removes single entry."""
    qa1 = await session_manager.add_qa("u1", "s1", "Q1", "C1", "A1")
    await session_manager.add_qa("u1", "s1", "Q2", "C2", "A2")
    ok = await session_manager.delete_qa("u1", "s1", qa1)
    assert ok

    entries = await session_manager.get_session("u1", "s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "Q2"


@pytest.mark.asyncio
async def test_delete_session(session_manager):
    """delete_session clears all entries."""
    await session_manager.add_qa("u1", "s1", "Q", "C", "A")
    ok = await session_manager.delete_session("u1", "s1")
    assert ok

    entries = await session_manager.get_session("u1", "s1")
    assert entries == []
