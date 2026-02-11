"""Integration tests for SessionManager with RedisAdapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.infrastructure.session.session_manager import SessionManager


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

    async def flushdb(self):
        self.data.clear()


@pytest.fixture
def redis_adapter():
    """RedisAdapter with in-memory backend."""
    store = _InMemoryRedisList()
    patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
    with (
        patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
        patch(f"{patch_mod}.aioredis.Redis", return_value=store),
    ):
        from cognee.infrastructure.databases.cache.redis.RedisAdapter import RedisAdapter

        yield RedisAdapter(host="localhost", port=6379)


@pytest.fixture
def session_manager(redis_adapter):
    """SessionManager wired to RedisAdapter."""
    return SessionManager(cache_engine=redis_adapter)


@pytest.mark.asyncio
async def test_add_qa_and_get_session(session_manager):
    """Add QA via SessionManager and retrieve via get_session."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )
    assert qa_id is not None

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "Q1?"
    assert entries[0]["answer"] == "A1."
    assert entries[0]["qa_id"] == qa_id


@pytest.mark.asyncio
async def test_get_session_formatted(session_manager):
    """get_session with formatted=True returns prompt string."""
    await session_manager.add_qa(
        user_id="u1", question="Q?", context="C", answer="A", session_id="s1"
    )
    formatted = await session_manager.get_session(user_id="u1", formatted=True, session_id="s1")
    assert isinstance(formatted, str)
    assert "Previous conversation" in formatted and "Q?" in formatted


@pytest.mark.asyncio
async def test_update_qa(session_manager):
    """update_qa updates entry via RedisAdapter."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.update_qa(
        user_id="u1", qa_id=qa_id, question="Q updated?", session_id="s1"
    )
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0]["question"] == "Q updated?"


@pytest.mark.asyncio
async def test_add_feedback(session_manager):
    """add_feedback sets feedback on entry."""
    qa_id = await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.add_feedback(
        user_id="u1", qa_id=qa_id, feedback_score=5, session_id="s1"
    )
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0]["feedback_score"] == 5


@pytest.mark.asyncio
async def test_delete_feedback(session_manager):
    """delete_feedback clears feedback."""
    qa_id = await session_manager.add_qa(
        user_id="u1",
        question="Q",
        context="C",
        answer="A",
        session_id="s1",
        feedback_text="good",
        feedback_score=4,
    )
    ok = await session_manager.delete_feedback(user_id="u1", qa_id=qa_id, session_id="s1")
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries[0].get("feedback_score") is None
    assert entries[0].get("feedback_text") is None


@pytest.mark.asyncio
async def test_delete_qa(session_manager):
    """delete_qa removes single entry."""
    qa1 = await session_manager.add_qa(
        user_id="u1", question="Q1", context="C1", answer="A1", session_id="s1"
    )
    await session_manager.add_qa(
        user_id="u1", question="Q2", context="C2", answer="A2", session_id="s1"
    )
    ok = await session_manager.delete_qa(user_id="u1", qa_id=qa1, session_id="s1")
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "Q2"


@pytest.mark.asyncio
async def test_delete_session(session_manager):
    """delete_session clears all entries."""
    await session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )
    ok = await session_manager.delete_session(user_id="u1", session_id="s1")
    assert ok

    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert entries == []


@pytest.mark.asyncio
async def test_generate_completion_with_session_saves_qa(session_manager):
    """generate_completion_with_session runs completion and saves QA to session (LLM mocked)."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    with (
        patch("cognee.infrastructure.session.session_manager.session_user") as mock_session_user,
        patch("cognee.infrastructure.session.session_manager.CacheConfig") as mock_config_cls,
        patch(
            "cognee.infrastructure.session.session_manager.generate_completion",
            new_callable=AsyncMock,
            return_value="Integration test answer",
        ),
    ):
        mock_session_user.get.return_value = mock_user
        mock_config = MagicMock()
        mock_config.caching = True
        mock_config_cls.return_value = mock_config

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="What is X?",
            context="Context about X.",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Integration test answer"
    entries = await session_manager.get_session(user_id="u1", session_id="s1")
    assert len(entries) == 1
    assert entries[0]["question"] == "What is X?"
    assert entries[0]["answer"] == "Integration test answer"
