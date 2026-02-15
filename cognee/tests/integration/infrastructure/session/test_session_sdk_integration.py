import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cognee
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.session_manager import SessionManager


def _session_module():
    """Real session.py module (package __init__ replaces session with a SimpleNamespace)."""
    import cognee.api.v1.session  # noqa: F401

    return sys.modules["cognee.api.v1.session.session"]


def _user(id_: str):
    """Minimal user for SDK calls."""
    return SimpleNamespace(id=id_)


class _InMemoryRedisList:
    """Minimal in-memory Redis list emulation for Redis-backed session tests."""

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


@pytest.fixture(params=["fs", "redis"])
def sdk_uses_session_manager(request):
    """
    SessionManager backed by either FsCache or in-memory Redis; SDK is patched to use it.
    Tests run twice (once per backend).
    """
    backend = request.param
    if backend == "fs":
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
                return_value={"data_root_directory": tmpdir},
            ):
                from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                    FSCacheAdapter,
                )

                adapter = FSCacheAdapter()
                session_manager = SessionManager(cache_engine=adapter)
                with patch.object(
                    _session_module(),
                    "get_session_manager",
                    return_value=session_manager,
                ):
                    yield session_manager
                adapter.cache.close()
    else:
        store = _InMemoryRedisList()
        patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
        with (
            patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
            patch(f"{patch_mod}.aioredis.Redis", return_value=store),
        ):
            from cognee.infrastructure.databases.cache.redis.RedisAdapter import (
                RedisAdapter,
            )

            adapter = RedisAdapter(host="localhost", port=6379)
            session_manager = SessionManager(cache_engine=adapter)
            with patch.object(
                _session_module(),
                "get_session_manager",
                return_value=session_manager,
            ):
                yield session_manager


@pytest.mark.asyncio
async def test_sdk_get_session_returns_empty_when_no_data(sdk_uses_session_manager):
    """cognee.session.get_session returns [] when session has no QAs."""
    user = _user("u1")
    result = await cognee.session.get_session(session_id="s1", user=user)
    assert result == []


@pytest.mark.asyncio
async def test_sdk_get_session_returns_entries_after_add_qa(sdk_uses_session_manager):
    """Add QA via SessionManager, then retrieve via cognee.session.get_session."""
    user = _user("u1")
    qa_id = await sdk_uses_session_manager.add_qa(
        user_id="u1", question="Q1?", context="ctx1", answer="A1.", session_id="s1"
    )
    assert qa_id is not None

    result = await cognee.session.get_session(session_id="s1", user=user)
    assert len(result) == 1
    assert isinstance(result[0], SessionQAEntry)
    assert result[0].question == "Q1?"
    assert result[0].answer == "A1."
    assert result[0].qa_id == qa_id


@pytest.mark.asyncio
async def test_sdk_get_session_default_session_id(sdk_uses_session_manager):
    """get_session() without session_id uses default_session."""
    user = _user("u1")
    await sdk_uses_session_manager.add_qa(
        user_id="u1", question="Q?", context="C", answer="A", session_id="default_session"
    )

    result = await cognee.session.get_session(user=user)
    assert len(result) == 1
    assert result[0].question == "Q?"


@pytest.mark.asyncio
async def test_sdk_get_session_last_n(sdk_uses_session_manager):
    """get_session(last_n=N) returns only last N entries."""
    user = _user("u1")
    for i in range(3):
        await sdk_uses_session_manager.add_qa(
            user_id="u1",
            question=f"Q{i}?",
            context=f"C{i}",
            answer=f"A{i}.",
            session_id="s1",
        )

    result = await cognee.session.get_session(session_id="s1", user=user, last_n=2)
    assert len(result) == 2
    assert result[0].question == "Q1?"
    assert result[1].question == "Q2?"


@pytest.mark.asyncio
async def test_sdk_add_feedback_roundtrip(sdk_uses_session_manager):
    """add_feedback via SDK updates entry; get_session returns updated feedback."""
    user = _user("u1")
    qa_id = await sdk_uses_session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )

    ok = await cognee.session.add_feedback(
        session_id="s1",
        qa_id=qa_id,
        feedback_text="Very helpful!",
        feedback_score=5,
        user=user,
    )
    assert ok is True

    entries = await cognee.session.get_session(session_id="s1", user=user)
    assert len(entries) == 1
    assert entries[0].feedback_text == "Very helpful!"
    assert entries[0].feedback_score == 5


@pytest.mark.asyncio
async def test_sdk_delete_feedback_roundtrip(sdk_uses_session_manager):
    """delete_feedback via SDK clears feedback on entry."""
    user = _user("u1")
    qa_id = await sdk_uses_session_manager.add_qa(
        user_id="u1",
        question="Q",
        context="C",
        answer="A",
        session_id="s1",
        feedback_text="good",
        feedback_score=4,
    )

    ok = await cognee.session.delete_feedback(session_id="s1", qa_id=qa_id, user=user)
    assert ok is True

    entries = await cognee.session.get_session(session_id="s1", user=user)
    assert len(entries) == 1
    assert entries[0].feedback_text is None
    assert entries[0].feedback_score is None


@pytest.mark.asyncio
async def test_sdk_add_feedback_returns_false_when_qa_not_found(sdk_uses_session_manager):
    """add_feedback returns False when qa_id does not exist."""
    user = _user("u1")
    await sdk_uses_session_manager.add_qa(
        user_id="u1", question="Q", context="C", answer="A", session_id="s1"
    )

    ok = await cognee.session.add_feedback(
        session_id="s1", qa_id="nonexistent-qa-id", feedback_text="x", user=user
    )
    assert ok is False


@pytest.mark.asyncio
async def test_sdk_delete_feedback_returns_false_when_qa_not_found(sdk_uses_session_manager):
    """delete_feedback returns False when qa_id does not exist."""
    user = _user("u1")
    ok = await cognee.session.delete_feedback(session_id="s1", qa_id="nonexistent-qa-id", user=user)
    assert ok is False


@pytest.mark.asyncio
async def test_sdk_explicit_user_id_passed_to_cache(sdk_uses_session_manager):
    """Session SDK passes resolved user id to SessionManager (isolation by user)."""
    user1 = _user("user-one")
    user2 = _user("user-two")
    await sdk_uses_session_manager.add_qa(
        user_id="user-one", question="Q1", context="C1", answer="A1", session_id="shared"
    )
    await sdk_uses_session_manager.add_qa(
        user_id="user-two", question="Q2", context="C2", answer="A2", session_id="shared"
    )

    entries1 = await cognee.session.get_session(session_id="shared", user=user1)
    entries2 = await cognee.session.get_session(session_id="shared", user=user2)

    assert len(entries1) == 1 and entries1[0].question == "Q1"
    assert len(entries2) == 1 and entries2[0].question == "Q2"
