import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.modules.users.exceptions.exceptions import UserNotFoundError


def _user(id_: str):
    """Minimal user: only .id."""
    return SimpleNamespace(id=id_)


@pytest.fixture
def session_user_none():
    """session_user.get() returns None."""
    with patch("cognee.api.v1.session.session.session_user", SimpleNamespace(get=lambda: None)):
        yield


@pytest.fixture
def session_user_ctx():
    """session_user.get() returns a context user."""
    with patch(
        "cognee.api.v1.session.session.session_user",
        SimpleNamespace(get=lambda: _user("ctx-user-id")),
    ):
        yield


@pytest.fixture
def sm():
    """Minimal SessionManager: only get_session, add_feedback, delete_feedback."""
    s = SimpleNamespace()
    s.get_session = AsyncMock(return_value=[])
    s.add_feedback = AsyncMock(return_value=True)
    s.delete_feedback = AsyncMock(return_value=True)
    with patch("cognee.api.v1.session.session.get_session_manager", return_value=s):
        yield s


# Resolving the user (explicit, context, or default)


class TestResolveUser:
    """User resolution: explicit user, context user, default user, and failures."""

    @pytest.mark.asyncio
    async def test_get_session_uses_explicit_user(self, sm):
        from cognee.api.v1.session.session import get_session

        await get_session(session_id="s1", user=_user("explicit-id"))
        sm.get_session.assert_called_once()
        assert sm.get_session.call_args.kwargs["user_id"] == "explicit-id"

    @pytest.mark.asyncio
    async def test_get_session_raises_when_explicit_user_has_no_id(self, sm):
        from cognee.api.v1.session.session import get_session

        with pytest.raises(CogneeValidationError) as exc_info:
            await get_session(session_id="s1", user=SimpleNamespace(id=None))
        assert "must have an id" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_get_session_uses_context_user_when_user_none(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        await get_session(session_id="s1")
        assert sm.get_session.call_args.kwargs["user_id"] == "ctx-user-id"

    @pytest.mark.asyncio
    async def test_get_session_uses_default_user_when_no_context(self, session_user_none, sm):
        from cognee.api.v1.session.session import get_session

        with patch(
            "cognee.api.v1.session.session.get_default_user",
            return_value=_user("default-id"),
        ):
            await get_session(session_id="s1")
        assert sm.get_session.call_args.kwargs["user_id"] == "default-id"

    @pytest.mark.asyncio
    async def test_get_session_raises_when_default_user_fails(self, session_user_none, sm):
        from cognee.api.v1.session.session import get_session

        with patch(
            "cognee.api.v1.session.session.get_default_user",
            side_effect=UserNotFoundError(),
        ):
            with pytest.raises(CogneeValidationError) as exc_info:
                await get_session(session_id="s1")
        assert "Session prerequisites" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_add_feedback_uses_explicit_user(self, sm):
        from cognee.api.v1.session.session import add_feedback

        await add_feedback(session_id="s1", qa_id="q1", user=_user("explicit-fb"))
        sm.add_feedback.assert_called_once()
        assert sm.add_feedback.call_args.kwargs["user_id"] == "explicit-fb"

    @pytest.mark.asyncio
    async def test_add_feedback_raises_when_explicit_user_has_no_id(self, sm):
        from cognee.api.v1.session.session import add_feedback

        with pytest.raises(CogneeValidationError):
            await add_feedback(session_id="s1", qa_id="q1", user=SimpleNamespace(id=None))

    @pytest.mark.asyncio
    async def test_delete_feedback_uses_resolved_user(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import delete_feedback

        await delete_feedback(session_id="s1", qa_id="q1")
        sm.delete_feedback.assert_called_once_with(
            user_id="ctx-user-id", session_id="s1", qa_id="q1"
        )


# get_session


class TestGetSession:
    """Tests for get_session."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_raw_empty(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        result = await get_session(session_id="s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_raw_none(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        sm.get_session.return_value = None
        result = await get_session(session_id="s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_converts_dict_entries_to_session_qa_entry(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        sm.get_session.return_value = [
            {
                "time": "2024-01-01T12:00:00",
                "question": "Q?",
                "context": "C",
                "answer": "A",
                "qa_id": "qa-1",
                "feedback_text": None,
                "feedback_score": None,
            }
        ]
        result = await get_session(session_id="s1")
        assert len(result) == 1
        assert isinstance(result[0], SessionQAEntry)
        assert result[0].qa_id == "qa-1"
        assert result[0].question == "Q?"

    @pytest.mark.asyncio
    async def test_appends_session_qa_entry_as_is(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        entry = SessionQAEntry(
            time="2024-01-01T12:00:00", question="Q", context="C", answer="A", qa_id="e1"
        )
        sm.get_session.return_value = [entry]
        result = await get_session(session_id="s1")
        assert len(result) == 1
        assert result[0] is entry

    @pytest.mark.asyncio
    async def test_skips_invalid_dict_entry(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        valid = {
            "time": "2024-01-01T12:00:00",
            "question": "Q",
            "context": "C",
            "answer": "A",
            "qa_id": "v1",
        }
        invalid = {"time": "x", "question": "Q"}
        sm.get_session.return_value = [valid, invalid]
        result = await get_session(session_id="s1")
        assert len(result) == 1
        assert result[0].qa_id == "v1"

    @pytest.mark.asyncio
    async def test_skips_non_dict_non_session_qa_entry(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        valid_dict = {
            "time": "2024-01-01T12:00:00",
            "question": "Q",
            "context": "C",
            "answer": "A",
            "qa_id": "v1",
        }
        sm.get_session.return_value = [valid_dict, "not-a-dict", 42]
        result = await get_session(session_id="s1")
        assert len(result) == 1
        assert result[0].qa_id == "v1"

    @pytest.mark.asyncio
    async def test_returns_empty_on_session_manager_exception(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        sm.get_session.side_effect = RuntimeError("cache down")
        result = await get_session(session_id="s1")
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_session_id_and_last_n(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        await get_session(session_id="my_session", last_n=5)
        kw = sm.get_session.call_args.kwargs
        assert kw["session_id"] == "my_session"
        assert kw["last_n"] == 5
        assert kw["formatted"] is False

    @pytest.mark.asyncio
    async def test_default_session_id(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import get_session

        await get_session()
        assert sm.get_session.call_args.kwargs["session_id"] == "default_session"


# add_feedback


class TestAddFeedback:
    """Tests for add_feedback."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import add_feedback

        result = await add_feedback(
            session_id="s1", qa_id="q1", feedback_text="good", feedback_score=5
        )
        assert result is True
        kw = sm.add_feedback.call_args.kwargs
        assert kw["user_id"] == "ctx-user-id"
        assert kw["session_id"] == "s1"
        assert kw["qa_id"] == "q1"
        assert kw["feedback_text"] == "good"
        assert kw["feedback_score"] == 5

    @pytest.mark.asyncio
    async def test_returns_false_when_qa_not_found(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import add_feedback

        sm.add_feedback.return_value = False
        result = await add_feedback(session_id="s1", qa_id="nonexistent", feedback_text="x")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_session_manager_exception(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import add_feedback

        sm.add_feedback.side_effect = RuntimeError("cache error")
        result = await add_feedback(session_id="s1", qa_id="q1", feedback_text="ok")
        assert result is False

    @pytest.mark.asyncio
    async def test_passes_optional_feedback_params(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import add_feedback

        await add_feedback(session_id="s1", qa_id="q1")
        kw = sm.add_feedback.call_args.kwargs
        assert kw["feedback_text"] is None
        assert kw["feedback_score"] is None


# delete_feedback


class TestDeleteFeedback:
    """Tests for delete_feedback."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import delete_feedback

        result = await delete_feedback(session_id="s1", qa_id="q1")
        assert result is True
        sm.delete_feedback.assert_called_once_with(
            user_id="ctx-user-id", session_id="s1", qa_id="q1"
        )

    @pytest.mark.asyncio
    async def test_returns_false_when_qa_not_found(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import delete_feedback

        sm.delete_feedback.return_value = False
        result = await delete_feedback(session_id="s1", qa_id="nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_session_manager_exception(self, session_user_ctx, sm):
        from cognee.api.v1.session.session import delete_feedback

        sm.delete_feedback.side_effect = RuntimeError("cache error")
        result = await delete_feedback(session_id="s1", qa_id="q1")
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_explicit_user(self, session_user_none, sm):
        from cognee.api.v1.session.session import delete_feedback

        await delete_feedback(session_id="s1", qa_id="q1", user=_user("del-user"))
        assert sm.delete_feedback.call_args.kwargs["user_id"] == "del-user"


# Session namespace and package exports


class TestSessionNamespace:
    """Session namespace and package exports."""

    def test_session_has_get_session_add_feedback_delete_feedback(self):
        """cognee.session exposes get_session, add_feedback, delete_feedback."""
        from cognee.api.v1.session import session

        assert hasattr(session, "get_session")
        assert hasattr(session, "add_feedback")
        assert hasattr(session, "delete_feedback")
        assert callable(session.get_session)
        assert callable(session.add_feedback)
        assert callable(session.delete_feedback)

    def test_session_qa_entry_exported(self):
        """SessionQAEntry is exported from session package."""
        from cognee.api.v1.session import SessionQAEntry as Exported

        from cognee.infrastructure.databases.cache.models import SessionQAEntry

        assert Exported is SessionQAEntry

    def test_all_exported(self):
        """__all__ includes get_session, add_feedback, delete_feedback, session, SessionQAEntry."""
        from cognee.api.v1.session import __all__

        assert "get_session" in __all__
        assert "add_feedback" in __all__
        assert "delete_feedback" in __all__
        assert "session" in __all__
        assert "SessionQAEntry" in __all__
