import pytest
from unittest.mock import AsyncMock, MagicMock

from cognee.infrastructure.databases.exceptions import SessionParameterValidationError
from cognee.infrastructure.session.session_manager import (
    SessionManager,
    _validate_session_params,
)


class TestValidateSessionParams:
    """Tests for _validate_session_params."""

    def test_valid_params(self):
        """Valid user_id and session_id do not raise."""
        _validate_session_params(user_id="u1", session_id="s1")
        _validate_session_params(user_id="u1", session_id="s1", qa_id="q1")

    def test_empty_user_id_raises(self):
        """Empty user_id raises SessionParameterValidationError."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="", session_id="s1")
        assert "user_id" in exc_info.value.message

    def test_empty_session_id_raises(self):
        """Empty session_id raises SessionParameterValidationError."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="")
        assert "session_id" in exc_info.value.message

    def test_whitespace_user_id_raises(self):
        """Whitespace-only user_id raises."""
        with pytest.raises(SessionParameterValidationError):
            _validate_session_params(user_id="  ", session_id="s1")

    def test_empty_qa_id_raises(self):
        """Empty qa_id raises when provided."""
        with pytest.raises(SessionParameterValidationError) as exc_info:
            _validate_session_params(user_id="u1", session_id="s1", qa_id="")
        assert "qa_id" in exc_info.value.message


class TestSessionManager:
    """Unit tests for SessionManager with mocked cache."""

    @pytest.fixture
    def mock_cache(self):
        """Mock cache engine."""
        cache = MagicMock()
        cache.create_qa_entry = AsyncMock()
        cache.get_all_qa_entries = AsyncMock(return_value=[])
        cache.get_latest_qa_entries = AsyncMock(return_value=[])
        cache.update_qa_entry = AsyncMock(return_value=True)
        cache.delete_feedback = AsyncMock(return_value=True)
        cache.delete_qa_entry = AsyncMock(return_value=True)
        cache.delete_session = AsyncMock(return_value=True)
        return cache

    @pytest.fixture
    def sm(self, mock_cache):
        """SessionManager with mocked cache."""
        return SessionManager(cache_engine=mock_cache)

    @pytest.fixture
    def sm_unavailable(self):
        """SessionManager with no cache."""
        return SessionManager(cache_engine=None)

    def test_normalize_session_id(self, sm):
        """normalize_session_id returns default when None."""
        assert sm.normalize_session_id(None) == "default_session"
        assert sm.normalize_session_id("custom") == "custom"

    def test_is_available(self, sm, sm_unavailable):
        """is_available reflects cache presence."""
        assert sm.is_available is True
        assert sm_unavailable.is_available is False

    @pytest.mark.asyncio
    async def test_add_qa_returns_qa_id(self, sm, mock_cache):
        """add_qa returns generated qa_id and calls cache."""
        qa_id = await sm.add_qa("u1", "s1", "Q", "C", "A")
        assert qa_id is not None
        mock_cache.create_qa_entry.assert_called_once()
        call_kw = mock_cache.create_qa_entry.call_args.kwargs
        assert call_kw["user_id"] == "u1"
        assert call_kw["session_id"] == "s1"
        assert call_kw["question"] == "Q"
        assert call_kw["answer"] == "A"
        assert call_kw["qa_id"] == qa_id

    @pytest.mark.asyncio
    async def test_add_qa_unavailable_returns_none(self, sm_unavailable):
        """add_qa returns None when cache unavailable."""
        assert await sm_unavailable.add_qa("u1", "s1", "Q", "C", "A") is None

    @pytest.mark.asyncio
    async def test_add_qa_invalid_params_raises(self, sm):
        """add_qa raises on invalid user_id or session_id."""
        with pytest.raises(SessionParameterValidationError):
            await sm.add_qa("", "s1", "Q", "C", "A")
        with pytest.raises(SessionParameterValidationError):
            await sm.add_qa("u1", "", "Q", "C", "A")

    def test_format_entries_empty(self):
        """format_entries returns empty string for empty list."""
        assert SessionManager.format_entries([]) == ""

    def test_format_entries_formats(self):
        """format_entries produces expected format."""
        entries = [
            {"time": "t1", "question": "Q1", "context": "C1", "answer": "A1"},
        ]
        out = SessionManager.format_entries(entries)
        assert "Previous conversation" in out
        assert "Q1" in out and "A1" in out

    @pytest.mark.asyncio
    async def test_get_session_calls_cache(self, sm, mock_cache):
        """get_session delegates to cache."""
        mock_cache.get_all_qa_entries.return_value = [
            {"qa_id": "1", "question": "Q", "context": "C", "answer": "A", "time": "t"}
        ]
        entries = await sm.get_session("u1", "s1")
        assert len(entries) == 1
        assert entries[0]["question"] == "Q"
        mock_cache.get_all_qa_entries.assert_called_once_with("u1", "s1")

    @pytest.mark.asyncio
    async def test_get_session_formatted(self, sm, mock_cache):
        """get_session with formatted=True returns string."""
        mock_cache.get_all_qa_entries.return_value = [
            {"qa_id": "1", "question": "Q", "context": "C", "answer": "A", "time": "t"}
        ]
        out = await sm.get_session("u1", "s1", formatted=True)
        assert isinstance(out, str)
        assert "Previous conversation" in out and "Q" in out

    @pytest.mark.asyncio
    async def test_get_session_unavailable_returns_empty(self, sm_unavailable):
        """get_session returns empty list when cache unavailable."""
        assert await sm_unavailable.get_session("u1", "s1") == []
        assert await sm_unavailable.get_session("u1", "s1", formatted=True) == ""

    @pytest.mark.asyncio
    async def test_update_qa_calls_cache(self, sm, mock_cache):
        """update_qa delegates to cache."""
        ok = await sm.update_qa("u1", "s1", "q1", question="Q2")
        assert ok is True
        mock_cache.update_qa_entry.assert_called_once_with(
            user_id="u1",
            session_id="s1",
            qa_id="q1",
            question="Q2",
            context=None,
            answer=None,
            feedback_text=None,
            feedback_score=None,
        )

    @pytest.mark.asyncio
    async def test_delete_feedback_calls_cache(self, sm, mock_cache):
        """delete_feedback delegates to cache."""
        ok = await sm.delete_feedback("u1", "s1", "q1")
        assert ok is True
        mock_cache.delete_feedback.assert_called_once_with(
            user_id="u1", session_id="s1", qa_id="q1"
        )

    @pytest.mark.asyncio
    async def test_delete_qa_calls_cache(self, sm, mock_cache):
        """delete_qa delegates to cache."""
        ok = await sm.delete_qa("u1", "s1", "q1")
        assert ok is True
        mock_cache.delete_qa_entry.assert_called_once_with(
            user_id="u1", session_id="s1", qa_id="q1"
        )

    @pytest.mark.asyncio
    async def test_delete_session_calls_cache(self, sm, mock_cache):
        """delete_session delegates to cache."""
        ok = await sm.delete_session("u1", "s1")
        assert ok is True
        mock_cache.delete_session.assert_called_once_with(user_id="u1", session_id="s1")
