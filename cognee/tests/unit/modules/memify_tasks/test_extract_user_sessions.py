import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.tasks.memify.extract_user_sessions import extract_user_sessions
from cognee.exceptions import CogneeSystemError
from cognee.modules.users.models import User

# Get the actual module object (not the function) for patching
extract_user_sessions_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock(spec=User)
    user.id = "test-user-123"
    return user


@pytest.fixture
def mock_qa_data():
    """Create mock Q&A data."""
    return [
        {
            "question": "What is cognee?",
            "context": "context about cognee",
            "answer": "Cognee is a knowledge graph solution",
            "time": "2025-01-01T12:00:00",
        },
        {
            "question": "How does it work?",
            "context": "how it works context",
            "answer": "It processes data and creates graphs",
            "time": "2025-01-01T12:05:00",
        },
    ]


@pytest.mark.asyncio
async def test_extract_user_sessions_success(mock_user, mock_qa_data):
    """Test successful extraction of sessions."""
    mock_cache_engine = AsyncMock()
    mock_cache_engine.get_all_qas.return_value = mock_qa_data

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module, "get_cache_engine", return_value=mock_cache_engine
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["test_session"]):
            sessions.append(session)

        assert len(sessions) == 1
        assert "Session ID: test_session" in sessions[0]
        assert "Question: What is cognee?" in sessions[0]
        assert "Answer: Cognee is a knowledge graph solution" in sessions[0]
        assert "Question: How does it work?" in sessions[0]
        assert "Answer: It processes data and creates graphs" in sessions[0]


@pytest.mark.asyncio
async def test_extract_user_sessions_multiple_sessions(mock_user, mock_qa_data):
    """Test extraction of multiple sessions."""
    mock_cache_engine = AsyncMock()
    mock_cache_engine.get_all_qas.return_value = mock_qa_data

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module, "get_cache_engine", return_value=mock_cache_engine
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["session1", "session2"]):
            sessions.append(session)

        assert len(sessions) == 2
        assert mock_cache_engine.get_all_qas.call_count == 2


@pytest.mark.asyncio
async def test_extract_user_sessions_no_data(mock_user, mock_qa_data):
    """Test extraction handles empty data parameter."""
    mock_cache_engine = AsyncMock()
    mock_cache_engine.get_all_qas.return_value = mock_qa_data

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module, "get_cache_engine", return_value=mock_cache_engine
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions(None, session_ids=["test_session"]):
            sessions.append(session)

        assert len(sessions) == 1


@pytest.mark.asyncio
async def test_extract_user_sessions_no_session_ids(mock_user):
    """Test extraction handles no session IDs provided."""
    mock_cache_engine = AsyncMock()

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module, "get_cache_engine", return_value=mock_cache_engine
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=None):
            sessions.append(session)

        assert len(sessions) == 0
        mock_cache_engine.get_all_qas.assert_not_called()


@pytest.mark.asyncio
async def test_extract_user_sessions_empty_qa_data(mock_user):
    """Test extraction handles empty Q&A data."""
    mock_cache_engine = AsyncMock()
    mock_cache_engine.get_all_qas.return_value = []

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module, "get_cache_engine", return_value=mock_cache_engine
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["empty_session"]):
            sessions.append(session)

        assert len(sessions) == 0


@pytest.mark.asyncio
async def test_extract_user_sessions_cache_error_handling(mock_user, mock_qa_data):
    """Test extraction continues on cache error for specific session."""
    mock_cache_engine = AsyncMock()
    mock_cache_engine.get_all_qas.side_effect = [
        mock_qa_data,
        Exception("Cache error"),
        mock_qa_data,
    ]

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module, "get_cache_engine", return_value=mock_cache_engine
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions(
            [{}], session_ids=["session1", "session2", "session3"]
        ):
            sessions.append(session)

        assert len(sessions) == 2
