import pytest
from unittest.mock import AsyncMock, patch

from cognee.tasks.memify.cognify_session import cognify_session
from cognee.exceptions import CogneeValidationError, CogneeSystemError


@pytest.fixture
def mock_cognee():
    """Create a mock cognee module."""
    cognee_mock = AsyncMock()
    cognee_mock.add = AsyncMock()
    cognee_mock.cognify = AsyncMock()
    return cognee_mock


@pytest.mark.asyncio
async def test_cognify_session_success(mock_cognee):
    """Test successful cognification of session data."""
    session_data = (
        "Session ID: test_session\n\nQuestion: What is AI?\n\nAnswer: AI is artificial intelligence"
    )

    with patch("cognee.tasks.memify.cognify_session.cognee", mock_cognee):
        await cognify_session(session_data)

        mock_cognee.add.assert_called_once_with(session_data, node_set=["user_sessions_from_cache"])
        mock_cognee.cognify.assert_called_once()


@pytest.mark.asyncio
async def test_cognify_session_empty_string():
    """Test cognification fails with empty string."""
    with pytest.raises(CogneeValidationError) as exc_info:
        await cognify_session("")

    assert "Session data cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_whitespace_string():
    """Test cognification fails with whitespace-only string."""
    with pytest.raises(CogneeValidationError) as exc_info:
        await cognify_session("   \n\t  ")

    assert "Session data cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_none_data():
    """Test cognification fails with None data."""
    with pytest.raises(CogneeValidationError) as exc_info:
        await cognify_session(None)

    assert "Session data cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_add_failure(mock_cognee):
    """Test cognification handles cognee.add failure."""
    session_data = "Session ID: test\n\nQuestion: test?"
    mock_cognee.add.side_effect = Exception("Add operation failed")

    with patch("cognee.tasks.memify.cognify_session.cognee", mock_cognee):
        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_session(session_data)

        assert "Failed to cognify session data" in str(exc_info.value)
        assert "Add operation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_cognify_failure(mock_cognee):
    """Test cognification handles cognify failure."""
    session_data = "Session ID: test\n\nQuestion: test?"
    mock_cognee.cognify.side_effect = Exception("Cognify operation failed")

    with patch("cognee.tasks.memify.cognify_session.cognee", mock_cognee):
        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_session(session_data)

        assert "Failed to cognify session data" in str(exc_info.value)
        assert "Cognify operation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_re_raises_validation_error(mock_cognee):
    """Test that CogneeValidationError is re-raised as-is."""
    with pytest.raises(CogneeValidationError):
        await cognify_session("")

    # Verify add and cognify were not called
    mock_cognee.add.assert_not_called()
    mock_cognee.cognify.assert_not_called()


@pytest.mark.asyncio
async def test_cognify_session_with_special_characters(mock_cognee):
    """Test cognification with special characters."""
    session_data = "Session: test™ © Question: What's special? Answer: Cognee is special!"

    with patch("cognee.tasks.memify.cognify_session.cognee", mock_cognee):
        await cognify_session(session_data)

        mock_cognee.add.assert_called_once_with(session_data, node_set=["user_sessions_from_cache"])
        mock_cognee.cognify.assert_called_once()
