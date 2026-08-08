import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.infrastructure.session.session_persist_watermark import SessionPersistWindow
from cognee.tasks.memify.cognify_session import cognify_session
from cognee.exceptions import CogneeValidationError, CogneeSystemError

# Get the actual module object (not the function) for patching
cognify_session_module = sys.modules["cognee.tasks.memify.cognify_session"]


def _window(text: str, persisted_qa_count: int = 1) -> SessionPersistWindow:
    return SessionPersistWindow(
        user_id="test-user-123",
        session_id="test_session",
        text=text,
        persisted_qa_count=persisted_qa_count,
    )


def _mock_session_manager():
    mock_sm = MagicMock()
    mock_sm.update_session_context_entry = AsyncMock(return_value=True)
    mock_sm.create_session_context_entry = AsyncMock(return_value=True)
    return mock_sm


@pytest.mark.asyncio
async def test_cognify_session_success():
    """Test successful cognification of a session window."""
    window = _window(
        "Session ID: test_session\n\nQuestion: What is AI?\n\nAnswer: AI is artificial intelligence",
        persisted_qa_count=1,
    )
    mock_sm = _mock_session_manager()

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(cognify_session_module, "get_session_manager", return_value=mock_sm),
    ):
        await cognify_session(window, dataset_id="123")

        mock_add.assert_called_once_with(
            window.text,
            dataset_id="123",
            node_set=["user_sessions_from_cache"],
            user=None,
        )
        mock_cognify.assert_called_once_with(datasets=["123"], user=None)
        # Watermark advanced after successful cognify.
        mock_sm.update_session_context_entry.assert_called_once()


@pytest.mark.asyncio
async def test_cognify_session_accepts_batched_windows():
    """The pipeline runner delivers windows in list batches; each is processed."""
    windows = [
        _window("Question: q1?\n\nAnswer: a1\n\n", persisted_qa_count=1),
        _window("Question: q2?\n\nAnswer: a2\n\n", persisted_qa_count=2),
    ]
    mock_sm = _mock_session_manager()

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(cognify_session_module, "get_session_manager", return_value=mock_sm),
    ):
        await cognify_session(windows, dataset_id="123")

        assert mock_add.call_count == 2
        assert mock_cognify.call_count == 2
        assert mock_sm.update_session_context_entry.call_count == 2


@pytest.mark.asyncio
async def test_cognify_session_empty_window_text():
    """Test cognification fails with a whitespace-only window."""
    with pytest.raises(CogneeValidationError) as exc_info:
        await cognify_session(_window("   \n\t  "))

    assert "Session window cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_none_data():
    """Test cognification fails with None data."""
    with pytest.raises(CogneeValidationError) as exc_info:
        await cognify_session(None)

    assert "Session window cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_plain_string_rejected():
    """The old string contract is gone: only SessionPersistWindow is accepted."""
    with pytest.raises(CogneeValidationError):
        await cognify_session("Session ID: test\n\nQuestion: test?")


@pytest.mark.asyncio
async def test_cognify_session_add_failure():
    """Test cognification handles cognee.add failure without advancing the watermark."""
    mock_sm = _mock_session_manager()

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
        patch.object(cognify_session_module, "get_session_manager", return_value=mock_sm),
    ):
        mock_add.side_effect = Exception("Add operation failed")

        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_session(_window("Question: test?"))

        assert "Failed to cognify session data" in str(exc_info.value)
        assert "Add operation failed" in str(exc_info.value)
        mock_sm.update_session_context_entry.assert_not_called()
        mock_sm.create_session_context_entry.assert_not_called()


@pytest.mark.asyncio
async def test_cognify_session_cognify_failure():
    """Test cognification handles cognify failure without advancing the watermark."""
    mock_sm = _mock_session_manager()

    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(cognify_session_module, "get_session_manager", return_value=mock_sm),
    ):
        mock_cognify.side_effect = Exception("Cognify operation failed")

        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_session(_window("Question: test?"))

        assert "Failed to cognify session data" in str(exc_info.value)
        assert "Cognify operation failed" in str(exc_info.value)
        mock_sm.update_session_context_entry.assert_not_called()
        mock_sm.create_session_context_entry.assert_not_called()


@pytest.mark.asyncio
async def test_cognify_session_re_raises_validation_error():
    """Test that CogneeValidationError is re-raised as-is."""
    with pytest.raises(CogneeValidationError):
        await cognify_session([])


@pytest.mark.asyncio
async def test_cognify_session_with_special_characters():
    """Test cognification with special characters."""
    window = _window("Session: test™ © Question: What's special? Answer: Cognee is special!")
    mock_sm = _mock_session_manager()

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(cognify_session_module, "get_session_manager", return_value=mock_sm),
    ):
        await cognify_session(window, dataset_id="123")

        mock_add.assert_called_once_with(
            window.text,
            dataset_id="123",
            node_set=["user_sessions_from_cache"],
            user=None,
        )
        mock_cognify.assert_called_once_with(datasets=["123"], user=None)


@pytest.mark.asyncio
async def test_cognify_session_passes_user_to_add_and_cognify():
    """Test user is forwarded to cognee.add/cognee.cognify."""
    window = _window(
        "Session ID: test_session\n\nQuestion: What is AI?\n\nAnswer: AI is artificial intelligence"
    )
    user = object()
    mock_sm = _mock_session_manager()

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(cognify_session_module, "get_session_manager", return_value=mock_sm),
    ):
        await cognify_session(window, dataset_id="123", user=user)

        mock_add.assert_called_once_with(
            window.text,
            dataset_id="123",
            node_set=["user_sessions_from_cache"],
            user=user,
        )
        mock_cognify.assert_called_once_with(datasets=["123"], user=user)
