import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.context_global_variables import session_user


def create_mock_cache_engine(qa_history=None):
    """Mocking cache engine as it is tested somewhere else"""
    mock_cache = AsyncMock()

    if qa_history is None:
        qa_history = []

    mock_cache.get_latest_qa = AsyncMock(return_value=qa_history)
    mock_cache.add_qa = AsyncMock(return_value=None)

    return mock_cache


def create_mock_user():
    """Create a mock user without database access"""
    mock_user = MagicMock()
    mock_user.id = "test-user-id-123"
    return mock_user


class TestConversationHistoryUtils:
    """Test the two utility functions: get_conversation_history and save_to_session_cache"""

    @pytest.mark.asyncio
    async def test_get_conversation_history_returns_empty_when_no_history(self):
        """Test get_conversation_history returns empty string when no history exists."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])

        with patch(
            "cognee.infrastructure.databases.cache.get_cache_engine.get_cache_engine",
            return_value=mock_cache,
        ):
            from cognee.modules.retrieval.utils.session_cache import get_conversation_history

            result = await get_conversation_history(session_id="test_session")

            assert result == ""

    @pytest.mark.asyncio
    async def test_get_conversation_history_formats_history_correctly(self):
        """Test get_conversation_history formats Q&A history with correct structure."""
        user = create_mock_user()
        session_user.set(user)

        mock_history = [
            {
                "time": "2024-01-15 10:30:45",
                "question": "What is AI?",
                "context": "AI is artificial intelligence",
                "answer": "AI stands for Artificial Intelligence",
            }
        ]
        mock_cache = create_mock_cache_engine(mock_history)

        with patch(
            "cognee.infrastructure.databases.cache.get_cache_engine.get_cache_engine",
            return_value=mock_cache,
        ):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                # Enable caching
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    get_conversation_history,
                )

                result = await get_conversation_history(session_id="test_session")

                assert "Previous conversation:" in result
                assert "[2024-01-15 10:30:45]" in result
                assert "QUESTION: What is AI?" in result
                assert "CONTEXT: AI is artificial intelligence" in result
                assert "ANSWER: AI stands for Artificial Intelligence" in result

    @pytest.mark.asyncio
    async def test_save_to_session_cache_saves_correctly(self):
        """Test save_to_session_cache calls add_qa with correct parameters."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])

        with patch(
            "cognee.infrastructure.databases.cache.get_cache_engine.get_cache_engine",
            return_value=mock_cache,
        ):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                # Enable caching
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_to_session_cache,
                )

                result = await save_to_session_cache(
                    query="What is Python?",
                    context_summary="Python is a programming language",
                    answer="Python is a high-level programming language",
                    session_id="my_session",
                )

                assert result is True
                mock_cache.add_qa.assert_called_once()

                call_kwargs = mock_cache.add_qa.call_args.kwargs
                assert call_kwargs["question"] == "What is Python?"
                assert call_kwargs["context"] == "Python is a programming language"
                assert call_kwargs["answer"] == "Python is a high-level programming language"
                assert call_kwargs["session_id"] == "my_session"

    @pytest.mark.asyncio
    async def test_save_to_session_cache_uses_default_session_when_none(self):
        """Test save_to_session_cache uses 'default_session' when session_id is None."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])

        with patch(
            "cognee.infrastructure.databases.cache.get_cache_engine.get_cache_engine",
            return_value=mock_cache,
        ):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                # Enable caching
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_to_session_cache,
                )

                result = await save_to_session_cache(
                    query="Test question",
                    context_summary="Test context",
                    answer="Test answer",
                    session_id=None,
                )

                assert result is True
                call_kwargs = mock_cache.add_qa.call_args.kwargs
                assert call_kwargs["session_id"] == "default_session"
