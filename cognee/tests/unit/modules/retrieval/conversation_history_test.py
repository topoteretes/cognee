import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.context_global_variables import session_user
import importlib


def create_mock_cache_engine(qa_history=None):
    mock_cache = AsyncMock()
    if qa_history is None:
        qa_history = []
    mock_cache.get_latest_qa = AsyncMock(return_value=qa_history)
    mock_cache.add_qa = AsyncMock(return_value=None)
    return mock_cache


def create_mock_user():
    mock_user = MagicMock()
    mock_user.id = "test-user-id-123"
    return mock_user


class TestConversationHistoryUtils:
    @pytest.mark.asyncio
    async def test_get_conversation_history_returns_empty_when_no_history(self):
        user = create_mock_user()
        session_user.set(user)
        mock_cache = create_mock_cache_engine([])

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
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

        # Import the real module to patch safely
        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
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
        """Test save_conversation_history calls add_qa with correct parameters."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_conversation_history,
                )

                result = await save_conversation_history(
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
        """Test save_conversation_history uses 'default_session' when session_id is None."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_conversation_history,
                )

                result = await save_conversation_history(
                    query="Test question",
                    context_summary="Test context",
                    answer="Test answer",
                    session_id=None,
                )

                assert result is True
                call_kwargs = mock_cache.add_qa.call_args.kwargs
                assert call_kwargs["session_id"] == "default_session"

    @pytest.mark.asyncio
    async def test_save_conversation_history_no_user_id(self):
        """Test save_conversation_history returns False when user_id is None."""
        session_user.set(None)

        with patch("cognee.modules.retrieval.utils.session_cache.CacheConfig") as MockCacheConfig:
            mock_config = MagicMock()
            mock_config.caching = True
            MockCacheConfig.return_value = mock_config

            from cognee.modules.retrieval.utils.session_cache import (
                save_conversation_history,
            )

            result = await save_conversation_history(
                query="Test question",
                context_summary="Test context",
                answer="Test answer",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_save_conversation_history_caching_disabled(self):
        """Test save_conversation_history returns False when caching is disabled."""
        user = create_mock_user()
        session_user.set(user)

        with patch("cognee.modules.retrieval.utils.session_cache.CacheConfig") as MockCacheConfig:
            mock_config = MagicMock()
            mock_config.caching = False
            MockCacheConfig.return_value = mock_config

            from cognee.modules.retrieval.utils.session_cache import (
                save_conversation_history,
            )

            result = await save_conversation_history(
                query="Test question",
                context_summary="Test context",
                answer="Test answer",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_save_conversation_history_cache_engine_none(self):
        """Test save_conversation_history returns False when cache_engine is None."""
        user = create_mock_user()
        session_user.set(user)

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=None):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_conversation_history,
                )

                result = await save_conversation_history(
                    query="Test question",
                    context_summary="Test context",
                    answer="Test answer",
                )

                assert result is False

    @pytest.mark.asyncio
    async def test_save_conversation_history_cache_connection_error(self):
        """Test save_conversation_history handles CacheConnectionError gracefully."""
        user = create_mock_user()
        session_user.set(user)

        from cognee.infrastructure.databases.exceptions import CacheConnectionError

        mock_cache = create_mock_cache_engine([])
        mock_cache.add_qa = AsyncMock(side_effect=CacheConnectionError("Connection failed"))

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_conversation_history,
                )

                result = await save_conversation_history(
                    query="Test question",
                    context_summary="Test context",
                    answer="Test answer",
                )

                assert result is False

    @pytest.mark.asyncio
    async def test_save_conversation_history_generic_exception(self):
        """Test save_conversation_history handles generic exceptions gracefully."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])
        mock_cache.add_qa = AsyncMock(side_effect=ValueError("Unexpected error"))

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    save_conversation_history,
                )

                result = await save_conversation_history(
                    query="Test question",
                    context_summary="Test context",
                    answer="Test answer",
                )

                assert result is False

    @pytest.mark.asyncio
    async def test_get_conversation_history_no_user_id(self):
        """Test get_conversation_history returns empty string when user_id is None."""
        session_user.set(None)

        with patch("cognee.modules.retrieval.utils.session_cache.CacheConfig") as MockCacheConfig:
            mock_config = MagicMock()
            mock_config.caching = True
            MockCacheConfig.return_value = mock_config

            from cognee.modules.retrieval.utils.session_cache import (
                get_conversation_history,
            )

            result = await get_conversation_history(session_id="test_session")

            assert result == ""

    @pytest.mark.asyncio
    async def test_get_conversation_history_caching_disabled(self):
        """Test get_conversation_history returns empty string when caching is disabled."""
        user = create_mock_user()
        session_user.set(user)

        with patch("cognee.modules.retrieval.utils.session_cache.CacheConfig") as MockCacheConfig:
            mock_config = MagicMock()
            mock_config.caching = False
            MockCacheConfig.return_value = mock_config

            from cognee.modules.retrieval.utils.session_cache import (
                get_conversation_history,
            )

            result = await get_conversation_history(session_id="test_session")

            assert result == ""

    @pytest.mark.asyncio
    async def test_get_conversation_history_default_session(self):
        """Test get_conversation_history uses 'default_session' when session_id is None."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    get_conversation_history,
                )

                await get_conversation_history(session_id=None)

                mock_cache.get_latest_qa.assert_called_once_with(str(user.id), "default_session")

    @pytest.mark.asyncio
    async def test_get_conversation_history_cache_engine_none(self):
        """Test get_conversation_history returns empty string when cache_engine is None."""
        user = create_mock_user()
        session_user.set(user)

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=None):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    get_conversation_history,
                )

                result = await get_conversation_history(session_id="test_session")

                assert result == ""

    @pytest.mark.asyncio
    async def test_get_conversation_history_cache_connection_error(self):
        """Test get_conversation_history handles CacheConnectionError gracefully."""
        user = create_mock_user()
        session_user.set(user)

        from cognee.infrastructure.databases.exceptions import CacheConnectionError

        mock_cache = create_mock_cache_engine([])
        mock_cache.get_latest_qa = AsyncMock(side_effect=CacheConnectionError("Connection failed"))

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    get_conversation_history,
                )

                result = await get_conversation_history(session_id="test_session")

                assert result == ""

    @pytest.mark.asyncio
    async def test_get_conversation_history_generic_exception(self):
        """Test get_conversation_history handles generic exceptions gracefully."""
        user = create_mock_user()
        session_user.set(user)

        mock_cache = create_mock_cache_engine([])
        mock_cache.get_latest_qa = AsyncMock(side_effect=ValueError("Unexpected error"))

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
                mock_config = MagicMock()
                mock_config.caching = True
                MockCacheConfig.return_value = mock_config

                from cognee.modules.retrieval.utils.session_cache import (
                    get_conversation_history,
                )

                result = await get_conversation_history(session_id="test_session")

                assert result == ""

    @pytest.mark.asyncio
    async def test_get_conversation_history_missing_keys(self):
        """Test get_conversation_history handles missing keys in history entries."""
        user = create_mock_user()
        session_user.set(user)

        mock_history = [
            {
                "time": "2024-01-15 10:30:45",
                "question": "What is AI?",
            },
            {
                "context": "AI is artificial intelligence",
                "answer": "AI stands for Artificial Intelligence",
            },
            {},
        ]
        mock_cache = create_mock_cache_engine(mock_history)

        cache_module = importlib.import_module(
            "cognee.infrastructure.databases.cache.get_cache_engine"
        )

        with patch.object(cache_module, "get_cache_engine", return_value=mock_cache):
            with patch(
                "cognee.modules.retrieval.utils.session_cache.CacheConfig"
            ) as MockCacheConfig:
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
                assert "Unknown time" in result
                assert "CONTEXT: AI is artificial intelligence" in result
                assert "ANSWER: AI stands for Artificial Intelligence" in result
