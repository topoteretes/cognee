import pytest
import asyncio
import numpy as np
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.infrastructure.databases.vector.embeddings.context_window_handler import (
    handle_context_window_exceeded,
)
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
    OllamaEmbeddingEngine,
)
import litellm


class TestContextWindowHandler:
    """Test the shared context window handler."""

    @pytest.mark.asyncio
    async def test_handle_context_window_multiple_texts(self):
        """Test handling multiple texts that exceed context window."""
        mock_embed = AsyncMock(
            side_effect=[
                Exception("Context window exceeded"),
                [[0.1, 0.2], [0.3, 0.4]],  # Left half
                [[0.5, 0.6], [0.7, 0.8]],  # Right half
            ]
        )

        result = await handle_context_window_exceeded(
            mock_embed, ["text1", "text2", "text3", "text4"]
        )
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        assert mock_embed.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_context_window_single_text(self):
        """Test handling single text that exceeds context window."""
        mock_embed = AsyncMock(
            side_effect=[
                Exception("Context window exceeded"),
                [[0.1, 0.2]],  # Left part
                [[0.3, 0.4]],  # Right part
            ]
        )

        long_text = "a" * 10000
        result = await handle_context_window_exceeded(mock_embed, [long_text])
        assert len(result) == 1
        # Use pytest.approx for floating point comparison
        assert result[0] == pytest.approx([0.2, 0.3])  # (0.1+0.3)/2, (0.2+0.4)/2

    @pytest.mark.asyncio
    async def test_handle_context_window_non_context_error(self):
        """Test that non-context errors are re-raised."""
        mock_embed = AsyncMock(side_effect=Exception("Generic error"))

        with pytest.raises(Exception, match="Generic error"):
            await handle_context_window_exceeded(mock_embed, ["test"])

    @pytest.mark.asyncio
    async def test_handle_context_window_max_retries(self):
        """Test max retries limit."""
        mock_embed = AsyncMock(side_effect=Exception("Context window exceeded"))

        with pytest.raises(Exception, match="Context window exceeded"):
            await handle_context_window_exceeded(mock_embed, ["test"], max_retries=0)


class TestLiteLLMEmbeddingEngine:
    """Test LiteLLM embedding engine context window handling."""

    @pytest.mark.asyncio
    async def test_litellm_context_window_handling(self):
        """Test that LiteLLM uses context window handler."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            engine = LiteLLMEmbeddingEngine(dimensions=2)

            # Mock litellm.aembedding to raise context window error
            with patch(
                "litellm.aembedding",
                new_callable=AsyncMock,
                side_effect=litellm.exceptions.ContextWindowExceededError(
                    model="text-embedding-3-large",
                    llm_provider="openai",
                    message="Context window exceeded",
                ),
            ):
                with patch(
                    "cognee.infrastructure.databases.vector.embeddings.context_window_handler.handle_context_window_exceeded",
                    new_callable=AsyncMock,
                    return_value=[[0.1, 0.2]],
                ) as mock_handler:
                    result = await engine.embed_text(["test"])
                    assert result == [[0.1, 0.2]]
                    mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_litellm_raw_embedding(self):
        """Test raw embedding without context window issues."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            engine = LiteLLMEmbeddingEngine(dimensions=2)

            # Mock successful embedding
            mock_response = SimpleNamespace(data=[{"embedding": [0.1, 0.2]}])

            with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
                result = await engine.embed_text(["test"])
                assert result == [[0.1, 0.2]]


class TestFastembedEmbeddingEngine:
    """Test Fastembed embedding engine context window handling."""

    @pytest.mark.asyncio
    async def test_fastembed_context_window_handling(self):
        """Test that Fastembed uses context window handler."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            # Mock TextEmbedding to avoid model loading
            with patch(
                "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine.TextEmbedding"
            ) as mock_text_embedding:
                mock_text_embedding.return_value = MagicMock()
                engine = FastembedEmbeddingEngine(
                    model="sentence-transformers/all-MiniLM-L6-v2", dimensions=384
                )

                # Mock the embed method to raise context window error
                with patch.object(
                    engine.embedding_model,
                    "embed",
                    side_effect=Exception("Context window exceeded"),
                ):
                    with patch(
                        "cognee.infrastructure.databases.vector.embeddings.context_window_handler.handle_context_window_exceeded",
                        new_callable=AsyncMock,
                        return_value=[[0.1, 0.2]],
                    ) as mock_handler:
                        result = await engine.embed_text(["test"])
                        assert result == [[0.1, 0.2]]
                        mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fastembed_context_error_detection(self):
        """Test that Fastembed detects context window errors."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            with patch(
                "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine.TextEmbedding"
            ) as mock_text_embedding:
                mock_text_embedding.return_value = MagicMock()
                engine = FastembedEmbeddingEngine(
                    model="sentence-transformers/all-MiniLM-L6-v2", dimensions=384
                )

                with patch.object(
                    engine.embedding_model, "embed", side_effect=Exception("token limit exceeded")
                ):
                    with patch(
                        "cognee.infrastructure.databases.vector.embeddings.context_window_handler.handle_context_window_exceeded",
                        new_callable=AsyncMock,
                        return_value=[[0.1, 0.2]],
                    ) as mock_handler:
                        result = await engine.embed_text(["test"])
                        assert result == [[0.1, 0.2]]
                        mock_handler.assert_awaited_once()


class TestOllamaEmbeddingEngine:
    """Test Ollama embedding engine context window handling."""

    @pytest.mark.asyncio
    async def test_ollama_context_window_handling(self):
        """Test that Ollama uses context window handler."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            with patch(
                "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine.HuggingFaceTokenizer"
            ) as mock_tokenizer:
                mock_tokenizer.return_value = MagicMock()
                engine = OllamaEmbeddingEngine(dimensions=2)

                # Mock _get_embedding to raise context window error
                with patch.object(
                    engine,
                    "_get_embedding",
                    new_callable=AsyncMock,
                    side_effect=ValueError("context window exceeded for embedding model"),
                ):
                    with patch(
                        "cognee.infrastructure.databases.vector.embeddings.context_window_handler.handle_context_window_exceeded",
                        new_callable=AsyncMock,
                        return_value=[[0.1, 0.2]],
                    ) as mock_handler:
                        result = await engine.embed_text(["test"])
                        assert result == [[0.1, 0.2]]
                        mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ollama_context_error_detection(self):
        """Test that Ollama detects context window errors."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            with patch(
                "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine.HuggingFaceTokenizer"
            ) as mock_tokenizer:
                mock_tokenizer.return_value = MagicMock()
                engine = OllamaEmbeddingEngine(dimensions=2)

                with patch.object(
                    engine,
                    "_get_embedding",
                    new_callable=AsyncMock,
                    side_effect=ValueError("context length exceeded"),
                ):
                    with patch(
                        "cognee.infrastructure.databases.vector.embeddings.context_window_handler.handle_context_window_exceeded",
                        new_callable=AsyncMock,
                        return_value=[[0.1, 0.2]],
                    ) as mock_handler:
                        result = await engine.embed_text(["test"])
                        assert result == [[0.1, 0.2]]
                        mock_handler.assert_awaited_once()


class TestIntegration:
    """Integration tests for all embedding engines."""

    @pytest.mark.asyncio
    async def test_all_engines_consistent_handling(self):
        """Test that all engines handle context windows consistently."""
        with patch.dict(os.environ, {"MOCK_EMBEDDING": "false"}):
            engines = [
                (
                    LiteLLMEmbeddingEngine(dimensions=2),
                    "litellm.aembedding",
                    litellm.exceptions.ContextWindowExceededError(
                        model="text-embedding-3-large",
                        llm_provider="openai",
                        message="Context window exceeded",
                    ),
                ),
            ]

            # Only test LiteLLM for now to avoid dependency issues
            for engine, mock_target, exception in engines:
                # Mock the actual embedding call to raise context window error
                with patch(mock_target, new_callable=AsyncMock, side_effect=exception):
                    with patch(
                        "cognee.infrastructure.databases.vector.embeddings.context_window_handler.handle_context_window_exceeded",
                        new_callable=AsyncMock,
                        return_value=[[0.1, 0.2]],
                    ) as mock_handler:
                        result = await engine.embed_text(["test"])
                        assert result == [[0.1, 0.2]]
                        mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embedding_pooling_accuracy(self):
        """Test that embedding pooling produces correct averages."""
        # Test the pooling logic directly
        left_vec = [0.2, 0.4, 0.6]
        right_vec = [0.8, 0.2, 0.4]
        expected = [(0.2 + 0.8) / 2, (0.4 + 0.2) / 2, (0.6 + 0.4) / 2]

        mock_embed = AsyncMock(
            side_effect=[Exception("Context window exceeded"), [left_vec], [right_vec]]
        )

        result = await handle_context_window_exceeded(mock_embed, ["very long text"])
        assert result[0] == pytest.approx(expected)
