import pytest
from unittest.mock import Mock, patch
from cognee.infrastructure.databases.exceptions import EmbeddingException


@pytest.mark.asyncio
async def test_litellm_embedding_exception_no_retry():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine.LiteLLMEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
            LiteLLMEmbeddingEngine,
        )

        engine = LiteLLMEmbeddingEngine(model="test-model", dimensions=2)

    call_count = 0

    def mock_sanitize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise EmbeddingException("deterministic error")

    with patch(
        "cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine.sanitize_embedding_text_inputs",
        side_effect=mock_sanitize,
    ):
        with pytest.raises(EmbeddingException):
            await engine.embed_text(["test"])

    # Tenacity should not retry since EmbeddingException is in retry_if_not_exception_type
    assert call_count == 1


@pytest.mark.asyncio
async def test_fastembed_embedding_exception_no_retry():
    pytest.importorskip("fastembed")

    with (
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine.FastembedEmbeddingEngine.get_tokenizer",
            return_value=Mock(),
        ),
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine.TextEmbedding",
            return_value=Mock(),
        ),
    ):
        from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
            FastembedEmbeddingEngine,
        )

        engine = FastembedEmbeddingEngine(model="test-model", dimensions=2)

    call_count = 0

    def mock_sanitize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise EmbeddingException("deterministic error")

    with patch(
        "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine.sanitize_embedding_text_inputs",
        side_effect=mock_sanitize,
    ):
        with pytest.raises(EmbeddingException):
            await engine.embed_text(["test"])

    assert call_count == 1


@pytest.mark.asyncio
async def test_openai_compatible_embedding_exception_no_retry():
    with (
        patch(
            "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine.OpenAICompatibleEmbeddingEngine.get_tokenizer",
            return_value=Mock(),
        ),
        patch(
            "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine.AsyncOpenAI",
            return_value=Mock(),
        ),
    ):
        from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
            OpenAICompatibleEmbeddingEngine,
            EmbeddingException as OpenAICompatibleEmbeddingException,
        )

        engine = OpenAICompatibleEmbeddingEngine(model="test-model", dimensions=2)

    call_count = 0

    def mock_sanitize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise OpenAICompatibleEmbeddingException("deterministic error")

    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine.sanitize_embedding_text_inputs",
        side_effect=mock_sanitize,
    ):
        with pytest.raises(OpenAICompatibleEmbeddingException):
            await engine.embed_text(["test"])

    assert call_count == 1
