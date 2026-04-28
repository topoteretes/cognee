from unittest.mock import Mock, patch

import pytest

from cognee.infrastructure.databases.exceptions import EmbeddingException


@pytest.mark.asyncio
async def test_ollama_embedding_splits_batch_on_context_window_error():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine."
        "OllamaEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
            OllamaEmbeddingEngine,
        )

        engine = OllamaEmbeddingEngine(model="test-model", dimensions=2)

    async def fake_get_embedding(prompt):
        if prompt == "too long":
            raise ValueError("input length exceeds context length")
        # "too long" (8 chars) splits into third=8//3=2 → left="too " (s[:4]), right="o long" (s[2:])
        mapping = {
            "too ": [1.0, 1.0],
            "o long": [2.0, 2.0],
            "ok 1": [1.0, 1.0],
            "ok 2": [2.0, 2.0],
        }
        return mapping[prompt]

    with patch.object(engine, "_get_embedding", side_effect=fake_get_embedding):
        result = await engine.embed_text(["too long", "ok 1", "ok 2"])

    # "too long" → pooled([1.0,1.0], [2.0,2.0]) = [1.5,1.5]; others pass through
    assert result == [[1.5, 1.5], [1.0, 1.0], [2.0, 2.0]]


@pytest.mark.asyncio
async def test_ollama_embedding_raises_when_short_text_still_exceeds_context_window():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine."
        "OllamaEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
            OllamaEmbeddingEngine,
        )

        engine = OllamaEmbeddingEngine(model="test-model", dimensions=2)

    async def always_fail(_prompt):
        raise ValueError("maximum context length exceeded")

    with patch.object(engine, "_get_embedding", side_effect=always_fail):
        with pytest.raises(EmbeddingException, match="too short to split further"):
            await engine.embed_text(["ab"])


@pytest.mark.asyncio
async def test_fastembed_splits_batch_on_context_window_error():
    pytest.importorskip("fastembed")

    with (
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine."
            "FastembedEmbeddingEngine.get_tokenizer",
            return_value=Mock(),
        ),
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine."
            "TextEmbedding",
        ) as mock_text_embedding,
    ):
        embedding_model = Mock()
        mock_text_embedding.return_value = embedding_model

        def fake_embed(inputs, batch_size, parallel):
            if inputs == ["too long", "ok 1", "ok 2"]:
                raise RuntimeError("input length exceeds context window")
            if inputs == ["too long", "ok 1"]:
                raise RuntimeError("maximum tokens exceeded")
            if inputs == ["too long"]:
                raise RuntimeError("context length exceeded")
            # "too long" (8 chars) splits into third=8//3=2 → left="too " (s[:4]), right="o long" (s[2:])
            if inputs == ["too "]:
                return iter([[1.0, 1.0]])
            if inputs == ["o long"]:
                return iter([[2.0, 2.0]])
            if inputs == ["ok 1"]:
                return iter([[3.0, 3.0]])
            if inputs == ["ok 2"]:
                return iter([[4.0, 4.0]])
            raise AssertionError(f"Unexpected inputs: {inputs}")

        embedding_model.embed.side_effect = fake_embed

        from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
            FastembedEmbeddingEngine,
        )

        engine = FastembedEmbeddingEngine(model="test-model", dimensions=2)
        result = await engine.embed_text(["too long", "ok 1", "ok 2"])

    assert result == [[1.5, 1.5], [3.0, 3.0], [4.0, 4.0]]


@pytest.mark.asyncio
async def test_fastembed_raises_when_short_text_still_exceeds_context_window():
    pytest.importorskip("fastembed")

    with (
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine."
            "FastembedEmbeddingEngine.get_tokenizer",
            return_value=Mock(),
        ),
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine."
            "TextEmbedding",
        ) as mock_text_embedding,
    ):
        embedding_model = Mock()
        embedding_model.embed.side_effect = RuntimeError("context window exceeded")
        mock_text_embedding.return_value = embedding_model

        from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
            FastembedEmbeddingEngine,
        )

        engine = FastembedEmbeddingEngine(model="test-model", dimensions=2)

        with pytest.raises(EmbeddingException, match="too short to split further"):
            await engine.embed_text(["ab"])
