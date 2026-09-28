"""A chunk can never be longer than what the fastembed model embeds: fastembed truncates
silently at the model's own limit, so the engine lowers its token limit to it."""

from unittest.mock import MagicMock, patch

import cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine as engine_module
from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
    fastembed_model_max_tokens,
)

MODEL = "BAAI/bge-small-en-v1.5"


def _engine(configured: int, truncation):
    """An engine over a fastembed model whose tokenizer reports ``truncation``."""
    with (
        patch.object(engine_module, "TextEmbedding") as text_embedding,
        patch.object(engine_module, "resolve_embedding_tokenizer") as resolve,
    ):
        text_embedding.return_value.model.tokenizer.truncation = truncation
        engine = FastembedEmbeddingEngine(
            model=MODEL, dimensions=384, max_completion_tokens=configured
        )
    return engine, resolve


def test_a_configured_limit_above_the_models_is_lowered_to_the_models():
    engine, resolve = _engine(8191, {"max_length": 512, "direction": "right"})

    assert engine.max_completion_tokens == 512
    resolve.assert_called_once_with(provider="fastembed", model=MODEL, max_completion_tokens=512)


def test_a_configured_limit_below_the_models_is_kept():
    engine, resolve = _engine(256, {"max_length": 512, "direction": "right"})

    assert engine.max_completion_tokens == 256
    resolve.assert_called_once_with(provider="fastembed", model=MODEL, max_completion_tokens=256)


def test_a_model_without_a_known_limit_keeps_the_configured_one():
    engine, _ = _engine(8191, None)

    assert engine.max_completion_tokens == 8191


def test_the_model_limit_is_read_from_the_loaded_tokenizer():
    model = MagicMock()
    model.model.tokenizer.truncation = {"max_length": 512}
    assert fastembed_model_max_tokens(model) == 512

    model.model.tokenizer.truncation = None
    assert fastembed_model_max_tokens(model) is None

    assert fastembed_model_max_tokens(object()) is None
