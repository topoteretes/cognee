"""Each embedding engine must delegate tokenizer selection to
``resolve_embedding_tokenizer`` (issue #3646).

The resolver's own selection logic is covered in ``test_tokenizer_resolver.py``.
These tests lock in the *wiring*: if an engine re-hardcodes a tokenizer in its
``get_tokenizer()`` (the exact fastembed ``gpt-4o`` bug this PR fixed), the
matching test fails. Engines are built with ``__new__`` and only the attributes
``get_tokenizer`` reads are set, so no network, API keys, or optional deps are
touched.
"""

from unittest.mock import patch

import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
    OllamaEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
    OpenAICompatibleEmbeddingEngine,
)

_BASE = "cognee.infrastructure.databases.vector.embeddings"
_SENTINEL = object()


def _engine(cls, **attrs):
    """Build an engine without running __init__; set only get_tokenizer's inputs."""
    engine = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(engine, name, value)
    return engine


def test_fastembed_delegates_to_resolver():
    pytest.importorskip("fastembed", reason="fastembed extra not installed")
    from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
        FastembedEmbeddingEngine,
    )

    engine = _engine(
        FastembedEmbeddingEngine,
        model="BAAI/bge-small-en-v1.5",
        max_completion_tokens=256,
    )
    with patch(
        f"{_BASE}.FastembedEmbeddingEngine.resolve_embedding_tokenizer",
        return_value=_SENTINEL,
    ) as mock:
        assert engine.get_tokenizer() is _SENTINEL
    mock.assert_called_once_with(
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        max_completion_tokens=256,
    )


def test_litellm_delegates_and_strips_vllm_prefix():
    engine = _engine(
        LiteLLMEmbeddingEngine,
        provider="openai",
        model="hosted_vllm/BAAI/bge-m3",
        max_completion_tokens=100,
    )
    with patch(
        f"{_BASE}.LiteLLMEmbeddingEngine.resolve_embedding_tokenizer",
        return_value=_SENTINEL,
    ) as mock:
        assert engine.get_tokenizer() is _SENTINEL
    mock.assert_called_once_with(
        provider="openai",
        model="BAAI/bge-m3",  # the hosted_vllm/ routing prefix is stripped first
        max_completion_tokens=100,
    )


def test_ollama_delegates_with_override():
    engine = _engine(
        OllamaEmbeddingEngine,
        model="avr/sfr-embedding-mistral:latest",
        max_completion_tokens=512,
        huggingface_tokenizer_name="Salesforce/SFR-Embedding-Mistral",
    )
    with patch(
        f"{_BASE}.OllamaEmbeddingEngine.resolve_embedding_tokenizer",
        return_value=_SENTINEL,
    ) as mock:
        assert engine.get_tokenizer() is _SENTINEL
    mock.assert_called_once_with(
        provider="ollama",
        model="avr/sfr-embedding-mistral:latest",
        max_completion_tokens=512,
        huggingface_tokenizer="Salesforce/SFR-Embedding-Mistral",
    )


def test_openai_compatible_delegates_to_resolver():
    engine = _engine(
        OpenAICompatibleEmbeddingEngine,
        model="BAAI/bge-m3",
        max_completion_tokens=128,
    )
    with patch(
        f"{_BASE}.OpenAICompatibleEmbeddingEngine.resolve_embedding_tokenizer",
        return_value=_SENTINEL,
    ) as mock:
        assert engine.get_tokenizer() is _SENTINEL
    mock.assert_called_once_with(
        provider="openai_compatible",
        model="BAAI/bge-m3",
        max_completion_tokens=128,
    )
