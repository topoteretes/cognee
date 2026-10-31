"""Chunks are never sized past what the embedding model accepts (SDK-868).

The config default used to be OpenAI's 8191 for every provider, and models that
accept less (fastembed's default reads 512 tokens) truncated silently, so most of
every chunk was missing from its vector. Each engine now lowers its limit to the
model's own, from the best source its provider has; the sources are checked here
with the providers mocked, and ``effective_input_limit`` with its log levels.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.input_limit import (
    DEFAULT_EMBEDDING_INPUT_CAP,
    effective_input_limit,
    init_input_limit,
    litellm_input_limit,
    resolve_input_limit,
    sane_limit,
)
from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
    OllamaEmbeddingEngine,
)
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.HuggingFace import adapter as hf_adapter
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer


def test_sane_limit_accepts_only_a_positive_int():
    assert sane_limit(512) == 512
    assert sane_limit(0) is None
    assert sane_limit(-1) is None
    assert sane_limit(True) is None
    assert sane_limit("512") is None
    assert sane_limit(None) is None


def test_litellm_input_limit_knows_hosted_models_by_any_spelling():
    # litellm keys OpenAI models by bare name and Azure ones with the prefix.
    assert litellm_input_limit("text-embedding-3-large") == 8191
    assert litellm_input_limit("openai/text-embedding-3-large", "openai") == 8191
    assert litellm_input_limit("text-embedding-3-large", "azure") == 8191
    assert litellm_input_limit("mistral/mistral-embed", "mistral") == 8192


def test_litellm_input_limit_ignores_chat_model_entries():
    # A bare name can match a chat model; its max_tokens is an output limit.
    assert litellm_input_limit("gpt-4o", "openai") is None


def test_litellm_input_limit_is_none_for_unknown_models():
    # (No ``ollama`` case: for that provider litellm asks the local Ollama server,
    # so the answer depends on what is running on the machine.)
    assert litellm_input_limit("my-company/private-embedder") is None
    assert litellm_input_limit("private-embedder", "custom") is None
    assert litellm_input_limit(None) is None


def test_huggingface_tokenizer_knows_its_models_limit_less_special_tokens():
    # The repo declares 512 and the model adds [CLS] and [SEP] itself.
    with (
        patch.object(hf_adapter, "_load", return_value=(lambda text: text.split(), 2)),
        patch.object(hf_adapter, "_declared_input_limit", return_value=512),
    ):
        assert HuggingFaceTokenizer(model="org/model").model_input_limit == 510

    # A repo that declares no limit.
    with (
        patch.object(hf_adapter, "_load", return_value=(lambda text: text.split(), 2)),
        patch.object(hf_adapter, "_declared_input_limit", return_value=None),
    ):
        assert HuggingFaceTokenizer(model="org/model").model_input_limit is None

    # A TikToken fallback says nothing about the embedding model.
    assert TikTokenTokenizer(model=None).model_input_limit is None


def _fastembed_engine_with(embedding_model) -> FastembedEmbeddingEngine:
    engine = FastembedEmbeddingEngine.__new__(FastembedEmbeddingEngine)
    engine.embedding_model = embedding_model
    return engine


@pytest.mark.asyncio
async def test_fastembed_input_limit_reads_the_loaded_tokenizer_truncation_less_special_tokens():
    model = MagicMock()
    model.model.tokenizer.truncation = {"max_length": 512, "direction": "right"}
    model.model.tokenizer.num_special_tokens_to_add.return_value = 2
    assert await _fastembed_engine_with(model).input_limit() == 510

    model.model.tokenizer.truncation = None
    assert await _fastembed_engine_with(model).input_limit() is None

    assert await _fastembed_engine_with(object()).input_limit() is None


def _ollama_engine_with(endpoint, model) -> OllamaEmbeddingEngine:
    engine = OllamaEmbeddingEngine.__new__(OllamaEmbeddingEngine)
    engine.endpoint, engine.model = endpoint, model
    return engine


@pytest.mark.asyncio
async def test_ollama_input_limit_reads_context_length_from_api_show(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "key")
    response = MagicMock()
    response.json.return_value = {
        "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}
    }
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)) as post:
        limit = await _ollama_engine_with(
            "http://localhost:11434/api/embed", "qwen3-embedding"
        ).input_limit()

    assert limit == 40960
    assert post.call_args.args[0] == "http://localhost:11434/api/show"
    assert post.call_args.kwargs["json"] == {"model": "qwen3-embedding"}
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer key"}


@pytest.mark.asyncio
async def test_ollama_input_limit_is_none_when_the_server_cannot_answer(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    native = "http://localhost:11434/api/embed"
    refused = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with patch.object(httpx.AsyncClient, "post", refused):
        assert await _ollama_engine_with(native, "m").input_limit() is None

    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=None, response=None
    )
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
        assert await _ollama_engine_with(native, "not-pulled").input_limit() is None

    # Not an Ollama-shaped endpoint: nothing to ask.
    with patch.object(httpx.AsyncClient, "post", AsyncMock()) as post:
        assert await _ollama_engine_with("http://proxy/v1/embeddings", "m").input_limit() is None
        assert await _ollama_engine_with(None, "m").input_limit() is None
    post.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_input_limit_asks_the_provider_once_and_applies_the_result():
    engine = SimpleNamespace(
        model="m", input_limit_source="test", input_limit=AsyncMock(return_value=512)
    )
    init_input_limit(engine, configured=None)
    assert engine.max_completion_tokens == DEFAULT_EMBEDDING_INPUT_CAP  # the cap, until resolved

    assert await resolve_input_limit(engine) == 512
    assert await resolve_input_limit(engine) == 512
    assert engine.model_input_limit == 512 and engine.max_completion_tokens == 512
    engine.input_limit.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_input_limit_leaves_an_engine_without_the_method_alone():
    legacy = SimpleNamespace(max_completion_tokens=1234)  # a third-party adapter
    assert await resolve_input_limit(legacy) == 1234


def test_effective_limit_lowers_the_default_cap_to_the_model_at_info(caplog):
    with caplog.at_level(logging.INFO):
        limit = effective_input_limit(
            configured=None, model_limit=512, model="BAAI/bge-small-en-v1.5", source="test"
        )

    assert limit == 512
    records = [r for r in caplog.records if "accepts 512 tokens" in r.message]
    assert records and records[0].levelno == logging.INFO


def test_effective_limit_warns_when_the_configured_cap_exceeds_the_model(caplog):
    with caplog.at_level(logging.WARNING):
        limit = effective_input_limit(
            configured=8191, model_limit=512, model="BAAI/bge-small-en-v1.5", source="test"
        )

    assert limit == 512
    records = [r for r in caplog.records if "EMBEDDING_MAX_COMPLETION_TOKENS=8191" in r.message]
    assert records and records[0].levelno == logging.WARNING
    assert "512" in records[0].message


def test_effective_limit_keeps_a_cap_below_the_model_limit():
    assert effective_input_limit(configured=256, model_limit=512, model="m", source="t") == 256
    assert effective_input_limit(configured=None, model_limit=8191, model="m", source="t") == (
        DEFAULT_EMBEDDING_INPUT_CAP
    )


def test_effective_limit_keeps_the_cap_and_warns_when_the_model_is_unknown(caplog):
    with caplog.at_level(logging.INFO):
        assert effective_input_limit(configured=None, model_limit=None, model="m", source="t") == (
            DEFAULT_EMBEDDING_INPUT_CAP
        )
        assert effective_input_limit(configured=9000, model_limit=None, model="m", source="t") == (
            9000
        )

    unknown = [r for r in caplog.records if "Could not determine" in r.message]
    assert len(unknown) == 2 and all(r.levelno == logging.WARNING for r in unknown)


def test_default_cap_is_4096():
    assert DEFAULT_EMBEDDING_INPUT_CAP == 4096
