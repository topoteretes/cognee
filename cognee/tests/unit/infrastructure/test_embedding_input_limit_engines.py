"""Each embedding engine embeds by ``min(cap, model limit)``, and never cuts input silently.

Per engine: where ``input_limit()`` gets the model's limit, that the default cap
(4096) is lowered to it, that an explicit ``EMBEDDING_MAX_COMPLETION_TOKENS`` above
it is lowered with a warning, and that an unknown model keeps the cap. The limit
is resolved once, asynchronously, by ``resolve_input_limit`` (the constructor only
records the cap). Then the two silent-truncation providers: fastembed embeds an
over-length text in parts instead of letting the model cut it, and Ollama is asked
to reject over-length input (``truncate: false``) so the existing split path runs.
Finally the chunk size a pipeline runs with: an explicit ``chunk_size`` above the
limit is lowered with a warning. Providers are mocked; nothing here loads a model
or a server.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest
from typing_extensions import Self

import cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine as fastembed_module
from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.input_limit import (
    DEFAULT_EMBEDDING_INPUT_CAP,
    resolve_input_limit,
)
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
    OllamaEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
    OpenAICompatibleEmbeddingEngine,
)
from cognee.infrastructure.llm import utils as llm_utils
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.HuggingFace import adapter as hf_adapter
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer

BGE = "BAAI/bge-small-en-v1.5"


def _hf_tokenizer(model_max_length: int) -> HuggingFaceTokenizer:
    """A resolved HuggingFace tokenizer whose repo declares ``model_max_length``
    and whose model adds no special tokens (so the limit is used as is)."""
    with (
        patch.object(hf_adapter, "_load", return_value=(lambda text: text.split(), 0)),
        patch.object(hf_adapter, "_declared_input_limit", return_value=model_max_length),
    ):
        return HuggingFaceTokenizer(model="org/model")


# ---------------------------------------------------------------------------
# fastembed
# ---------------------------------------------------------------------------


def _fastembed_engine(monkeypatch, configured, truncation):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    with (
        patch.object(fastembed_module, "TextEmbedding") as text_embedding,
        patch.object(fastembed_module, "resolve_embedding_tokenizer", return_value=MagicMock()),
    ):
        tokenizer = text_embedding.return_value.model.tokenizer
        tokenizer.truncation = truncation
        tokenizer.num_special_tokens_to_add.return_value = 2  # [CLS] and [SEP]
        return FastembedEmbeddingEngine(model=BGE, dimensions=4, max_completion_tokens=configured)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 510), (8191, 510), (256, 256)],
    ids=["default_cap_lowered", "explicit_cap_lowered", "cap_below_model_kept"],
)
async def test_fastembed_limit_comes_from_the_loaded_model(monkeypatch, configured, expected):
    # 512 is what the model reads, 2 of them its own [CLS]/[SEP].
    engine = _fastembed_engine(monkeypatch, configured, {"max_length": 512})

    assert await resolve_input_limit(engine) == expected
    assert engine.model_input_limit == 510


@pytest.mark.asyncio
async def test_fastembed_unknown_model_limit_keeps_the_cap(monkeypatch):
    engine = _fastembed_engine(monkeypatch, None, None)

    assert await resolve_input_limit(engine) == DEFAULT_EMBEDDING_INPUT_CAP
    assert engine.model_input_limit is None


@pytest.mark.asyncio
async def test_fastembed_embeds_over_length_text_in_parts_instead_of_cutting_it(monkeypatch):
    engine = _fastembed_engine(monkeypatch, None, {"max_length": 512})
    long_text = "x" * 3000
    embedded = []

    def encode_batch(texts):
        # The model's tokenizer reports what it would cut off as ``overflowing``;
        # here anything longer than 2500 characters, so one split (2000 + 2000,
        # overlapping) is enough.
        return [SimpleNamespace(overflowing=[1] if len(text) > 2500 else []) for text in texts]

    def embed(texts, **kwargs):
        embedded.extend(texts)
        return [np.array([len(text), 1.0, 1.0, 1.0]) for text in texts]

    engine.embedding_model.model.tokenizer.encode_batch = encode_batch
    engine.embedding_model.embed = embed

    (vector,) = await engine.embed_text([long_text])

    assert long_text not in embedded, "the over-length text must never reach the model whole"
    assert len(embedded) == 2 and all(len(part) <= 2000 for part in embedded)
    assert len(vector) == 4
    assert vector[0] == pytest.approx(np.mean([len(part) for part in embedded]))  # pooled


@pytest.mark.asyncio
async def test_fastembed_batch_with_one_over_length_item_returns_one_vector_per_input(monkeypatch):
    engine = _fastembed_engine(monkeypatch, None, {"max_length": 512})
    engine.embedding_model.model.tokenizer.encode_batch = lambda texts: [
        SimpleNamespace(overflowing=[1] if len(text) > 2500 else []) for text in texts
    ]
    engine.embedding_model.embed = lambda texts, **kwargs: [np.ones(4) for _ in texts]

    vectors = await engine.embed_text(["short", "y" * 3000, "short too"])

    assert len(vectors) == 3 and all(len(vector) == 4 for vector in vectors)


@pytest.mark.asyncio
async def test_fastembed_length_check_failure_does_not_break_embedding(monkeypatch):
    engine = _fastembed_engine(monkeypatch, None, {"max_length": 512})
    engine.embedding_model.model.tokenizer.encode_batch = MagicMock(side_effect=RuntimeError("no"))
    engine.embedding_model.embed = lambda texts, **kwargs: [np.ones(4) for _ in texts]

    assert len(await engine.embed_text(["hello"])) == 1


# ---------------------------------------------------------------------------
# litellm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, DEFAULT_EMBEDDING_INPUT_CAP), (8191, 8191), (9000, 8191)],
    ids=["default_cap_below_model", "explicit_at_model", "explicit_above_model_lowered"],
)
async def test_litellm_limit_for_a_hosted_model_comes_from_litellms_table(configured, expected):
    engine = LiteLLMEmbeddingEngine(
        model="openai/text-embedding-3-large",
        provider="openai",
        dimensions=4,
        max_completion_tokens=configured,
    )

    assert await resolve_input_limit(engine) == expected
    assert engine.model_input_limit == 8191


@pytest.mark.asyncio
async def test_litellm_explicit_cap_above_the_model_warns(caplog):
    engine = LiteLLMEmbeddingEngine(
        model="openai/text-embedding-3-large",
        provider="openai",
        dimensions=4,
        max_completion_tokens=9000,
    )
    with caplog.at_level(logging.WARNING):
        await resolve_input_limit(engine)

    assert any(
        "EMBEDDING_MAX_COMPLETION_TOKENS=9000" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_litellm_limit_for_a_self_hosted_repo_comes_from_its_tokenizer():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine."
        "resolve_embedding_tokenizer",
        return_value=_hf_tokenizer(512),
    ):
        engine = LiteLLMEmbeddingEngine(model=f"hosted_vllm/{BGE}", provider="custom", dimensions=4)

    assert await resolve_input_limit(engine) == 512
    assert engine.model_input_limit == 512


@pytest.mark.asyncio
async def test_litellm_unknown_model_keeps_the_cap():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine."
        "resolve_embedding_tokenizer",
        return_value=TikTokenTokenizer(model=None),
    ):
        engine = LiteLLMEmbeddingEngine(
            model="my-private-embedder", provider="custom", dimensions=4
        )

    assert await resolve_input_limit(engine) == DEFAULT_EMBEDDING_INPUT_CAP
    assert engine.model_input_limit is None


# ---------------------------------------------------------------------------
# openai-compatible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_compatible_limit_comes_from_the_served_repos_tokenizer():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine."
        "resolve_embedding_tokenizer",
        return_value=_hf_tokenizer(512),
    ):
        engine = OpenAICompatibleEmbeddingEngine(model=BGE, dimensions=4)

    assert await resolve_input_limit(engine) == 512
    assert engine.model_input_limit == 512


@pytest.mark.asyncio
async def test_openai_compatible_hosted_model_name_uses_litellms_table():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine."
        "resolve_embedding_tokenizer",
        return_value=TikTokenTokenizer(model=None),
    ):
        engine = OpenAICompatibleEmbeddingEngine(
            model="text-embedding-3-small", dimensions=4, max_completion_tokens=9000
        )

    assert await resolve_input_limit(engine) == 8191
    assert engine.model_input_limit == 8191


@pytest.mark.asyncio
async def test_openai_compatible_unknown_model_keeps_the_cap():
    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine."
        "resolve_embedding_tokenizer",
        return_value=TikTokenTokenizer(model=None),
    ):
        engine = OpenAICompatibleEmbeddingEngine(model="default", dimensions=4)

    assert await resolve_input_limit(engine) == DEFAULT_EMBEDDING_INPUT_CAP
    assert engine.model_input_limit is None


# ---------------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------------


def _ollama_engine(monkeypatch, configured, endpoint="http://localhost:11434/api/embed"):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    with patch.object(OllamaEmbeddingEngine, "get_tokenizer", return_value=MagicMock()):
        return OllamaEmbeddingEngine(
            model="nomic-embed-text",
            dimensions=4,
            max_completion_tokens=configured,
            endpoint=endpoint,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "model_limit", "expected"),
    [
        (None, 512, 512),
        (8191, 512, 512),
        (None, 8192, DEFAULT_EMBEDDING_INPUT_CAP),
        (None, None, 4096),
    ],
    ids=["default_cap_lowered", "explicit_cap_lowered", "cap_below_model", "unknown_keeps_cap"],
)
async def test_ollama_limit_comes_from_api_show(monkeypatch, configured, model_limit, expected):
    lookup = AsyncMock(return_value=model_limit)
    with patch.object(OllamaEmbeddingEngine, "input_limit", lookup):
        engine = _ollama_engine(monkeypatch, configured)
        assert lookup.await_count == 0, "the constructor must not ask the server"

        assert await resolve_input_limit(engine) == expected

    assert engine.model_input_limit == model_limit
    lookup.assert_awaited_once()


class _FakeAiohttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


@pytest.mark.asyncio
async def test_ollama_asks_the_server_to_reject_over_length_input(monkeypatch):
    engine = _ollama_engine(monkeypatch, None)
    payloads = []

    def _fake_post(self, url, *, json, **kwargs):
        payloads.append(json)
        return _FakeAiohttpResponse({"embeddings": [[0.1, 0.2, 0.3, 0.4]]})

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)

    await engine.embed_text(["hello"])

    assert payloads[0]["truncate"] is False


@pytest.mark.asyncio
async def test_ollama_does_not_send_truncate_to_an_openai_shaped_endpoint(monkeypatch):
    """A LiteLLM proxy or /v1 endpoint rejects unknown fields, so truncate stays Ollama-only."""
    engine = _ollama_engine(monkeypatch, None, endpoint="http://proxy:4000/v1/embeddings")
    payloads = []

    def _fake_post(self, url, *, json, **kwargs):
        payloads.append(json)
        return _FakeAiohttpResponse({"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]})

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)

    await engine.embed_text(["hello"])

    assert "truncate" not in payloads[0]


@pytest.mark.asyncio
async def test_ollama_over_length_rejection_is_embedded_in_parts_without_an_error_log(
    monkeypatch, caplog
):
    engine = _ollama_engine(monkeypatch, None)
    rejected = {"count": 0}

    def _fake_post(self, url, *, json, **kwargs):
        if len(json["input"]) > 2500:
            rejected["count"] += 1
            return _FakeAiohttpResponse({"error": "the input length exceeds the context length"})
        return _FakeAiohttpResponse({"embeddings": [[1.0, 1.0, 1.0, 1.0]]})

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)

    with caplog.at_level(logging.INFO):
        (vector,) = await engine.embed_text(["z" * 3000])

    assert len(vector) == 4 and rejected["count"] == 1
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# the chunk size a pipeline runs with
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_chunk_size_warnings(monkeypatch):
    monkeypatch.setattr(llm_utils, "_chunk_size_warnings_issued", set())


def _vector_engine_with(limit: int, model_limit: int | None, model: str = BGE):
    """A vector engine whose embedding engine already has its limit resolved."""
    engine = SimpleNamespace(
        max_completion_tokens=limit,
        model_input_limit=model_limit,
        model=model,
        input_limit_resolved=True,
    )
    return AsyncMock(return_value=SimpleNamespace(embedding_engine=engine))


@pytest.mark.asyncio
async def test_resolve_chunk_size_uses_the_automatic_size_when_none_is_given():
    with patch.object(llm_utils, "get_max_chunk_tokens", AsyncMock(return_value=512)):
        assert await llm_utils.resolve_chunk_size(None) == 512
        assert await llm_utils.resolve_chunk_size(0) == 512


@pytest.mark.asyncio
async def test_resolve_chunk_size_rejects_a_negative_size():
    with pytest.raises(ValueError, match="chunk_size"):
        await llm_utils.resolve_chunk_size(-1)


@pytest.mark.asyncio
async def test_resolve_chunk_size_keeps_an_explicit_size_within_the_limit():
    with patch(
        "cognee.infrastructure.databases.vector.get_vector_engine_async",
        _vector_engine_with(limit=512, model_limit=512),
    ):
        assert await llm_utils.resolve_chunk_size(300) == 300
        assert await llm_utils.resolve_chunk_size(512) == 512


@pytest.mark.asyncio
async def test_resolve_chunk_size_lowers_a_size_the_model_cannot_embed_with_a_warning(caplog):
    with (
        patch(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _vector_engine_with(limit=512, model_limit=512),
        ),
        caplog.at_level(logging.WARNING),
    ):
        assert await llm_utils.resolve_chunk_size(8191) == 512

    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "chunk_size=8191" in warning.message
    assert BGE in warning.message and "512" in warning.message
    assert "dropped from the embedding" in warning.message


@pytest.mark.asyncio
async def test_resolve_chunk_size_lowers_a_size_above_the_cap_and_names_the_setting(caplog):
    # The model accepts 8191, but the cap (default 4096) is what the engine embeds by.
    with (
        patch(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _vector_engine_with(
                limit=4096, model_limit=8191, model="openai/text-embedding-3-large"
            ),
        ),
        caplog.at_level(logging.WARNING),
    ):
        assert await llm_utils.resolve_chunk_size(6000) == 4096

    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "EMBEDDING_MAX_COMPLETION_TOKENS" in warning.message
    assert "8191" in warning.message  # tells the user how far the cap can be raised


@pytest.mark.asyncio
async def test_automatic_chunk_size_is_the_embedding_limit_or_half_the_llm_context():
    llm = SimpleNamespace(
        llm_model="unknown-model", llm_provider="openai", llm_max_completion_tokens=16384
    )
    with (
        patch(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _vector_engine_with(limit=512, model_limit=512),
        ),
        patch("cognee.infrastructure.llm.config.get_llm_context_config", return_value=llm),
    ):
        assert await llm_utils.get_max_chunk_tokens() == 512

    llm.llm_max_completion_tokens = 4000
    with (
        patch(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _vector_engine_with(limit=4096, model_limit=8191),
        ),
        patch("cognee.infrastructure.llm.config.get_llm_context_config", return_value=llm),
    ):
        assert await llm_utils.get_max_chunk_tokens() == 2000
