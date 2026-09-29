"""Chunks are never sized past what the embedding model accepts (SDK-868).

The config default used to be OpenAI's 8191 for every provider, and models that
accept less (fastembed's default reads 512 tokens) truncated silently, so most of
every chunk was missing from its vector. Each engine now lowers its limit to the
model's own, from the best source its provider has; the sources are checked here
with the providers mocked, and ``effective_input_limit`` with its log levels.
"""

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cognee.infrastructure.databases.vector.embeddings.input_limit import (
    DEFAULT_EMBEDDING_INPUT_CAP,
    effective_input_limit,
    fastembed_input_limit,
    huggingface_tokenizer_limit,
    litellm_input_limit,
    ollama_input_limit,
    sane_limit,
)
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
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


def test_litellm_input_limit_is_none_for_unknown_models():
    # (No ``ollama`` case: for that provider litellm asks the local Ollama server,
    # so the answer depends on what is running on the machine.)
    assert litellm_input_limit("my-company/private-embedder") is None
    assert litellm_input_limit("private-embedder", "custom") is None
    assert litellm_input_limit(None) is None


def test_huggingface_tokenizer_limit_reads_model_max_length():
    tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tokenizer.tokenizer = MagicMock(model_max_length=512)
    assert huggingface_tokenizer_limit(tokenizer) == 512

    # A TikToken fallback says nothing about the embedding model.
    assert huggingface_tokenizer_limit(TikTokenTokenizer(model=None)) is None


def test_fastembed_input_limit_reads_the_loaded_tokenizer_truncation():
    model = MagicMock()
    model.model.tokenizer.truncation = {"max_length": 512, "direction": "right"}
    assert fastembed_input_limit(model) == 512

    model.model.tokenizer.truncation = None
    assert fastembed_input_limit(model) is None

    assert fastembed_input_limit(object()) is None


def test_ollama_input_limit_reads_context_length_from_api_show():
    response = MagicMock()
    response.json.return_value = {
        "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}
    }
    with patch.object(httpx, "post", return_value=response) as post:
        limit = ollama_input_limit("http://localhost:11434/api/embed", "qwen3-embedding", "key")

    assert limit == 40960
    assert post.call_args.args[0] == "http://localhost:11434/api/show"
    assert post.call_args.kwargs["json"] == {"model": "qwen3-embedding"}
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer key"}


def test_ollama_input_limit_is_none_when_the_server_cannot_answer():
    with patch.object(httpx, "post", side_effect=httpx.ConnectError("refused")):
        assert ollama_input_limit("http://localhost:11434/api/embed", "m", None) is None

    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=None, response=None
    )
    with patch.object(httpx, "post", return_value=response):
        assert ollama_input_limit("http://localhost:11434/api/embed", "not-pulled", None) is None

    # Not an Ollama-shaped endpoint: nothing to ask.
    with patch.object(httpx, "post") as post:
        assert ollama_input_limit("http://proxy/v1/embeddings", "m", None) is None
        assert ollama_input_limit(None, "m", None) is None
    post.assert_not_called()


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


def test_effective_limit_keeps_the_cap_and_says_so_when_the_model_is_unknown(caplog):
    with caplog.at_level(logging.INFO):
        assert effective_input_limit(configured=None, model_limit=None, model="m", source="t") == (
            DEFAULT_EMBEDDING_INPUT_CAP
        )
        assert effective_input_limit(configured=9000, model_limit=None, model="m", source="t") == (
            9000
        )

    unknown = [r for r in caplog.records if "Could not determine" in r.message]
    assert len(unknown) == 2 and all(r.levelno == logging.INFO for r in unknown)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_default_cap_is_4096():
    assert DEFAULT_EMBEDDING_INPUT_CAP == 4096
