"""COGNEE_EMBEDDING_WORKER_* settings and EMBEDDING_PROVIDER=embedding_worker wiring."""

from __future__ import annotations

import sys

import pytest

import cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine as engine_module
from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
    EmbeddingWorkerConfigError,
    get_embedding_worker_settings,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (
    EmbeddingWorkerEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    create_embedding_engine,
)

PREFIX = "COGNEE_EMBEDDING_WORKER_"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "TRANSPORT",
        "ENDPOINT",
        "AMQP_QUEUE",
        "EXPECTED_MODEL",
        "MAX_CONCURRENT_REQUESTS",
        "CONNECT_TIMEOUT_SECS",
        "REQUEST_TIMEOUT_SECS",
    ):
        monkeypatch.delenv(PREFIX + name, raising=False)
    monkeypatch.setattr(engine_module, "resolve_embedding_tokenizer", lambda **_: object())


def test_defaults_match_the_rust_sdk():
    settings = get_embedding_worker_settings()
    assert settings.transport == "amqp"
    assert settings.endpoint is None
    assert settings.amqp_queue == "embedding_worker.embed"
    assert settings.expected_model is None
    assert settings.max_concurrent_requests == 16
    assert settings.connect_timeout_secs == 5
    assert settings.request_timeout_secs == 300


def test_rabbitmq_is_an_alias_of_amqp(monkeypatch):
    monkeypatch.setenv(PREFIX + "TRANSPORT", " RabbitMQ ")
    assert get_embedding_worker_settings().transport == "amqp"


@pytest.mark.parametrize("transport", ["http", "grpc"])
def test_http_and_grpc_are_not_supported_yet(monkeypatch, transport):
    monkeypatch.setenv(PREFIX + "TRANSPORT", transport)
    monkeypatch.setenv(PREFIX + "ENDPOINT", "http://worker:8080")
    with pytest.raises(EmbeddingWorkerConfigError, match="not supported yet"):
        get_embedding_worker_settings().require_amqp_endpoint()


def test_an_unknown_transport_is_a_config_error(monkeypatch):
    monkeypatch.setenv(PREFIX + "TRANSPORT", "kafka")
    with pytest.raises(EmbeddingWorkerConfigError, match=PREFIX + "TRANSPORT"):
        get_embedding_worker_settings()


def test_malformed_numbers_are_a_config_error(monkeypatch):
    monkeypatch.setenv(PREFIX + "MAX_CONCURRENT_REQUESTS", "0")
    with pytest.raises(EmbeddingWorkerConfigError, match=PREFIX + "MAX_CONCURRENT_REQUESTS"):
        get_embedding_worker_settings()


def test_the_endpoint_is_required(monkeypatch):
    monkeypatch.setenv(PREFIX + "ENDPOINT", "  ")
    with pytest.raises(EmbeddingWorkerConfigError, match="broker URL"):
        get_embedding_worker_settings().require_amqp_endpoint()


def test_the_endpoint_must_be_amqp_and_is_masked(monkeypatch):
    monkeypatch.setenv(PREFIX + "ENDPOINT", "http://user:s3cret@broker:5672")
    with pytest.raises(EmbeddingWorkerConfigError) as raised:
        get_embedding_worker_settings().require_amqp_endpoint()
    assert "s3cret" not in str(raised.value)
    assert "amqp(s)://" in raised.value.message


def build(provider="embedding_worker", **overrides):
    arguments = {
        "embedding_provider": provider,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimensions": 384,
        "embedding_max_completion_tokens": 512,
        "embedding_endpoint": None,
        "embedding_api_key": None,
        "embedding_api_version": None,
        "embedding_batch_size": 16,
        "huggingface_tokenizer": None,
        "llm_api_key": None,
        "llm_provider": None,
        "embedding_input_type": None,
        "embedding_encoding_format": None,
    }
    arguments.update(overrides)
    # Uncached, so each test sees its own environment.
    return create_embedding_engine.__wrapped__(**arguments)


def test_the_provider_selects_the_engine(monkeypatch):
    monkeypatch.setenv(PREFIX + "ENDPOINT", "amqp://guest:guest@broker:5672/%2f")
    engine = build(embedding_input_type="query")
    assert isinstance(engine, EmbeddingWorkerEmbeddingEngine)
    assert engine.get_vector_size() == 384
    assert engine.get_batch_size() == 16
    assert engine.input_cap == 512
    assert engine.input_type == "query"
    assert engine.model == "BAAI/bge-small-en-v1.5"


def test_the_engine_fails_fast_without_an_endpoint():
    with pytest.raises(EmbeddingWorkerConfigError, match=PREFIX + "ENDPOINT"):
        build()


def test_without_embedding_dimensions_the_engine_waits_for_the_worker(monkeypatch):
    monkeypatch.setenv(PREFIX + "ENDPOINT", "amqp://guest:guest@broker:5672/%2f")
    engine = build(embedding_dimensions=None)
    assert engine.dimensions is None and engine.configured_dimensions is None


def test_the_config_leaves_the_size_unset_for_the_worker(monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig

    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    worker = EmbeddingConfig(
        embedding_provider="embedding_worker", embedding_model="BAAI/bge-small-en-v1.5"
    )
    assert worker.embedding_dimensions is None  # not the 3072 fallback
    explicit = EmbeddingConfig(embedding_provider="embedding_worker", embedding_dimensions=384)
    assert explicit.embedding_dimensions == 384
    other = EmbeddingConfig(embedding_provider="openai_compatible", embedding_model="unknown/x")
    assert other.embedding_dimensions is not None


def test_a_missing_aio_pika_fails_when_the_engine_is_built(monkeypatch):
    monkeypatch.setenv(PREFIX + "ENDPOINT", "amqp://guest:guest@broker:5672/%2f")
    amqp_module = "cognee.infrastructure.databases.vector.embeddings.embedding_worker.amqp"
    monkeypatch.delitem(sys.modules, amqp_module, raising=False)
    # `from . import amqp` would otherwise find the already imported submodule here.
    monkeypatch.delattr(sys.modules[amqp_module.rsplit(".", 1)[0]], "amqp", raising=False)
    monkeypatch.delitem(sys.modules, "cognee.infrastructure.remote_worker.amqp_rpc", raising=False)
    monkeypatch.setitem(sys.modules, "aio_pika", None)  # import aio_pika -> ImportError
    with pytest.raises(EmbeddingWorkerConfigError, match="aio-pika") as raised:
        build()
    assert "remote-worker-amqp" in raised.value.remediation
