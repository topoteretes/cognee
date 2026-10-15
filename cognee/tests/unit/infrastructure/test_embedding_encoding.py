"""embeddings/encoding.py: which routes request base64, and how responses decode."""

import base64
import struct

import litellm
import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError
from cognee.infrastructure.databases.vector.embeddings.encoding import (
    base64_supported,
    decode_embedding,
    resolve_encoding_format,
)
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    create_embedding_engine,
)


@pytest.mark.parametrize(
    ("model", "endpoint", "expected"),
    [
        # OpenAI's own API: base64.
        ("openai/text-embedding-3-large", None, "base64"),
        ("text-embedding-3-small", None, "base64"),
        ("text-embedding-ada-002", None, "base64"),
        ("openai/text-embedding-3-large", "https://api.openai.com/v1", "base64"),
        # Everything else keeps float.
        ("openai/text-embedding-3-large", "https://proxy.invalid/v1", "float"),
        ("openai/BAAI/bge-m3", "http://localhost:8000/v1", "float"),
        ("azure/my-deployment", "https://example.openai.azure.com", "float"),
        ("openrouter/openai/text-embedding-3-small", None, "float"),
        ("gemini/text-embedding-004", None, "float"),
        ("bedrock/amazon.titan-embed-text-v2:0", None, "float"),
        ("nvidia_nim/nv-embedqa-e5-v5", None, "float"),
        ("hosted_vllm/BAAI/bge-m3", "http://localhost:8000/v1", "float"),
        ("BAAI/bge-m3", "http://localhost:8000/v1", "float"),
        ("nomic-embed-text", None, "float"),
        (None, None, "float"),
    ],
)
def test_default_format_per_route(model, endpoint, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    assert resolve_encoding_format(model, endpoint, None) == expected


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    [
        # litellm sends an endpoint-less OpenAI request to these when set.
        ("OPENAI_BASE_URL", "http://localhost:4000", "float"),
        ("OPENAI_API_BASE", "https://gateway.example.com/v1", "float"),
        ("litellm.api_base", "http://localhost:4000", "float"),
        ("OPENAI_BASE_URL", "https://api.openai.com/v1", "base64"),
    ],
)
def test_default_format_follows_litellm_base_url(setting, value, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    if setting == "litellm.api_base":
        monkeypatch.setattr(litellm, "api_base", value)
    else:
        monkeypatch.setenv(setting, value)

    assert resolve_encoding_format("openai/text-embedding-3-large", None, None) == expected
    # An explicit endpoint still wins over litellm's fallbacks.
    assert (
        resolve_encoding_format("openai/text-embedding-3-large", "https://api.openai.com/v1", None)
        == "base64"
    )


@pytest.mark.parametrize("configured", ["base64", "float"])
def test_configured_format_wins(configured):
    assert resolve_encoding_format("openai/text-embedding-3-large", None, configured) == configured
    assert resolve_encoding_format("openai/m", "http://localhost:8000/v1", configured) == configured


def test_decode_embedding_handles_both_formats():
    vector = [0.0123, -0.5, 0.25, 1e-7]
    float32_vector = [struct.unpack("<f", struct.pack("<f", v))[0] for v in vector]
    encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()

    decoded = decode_embedding(encoded, "base64")

    assert decoded == float32_vector
    assert all(isinstance(v, float) for v in decoded)
    assert decode_embedding(float32_vector, "float") is float32_vector


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        # A LiteLLM proxy in front of Bedrock answers base64 with 1-bit vectors as a list.
        ([1.0, 0.0, 1.0, 1.0], "got list"),
        ("not base64!", "does not decode"),
        # Five bytes: not a whole number of float32 values.
        (base64.b64encode(b"\x00\x00\x00\x00\x00").decode(), "does not decode"),
    ],
)
def test_decode_embedding_rejects_what_is_not_base64_float32(payload, detail):
    with pytest.raises(EmbeddingEncodingError, match=detail) as info:
        decode_embedding(payload, "base64")
    assert "EMBEDDING_ENCODING_FORMAT" in str(info.value)


@pytest.mark.parametrize(
    ("provider", "supported"),
    [
        # Engines that send an OpenAI-style request: trusted, the reply is checked.
        ("openai", True),
        ("custom", True),
        ("openai_compatible", True),
        ("azure", True),
        ("bedrock", True),
        # Engines with no wire format to request.
        ("fastembed", False),
        ("ollama", False),
    ],
)
def test_base64_supported_everywhere_but_engines_without_a_wire(provider, supported):
    assert base64_supported(provider) is supported


def _create(provider, model, encoding_format):
    return create_embedding_engine(
        provider, model, 4, None, None, None, None, 36, None, None, None, None, encoding_format
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [("fastembed", "BAAI/bge-small-en-v1.5"), ("ollama", "nomic-embed-text")],
)
def test_base64_on_an_engine_without_a_wire_is_a_configuration_error(provider, model):
    with pytest.raises(CogneeConfigurationError, match="EMBEDDING_ENCODING_FORMAT"):
        _create(provider, model, "base64")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        # A proxy or gateway route: cognee cannot see what is behind it.
        ("custom", "litellm_proxy/text-embedding-3-large"),
        ("custom", "openai/text-embedding-3-large"),
        # litellm translates the format for these; the first reply shows whether it held.
        ("bedrock", "bedrock/amazon.titan-embed-text-v2:0"),
        ("azure", "azure/my-deployment"),
    ],
)
def test_configured_base64_is_trusted_at_creation(provider, model, monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    engine = _create(provider, model, "base64")
    assert engine.encoding_format == "base64"
