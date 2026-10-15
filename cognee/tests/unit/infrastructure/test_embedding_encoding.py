"""embeddings/encoding.py: which routes request base64, and how responses decode."""

import base64
import struct

import pytest

from cognee.exceptions import CogneeConfigurationError
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
def test_default_format_per_route(model, endpoint, expected):
    assert resolve_encoding_format(model, endpoint, None) == expected


@pytest.mark.parametrize("configured", ["base64", "float"])
def test_configured_format_wins(configured):
    assert resolve_encoding_format("openai/text-embedding-3-large", None, configured) == configured
    assert resolve_encoding_format("openai/m", "http://localhost:8000/v1", configured) == configured


def test_decode_embedding_handles_both_formats():
    vector = [0.0123, -0.5, 0.25, 1e-7]
    float32_vector = [struct.unpack("<f", struct.pack("<f", v))[0] for v in vector]
    encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()

    decoded = decode_embedding(encoded)

    assert decoded == float32_vector
    assert all(isinstance(v, float) for v in decoded)
    assert decode_embedding(float32_vector) is float32_vector


@pytest.mark.parametrize(
    ("provider", "model", "supported"),
    [
        # Tested: OpenAI's embeddings request, sent unchanged.
        ("openai", "openai/text-embedding-3-large", True),
        ("openai", "text-embedding-3-small", True),
        ("custom", "openai/BAAI/bge-m3", True),
        ("openai_compatible", "nomic-embed-text", True),
        # Untested, or litellm translates encoding_format for the provider.
        ("azure", "azure/my-deployment", False),
        ("bedrock", "bedrock/amazon.titan-embed-text-v2:0", False),
        ("cohere", "cohere/embed-english-v3.0", False),
        ("gemini", "gemini/text-embedding-004", False),
        ("mistral", "mistral/mistral-embed", False),
        ("custom", "openrouter/openai/text-embedding-3-small", False),
        ("custom", "hosted_vllm/BAAI/bge-m3", False),
        ("custom", "BAAI/bge-m3", False),
        ("fastembed", "BAAI/bge-small-en-v1.5", False),
        ("ollama", "nomic-embed-text", False),
    ],
)
def test_base64_supported_only_on_tested_routes(provider, model, supported):
    assert base64_supported(provider, model) is supported


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("fastembed", "BAAI/bge-small-en-v1.5"),
        ("ollama", "nomic-embed-text"),
        ("bedrock", "bedrock/amazon.titan-embed-text-v2:0"),
        ("azure", "azure/my-deployment"),
    ],
)
def test_base64_on_an_unsupported_route_is_a_configuration_error(provider, model):
    with pytest.raises(CogneeConfigurationError, match="EMBEDDING_ENCODING_FORMAT"):
        create_embedding_engine(
            provider,
            model,
            4,
            None,
            None,
            None,
            None,
            36,
            None,
            None,
            None,
            None,
            "base64",
        )
