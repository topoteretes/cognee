"""Wire format of embedding responses: base64 or float JSON.

An OpenAI-style embeddings API returns each vector either as a JSON array of
decimals ("float") or as one base64 string of little-endian float32 values
("base64"). The values are the same; base64 is far cheaper to parse, because
the openai SDK builds a float response one element at a time.

base64 is used automatically only on routes known to accept it: OpenAI's own
API. Every other route sends the request cognee always sent and gets floats,
unless EMBEDDING_ENCODING_FORMAT=base64 opts it in. An opted-in route is
trusted and checked on the first reply: a provider that cannot honour base64
answers with a list, bad bytes or a 400/422, and each fails that call for good.
"""

import base64
import binascii
import os
import re
from typing import Literal
from urllib.parse import urlparse

import litellm
import numpy as np

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError

EncodingFormat = Literal["base64", "float"]
ENCODING_FORMATS = ("base64", "float")

ENCODING_FORMAT_HINT = (
    "EMBEDDING_ENCODING_FORMAT=base64 is set, and this route rejected it. "
    "Fix: comment out or remove EMBEDDING_ENCODING_FORMAT in your .env file or environment."
)


def validate_encoding_format(configured: str | None) -> EncodingFormat | None:
    """The configured format, or a configuration error for anything but the two values.

    The env path is typed, but cognee.config.set_embedding_config() stores whatever it
    is given, and a value like "Base64" would send no format and then reject the reply.
    """
    if configured is None or configured in ENCODING_FORMATS:
        return configured
    raise CogneeConfigurationError(
        f"EMBEDDING_ENCODING_FORMAT must be 'base64' or 'float', got {configured!r}.",
        remediation="Set EMBEDDING_ENCODING_FORMAT to base64 or float, or remove it.",
    )


_OPENAI_API_HOST = "api.openai.com"


def _routes_through_openai_handler(model: str | None) -> bool:
    """Whether litellm sends this model through its OpenAI handler, which forwards
    encoding_format unchanged. Other handlers translate or reject it."""
    if not model:
        return False
    if "/" in model:
        return model.split("/", 1)[0].lower() == "openai"
    # An unprefixed model resolves by litellm's model registry.
    return litellm.model_cost.get(model, {}).get("litellm_provider") == "openai"


def _litellm_openai_endpoint(endpoint: str | None) -> str | None:
    """The base URL litellm's OpenAI handler sends to, in litellm's own order.

    Mirrors litellm's lookup (api_base, litellm.api_base, OPENAI_BASE_URL,
    OPENAI_API_BASE), which cognee cannot call before the request is built. A
    mismatch only affects the default: base64 to a proxy that then rejects it, loudly.
    """
    return (
        endpoint or litellm.api_base or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    )


def _routes_to_openai_api(model: str | None, endpoint: str | None) -> bool:
    """Whether litellm sends this model to OpenAI's own embeddings API."""
    endpoint = _litellm_openai_endpoint(endpoint)
    if endpoint and urlparse(endpoint).hostname != _OPENAI_API_HOST:
        return False
    return _routes_through_openai_handler(model)


def base64_supported(provider: str) -> bool:
    """Whether EMBEDDING_ENCODING_FORMAT=base64 can be sent at all for this provider.

    Only the engines with no wire format are refused: fastembed embeds in-process and
    the ollama engine calls Ollama's native API, which has no such parameter. Every
    other engine sends the request and decode_embedding() checks the reply.
    """
    return provider not in ("fastembed", "ollama")


def resolve_encoding_format(
    model: str | None, endpoint: str | None, configured: EncodingFormat | None
) -> EncodingFormat:
    """The format to request: the configured one, else base64 only for OpenAI's own API."""
    configured = validate_encoding_format(configured)
    if configured is not None:
        return configured
    return "base64" if _routes_to_openai_api(model, endpoint) else "float"


def decode_embedding(embedding: list[float] | str, encoding_format: EncodingFormat) -> list[float]:
    """Return the vector as floats.

    With base64 requested, the server must answer with a base64 string of little-endian
    float32 values. A list means a provider or proxy translated the request into its own
    format (a LiteLLM proxy in front of Bedrock returns 1-bit vectors as a list of 0/1),
    so it raises EmbeddingEncodingError instead of storing garbage. With float, the list
    passes through unchanged.
    """
    if encoding_format == "float":
        return embedding
    if not isinstance(embedding, str):
        raise EmbeddingEncodingError(f"got {type(embedding).__name__}")
    try:
        vector = np.frombuffer(base64.b64decode(embedding, validate=True), dtype="<f4").tolist()
    except (binascii.Error, ValueError) as error:
        raise EmbeddingEncodingError(f"payload does not decode as float32: {error}") from error
    if not vector:
        raise EmbeddingEncodingError("payload is empty")
    return vector


_ENCODING_REJECTION_RE = re.compile(r"encoding_format|base64", re.IGNORECASE)


def rejects_encoding_format(error: BaseException) -> bool:
    """Whether a 400/422 is about the encoding, so ENCODING_FORMAT_HINT fits it.

    A server that rejects base64 says so ("unsupported encoding_format", "Input should
    be 'float' or 'base64'"); a bad model name or dimensions does not, and the hint
    would send the user after the wrong fix.
    """
    return bool(_ENCODING_REJECTION_RE.search(str(error)))
