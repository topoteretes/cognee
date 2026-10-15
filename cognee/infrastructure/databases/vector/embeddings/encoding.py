"""Wire format of embedding responses: base64 or float JSON.

An OpenAI-style embeddings API returns each vector either as a JSON array of
decimals ("float") or as one base64 string of little-endian float32 values
("base64"). The values are the same; base64 is far cheaper to parse, because
the openai SDK builds a float response one element at a time.

base64 is used automatically only on routes known to accept it: OpenAI's own
API. Every other route sends the request cognee always sent and gets floats,
unless EMBEDDING_ENCODING_FORMAT=base64 opts it in.
"""

import base64
from typing import Literal
from urllib.parse import urlparse

import litellm
import numpy as np

EncodingFormat = Literal["base64", "float"]

ENCODING_FORMAT_HINT = (
    "EMBEDDING_ENCODING_FORMAT=base64 is set; if this endpoint does not support "
    "base64 embeddings, set EMBEDDING_ENCODING_FORMAT=float."
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


def _routes_to_openai_api(model: str | None, endpoint: str | None) -> bool:
    """Whether litellm sends this model to OpenAI's own embeddings API."""
    if endpoint and urlparse(endpoint).hostname != _OPENAI_API_HOST:
        return False
    return _routes_through_openai_handler(model)


def base64_supported(provider: str, model: str | None) -> bool:
    """Whether EMBEDDING_ENCODING_FORMAT=base64 may be set for this provider and model.

    Only routes that send OpenAI's embeddings request unchanged, which are the ones
    tested: the openai_compatible engine and litellm's OpenAI handler. litellm maps
    base64 to other providers' own formats, e.g. Bedrock's 1-bit "binary" vectors.
    """
    if provider == "openai_compatible":
        return True
    if provider in ("fastembed", "ollama"):
        return False
    return _routes_through_openai_handler(model)


def resolve_encoding_format(
    model: str | None, endpoint: str | None, configured: EncodingFormat | None
) -> EncodingFormat:
    """The format to request: the configured one, else base64 only for OpenAI's own API."""
    if configured is not None:
        return configured
    return "base64" if _routes_to_openai_api(model, endpoint) else "float"


def decode_embedding(embedding: list[float] | str) -> list[float]:
    """Return the vector as floats; a base64 payload holds little-endian float32 values.

    A float list passes through unchanged, since a server may ignore encoding_format.
    """
    if isinstance(embedding, str):
        return np.frombuffer(base64.b64decode(embedding), dtype="<f4").tolist()
    return embedding
