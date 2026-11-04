"""What a transport does, and the embedding worker's ``/v1/embed`` wire contract.

A transport sends one request and answers readiness; retries, the request
budget, readiness caching, model and size checks, splitting over-long inputs and
failure scope live in the engine. Requests are the worker's ``POST /v1/embed``
JSON body, which RabbitMQ sends as is (see the worker's ``docs/api.md``).
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from .errors import EmbeddingWorkerRuntimeError

InputType = Literal["document", "query"]


@dataclass(frozen=True)
class EmbedReply:
    vectors: list[list[float]]
    model: str
    dimensions: int
    prompt_tokens: int
    elapsed_ms: float | None


class EmbeddingWorkerTransport(Protocol):
    #: "amqp" (the only transport so far).
    name: str
    #: The endpoint with any credentials masked, for logs and errors.
    endpoint_label: str

    async def embed(self, payload: dict[str, Any], *, request_id: str) -> EmbedReply:
        """Send one request. Raises an :class:`EmbeddingWorkerError` subclass on failure."""
        ...

    async def readiness(self) -> None:
        """Probe the worker once. Raises :class:`EmbeddingWorkerUnavailableError` when not ready."""
        ...

    async def aclose(self) -> None: ...


def embed_request(inputs: Sequence[str], input_type: InputType) -> dict[str, Any]:
    """The ``/v1/embed`` body. ``dimensions`` is never sent: the model's full size is
    what EMBEDDING_DIMENSIONS must match, and ``normalize`` / ``truncate`` keep the
    worker's defaults (unit vectors; over-long inputs rejected so they are split)."""
    return {"inputs": list(inputs), "input_type": input_type, "encoding_format": "base64"}


def parse_embed_reply(body: bytes, expected_inputs: int) -> EmbedReply:
    """Parse a success body; anything malformed is a runtime fault of the worker."""
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EmbeddingWorkerRuntimeError(
            f"The embedding worker sent invalid JSON: {error}"
        ) from error
    if not isinstance(data, dict):
        raise EmbeddingWorkerRuntimeError("The embedding worker's reply is not a JSON object.")
    embeddings = data.get("embeddings")
    model = data.get("model")
    dimensions = data.get("dimensions")
    usage = data.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    elapsed_ms = data.get("elapsed_ms")
    if (
        not isinstance(embeddings, list)
        or not isinstance(model, str)
        or not _is_int(dimensions)
        or not _is_int(prompt_tokens)
    ):
        raise EmbeddingWorkerRuntimeError(
            "The embedding worker's reply lacks `embeddings`, `model`, `dimensions` or "
            "`usage.prompt_tokens`."
        )
    if data.get("encoding_format", "base64") != "base64":
        raise EmbeddingWorkerRuntimeError(
            f"The embedding worker answered in {data.get('encoding_format')!r}, not base64."
        )
    if len(embeddings) != expected_inputs:
        raise EmbeddingWorkerRuntimeError(
            f"The embedding worker returned {len(embeddings)} vectors for {expected_inputs} inputs."
        )
    if elapsed_ms is not None and not isinstance(elapsed_ms, (int, float)):
        raise EmbeddingWorkerRuntimeError("The embedding worker's `elapsed_ms` is not a number.")
    return EmbedReply(
        vectors=[_decode(vector, dimensions) for vector in embeddings],
        model=model,
        dimensions=dimensions,
        prompt_tokens=prompt_tokens,
        elapsed_ms=float(elapsed_ms) if elapsed_ms is not None else None,
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decode(encoded: Any, dimensions: int) -> list[float]:
    """Base64 of little-endian float32, ``dimensions`` long and finite."""
    if not isinstance(encoded, str):
        raise EmbeddingWorkerRuntimeError(
            f"The embedding worker sent a {type(encoded).__name__} vector, not base64."
        )
    try:
        vector = np.frombuffer(base64.b64decode(encoded, validate=True), dtype="<f4")
    except (binascii.Error, ValueError) as error:
        raise EmbeddingWorkerRuntimeError(
            f"The embedding worker sent a vector that does not decode as float32: {error}"
        ) from error
    if vector.size != dimensions:
        raise EmbeddingWorkerRuntimeError(
            f"The embedding worker sent a {vector.size}-component vector in a reply that "
            f"declares {dimensions} dimensions."
        )
    if not np.isfinite(vector).all():
        raise EmbeddingWorkerRuntimeError("The embedding worker sent NaN or infinity in a vector.")
    return vector.tolist()
