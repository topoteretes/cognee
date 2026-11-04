"""RabbitMQ transport to the embedding worker: the ``/v1/embed`` body over the
RPC client the GLiNER worker client uses too
(:class:`cognee.infrastructure.remote_worker.amqp_rpc.AmqpRpcClient`: direct
reply-to, correlation ids, ``x-request-id``, ``expiration`` = request timeout,
publisher confirms with ``mandatory``, duplicate replies dropped, readiness =
at least one consumer on the queue).

Needs aio-pika, from the ``gliner-remote-amqp`` extra.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import (
    EmbeddingWorkerConfigError,
    EmbeddingWorkerInputTooLongError,
    EmbeddingWorkerRejectedError,
    EmbeddingWorkerRuntimeError,
    EmbeddingWorkerTimeoutError,
    EmbeddingWorkerUnauthorizedError,
    EmbeddingWorkerUnavailableError,
)
from .protocol import EmbedReply, parse_embed_reply

try:
    from cognee.infrastructure.remote_worker.amqp_rpc import AmqpRpcClient, AmqpRpcErrors
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise EmbeddingWorkerConfigError(
        "EMBEDDING_PROVIDER=embedding_worker needs aio-pika for its RabbitMQ transport, "
        "which is not installed.",
        remediation='Install it with: pip install "cognee[gliner-remote-amqp]"',
    ) from error

_RPC_ERRORS = AmqpRpcErrors(
    unavailable=EmbeddingWorkerUnavailableError,
    timeout=EmbeddingWorkerTimeoutError,
    unauthorized=EmbeddingWorkerUnauthorizedError,
)
# The worker's error codes (its docs/rabbitmq.md); unknown codes are runtime faults.
_ERROR_CODES = {
    "invalid_request": EmbeddingWorkerRejectedError,
    "input_too_long": EmbeddingWorkerInputTooLongError,
    "unavailable": EmbeddingWorkerUnavailableError,
    "internal": EmbeddingWorkerRuntimeError,
}


def reply_to_result(body: bytes, endpoint_label: str, expected_inputs: int) -> EmbedReply:
    """A reply body is either the ``/v1/embed`` success JSON or ``{"error": {code, message}}``."""
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        code = data["error"].get("code")
        message = data["error"].get("message") or ""
        error_class = _ERROR_CODES.get(code, EmbeddingWorkerRuntimeError)
        raise error_class(f"The embedding worker on {endpoint_label} answered {code}: {message}")
    return parse_embed_reply(body, expected_inputs)


class AmqpEmbeddingTransport:
    name = "amqp"

    def __init__(self, url: str, *, queue: str, connect_timeout: float, request_timeout: float):
        self._rpc = AmqpRpcClient(
            url,
            queue=queue,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            errors=_RPC_ERRORS,
            worker_name="embedding worker",
        )
        self.endpoint_label = self._rpc.endpoint_label

    async def embed(self, payload: dict[str, Any], *, request_id: str) -> EmbedReply:
        body = await self._rpc.call(json.dumps(payload).encode(), request_id=request_id)
        return reply_to_result(body, self.endpoint_label, len(payload["inputs"]))

    async def readiness(self) -> None:
        await self._rpc.consumer_count()

    async def aclose(self) -> None:
        await self._rpc.aclose()
