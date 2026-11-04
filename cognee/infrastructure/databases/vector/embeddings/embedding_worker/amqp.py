"""RabbitMQ transport to the embedding worker: the ``/v1/embed`` body (and the
``type=info`` request for the worker's model info) over the RPC client the GLiNER
worker client uses too
(:class:`cognee.infrastructure.remote_worker.amqp_rpc.AmqpRpcClient`: direct
reply-to, correlation ids, ``x-request-id``, ``expiration`` = request timeout,
publisher confirms with ``mandatory``, duplicate replies dropped, readiness =
at least one consumer on the queue).

Needs aio-pika, from the ``remote-worker-amqp`` extra (alias ``gliner-remote-amqp``).
The engine imports this module when it is built, so a missing aio-pika fails then.
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
from .protocol import EmbedReply, WorkerInfo, parse_embed_reply, parse_info_reply

try:
    from cognee.infrastructure.remote_worker.amqp_rpc import AmqpRpcClient, AmqpRpcErrors
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise EmbeddingWorkerConfigError(
        "EMBEDDING_PROVIDER=embedding_worker needs aio-pika for its RabbitMQ transport, "
        "which is not installed.",
        remediation='Install it with: pip install "cognee[remote-worker-amqp]"',
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


#: AMQP ``type`` of an info request; requests without a type are embed requests.
INFO_MESSAGE_TYPE = "info"


def _raise_error_reply(body: bytes, endpoint_label: str) -> None:
    """Raise the typed error for an ``{"error": {code, message}}`` body; return otherwise."""
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        code = data["error"].get("code")
        message = data["error"].get("message") or ""
        error_class = _ERROR_CODES.get(code, EmbeddingWorkerRuntimeError)
        raise error_class(f"The embedding worker on {endpoint_label} answered {code}: {message}")


def reply_to_result(body: bytes, endpoint_label: str, expected_inputs: int) -> EmbedReply:
    """A reply body is either the ``/v1/embed`` success JSON or ``{"error": {code, message}}``."""
    _raise_error_reply(body, endpoint_label)
    return parse_embed_reply(body, expected_inputs)


def info_reply_to_result(body: bytes, endpoint_label: str) -> WorkerInfo:
    """An info reply is the ``/v1/info`` JSON or ``{"error": {code, message}}``."""
    _raise_error_reply(body, endpoint_label)
    return parse_info_reply(body)


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

    async def info(self, *, request_id: str) -> WorkerInfo:
        body = await self._rpc.call(b"{}", request_id=request_id, message_type=INFO_MESSAGE_TYPE)
        return info_reply_to_result(body, self.endpoint_label)

    async def readiness(self) -> None:
        await self._rpc.consumer_count()

    async def aclose(self) -> None:
        await self._rpc.aclose()
