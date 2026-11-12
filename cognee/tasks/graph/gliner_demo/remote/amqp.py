"""RabbitMQ transport: RPC on the worker's queue with direct reply-to.

Needs the ``gliner-remote-amqp`` extra (aio-pika). The RPC itself (session,
publishing, reply routing, readiness) is
:class:`cognee.infrastructure.remote_worker.amqp_rpc.AmqpRpcClient`, shared with
the embedding worker client; this module adds the GLiNER body, reply and errors.
One session per transport: a connection, a channel in publisher-confirm mode,
and a no-ack consumer on the ``amq.rabbitmq.reply-to`` pseudo-queue that routes
replies by correlation id. It is created lazily and rebuilt after the connection
or channel drops.

Each attempt publishes the HTTP JSON body to the default exchange as a
transient, ``mandatory`` message with its own correlation id, an
``x-request-id`` header and an ``expiration`` equal to the request timeout, so
a worker never starts work its client has given up on. Unroutable (no queue),
nacked or unconfirmed requests are :class:`GlinerWorkerUnavailableError` and
retried; a confirmed request left unanswered is :class:`GlinerWorkerTimeoutError`. The worker delivers at least once, so a duplicate reply is dropped.

Readiness passively declares the queue on a throwaway channel (a missing queue
closes the channel it is asked on) and requires at least one consumer.
Authentication is the broker's: credentials go in the URL, which is masked in
every log line and error.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cognee.shared.logging_utils import get_logger

from .errors import (
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerTimeoutError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from .protocol import WorkerReadiness, WorkerReply, parse_json_reply, redact_url

try:
    from cognee.infrastructure.remote_worker.amqp_rpc import REPLY_TO, AmqpRpcClient, AmqpRpcErrors
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise GlinerRemoteConfigError(
        "COGNEE_GLINER_TRANSPORT=amqp needs aio-pika, which is not installed.",
        remediation='Install it with: pip install "cognee[gliner-remote-amqp]"',
    ) from error

logger = get_logger("gliner.remote.amqp")

__all__ = ["REPLY_TO", "AmqpWorkerTransport", "reply_to_result"]

_RPC_ERRORS = AmqpRpcErrors(
    unavailable=GlinerWorkerUnavailableError,
    timeout=GlinerWorkerTimeoutError,
    unauthorized=GlinerWorkerUnauthorizedError,
)
_ERROR_CODES = {
    "invalid_request": GlinerWorkerRejectedError,
    "unavailable": GlinerWorkerUnavailableError,
    "internal": GlinerWorkerRuntimeError,
}


def reply_to_result(body: bytes, endpoint_label: str, max_response_bytes: int) -> WorkerReply:
    """A reply body is either the HTTP success JSON or ``{"error": {code, message}}``."""
    if len(body) > max_response_bytes:
        raise GlinerWorkerRuntimeError(
            f"The GLiNER worker's reply ({len(body)} bytes) exceeds the {max_response_bytes}-byte "
            "cap (COGNEE_GLINER_MAX_RESPONSE_BYTES)."
        )
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        code = data["error"].get("code")
        message = data["error"].get("message") or ""
        error_class = _ERROR_CODES.get(code, GlinerWorkerRuntimeError)
        return_message = f"The GLiNER worker on {endpoint_label} answered {code}: {message}"
        raise error_class(return_message)
    return parse_json_reply(body)


class AmqpWorkerTransport:
    name = "amqp"

    def __init__(
        self,
        url: str,
        *,
        queue: str,
        connect_timeout: float,
        request_timeout: float,
        max_response_bytes: int,
    ):
        if not url.startswith(("amqp://", "amqps://")):
            raise GlinerRemoteConfigError(
                "COGNEE_GLINER_ENDPOINT must be an amqp(s):// broker URL for the RabbitMQ "
                f"transport, got {redact_url(url)!r}."
            )
        self._rpc = AmqpRpcClient(
            url,
            queue=queue,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            errors=_RPC_ERRORS,
            worker_name="GLiNER worker",
        )
        self.endpoint_label = self._rpc.endpoint_label
        self._max_response_bytes = max_response_bytes

    async def extract(self, payload: Mapping[str, Any], *, request_id: str) -> WorkerReply:
        body = await self._rpc.call(json.dumps(payload).encode(), request_id=request_id)
        return reply_to_result(body, self.endpoint_label, self._max_response_bytes)

    async def readiness(self) -> WorkerReadiness:
        await self._rpc.consumer_count()
        return WorkerReadiness()

    async def aclose(self) -> None:
        await self._rpc.aclose()
