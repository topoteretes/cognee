"""RabbitMQ transport: RPC on the worker's queue with direct reply-to.

Needs the ``gliner-remote-amqp`` extra (aio-pika). One session per transport:
a connection, a channel in publisher-confirm mode, and a no-ack consumer on the
``amq.rabbitmq.reply-to`` pseudo-queue that routes replies by correlation id.
It is created lazily and rebuilt after the connection or channel drops.

Each attempt publishes the HTTP JSON body to the default exchange as a
transient, ``mandatory`` message with its own correlation id, an
``x-request-id`` header and an ``expiration`` equal to the request timeout, so
a worker never starts work its client has given up on. Unroutable (no queue),
nacked or unanswered requests are :class:`GlinerWorkerUnavailableError` and
retried. The worker delivers at least once, so a duplicate reply is dropped.

Readiness passively declares the queue on a throwaway channel (a missing queue
closes the channel it is asked on) and requires at least one consumer.
Authentication is the broker's: credentials go in the URL, which is masked in
every log line and error.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import Mapping
from typing import Any

from cognee.shared.logging_utils import get_logger

from .errors import (
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from .protocol import WorkerReadiness, WorkerReply, parse_json_reply, redact_url

try:
    import aio_pika
    import aiormq
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise GlinerRemoteConfigError(
        "COGNEE_GLINER_TRANSPORT=amqp needs aio-pika, which is not installed.",
        remediation='Install it with: pip install "cognee[gliner-remote-amqp]"',
    ) from error

logger = get_logger("gliner.remote.amqp")

REPLY_TO = "amq.rabbitmq.reply-to"
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


class _Session:
    def __init__(self, connection: Any, channel: Any):
        self.connection = connection
        self.channel = channel
        self.pending: dict[str, asyncio.Future[bytes]] = {}

    @property
    def usable(self) -> bool:
        return not self.connection.is_closed and not self.channel.is_closed

    async def on_reply(self, message: Any) -> None:
        future = self.pending.pop(message.correlation_id or "", None)
        # Unknown ids are stray or duplicate replies (at-least-once delivery).
        if future is not None and not future.done():
            future.set_result(message.body)

    def fail_pending(self, reason: str) -> None:
        for future in self.pending.values():
            if not future.done():
                future.set_exception(GlinerWorkerUnavailableError(reason))
        self.pending.clear()


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
        self._url = url
        self._queue = queue
        self.endpoint_label = f"{redact_url(url)} queue {queue}"
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._max_response_bytes = max_response_bytes
        self._session: _Session | None = None
        self._session_lock = asyncio.Lock()

    async def extract(self, payload: Mapping[str, Any], *, request_id: str) -> WorkerReply:
        session = await self._get_session()
        correlation_id = uuid.uuid4().hex
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        session.pending[correlation_id] = future
        message = aio_pika.Message(
            body=json.dumps(payload).encode(),
            content_type="application/json",
            correlation_id=correlation_id,
            reply_to=REPLY_TO,
            # Seconds as a string on the wire; the broker drops the request once
            # nobody waits for its reply any more.
            expiration=self._request_timeout,
            headers={"x-request-id": request_id},
            delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
        )
        try:
            # The publish and its confirm share the request deadline: a broker
            # under a memory or disk alarm blocks publishers without closing them.
            async with _timeout(self._request_timeout):
                await session.channel.default_exchange.publish(
                    message, routing_key=self._queue, mandatory=True
                )
                body = await future
        except (TimeoutError, asyncio.TimeoutError) as error:
            raise GlinerWorkerUnavailableError(
                f"No reply from the GLiNER worker on {self.endpoint_label} within "
                f"{self._request_timeout:g}s (or the broker, e.g. under a memory alarm, did "
                "not accept the request)."
            ) from error
        except GlinerWorkerUnavailableError:
            raise
        except aio_pika.exceptions.DeliveryError as error:
            raise GlinerWorkerUnavailableError(
                f"The broker could not route the request to {self.endpoint_label} "
                "(no such queue, or nacked)."
            ) from error
        except (aiormq.exceptions.AMQPError, ConnectionError, OSError) as error:
            raise self._connection_error(error) from error
        finally:
            session.pending.pop(correlation_id, None)
        return reply_to_result(body, self.endpoint_label, self._max_response_bytes)

    async def readiness(self) -> WorkerReadiness:
        try:
            async with _timeout(self._connect_timeout):
                session = await self._get_session()
                channel = await session.connection.channel(publisher_confirms=False)
                try:
                    queue = await channel.declare_queue(self._queue, passive=True)
                finally:
                    if not channel.is_closed:
                        await channel.close()
        except (TimeoutError, asyncio.TimeoutError) as error:
            raise GlinerWorkerUnavailableError(
                f"The broker for {self.endpoint_label} did not answer within "
                f"{self._connect_timeout:g}s."
            ) from error
        except aiormq.exceptions.ChannelNotFoundEntity as error:
            raise GlinerWorkerUnavailableError(
                f"No queue {self._queue!r} on {redact_url(self._url)}: no GLiNER worker has "
                "declared it yet."
            ) from error
        except GlinerWorkerUnavailableError:
            raise
        except (aiormq.exceptions.AMQPError, ConnectionError, OSError) as error:
            raise self._connection_error(error) from error
        consumers = queue.declaration_result.consumer_count
        if not consumers:
            raise GlinerWorkerUnavailableError(
                f"No GLiNER worker is consuming from {self.endpoint_label}."
            )
        return WorkerReadiness()

    async def aclose(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.fail_pending("The GLiNER transport was closed.")
            if not session.connection.is_closed:
                await session.connection.close()

    async def _get_session(self) -> _Session:
        session = self._session
        if session is not None and session.usable:
            return session
        async with self._session_lock:
            if self._session is not None and self._session.usable:
                return self._session
            if self._session is not None:
                self._session.fail_pending("The broker connection dropped.")
            self._session = await self._connect()
            return self._session

    async def _connect(self) -> _Session:
        try:
            async with _timeout(self._connect_timeout):
                connection = await aio_pika.connect(self._url, timeout=self._connect_timeout)
                channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
                session = _Session(connection, channel)
                reply_queue = await channel.get_queue(REPLY_TO, ensure=False)
                await reply_queue.consume(session.on_reply, no_ack=True)
        except (TimeoutError, asyncio.TimeoutError) as error:
            raise GlinerWorkerUnavailableError(
                f"Cannot connect to the broker for {self.endpoint_label} within "
                f"{self._connect_timeout:g}s."
            ) from error
        except (aiormq.exceptions.AMQPError, ConnectionError, OSError) as error:
            raise self._connection_error(error) from error

        def dropped(*_args: Any) -> None:
            session.fail_pending(f"The broker connection for {self.endpoint_label} dropped.")

        connection.close_callbacks.add(dropped)
        channel.close_callbacks.add(dropped)
        return session

    def _connection_error(self, error: BaseException) -> Exception:
        if isinstance(error, aiormq.exceptions.ProbableAuthenticationError):
            return GlinerWorkerUnauthorizedError(
                f"The broker for {self.endpoint_label} refused the credentials in the URL."
            )
        return GlinerWorkerUnavailableError(
            f"Cannot reach the broker for {self.endpoint_label} ({type(error).__name__})."
        )


def _timeout(seconds: float):
    """An async deadline: ``asyncio.timeout`` where it exists (3.11+)."""
    if sys.version_info >= (3, 11):
        return asyncio.timeout(seconds)
    return _Timeout310(seconds)


class _Timeout310:
    """Python 3.10 lacks ``asyncio.timeout``: cancel the task when the deadline passes."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._handle: asyncio.TimerHandle | None = None
        self._task: asyncio.Task | None = None
        self._expired = False

    async def __aenter__(self) -> None:
        self._task = asyncio.current_task()
        self._handle = asyncio.get_running_loop().call_later(self._seconds, self._expire)

    def _expire(self) -> None:
        self._expired = True
        if self._task is not None:
            self._task.cancel()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.cancel()
        if self._expired and exc_type is asyncio.CancelledError:
            raise asyncio.TimeoutError()
        return False
