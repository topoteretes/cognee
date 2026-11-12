"""RabbitMQ RPC to a model worker's queue with direct reply-to.

Needs aio-pika (the ``remote-worker-amqp`` extra, alias ``gliner-remote-amqp``).
One session per client: a connection, a channel in publisher-confirm mode, and a
no-ack consumer on the ``amq.rabbitmq.reply-to`` pseudo-queue that routes replies
by correlation id. It is created lazily and rebuilt after the connection or
channel drops; a connection that is not kept (its setup failed, or its channel
closed under it) is closed.

Each call publishes a JSON body to the default exchange as a transient,
``mandatory`` message with its own correlation id, an ``x-request-id`` header and
an ``expiration`` equal to the request timeout, so a worker never starts work its
client has given up on. Unroutable (no queue), nacked or unconfirmed requests are
"unavailable" (retried by the callers); a confirmed request left unanswered is a
"timeout". Workers deliver at least once, so a duplicate reply is dropped.

Readiness passively declares the queue on a throwaway channel (a missing queue
closes the channel it is asked on) and requires at least one consumer.
Authentication is the broker's: credentials go in the URL, which is masked in
every log line and error.

The client knows nothing of request or reply bodies, and raises the error
classes its caller hands it (:class:`AmqpRpcErrors`), so each worker client keeps
its own typed failures.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass
from typing import Any

import aio_pika
import aiormq

from cognee.shared.logging_utils import get_logger

from .urls import redact_url

logger = get_logger("remote_worker.amqp_rpc")

REPLY_TO = "amq.rabbitmq.reply-to"


@dataclass(frozen=True)
class AmqpRpcErrors:
    """The caller's error classes; each is built from one message string."""

    #: The broker or worker cannot take the request now. Callers retry it.
    unavailable: type[Exception]
    #: The broker accepted the request and no reply came in time. Not resent.
    timeout: type[Exception]
    #: The broker refused the credentials in the URL.
    unauthorized: type[Exception]


class AmqpRpcClient:
    def __init__(
        self,
        url: str,
        *,
        queue: str,
        connect_timeout: float,
        request_timeout: float,
        errors: AmqpRpcErrors,
        worker_name: str,
    ):
        self._url = url
        self._queue = queue
        self.endpoint_label = f"{redact_url(url)} queue {queue}"
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._errors = errors
        # "GLiNER worker", "embedding worker": how messages name the other side.
        self._worker_name = worker_name
        self._session: _Session | None = None
        # The (re)connect in flight, if any: every caller that needs a session
        # while it runs shares its outcome instead of queueing for its own.
        self._connecting: asyncio.Future[_Session] | None = None

    async def call(self, body: bytes, *, request_id: str, message_type: str | None = None) -> bytes:
        """Publish one request and return the body of its reply.

        ``message_type`` is the AMQP ``type`` property; workers that serve more than
        one kind of message (the embedding worker's ``info``) dispatch on it.
        """
        session = await self._get_session()
        correlation_id = uuid.uuid4().hex
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        session.pending[correlation_id] = future
        message = aio_pika.Message(
            body=body,
            content_type="application/json",
            correlation_id=correlation_id,
            reply_to=REPLY_TO,
            # Seconds as a string on the wire; the broker drops the request once
            # nobody waits for its reply any more.
            expiration=self._request_timeout,
            headers={"x-request-id": request_id},
            type=message_type,
            delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
        )
        published = False
        try:
            # The publish and its confirm share the request deadline: a broker
            # under a memory or disk alarm blocks publishers without closing them.
            async with _timeout(self._request_timeout):
                await session.channel.default_exchange.publish(
                    message, routing_key=self._queue, mandatory=True
                )
                published = True
                return await future
        except (TimeoutError, asyncio.TimeoutError) as error:
            if published:
                # The broker took the request and no reply came: a worker may
                # still be running it, so it is not resent.
                raise self._errors.timeout(
                    f"No reply from the {self._worker_name} on {self.endpoint_label} within "
                    f"{self._request_timeout:g}s."
                ) from error
            raise self._errors.unavailable(
                f"The broker for {self.endpoint_label} did not accept the request within "
                f"{self._request_timeout:g}s (e.g. under a memory or disk alarm)."
            ) from error
        except self._errors.unavailable:
            raise
        except aio_pika.exceptions.DeliveryError as error:
            raise self._errors.unavailable(
                f"The broker could not route the request to {self.endpoint_label} "
                "(no such queue, or nacked)."
            ) from error
        except (aiormq.exceptions.AMQPError, ConnectionError, OSError) as error:
            raise self._connection_error(error) from error
        finally:
            session.pending.pop(correlation_id, None)

    async def consumer_count(self) -> int:
        """Consumers on the queue; raises "unavailable" when there are none."""
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
            raise self._errors.unavailable(
                f"The broker for {self.endpoint_label} did not answer within "
                f"{self._connect_timeout:g}s."
            ) from error
        except aiormq.exceptions.ChannelNotFoundEntity as error:
            raise self._errors.unavailable(
                f"No queue {self._queue!r} on {redact_url(self._url)}: no {self._worker_name} "
                "has declared it yet."
            ) from error
        except self._errors.unavailable:
            raise
        except (aiormq.exceptions.AMQPError, ConnectionError, OSError) as error:
            raise self._connection_error(error) from error
        consumers = queue.declaration_result.consumer_count
        if not consumers:
            raise self._errors.unavailable(
                f"No {self._worker_name} is consuming from {self.endpoint_label}."
            )
        return consumers

    async def aclose(self) -> None:
        connecting, self._connecting = self._connecting, None
        if connecting is not None and not connecting.done():
            # Its _connect closes what it opened when cancelled.
            connecting.cancel()
        session, self._session = self._session, None
        if session is not None:
            session.fail_pending(f"The {self._worker_name} transport was closed.")
            if not session.connection.is_closed:
                await session.connection.close()

    async def _get_session(self) -> _Session:
        """The live session, or the outcome of the one (re)connect in flight.

        Single-flight: the first caller to find no usable session starts a
        connect and every caller arriving while it runs awaits the same one, so
        N callers cost one connect timeout, not N in a row. Each waiter is
        shielded from the others' cancellation.
        """
        session = self._session
        if session is not None and session.usable:
            return session
        if self._connecting is None:
            stale, self._session = self._session, None
            connecting = asyncio.ensure_future(self._reconnect(stale))
            self._connecting = connecting

            def finished(done: asyncio.Future) -> None:
                if self._connecting is done:
                    self._connecting = None
                if not done.cancelled():
                    done.exception()  # retrieved even when every waiter was cancelled

            connecting.add_done_callback(finished)
        return await asyncio.shield(self._connecting)

    async def _reconnect(self, stale: _Session | None) -> _Session:
        if stale is not None:
            stale.fail_pending("The broker connection dropped.")
            # The channel may have closed while the connection stayed open (a
            # channel-level error): close it, or every rebuild leaks one.
            await _close_quietly(stale.connection)
        session = await self._connect()
        self._session = session
        return session

    async def _connect(self) -> _Session:
        connection = None
        try:
            async with _timeout(self._connect_timeout):
                connection = await aio_pika.connect(self._url, timeout=self._connect_timeout)
                channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
                session = _Session(connection, channel, self._errors.unavailable)
                reply_queue = await channel.get_queue(REPLY_TO, ensure=False)
                await reply_queue.consume(session.on_reply, no_ack=True)
        except BaseException as error:
            # Connected, but the channel or the reply consumer failed (or the
            # deadline or a cancellation hit): the connection is not kept, so close it.
            if connection is not None:
                await _close_quietly(connection)
            if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
                raise self._errors.unavailable(
                    f"Cannot connect to the broker for {self.endpoint_label} within "
                    f"{self._connect_timeout:g}s."
                ) from error
            if isinstance(error, (aiormq.exceptions.AMQPError, ConnectionError, OSError)):
                raise self._connection_error(error) from error
            raise

        def dropped(*_args: Any) -> None:
            session.fail_pending(f"The broker connection for {self.endpoint_label} dropped.")

        connection.close_callbacks.add(dropped)
        channel.close_callbacks.add(dropped)
        return session

    def _connection_error(self, error: BaseException) -> Exception:
        if isinstance(error, aiormq.exceptions.ProbableAuthenticationError):
            return self._errors.unauthorized(
                f"The broker for {self.endpoint_label} refused the credentials in the URL."
            )
        if isinstance(error, aiormq.exceptions.ChannelAccessRefused):
            # ACCESS_REFUSED (403) on a declare or publish: the user may log in but
            # lacks permission on the vhost, queue or exchange. Retrying cannot fix it.
            return self._errors.unauthorized(
                f"The broker for {self.endpoint_label} refused access for the user in the URL "
                f"({error}); check the user's permissions on the vhost and queue."
            )
        return self._errors.unavailable(
            f"Cannot reach the broker for {self.endpoint_label} ({type(error).__name__})."
        )


class _Session:
    def __init__(self, connection: Any, channel: Any, unavailable: type[Exception]):
        self.connection = connection
        self.channel = channel
        self.pending: dict[str, asyncio.Future[bytes]] = {}
        self._unavailable = unavailable

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
                future.set_exception(self._unavailable(reason))
        self.pending.clear()


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


async def _close_quietly(connection: Any) -> None:
    """Close a connection this client gives up on; failing to close it is not an error."""
    try:
        if not connection.is_closed:
            await connection.close()
    except Exception:  # best effort: the caller's own outcome matters more
        logger.debug("Closing an abandoned broker connection failed", exc_info=True)
