"""The shared RabbitMQ RPC client: connections it gives up on are closed.

Against fakes of aio-pika's connection and channel; the broker behaviour is
covered by the GLiNER and embedding worker transport tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("aio_pika")

import aiormq

from cognee.infrastructure.remote_worker import amqp_rpc
from cognee.infrastructure.remote_worker.amqp_rpc import (
    AmqpRpcClient,
    AmqpRpcErrors,
)


class Unavailable(Exception):
    pass


class Timeout(Exception):
    pass


class Unauthorized(Exception):
    pass


class FakeQueue:
    def __init__(self, fail: BaseException | None):
        self.fail = fail

    async def consume(self, callback, no_ack: bool):
        if self.fail is not None:
            raise self.fail


class FakeChannel:
    def __init__(self, *, consume_fails: BaseException | None = None):
        self.is_closed = False
        self.close_callbacks: set = set()
        self.consume_fails = consume_fails

    async def get_queue(self, name, ensure):
        return FakeQueue(self.consume_fails)


class FakeConnection:
    def __init__(
        self,
        *,
        channel_fails: BaseException | None = None,
        consume_fails: BaseException | None = None,
    ):
        self.is_closed = False
        self.closed = 0
        self.close_callbacks: set = set()
        self.channel_fails = channel_fails
        self.consume_fails = consume_fails

    async def channel(self, **_kwargs):
        if self.channel_fails is not None:
            raise self.channel_fails
        return FakeChannel(consume_fails=self.consume_fails)

    async def close(self):
        self.closed += 1
        self.is_closed = True


def client() -> AmqpRpcClient:
    return AmqpRpcClient(
        "amqp://guest:guest@broker:5672/%2f",
        queue="q",
        connect_timeout=2,
        request_timeout=2,
        errors=AmqpRpcErrors(unavailable=Unavailable, timeout=Timeout, unauthorized=Unauthorized),
        worker_name="test worker",
    )


def connecting(monkeypatch, connections: list[FakeConnection]) -> None:
    pending = list(connections)

    async def connect(url, timeout):
        return pending.pop(0)

    monkeypatch.setattr(amqp_rpc.aio_pika, "connect", connect)


@pytest.mark.parametrize(
    "failure",
    [
        {"channel_fails": aiormq.exceptions.ChannelClosed(None, None)},
        {"consume_fails": aiormq.exceptions.ChannelAccessRefused(None)},
        {"consume_fails": ConnectionResetError()},
    ],
)
@pytest.mark.asyncio
async def test_a_connection_whose_session_setup_fails_is_closed(monkeypatch, failure):
    connection = FakeConnection(**failure)
    connecting(monkeypatch, [connection])
    with pytest.raises(Unavailable):
        await client()._get_session()
    assert connection.closed == 1


@pytest.mark.asyncio
async def test_a_session_whose_channel_closed_has_its_connection_closed_on_rebuild(monkeypatch):
    first, second = FakeConnection(), FakeConnection()
    connecting(monkeypatch, [first, second])
    rpc = client()
    session = await rpc._get_session()
    assert session.connection is first

    session.channel.is_closed = True  # a channel-level error; the connection stays open
    rebuilt = await rpc._get_session()

    assert rebuilt.connection is second
    assert first.closed == 1
    assert second.closed == 0


@pytest.mark.asyncio
async def test_a_session_whose_connection_dropped_is_rebuilt_without_closing_it_again(monkeypatch):
    first, second = FakeConnection(), FakeConnection()
    connecting(monkeypatch, [first, second])
    rpc = client()
    await rpc._get_session()
    first.is_closed = True  # dropped by the broker
    assert (await rpc._get_session()).connection is second
    assert first.closed == 0
