"""Remote GLiNER RabbitMQ transport (SDK-980).

Reply mapping is unit-tested here. The broker tests need a RabbitMQ and run when
``COGNEE_GLINER_AMQP_TEST_URL`` is set (e.g. ``amqp://guest:guest@localhost:5672/``);
a fake worker on a private queue answers them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid

import pytest

aio_pika = pytest.importorskip("aio_pika")

from cognee.tasks.graph.gliner_demo.remote import (  # noqa: E402
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from cognee.tasks.graph.gliner_demo.remote.amqp import (  # noqa: E402
    AmqpWorkerTransport,
    reply_to_result,
)

BROKER_URL = os.environ.get("COGNEE_GLINER_AMQP_TEST_URL")
needs_broker = pytest.mark.skipif(not BROKER_URL, reason="COGNEE_GLINER_AMQP_TEST_URL is not set")

PAYLOAD = {
    "inputs": [{"id": "0", "text": "Alice works for Acme."}],
    "schema": {"entities": {"person": ""}, "relations": {}},
    "options": {"window_words": 384, "window_overlap_words": 64},
}


def success_body(payload: dict) -> bytes:
    return json.dumps(
        {
            "items": [
                {"id": item["id"], "extraction": {"entities": {"person": [item["text"]]}}}
                for item in payload["inputs"]
            ],
            "model": "fake/model",
            "elapsed_ms": 1.0,
            "windowed": True,
        }
    ).encode()


# --------------------------------------------------------------------------- #
# Reply mapping
# --------------------------------------------------------------------------- #


def test_success_reply_is_the_http_body():
    reply = reply_to_result(success_body(PAYLOAD), "label", 1 << 20)
    assert reply.items == [("0", {"entities": {"person": ["Alice works for Acme."]}})]
    assert reply.windowed is True


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("invalid_request", GlinerWorkerRejectedError),
        ("unavailable", GlinerWorkerUnavailableError),
        ("internal", GlinerWorkerRuntimeError),
        ("something_new", GlinerWorkerRuntimeError),
    ],
)
def test_error_replies_map_like_http_statuses(code, error):
    body = json.dumps({"error": {"code": code, "message": "details"}}).encode()
    with pytest.raises(error, match="details"):
        reply_to_result(body, "label", 1 << 20)


def test_oversized_and_malformed_replies_are_runtime_errors():
    with pytest.raises(GlinerWorkerRuntimeError, match="cap"):
        reply_to_result(success_body(PAYLOAD), "label", 10)
    with pytest.raises(GlinerWorkerRuntimeError):
        reply_to_result(b"not json", "label", 1 << 20)


@pytest.mark.parametrize("url", ["http://broker:5672", "broker:5672"])
def test_endpoint_must_be_an_amqp_url(url):
    with pytest.raises(GlinerRemoteConfigError):
        AmqpWorkerTransport(
            url, queue="q", connect_timeout=1, request_timeout=1, max_response_bytes=1
        )


def test_credentials_are_masked_in_the_label():
    transport = AmqpWorkerTransport(
        "amqp://guest:s3cret@broker:5672/%2f",
        queue="gliner_worker.extract",
        connect_timeout=1,
        request_timeout=1,
        max_response_bytes=1,
    )
    assert "s3cret" not in transport.endpoint_label
    assert transport.endpoint_label == "amqp://***@broker:5672/%2f queue gliner_worker.extract"


@pytest.mark.asyncio
async def test_a_refused_connection_is_unavailable():
    transport = AmqpWorkerTransport(
        "amqp://guest:guest@127.0.0.1:1/",
        queue="q",
        connect_timeout=2,
        request_timeout=2,
        max_response_bytes=1 << 20,
    )
    with pytest.raises(GlinerWorkerUnavailableError):
        await transport.readiness()
    with pytest.raises(GlinerWorkerUnavailableError):
        await transport.extract(PAYLOAD, request_id="r")


# --------------------------------------------------------------------------- #
# Against a broker
# --------------------------------------------------------------------------- #


@contextlib.asynccontextmanager
async def fake_worker(queue: str, respond):
    """Consume ``queue`` and answer each request with ``respond(message) -> [bodies]``."""
    connection = await aio_pika.connect(BROKER_URL)
    channel = await connection.channel()
    # Durable: RabbitMQ 4 refuses transient non-exclusive queues, and an exclusive
    # one would refuse the transport's passive declare.
    declared = await channel.declare_queue(queue, durable=True, auto_delete=True)
    seen: list = []

    async def on_message(message):
        async with message.process():
            seen.append(message)
            for body in respond(message):
                await channel.default_exchange.publish(
                    aio_pika.Message(body=body, correlation_id=message.correlation_id),
                    routing_key=message.reply_to,
                )

    await declared.consume(on_message)
    try:
        yield seen
    finally:
        await connection.close()


def transport_for(queue: str, *, request_timeout: float = 5.0) -> AmqpWorkerTransport:
    return AmqpWorkerTransport(
        BROKER_URL,
        queue=queue,
        connect_timeout=5,
        request_timeout=request_timeout,
        max_response_bytes=1 << 20,
    )


def private_queue() -> str:
    return f"cognee-test-{uuid.uuid4().hex}"


@needs_broker
@pytest.mark.asyncio
async def test_request_properties_and_reply_routing():
    queue = private_queue()
    async with fake_worker(queue, lambda m: [success_body(json.loads(m.body))]) as seen:
        transport = transport_for(queue)
        assert (await transport.readiness()).model is None
        reply = await transport.extract(PAYLOAD, request_id="req-1")
        await transport.aclose()

    [message] = seen
    assert json.loads(message.body) == PAYLOAD
    assert message.headers["x-request-id"] == "req-1"
    # The broker rewrites the pseudo-queue name into a per-channel one.
    assert message.reply_to.startswith("amq.rabbitmq.reply-to")
    assert message.content_type == "application/json"
    assert message.expiration is not None
    assert reply.items == [("0", {"entities": {"person": ["Alice works for Acme."]}})]


@needs_broker
@pytest.mark.asyncio
async def test_concurrent_requests_get_their_own_replies():
    queue = private_queue()
    async with fake_worker(queue, lambda m: [success_body(json.loads(m.body))]):
        transport = transport_for(queue)
        payloads = [
            {**PAYLOAD, "inputs": [{"id": "0", "text": f"Name{i} here."}]} for i in range(8)
        ]
        replies = await asyncio.gather(
            *(transport.extract(p, request_id=str(i)) for i, p in enumerate(payloads))
        )
        await transport.aclose()
    assert [r.items[0][1]["entities"]["person"][0] for r in replies] == [
        f"Name{i} here." for i in range(8)
    ]


@needs_broker
@pytest.mark.asyncio
async def test_duplicate_and_stray_replies_are_dropped():
    queue = private_queue()

    def twice(message):
        body = success_body(json.loads(message.body))
        return [body, body]

    async with fake_worker(queue, twice):
        transport = transport_for(queue)
        first = await transport.extract(PAYLOAD, request_id="a")
        second = await transport.extract(PAYLOAD, request_id="b")
        await transport.aclose()
    assert first.items == second.items


@needs_broker
@pytest.mark.asyncio
async def test_error_replies_raise_their_class():
    queue = private_queue()
    error = json.dumps({"error": {"code": "invalid_request", "message": "bad schema"}}).encode()
    async with fake_worker(queue, lambda _m: [error]):
        transport = transport_for(queue)
        with pytest.raises(GlinerWorkerRejectedError, match="bad schema"):
            await transport.extract(PAYLOAD, request_id="r")
        await transport.aclose()


@needs_broker
@pytest.mark.asyncio
async def test_no_reply_within_the_timeout_is_unavailable():
    queue = private_queue()
    async with fake_worker(queue, lambda _m: []):
        transport = transport_for(queue, request_timeout=0.5)
        with pytest.raises(GlinerWorkerUnavailableError, match="No reply"):
            await transport.extract(PAYLOAD, request_id="r")
        await transport.aclose()


@needs_broker
@pytest.mark.asyncio
async def test_a_missing_queue_is_unavailable_for_readiness_and_requests():
    transport = transport_for(private_queue())
    with pytest.raises(GlinerWorkerUnavailableError, match="No queue"):
        await transport.readiness()
    with pytest.raises(GlinerWorkerUnavailableError, match="could not route"):
        await transport.extract(PAYLOAD, request_id="r")
    # The session survives both: a later readiness check still answers.
    with pytest.raises(GlinerWorkerUnavailableError, match="No queue"):
        await transport.readiness()
    await transport.aclose()


@needs_broker
@pytest.mark.asyncio
async def test_a_queue_nobody_consumes_is_not_ready():
    queue = private_queue()
    connection = await aio_pika.connect(BROKER_URL)
    channel = await connection.channel()
    await channel.declare_queue(queue, durable=True)
    try:
        transport = transport_for(queue)
        with pytest.raises(GlinerWorkerUnavailableError, match="No GLiNER worker is consuming"):
            await transport.readiness()
        await transport.aclose()
    finally:
        await channel.queue_delete(queue)
        await connection.close()


@needs_broker
@pytest.mark.asyncio
async def test_wrong_credentials_are_unauthorized():
    from urllib.parse import urlsplit, urlunsplit

    split = urlsplit(BROKER_URL)
    host = split.netloc.rsplit("@", 1)[-1]
    bad = urlunsplit(split._replace(netloc=f"nobody:wrong@{host}"))
    transport = AmqpWorkerTransport(
        bad, queue="q", connect_timeout=5, request_timeout=5, max_response_bytes=1 << 20
    )
    with pytest.raises(GlinerWorkerUnauthorizedError):
        await transport.readiness()
