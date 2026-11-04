"""Embedding worker RabbitMQ transport.

Reply mapping and decoding are unit-tested here. The broker tests need a
RabbitMQ and run when ``COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL`` is set (e.g.
``amqp://guest:guest@localhost:5672/``); a fake worker on a private queue answers
them. The RPC core is shared with the remote GLiNER client, whose broker tests
cover it too (``test_gliner_remote_amqp.py``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import uuid

import numpy as np
import pytest

aio_pika = pytest.importorskip("aio_pika")

import cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine as engine_module  # noqa: E402
from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (  # noqa: E402
    EmbeddingWorkerInputTooLongError,
    EmbeddingWorkerRejectedError,
    EmbeddingWorkerRuntimeError,
    EmbeddingWorkerSettings,
    EmbeddingWorkerTimeoutError,
    EmbeddingWorkerUnauthorizedError,
    EmbeddingWorkerUnavailableError,
    close_embedding_worker_transports,
)
from cognee.infrastructure.databases.vector.embeddings.embedding_worker.amqp import (  # noqa: E402
    AmqpEmbeddingTransport,
    reply_to_result,
)
from cognee.infrastructure.databases.vector.embeddings.embedding_worker.protocol import (  # noqa: E402
    embed_request,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (  # noqa: E402
    EmbeddingWorkerEmbeddingEngine,
)

BROKER_URL = os.environ.get("COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL")
needs_broker = pytest.mark.skipif(
    not BROKER_URL, reason="COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL is not set"
)

DIMS = 3
PAYLOAD = embed_request(["Alice works for Acme.", "Bob lives in Berlin."], "document")


def encode(vector: list[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()


def vector_for(text: str) -> list[float]:
    return [float(len(text)), 0.5, -1.0]


def success_body(payload: dict, *, model: str = "fake/model", dimensions: int = DIMS) -> bytes:
    return json.dumps(
        {
            "embeddings": [encode(vector_for(text)) for text in payload["inputs"]],
            "model": model,
            "dimensions": dimensions,
            "encoding_format": "base64",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
            "truncated_inputs": 0,
            "elapsed_ms": 1.5,
        }
    ).encode()


# --------------------------------------------------------------------------- #
# Reply mapping
# --------------------------------------------------------------------------- #


def test_success_reply_is_the_v1_embed_body_with_base64_vectors():
    reply = reply_to_result(success_body(PAYLOAD), "label", 2)
    assert reply.vectors == [vector_for(text) for text in PAYLOAD["inputs"]]
    assert reply.model == "fake/model"
    assert reply.dimensions == DIMS
    assert reply.prompt_tokens == 7
    assert reply.elapsed_ms == 1.5


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("invalid_request", EmbeddingWorkerRejectedError),
        ("input_too_long", EmbeddingWorkerInputTooLongError),
        ("unavailable", EmbeddingWorkerUnavailableError),
        ("internal", EmbeddingWorkerRuntimeError),
        ("something_new", EmbeddingWorkerRuntimeError),
    ],
)
def test_error_replies_map_to_their_class(code, error):
    body = json.dumps({"error": {"code": code, "message": "details"}}).encode()
    with pytest.raises(error, match="details") as raised:
        reply_to_result(body, "label", 2)
    assert type(raised.value) is error


def _mutated(**changes) -> bytes:
    data = json.loads(success_body(PAYLOAD))
    data.update(changes)
    return json.dumps(data).encode()


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"not json", "invalid JSON"),
        (b"[]", "not a JSON object"),
        (_mutated(usage={}), "usage.prompt_tokens"),
        (_mutated(dimensions="3"), "dimensions"),
        (_mutated(embeddings=[encode([1, 2, 3])]), "1 vectors for 2 inputs"),
        (_mutated(embeddings=[encode([1, 2]), encode([1, 2, 3])]), "2-component"),
        (_mutated(embeddings=["!!", encode([1, 2, 3])]), "float32"),
        (_mutated(embeddings=[[1.0, 2.0, 3.0], encode([1, 2, 3])]), "not base64"),
        (_mutated(embeddings=[encode([1, float("nan"), 3])] * 2), "NaN"),
        (_mutated(encoding_format="float"), "base64"),
    ],
)
def test_malformed_replies_are_runtime_errors(body, match):
    with pytest.raises(EmbeddingWorkerRuntimeError, match=match):
        reply_to_result(body, "label", 2)


def test_credentials_are_masked_in_the_label():
    transport = AmqpEmbeddingTransport(
        "amqp://guest:s3cret@broker:5672/%2f",
        queue="embedding_worker.embed",
        connect_timeout=1,
        request_timeout=1,
    )
    assert "s3cret" not in transport.endpoint_label
    assert transport.endpoint_label == "amqp://***@broker:5672/%2f queue embedding_worker.embed"


@pytest.mark.asyncio
async def test_a_refused_connection_is_unavailable():
    transport = AmqpEmbeddingTransport(
        "amqp://guest:guest@127.0.0.1:1/", queue="q", connect_timeout=2, request_timeout=2
    )
    with pytest.raises(EmbeddingWorkerUnavailableError):
        await transport.readiness()
    with pytest.raises(EmbeddingWorkerUnavailableError):
        await transport.embed(PAYLOAD, request_id="r")


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


def answer(message) -> list[bytes]:
    return [success_body(json.loads(message.body))]


def transport_for(queue: str, *, request_timeout: float = 5.0) -> AmqpEmbeddingTransport:
    return AmqpEmbeddingTransport(
        BROKER_URL, queue=queue, connect_timeout=5, request_timeout=request_timeout
    )


def private_queue() -> str:
    return f"cognee-test-{uuid.uuid4().hex}"


@needs_broker
@pytest.mark.asyncio
async def test_request_properties_and_reply_routing():
    queue = private_queue()
    async with fake_worker(queue, answer) as seen:
        transport = transport_for(queue)
        await transport.readiness()
        reply = await transport.embed(PAYLOAD, request_id="req-1")
        await transport.aclose()

    [message] = seen
    assert json.loads(message.body) == PAYLOAD
    assert message.headers["x-request-id"] == "req-1"
    # The broker rewrites the pseudo-queue name into a per-channel one.
    assert message.reply_to.startswith("amq.rabbitmq.reply-to")
    assert message.content_type == "application/json"
    assert message.expiration is not None
    assert reply.vectors == [vector_for(text) for text in PAYLOAD["inputs"]]


@needs_broker
@pytest.mark.asyncio
async def test_concurrent_requests_get_their_own_replies():
    queue = private_queue()
    async with fake_worker(queue, answer):
        transport = transport_for(queue)
        texts = [f"text number {'x' * i}" for i in range(8)]
        replies = await asyncio.gather(
            *(
                transport.embed(embed_request([text], "document"), request_id=str(i))
                for i, text in enumerate(texts)
            )
        )
        await transport.aclose()
    assert [reply.vectors[0] for reply in replies] == [vector_for(text) for text in texts]


@needs_broker
@pytest.mark.asyncio
async def test_duplicate_replies_are_dropped():
    queue = private_queue()
    async with fake_worker(queue, lambda message: answer(message) * 2):
        transport = transport_for(queue)
        first = await transport.embed(PAYLOAD, request_id="a")
        second = await transport.embed(PAYLOAD, request_id="b")
        await transport.aclose()
    assert first.vectors == second.vectors


@needs_broker
@pytest.mark.asyncio
async def test_error_replies_raise_their_class():
    queue = private_queue()
    error = json.dumps({"error": {"code": "input_too_long", "message": "input 0"}}).encode()
    async with fake_worker(queue, lambda _m: [error]):
        transport = transport_for(queue)
        with pytest.raises(EmbeddingWorkerInputTooLongError, match="input 0"):
            await transport.embed(PAYLOAD, request_id="r")
        await transport.aclose()


@needs_broker
@pytest.mark.asyncio
async def test_no_reply_to_an_accepted_request_times_out():
    queue = private_queue()
    async with fake_worker(queue, lambda _m: []):
        transport = transport_for(queue, request_timeout=0.5)
        with pytest.raises(EmbeddingWorkerTimeoutError, match="No reply from the embedding worker"):
            await transport.embed(PAYLOAD, request_id="r")
        await transport.aclose()


@needs_broker
@pytest.mark.asyncio
async def test_a_missing_queue_is_unavailable_for_readiness_and_requests():
    transport = transport_for(private_queue())
    with pytest.raises(EmbeddingWorkerUnavailableError, match="No queue"):
        await transport.readiness()
    with pytest.raises(EmbeddingWorkerUnavailableError, match="could not route"):
        await transport.embed(PAYLOAD, request_id="r")
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
        with pytest.raises(
            EmbeddingWorkerUnavailableError, match="No embedding worker is consuming"
        ):
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
    transport = AmqpEmbeddingTransport(bad, queue="q", connect_timeout=5, request_timeout=5)
    with pytest.raises(EmbeddingWorkerUnauthorizedError):
        await transport.readiness()


@needs_broker
@pytest.mark.asyncio
async def test_the_engine_splits_over_long_inputs_through_the_broker(monkeypatch):
    """The settings-built, shared transport end to end, with a worker whose model
    accepts at most 20 characters per input."""
    monkeypatch.setattr(engine_module, "resolve_embedding_tokenizer", lambda **_: object())
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)

    def limited(message) -> list[bytes]:
        payload = json.loads(message.body)
        if any(len(text) > 20 for text in payload["inputs"]):
            error = {"error": {"code": "input_too_long", "message": "too many tokens"}}
            return [json.dumps(error).encode()]
        return [success_body(payload)]

    queue = private_queue()
    async with fake_worker(queue, limited) as seen:
        engine = EmbeddingWorkerEmbeddingEngine(
            model="fake/model",
            dimensions=DIMS,
            batch_size=8,
            settings=EmbeddingWorkerSettings(endpoint=BROKER_URL, amqp_queue=queue),
        )
        try:
            result = await engine.embed_text(["short", "a" * 30])
        finally:
            await close_embedding_worker_transports()

    assert result[0] == vector_for("short")
    assert result[1] == vector_for("a" * 20)  # two 20-character parts, averaged
    assert result.tokens == 7 * 3  # three successful requests
    assert len(seen) == 5  # the batch, its halves, and the two parts of the long text
