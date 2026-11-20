"""Embedding worker RabbitMQ transport.

Reply mapping and decoding are unit-tested here. The broker tests need a
RabbitMQ and run when ``COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL`` is set (e.g.
``amqp://guest:guest@localhost:5672/%2f``); a fake worker on a private queue answers
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
    info_reply_to_result,
    reply_to_result,
)
from cognee.infrastructure.databases.vector.embeddings.embedding_worker.protocol import (  # noqa: E402
    embed_request,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (  # noqa: E402
    EmbeddingWorkerEmbeddingEngine,
)

BROKER_URL = os.environ.get("COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL")
# A user that may log in but has no permissions on the vhost, e.g. after
# ``rabbitmqctl add_user limited secret; rabbitmqctl set_permissions -p / limited "^$" "^$" "^$"``.
RESTRICTED_URL = os.environ.get("COGNEE_EMBEDDING_WORKER_AMQP_TEST_RESTRICTED_URL")
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


def info_body(*, model: str = "fake/model", dimensions: int = DIMS, **changes) -> bytes:
    """The worker's ``GET /v1/info`` body, as its RabbitMQ ``type=info`` reply."""
    data = {
        "status": "ready",
        "model": model,
        "dimensions": dimensions,
        "max_input_tokens": 64,
        "matryoshka": False,
        "asymmetric": True,
        "tokenizer": "fake",
        "pooling": "cls",
        "precision": "fp32",
        "limits": {"max_inputs": 8, "max_text_chars": 10000, "max_input_tokens": 64},
        "batching": {"max_batch_size": 32, "max_batch_tokens": 8192},
        "features": ["input_type", "truncate", "base64", "openai", "info"],
        "device": None,
    }
    data.update(changes)
    return json.dumps(data).encode()


# --------------------------------------------------------------------------- #
# Reply mapping
# --------------------------------------------------------------------------- #


def test_info_optional_fields_default_and_new_budgets_are_read():
    minimal = json.dumps({"model": "m", "dimensions": 3, "max_input_tokens": 512}).encode()
    info = info_reply_to_result(minimal, "label")
    assert info.max_inputs == 256
    assert info.special_tokens is None and info.max_text_tokens is None and info.tokenizer is None
    full = info_body(
        special_tokens=2,
        max_text_tokens={"document": 510, "query": 502},
        tokenizer="BAAI/bge-small-en-v1.5",
    )
    info = info_reply_to_result(full, "label")
    assert info.special_tokens == 2
    assert info.max_text_tokens == {"document": 510, "query": 502}
    assert info.tokenizer == "BAAI/bge-small-en-v1.5"


def test_input_too_long_lists_every_offending_input():
    body = json.dumps(
        {
            "error": {
                "code": "input_too_long",
                "message": "input 1 has 530 tokens",
                "inputs": [
                    {"index": 4, "chars": 250000, "tokens": None},
                    {"index": 1, "chars": 2410, "tokens": 530},
                ],
                "max_input_tokens": 512,
                "max_text_chars": 200000,
            }
        }
    ).encode()
    with pytest.raises(EmbeddingWorkerInputTooLongError) as raised:
        reply_to_result(body, "label", 5)
    assert raised.value.indices == [1, 4]
    assert raised.value.max_input_tokens == 512 and raised.value.max_text_chars == 200000


@pytest.mark.parametrize(
    "inputs",
    [None, [], [{"chars": 3}], [{"index": "1"}], [{"index": -1}], "1"],
)
def test_input_too_long_without_a_usable_list_has_no_indices(inputs):
    error = {"code": "input_too_long", "message": "input 0 has too many tokens"}
    if inputs is not None:
        error["inputs"] = inputs
    with pytest.raises(EmbeddingWorkerInputTooLongError) as raised:
        reply_to_result(json.dumps({"error": error}).encode(), "label", 2)
    assert raised.value.indices is None


def test_info_reply_is_the_v1_info_body():
    info = info_reply_to_result(info_body(), "label")
    assert (info.model, info.dimensions, info.max_input_tokens, info.max_inputs) == (
        "fake/model",
        DIMS,
        64,
        8,
    )
    assert info.raw["features"][-1] == "info"


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        info_body(dimensions=0),
        info_body(dimensions="3"),
        info_body(model=""),
        info_body(max_input_tokens=None),
        json.dumps({"model": "m", "dimensions": 3}).encode(),
        info_body(limits=[]),
        info_body(limits={"max_inputs": True}),
        info_body(limits={"max_inputs": 0}),
        info_body(special_tokens=-1),
        info_body(special_tokens="2"),
        info_body(max_text_tokens={"document": 0}),
        info_body(max_text_tokens=[510]),
        info_body(tokenizer=5),
    ],
)
def test_malformed_info_is_a_runtime_error(body):
    with pytest.raises(EmbeddingWorkerRuntimeError, match="info"):
        info_reply_to_result(body, "label")


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("invalid_request", EmbeddingWorkerRejectedError),  # a worker that predates info
        ("unavailable", EmbeddingWorkerUnavailableError),
    ],
)
def test_info_error_replies_map_to_their_class(code, error):
    body = json.dumps({"error": {"code": code, "message": "details"}}).encode()
    with pytest.raises(error, match="details"):
        info_reply_to_result(body, "label")


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
async def fake_worker(queue: str, respond, *, info: bytes | None = None):
    """Consume ``queue`` and answer each request with ``respond(message) -> [bodies]``;
    ``type=info`` messages get ``info`` (default: :func:`info_body`). Only the
    embed requests are recorded."""
    info = info if info is not None else info_body()
    connection = await aio_pika.connect(BROKER_URL)
    channel = await connection.channel()
    # Durable: RabbitMQ 4 refuses transient non-exclusive queues, and an exclusive
    # one would refuse the transport's passive declare.
    declared = await channel.declare_queue(queue, durable=True, auto_delete=True)
    seen: list = []

    async def on_message(message):
        async with message.process():
            if message.type == "info":
                bodies = [info]
            else:
                seen.append(message)
                bodies = respond(message)
            for body in bodies:
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
    assert message.type is None  # an embed request carries no type


@needs_broker
@pytest.mark.asyncio
async def test_info_is_a_typed_message_answered_with_the_info_body():
    queue = private_queue()
    received: list = []
    connection = await aio_pika.connect(BROKER_URL)
    channel = await connection.channel()
    declared = await channel.declare_queue(queue, durable=True, auto_delete=True)

    async def on_message(message):
        async with message.process():
            received.append(message)
            await channel.default_exchange.publish(
                aio_pika.Message(body=info_body(), correlation_id=message.correlation_id),
                routing_key=message.reply_to,
            )

    await declared.consume(on_message)
    try:
        transport = transport_for(queue)
        info = await transport.info(request_id="info-1")
        await transport.aclose()
    finally:
        await connection.close()
    [message] = received
    assert message.type == "info"
    assert json.loads(message.body) == {}
    assert message.headers["x-request-id"] == "info-1"
    assert message.reply_to.startswith("amq.rabbitmq.reply-to")
    assert info.dimensions == DIMS and info.max_inputs == 8


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
    accepts at most 20 characters per input. EMBEDDING_DIMENSIONS is unset: the
    size comes from the worker's info."""
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
            dimensions=None,
            batch_size=8,
            settings=EmbeddingWorkerSettings(endpoint=BROKER_URL, amqp_queue=queue),
        )
        try:
            result = await engine.embed_text(["short", "a" * 30])
        finally:
            await close_embedding_worker_transports()

    assert engine.get_vector_size() == DIMS
    assert result[0] == vector_for("short")
    # Two 20-character parts, averaged and scaled back to unit length.
    expected = np.asarray(vector_for("a" * 20))
    assert result[1] == pytest.approx((expected / np.linalg.norm(expected)).tolist())
    assert result.tokens == 7 * 2  # two successful requests
    # An older worker lists nothing: the batch, its halves, then the long text's
    # two parts in one request.
    assert [json.loads(message.body)["inputs"] for message in seen][-1] == ["a" * 20, "a" * 20]
    assert len(seen) == 4


@needs_broker
@pytest.mark.asyncio
async def test_a_worker_without_info_support_falls_back_to_the_config(monkeypatch):
    """An older worker parses the info message as an embed request and answers
    invalid_request: the engine uses EMBEDDING_DIMENSIONS and goes on."""
    monkeypatch.setattr(engine_module, "resolve_embedding_tokenizer", lambda **_: object())
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    old = json.dumps({"error": {"code": "invalid_request", "message": "inputs: missing"}})
    queue = private_queue()
    async with fake_worker(queue, answer, info=old.encode()):
        engine = EmbeddingWorkerEmbeddingEngine(
            model="fake/model",
            dimensions=DIMS,
            settings=EmbeddingWorkerSettings(endpoint=BROKER_URL, amqp_queue=queue),
        )
        try:
            result = await engine.embed_text(["x"])
        finally:
            await close_embedding_worker_transports()
    assert engine.worker_info is None
    assert result[0] == vector_for("x")


@needs_broker
@pytest.mark.skipif(
    not RESTRICTED_URL, reason="COGNEE_EMBEDDING_WORKER_AMQP_TEST_RESTRICTED_URL is not set"
)
@pytest.mark.asyncio
async def test_access_refused_on_the_queue_is_unauthorized_not_unavailable():
    queue = private_queue()
    async with fake_worker(queue, answer):
        transport = AmqpEmbeddingTransport(
            RESTRICTED_URL, queue=queue, connect_timeout=5, request_timeout=5
        )
        with pytest.raises(EmbeddingWorkerUnauthorizedError, match="ACCESS_REFUSED"):
            await transport.readiness()  # passive declare: configure access refused
        with pytest.raises(EmbeddingWorkerUnauthorizedError, match="ACCESS_REFUSED"):
            await transport.embed(PAYLOAD, request_id="r")  # publish: write access refused
        await transport.aclose()


@needs_broker
@pytest.mark.asyncio
async def test_the_engine_splits_exactly_the_listed_inputs_in_one_round_trip(monkeypatch):
    """A worker that lists its over-long inputs, as the current one does."""
    monkeypatch.setattr(engine_module, "resolve_embedding_tokenizer", lambda **_: object())
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)

    def limited(message) -> list[bytes]:
        payload = json.loads(message.body)
        listed = [
            {"index": i, "chars": len(text), "tokens": len(text)}
            for i, text in enumerate(payload["inputs"])
            if len(text) > 20
        ]
        if listed:
            error = {
                "code": "input_too_long",
                "message": f"input {listed[0]['index']} has too many tokens",
                "inputs": listed,
                "max_input_tokens": 20,
                "max_text_chars": 1000,
            }
            return [json.dumps({"error": error}).encode()]
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
            result = await engine.embed_text(["short", "a" * 30, "b" * 30])
        finally:
            await close_embedding_worker_transports()

    assert [json.loads(message.body)["inputs"] for message in seen] == [
        ["short", "a" * 30, "b" * 30],
        ["short", "a" * 20, "a" * 20, "b" * 20, "b" * 20],
    ]
    assert result[0] == vector_for("short")
    expected = np.asarray(vector_for("a" * 20))
    assert result[1] == pytest.approx((expected / np.linalg.norm(expected)).tolist())
