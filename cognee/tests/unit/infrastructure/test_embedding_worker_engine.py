"""EMBEDDING_PROVIDER=embedding_worker: the engine, against a fake transport.

Batching, blank inputs, input type, token accounting, the retry and failure
scope rules shared with the remote GLiNER client (unavailable retried, timeouts
not resent, fatal errors latched), the model and vector-size checks, and
splitting on ``input_too_long``. The RabbitMQ transport is tested in
``test_embedding_worker_amqp.py``; settings and factory wiring in
``test_embedding_worker_settings.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

import cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine as engine_module
from cognee.infrastructure.databases.exceptions import (
    EmbeddingContextWindowTooSmallError,
    EmbeddingException,
)
from cognee.infrastructure.databases.vector.embeddings.embedding_worker import (
    EmbeddingWorkerConfigError,
    EmbeddingWorkerDimensionsMismatchError,
    EmbeddingWorkerInputTooLongError,
    EmbeddingWorkerModelMismatchError,
    EmbeddingWorkerRejectedError,
    EmbeddingWorkerRuntimeError,
    EmbeddingWorkerSettings,
    EmbeddingWorkerTimeoutError,
    EmbeddingWorkerUnauthorizedError,
    EmbeddingWorkerUnavailableError,
)
from cognee.infrastructure.databases.vector.embeddings.embedding_worker.protocol import EmbedReply
from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (
    READINESS_TTL_SECS,
    EmbeddingWorkerEmbeddingEngine,
)
from cognee.infrastructure.remote_worker.retry import RetryPolicy

DIMS = 4
MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    monkeypatch.setattr(engine_module, "resolve_embedding_tokenizer", lambda **_: object())
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)


def vector_for(text: str) -> list[float]:
    return [float(len(text)), 1.0, 0.0, 0.0]


def reply_for(payload: dict, *, model: str = MODEL, dimensions: int = DIMS) -> EmbedReply:
    return EmbedReply(
        vectors=[vector_for(text) for text in payload["inputs"]],
        model=model,
        dimensions=dimensions,
        prompt_tokens=sum(len(text.split()) for text in payload["inputs"]),
        elapsed_ms=1.0,
    )


class FakeTransport:
    name = "fake"
    endpoint_label = "fake://worker queue q"

    def __init__(
        self,
        respond: Callable[[dict], Any] = reply_for,
        ready: Callable[[], None] = lambda: None,
    ):
        self.respond = respond
        self.ready = ready
        self.requests: list[tuple[dict, str]] = []
        self.probes = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def embed(self, payload: dict, *, request_id: str) -> EmbedReply:
        self.requests.append((payload, request_id))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            outcome = self.respond(payload)
        finally:
            self.in_flight -= 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def readiness(self) -> None:
        self.probes += 1
        outcome = self.ready()
        if isinstance(outcome, BaseException):
            raise outcome

    async def aclose(self) -> None:
        pass


def outcomes(*items: Any) -> Callable[[dict], Any]:
    """Respond with ``items`` in turn, then succeed."""
    queue = list(items)

    def respond(payload: dict) -> Any:
        return queue.pop(0) if queue else reply_for(payload)

    return respond


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_engine(
    transport: FakeTransport,
    *,
    batch_size: int = 2,
    input_type: str | None = None,
    dimensions: int = DIMS,
    max_completion_tokens: int | None = None,
    clock: Callable[[], float] | None = None,
    sleeps: list[float] | None = None,
    **settings: Any,
) -> EmbeddingWorkerEmbeddingEngine:
    async def sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    return EmbeddingWorkerEmbeddingEngine(
        model=MODEL,
        dimensions=dimensions,
        max_completion_tokens=max_completion_tokens,
        batch_size=batch_size,
        input_type=input_type,
        settings=EmbeddingWorkerSettings(endpoint="amqp://broker", **settings),
        transport=transport,
        retry=RetryPolicy(attempts=3, base_delay=0.01, max_delay=0.02),
        sleep=sleep,
        clock=clock or Clock(),
    )


# --------------------------------------------------------------------------- #
# Requests and replies
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_texts_go_out_in_batches_and_come_back_in_order():
    transport = FakeTransport()
    engine = make_engine(transport, batch_size=2)
    texts = ["a b", "ccc", "dddd e", "f", "gg hh ii"]

    result = await engine.embed_text(texts)

    assert [payload["inputs"] for payload, _ in transport.requests] == [
        ["a b", "ccc"],
        ["dddd e", "f"],
        ["gg hh ii"],
    ]
    assert list(result) == [vector_for(text) for text in texts]
    assert result.tokens == 2 + 1 + 2 + 1 + 3  # usage.prompt_tokens summed over requests
    payload = transport.requests[0][0]
    assert payload == {
        "inputs": ["a b", "ccc"],
        "input_type": "document",
        "encoding_format": "base64",
    }  # no dimensions, normalize or truncate: the worker's defaults apply
    assert engine.served_model == MODEL


@pytest.mark.asyncio
async def test_blank_texts_are_never_sent_blank_and_get_zero_vectors():
    transport = FakeTransport()
    engine = make_engine(transport, batch_size=8)

    result = await engine.embed_text(["hello", "   ", ""])

    [(payload, _)] = transport.requests
    assert all(text.strip() for text in payload["inputs"])
    assert result[0] == vector_for("hello")
    assert result[1] == result[2] == [0.0] * DIMS


@pytest.mark.parametrize(
    ("configured", "sent"),
    [(None, "document"), ("query", "query"), ("QUERY", "query"), ("passage", "document")],
)
@pytest.mark.asyncio
async def test_input_type_is_document_unless_configured_as_query(configured, sent):
    transport = FakeTransport()
    await make_engine(transport, input_type=configured).embed_text(["x"])
    assert transport.requests[0][0]["input_type"] == sent


@pytest.mark.asyncio
async def test_mock_embedding_sends_nothing(monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    transport = FakeTransport()
    result = await make_engine(transport).embed_text(["a", "b"])
    assert list(result) == [[0.0] * DIMS] * 2 and result.tokens == 0
    assert transport.requests == [] and transport.probes == 0


@pytest.mark.asyncio
async def test_requests_share_the_concurrency_budget():
    transport = FakeTransport()
    engine = make_engine(transport, batch_size=1, max_concurrent_requests=2)
    await engine.embed_text([f"text {i}" for i in range(10)])
    assert len(transport.requests) == 10
    assert transport.max_in_flight <= 2


def test_dimensions_are_required():
    with pytest.raises(EmbeddingWorkerConfigError, match="EMBEDDING_DIMENSIONS"):
        make_engine(FakeTransport(), dimensions=0)


@pytest.mark.asyncio
async def test_input_limit_comes_from_the_config():
    assert await make_engine(FakeTransport(), max_completion_tokens=512).input_limit() == 512
    assert await make_engine(FakeTransport()).input_limit() is None


def test_errors_are_embedding_exceptions():
    # The API and the pipelines handle them like every other embedding failure.
    assert issubclass(EmbeddingWorkerUnavailableError, EmbeddingException)
    assert issubclass(EmbeddingWorkerInputTooLongError, EmbeddingWorkerRejectedError)


# --------------------------------------------------------------------------- #
# Retries and failure scope
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unavailable_is_retried_with_the_same_request_id():
    sleeps: list[float] = []
    transport = FakeTransport(
        outcomes(EmbeddingWorkerUnavailableError("down"), EmbeddingWorkerUnavailableError("down"))
    )
    result = await make_engine(transport, sleeps=sleeps).embed_text(["x"])
    assert result[0] == vector_for("x")
    assert len(transport.requests) == 3
    assert len({request_id for _, request_id in transport.requests}) == 1
    assert len(sleeps) == 2


@pytest.mark.parametrize(
    "error",
    [EmbeddingWorkerRejectedError("bad"), EmbeddingWorkerRuntimeError("boom")],
)
@pytest.mark.asyncio
async def test_rejected_and_runtime_errors_fail_the_call_only(error):
    transport = FakeTransport(outcomes(error))
    engine = make_engine(transport)
    with pytest.raises(type(error)):
        await engine.embed_text(["x"])
    assert len(transport.requests) == 1  # not retried
    # Not latched: the next call goes through.
    assert (await engine.embed_text(["y"]))[0] == vector_for("y")


@pytest.mark.asyncio
async def test_a_timeout_is_not_resent_and_rechecks_readiness():
    transport = FakeTransport(outcomes(EmbeddingWorkerTimeoutError("slow")))
    engine = make_engine(transport)
    with pytest.raises(EmbeddingWorkerTimeoutError):
        await engine.embed_text(["x"])
    assert len(transport.requests) == 1
    assert transport.probes == 2  # the first readiness check, then a fresh one
    assert (await engine.embed_text(["y"]))[0] == vector_for("y")


@pytest.mark.asyncio
async def test_a_timeout_from_a_worker_that_went_away_is_an_outage():
    state: dict[str, Any] = {"ready": None}

    def worker_dies(_payload: dict) -> Any:
        state["ready"] = EmbeddingWorkerUnavailableError("no consumer")
        return EmbeddingWorkerTimeoutError("slow")

    transport = FakeTransport(worker_dies, ready=lambda: state["ready"])
    with pytest.raises(EmbeddingWorkerUnavailableError):
        await make_engine(transport).embed_text(["x"])
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_an_outage_is_latched_until_a_worker_is_back():
    sleeps: list[float] = []
    down = EmbeddingWorkerUnavailableError("no consumer")
    state = {"ready": down}
    transport = FakeTransport(ready=lambda: state["ready"])
    engine = make_engine(transport, sleeps=sleeps)

    with pytest.raises(EmbeddingWorkerUnavailableError):
        await engine.embed_text(["x"])
    assert transport.probes == 3  # every retry spent

    # Latched: one probe, no retries, no request.
    sleeps.clear()
    with pytest.raises(EmbeddingWorkerUnavailableError):
        await engine.embed_text(["x"])
    assert transport.probes == 4 and sleeps == [] and transport.requests == []

    # A worker consumes again: the latch is released.
    state["ready"] = None
    assert (await engine.embed_text(["x"]))[0] == vector_for("x")


@pytest.mark.asyncio
async def test_refused_credentials_are_latched():
    transport = FakeTransport(ready=lambda: EmbeddingWorkerUnauthorizedError("refused"))
    engine = make_engine(transport)
    with pytest.raises(EmbeddingWorkerUnauthorizedError):
        await engine.embed_text(["x"])
    with pytest.raises(EmbeddingWorkerUnauthorizedError):
        await engine.embed_text(["x"])
    assert transport.probes == 1


@pytest.mark.asyncio
async def test_the_wrong_model_is_fatal_and_latched():
    transport = FakeTransport(lambda payload: reply_for(payload, model="other/model"))
    engine = make_engine(transport, expected_model=MODEL)
    with pytest.raises(EmbeddingWorkerModelMismatchError, match="other/model"):
        await engine.embed_text(["x"])
    with pytest.raises(EmbeddingWorkerModelMismatchError):
        await engine.embed_text(["y"])
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_any_model_is_accepted_without_an_expected_one():
    transport = FakeTransport(lambda payload: reply_for(payload, model="other/model"))
    engine = make_engine(transport)
    await engine.embed_text(["x"])
    assert engine.served_model == "other/model"


@pytest.mark.asyncio
async def test_a_vector_size_other_than_embedding_dimensions_is_fatal():
    transport = FakeTransport(lambda payload: reply_for(payload, dimensions=DIMS * 2))
    engine = make_engine(transport)
    with pytest.raises(EmbeddingWorkerDimensionsMismatchError, match="EMBEDDING_DIMENSIONS"):
        await engine.embed_text(["x"])
    with pytest.raises(EmbeddingWorkerDimensionsMismatchError):
        await engine.embed_text(["y"])
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_readiness_is_cached_for_its_ttl():
    clock = Clock()
    transport = FakeTransport()
    engine = make_engine(transport, clock=clock)
    await engine.embed_text(["a"])
    await engine.embed_text(["b"])
    assert transport.probes == 1
    clock.now += READINESS_TTL_SECS
    await engine.embed_text(["c"])
    assert transport.probes == 2


@pytest.mark.asyncio
async def test_concurrent_calls_share_one_readiness_probe():
    transport = FakeTransport()
    engine = make_engine(transport)
    await asyncio.gather(*(engine.embed_text([f"t{i}"]) for i in range(5)))
    assert transport.probes == 1


# --------------------------------------------------------------------------- #
# input_too_long: split and pool
# --------------------------------------------------------------------------- #


def too_long_over(limit: int) -> Callable[[dict], Any]:
    """A worker whose model accepts inputs of at most ``limit`` characters."""

    def respond(payload: dict) -> Any:
        if any(len(text) > limit for text in payload["inputs"]):
            return EmbeddingWorkerInputTooLongError("input 0 has too many tokens")
        return reply_for(payload)

    return respond


@pytest.mark.asyncio
async def test_input_too_long_splits_the_request_and_pools_the_long_text():
    transport = FakeTransport(too_long_over(10))
    engine = make_engine(transport, batch_size=4)
    long_text = "abcdefghijkl"  # 12 characters: thirds of 4, parts of 8

    result = await engine.embed_text(["short", long_text])

    assert result[0] == vector_for("short")
    # The two overlapping 8-character parts, averaged.
    assert result[1] == [8.0, 1.0, 0.0, 0.0]
    sent = [payload["inputs"] for payload, _ in transport.requests]
    assert sent[0] == ["short", long_text]  # rejected as a whole
    assert ["short"] in sent and ["abcdefgh"] in sent and ["efghijkl"] in sent
    assert result.tokens == 1 + 1 + 1


@pytest.mark.asyncio
async def test_an_input_too_short_to_split_fails():
    transport = FakeTransport(lambda _payload: EmbeddingWorkerInputTooLongError("too long"))
    with pytest.raises(EmbeddingContextWindowTooSmallError):
        await make_engine(transport).embed_text(["ab"])
