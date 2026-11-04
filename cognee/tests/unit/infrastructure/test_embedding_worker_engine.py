"""EMBEDDING_PROVIDER=embedding_worker: the engine, against a fake transport.

Batching, blank inputs, input type, token accounting, the retry and failure
scope rules shared with the remote GLiNER client (unavailable retried, timeouts
not resent, fatal errors latched), the worker's model info (vector size, input
limit, batch size, and the fallback for workers without it), the model and
vector-size checks, and splitting on ``input_too_long``. The RabbitMQ transport is tested in
``test_embedding_worker_amqp.py``; settings and factory wiring in
``test_embedding_worker_settings.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
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
from cognee.infrastructure.databases.vector.embeddings.embedding_worker.protocol import (
    EmbedReply,
    WorkerInfo,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingWorkerEmbeddingEngine import (
    READINESS_TTL_SECS,
    EmbeddingWorkerEmbeddingEngine,
    _gather_or_cancel,
    pool_vectors,
)
from cognee.infrastructure.remote_worker.retry import RetryPolicy

DIMS = 4
MODEL = "BAAI/bge-small-en-v1.5"
WORKER_MAX_INPUT_TOKENS = 512


def worker_info(
    *,
    model: str = MODEL,
    dimensions: int = DIMS,
    max_input_tokens: int = WORKER_MAX_INPUT_TOKENS,
    max_inputs: int = 256,
) -> WorkerInfo:
    return WorkerInfo(
        model=model,
        dimensions=dimensions,
        max_input_tokens=max_input_tokens,
        max_inputs=max_inputs,
        raw={"status": "ready", "model": model},
    )


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
        info: Callable[[], Any] = worker_info,
    ):
        self.respond = respond
        self.ready = ready
        self.info_outcome = info
        self.requests: list[tuple[dict, str]] = []
        self.probes = 0
        self.info_requests = 0
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

    async def info(self, *, request_id: str) -> WorkerInfo:
        self.info_requests += 1
        await asyncio.sleep(0)
        outcome = self.info_outcome()
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
    dimensions: int | None = DIMS,
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


def test_non_positive_dimensions_are_a_config_error():
    with pytest.raises(EmbeddingWorkerConfigError, match="EMBEDDING_DIMENSIONS"):
        make_engine(FakeTransport(), dimensions=0)


def test_a_batch_size_over_the_workers_default_limit_is_refused():
    with pytest.raises(EmbeddingWorkerConfigError, match="EMBEDDING_BATCH_SIZE=257"):
        make_engine(FakeTransport(), batch_size=257)
    assert make_engine(FakeTransport(), batch_size=256).get_batch_size() == 256


def test_query_input_type_warns_that_documents_are_embedded_as_queries(caplog):
    with caplog.at_level(logging.WARNING):
        make_engine(FakeTransport(), input_type="query")
    assert "documents included" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        make_engine(FakeTransport())
    assert "documents included" not in caplog.text


def test_mock_embedding_needs_configured_dimensions(monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    with pytest.raises(EmbeddingWorkerConfigError, match="MOCK_EMBEDDING"):
        make_engine(FakeTransport(), dimensions=None)


# --------------------------------------------------------------------------- #
# The worker's model info
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_info_is_fetched_once_after_the_first_probe_and_kept():
    clock = Clock()
    transport = FakeTransport()
    engine = make_engine(transport, clock=clock)
    await engine.embed_text(["a"])
    await engine.embed_text(["b"])
    assert (transport.probes, transport.info_requests) == (1, 1)
    assert engine.worker_info == worker_info()
    for _ in range(3):  # later probes do not ask again: info would queue behind the backlog
        clock.now += READINESS_TTL_SECS
        await engine.embed_text(["c"])
    assert (transport.probes, transport.info_requests) == (4, 1)


@pytest.mark.asyncio
async def test_info_is_fetched_again_when_a_latched_outage_is_released():
    state: dict[str, Any] = {"ready": None, "info": worker_info()}
    transport = FakeTransport(ready=lambda: state["ready"], info=lambda: state["info"])
    engine = make_engine(transport, dimensions=None, max_completion_tokens=None)
    await engine.embed_text(["a"])
    assert transport.info_requests == 1

    state["ready"] = EmbeddingWorkerUnavailableError("no consumer")
    with pytest.raises(EmbeddingWorkerUnavailableError):
        await engine._ready(fresh=True)  # the outage outlives its retries
    engine._latch(EmbeddingWorkerUnavailableError("no consumer"))

    # Another worker comes back, with other limits: its info is asked for.
    state["ready"] = None
    state["info"] = worker_info(max_inputs=1)
    await engine.embed_text(["b", "c"])
    assert transport.info_requests == 2
    assert engine.get_batch_size() == 1


@pytest.mark.asyncio
async def test_a_malformed_info_reply_fails_readiness():
    answers: list[Any] = [EmbeddingWorkerRuntimeError("info lacks dimensions")]
    transport = FakeTransport(info=lambda: answers.pop(0) if answers else worker_info())
    engine = make_engine(transport)
    with pytest.raises(EmbeddingWorkerRuntimeError, match="info"):
        await engine.embed_text(["x"])
    assert transport.requests == []
    # Not latched and not cached as ready: the next call probes and asks again.
    assert (await engine.embed_text(["y"]))[0] == vector_for("y")
    assert (transport.probes, transport.info_requests) == (2, 2)


@pytest.mark.asyncio
async def test_an_info_outage_that_outlives_its_retries_fails_readiness_and_is_latched():
    transport = FakeTransport(info=lambda: EmbeddingWorkerUnavailableError("loading"))
    engine = make_engine(transport)
    with pytest.raises(EmbeddingWorkerUnavailableError):
        await engine.embed_text(["x"])
    assert transport.info_requests == 3 and transport.requests == []
    assert isinstance(engine._failure, EmbeddingWorkerUnavailableError)


@pytest.mark.asyncio
async def test_unavailable_info_is_retried():
    sleeps: list[float] = []
    answers = [EmbeddingWorkerUnavailableError("loading")]
    transport = FakeTransport(info=lambda: answers.pop(0) if answers else worker_info())
    engine = make_engine(transport, sleeps=sleeps)
    await engine.embed_text(["a"])
    assert transport.info_requests == 2 and len(sleeps) == 1


@pytest.mark.asyncio
async def test_dimensions_come_from_the_worker_when_unset():
    transport = FakeTransport()
    engine = make_engine(transport, dimensions=None)
    assert engine.dimensions is None
    result = await engine.embed_text(["hello", " "])
    assert engine.get_vector_size() == DIMS
    assert result[1] == [0.0] * DIMS


@pytest.mark.asyncio
async def test_resolve_vector_size_asks_the_worker_on_the_loop():
    transport = FakeTransport()
    engine = make_engine(transport, dimensions=None)
    assert await engine.resolve_vector_size() == DIMS
    assert transport.info_requests == 1 and transport.requests == []


def test_get_vector_size_before_first_contact_fetches_info_synchronously():
    transport = FakeTransport()
    engine = make_engine(transport, dimensions=None)
    assert engine.get_vector_size() == DIMS
    assert engine.get_vector_size() == DIMS
    assert transport.info_requests == 1  # learned once


@pytest.mark.asyncio
async def test_get_vector_size_works_from_inside_a_running_loop():
    # The way vector adapters call it: sync, from async code.
    transport = FakeTransport()
    engine = make_engine(transport, dimensions=None)
    assert engine.get_vector_size() == DIMS
    assert (await engine.embed_text(["x"]))[0] == vector_for("x")


def test_get_vector_size_with_a_configured_size_never_asks():
    transport = FakeTransport()
    assert make_engine(transport).get_vector_size() == DIMS
    assert transport.info_requests == 0 and transport.probes == 0


def test_get_vector_size_against_a_dead_worker_raises_and_latches():
    transport = FakeTransport(ready=lambda: EmbeddingWorkerUnavailableError("down"))
    engine = make_engine(transport, dimensions=None)
    with pytest.raises(EmbeddingWorkerUnavailableError):
        engine.get_vector_size()
    assert isinstance(engine._failure, EmbeddingWorkerUnavailableError)


@pytest.mark.asyncio
async def test_configured_dimensions_that_differ_from_the_info_are_fatal():
    transport = FakeTransport(info=lambda: worker_info(dimensions=384))
    engine = make_engine(transport, dimensions=768)
    with pytest.raises(EmbeddingWorkerDimensionsMismatchError) as raised:
        await engine.embed_text(["x"])
    message = raised.value.message
    assert "384" in message and "768" in message and "EMBEDDING_DIMENSIONS" in message
    with pytest.raises(EmbeddingWorkerDimensionsMismatchError):
        await engine.embed_text(["y"])
    assert transport.requests == [] and transport.probes == 1  # latched, nothing sent


@pytest.mark.asyncio
async def test_a_worker_whose_size_changes_after_it_was_learned_is_fatal():
    # The info is not asked for again; the swapped worker's replies give it away.
    state = {"dimensions": DIMS}
    transport = FakeTransport(lambda payload: reply_for(payload, dimensions=state["dimensions"]))
    engine = make_engine(transport, dimensions=None)
    await engine.embed_text(["x"])
    state["dimensions"] = DIMS * 2
    with pytest.raises(EmbeddingWorkerDimensionsMismatchError, match="first learned"):
        await engine.embed_text(["y"])
    assert transport.info_requests == 1


@pytest.mark.asyncio
async def test_the_expected_model_is_checked_against_the_info():
    transport = FakeTransport(info=lambda: worker_info(model="other/model"))
    engine = make_engine(transport, expected_model=MODEL)
    with pytest.raises(EmbeddingWorkerModelMismatchError, match="other/model"):
        await engine.embed_text(["x"])
    with pytest.raises(EmbeddingWorkerModelMismatchError):
        await engine.embed_text(["x"])
    assert transport.requests == []


@pytest.mark.asyncio
async def test_requests_carry_at_most_the_workers_max_inputs():
    transport = FakeTransport(info=lambda: worker_info(max_inputs=2))
    engine = make_engine(transport, batch_size=3)
    assert engine.get_batch_size() == 3  # before info
    await engine.embed_text(["a", "b", "c", "d", "e"])
    assert [len(payload["inputs"]) for payload, _ in transport.requests] == [2, 2, 1]
    assert engine.get_batch_size() == 2


@pytest.mark.asyncio
async def test_a_smaller_batch_size_is_kept():
    transport = FakeTransport(info=lambda: worker_info(max_inputs=200))
    engine = make_engine(transport, batch_size=3)
    await engine.embed_text(["a"])
    assert engine.get_batch_size() == 3


@pytest.mark.parametrize(
    ("configured", "worker", "expected"),
    [
        (None, 512, 512),  # unset: the worker's
        (256, 512, 256),  # smaller config lowers it
        (512, 512, 512),
        (8192, 512, 512),  # larger config: warned, the worker's
    ],
)
@pytest.mark.asyncio
async def test_input_limit_is_the_workers_lowered_by_the_config(
    configured, worker, expected, caplog
):
    transport = FakeTransport(info=lambda: worker_info(max_input_tokens=worker))
    engine = make_engine(transport, max_completion_tokens=configured)
    with caplog.at_level(logging.WARNING):
        assert await engine.input_limit() == expected
    assert ("exceeds the 512 tokens" in caplog.text) == (configured == 8192)


@pytest.mark.asyncio
async def test_input_limit_without_info_is_the_config():
    unsupported = EmbeddingWorkerRejectedError("unknown message")
    assert (
        await make_engine(
            FakeTransport(info=lambda: unsupported), max_completion_tokens=512
        ).input_limit()
        == 512
    )
    assert await make_engine(FakeTransport(info=lambda: unsupported)).input_limit() is None


@pytest.mark.asyncio
async def test_a_worker_without_info_falls_back_to_the_config_and_warns_once(caplog):
    clock = Clock()
    transport = FakeTransport(info=lambda: EmbeddingWorkerRejectedError("invalid_request"))
    engine = make_engine(transport, batch_size=3, clock=clock)
    with caplog.at_level(logging.WARNING):
        await engine.embed_text(["a", "b", "c", "d"])
        clock.now += READINESS_TTL_SECS
        await engine.embed_text(["e"])
    assert caplog.text.count("does not report its model info") == 1
    assert transport.info_requests == 2  # no info cached: asked again on the next probe
    assert engine.worker_info is None and engine.get_batch_size() == 3
    assert [len(payload["inputs"]) for payload, _ in transport.requests] == [3, 1, 1]


@pytest.mark.asyncio
async def test_a_worker_without_info_needs_configured_dimensions():
    transport = FakeTransport(info=lambda: EmbeddingWorkerRejectedError("invalid_request"))
    engine = make_engine(transport, dimensions=None)
    with pytest.raises(EmbeddingWorkerConfigError, match="EMBEDDING_DIMENSIONS"):
        await engine.embed_text(["x"])
    assert transport.requests == []


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
    # The two overlapping 8-character parts, averaged and L2-normalized.
    assert result[1] == pytest.approx([8 / math.sqrt(65), 1 / math.sqrt(65), 0.0, 0.0])
    sent = [payload["inputs"] for payload, _ in transport.requests]
    assert sent[0] == ["short", long_text]  # rejected as a whole
    assert ["short"] in sent and ["abcdefgh"] in sent and ["efghijkl"] in sent
    assert result.tokens == 1 + 1 + 1


@pytest.mark.asyncio
async def test_an_input_too_short_to_split_fails():
    transport = FakeTransport(lambda _payload: EmbeddingWorkerInputTooLongError("too long"))
    with pytest.raises(EmbeddingContextWindowTooSmallError):
        await make_engine(transport).embed_text(["ab"])


@pytest.mark.asyncio
async def test_a_text_is_split_into_its_first_two_thirds_and_its_last_two_thirds():
    transport = FakeTransport(too_long_over(6))
    text = "abcdefghi"  # t = 3: "abcdef" and "defghi"
    await make_engine(transport).embed_text([text])
    sent = [payload["inputs"] for payload, _ in transport.requests]
    assert sent == [[text], ["abcdef"], ["defghi"]] or sent == [[text], ["defghi"], ["abcdef"]]


@pytest.mark.asyncio
async def test_parts_still_too_long_are_split_again_and_the_result_is_a_unit_vector():
    transport = FakeTransport(too_long_over(4))
    [vector] = await make_engine(transport).embed_text(["abcdefghijkl"])
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-9)
    # 12 -> 8 + 8; each 8 -> 4 (fits) + 6; each 6 -> 4 + 4: 1 + 2 * (1 + 1 + 1 + 2) requests.
    assert len(transport.requests) == 11
    fitted = [
        payload["inputs"][0] for payload, _ in transport.requests if len(payload["inputs"][0]) <= 4
    ]
    assert len(fitted) == 6


def test_pooled_vectors_are_unit_length():
    pooled = pool_vectors([1.0, 0.0], [0.0, 1.0])
    assert pooled == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    assert pool_vectors([0.0, 0.0], [0.0, 0.0]) == [0.0, 0.0]


@pytest.mark.asyncio
async def test_a_failing_half_cancels_its_sibling():
    sibling_cancelled = asyncio.Event()

    async def slow() -> int:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return 1

    async def fails() -> int:
        await asyncio.sleep(0)
        raise EmbeddingWorkerRuntimeError("boom")

    with pytest.raises(EmbeddingWorkerRuntimeError):
        await asyncio.wait_for(_gather_or_cancel([slow(), fails()]), timeout=2)
    assert sibling_cancelled.is_set()


@pytest.mark.asyncio
async def test_a_failing_part_of_a_split_text_stops_the_other_part():
    started: list[str] = []
    release = asyncio.Event()

    class SlowRight(FakeTransport):
        async def embed(self, payload: dict, *, request_id: str) -> EmbedReply:
            [text] = payload["inputs"]
            started.append(text)
            if len(text) > 8:
                raise EmbeddingWorkerInputTooLongError("too long")
            if text.startswith("a"):
                raise EmbeddingWorkerRuntimeError("left part failed")
            await release.wait()  # the right part would hang without cancellation
            return reply_for(payload)

    engine = make_engine(SlowRight())
    with pytest.raises(EmbeddingWorkerRuntimeError, match="left part failed"):
        await asyncio.wait_for(engine.embed_text(["abcdefghijkl"]), timeout=2)
    assert "efghijkl" in started
