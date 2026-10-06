"""Remote GLiNER adapter (SDK-980): batching, retries, readiness, limits, failure scope.

Deterministic: a scripted in-memory transport stands in for the worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from cognee.tasks.graph.gliner_demo.remote import (
    GlinerRemoteConfigError,
    GlinerWorkerIncompatibleError,
    GlinerWorkerModelMismatchError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
    RemoteGlinerAdapter,
    RemoteGlinerSettings,
)
from cognee.tasks.graph.gliner_demo.remote.protocol import (
    WorkerLimits,
    WorkerReadiness,
    WorkerReply,
)
from cognee.tasks.graph.gliner_demo.remote.retry import RetryPolicy
from cognee.tasks.graph.gliner_demo.schema import GlinerSchema

SCHEMA = GlinerSchema({"person": "", "organization": "A company"}, {"works_for": ""}, "caller")
MODEL = "fastino/gliner2.5-base-v1"
READY = WorkerReadiness(
    model=MODEL,
    limits=WorkerLimits(max_inputs=128, max_text_chars=200_000, max_batch_size=256),
    features=frozenset({"windowing"}),
)


def extraction_for(text: str) -> dict[str, Any]:
    return {"entities": {"person": [{"text": text.split()[0], "start": 0, "end": 1}]}}


class FakeTransport:
    """Answers each input with ``extraction_for(text)``; ``script`` can override a call."""

    name = "fake"
    endpoint_label = "fake://worker"

    def __init__(
        self,
        *,
        readiness: WorkerReadiness = READY,
        script: list[Callable[[dict], Any]] | None = None,
        windowed: bool = True,
        model: str = MODEL,
        delay: float = 0.0,
    ):
        self._readiness = readiness
        self.script = list(script or [])
        self.windowed = windowed
        self.model = model
        self.delay = delay
        self.payloads: list[dict] = []
        self.request_ids: list[str] = []
        self.readiness_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def extract(self, payload, *, request_id):
        self.payloads.append(payload)
        self.request_ids.append(request_id)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.script:
                outcome = self.script.pop(0)(payload)
                if outcome is not None:
                    return outcome
            return WorkerReply(
                items=[(item["id"], extraction_for(item["text"])) for item in payload["inputs"]],
                model=self.model,
                elapsed_ms=1.0,
                windowed=self.windowed and "window_words" in payload["options"],
            )
        finally:
            self.in_flight -= 1

    async def readiness(self):
        self.readiness_calls += 1
        return self._readiness

    async def aclose(self):
        pass


def raise_(error: Exception) -> Callable[[dict], Any]:
    def step(_payload):
        raise error

    return step


def make_adapter(transport, sleeps: list[float] | None = None, clock=None, **settings):
    async def fake_sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    kwargs = {"clock": clock} if clock is not None else {}
    return RemoteGlinerAdapter(
        RemoteGlinerSettings(transport="http", endpoint="http://worker:8080", **settings),
        transport=transport,
        retry=RetryPolicy(attempts=4, base_delay=0.25, max_delay=8.0),
        sleep=fake_sleep,
        **kwargs,
    )


async def extract(adapter, texts, **overrides):
    options = {"threshold": 0.5, "batch_size": 16, "window_words": 384, "window_overlap_words": 64}
    options.update(overrides)
    return await adapter.extract_batch(texts, SCHEMA, **options)


# --------------------------------------------------------------------------- #
# Requests and results
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_results_come_back_in_input_order_across_requests():
    transport = FakeTransport()
    adapter = make_adapter(transport, inputs_per_request=2)
    texts = [f"Name{i} works here." for i in range(5)]

    results = await extract(adapter, texts)

    assert results == [extraction_for(text) for text in texts]
    assert [len(payload["inputs"]) for payload in transport.payloads] == [2, 2, 1]
    assert adapter.model == MODEL


@pytest.mark.asyncio
async def test_payload_carries_schema_descriptions_verbatim_and_local_options():
    transport = FakeTransport()
    adapter = make_adapter(transport)

    await extract(adapter, ["Alice works for Acme."], threshold=0.4, batch_size=8)

    [payload] = transport.payloads
    assert payload["inputs"] == [{"id": "0", "text": "Alice works for Acme."}]
    # "" stays "": the worker prompts with it exactly as the local runtime does.
    assert payload["schema"] == {
        "entities": {"person": "", "organization": "A company"},
        "relations": {"works_for": ""},
    }
    assert payload["options"] == {
        "threshold": 0.4,
        "batch_size": 8,
        "include_confidence": True,
        "include_spans": True,
        "overlap_policy": "longest",
        "window_words": 384,
        "window_overlap_words": 64,
    }


@pytest.mark.asyncio
async def test_blank_chunks_get_empty_results_without_a_request():
    transport = FakeTransport()
    adapter = make_adapter(transport)

    results = await extract(adapter, ["  ", "Bob joined.", "\n"])

    assert results == [{}, extraction_for("Bob joined."), {}]
    assert [item["text"] for item in transport.payloads[0]["inputs"]] == ["Bob joined."]

    transport.payloads.clear()
    assert await extract(adapter, ["", " "]) == [{}, {}]
    assert transport.payloads == []


@pytest.mark.asyncio
async def test_empty_schema_or_no_texts_send_nothing():
    transport = FakeTransport()
    adapter = make_adapter(transport)

    assert await adapter.extract_batch(
        ["text"],
        GlinerSchema(),
        threshold=0.5,
        batch_size=16,
        window_words=384,
        window_overlap_words=64,
    ) == [{}]
    assert await extract(adapter, []) == []
    assert transport.payloads == [] and transport.readiness_calls == 0


@pytest.mark.asyncio
async def test_probe_is_one_unwindowed_spanless_request():
    transport = FakeTransport()
    adapter = make_adapter(transport)

    result = await adapter.extract_once("Alice works for Acme.", SCHEMA, threshold=0.3)

    assert result == extraction_for("Alice works for Acme.")
    [payload] = transport.payloads
    assert payload["options"] == {
        "threshold": 0.3,
        "batch_size": 1,
        "include_confidence": False,
        "include_spans": False,
        "overlap_policy": "longest",
    }
    assert await adapter.extract_once("   ", SCHEMA, threshold=0.3) == {}
    assert len(transport.payloads) == 1


@pytest.mark.asyncio
async def test_in_flight_requests_are_bounded_per_call_and_per_worker():
    transport = FakeTransport(delay=0.01)
    adapter = make_adapter(
        transport, inputs_per_request=1, max_in_flight_requests=3, max_concurrent_requests=8
    )
    await extract(adapter, [f"Name{i} here." for i in range(10)])
    assert transport.max_in_flight == 3

    transport = FakeTransport(delay=0.01)
    adapter = make_adapter(
        transport, inputs_per_request=1, max_in_flight_requests=8, max_concurrent_requests=2
    )
    await extract(adapter, [f"Name{i} here." for i in range(10)])
    assert transport.max_in_flight == 2


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unavailable_is_retried_under_one_request_id_with_backoff():
    sleeps: list[float] = []
    transport = FakeTransport(
        script=[
            raise_(GlinerWorkerUnavailableError("503")),
            raise_(GlinerWorkerUnavailableError("503", retry_after=5.0)),
        ]
    )
    adapter = make_adapter(transport, sleeps=sleeps)

    results = await extract(adapter, ["Alice works."])

    assert results == [extraction_for("Alice works.")]
    assert len(transport.request_ids) == 3 and len(set(transport.request_ids)) == 1
    assert 0.125 <= sleeps[0] <= 0.25
    assert sleeps[1] == 5.0  # Retry-After asks for longer than the backoff


@pytest.mark.asyncio
async def test_retry_after_is_capped():
    assert RetryPolicy(max_delay=8.0).delay(0, retry_after=60.0) == 8.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        GlinerWorkerRejectedError("422"),
        GlinerWorkerRuntimeError("500"),
        GlinerWorkerUnauthorizedError("401"),
    ],
)
async def test_non_transient_failures_are_not_retried(error):
    transport = FakeTransport(script=[raise_(error)])
    adapter = make_adapter(transport, sleeps=[])

    with pytest.raises(type(error)):
        await extract(adapter, ["Alice works."])
    assert len(transport.payloads) == 1


# --------------------------------------------------------------------------- #
# Failure scope
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_outage_that_outlives_the_retries_fails_the_rest_of_the_run_at_once():
    transport = FakeTransport(script=[raise_(GlinerWorkerUnavailableError("down"))] * 4)
    adapter = make_adapter(transport, sleeps=[])

    with pytest.raises(GlinerWorkerUnavailableError):
        await extract(adapter, ["Alice works."])
    assert len(transport.payloads) == 4

    # The next document fails immediately, without another round of retries.
    with pytest.raises(GlinerWorkerUnavailableError):
        await extract(adapter, ["Bob works."])
    with pytest.raises(GlinerWorkerUnavailableError):
        await adapter.extract_once("Bob works.", SCHEMA, threshold=0.5)
    assert len(transport.payloads) == 4


@pytest.mark.asyncio
async def test_a_rejected_document_does_not_fail_the_next_one():
    transport = FakeTransport(script=[raise_(GlinerWorkerRejectedError("422"))])
    adapter = make_adapter(transport)

    with pytest.raises(GlinerWorkerRejectedError):
        await extract(adapter, ["Alice works."])
    assert await extract(adapter, ["Bob works."]) == [extraction_for("Bob works.")]


@pytest.mark.asyncio
async def test_one_failing_request_cancels_its_siblings():
    calls = 0

    def fail_first(_payload):
        nonlocal calls
        calls += 1
        raise GlinerWorkerRuntimeError("boom")

    transport = FakeTransport(script=[fail_first], delay=0.01)
    adapter = make_adapter(transport, inputs_per_request=1, max_in_flight_requests=1)

    with pytest.raises(GlinerWorkerRuntimeError):
        await extract(adapter, [f"Name{i} here." for i in range(5)])
    # In-flight limit 1: the failure cancels the waiting requests. At most the one
    # that took the freed slot before the cancellation landed was sent.
    assert len(transport.payloads) <= 2


# --------------------------------------------------------------------------- #
# Replies
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items",
    [
        [],  # wrong count
        [("0", {}), ("0", {})],  # duplicate id
        [("7", {})],  # unknown id
    ],
)
async def test_malformed_replies_fail_the_document(items):
    texts = ["Alice works.", "Bob works."] if len(items) == 2 else ["Alice works."]
    reply = WorkerReply(items=items, model=MODEL, elapsed_ms=None, windowed=True)
    transport = FakeTransport(script=[lambda _payload: reply])
    adapter = make_adapter(transport)

    with pytest.raises(GlinerWorkerRuntimeError):
        await extract(adapter, texts)
    # Scoped to the document: the next one goes through.
    assert await extract(adapter, ["Carol works."]) == [extraction_for("Carol works.")]


@pytest.mark.asyncio
async def test_a_worker_that_ignores_window_words_is_incompatible():
    transport = FakeTransport(windowed=False, readiness=WorkerReadiness(model=MODEL))
    adapter = make_adapter(transport)

    with pytest.raises(GlinerWorkerIncompatibleError, match="ignored window_words"):
        await extract(adapter, ["Alice works."])


@pytest.mark.asyncio
async def test_unwindowed_probe_does_not_need_the_windowed_flag():
    transport = FakeTransport(windowed=False, readiness=WorkerReadiness(model=MODEL))
    adapter = make_adapter(transport)

    assert await adapter.extract_once("Alice works.", SCHEMA, threshold=0.5)


@pytest.mark.asyncio
async def test_expected_model_is_enforced_on_readiness_and_replies():
    adapter = make_adapter(FakeTransport(), expected_model="other/model")
    with pytest.raises(GlinerWorkerModelMismatchError):
        await adapter.ensure_ready()

    # Transports whose probe names no model are checked on the first reply.
    transport = FakeTransport(readiness=WorkerReadiness())
    adapter = make_adapter(transport, expected_model="other/model")
    with pytest.raises(GlinerWorkerModelMismatchError):
        await extract(adapter, ["Alice works."])
    with pytest.raises(GlinerWorkerModelMismatchError):
        await extract(adapter, ["Bob works."])
    assert len(transport.payloads) == 1


@pytest.mark.asyncio
async def test_replicas_serving_different_models_keep_the_first_one_recorded():
    transport = FakeTransport(readiness=WorkerReadiness(), model="model/a")
    adapter = make_adapter(transport)
    await extract(adapter, ["Alice works."])
    transport.model = "model/b"

    # Logged as a warning, not an error: no model was required.
    assert await extract(adapter, ["Bob works."]) == [extraction_for("Bob works.")]
    assert adapter.model == "model/a"


# --------------------------------------------------------------------------- #
# Readiness and limits
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_readiness_is_cached_for_its_ttl():
    now = [0.0]
    transport = FakeTransport()
    adapter = make_adapter(transport, clock=lambda: now[0])

    await adapter.ensure_ready()
    await extract(adapter, ["Alice works."])
    assert transport.readiness_calls == 1

    now[0] = 31.0
    await extract(adapter, ["Bob works."])
    assert transport.readiness_calls == 2


@pytest.mark.asyncio
async def test_readiness_retries_a_worker_that_is_still_loading():
    sleeps: list[float] = []
    transport = FakeTransport()
    attempts = 0
    real = transport.readiness

    async def loading():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise GlinerWorkerUnavailableError("model is not loaded")
        return await real()

    transport.readiness = loading
    adapter = make_adapter(transport, sleeps=sleeps)

    assert (await adapter.ensure_ready()).model == MODEL
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_a_worker_without_windowing_fails_at_readiness():
    transport = FakeTransport(readiness=WorkerReadiness(model=MODEL, features=frozenset()))
    adapter = make_adapter(transport)

    with pytest.raises(GlinerWorkerIncompatibleError):
        await adapter.ensure_ready()
    with pytest.raises(GlinerWorkerIncompatibleError):
        await extract(adapter, ["Alice works."])
    assert transport.payloads == []


@pytest.mark.asyncio
async def test_settings_beyond_the_workers_limits_are_refused_before_sending():
    limits = WorkerReadiness(
        model=MODEL,
        limits=WorkerLimits(max_inputs=4, max_text_chars=10, max_batch_size=32),
        features=frozenset({"windowing"}),
    )
    with pytest.raises(GlinerRemoteConfigError, match="INPUTS_PER_REQUEST"):
        await make_adapter(FakeTransport(readiness=limits), inputs_per_request=8).ensure_ready()

    transport = FakeTransport(readiness=limits)
    adapter = make_adapter(transport, inputs_per_request=4)
    with pytest.raises(GlinerRemoteConfigError, match="batch cap"):
        await extract(adapter, ["short"], batch_size=64)

    transport = FakeTransport(readiness=limits)
    adapter = make_adapter(transport, inputs_per_request=4)
    with pytest.raises(GlinerWorkerRejectedError, match="11 characters"):
        await extract(adapter, ["short", "x" * 11])
    assert transport.payloads == []
    # A too-long chunk fails its document only.
    assert await extract(adapter, ["short"]) == [extraction_for("short")]


def test_a_remote_transport_needs_an_endpoint():
    with pytest.raises(GlinerRemoteConfigError, match="COGNEE_GLINER_ENDPOINT"):
        RemoteGlinerAdapter(RemoteGlinerSettings(transport="http"))


@pytest.mark.asyncio
async def test_a_transport_that_cannot_be_built_fails_the_run_cleanly(monkeypatch):
    from cognee.tasks.graph.gliner_demo.remote import adapter as adapter_module

    def missing_extra(_settings):
        raise GlinerRemoteConfigError("grpcio is not installed")

    monkeypatch.setattr(adapter_module, "build_transport", missing_extra)
    adapter = RemoteGlinerAdapter(
        RemoteGlinerSettings(transport="grpc", endpoint="http://user:pw@worker:50051")
    )
    assert adapter.endpoint_label == "http://***@worker:50051"

    with pytest.raises(GlinerRemoteConfigError, match="grpcio"):
        await adapter.ensure_ready()
    with pytest.raises(GlinerRemoteConfigError, match="grpcio"):
        await extract(adapter, ["Alice works."])


@pytest.mark.asyncio
@pytest.mark.parametrize(("words", "overlap"), [(1025, 64), (384, 193), (8, 5)])
async def test_windows_beyond_the_workers_caps_fail_the_run_before_sending(words, overlap):
    transport = FakeTransport()
    adapter = make_adapter(transport)

    with pytest.raises(GlinerRemoteConfigError, match="half of it"):
        await extract(adapter, ["Alice works."], window_words=words, window_overlap_words=overlap)
    assert transport.payloads == []
    # Every document of the run would fail the same way.
    with pytest.raises(GlinerRemoteConfigError):
        await extract(adapter, ["Bob works."])


@pytest.mark.asyncio
async def test_windows_at_the_caps_are_sent():
    transport = FakeTransport()
    adapter = make_adapter(transport)
    await extract(adapter, ["Alice works."], window_words=1024, window_overlap_words=512)
    assert transport.payloads[0]["options"]["window_overlap_words"] == 512
