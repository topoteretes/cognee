"""Remote GLiNER gRPC transport (SDK-980) against an in-process fake worker."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util

import pytest

grpc = pytest.importorskip("grpc")

from cognee.tasks.graph.gliner_demo.remote import (  # noqa: E402
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
    grpc_descriptors,
)
from cognee.tasks.graph.gliner_demo.remote.grpc import (  # noqa: E402
    GrpcWorkerTransport,
    _messages,
    classify_status,
)

M = _messages()
PAYLOAD = {
    "inputs": [{"id": "0", "text": "Alice works for Acme."}, {"id": "1", "text": "Bob."}],
    "schema": {
        "entities": {"person": "", "organization": "A company", "location": None},
        "relations": {"works_for": ""},
    },
    "options": {
        "threshold": 0.4,
        "batch_size": 8,
        "include_confidence": True,
        "include_spans": True,
        "overlap_policy": "longest",
        "window_words": 384,
        "window_overlap_words": 64,
    },
}


class FakeWorker:
    def __init__(self):
        self.requests = []
        self.metadata = []
        self.health = M["HealthCheckResponse"].SERVING
        self.abort: tuple | None = None
        self.stall: float = 0.0

    async def extract(self, request, context):
        self.requests.append(request)
        self.metadata.append(dict(context.invocation_metadata()))
        if self.stall:
            await asyncio.sleep(self.stall)
        if self.abort:
            await context.abort(*self.abort)
        response = M["ExtractResponse"](model="fake/model", elapsed_ms=3.5, windowed=True)
        for item in request.inputs:
            result = response.items.add(id=item.id)
            # Map entries inserted against schema order: the client restores it.
            result.entities["location"].mentions.add()
            result.entities["person"].mentions.add(text="Alice", confidence=0.9, start=0, end=5)
            result.entities["organization"].mentions.add(text="Acme")
            pair = result.relations["works_for"].relations.add()
            pair.head.text, pair.tail.text = "Alice", "Acme"
        return response

    async def check(self, request, context):
        return M["HealthCheckResponse"](status=self.health)


@contextlib.asynccontextmanager
async def serve(worker: FakeWorker):
    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "gliner_worker.v1.GlinerWorker",
                {
                    "Extract": grpc.unary_unary_rpc_method_handler(
                        worker.extract,
                        request_deserializer=M["ExtractRequest"].FromString,
                        response_serializer=M["ExtractResponse"].SerializeToString,
                    )
                },
            ),
            grpc.method_handlers_generic_handler(
                "grpc.health.v1.Health",
                {
                    "Check": grpc.unary_unary_rpc_method_handler(
                        worker.check,
                        request_deserializer=M["HealthCheckRequest"].FromString,
                        response_serializer=M["HealthCheckResponse"].SerializeToString,
                    )
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await server.stop(None)


def transport_for(endpoint: str, *, api_key=None, request_timeout=5.0) -> GrpcWorkerTransport:
    return GrpcWorkerTransport(
        endpoint,
        api_key=api_key,
        connect_timeout=2.0,
        request_timeout=request_timeout,
        max_response_bytes=1024 * 1024,
    )


@pytest.mark.asyncio
async def test_extract_maps_the_payload_and_the_typed_reply():
    worker = FakeWorker()
    async with serve(worker) as endpoint:
        transport = transport_for(endpoint, api_key="s3cret")
        reply = await transport.extract(PAYLOAD, request_id="req-1")
        await transport.aclose()

    [request] = worker.requests
    assert [(i.id, i.text) for i in request.inputs] == [
        ("0", "Alice works for Acme."),
        ("1", "Bob."),
    ]
    labels = [(label.name, label.HasField("description")) for label in request.schema.entities]
    # "" is sent as a present, empty description; None as no description.
    assert labels == [("person", True), ("organization", True), ("location", False)]
    assert request.schema.entities[0].description == ""
    assert request.options.window_words == 384
    assert request.options.window_overlap_words == 64
    assert request.options.overlap_policy == "longest"
    assert worker.metadata[0]["authorization"] == "Bearer s3cret"
    assert worker.metadata[0]["x-request-id"] == "req-1"

    assert (reply.model, reply.elapsed_ms, reply.windowed) == ("fake/model", 3.5, True)
    item_id, extraction = reply.items[0]
    assert item_id == "0"
    # Labels back in schema order, gliner2's shapes per mention.
    assert list(extraction["entities"]) == ["person", "organization", "location"]
    assert extraction["entities"]["person"] == [
        {"text": "Alice", "confidence": pytest.approx(0.9), "start": 0, "end": 5}
    ]
    assert extraction["entities"]["organization"] == ["Acme"]
    assert extraction["relation_extraction"] == {"works_for": [["Alice", "Acme"]]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "error"),
    [
        (grpc.StatusCode.UNAVAILABLE, GlinerWorkerUnavailableError),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, GlinerWorkerUnavailableError),
        (grpc.StatusCode.UNAUTHENTICATED, GlinerWorkerUnauthorizedError),
        (grpc.StatusCode.PERMISSION_DENIED, GlinerWorkerUnauthorizedError),
        (grpc.StatusCode.INVALID_ARGUMENT, GlinerWorkerRejectedError),
        (grpc.StatusCode.INTERNAL, GlinerWorkerRuntimeError),
    ],
)
async def test_statuses_map_to_failure_classes(code, error):
    worker = FakeWorker()
    worker.abort = (code, "nope")
    async with serve(worker) as endpoint:
        transport = transport_for(endpoint)
        with pytest.raises(error, match="nope"):
            await transport.extract(PAYLOAD, request_id="r")
        await transport.aclose()


def test_cancelled_and_deadline_exceeded_are_retryable():
    assert isinstance(classify_status("CANCELLED", ""), GlinerWorkerUnavailableError)
    assert isinstance(classify_status("DEADLINE_EXCEEDED", ""), GlinerWorkerUnavailableError)
    assert isinstance(classify_status("ABORTED", ""), GlinerWorkerUnavailableError)
    assert isinstance(classify_status("UNKNOWN", ""), GlinerWorkerRuntimeError)


@pytest.mark.asyncio
async def test_a_call_past_the_request_timeout_is_unavailable():
    worker = FakeWorker()
    worker.stall = 1.0
    async with serve(worker) as endpoint:
        transport = transport_for(endpoint, request_timeout=0.2)
        with pytest.raises(GlinerWorkerUnavailableError, match="DEADLINE_EXCEEDED"):
            await transport.extract(PAYLOAD, request_id="r")
        await transport.aclose()


@pytest.mark.asyncio
async def test_requests_are_balanced_round_robin_over_endpoints():
    first, second = FakeWorker(), FakeWorker()
    async with serve(first) as a, serve(second) as b:
        transport = transport_for(f"{a},{b}")
        for _ in range(4):
            await transport.extract(PAYLOAD, request_id="r")
        await transport.aclose()
    assert (len(first.requests), len(second.requests)) == (2, 2)


@pytest.mark.asyncio
async def test_readiness_follows_the_health_service():
    worker = FakeWorker()
    async with serve(worker) as endpoint:
        transport = transport_for(endpoint)
        readiness = await transport.readiness()
        assert readiness.model is None and readiness.features is None

        worker.health = M["HealthCheckResponse"].NOT_SERVING
        with pytest.raises(GlinerWorkerUnavailableError, match="not serving"):
            await transport.readiness()
        await transport.aclose()


@pytest.mark.asyncio
async def test_one_serving_replica_is_enough():
    down, up = FakeWorker(), FakeWorker()
    down.health = M["HealthCheckResponse"].NOT_SERVING
    async with serve(down) as a, serve(up) as b:
        transport = transport_for(f"{a},{b}")
        await transport.readiness()
        await transport.aclose()


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_is_unavailable():
    transport = transport_for("http://127.0.0.1:1")
    with pytest.raises(GlinerWorkerUnavailableError):
        await transport.readiness()
    with pytest.raises(GlinerWorkerUnavailableError):
        await transport.extract(PAYLOAD, request_id="r")
    await transport.aclose()


@pytest.mark.parametrize("endpoint", ["ftp://worker:1", "http://user:pw@worker:1", " , "])
def test_bad_endpoints_are_configuration_errors(endpoint):
    with pytest.raises(GlinerRemoteConfigError):
        transport_for(endpoint)


def test_https_endpoints_use_tls_without_connecting():
    transport = transport_for("https://worker.example:443")
    assert transport.endpoint_label == "https://worker.example:443"


@pytest.mark.skipif(
    importlib.util.find_spec("grpc_tools") is None, reason="grpcio-tools is not installed"
)
@pytest.mark.parametrize("name", sorted(grpc_descriptors.PROTO_FILES))
def test_serialized_descriptors_match_the_proto_files(name):
    compiled = grpc_descriptors.compile_proto(grpc_descriptors.PROTO_FILES[name])
    assert compiled == getattr(grpc_descriptors, name), (
        "Regenerate with: python -m cognee.tasks.graph.gliner_demo.remote.grpc_descriptors"
    )
