"""Remote GLiNER HTTP transport (SDK-980) against a scripted worker (httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from cognee.tasks.graph.gliner_demo.remote import (
    GlinerRemoteConfigError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnauthorizedError,
    GlinerWorkerUnavailableError,
)
from cognee.tasks.graph.gliner_demo.remote.http import HttpWorkerTransport
from cognee.tasks.graph.gliner_demo.remote.protocol import redact_url

ENDPOINT = "http://worker:8080"
PAYLOAD = {
    "inputs": [{"id": "0", "text": "Alice works for Acme."}],
    "schema": {"entities": {"person": ""}, "relations": {}},
    "options": {"window_words": 384, "window_overlap_words": 64},
}
REPLY = {
    "items": [{"id": "0", "extraction": {"entities": {"person": [{"text": "Alice"}]}}}],
    "model": "fastino/gliner2.5-base-v1",
    "elapsed_ms": 12.5,
    "windowed": True,
}
READY = {
    "status": "ready",
    "model": "fastino/gliner2.5-base-v1",
    "limits": {"max_inputs": 128, "max_text_chars": 200000, "max_batch_size": 256},
    "features": ["windowing"],
}


def transport_for(handler, *, api_key=None, max_response_bytes=1024 * 1024, endpoint=ENDPOINT):
    client = httpx.AsyncClient(base_url=endpoint, transport=httpx.MockTransport(handler))
    return HttpWorkerTransport(
        endpoint,
        api_key=api_key,
        connect_timeout=1,
        request_timeout=5,
        max_response_bytes=max_response_bytes,
        client=client,
    )


@pytest.mark.asyncio
async def test_extract_posts_the_payload_with_auth_and_request_id():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    reply = await transport_for(handler, api_key="s3cret").extract(PAYLOAD, request_id="req-1")

    [request] = seen
    assert request.method == "POST" and request.url.path == "/v1/extract"
    assert json.loads(request.content) == PAYLOAD
    assert request.headers["authorization"] == "Bearer s3cret"
    assert request.headers["x-request-id"] == "req-1"
    assert reply.items == [("0", {"entities": {"person": [{"text": "Alice"}]}})]
    assert (reply.model, reply.elapsed_ms, reply.windowed) == (REPLY["model"], 12.5, True)


@pytest.mark.asyncio
async def test_no_authorization_header_without_a_key():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    await transport_for(handler).extract(PAYLOAD, request_id="r")
    assert "authorization" not in seen[0].headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, GlinerWorkerUnavailableError),
        (502, GlinerWorkerUnavailableError),
        (503, GlinerWorkerUnavailableError),
        (504, GlinerWorkerUnavailableError),
        (401, GlinerWorkerUnauthorizedError),
        (403, GlinerWorkerUnauthorizedError),
        (400, GlinerWorkerRejectedError),
        (413, GlinerWorkerRejectedError),
        (422, GlinerWorkerRejectedError),
        (500, GlinerWorkerRuntimeError),
        (404, GlinerWorkerRuntimeError),
    ],
)
async def test_statuses_map_to_failure_classes(status, error):
    transport = transport_for(lambda _r: httpx.Response(status, json={"detail": "nope"}))
    with pytest.raises(error, match="nope"):
        await transport.extract(PAYLOAD, request_id="r")


@pytest.mark.asyncio
async def test_retry_after_seconds_are_passed_on_and_dates_ignored():
    transport = transport_for(lambda _r: httpx.Response(503, headers={"retry-after": "3"}))
    with pytest.raises(GlinerWorkerUnavailableError) as info:
        await transport.extract(PAYLOAD, request_id="r")
    assert info.value.retry_after == 3.0

    dated = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    transport = transport_for(lambda _r: httpx.Response(503, headers=dated))
    with pytest.raises(GlinerWorkerUnavailableError) as info:
        await transport.extract(PAYLOAD, request_id="r")
    assert info.value.retry_after is None


@pytest.mark.asyncio
async def test_unreachable_worker_is_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(GlinerWorkerUnavailableError, match="Cannot reach"):
        await transport_for(handler).extract(PAYLOAD, request_id="r")

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(GlinerWorkerUnavailableError):
        await transport_for(timeout).extract(PAYLOAD, request_id="r")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        json.dumps({"items": [{"id": 0, "extraction": {}}], "model": "m"}).encode(),
        json.dumps({"items": [], "model": "m", "windowed": "yes"}).encode(),
        json.dumps({"items": []}).encode(),
    ],
)
async def test_malformed_replies_are_runtime_errors(body):
    transport = transport_for(lambda _r: httpx.Response(200, content=body))
    with pytest.raises(GlinerWorkerRuntimeError):
        await transport.extract(PAYLOAD, request_id="r")


@pytest.mark.asyncio
async def test_replies_over_the_cap_are_refused_by_header_and_by_stream():
    big = json.dumps(REPLY).encode()
    transport = transport_for(lambda _r: httpx.Response(200, content=big), max_response_bytes=10)
    with pytest.raises(GlinerWorkerRuntimeError, match="cap"):
        await transport.extract(PAYLOAD, request_id="r")

    async def chunks():
        for _ in range(4):
            yield b"x" * 8

    # No content-length: refused once the stream passes the cap.
    streamed = transport_for(
        lambda _r: httpx.Response(200, content=chunks()), max_response_bytes=10
    )
    with pytest.raises(GlinerWorkerRuntimeError, match="cap"):
        await streamed.extract(PAYLOAD, request_id="r")


@pytest.mark.asyncio
async def test_readiness_reads_model_limits_and_features():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=READY)

    readiness = await transport_for(handler).readiness()

    assert seen[0].method == "GET" and seen[0].url.path == "/readyz"
    assert readiness.model == "fastino/gliner2.5-base-v1"
    assert readiness.limits.max_inputs == 128
    assert readiness.limits.max_text_chars == 200000
    assert readiness.limits.max_batch_size == 256
    assert readiness.features == frozenset({"windowing"})


@pytest.mark.asyncio
async def test_readiness_of_an_older_worker_has_no_features_or_limits():
    old = {"status": "ready", "model": "m"}
    readiness = await transport_for(lambda _r: httpx.Response(200, json=old)).readiness()
    assert readiness.features == frozenset()
    assert readiness.limits.max_inputs is None


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"status": "ready"}, {"status": "alive", "model": "m"}])
async def test_readiness_is_strict_about_what_a_worker_looks_like(body):
    # A 200 from another service on the wrong port must not count as ready.
    with pytest.raises(GlinerWorkerRuntimeError, match="did not answer like a GLiNER worker"):
        await transport_for(lambda _r: httpx.Response(200, json=body)).readiness()


@pytest.mark.asyncio
async def test_readiness_while_the_model_loads_is_unavailable():
    transport = transport_for(lambda _r: httpx.Response(503, json={"detail": "loading"}))
    with pytest.raises(GlinerWorkerUnavailableError):
        await transport.readiness()


@pytest.mark.parametrize("endpoint", ["worker:8080", "ftp://worker", "amqp://x@y:5672"])
def test_endpoint_must_be_an_http_url(endpoint):
    with pytest.raises(GlinerRemoteConfigError):
        HttpWorkerTransport(
            endpoint, api_key=None, connect_timeout=1, request_timeout=1, max_response_bytes=1
        )


@pytest.mark.asyncio
async def test_credentials_in_the_endpoint_never_reach_errors():
    endpoint = "http://user:p@ss@worker:8080"
    transport = transport_for(
        lambda _r: httpx.Response(500, json={"detail": "x"}), endpoint=endpoint
    )
    assert transport.endpoint_label == "http://***@worker:8080"
    with pytest.raises(GlinerWorkerRuntimeError) as info:
        await transport.extract(PAYLOAD, request_id="r")
    assert "p@ss" not in str(info.value)


def test_redact_url_masks_each_comma_separated_part():
    assert redact_url("amqp://guest:guest@rabbit:5672/%2f") == "amqp://***@rabbit:5672/%2f"
    assert redact_url("http://a:1,http://u:p@b:2") == "http://a:1,http://***@b:2"
