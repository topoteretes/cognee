"""SSE transport for POST /v1/recall.

Three things have to hold. Streaming is opt-in, so no existing client is moved
onto it by accident. A streamed request returns the *same* payload a JSON one
would — deltas are a live preview, `final` is the answer. And a request that
fails still fails the same way: the status code is decided before the first byte
goes out, so a permission denial or a bad `scope` is a 403/422 on both paths
rather than a 200 carrying an opaque error.
"""

import asyncio
import importlib
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from cognee.infrastructure.llm.LLMGateway import LLMGateway

from cognee.api.sse import wants_event_stream
from cognee.api.v1.recall import recall_stream
from cognee.api.v1.recall.routers.get_recall_router import get_recall_router
from cognee.exceptions import CogneeApiError
from cognee.infrastructure.llm.streaming.token_sink import (
    get_active_token_sink,
    answer_scope,
)
from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry
from cognee.modules.recall.types.SearchResultItem import SearchResultKind
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_authenticated_user

# The package re-exports `recall`, so plain `import cognee.api.v1.recall`
# resolves to the function; the handler imports the name from the module, so the
# module object is what has to be patched.
recall_pkg = importlib.import_module("cognee.api.v1.recall")

ANSWER = ["Neon ", "was ", "chosen."]
PAYLOAD = [
    ResponseGraphEntry(
        kind=SearchResultKind.GRAPH_COMPLETION,
        search_type=SearchType.GRAPH_COMPLETION,
        text="Neon was chosen.",
        source="graph",
    )
]
ENCODED = jsonable_encoder(PAYLOAD)


@contextmanager
def _flag(enabled: bool = True, adapter_streams: bool = True):
    """Set both preconditions for promotion — the flag and the adapter
    capability — so whether a request streams is decided by the test, not by
    whatever LLM provider the host environment happens to configure. Unpinned,
    a host where the capability probe resolves False turns every streamed
    request here into the zero-delta path and the delta/reset/answer_done
    assertions fail. patch.object because the LLMGateway class shadows its
    module and string targets land on the class under Python 3.10's mock."""
    with (
        patch(
            "cognee.infrastructure.llm.config.get_llm_context_config",
            return_value=SimpleNamespace(llm_answer_streaming=enabled),
        ),
        patch.object(
            LLMGateway,
            "supports_answer_streaming",
            return_value=adapter_streams,
        ),
    ):
        yield


async def _recall_that_streams(**_kwargs):
    """Stand-in for recall(): streams through the real promotion helper."""
    async with answer_scope(stage="generating"):
        sink = get_active_token_sink()
        for token in ANSWER:
            if sink:
                sink.put_delta(token)
    return PAYLOAD


async def _recall_that_does_not_stream(**_kwargs):
    """A sessionless recall, only_context, or a non-streaming retriever."""
    return PAYLOAD


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(get_recall_router(), prefix="/recall")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    monkeypatch.setattr(recall_pkg, "recall", _recall_that_streams)
    with TestClient(app) as test_client:
        yield test_client


def _frames(text: str) -> list[tuple[str, str]]:
    out = []
    for block in text.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        event = next((line[7:] for line in lines if line.startswith("event: ")), None)
        data = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if event:
            out.append((event, data))
    return out


def _keepalives(text: str) -> int:
    return text.count(": keepalive")


# --------------------------- negotiation ---------------------------


@pytest.mark.parametrize(
    "accept,flag,expected",
    [
        ("text/event-stream", None, True),
        ("text/event-stream; charset=utf-8", None, True),
        ("application/json;q=0.5, text/event-stream", None, True),
        ("TEXT/EVENT-STREAM", None, True),  # media types are case-insensitive
        # fetch/httpx/requests all send this; a tie must stay on JSON.
        ("*/*", None, False),
        ("application/json", None, False),
        (None, None, False),
        # The MCP Streamable-HTTP header. Listing both is not a preference for
        # SSE, and flipping these clients would break them on the first frame.
        ("application/json, text/event-stream", None, False),
        # An explicit refusal must be honoured, not read as opting in.
        ("application/json, text/event-stream;q=0", None, False),
        # The body flag decides outright, in both directions.
        ("application/json", True, True),
        ("text/event-stream", False, False),
    ],
)
def test_streaming_is_opt_in(accept, flag, expected):
    assert wants_event_stream(accept, flag) is expected


def test_default_request_still_gets_json(client):
    """The whole point of negotiating on Accept: nobody changes until they ask."""
    with _flag():
        response = client.post("/recall", json={"query": "why neon?"})

    assert response.status_code == 200
    assert response.json() == ENCODED
    assert "text/event-stream" not in response.headers["content-type"]


def test_stream_true_opts_in_without_an_accept_header(client):
    """`stream: true` is the obvious thing for a client to send; ignoring it —
    as an Accept-only check does — silently returns JSON instead."""
    with _flag():
        response = client.post("/recall", json={"query": "why neon?", "stream": True})

    assert response.headers["content-type"].startswith("text/event-stream")


def test_stream_false_forces_json_despite_the_header(client):
    with _flag():
        response = client.post(
            "/recall",
            json={"query": "why neon?", "stream": False},
            headers={"Accept": "text/event-stream"},
        )

    assert response.json() == ENCODED


def test_hop_by_hop_headers_are_not_set(client):
    """`Connection` is hop-by-hop; an ASGI app must not set it and HTTP/2
    forbids it outright."""
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    assert "connection" not in {k.lower() for k in response.headers}
    assert response.headers["x-accel-buffering"] == "no"


# --------------------------- the stream itself ---------------------------


def test_tokens_arrive_as_deltas_then_the_same_payload_as_final(client):
    with _flag():
        response = client.post(
            "/recall",
            json={"query": "why neon?"},
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    frames = _frames(response.text)
    kinds = [event for event, _ in frames]
    assert kinds[0] == "stage"
    assert kinds[-1] == "final"
    assert "answer_done" in kinds

    streamed = "".join(json.loads(data)["text"] for event, data in frames if event == "delta")
    assert streamed == "Neon was chosen."

    # The guarantee that makes streaming safe to turn on: the two transports
    # return the same answer, so a client gives up nothing by opting in.
    with _flag():
        plain = client.post("/recall", json={"query": "why neon?"})
    final = json.loads(frames[-1][1])["results"]
    assert final == plain.json() == ENCODED


def test_a_recall_that_streams_nothing_still_completes(client, monkeypatch):
    """Sessionless recall, sequential mode, only_context: zero deltas is normal,
    and the request must still terminate with the full payload."""
    monkeypatch.setattr(recall_pkg, "recall", _recall_that_does_not_stream)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    frames = _frames(response.text)
    assert [event for event, _ in frames if event == "delta"] == []
    assert frames[-1][0] == "final"
    assert json.loads(frames[-1][1])["results"] == ENCODED


def test_flag_off_streams_no_tokens_but_still_answers(client):
    with _flag(False):
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    frames = _frames(response.text)
    assert [event for event, _ in frames if event == "delta"] == []
    assert frames[-1][0] == "final"


def test_a_retry_tells_the_client_to_discard_the_partial_answer(client, monkeypatch):
    """Tenacity re-runs the whole call, which re-streams from the beginning."""

    async def _retried(**_kwargs):
        async with answer_scope(stage="generating"):
            sink = get_active_token_sink()
            sink.put_delta("partial")
            sink.begin_attempt()  # what a retry does
            sink.put_delta("complete")
        return PAYLOAD

    monkeypatch.setattr(recall_pkg, "recall", _retried)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    kinds = [event for event, _ in _frames(response.text)]
    assert "reset" in kinds
    assert kinds.index("reset") < kinds.index("final")


def test_keepalives_cover_the_silent_persistence_phase(client, monkeypatch):
    """The sink closes with the last token, but commit_turn, add_qa and
    serialisation still have to run. Without keepalives that gap can exceed a
    proxy's idle timeout and `final` — the authoritative payload — is lost."""

    async def _slow_persistence(**_kwargs):
        async with answer_scope(stage="generating"):
            get_active_token_sink().put_delta("answer")
        await asyncio.sleep(0.25)  # persistence, emitting nothing
        return PAYLOAD

    monkeypatch.setattr(recall_pkg, "recall", _slow_persistence)
    monkeypatch.setattr(recall_stream, "KEEPALIVE_SECONDS", 0.05)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    assert _keepalives(response.text) >= 1, "no keepalive during the quiet phase"
    assert _frames(response.text)[-1][0] == "final"


# --------------------------- failure ---------------------------


def test_a_failure_before_any_output_keeps_its_status_code(client, monkeypatch):
    """Once a byte is sent the status is fixed at 200, so the decision is made
    first. A 422 must not become a 200 carrying an opaque error."""

    async def _bad_scope(**_kwargs):
        raise ValueError("scope must be one of: graph, session, trace")

    monkeypatch.setattr(recall_pkg, "recall", _bad_scope)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    assert response.status_code == 422
    # Deliberately does NOT assert the message text. The router rebuilds it
    # rather than echoing the exception, so a ValueError raised deeper in the
    # recall path cannot leak its text to the client (SDK-463). What this test
    # owns is the status code and the transport: a failure decided before the
    # first byte stays JSON and keeps its real status, instead of becoming a
    # 200 carrying an error frame.
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert response.json()["error"]


def test_a_cognee_error_before_output_is_left_to_the_global_handler(client, monkeypatch):
    """CogneeApiError subclasses carry their own status (403, 402, 409) and the
    JSON path re-raises them; the streamed path must not swallow that."""

    class _Denied(CogneeApiError):
        def __init__(self):
            super().__init__(message="No access to dataset", status_code=403)

    async def _denied(**_kwargs):
        raise _Denied()

    monkeypatch.setattr(recall_pkg, "recall", _denied)
    with _flag():
        with pytest.raises(_Denied):
            client.post("/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"})


def test_a_failure_after_output_arrives_as_a_single_error_event(client, monkeypatch):
    """The 200 and the first frames are already sent, so the failure can only
    reach the client as an event — and only one, since a client that treats
    `error` as terminal has already stopped reading."""

    async def _fails_mid_answer(**_kwargs):
        async with answer_scope(stage="generating"):
            get_active_token_sink().put_delta("half an ans")
            raise RuntimeError("graph unavailable")

    monkeypatch.setattr(recall_pkg, "recall", _fails_mid_answer)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    assert response.status_code == 200
    kinds = [event for event, _ in _frames(response.text)]
    assert kinds.count("error") == 1, kinds
    assert "final" not in kinds
    assert kinds[-1] == "error"


def test_a_failure_after_output_still_reports_the_status_it_would_have_had(client, monkeypatch):
    """Credit exhaustion is the failure this most needs to survive the transport.

    The 200 is already committed by the time the answer call runs, so a client
    that reads only the status line cannot be helped — but the frame can still
    say 402, and it must, because "top up your credits" and "transient fault"
    need different handling. Both error paths carry `status` for that reason;
    this pins the sink one, which is the path a mid-answer failure takes.
    """

    class NoCredit(Exception):
        def __init__(self):
            super().__init__("insufficient credit")
            self.status_code = 402

    async def _runs_out_of_credit(**_kwargs):
        async with answer_scope(stage="generating"):
            get_active_token_sink().put_delta("half an ans")
            raise NoCredit()

    monkeypatch.setattr(recall_pkg, "recall", _runs_out_of_credit)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    assert response.status_code == 200
    errors = [json.loads(data) for event, data in _frames(response.text) if event == "error"]
    assert len(errors) == 1
    assert errors[0]["status"] == 402
    assert "insufficient credit" not in errors[0]["message"]


def test_a_failure_with_no_status_falls_back_to_the_catch_all(client, monkeypatch):
    """A bare exception carries no status, so the frame uses the same 409 the
    route's catch-all would — the shape never varies."""

    async def _plain_failure(**_kwargs):
        async with answer_scope(stage="generating"):
            get_active_token_sink().put_delta("half an ans")
            raise RuntimeError("graph unavailable")

    monkeypatch.setattr(recall_pkg, "recall", _plain_failure)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    errors = [json.loads(data) for event, data in _frames(response.text) if event == "error"]
    assert len(errors) == 1 and errors[0]["status"] == 409


def test_the_error_event_does_not_leak_provider_detail(client, monkeypatch):
    """Provider errors embed the rendered prompt — the whole retrieved graph
    context — plus endpoints and request bodies."""
    secret = "ContentPolicy: <entire graph context, api_base=https://internal>"

    async def _leaky(**_kwargs):
        async with answer_scope(stage="generating"):
            get_active_token_sink().put_delta("x")
            raise RuntimeError(secret)

    monkeypatch.setattr(recall_pkg, "recall", _leaky)
    with _flag():
        response = client.post(
            "/recall", json={"query": "q"}, headers={"Accept": "text/event-stream"}
        )

    assert secret not in response.text
    assert "api_base" not in response.text


# --------------------------- disconnect ---------------------------


@pytest.mark.asyncio
async def test_a_disconnect_detaches_the_sink_and_lets_the_recall_finish():
    """Starlette cancels the response generator when the client goes away. That
    must stop buffering without cancelling the recall — the turn is still worth
    answering and persisting, and the LLM has already been charged for it.

    Polling request.is_disconnected() cannot do this: it reads the same ASGI
    receive channel Starlette's own disconnect listener is consuming, and it is
    only reached on a keepalive tick, so a disconnect while tokens are flowing
    would never be noticed at all.
    """
    finished = asyncio.Event()

    async def _long_recall(**_kwargs):
        async with answer_scope(stage="generating"):
            sink = get_active_token_sink()
            sink.put_delta("first")
            await asyncio.sleep(0.05)
            sink.put_delta("second")
        finished.set()
        return PAYLOAD

    with _flag():
        started = await recall_stream.begin_recall_stream(_long_recall)
        frames = started.frames()
        await frames.__anext__()  # the client reads one frame, then vanishes
        await frames.aclose()  # what Starlette does on http.disconnect

        assert started._sink._detached is True, "sink must stop buffering"
        await asyncio.wait_for(finished.wait(), timeout=1.0)
        assert await asyncio.wait_for(started._task, timeout=1.0) == PAYLOAD


@pytest.mark.asyncio
async def test_dropped_preview_frames_are_signalled_before_final():
    """Drops happen exactly when the consumer is slower than the model, so the
    rendered answer has silent gaps. `final` is authoritative and still coming;
    the client needs telling to stop trusting what it drew."""

    async def _floods(**_kwargs):
        async with answer_scope(stage="generating"):
            sink = get_active_token_sink()
            for index in range(50):
                sink.put_delta(f"token{index}")
        return PAYLOAD

    with _flag(), patch.object(recall_stream, "KEEPALIVE_SECONDS", 5.0):
        with patch("cognee.infrastructure.llm.streaming.token_sink.MAX_BUFFERED_EVENTS", 4):
            started = await recall_stream.begin_recall_stream(_floods)
            body = "".join([frame async for frame in started.frames()])

    kinds = [event for event, _ in _frames(body)]
    assert "reset" in kinds, "a truncated preview must be signalled"
    assert kinds[-1] == "final"
