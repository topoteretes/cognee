"""SSE transport for POST /v1/recall.

The contract that matters: streaming is opt-in, and a streamed request returns
the *same* payload a normal one would. Deltas are a live preview; ``final`` is
the answer. These tests drive the real mounted route so the negotiation, the
ContextVar plumbing and the relay are exercised together — the parts that look
fine in isolation and break when wired.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from cognee.api.v1.recall.recall_stream import wants_event_stream
from cognee.api.v1.recall.routers.get_recall_router import get_recall_router
from cognee.infrastructure.llm.streaming.token_sink import (
    get_active_token_sink,
    stream_answer_tokens,
)
from cognee.modules.search.types import SearchType
from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry
from cognee.modules.recall.types.SearchResultItem import SearchResultKind
from cognee.modules.users.methods import get_authenticated_user

# The package re-exports `recall`, so plain `import cognee.api.v1.recall`
# resolves to the function; the handler imports the name from the module, so the
# module object is what has to be patched.
recall_pkg = importlib.import_module("cognee.api.v1.recall")

ANSWER = ["Neon ", "was ", "chosen."]
# Real RecallResponse instances, not hand-written dicts: the JSON path is
# validated against response_model while the streaming path encodes what recall
# returned, so only a faithful payload can show the two agreeing.
PAYLOAD = [
    ResponseGraphEntry(
        kind=SearchResultKind.GRAPH_COMPLETION,
        search_type=SearchType.GRAPH_COMPLETION,
        text="Neon was chosen.",
        source="graph",
    )
]
ENCODED = jsonable_encoder(PAYLOAD)


def _flag(enabled: bool = True):
    return patch(
        "cognee.infrastructure.llm.config.get_llm_context_config",
        return_value=SimpleNamespace(llm_answer_streaming=enabled),
    )


async def _recall_that_streams(**_kwargs):
    """Stand-in for recall(): streams through the real promotion helper."""
    async with stream_answer_tokens(stage="generating"):
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
    """(event, data) pairs, ignoring keepalive comments."""
    out = []
    for block in text.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        event = next((line[7:] for line in lines if line.startswith("event: ")), None)
        data = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if event:
            out.append((event, data))
    return out


# --------------------------- negotiation ---------------------------


@pytest.mark.parametrize(
    "accept,flag,expected",
    [
        ("text/event-stream", None, True),
        ("text/event-stream; charset=utf-8", None, True),
        ("application/json, text/event-stream", None, True),
        # fetch() sends this unless told otherwise — every existing caller.
        ("*/*", None, False),
        ("application/json", None, False),
        (None, None, False),
        # Explicit opt-out for a client that cannot set its own Accept header.
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


def test_stream_false_forces_json_despite_the_header(client):
    with _flag():
        response = client.post(
            "/recall",
            json={"query": "why neon?", "stream": False},
            headers={"Accept": "text/event-stream"},
        )

    assert response.json() == ENCODED


# --------------------------- the stream itself ---------------------------


def test_tokens_arrive_as_deltas_then_the_same_payload_as_final(client):
    with _flag():
        response = client.post(
            "/recall",
            json={"query": "why neon?"},
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # nginx buffers proxied responses by default, which would hold every token
    # until the answer finished — the exact thing streaming exists to avoid.
    assert response.headers["x-accel-buffering"] == "no"

    frames = _frames(response.text)
    kinds = [event for event, _ in frames]
    assert kinds[0] == "stage", "retrieval starts before any token exists"
    assert kinds[-1] == "final"
    assert "answer_done" in kinds

    import json

    streamed = "".join(json.loads(data)["text"] for event, data in frames if event == "delta")
    assert streamed == "Neon was chosen."

    # The guarantee that makes streaming safe to turn on: the two transports
    # return the same answer, so a client can stream without giving anything up.
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
            "/recall",
            json={"query": "why neon?"},
            headers={"Accept": "text/event-stream"},
        )

    frames = _frames(response.text)
    assert [event for event, _ in frames if event == "delta"] == []
    assert frames[-1][0] == "final"


def test_flag_off_streams_no_tokens_but_still_answers(client):
    """The engine is inert until LLM_ANSWER_STREAMING is on; the transport must
    degrade to a single final event rather than hanging."""
    with _flag(False):
        response = client.post(
            "/recall",
            json={"query": "why neon?"},
            headers={"Accept": "text/event-stream"},
        )

    frames = _frames(response.text)
    assert [event for event, _ in frames if event == "delta"] == []
    assert frames[-1][0] == "final"


def test_a_failing_recall_reports_an_error_event(client, monkeypatch):
    """The 200 and the first frame are already sent, so the failure can only
    reach the client as an event."""

    async def _boom(**_kwargs):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(recall_pkg, "recall", _boom)
    with _flag():
        response = client.post(
            "/recall",
            json={"query": "why neon?"},
            headers={"Accept": "text/event-stream"},
        )

    kinds = [event for event, _ in _frames(response.text)]
    assert kinds[-1] == "error"
    assert "final" not in kinds
