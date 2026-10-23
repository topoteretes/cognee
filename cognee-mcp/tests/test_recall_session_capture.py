"""recall records each query as a turn of the MCP conversation (SDK-1121).

MCP clients such as Claude Desktop have no hook surface, so the only time the
server sees the conversation is a tool call. recall already runs on every
turn, so it stores its query in a session derived from the request context,
and improve defaults to that session. No model-visible cost: no extra tool,
and the write never blocks the search.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from mcp.server.lowlevel.server import request_ctx

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from src import server  # noqa: E402


def _fake_client(**overrides):
    attrs = {
        "recall": AsyncMock(return_value=[{"text": "hit"}]),
        "get_recall_state": AsyncMock(),
        "remember": AsyncMock(return_value={"status": "stored"}),
        "improve": AsyncMock(return_value={"status": "completed", "stages": []}),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class _Session:
    """Stands in for mcp's ServerSession: weak-referenceable, carries clientInfo."""

    def __init__(self, client_name="Claude Desktop"):
        info = SimpleNamespace(name=client_name) if client_name else None
        self.client_params = SimpleNamespace(clientInfo=info)


@pytest.fixture
def in_request():
    """Enter a fake MCP request context; yields a setter to pick transport/session."""
    tokens = []

    def enter(session=None, headers=None):
        request = SimpleNamespace(headers=headers) if headers is not None else None
        ctx = SimpleNamespace(session=session or _Session(), request=request)
        tokens.append(request_ctx.set(ctx))
        return ctx

    yield enter
    for token in reversed(tokens):
        try:
            request_ctx.reset(token)
        except ValueError:
            # An async test body runs in its own copied context, which pytest
            # discards with the value in it; only a sync test needs the reset.
            pass


async def _drain():
    if server._background_tasks:
        await asyncio.gather(*server._background_tasks, return_exceptions=True)


# --- session id derivation -------------------------------------------------------


def test_no_request_context_means_no_session():
    assert server._derived_session_id() is None


def test_http_uses_transport_session_id_prefixed_with_client(in_request):
    in_request(headers={"mcp-session-id": "1d224079f190443b86d6c02b0024fcf9"})
    assert server._derived_session_id() == "claude_desktop_1d224079f190443b86d6c02b0024fcf9"


def test_stdio_mints_one_id_per_connection(in_request):
    first = _Session()
    in_request(session=first)
    a = server._derived_session_id()
    b = server._derived_session_id()
    assert a == b
    assert a.startswith("claude_desktop_") and len(a) > len("claude_desktop_")

    in_request(session=_Session())
    assert server._derived_session_id() != a


def test_unknown_client_falls_back_to_mcp_prefix(in_request):
    in_request(session=_Session(client_name=None), headers={"mcp-session-id": "abc"})
    assert server._derived_session_id() == "mcp_abc"


def test_env_flag_disables_derivation(monkeypatch, in_request):
    monkeypatch.setenv("COGNEE_MCP_SESSION_CAPTURE", "false")
    in_request(headers={"mcp-session-id": "abc"})
    assert server._derived_session_id() is None


# --- recall stores the turn ------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_stores_query_in_derived_session(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "claude_desktop_memory")
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.recall("what did we decide about pricing?")
    await _drain()

    assert result[0].text.endswith("hit")
    fake.remember.assert_awaited_once_with(
        data="what did we decide about pricing?",
        dataset_name="claude_desktop_memory",
        session_id="claude_desktop_abc",
        self_improvement=False,
    )
    # The search itself is unchanged: no derived session is injected into it.
    assert fake.recall.call_args.kwargs["session_id"] is None
    # Not registered as an indexing job, or an empty recall would say "still indexing".
    assert not server._background_task_datasets


@pytest.mark.asyncio
async def test_recall_stores_into_the_first_requested_dataset(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("q", datasets="project_a,project_b")
    await _drain()

    assert fake.remember.call_args.kwargs["dataset_name"] == "project_a"


@pytest.mark.asyncio
async def test_recall_store_false_writes_nothing(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("off the record", store=False)
    await _drain()

    fake.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_outside_mcp_request_writes_nothing(monkeypatch):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)

    await server.recall("q")
    await _drain()

    fake.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_failure_never_breaks_recall(monkeypatch, in_request):
    fake = _fake_client(remember=AsyncMock(side_effect=RuntimeError("cache down")))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.recall("q")
    await _drain()

    assert result[0].text.endswith("hit")
    assert "Error" not in result[0].text


@pytest.mark.asyncio
async def test_capture_tolerates_client_without_remember(monkeypatch, in_request):
    """Older/partial clients (and the existing recall tests' stubs) lack remember."""
    fake = SimpleNamespace(recall=AsyncMock(return_value=[{"text": "hit"}]))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.recall("q")
    await _drain()

    assert result[0].text.endswith("hit")


# --- improve picks the session up ------------------------------------------------


@pytest.mark.asyncio
async def test_improve_defaults_to_derived_session(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "claude_desktop_memory")
    in_request(headers={"mcp-session-id": "abc"})

    await server.improve()

    assert fake.improve.call_args.kwargs["session_ids"] == ["claude_desktop_abc"]


@pytest.mark.asyncio
async def test_improve_explicit_session_ids_still_win(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.improve(session_ids="s1, s2")

    assert fake.improve.call_args.kwargs["session_ids"] == ["s1", "s2"]


@pytest.mark.asyncio
async def test_improve_without_request_keeps_no_session_ids(monkeypatch):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)

    await server.improve()

    assert fake.improve.call_args.kwargs["session_ids"] is None


# --- explicit remember is unchanged ----------------------------------------------


@pytest.mark.asyncio
async def test_remember_without_session_id_stays_permanent(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.remember(data="keep this", dataset_name="ds")

    assert fake.remember.call_args.kwargs["session_id"] is None
    assert "permanently" in result[0].text


# --- through a real MCP client ---------------------------------------------------


@pytest.mark.asyncio
async def test_real_mcp_client_conversation_lands_in_one_session(monkeypatch):
    """Two recalls over one connection share a session named after the client."""
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "mcp_memory")

    async with Client(server.mcp) as client:
        await client.call_tool("recall", {"query": "first turn"})
        await client.call_tool("recall", {"query": "second turn"})
        await _drain()

    sessions = {call.kwargs["session_id"] for call in fake.remember.await_args_list}
    assert fake.remember.await_count == 2
    assert len(sessions) == 1
    (session_id,) = sessions
    assert session_id.startswith("mcp_")  # the in-memory client's clientInfo.name is "mcp"
    assert [c.kwargs["data"] for c in fake.remember.await_args_list] == [
        "first turn",
        "second turn",
    ]
