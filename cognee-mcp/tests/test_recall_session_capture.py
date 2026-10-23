"""recall records each query as a turn of the MCP conversation (SDK-1121).

MCP clients such as Claude Desktop have no hook surface, so the only time the
server sees the conversation is a tool call. recall already runs on every
turn, so it stores its query, as a question, in a session derived from the
request context and bound to the client's default dataset; once the
conversation has been silent for the idle gap the server improves it. No
model-visible cost: no extra tool, and the write never blocks the search.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Client
from mcp.server.lowlevel.server import request_ctx

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from src import server  # noqa: E402
from src.cognee_client import CogneeClient  # noqa: E402

T0 = 1_700_000_000.0


def _fake_client(**overrides):
    attrs = {
        "recall": AsyncMock(return_value=[{"text": "hit"}]),
        "get_recall_state": AsyncMock(),
        "remember": AsyncMock(return_value={"status": "stored"}),
        "remember_qa": AsyncMock(return_value={"status": "session_stored"}),
        "improve": AsyncMock(return_value={"status": "completed", "stages": []}),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class _Session:
    """Stands in for mcp's ServerSession: carries clientInfo."""

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


@pytest.fixture
def clock(monkeypatch):
    """Frozen wall clock for the conversation tracker; call it to move time."""
    state = {"now": T0}
    monkeypatch.setattr(server, "_now", lambda: state["now"])

    def advance(seconds):
        state["now"] += seconds

    return advance


@pytest.fixture
def idle(monkeypatch):
    """Set the idle gap (seconds) that ends a conversation and triggers its improve."""

    def set_idle(seconds):
        monkeypatch.setenv("COGNEE_MCP_SESSION_IDLE_SECONDS", str(seconds))

    return set_idle


async def _drain():
    if server._background_tasks:
        await asyncio.gather(*server._background_tasks, return_exceptions=True)


async def _let_timers_fire(seconds=0.05):
    """Give call_later timers, and the tasks they spawn, a chance to run."""
    await asyncio.sleep(seconds)
    await _drain()


def _state_file(client="claude_desktop"):
    return server._conversation_state_dir() / f"{client}.json"


def _state(client="claude_desktop"):
    return json.loads(_state_file(client).read_text())


# --- session id derivation -------------------------------------------------------


def test_no_request_context_means_no_session():
    assert server._derived_session_id() is None
    assert server._derived_session_id(touch=True) is None


def test_http_uses_transport_session_id_prefixed_with_client(in_request):
    in_request(headers={"mcp-session-id": "1d224079f190443b86d6c02b0024fcf9"})
    assert server._derived_session_id() == "claude_desktop_1d224079f190443b86d6c02b0024fcf9"


def test_unknown_client_falls_back_to_mcp_prefix(in_request):
    in_request(session=_Session(client_name=None), headers={"mcp-session-id": "abc"})
    assert server._derived_session_id() == "mcp_abc"


def test_env_flag_disables_derivation(monkeypatch, in_request):
    monkeypatch.setenv("COGNEE_MCP_SESSION_CAPTURE", "false")
    in_request(headers={"mcp-session-id": "abc"})
    assert server._derived_session_id(touch=True) is None


def test_stdio_has_one_conversation_per_client(in_request, clock):
    """Without a transport id the conversation belongs to the client, across connections."""
    in_request(session=_Session())
    assert server._derived_session_id() is None  # a read never starts one

    first = server._derived_session_id(touch=True)
    assert first.startswith("claude_desktop_") and len(first) > len("claude_desktop_")
    assert server._derived_session_id() == first

    in_request(session=_Session())  # another connection of the same client
    assert server._derived_session_id(touch=True) == first

    in_request(session=_Session(client_name="Cursor"))
    assert server._derived_session_id(touch=True).startswith("cursor_")


def test_stdio_conversation_is_remembered_across_a_restart(in_request, clock):
    in_request()
    first = server._derived_session_id(touch=True)
    assert _state()["session_id"] == first

    server._conversations.clear()  # a new server process starts with an empty memory
    assert server._derived_session_id(touch=True) == first


def test_unreadable_state_falls_back_to_a_fresh_conversation(in_request, clock):
    path = _state_file()
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    in_request()

    session_id = server._derived_session_id(touch=True)

    assert session_id.startswith("claude_desktop_")
    assert _state()["session_id"] == session_id  # the broken file was replaced


@pytest.mark.asyncio
async def test_stdio_conversation_ends_after_the_idle_gap(monkeypatch, in_request, clock, idle):
    idle(100)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request()
    first = server._derived_session_id(touch=True)
    server._mark_turn_captured(first, "claude_desktop_memory")

    clock(100)
    assert server._derived_session_id(touch=True) == first  # at the gap: still the same
    clock(1)
    assert server._derived_session_id() == first  # a read after the gap names the ended one
    second = server._derived_session_id(touch=True)

    assert second != first
    assert _state()["session_id"] == second
    # The ended conversation still held undistilled turns: it is improved right away
    # and, once that succeeded, dropped from the client's pending list.
    assert _state()["pending"] == [{"session_id": first, "dataset_name": "claude_desktop_memory"}]
    await _let_timers_fire()
    fake.improve.assert_awaited_once_with(dataset_name="claude_desktop_memory", session_ids=[first])
    assert _state()["pending"] == []


@pytest.mark.asyncio
async def test_idle_zero_keeps_one_conversation_and_never_improves(
    monkeypatch, in_request, clock, idle
):
    idle(0)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request()
    first = server._derived_session_id(touch=True)
    server._mark_turn_captured(first, "claude_desktop_memory")

    clock(10**6)

    assert server._derived_session_id(touch=True) == first
    assert not server._idle_timers
    await _let_timers_fire()
    fake.improve.assert_not_awaited()


# --- recall stores the turn ------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_stores_the_query_as_a_question_in_the_clients_dataset(
    monkeypatch, in_request
):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "claude_desktop_memory")
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.recall("what did we decide about pricing?")
    await _drain()

    assert result[0].text.endswith("hit")
    fake.remember_qa.assert_awaited_once_with(
        question="what did we decide about pricing?",
        dataset_name="claude_desktop_memory",
        session_id="claude_desktop_abc",
    )
    # The search itself is unchanged: no derived session is injected into it.
    assert fake.recall.call_args.kwargs["session_id"] is None
    # Not registered as an indexing job, or an empty recall would say "still indexing".
    assert not server._background_task_datasets
    # The conversation now waits for its idle gap.
    assert "claude_desktop_abc" in server._idle_timers


@pytest.mark.asyncio
async def test_capture_starts_only_after_the_search_finished(monkeypatch, in_request):
    """On a fresh install the first search and the first write both create the
    default user; the loser raises IntegrityError on users.email and the turn is
    lost (seen end to end over stdio). So the write is scheduled after the search."""
    fake = _fake_client()
    remember_scheduled_during_search = []

    async def recall(**kwargs):
        remember_scheduled_during_search.append(bool(server._background_tasks))
        return [{"text": "hit"}]

    fake.recall = recall
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("q")
    await _drain()

    assert remember_scheduled_during_search == [False]
    fake.remember_qa.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_search_stores_nothing(monkeypatch, in_request):
    fake = _fake_client(recall=AsyncMock(side_effect=RuntimeError("search down")))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.recall("q")
    await _drain()

    assert "Recall failed" in result[0].text
    fake.remember_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_scoped_to_other_datasets_still_captures_into_the_clients_dataset(
    monkeypatch, in_request
):
    """A session belongs to one dataset: the entry route binds it on the first write
    and refuses another, so the turn never follows the search's dataset scope."""
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "claude_desktop_memory")
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("q", datasets="project_a,project_b")
    await _drain()

    assert fake.recall.call_args.kwargs["datasets"] == ["project_a", "project_b"]
    assert fake.remember_qa.call_args.kwargs["dataset_name"] == "claude_desktop_memory"


@pytest.mark.asyncio
async def test_recall_store_false_writes_nothing(monkeypatch, in_request):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("off the record", store=False)
    await _drain()

    fake.remember_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_with_its_own_session_id_is_not_captured(monkeypatch, in_request):
    """A caller that manages a cognee session owns that session's turns."""
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("q", session_id="my-session")
    await _drain()

    assert fake.recall.call_args.kwargs["session_id"] == "my-session"
    fake.remember_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_code_query_is_not_captured(monkeypatch, in_request):
    """A structured code operation is not a question anyone asked."""
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    await server.recall("ingest_data", scope="code", code_query={"operation": "explore"})
    await _drain()

    fake.remember_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_outside_mcp_request_writes_nothing(monkeypatch):
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)

    await server.recall("q")
    await _drain()

    fake.remember_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_failure_never_breaks_recall_and_quiets_down(monkeypatch, in_request):
    fake = _fake_client(remember_qa=AsyncMock(side_effect=RuntimeError("cache down")))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    for _ in range(3):
        result = await server.recall("q")
        await _drain()
        assert result[0].text.endswith("hit")
        assert "Error" not in result[0].text

    assert server._capture_failures == 3
    assert not server._idle_timers  # nothing landed, so nothing to improve

    fake.remember_qa.side_effect = None
    await server.recall("q")
    await _drain()

    assert server._capture_failures == 0
    assert "claude_desktop_abc" in server._idle_timers


@pytest.mark.asyncio
async def test_capture_tolerates_client_without_remember_qa(monkeypatch, in_request):
    """Older/partial clients (and the existing recall tests' stubs) lack remember_qa."""
    fake = SimpleNamespace(recall=AsyncMock(return_value=[{"text": "hit"}]))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request(headers={"mcp-session-id": "abc"})

    result = await server.recall("q")
    await _drain()

    assert result[0].text.endswith("hit")


# --- the conversation is distilled after the idle gap ----------------------------


@pytest.mark.asyncio
async def test_turn_is_improved_once_the_conversation_goes_quiet(monkeypatch, in_request, idle):
    idle(0.05)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "claude_desktop_memory")
    in_request()  # stdio: the conversation is persisted

    await server.recall("first turn")
    await server.recall("second turn")
    await _drain()
    session_id = fake.remember_qa.call_args.kwargs["session_id"]
    assert _state()["distilled"] is False
    fake.improve.assert_not_awaited()

    await _let_timers_fire(0.3)

    fake.improve.assert_awaited_once_with(
        dataset_name="claude_desktop_memory", session_ids=[session_id]
    )
    assert _state()["distilled"] is True
    assert not server._idle_timers


@pytest.mark.asyncio
async def test_a_newer_turn_restarts_the_idle_clock(monkeypatch, idle):
    idle(60)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)

    server._schedule_session_improve("s", "ds", delay=60)
    first = server._idle_timers["s"]
    server._schedule_session_improve("s", "ds", delay=60)
    second = server._idle_timers["s"]

    assert first.cancelled() and second is not first
    fake.improve.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_after_the_gap_starts_a_new_conversation(monkeypatch, in_request, idle):
    """The timer fired and the improve is still running when the next turn arrives:
    that turn opens a new conversation, the old improve finishes undisturbed and is
    not started a second time, and the new conversation is improved after its own gap."""
    idle(0.05)
    release = asyncio.Event()

    async def slow_improve(**kwargs):
        await release.wait()
        return {"status": "completed", "stages": []}

    fake = _fake_client(improve=AsyncMock(side_effect=slow_improve))
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_agent_scoped_default_dataset", lambda: "ds")
    in_request()

    await server.recall("first")
    await asyncio.sleep(0.1)  # the gap passes; the improve is now waiting on `release`
    first = fake.remember_qa.call_args.kwargs["session_id"]
    assert fake.improve.await_count == 1
    await server.recall("second")
    await asyncio.sleep(0.1)  # its own gap passes too
    second = fake.remember_qa.call_args.kwargs["session_id"]
    release.set()
    await _let_timers_fire(0.1)

    assert second != first
    assert [c.kwargs["session_ids"] for c in fake.improve.await_args_list] == [[first], [second]]
    state = _state()
    assert (state["session_id"], state["dataset_name"], state["distilled"], state["pending"]) == (
        second,
        "ds",
        True,
        [],
    )
    assert not server._idle_timers


@pytest.mark.asyncio
async def test_an_ended_conversation_stays_pending_until_an_improve_succeeds(
    monkeypatch, in_request, clock, idle
):
    idle(100)
    fake = _fake_client(improve=AsyncMock(side_effect=RuntimeError("llm down")))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request()
    first = server._derived_session_id(touch=True)
    server._mark_turn_captured(first, "claude_desktop_memory")

    clock(101)
    second = server._derived_session_id(touch=True)  # ends `first`, improves it now
    await _let_timers_fire()

    assert second != first
    fake.improve.assert_awaited_once()
    assert _state()["pending"] == [{"session_id": first, "dataset_name": "claude_desktop_memory"}]

    # The next server start retries it; success takes it off the list.
    fake.improve = AsyncMock(return_value={"status": "completed", "stages": []})
    server._conversations.clear()
    server._resume_conversations()
    await _let_timers_fire()

    fake.improve.assert_awaited_once_with(dataset_name="claude_desktop_memory", session_ids=[first])
    assert _state()["pending"] == []
    assert _state()["session_id"] == second


@pytest.mark.asyncio
async def test_improve_failure_leaves_the_conversation_for_the_next_attempt(
    monkeypatch, in_request, idle
):
    idle(0.05)
    fake = _fake_client(improve=AsyncMock(side_effect=RuntimeError("llm down")))
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request()

    await server.recall("q")
    await _drain()
    await _let_timers_fire(0.3)

    fake.improve.assert_awaited_once()
    assert _state()["distilled"] is False


@pytest.mark.asyncio
async def test_server_start_distills_what_the_last_server_left(monkeypatch, clock, idle):
    idle(100)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    _state_file().parent.mkdir(parents=True)
    _state_file("claude_desktop").write_text(
        json.dumps(
            {
                "session_id": "claude_desktop_old",
                "dataset_name": "claude_desktop_memory",
                "last_activity": T0 - 1000,
                "distilled": False,
            }
        )
    )
    _state_file("cursor").write_text(
        json.dumps(
            {
                "session_id": "cursor_done",
                "dataset_name": "cursor_memory",
                "last_activity": T0 - 1000,
                "distilled": True,
            }
        )
    )

    server._resume_conversations()
    await _let_timers_fire()

    fake.improve.assert_awaited_once_with(
        dataset_name="claude_desktop_memory", session_ids=["claude_desktop_old"]
    )
    assert _state()["distilled"] is True


@pytest.mark.asyncio
async def test_server_start_waits_out_the_rest_of_a_live_gap(monkeypatch, clock, idle):
    idle(100)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    _state_file().parent.mkdir(parents=True)
    _state_file().write_text(
        json.dumps(
            {
                "session_id": "claude_desktop_live",
                "dataset_name": "ds",
                "last_activity": T0 - 10,
                "distilled": False,
            }
        )
    )

    server._resume_conversations()
    await _let_timers_fire()

    fake.improve.assert_not_awaited()
    assert "claude_desktop_live" in server._idle_timers


@pytest.mark.asyncio
async def test_shutdown_drops_sleeping_timers_but_keeps_the_conversation_undistilled(
    monkeypatch, in_request, idle
):
    idle(60)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request()
    session_id = server._derived_session_id(touch=True)
    server._mark_turn_captured(session_id, "ds")
    assert session_id in server._idle_timers

    server._cancel_idle_timers()
    await _let_timers_fire()

    assert not server._idle_timers
    fake.improve.assert_not_awaited()
    assert _state()["distilled"] is False


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
async def test_improve_default_reaches_the_conversation_that_just_ended(
    monkeypatch, in_request, clock, idle
):
    idle(100)
    fake = _fake_client()
    monkeypatch.setattr(server, "cognee_client", fake)
    in_request()
    session_id = server._derived_session_id(touch=True)
    server._mark_turn_captured(session_id, "claude_desktop_memory")
    clock(500)

    await server.improve()

    assert fake.improve.call_args.kwargs["session_ids"] == [session_id]
    assert fake.improve.call_args.kwargs["dataset_name"] == "claude_desktop_memory"


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

    sessions = {call.kwargs["session_id"] for call in fake.remember_qa.await_args_list}
    assert fake.remember_qa.await_count == 2
    assert len(sessions) == 1
    (session_id,) = sessions
    assert session_id.startswith("mcp_")  # the in-memory client's clientInfo.name is "mcp"
    assert [c.kwargs["question"] for c in fake.remember_qa.await_args_list] == [
        "first turn",
        "second turn",
    ]
    assert {c.kwargs["dataset_name"] for c in fake.remember_qa.await_args_list} == {"mcp_memory"}
    assert _state("mcp")["session_id"] == session_id


# --- CogneeClient.remember_qa ----------------------------------------------------


@pytest.mark.asyncio
async def test_cognee_client_api_remember_qa_posts_a_question_side_entry():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "session_stored", "entry_id": "qa-1"})

    client = CogneeClient(api_url="http://cognee.local", api_token="token")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        result = await client.remember_qa(
            question="what did we decide?", dataset_name="ds", session_id="claude_desktop_abc"
        )
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember/entry"
    assert json.loads(requests[0].content.decode()) == {
        "entry": {"type": "qa", "question": "what did we decide?", "answer": "", "context": ""},
        "dataset_name": "ds",
        "session_id": "claude_desktop_abc",
        "self_improvement": False,
    }
    assert result["entry_id"] == "qa-1"


@pytest.mark.asyncio
async def test_cognee_client_direct_remember_qa_passes_a_qa_entry(monkeypatch):
    from cognee.memory import QAEntry

    client = CogneeClient()
    user = object()
    remember = AsyncMock(return_value=SimpleNamespace(status="session_stored", entry_id="qa-1"))
    monkeypatch.setattr(client, "cognee", SimpleNamespace(remember=remember))
    monkeypatch.setattr(client, "_default_user", AsyncMock(return_value=user))

    result = await client.remember_qa(question="q", dataset_name="ds", session_id="s")

    (entry,) = remember.call_args.args
    assert isinstance(entry, QAEntry)
    assert (entry.question, entry.answer) == ("q", "")
    assert remember.call_args.kwargs == {"dataset_name": "ds", "session_id": "s", "user": user}
    assert result == {
        "status": "session_stored",
        "dataset_name": "ds",
        "session_id": "s",
        "entry_id": "qa-1",
    }
