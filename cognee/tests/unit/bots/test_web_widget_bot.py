"""Deterministic, mocked tests for the web-widget chat-memory adapter.

These run in CI with no real LLM keys: every cognee memory call
(``recall`` / ``remember`` / ``forget`` / session manager) is monkeypatched so
the tests assert the adapter's *behavior* — session scoping, opt-out, citation
parsing, and per-conversation forget — without touching a provider. This
mirrors the mocked-LLM harness in issue #3601.

Crucially, ``fake_recall`` returns cognee's **real** shape: the answer text with
an appended ``Evidence:`` block (that is how ``include_references=True`` surfaces
sources — see cognee/modules/retrieval/utils/references.py), not a fabricated
structured ``references`` list.

Run just this file::

    uv run pytest cognee/tests/unit/bots/test_web_widget_bot.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Make the web-widget example importable (it ships its own thin adapter).
WIDGET_DIR = Path(__file__).resolve().parents[4] / "examples" / "bots" / "web_widget"
sys.path.insert(0, str(WIDGET_DIR))

import adapter as adapter_mod  # noqa: E402
from adapter import ChatMemoryAdapter  # noqa: E402
from citations import split_evidence  # noqa: E402
from cognee.modules.data.exceptions import DatasetNotFoundError  # noqa: E402

# A graph completion exactly as recall(include_references=True) returns it:
# the answer prose followed by an appended, grounded "Evidence:" block.
GRAPH_ANSWER = "Cognee stores memory as a knowledge graph."
GRAPH_ENTRY = {
    "source": "graph",
    "dataset_name": "web:demo:docs",
    "score": 0.91,
    "text": (
        f"{GRAPH_ANSWER}\n\n"
        "Evidence:\n"
        "- chunk 1 of document guide.md (data_id: d1, chunk_id: c1): "
        '"Cognee turns raw data into a knowledge graph."'
    ),
}


class _FakeUser:
    id = "test-user"


class _FakeSessionManager:
    def __init__(self):
        self.deleted = []

    async def delete_session(self, *, user_id, session_id):
        self.deleted.append((user_id, session_id))
        return True


@pytest.fixture
def cognee_calls(monkeypatch):
    """Record every cognee memory call the adapter makes; recall returns a
    realistic graph completion with an Evidence block."""
    calls = {"remember": [], "recall": []}

    async def fake_remember(data, **kwargs):
        calls["remember"].append({"data": data, **kwargs})
        return {"status": "session_stored"}

    async def fake_recall(**kwargs):
        calls["recall"].append(kwargs)
        return [GRAPH_ENTRY]

    monkeypatch.setattr(adapter_mod.cognee, "remember", fake_remember)
    monkeypatch.setattr(adapter_mod.cognee, "recall", fake_recall)

    session_manager = _FakeSessionManager()
    monkeypatch.setattr(adapter_mod, "get_session_manager", lambda: session_manager)

    async def fake_default_user():
        return _FakeUser()

    monkeypatch.setattr(adapter_mod, "get_default_user", fake_default_user)

    return calls, session_manager


def test_session_id_convention():
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="acme", visitor_id="v1", conversation_id="c1")
    assert conv.session_id == "web:acme:v1:c1"
    assert adapter.docs_dataset("acme") == "web:acme:docs"


def test_split_evidence_parses_bullets_and_strips_block():
    prose, citations = split_evidence(GRAPH_ENTRY["text"])
    # The Evidence block is stripped from the prose shown to the user.
    assert prose == GRAPH_ANSWER
    assert "Evidence:" not in prose
    # ...and each bullet becomes a citation with its document + ids.
    assert len(citations) == 1
    assert citations[0].document == "guide.md"
    assert citations[0].data_id == "d1"
    assert citations[0].chunk_id == "c1"
    assert citations[0].snippet == "Cognee turns raw data into a knowledge graph."


def test_split_evidence_without_block_yields_no_citations():
    prose, citations = split_evidence("You told me your name is Ada.")
    assert prose == "You told me your name is Ada."
    assert citations == []


def test_answer_returns_clean_text_and_citations(cognee_calls):
    calls, _ = cognee_calls
    adapter = ChatMemoryAdapter(top_k=5)
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    answer = asyncio.run(adapter.answer(conversation=conv, query="What is cognee?"))

    # The raw Evidence block never leaks into the displayed answer.
    assert answer.text == GRAPH_ANSWER
    assert "Evidence:" not in answer.text
    assert answer.session_id == "web:demo:v1:c1"
    assert [c.document for c in answer.citations] == ["guide.md"]
    assert answer.as_dict()["answer"] == GRAPH_ANSWER

    # recall must be session-scoped, docs-scoped, and ask for references.
    recall_kwargs = calls["recall"][0]
    assert recall_kwargs["session_id"] == "web:demo:v1:c1"
    assert recall_kwargs["datasets"] == ["web:demo:docs"]
    assert recall_kwargs["include_references"] is True
    assert recall_kwargs["top_k"] == 5


def test_answer_prefers_generated_completion_over_session_turns(monkeypatch, cognee_calls):
    """A prior/echoed session turn must never be shown as the answer."""
    calls, _ = cognee_calls

    async def fake_recall(**kwargs):
        calls["recall"].append(kwargs)
        # recall returns session entries *before* the graph completion.
        return [
            {"source": "session", "answer": "user: What is cognee?"},
            {"source": "session", "answer": "A stale answer from a previous turn."},
            GRAPH_ENTRY,
        ]

    monkeypatch.setattr(adapter_mod.cognee, "recall", fake_recall)
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    answer = asyncio.run(adapter.answer(conversation=conv, query="What is cognee?"))

    assert answer.text == GRAPH_ANSWER  # not the echoed question or stale turn


def test_answer_opt_out_recalls_without_session(cognee_calls):
    calls, _ = cognee_calls
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    asyncio.run(adapter.answer(conversation=conv, query="hi", remember=False, use_docs=False))

    recall_kwargs = calls["recall"][0]
    assert recall_kwargs["session_id"] is None  # nothing is persisted
    assert recall_kwargs["datasets"] is None  # docs mode off


def test_answer_falls_back_when_docs_dataset_missing(monkeypatch, cognee_calls):
    """A never-seeded docs dataset degrades to a session/graph recall, not a 500."""
    calls, _ = cognee_calls

    async def fake_recall(**kwargs):
        calls["recall"].append(kwargs)
        if kwargs["datasets"]:
            raise DatasetNotFoundError(message="No datasets found.")
        return [GRAPH_ENTRY]

    monkeypatch.setattr(adapter_mod.cognee, "recall", fake_recall)
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    answer = asyncio.run(adapter.answer(conversation=conv, query="What is cognee?"))

    assert answer.text == GRAPH_ANSWER
    assert calls["recall"][0]["datasets"] == ["web:demo:docs"]  # tried docs first
    assert calls["recall"][1]["datasets"] is None  # then fell back


def test_forget_clears_only_this_conversation(cognee_calls):
    _, session_manager = cognee_calls
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    cleared = asyncio.run(adapter.forget(conversation=conv))

    assert cleared is True
    assert session_manager.deleted == [("test-user", "web:demo:v1:c1")]


# --- Server flow (thin FastAPI glue over the adapter) -----------------------


@pytest.fixture
def client(cognee_calls, monkeypatch):
    """A TestClient over the widget server with cognee mocked at the boundary."""
    from fastapi.testclient import TestClient

    import server as server_mod

    calls, session_manager = cognee_calls
    monkeypatch.setattr(server_mod.adapter, "top_k", 8)
    with TestClient(server_mod.app) as test_client:
        yield test_client, calls, session_manager


def test_chat_endpoint_returns_answer_and_citations(client):
    test_client, _, _ = client
    resp = test_client.post(
        "/api/chat", json={"message": "What is cognee?", "conversation_id": "c1"}
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["answer"] == GRAPH_ANSWER
    assert body["session_id"] == "web:demo:anonymous:c1"
    assert [c["document"] for c in body["citations"]] == ["guide.md"]


def test_chat_forget_command_is_not_answered(client):
    test_client, calls, session_manager = client
    before = len(calls["recall"])
    resp = test_client.post("/api/chat", json={"message": "/forget", "conversation_id": "c1"})
    assert resp.status_code == 200
    assert resp.json()["citations"] == []
    # A /forget clears the session and is NOT sent through recall.
    assert session_manager.deleted[-1] == ("test-user", "web:demo:anonymous:c1")
    assert len(calls["recall"]) == before


def test_forget_endpoint_clears_conversation(client):
    test_client, _, session_manager = client
    resp = test_client.post("/api/forget", json={"conversation_id": "c1"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["cleared"] is True
    assert session_manager.deleted[-1] == ("test-user", "web:demo:anonymous:c1")
