"""Deterministic, mocked tests for the web-widget chat-memory adapter.

These run in CI with no real LLM keys: every cognee memory call
(``remember`` / ``recall`` / ``forget`` / session manager) is monkeypatched
so the test asserts the adapter's *behavior* — session scoping, background
ingestion, opt-out, citations, and per-conversation forget — without
touching a provider. This mirrors the mocked-LLM harness in issue #3601.

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
from citations import extract_citations  # noqa: E402


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
    """Record every cognee memory call the adapter makes."""
    calls = {"remember": [], "recall": [], "forget": []}

    async def fake_remember(data, **kwargs):
        calls["remember"].append({"data": data, **kwargs})
        return {"status": "completed"}

    async def fake_recall(**kwargs):
        calls["recall"].append(kwargs)
        # Canned graph answer with an explicit citation to a docs chunk.
        return [
            {
                "source": "graph",
                "text": "Cognee stores memory as a knowledge graph.",
                "score": 0.91,
                "dataset_name": "web:demo:docs",
                "metadata": {
                    "references": [
                        {
                            "snippet": "Cognee turns raw data into a knowledge graph.",
                            "data_id": "doc-1",
                            "score": 0.91,
                        }
                    ]
                },
            }
        ]

    async def fake_forget(**kwargs):
        calls["forget"].append(kwargs)
        return {"removed": 1}

    monkeypatch.setattr(adapter_mod.cognee, "remember", fake_remember)
    monkeypatch.setattr(adapter_mod.cognee, "recall", fake_recall)
    monkeypatch.setattr(adapter_mod.cognee, "forget", fake_forget)

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


def test_answer_returns_text_and_citations(cognee_calls):
    calls, _ = cognee_calls
    adapter = ChatMemoryAdapter(top_k=5)
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    answer = asyncio.run(adapter.answer(conversation=conv, query="What is cognee?"))

    assert answer.text == "Cognee stores memory as a knowledge graph."
    assert answer.session_id == "web:demo:v1:c1"
    assert len(answer.citations) == 1
    assert answer.citations[0].reference == "doc-1"

    # recall must be session-scoped, docs-scoped, and ask for references.
    recall_kwargs = calls["recall"][0]
    assert recall_kwargs["session_id"] == "web:demo:v1:c1"
    assert recall_kwargs["datasets"] == ["web:demo:docs"]
    assert recall_kwargs["include_references"] is True
    assert recall_kwargs["top_k"] == 5


def test_ingest_opt_in_uses_background_session(cognee_calls):
    calls, _ = cognee_calls
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    stored = asyncio.run(adapter.ingest(conversation=conv, message="hello", role="user"))

    assert stored is True
    remember_kwargs = calls["remember"][0]
    assert remember_kwargs["session_id"] == "web:demo:v1:c1"
    assert remember_kwargs["run_in_background"] is True
    assert remember_kwargs["self_improvement"] is False
    assert remember_kwargs["data"] == "user: hello"


def test_ingest_opt_out_skips_remember(cognee_calls):
    calls, _ = cognee_calls
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    stored = asyncio.run(
        adapter.ingest(conversation=conv, message="secret", role="user", opt_in=False)
    )

    assert stored is False
    assert calls["remember"] == []


def test_forget_clears_only_this_conversation(cognee_calls):
    _, session_manager = cognee_calls
    adapter = ChatMemoryAdapter()
    conv = adapter.conversation(site_id="demo", visitor_id="v1", conversation_id="c1")

    cleared = asyncio.run(adapter.forget(conversation=conv))

    assert cleared is True
    assert session_manager.deleted == [("test-user", "web:demo:v1:c1")]


def test_extract_citations_fallback_when_no_references():
    # No explicit references -> cite the entry text so replies stay attributable.
    results = [{"source": "session", "answer": "You told me your name is Ada."}]
    citations = extract_citations(results)
    assert len(citations) == 1
    assert citations[0].source == "session"
    assert "Ada" in citations[0].snippet
