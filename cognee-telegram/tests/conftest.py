"""Shared fixtures. Patches cognee's memory API so tests are deterministic and
need no real LLM/API keys (the #3601 spirit, applied at the cognee boundary)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_cognee(monkeypatch):
    """Replace cognee.remember/recall/forget with AsyncMocks.

    The adapter calls ``cognee.<fn>`` at call time, so patching the module
    attributes is enough — no real ingestion, no keys.
    """
    import cognee

    remember = AsyncMock(return_value={"status": "session_stored"})
    recall = AsyncMock(return_value=[])
    forget = AsyncMock(return_value={"datasets_removed": 1})
    monkeypatch.setattr(cognee, "remember", remember)
    monkeypatch.setattr(cognee, "recall", recall)
    monkeypatch.setattr(cognee, "forget", forget)
    return SimpleNamespace(remember=remember, recall=recall, forget=forget)


@pytest.fixture
def graph_result():
    """Build a real ``ResponseGraphEntry`` (the shape recall actually returns)."""
    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry
    from cognee.modules.recall.types.SearchResultItem import SearchResultKind
    from cognee.modules.search.types.SearchType import SearchType

    def _make(text: str, source: str = "graph"):
        return ResponseGraphEntry(
            kind=SearchResultKind.GRAPH_COMPLETION,
            search_type=SearchType.GRAPH_COMPLETION,
            text=text,
            source=source,
        )

    return _make


@pytest.fixture
def session_result():
    """Build a real ``ResponseQAEntry`` (a session-cache hit)."""
    from cognee.modules.recall.types.RecallResponse import ResponseQAEntry

    def _make(answer: str):
        return ResponseQAEntry(
            time="2026-01-01T00:00:00", question="", context="", answer=answer, source="session"
        )

    return _make
