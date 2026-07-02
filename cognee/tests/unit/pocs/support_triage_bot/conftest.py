"""Shared test fixtures for the support-triage bot tests.

All tests are fully deterministic — zero real LLM keys, zero platform
tokens, zero network calls. Uses unittest.mock.patch to mock cognee APIs.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Ensure the bot package is importable from test modules.
# This must happen before any test module tries `from config import ...`.
_BOT_ROOT = str(
    Path(__file__).resolve().parents[5] / "examples" / "pocs" / "support_triage_bot"
)
if _BOT_ROOT not in sys.path:
    sys.path.insert(0, _BOT_ROOT)


def _import_bot_module(name: str):
    """Import a module from the bot package by name."""
    return importlib.import_module(name)


@pytest.fixture
def sample_resolved_threads():
    """Three pre-seeded resolved support threads."""
    models = _import_bot_module("models")
    SupportThread = models.SupportThread

    return [
        SupportThread(
            thread_id="T001",
            channel_id="support",
            reporter="alice",
            problem_summary="Auth timeout after token refresh",
            resolution_summary="Bumped token TTL from 1h to 24h in auth config",
            messages=[
                "Users are getting 401 errors after token refresh",
                "Looks like the token TTL is too short at 1 hour",
                "Fixed by bumping TTL to 24h",
            ],
            resolved_at=datetime(2026, 5, 1),
            thread_url="https://support.example.com/T001",
        ),
        SupportThread(
            thread_id="T002",
            channel_id="support",
            reporter="bob",
            problem_summary="Session expiry on mobile app",
            resolution_summary="Same root cause — mobile SDK used shorter TTL. Aligned to 24h.",
            messages=[
                "Mobile sessions expire too fast",
                "Mobile SDK TTL was 30min vs web 1h",
                "Aligned mobile TTL with web (24h)",
            ],
            resolved_at=datetime(2026, 5, 15),
            thread_url="https://support.example.com/T002",
        ),
        SupportThread(
            thread_id="T003",
            channel_id="support",
            reporter="carol",
            problem_summary="Database connection pool exhaustion",
            resolution_summary="Increased PgBouncer pool from 20 to 100 and added 30s timeout",
            messages=[
                "Getting too many connections errors during peak",
                "PgBouncer pool was set to 20",
                "Increased to 100, added 30s timeout and alerts",
            ],
            resolved_at=datetime(2026, 6, 1),
            thread_url="https://support.example.com/T003",
        ),
    ]


@pytest.fixture
def new_support_issue():
    """A new inbound support issue query."""
    return "Users are getting auth timeout errors after they refresh their session tokens"


@pytest.fixture
def unrelated_query():
    """A query completely unrelated to the seeded threads."""
    return "How do I cook a perfect pasta carbonara?"


@pytest.fixture
def bot_config():
    """A BotConfig with test defaults."""
    config_mod = _import_bot_module("config")
    return config_mod.BotConfig(
        dataset_name="test_support_threads",
        memory_scope="channel",
        top_k=5,
        min_relevance_score=0.0,
    )


def _make_mock_remember_result(
    *,
    status="completed",
    dataset_name="test_support_threads",
    items=None,
):
    """Create a mock RememberResult object."""
    result = MagicMock()
    result.status = status
    result.dataset_name = dataset_name
    result.dataset_id = str(uuid4())
    result.items_processed = len(items) if items else 0
    result.items = items or []
    return result


def _make_mock_recall_results(threads=None):
    """Create a list of mock RecallResponse (ResponseGraphEntry) objects."""
    if threads is None:
        return []

    results = []
    for i, thread in enumerate(threads):
        entry = SimpleNamespace(
            text="{}: {}".format(thread.problem_summary, thread.resolution_summary),
            score=0.95 - (i * 0.1),
            metadata={
                "thread_id": thread.thread_id,
                "thread_url": thread.thread_url,
            },
            source="graph",
            kind="graph_completion",
            search_type="GRAPH_COMPLETION",
        )
        results.append(entry)
    return results


@pytest.fixture
def mock_cognee(sample_resolved_threads):
    """Mock cognee.remember, cognee.recall, cognee.forget for deterministic tests."""
    data_id_1 = uuid4()
    data_id_2 = uuid4()

    remember_result = _make_mock_remember_result(
        items=[{"id": str(data_id_1), "name": "support_doc"}]
    )
    recall_results = _make_mock_recall_results(sample_resolved_threads[:2])
    forget_result = {
        "data_id": str(data_id_1),
        "dataset_id": str(uuid4()),
        "status": "success",
    }

    with patch("cognee.remember", new_callable=AsyncMock, return_value=remember_result) as mock_rem:
        with patch("cognee.recall", new_callable=AsyncMock, return_value=recall_results) as mock_rec:
            with patch("cognee.forget", new_callable=AsyncMock, return_value=forget_result) as mock_fgt:
                yield {
                    "remember": mock_rem,
                    "recall": mock_rec,
                    "forget": mock_fgt,
                    "remember_result": remember_result,
                    "recall_results": recall_results,
                    "forget_result": forget_result,
                    "data_id_1": data_id_1,
                    "data_id_2": data_id_2,
                }


@pytest.fixture
def mock_cognee_empty():
    """Mock cognee APIs with empty recall results (no matches)."""
    remember_result = _make_mock_remember_result(items=[])
    recall_results = []
    forget_result = {"status": "success"}

    with patch("cognee.remember", new_callable=AsyncMock, return_value=remember_result) as mock_rem:
        with patch("cognee.recall", new_callable=AsyncMock, return_value=recall_results) as mock_rec:
            with patch("cognee.forget", new_callable=AsyncMock, return_value=forget_result) as mock_fgt:
                yield {
                    "remember": mock_rem,
                    "recall": mock_rec,
                    "forget": mock_fgt,
                }
