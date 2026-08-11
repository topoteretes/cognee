"""Recall records the questions it answers, the same way /v1/search does.

Recall calls ``authorized_search`` directly, so it skips the logging that
``search()`` wraps around that call. Agents recall through this endpoint, so
without this their questions never reach recall history at all.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.modules.search.operations as search_operations
from cognee.api.v1.recall.recall import recall
from cognee.modules.search.types import SearchType

# cognee.modules.search.methods re-exports `search` the function, which shadows
# the module of the same name for dotted-path patching.
search_mod = importlib.import_module("cognee.modules.search.methods.search")
recall_mod = importlib.import_module("cognee.api.v1.recall.recall")


@pytest.fixture
def logged_queries(monkeypatch):
    """Capture log_query calls and stub out everything recall needs around them."""
    calls = []

    async def fake_log_query(query_text, query_type, user_id, dataset_id=None):
        calls.append(
            {
                "query_text": query_text,
                "query_type": query_type,
                "user_id": user_id,
                "dataset_id": dataset_id,
            }
        )
        return SimpleNamespace(id=uuid4())

    async def fake_authorized_search(**_kwargs):
        return []

    async def fake_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(search_operations, "log_query", fake_log_query)
    monkeypatch.setattr(search_mod, "authorized_search", fake_authorized_search)
    monkeypatch.setattr(
        recall_mod, "set_session_user_context_variable", fake_set_session_user_context_variable
    )
    return calls


@pytest.mark.asyncio
async def test_recall_logs_the_question(logged_queries):
    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()

    await recall(
        "how do I rotate my API keys?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[dataset_id],
        auto_route=False,
        user=user,
    )

    assert len(logged_queries) == 1
    logged = logged_queries[0]
    assert logged["query_text"] == "how do I rotate my API keys?"
    assert logged["query_type"] == SearchType.GRAPH_COMPLETION.value
    assert logged["user_id"] == user.id
    assert logged["dataset_id"] == dataset_id


@pytest.mark.asyncio
async def test_recall_leaves_dataset_id_null_when_it_fans_out(logged_queries):
    """Two datasets have no single dataset to attribute the question to."""
    user = SimpleNamespace(id=uuid4())

    await recall(
        "what changed last week?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4(), uuid4()],
        auto_route=False,
        user=user,
    )

    assert len(logged_queries) == 1
    assert logged_queries[0]["dataset_id"] is None


@pytest.mark.asyncio
async def test_recall_survives_a_logging_failure(logged_queries, monkeypatch):
    """A failed insert must never cost the caller their answer."""

    async def exploding_log_query(*_args, **_kwargs):
        raise RuntimeError("history database is down")

    monkeypatch.setattr(search_operations, "log_query", exploding_log_query)

    results = await recall(
        "does this still answer?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user_stub(),
    )

    assert results == []


def user_stub():
    return SimpleNamespace(id=uuid4())
