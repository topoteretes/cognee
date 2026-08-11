"""Recall records the questions it answers, the same way /v1/search does.

Recall calls ``authorized_search`` directly, so it skips the logging that
``search()`` wraps around that call. Agents recall through this endpoint, so
without this their questions never reach recall history at all.

A search fans out over every dataset it is given, so one recall produces one
query and one result row *per dataset* — collapsing them loses which dataset
was searched and which one answered.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.modules.search.operations as search_operations
from cognee.api.v1.recall.recall import recall
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

# cognee.modules.search.methods re-exports `search` the function, which shadows
# the module of the same name for dotted-path patching.
search_mod = importlib.import_module("cognee.modules.search.methods.search")
recall_mod = importlib.import_module("cognee.api.v1.recall.recall")
history_mod = importlib.import_module("cognee.modules.search.operations.log_search_history")


def payload(dataset_id, completion):
    return SearchResultPayload(
        dataset_id=dataset_id,
        completion=completion,
        search_type=SearchType.GRAPH_COMPLETION,
    )


@pytest.fixture
def history(monkeypatch):
    """Capture the rows log_search_history would write."""
    rows = {"queries": [], "results": []}

    async def fake_log_query(query_text, query_type, user_id, dataset_id=None):
        query = SimpleNamespace(id=uuid4())
        rows["queries"].append(
            {
                "id": query.id,
                "query_text": query_text,
                "query_type": query_type,
                "user_id": user_id,
                "dataset_id": dataset_id,
            }
        )
        return query

    async def fake_log_result(query_id, result, user_id, dataset_id=None):
        rows["results"].append(
            {"query_id": query_id, "value": result, "user_id": user_id, "dataset_id": dataset_id}
        )

    async def fake_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(history_mod, "log_query", fake_log_query)
    monkeypatch.setattr(history_mod, "log_result", fake_log_result)
    monkeypatch.setattr(
        recall_mod, "set_session_user_context_variable", fake_set_session_user_context_variable
    )
    return rows


def stub_search(monkeypatch, results):
    async def fake_authorized_search(**_kwargs):
        return results

    monkeypatch.setattr(search_mod, "authorized_search", fake_authorized_search)


@pytest.mark.asyncio
async def test_recall_logs_one_query_and_result_per_dataset(history, monkeypatch):
    user = SimpleNamespace(id=uuid4())
    first, second = uuid4(), uuid4()
    stub_search(
        monkeypatch,
        [payload(first, "answer from first"), payload(second, "answer from second")],
    )

    await recall(
        "how do I rotate my API keys?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[first, second],
        auto_route=False,
        user=user,
    )

    assert [row["dataset_id"] for row in history["queries"]] == [first, second]
    assert all(row["query_text"] == "how do I rotate my API keys?" for row in history["queries"])
    assert all(row["query_type"] == SearchType.GRAPH_COMPLETION.value for row in history["queries"])
    assert all(row["user_id"] == user.id for row in history["queries"])

    assert [row["dataset_id"] for row in history["results"]] == [first, second]
    assert [row["value"] for row in history["results"]] == [
        "answer from first",
        "answer from second",
    ]
    # Each result belongs to its own dataset's query, not to a shared one.
    assert [row["query_id"] for row in history["results"]] == [
        row["id"] for row in history["queries"]
    ]


@pytest.mark.asyncio
async def test_recall_records_no_dataset_when_the_payload_has_none(history, monkeypatch):
    """Access control off: the payload carries no dataset, so the row records None."""
    user = SimpleNamespace(id=uuid4())
    stub_search(monkeypatch, [payload(None, "the answer")])

    await recall(
        "what changed last week?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user,
    )

    assert [row["dataset_id"] for row in history["queries"]] == [None]
    assert [row["dataset_id"] for row in history["results"]] == [None]


@pytest.mark.asyncio
async def test_recall_logs_the_question_even_with_no_results(history, monkeypatch):
    """A question nothing could answer is exactly the one worth recording."""
    user = SimpleNamespace(id=uuid4())
    stub_search(monkeypatch, [])

    await recall(
        "anything about SOC 2?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4(), uuid4()],
        auto_route=False,
        user=user,
    )

    assert len(history["queries"]) == 1
    assert history["queries"][0]["dataset_id"] is None
    # The query still keeps its result row, empty — history stays 1:1.
    assert [row["value"] for row in history["results"]] == [""]


@pytest.mark.asyncio
async def test_recall_survives_when_logging_fails(history, monkeypatch, caplog):
    """History is best-effort bookkeeping on the read path: a broken history
    write logs a warning and must never fail the recall that produced it."""

    async def exploding_log_query(*_args, **_kwargs):
        raise RuntimeError("history database is down")

    monkeypatch.setattr(history_mod, "log_query", exploding_log_query)
    stub_search(monkeypatch, [payload(uuid4(), "the answer")])

    results = await recall(
        "does this still answer?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=SimpleNamespace(id=uuid4()),
    )

    assert results  # the answer was returned despite the broken history store
    assert history["queries"] == []  # nothing was recorded
    assert any("Failed to record search history" in r.message for r in caplog.records)
