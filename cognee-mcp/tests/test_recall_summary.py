"""Memory-hit summaries preserve the old body and diagnose only empty results."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastmcp import Client
from src import server
from src.cognee_client import CogneeClient
from src.server_utils import (
    RecallState,
    classify_recall_state,
    format_recall_body,
    format_recall_results,
    recall_items,
)


@pytest.mark.parametrize(
    "payload,count",
    [
        ([{"source": "session", "answer": "A"}, {"source": "graph", "text": "B"}], 2),
        ({"results": [{"text": "A"}]}, 1),
        ("an answer", 1),
        ([], 0),
        (None, 0),
        ({"results": []}, 0),
        ([{"source": "system", "status": "memory_warming_up", "text": "warming"}], 0),
    ],
)
def test_count_matches_entries_and_preserves_body(payload, count):
    assert len(recall_items(payload)) == count
    rendered = format_recall_results(payload)
    assert rendered.splitlines()[0]
    if count:
        assert rendered.splitlines()[0].startswith(f"{count} memor")
        assert rendered.partition("\n")[2] == format_recall_body(payload)


def test_summary_uses_only_returned_source_metadata_and_is_one_line():
    payload = [
        {"_source": "session", "answer": "A"},
        {"source": "graph", "dataset_name": "project\ndocs", "text": "B"},
    ]
    assert format_recall_results(payload).splitlines()[0] == (
        "2 memories found (1 from sessions, 1 from project docs)"
    )


def test_pydantic_results_and_markers():
    from cognee.modules.recall.types.RecallResponse import ResponseMarkerEntry

    marker = ResponseMarkerEntry(
        source="system", status="build_failed", text="Build failed", datapoint_count=0, threshold=1
    )
    assert recall_items([marker]) == []
    assert format_recall_results([marker]).startswith("memory indexing failed")


@pytest.mark.parametrize(
    "state,prefix",
    [
        (RecallState("indexing", 12, 40), "still indexing — 12/40 items processed"),
        (RecallState("indexing"), "still indexing — retry shortly"),
        (RecallState("build_failed"), "memory indexing failed — check cognify_status"),
        (RecallState("none"), "no matching memories"),
    ],
)
def test_explicit_empty_states(state, prefix):
    assert format_recall_results([], empty_state=state).startswith(prefix)


def test_no_pipeline_activity_is_simply_none():
    """No run to report on is "nothing", not a separate diagnosis.

    This used to assert "unknown" for an unavailable graph store, which only
    existed to stop a zero node count being read as "empty". Both states, and
    the graph read that produced them, are gone.
    """
    assert classify_recall_state({}) == RecallState("none")


@pytest.mark.parametrize("completed,total", [(1, None), (None, 40), (-1, 40), (41, 40), (True, 40)])
def test_invalid_progress_does_not_invent_a_fraction(completed, total):
    state = classify_recall_state(
        {
            "ds": {
                "status": "DATASET_PROCESSING_STARTED",
                "progress": {"completed_items": completed, "total_items": total},
            }
        }
    )
    assert state == RecallState("indexing")


@pytest.mark.asyncio
async def test_successful_recall_never_probes_or_changes_query(monkeypatch):
    payload = [{"source": "session", "answer": "cached memory"}]
    fake = SimpleNamespace(recall=AsyncMock(return_value=payload), get_recall_state=AsyncMock())
    monkeypatch.setattr(server, "cognee_client", fake)
    result = await server.recall("query", datasets="a,b", session_id="session", top_k=9)
    assert result[0].text == "1 memory found (1 from sessions)\n[session] cached memory"
    fake.get_recall_state.assert_not_awaited()
    assert fake.recall.call_args.kwargs["datasets"] == ["a", "b"]
    assert fake.recall.call_args.kwargs["top_k"] == 9


@pytest.mark.asyncio
async def test_marker_is_not_a_hit_and_indexing_state_is_checked(monkeypatch):
    marker = {"source": "system", "status": "memory_warming_up", "text": "Graph warming up"}
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[marker]),
        get_recall_state=AsyncMock(return_value=RecallState("indexing", 12, 40)),
    )
    monkeypatch.setattr(server, "cognee_client", fake)
    result = await server.recall("query", datasets="project")
    assert result[0].text.startswith("still indexing — 12/40 items processed")
    assert result[0].text.endswith("[system] Graph warming up")
    # The caller shares its remaining budget so every hop inside is bounded by it.
    fake.get_recall_state.assert_awaited_once_with(["project"], deadline=ANY)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError(), RuntimeError("unavailable")])
async def test_diagnostic_failure_does_not_turn_no_results_into_tool_error(monkeypatch, failure):
    fake = SimpleNamespace(
        recall=AsyncMock(return_value=[]), get_recall_state=AsyncMock(side_effect=failure)
    )
    monkeypatch.setattr(server, "cognee_client", fake)
    result = await server.recall("query")
    assert result[0].text == "no matching memories"


@pytest.mark.asyncio
async def test_mcp_wire_content_stays_text_with_old_body(monkeypatch):
    fake = SimpleNamespace(recall=AsyncMock(return_value=[{"text": "original answer"}]))
    monkeypatch.setattr(server, "cognee_client", fake)
    async with Client(server.mcp) as client:
        response = await client.call_tool("recall", {"query": "question"})
    assert len(response.content) == 1
    assert response.content[0].type == "text"
    assert response.content[0].text == "1 memory found\noriginal answer"
    assert response.content[0].meta == {"cognee/memory": {"count": 1, "state": "found"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["indexing", "none"])
async def test_api_diagnostics_are_scoped_and_bounded(state):
    """One status call, scoped to the named dataset, and no graph read.

    The graph-summary leg existed only to split "nothing" four ways; dropping it
    takes a whole round trip out of every empty recall.
    """
    client = CogneeClient(api_url="http://cognee.test", api_token="token")
    await client.client.aclose()
    seen = []

    def handle(request):
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer token"
        assert not request.url.path.endswith("/graph-summary")
        if request.url.path == "/api/v1/datasets/":
            return httpx.Response(
                200, json=[{"id": "target", "name": "project"}, {"id": "other", "name": "other"}]
            )
        assert request.url.path.endswith("/status/progress")
        assert request.url.params.get_list("dataset") == ["target"]
        assert set(request.url.params.get_list("pipeline")) == {
            "add_pipeline",
            "cognify_pipeline",
            "code_graph_pipeline",
        }
        status = (
            "DATASET_PROCESSING_STARTED" if state == "indexing" else "DATASET_PROCESSING_COMPLETED"
        )
        return httpx.Response(
            200,
            json={
                "target": {
                    "cognify_pipeline": {
                        "status": status,
                        "progress": {"completed_items": 12, "total_items": 40},
                    }
                }
            },
        )

    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        result = await client.get_recall_state(["project"])
        assert result.state == state
        assert len(seen) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_direct_diagnostics_authorize_before_reading(monkeypatch):
    methods = importlib.import_module("cognee.modules.data.methods")
    users = importlib.import_module("cognee.modules.users.methods")
    pipelines = importlib.import_module("cognee.modules.pipelines.operations.get_pipeline_status")
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4())
    authorize = AsyncMock(return_value=[dataset])
    progress = AsyncMock(return_value={})
    graph_counts = AsyncMock()
    monkeypatch.setattr(users, "get_default_user", AsyncMock(return_value=user))
    monkeypatch.setattr(methods, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(methods, "get_datasets_graph_counts", graph_counts)
    monkeypatch.setattr(pipelines, "get_pipeline_progress", progress)
    client = CogneeClient()
    assert (await client.get_recall_state(["project"])).state == "none"
    authorize.assert_awaited_once_with(["project"], "read", user)
    assert all(call.args[0] == [dataset.id] for call in progress.await_args_list)
    # The graph traversal is gone: locally it was a full per-dataset walk on the
    # first empty recall after a cognify, and it only served the four-way split.
    graph_counts.assert_not_awaited()


@pytest.mark.asyncio
async def test_diagnostic_deadline_cancels_probe(monkeypatch):
    cancelled = asyncio.Event()

    async def probe(_datasets, *, deadline=None):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    fake = SimpleNamespace(recall=AsyncMock(return_value=[]), get_recall_state=probe)
    monkeypatch.setattr(server, "cognee_client", fake)
    monkeypatch.setattr(server, "_RECALL_STATE_TIMEOUT_SECONDS", 0.01)
    result = await server.recall("query")
    assert cancelled.is_set()
    # A probe that overran is still just "nothing" to the caller; the separate
    # "status unavailable" wording went with the unknown state.
    assert result[0].text == "no matching memories"


@pytest.mark.asyncio
async def test_queued_ingestion_is_indexing_before_pipeline_record_exists(monkeypatch):
    gate = asyncio.Event()
    fake = SimpleNamespace(
        remember=AsyncMock(side_effect=lambda **kwargs: None),
        recall=AsyncMock(return_value=[]),
        get_recall_state=AsyncMock(return_value=RecallState("no_match")),
    )

    async def remember(**kwargs):
        await gate.wait()

    fake.remember = remember
    monkeypatch.setattr(server, "cognee_client", fake)
    await server.remember("synthetic data", dataset_name="target", background=True)
    tasks = list(server._background_task_datasets)
    try:
        result = await server.recall("query", datasets="target")
        assert result[0].text.startswith("still indexing")
        fake.get_recall_state.assert_not_awaited()
        result = await server.recall("query", datasets="unrelated")
        assert result[0].text == "no matching memories"
    finally:
        gate.set()
        await asyncio.gather(*tasks)
    assert not server._background_task_datasets


def test_completed_run_with_nothing_to_show_is_none():
    """A finished pipeline and an empty result are the same outcome for a caller.

    Previously split into not_indexed vs no_match, which cost a graph-summary
    round trip to tell apart and produced near-identical sentences.
    """
    state = classify_recall_state(
        {"ds": {"add_pipeline": {"status": "DATASET_PROCESSING_COMPLETED", "progress": None}}}
    )
    assert state == RecallState("none")


def test_errored_run_is_still_distinguished_from_none():
    """build_failed survives the collapse: it is the one "no results" case with
    a different action attached (go and look at cognify_status)."""
    assert classify_recall_state(
        {"ds": {"cognify_pipeline": {"status": "DATASET_PROCESSING_ERRORED"}}}
    ) == RecallState("build_failed")


def test_multiple_active_pipelines_do_not_double_count_progress():
    run = {
        "status": "DATASET_PROCESSING_STARTED",
        "progress": {"completed_items": 12, "total_items": 40},
    }
    assert classify_recall_state({"ds": {"add_pipeline": run, "cognify_pipeline": run}}) == (
        RecallState("indexing")
    )
