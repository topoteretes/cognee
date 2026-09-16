"""SDK-270: a graph-only recall whose every dataset skipped its LLM completion
reports a typed system marker instead of a bare [] — and never a 404."""

import importlib
import types
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from cognee.modules.recall.config import RecallConfig
from cognee.modules.recall.types.RecallResponse import (
    RecallResponse,
    ResponseGraphEntry,
    ResponseMarkerEntry,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchStatus, SearchType


def _make_user():
    return types.SimpleNamespace(id=uuid4(), tenant_id=None)


def _payload(status, completion=None, name="ds"):
    return SearchResultPayload(
        result_object=[],
        context="",
        completion=completion if completion is not None else [],
        search_type=SearchType.GRAPH_COMPLETION,
        status=status,
        dataset_name=name,
        dataset_id=uuid4(),
    )


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


@pytest.fixture
def warm_graph(monkeypatch):
    """Graph warm-up guard passes, so the graph lane really runs."""
    graph_warmup_mod = importlib.import_module("cognee.modules.recall.methods.graph_warmup")
    recall_config_mod = importlib.import_module("cognee.modules.recall.config")

    async def warm(user, dataset_ids):
        return graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_WARM, 5)

    monkeypatch.setattr(graph_warmup_mod, "get_graph_build_status", warm)
    monkeypatch.setattr(
        recall_config_mod, "get_recall_config", lambda: RecallConfig(_env_file=None)
    )


@pytest.fixture
def graph_lane(monkeypatch, api_recall_mod, warm_graph):
    """Stub the graph lane; set ``payloads[:]`` to control what search returns."""
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)

    permission_methods = importlib.import_module("cognee.modules.users.permissions.methods")

    async def fake_permission_datasets(user_id, permission_type, dataset_ids):
        return [types.SimpleNamespace(id=dataset_id) for dataset_id in dataset_ids]

    monkeypatch.setattr(
        permission_methods, "get_specific_user_permission_datasets", fake_permission_datasets
    )

    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")
    payloads: list = []

    async def fake_authorized_search(**kwargs):
        return list(payloads)

    async def dummy_log_search_history(*args, **kwargs):
        return None

    async def dummy_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(search_methods, "authorized_search", fake_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", dummy_log_search_history)
    monkeypatch.setattr(
        api_recall_mod, "set_session_user_context_variable", dummy_set_session_user_context_variable
    )
    return payloads


async def _recall(api_recall_mod, **overrides):
    kwargs = {
        "query_text": "q",
        "query_type": SearchType.GRAPH_COMPLETION,
        "dataset_ids": [uuid4()],
        "auto_route": False,
        "user": _make_user(),
    }
    kwargs.update(overrides)
    return await api_recall_mod.recall(**kwargs)


@pytest.mark.asyncio
async def test_every_dataset_graph_empty_yields_graph_empty_marker(api_recall_mod, graph_lane):
    graph_lane[:] = [_payload(SearchStatus.GRAPH_EMPTY, name="a")]

    out = await _recall(api_recall_mod)

    assert len(out) == 1
    marker = out[0]
    assert isinstance(marker, ResponseMarkerEntry)
    assert marker.source == "system"
    assert marker.status == "graph_empty"
    assert "cognif" in marker.text
    assert marker.datapoint_count is None


@pytest.mark.asyncio
async def test_retrieval_miss_yields_no_context_marker(api_recall_mod, graph_lane):
    graph_lane[:] = [_payload(SearchStatus.NO_CONTEXT)]

    out = await _recall(api_recall_mod)

    assert [entry.status for entry in out] == ["no_context"]
    assert isinstance(out[0], ResponseMarkerEntry)


@pytest.mark.asyncio
async def test_mixed_empty_and_populated_datasets_report_no_context(api_recall_mod, graph_lane):
    """One empty dataset beside a populated one that missed: the graph is not
    empty as a whole, so the marker is the weaker 'no_context'."""
    graph_lane[:] = [
        _payload(SearchStatus.GRAPH_EMPTY, name="fresh"),
        _payload(SearchStatus.NO_CONTEXT, name="populated"),
    ]

    out = await _recall(api_recall_mod)

    assert [entry.status for entry in out] == ["no_context"]


@pytest.mark.asyncio
async def test_an_answer_from_any_dataset_wins_over_sibling_markers(api_recall_mod, graph_lane):
    """The multi-dataset composition the 404 broke: an empty dataset must not
    hide a sibling's real answer, and it adds no marker beside it."""
    graph_lane[:] = [
        _payload(SearchStatus.GRAPH_EMPTY, name="fresh"),
        _payload(SearchStatus.OK, completion=["Jane proposed SQLite."], name="populated"),
    ]

    out = await _recall(api_recall_mod)

    assert len(out) == 1
    assert isinstance(out[0], ResponseGraphEntry)
    assert out[0].text == "Jane proposed SQLite."
    assert out[0].dataset_name == "populated"


@pytest.mark.asyncio
async def test_plain_empty_result_without_skip_status_stays_empty(api_recall_mod, graph_lane):
    """A status-less empty payload (non-generative types, or no payloads at all)
    is still a plain []: the marker is only for skipped completions."""
    graph_lane[:] = [_payload(SearchStatus.OK)]
    assert await _recall(api_recall_mod) == []

    graph_lane[:] = []
    assert await _recall(api_recall_mod) == []


@pytest.mark.asyncio
async def test_multi_source_recall_contributes_nothing_instead_of_a_marker(
    api_recall_mod, graph_lane
):
    """Mirrors the warm-up guard: with other sources in play the graph lane
    stays silent so the tools 'on_empty' fallback and session results are not
    crowded out by a system marker."""
    graph_lane[:] = [_payload(SearchStatus.GRAPH_EMPTY)]

    out = await _recall(api_recall_mod, scope=["session", "graph"])

    assert out == []


def test_marker_without_probe_fields_round_trips_through_union():
    adapter = TypeAdapter(RecallResponse)
    marker = adapter.validate_python(
        {"source": "system", "status": "no_context", "text": "No relevant memory found."}
    )
    assert isinstance(marker, ResponseMarkerEntry)
    assert marker.datapoint_count is None
    assert marker.threshold is None
