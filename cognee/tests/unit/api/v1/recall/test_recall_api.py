import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.search.types import SearchType


def _make_user(user_id=None, tenant_id=None):
    return types.SimpleNamespace(id=user_id or uuid4(), tenant_id=tenant_id)


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


@pytest.mark.asyncio
async def test_recall_dataset_ids_takes_precedence_over_datasets(monkeypatch, api_recall_mod):
    user = _make_user()
    explicit_id = uuid4()

    captured = {}

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        captured["dataset_ids"] = kwargs.get("dataset_ids")
        return []

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        captured["resolved_from_datasets"] = True
        return []

    def dummy_get_remote_client():
        return None

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")

    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["some_dataset_name"],
        dataset_ids=[explicit_id],
        auto_route=False,
        user=user,
    )

    assert out == []
    assert captured["dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured


@pytest.mark.asyncio
async def test_recall_remote_client_forwards_dataset_ids(monkeypatch, api_recall_mod):
    user = _make_user()
    explicit_id = uuid4()

    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["dataset_ids"] = kwargs.get("dataset_ids")
        return []

    dummy_remote_client = types.SimpleNamespace(recall=dummy_remote_recall)

    def dummy_get_remote_client():
        return dummy_remote_client

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        captured["resolved_from_datasets"] = True
        return []

    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["some_dataset_name"],
        dataset_ids=[explicit_id],
        auto_route=False,
        user=user,
    )

    assert out == []
    assert captured["dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured


@pytest.mark.asyncio
async def test_recall_session_with_dataset_ids_does_not_short_circuit_graph(
    monkeypatch, api_recall_mod
):
    user = _make_user()
    explicit_id = uuid4()

    captured = {}

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        captured["dataset_ids"] = kwargs.get("dataset_ids")
        return []

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        captured["resolved_from_datasets"] = True
        return []

    def dummy_get_remote_client():
        return None

    async def dummy_search_session(**kwargs):
        return []

    async def dummy_search_trace(**kwargs):
        return []

    async def dummy_fetch_graph_context(**kwargs):
        return []

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )
    monkeypatch.setattr(api_recall_mod, "_search_session", dummy_search_session)
    monkeypatch.setattr(api_recall_mod, "_search_trace", dummy_search_trace)
    monkeypatch.setattr(api_recall_mod, "_fetch_graph_context", dummy_fetch_graph_context)

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")

    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=None,
        dataset_ids=[explicit_id],
        session_id="session-1",
        user=user,
    )

    assert out == []
    assert captured["dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured
