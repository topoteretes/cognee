## The Objective of these tests is to cover the search - prepare search results behavior (later to be removed)

import types
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.search.types import SearchType


class DummyDataset(BaseModel):
    id: object
    name: str
    tenant_id: str | None = None
    owner_id: object


def _ds(name="ds1", tenant_id="t1"):
    return DummyDataset(id=uuid4(), name=name, tenant_id=tenant_id, owner_id=uuid4())


def _edge(rel="rel", n1="A", n2="B"):
    node1 = Node(str(uuid4()), attributes={"type": "Entity", "name": n1})
    node2 = Node(str(uuid4()), attributes={"type": "Entity", "name": n2})
    return Edge(node1, node2, attributes={"relationship_name": rel})


@pytest.fixture
def search_mod():
    import importlib

    return importlib.import_module("cognee.modules.search.methods.search")


@pytest.fixture(autouse=True)
def _patch_search_side_effects(monkeypatch, search_mod):
    """
    These tests validate prepare_search_result behavior *through* search.py.
    We only patch unavoidable side effects (telemetry + query/result logging).
    """

    async def dummy_log_query(_query_text, _query_type, _user_id):
        return types.SimpleNamespace(id="qid-1")

    async def dummy_log_result(*_args, **_kwargs):
        return None

    monkeypatch.setattr(search_mod, "send_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "log_query", dummy_log_query)
    monkeypatch.setattr(search_mod, "log_result", dummy_log_result)

    yield


@pytest.fixture(autouse=True)
def _patch_resolve_edges_to_text(monkeypatch):
    """
    Keep graph-text conversion deterministic and lightweight.
    """
    import importlib

    psr_mod = importlib.import_module("cognee.modules.search.utils.prepare_search_result")

    async def dummy_resolve_edges_to_text(_edges):
        return "EDGE_TEXT"

    monkeypatch.setattr(psr_mod, "resolve_edges_to_text", dummy_resolve_edges_to_text)

    yield


@pytest.mark.asyncio
async def test_search_access_control_edges_context_produces_graphs_and_context_map(
    monkeypatch, search_mod
):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")
    context = [_edge("likes")]

    async def dummy_authorized_search(**_kwargs):
        return [(["answer"], context, [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out[0]["dataset_name"] == "ds1"
    assert out[0]["dataset_tenant_id"] == "t1"
    assert out[0]["graphs"] is not None
    assert "ds1" in out[0]["graphs"]
    assert out[0]["graphs"]["ds1"]["nodes"]
    assert out[0]["graphs"]["ds1"]["edges"]
    assert out[0]["search_result"] == ["answer"]


@pytest.mark.asyncio
async def test_search_access_control_insights_context_produces_graphs_and_null_result(
    monkeypatch, search_mod
):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")
    insights = [
        (
            {"id": "n1", "type": "Entity", "name": "Alice"},
            {"relationship_name": "knows"},
            {"id": "n2", "type": "Entity", "name": "Bob"},
        )
    ]

    async def dummy_authorized_search(**_kwargs):
        return [(["something"], insights, [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out[0]["graphs"] is not None
    assert "ds1" in out[0]["graphs"]
    assert out[0]["search_result"] is None


@pytest.mark.asyncio
async def test_search_access_control_only_context_returns_context_text_map(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")

    async def dummy_authorized_search(**_kwargs):
        return [(None, ["a", "b"], [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        only_context=True,
        verbose=True,
    )

    assert out[0]["search_result"] == [{"ds1": "a\nb"}]


@pytest.mark.asyncio
async def test_search_access_control_results_edges_become_graph_result(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")
    results = [_edge("connected_to")]

    async def dummy_authorized_search(**_kwargs):
        return [(results, "ctx", [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert isinstance(out[0]["search_result"][0], dict)
    assert "nodes" in out[0]["search_result"][0]
    assert "edges" in out[0]["search_result"][0]


@pytest.mark.asyncio
async def test_search_use_combined_context_defaults_empty_datasets(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)

    async def dummy_authorized_search(**_kwargs):
        return ("answer", "ctx", [])

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=None,
        user=user,
        use_combined_context=True,
        verbose=True,
    )

    assert out.result == "answer"
    assert out.context == {"all available datasets": "ctx"}
    assert out.datasets[0].name == "all available datasets"


@pytest.mark.asyncio
async def test_search_access_control_context_str_branch(monkeypatch, search_mod):
    """Covers prepare_search_result(context is str) through search()."""
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")

    async def dummy_authorized_search(**_kwargs):
        return [(["answer"], "plain context", [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out[0]["graphs"] is None
    assert out[0]["search_result"] == ["answer"]


@pytest.mark.asyncio
async def test_search_access_control_context_empty_list_branch(monkeypatch, search_mod):
    """Covers prepare_search_result(context is empty list) through search()."""
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")

    async def dummy_authorized_search(**_kwargs):
        return [(["answer"], [], [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out[0]["graphs"] is None
    assert out[0]["search_result"] == ["answer"]


@pytest.mark.asyncio
async def test_search_access_control_multiple_results_list_branch(monkeypatch, search_mod):
    """Covers prepare_search_result(result list length > 1) through search()."""
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1", "t1")

    async def dummy_authorized_search(**_kwargs):
        return [(["r1", "r2"], "ctx", [ds])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out[0]["search_result"] == [["r1", "r2"]]


@pytest.mark.asyncio
async def test_search_access_control_defaults_empty_datasets(monkeypatch, search_mod):
    """
    Covers prepare_search_result(datasets empty list) through search().

    Note: in access-control mode, search.py expects datasets[0] to have `tenant_id`,
    but prepare_search_result defaults to SearchResultDataset which doesn't define it.
    We assert the current behavior (it raises) so refactors don't silently change it.
    """
    user = types.SimpleNamespace(id="u1", tenant_id=None)

    async def dummy_authorized_search(**_kwargs):
        return [(["answer"], "ctx", [])]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    with pytest.raises(AttributeError, match="tenant_id"):
        await search_mod.search(
            query_text="q",
            query_type=SearchType.CHUNKS,
            dataset_ids=None,
            user=user,
            verbose=True,
        )
