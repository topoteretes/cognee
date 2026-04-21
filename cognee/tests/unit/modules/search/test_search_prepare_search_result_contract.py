"""Contract tests for the normalized SearchResponse envelope.

Replaces the legacy ``_backwards_compatible_search_results`` coverage.
Each test drives ``search`` through its normal plumbing and asserts the
shape callers can rely on: a ``SearchResponse`` with a flat
``results`` list of ``SearchResultItem`` objects, each carrying a
renderable ``text``, a ``kind``, and the original payload in ``raw``.
"""

import types
from uuid import uuid4, uuid5, UUID, NAMESPACE_OID

import pytest
from pydantic import BaseModel

from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchResponse, SearchResultKind, SearchType


class DummyDataset(BaseModel):
    id: object
    name: str
    tenant_id: UUID | None = None
    owner_id: object


def _ds(name="ds1", tenant_id=uuid5(NAMESPACE_OID, "t1")):
    return DummyDataset(
        id=uuid5(NAMESPACE_OID, name), name=name, tenant_id=tenant_id, owner_id=uuid4()
    )


@pytest.fixture
def search_mod():
    import importlib

    return importlib.import_module("cognee.modules.search.methods.search")


@pytest.fixture(autouse=True)
def _patch_search_side_effects(monkeypatch, search_mod):
    async def dummy_log_query(_query_text, _query_type, _user_id):
        return types.SimpleNamespace(id="qid-1")

    async def dummy_log_result(*_args, **_kwargs):
        return None

    monkeypatch.setattr(search_mod, "send_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "log_query", dummy_log_query)
    monkeypatch.setattr(search_mod, "log_result", dummy_log_result)

    yield


@pytest.mark.asyncio
async def test_search_returns_search_response(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context="text",
                completion=["completion text"],
                search_type=SearchType.GRAPH_COMPLETION,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    response = await search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[ds.id],
        user=user,
    )

    assert isinstance(response, SearchResponse)
    assert response.query == "q"
    assert response.search_type == SearchType.GRAPH_COMPLETION.value
    assert response.total == 1
    assert len(response.results) == 1
    item = response.results[0]
    assert item.text == "completion text"
    assert item.kind == SearchResultKind.GRAPH_COMPLETION.value
    assert item.dataset_name == "ds1"
    assert item.dataset_id == str(uuid5(NAMESPACE_OID, "ds1"))


@pytest.mark.asyncio
async def test_search_flattens_completion_list(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object=None,
                context=None,
                completion=["first", "second", "third"],
                search_type=SearchType.GRAPH_COMPLETION,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    response = await search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[ds.id],
        user=user,
    )

    assert response.total == 3
    assert [r.text for r in response.results] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_search_chunks_preserves_dict_payload_in_raw(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")
    chunk = {"text": "chunk body", "id": "c1", "score": 0.42}

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object=None,
                context=None,
                completion=[chunk],
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    response = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
    )

    assert response.total == 1
    item = response.results[0]
    assert item.kind == SearchResultKind.CHUNK.value
    assert item.text == "chunk body"
    assert item.score == 0.42
    assert item.raw == chunk


@pytest.mark.asyncio
async def test_search_only_context_uses_context(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context=["ctx body"],
                completion=None,
                only_context=True,
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    response = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        only_context=True,
    )

    assert response.total == 1
    assert response.results[0].text == "ctx body"


@pytest.mark.asyncio
async def test_search_merges_results_across_datasets(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds1 = _ds("ds1")
    ds2 = _ds("ds2", tenant_id=uuid5(NAMESPACE_OID, "t2"))

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                completion=["a"],
                search_type=SearchType.GRAPH_COMPLETION,
                dataset_name=ds1.name,
                dataset_id=ds1.id,
                dataset_tenant_id=ds1.tenant_id,
            ),
            SearchResultPayload(
                completion=["b", "c"],
                search_type=SearchType.GRAPH_COMPLETION,
                dataset_name=ds2.name,
                dataset_id=ds2.id,
                dataset_tenant_id=ds2.tenant_id,
            ),
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    response = await search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[ds1.id, ds2.id],
        user=user,
    )

    assert response.total == 3
    assert [r.text for r in response.results] == ["a", "b", "c"]
    assert {r.dataset_name for r in response.results} == {"ds1", "ds2"}
