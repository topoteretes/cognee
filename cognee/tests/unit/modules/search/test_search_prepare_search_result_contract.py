## The Objective of these tests is to cover the search - prepare search results behavior (later to be removed)

import types
from uuid import uuid4, uuid5, UUID, NAMESPACE_OID

import pytest
from pydantic import BaseModel

from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


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


@pytest.mark.asyncio
async def test_search_backend_access_verbose(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context="text",
                completion="test",
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

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
    assert out[0]["dataset_tenant_id"] == uuid5(NAMESPACE_OID, "t1")
    assert out[0]["dataset_id"] == uuid5(NAMESPACE_OID, "ds1")
    assert out[0]["objects_result"] == "object"
    assert out[0]["context_result"] == "text"
    assert out[0]["text_result"] == "test"


@pytest.mark.asyncio
async def test_search_no_backend_access_verbose(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context="text",
                completion="test",
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out[0]["objects_result"] == "object"
    assert out[0]["context_result"] == "text"
    assert out[0]["text_result"] == "test"
    assert out[0].get("dataset_name") is None


@pytest.mark.asyncio
async def test_search_backend_access(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context="text",
                completion="test",
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
    )

    assert out[0]["dataset_name"] == "ds1"
    assert out[0]["dataset_id"] == uuid5(NAMESPACE_OID, "ds1")
    assert out[0]["dataset_tenant_id"] == uuid5(NAMESPACE_OID, "t1")
    assert (
        out[0]["search_result"] == "test"
    )  # Search result should be resolved to completion if it exists


@pytest.mark.asyncio
async def test_search_no_backend_access(monkeypatch, search_mod):
    user = types.SimpleNamespace(id="u1", tenant_id=None)
    ds = _ds("ds1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context="text",
                completion="test",
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
    )

    assert isinstance(out, list)
    assert out[0] == "test"  # Search result should be resolved to completion if it exists
