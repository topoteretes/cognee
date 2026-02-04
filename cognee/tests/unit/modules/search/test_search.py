import types
from uuid import uuid4, uuid5, NAMESPACE_OID

import pytest

from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def _make_user(user_id: str = "u1", tenant_id=None):
    return types.SimpleNamespace(id=user_id, tenant_id=tenant_id)


def _make_dataset(*, name="ds", tenant_id="t1", dataset_id=None, owner_id=None):
    return types.SimpleNamespace(
        id=uuid5(NAMESPACE_OID, name),
        name=name,
        tenant_id=uuid5(NAMESPACE_OID, tenant_id),
        owner_id=owner_id or uuid4(),
    )


@pytest.fixture
def search_mod():
    import importlib

    return importlib.import_module("cognee.modules.search.methods.search")


@pytest.fixture(autouse=True)
def _patch_side_effect_boundaries(monkeypatch, search_mod):
    """
    Keep production logic; patch only unavoidable side-effect boundaries.
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
async def test_search_access_control_returns_dataset_shaped_dicts(monkeypatch, search_mod):
    user = _make_user()
    ds = _make_dataset(name="ds1", tenant_id="t1")

    async def dummy_authorized_search(**_kwargs):
        assert _kwargs["dataset_ids"] == [ds.id]
        return [
            SearchResultPayload(
                result_object="object",
                context=["ctx"],
                completion=["r"],
                search_type=SearchType.CHUNKS,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out_non_verbose = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=False,
    )

    assert out_non_verbose == [
        {
            "search_result": ["r"],
            "dataset_id": ds.id,
            "dataset_name": "ds1",
            "dataset_tenant_id": uuid5(NAMESPACE_OID, "t1"),
        }
    ]


@pytest.mark.asyncio
async def test_search_access_control_only_context_returns_dataset_shaped_dicts(
    monkeypatch, search_mod
):
    user = _make_user()
    ds = _make_dataset(name="ds1", tenant_id="t1")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context=["ctx"],
                completion=None,
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
        only_context=True,
    )

    assert out == [
        {
            "search_result": ["ctx"],
            "dataset_id": ds.id,
            "dataset_name": "ds1",
            "dataset_tenant_id": uuid5(NAMESPACE_OID, "t1"),
        }
    ]


@pytest.mark.asyncio
async def test_authorized_search_delegates_to_search_in_datasets_context(monkeypatch, search_mod):
    user = _make_user()
    ds = _make_dataset(name="ds1")

    async def dummy_get_authorized_existing_datasets(*_args, **_kwargs):
        return [ds]

    expected = [
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

    async def dummy_search_in_datasets_context(**_kwargs):
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

    monkeypatch.setattr(
        search_mod, "get_authorized_existing_datasets", dummy_get_authorized_existing_datasets
    )
    monkeypatch.setattr(search_mod, "search_in_datasets_context", dummy_search_in_datasets_context)

    out = await search_mod.authorized_search(
        query_type=SearchType.CHUNKS,
        query_text="q",
        user=user,
        dataset_ids=[ds.id],
        only_context=False,
    )

    assert out == expected
