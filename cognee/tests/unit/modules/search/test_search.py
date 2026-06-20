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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_type",
    [SearchType.GRAPH_COMPLETION, SearchType.RAG_COMPLETION],
)
async def test_search_returns_results_for_all_datasets(monkeypatch, search_mod, query_type):
    """Regression guard for #2617.

    Multi-dataset GRAPH_COMPLETION / RAG_COMPLETION search must surface a
    result for EVERY authorized dataset, not just the first one. The bug
    silently dropped every dataset except ``results[0]`` and still returned
    200/success, so the assertion is specifically about not losing datasets.
    """
    user = _make_user()
    ds1 = _make_dataset(name="ds1", tenant_id="t1")
    ds2 = _make_dataset(name="ds2", tenant_id="t1")

    async def dummy_authorized_search(**_kwargs):
        # One payload per dataset, as produced by search_in_datasets_context.
        return [
            SearchResultPayload(
                result_object="object",
                context=[f"ctx-{ds.name}"],
                completion=[f"answer-{ds.name}"],
                search_type=query_type,
                dataset_name=ds.name,
                dataset_id=ds.id,
                dataset_tenant_id=ds.tenant_id,
            )
            for ds in (ds1, ds2)
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=query_type,
        dataset_ids=[ds1.id, ds2.id],
        user=user,
    )

    # Both datasets must be represented; nothing collapsed to results[0].
    assert len(out) == 2
    returned_dataset_ids = {row["dataset_id"] for row in out}
    assert returned_dataset_ids == {ds1.id, ds2.id}

    by_id = {row["dataset_id"]: row for row in out}
    assert by_id[ds1.id]["search_result"] == ["answer-ds1"]
    assert by_id[ds2.id]["search_result"] == ["answer-ds2"]


@pytest.mark.asyncio
async def test_search_no_access_control_keeps_all_dataset_results(monkeypatch, search_mod):
    """Regression guard for #2617 (access-control-disabled path).

    Without backend access control, multiple per-dataset payloads must not
    be collapsed: only a SINGLE result list is unwrapped for backwards
    compatibility, never a multi-result aggregation.
    """
    user = _make_user()
    ds1 = _make_dataset(name="ds1")
    ds2 = _make_dataset(name="ds2")

    async def dummy_authorized_search(**_kwargs):
        return [
            SearchResultPayload(
                result_object="object",
                context=["ctx1"],
                completion=["answer-ds1"],
                search_type=SearchType.RAG_COMPLETION,
                dataset_name=ds1.name,
                dataset_id=ds1.id,
                dataset_tenant_id=ds1.tenant_id,
            ),
            SearchResultPayload(
                result_object="object",
                context=["ctx2"],
                completion=["answer-ds2"],
                search_type=SearchType.RAG_COMPLETION,
                dataset_name=ds2.name,
                dataset_id=ds2.id,
                dataset_tenant_id=ds2.tenant_id,
            ),
        ]

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.RAG_COMPLETION,
        dataset_ids=[ds1.id, ds2.id],
        user=user,
    )

    # Two datasets -> two result lists preserved (no results[0]-only collapse).
    assert out == [["answer-ds1"], ["answer-ds2"]]


@pytest.mark.asyncio
async def test_search_in_datasets_context_scopes_each_dataset_when_access_control_disabled(
    monkeypatch, search_mod
):
    """Explicit datasets must be searched per-dataset even when ACCESS_CONTROL=false (#2867)."""
    user = _make_user()
    ds_one = _make_dataset(name="dataset_a")
    ds_two = _make_dataset(name="dataset_b")
    searched_dataset_ids = []

    async def fake_get_retriever_output(**kwargs):
        dataset = kwargs.get("dataset")
        searched_dataset_ids.append(dataset.id if dataset else None)
        return SearchResultPayload(
            result_object="object",
            context="ctx",
            completion="ok",
            search_type=SearchType.CHUNKS,
            dataset_name=dataset.name if dataset else None,
            dataset_id=dataset.id if dataset else None,
            dataset_tenant_id=dataset.tenant_id if dataset else None,
        )

    class DummyGraphEngine:
        async def is_empty(self):
            return False

    async def fake_get_graph_engine():
        return DummyGraphEngine()

    class DummyContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "get_retriever_output", fake_get_retriever_output)
    monkeypatch.setattr(search_mod, "get_graph_engine", fake_get_graph_engine)
    monkeypatch.setattr(
        search_mod,
        "set_database_global_context_variables",
        lambda *args, **kwargs: DummyContext(),
    )

    results = await search_mod.search_in_datasets_context(
        search_datasets=[ds_one, ds_two],
        query_type=SearchType.CHUNKS,
        query_text="q",
        user=user,
    )

    assert searched_dataset_ids == [ds_one.id, ds_two.id]
    assert len(results) == 2
    assert {result.dataset_id for result in results} == {ds_one.id, ds_two.id}


@pytest.mark.asyncio
async def test_search_passes_retriever_specific_config_to_authorized_search(
    monkeypatch, search_mod
):
    user = _make_user()
    ds = _make_dataset(name="ds1", tenant_id="t1")

    async def dummy_authorized_search(**_kwargs):
        assert _kwargs["feedback_influence"] == 0.25
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

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        feedback_influence=0.25,
    )

    assert out
