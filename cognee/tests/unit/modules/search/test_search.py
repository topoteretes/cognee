import types
from uuid import uuid4

import pytest

from cognee.modules.search.types import SearchType


def _make_user(user_id: str = "u1", tenant_id=None):
    return types.SimpleNamespace(id=user_id, tenant_id=tenant_id)


def _make_dataset(*, name="ds", tenant_id="t1", dataset_id=None, owner_id=None):
    return types.SimpleNamespace(
        id=dataset_id or uuid4(),
        name=name,
        tenant_id=tenant_id,
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

    async def dummy_prepare_search_result(search_result):
        if isinstance(search_result, tuple) and len(search_result) == 3:
            result, context, datasets = search_result
            return {"result": result, "context": context, "graphs": {}, "datasets": datasets}
        return {"result": None, "context": None, "graphs": {}, "datasets": []}

    monkeypatch.setattr(search_mod, "send_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "log_query", dummy_log_query)
    monkeypatch.setattr(search_mod, "log_result", dummy_log_result)
    monkeypatch.setattr(search_mod, "prepare_search_result", dummy_prepare_search_result)

    yield


@pytest.mark.asyncio
async def test_search_no_access_control_flattens_single_list_result(monkeypatch, search_mod):
    user = _make_user()

    async def dummy_no_access_control_search(**_kwargs):
        return (["r"], ["ctx"], [])

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "no_access_control_search", dummy_no_access_control_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=None,
        user=user,
    )

    assert out == ["r"]


@pytest.mark.asyncio
async def test_search_no_access_control_non_list_result_returns_list(monkeypatch, search_mod):
    """
    Covers the non-flattening back-compat branch in `search()`: if the single returned result is
    not a list, `search()` returns a list of results instead of flattening.
    """
    user = _make_user()

    async def dummy_no_access_control_search(**_kwargs):
        return ("r", ["ctx"], [])

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "no_access_control_search", dummy_no_access_control_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=None,
        user=user,
    )

    assert out == ["r"]


@pytest.mark.asyncio
async def test_search_no_access_control_only_context_returns_context(monkeypatch, search_mod):
    user = _make_user()

    async def dummy_no_access_control_search(**_kwargs):
        return (None, ["ctx"], [])

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "no_access_control_search", dummy_no_access_control_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=None,
        user=user,
        only_context=True,
    )

    assert out == ["ctx"]


@pytest.mark.asyncio
async def test_search_access_control_returns_dataset_shaped_dicts(monkeypatch, search_mod):
    user = _make_user()
    ds = _make_dataset(name="ds1", tenant_id="t1")

    async def dummy_authorized_search(**kwargs):
        assert kwargs["dataset_ids"] == [ds.id]
        return [("r", ["ctx"], [ds])]

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
            "dataset_tenant_id": "t1",
        }
    ]

    out_verbose = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds.id],
        user=user,
        verbose=True,
    )

    assert out_verbose == [
        {
            "search_result": ["r"],
            "dataset_id": ds.id,
            "dataset_name": "ds1",
            "dataset_tenant_id": "t1",
            "graphs": {},
        }
    ]


@pytest.mark.asyncio
async def test_search_access_control_only_context_returns_dataset_shaped_dicts(
    monkeypatch, search_mod
):
    user = _make_user()
    ds = _make_dataset(name="ds1", tenant_id="t1")

    async def dummy_authorized_search(**_kwargs):
        return [(None, "ctx", [ds])]

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

    assert out == [
        {
            "search_result": ["ctx"],
            "dataset_id": ds.id,
            "dataset_name": "ds1",
            "dataset_tenant_id": "t1",
            "graphs": {},
        }
    ]


@pytest.mark.asyncio
async def test_search_access_control_use_combined_context_returns_combined_model(
    monkeypatch, search_mod
):
    user = _make_user()
    ds1 = _make_dataset(name="ds1", tenant_id="t1")
    ds2 = _make_dataset(name="ds2", tenant_id="t1")

    async def dummy_authorized_search(**_kwargs):
        return ("answer", {"k": "v"}, [ds1, ds2])

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="q",
        query_type=SearchType.CHUNKS,
        dataset_ids=[ds1.id, ds2.id],
        user=user,
        use_combined_context=True,
    )

    assert out.result == "answer"
    assert out.context == {"k": "v"}
    assert out.graphs == {}
    assert [d.id for d in out.datasets] == [ds1.id, ds2.id]


@pytest.mark.asyncio
async def test_authorized_search_non_combined_delegates(monkeypatch, search_mod):
    user = _make_user()
    ds = _make_dataset(name="ds1")

    async def dummy_get_authorized_existing_datasets(*_args, **_kwargs):
        return [ds]

    expected = [("r", ["ctx"], [ds])]

    async def dummy_search_in_datasets_context(**kwargs):
        assert kwargs["use_combined_context"] is False if "use_combined_context" in kwargs else True
        return expected

    monkeypatch.setattr(
        search_mod, "get_authorized_existing_datasets", dummy_get_authorized_existing_datasets
    )
    monkeypatch.setattr(search_mod, "search_in_datasets_context", dummy_search_in_datasets_context)

    out = await search_mod.authorized_search(
        query_type=SearchType.CHUNKS,
        query_text="q",
        user=user,
        dataset_ids=[ds.id],
        use_combined_context=False,
        only_context=False,
    )

    assert out == expected


@pytest.mark.asyncio
async def test_authorized_search_use_combined_context_joins_string_context(monkeypatch, search_mod):
    user = _make_user()
    ds1 = _make_dataset(name="ds1")
    ds2 = _make_dataset(name="ds2")

    async def dummy_get_authorized_existing_datasets(*_args, **_kwargs):
        return [ds1, ds2]

    async def dummy_search_in_datasets_context(**kwargs):
        assert kwargs["only_context"] is True
        return [(None, ["a"], [ds1]), (None, ["b"], [ds2])]

    seen = {}

    async def dummy_get_completion(query_text, context, session_id=None):
        seen["query_text"] = query_text
        seen["context"] = context
        seen["session_id"] = session_id
        return ["answer"]

    async def dummy_get_search_type_tools(**_kwargs):
        return [dummy_get_completion, lambda *_a, **_k: None]

    monkeypatch.setattr(
        search_mod, "get_authorized_existing_datasets", dummy_get_authorized_existing_datasets
    )
    monkeypatch.setattr(search_mod, "search_in_datasets_context", dummy_search_in_datasets_context)
    monkeypatch.setattr(search_mod, "get_search_type_tools", dummy_get_search_type_tools)

    completion, combined_context, datasets = await search_mod.authorized_search(
        query_type=SearchType.CHUNKS,
        query_text="q",
        user=user,
        dataset_ids=[ds1.id, ds2.id],
        use_combined_context=True,
        session_id="s1",
    )

    assert combined_context == "a\nb"
    assert completion == ["answer"]
    assert datasets == [ds1, ds2]
    assert seen == {"query_text": "q", "context": "a\nb", "session_id": "s1"}


@pytest.mark.asyncio
async def test_authorized_search_use_combined_context_keeps_non_string_context(
    monkeypatch, search_mod
):
    user = _make_user()
    ds1 = _make_dataset(name="ds1")
    ds2 = _make_dataset(name="ds2")

    class DummyEdge:
        pass

    e1, e2 = DummyEdge(), DummyEdge()

    async def dummy_get_authorized_existing_datasets(*_args, **_kwargs):
        return [ds1, ds2]

    async def dummy_search_in_datasets_context(**_kwargs):
        return [(None, [e1], [ds1]), (None, [e2], [ds2])]

    async def dummy_get_completion(query_text, context, session_id=None):
        assert query_text == "q"
        assert context == [e1, e2]
        return ["answer"]

    async def dummy_get_search_type_tools(**_kwargs):
        return [dummy_get_completion]

    monkeypatch.setattr(
        search_mod, "get_authorized_existing_datasets", dummy_get_authorized_existing_datasets
    )
    monkeypatch.setattr(search_mod, "search_in_datasets_context", dummy_search_in_datasets_context)
    monkeypatch.setattr(search_mod, "get_search_type_tools", dummy_get_search_type_tools)

    completion, combined_context, datasets = await search_mod.authorized_search(
        query_type=SearchType.CHUNKS,
        query_text="q",
        user=user,
        dataset_ids=[ds1.id, ds2.id],
        use_combined_context=True,
    )

    assert combined_context == [e1, e2]
    assert completion == ["answer"]
    assert datasets == [ds1, ds2]


@pytest.mark.asyncio
async def test_search_in_datasets_context_two_tool_context_override_and_is_empty_branches(
    monkeypatch, search_mod
):
    ds1 = _make_dataset(name="ds1")
    ds2 = _make_dataset(name="ds2")

    async def dummy_set_database_global_context_variables(*_args, **_kwargs):
        return None

    class DummyGraphEngine:
        async def is_empty(self):
            return True

    async def dummy_get_graph_engine():
        return DummyGraphEngine()

    async def dummy_get_dataset_data(dataset_id):
        return [1] if dataset_id == ds1.id else []

    calls = {"completion": 0, "context": 0}

    async def dummy_get_context(_query_text: str):
        calls["context"] += 1
        return ["ctx"]

    async def dummy_get_completion(_query_text: str, _context, session_id=None):
        calls["completion"] += 1
        assert session_id == "s1"
        return ["r"]

    async def dummy_get_search_type_tools(**_kwargs):
        return [dummy_get_completion, dummy_get_context]

    monkeypatch.setattr(
        search_mod,
        "set_database_global_context_variables",
        dummy_set_database_global_context_variables,
    )
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(search_mod, "get_search_type_tools", dummy_get_search_type_tools)
    monkeypatch.setattr("cognee.modules.data.methods.get_dataset_data", dummy_get_dataset_data)

    out = await search_mod.search_in_datasets_context(
        search_datasets=[ds1, ds2],
        query_type=SearchType.CHUNKS,
        query_text="q",
        context=["pre_ctx"],
        session_id="s1",
    )

    assert out == [(["r"], ["pre_ctx"], [ds1]), (["r"], ["pre_ctx"], [ds2])]
    assert calls == {"completion": 2, "context": 0}


@pytest.mark.asyncio
async def test_search_in_datasets_context_two_tool_only_context_true(monkeypatch, search_mod):
    ds = _make_dataset(name="ds1")

    async def dummy_set_database_global_context_variables(*_args, **_kwargs):
        return None

    class DummyGraphEngine:
        async def is_empty(self):
            return False

    async def dummy_get_graph_engine():
        return DummyGraphEngine()

    async def dummy_get_context(query_text: str):
        assert query_text == "q"
        return ["ctx"]

    async def dummy_get_completion(*_args, **_kwargs):
        raise AssertionError("Completion should not be called when only_context=True")

    async def dummy_get_search_type_tools(**_kwargs):
        return [dummy_get_completion, dummy_get_context]

    monkeypatch.setattr(
        search_mod,
        "set_database_global_context_variables",
        dummy_set_database_global_context_variables,
    )
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(search_mod, "get_search_type_tools", dummy_get_search_type_tools)

    out = await search_mod.search_in_datasets_context(
        search_datasets=[ds],
        query_type=SearchType.CHUNKS,
        query_text="q",
        only_context=True,
    )

    assert out == [(None, ["ctx"], [ds])]


@pytest.mark.asyncio
async def test_search_in_datasets_context_unknown_tool_path(monkeypatch, search_mod):
    ds = _make_dataset(name="ds1")

    async def dummy_set_database_global_context_variables(*_args, **_kwargs):
        return None

    class DummyGraphEngine:
        async def is_empty(self):
            return False

    async def dummy_get_graph_engine():
        return DummyGraphEngine()

    async def dummy_unknown_tool(query_text: str):
        assert query_text == "q"
        return ["u"]

    async def dummy_get_search_type_tools(**_kwargs):
        return [dummy_unknown_tool]

    monkeypatch.setattr(
        search_mod,
        "set_database_global_context_variables",
        dummy_set_database_global_context_variables,
    )
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(search_mod, "get_search_type_tools", dummy_get_search_type_tools)

    out = await search_mod.search_in_datasets_context(
        search_datasets=[ds],
        query_type=SearchType.CODING_RULES,
        query_text="q",
    )

    assert out == [(["u"], "", [ds])]
