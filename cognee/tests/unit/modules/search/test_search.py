import types
from uuid import NAMESPACE_OID, uuid4, uuid5

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

    async def dummy_log_query(*_args, **_kwargs):
        return types.SimpleNamespace(id="qid-1")

    async def dummy_log_result(*_args, **_kwargs):
        return None

    monkeypatch.setattr(search_mod, "send_telemetry", lambda *a, **k: None)
    import importlib

    history_mod = importlib.import_module("cognee.modules.search.operations.log_search_history")
    monkeypatch.setattr(history_mod, "log_query", dummy_log_query)
    monkeypatch.setattr(history_mod, "log_result", dummy_log_result)

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


@pytest.mark.asyncio
async def test_only_context_fan_out_shares_one_history_reader(monkeypatch, search_mod):
    """Ten datasets must not mean ten embedding calls: one SharedSessionHistory per search."""
    from contextlib import asynccontextmanager

    from cognee.modules.retrieval.only_context_prompt import SharedSessionHistory

    user = _make_user()
    datasets = [_make_dataset(name="ds1"), _make_dataset(name="ds2"), _make_dataset(name="ds3")]
    captured = []

    @asynccontextmanager
    async def dummy_context(*_args, **_kwargs):
        yield

    class _Engine:
        async def is_empty(self):
            return False

    async def dummy_get_graph_engine():
        return _Engine()

    async def dummy_get_retriever_output(query_type, query_text, **kwargs):
        captured.append(kwargs)
        dataset = kwargs["dataset"]
        return SearchResultPayload(
            context="ctx",
            only_context=True,
            search_type=query_type,
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            dataset_tenant_id=dataset.tenant_id,
        )

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "set_database_global_context_variables", dummy_context)
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(search_mod, "get_retriever_output", dummy_get_retriever_output)

    await search_mod.search_in_datasets_context(
        search_datasets=datasets,
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="q",
        user=user,
        only_context=True,
        session_id="s1",
    )

    assert len(captured) == 3
    shared = captured[0]["shared_history"]
    assert isinstance(shared, SharedSessionHistory)
    assert all(call["shared_history"] is shared for call in captured)
    assert (shared.query, shared.session_id) == ("q", "s1")


@pytest.mark.asyncio
async def test_fan_out_without_only_context_creates_no_history_reader(monkeypatch, search_mod):
    from contextlib import asynccontextmanager

    captured = []

    @asynccontextmanager
    async def dummy_context(*_args, **_kwargs):
        yield

    class _Engine:
        async def is_empty(self):
            return False

    async def dummy_get_graph_engine():
        return _Engine()

    async def dummy_get_retriever_output(query_type, query_text, **kwargs):
        captured.append(kwargs)
        return SearchResultPayload(completion=["answer"], search_type=query_type)

    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "set_database_global_context_variables", dummy_context)
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(search_mod, "get_retriever_output", dummy_get_retriever_output)

    await search_mod.search_in_datasets_context(
        search_datasets=[_make_dataset(name="ds1")],
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="q",
        user=_make_user(),
    )

    assert captured[0]["shared_history"] is None


def _only_context_payloads():
    ds = _make_dataset(name="ds1")
    built = SearchResultPayload(
        context="ctx",
        only_context=True,
        user_prompt="The question is: `q` ... ctx",
        system_prompt="## Active session guidance\n- x\nTASK:answer",
        search_type=SearchType.GRAPH_COMPLETION,
        dataset_name=ds.name,
        dataset_id=ds.id,
        dataset_tenant_id=ds.tenant_id,
    )
    bare = SearchResultPayload(
        context=["chunk"],
        only_context=True,
        search_type=SearchType.CHUNKS,
        dataset_name=ds.name,
        dataset_id=ds.id,
        dataset_tenant_id=ds.tenant_id,
    )
    return built, bare


def test_only_context_search_result_is_the_user_prompt_when_built(monkeypatch, search_mod):
    """Completion types hand back the user prompt; retrieval-only types their context."""
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    built, bare = _only_context_payloads()

    out = search_mod._backwards_compatible_search_results([built, bare], verbose=False)

    assert out[0]["search_result"] == built.user_prompt
    assert isinstance(out[0]["search_result"], str)
    assert out[1]["search_result"] == ["chunk"]


@pytest.mark.parametrize("access_control", [True, False])
def test_verbose_results_carry_both_prompts_next_to_the_bare_context(
    monkeypatch, search_mod, access_control
):
    """`user_prompt_result` and `system_prompt_result` are always present in verbose
    output: the prompts when they were built, None otherwise — so a verbose caller can take
    any layer."""
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: access_control)
    built, bare = _only_context_payloads()

    out = search_mod._backwards_compatible_search_results([built, bare], verbose=True)

    assert out[0]["user_prompt_result"] == built.user_prompt
    assert out[0]["system_prompt_result"] == built.system_prompt
    assert out[0]["context_result"] == "ctx"
    assert out[1]["user_prompt_result"] is None
    assert out[1]["system_prompt_result"] is None
    assert out[1]["context_result"] == ["chunk"]


# --- SDK-270 / empty datasets: a dataset without searchable memory is an empty entry
# with its reason on ``error``; the search never raises NoDataError for it ---


def _fan_out_harness(monkeypatch, search_mod, *, datasets, graph_empty, data_items, outcomes):
    """Run the real per-dataset fan-out; stub only the boundaries around it.

    ``graph_empty`` / ``data_items`` / ``outcomes`` are keyed by dataset name. An
    outcome that is an exception is raised by get_retriever_output for that dataset;
    anything else is returned as its payload.
    """
    import importlib
    from contextlib import asynccontextmanager

    by_id = {dataset.id: dataset for dataset in datasets}
    current: dict = {}

    @asynccontextmanager
    async def dummy_context(dataset_id, *_args, **_kwargs):
        current["name"] = by_id[dataset_id].name
        yield

    class _Engine:
        async def is_empty(self):
            return graph_empty[current["name"]]

    async def dummy_get_graph_engine():
        return _Engine()

    async def dummy_get_dataset_data(dataset_id):
        return data_items[by_id[dataset_id].name]

    async def dummy_get_retriever_output(query_type, query_text, **kwargs):
        outcome = outcomes[kwargs["dataset"].name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    data_methods = importlib.import_module("cognee.modules.data.methods")
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(search_mod, "set_database_global_context_variables", dummy_context)
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(data_methods, "get_dataset_data", dummy_get_dataset_data)
    monkeypatch.setattr(search_mod, "get_retriever_output", dummy_get_retriever_output)


def _answer(dataset, text):
    return SearchResultPayload(
        result_object=["edge"],
        context="node1 -- rel -- node2",
        completion=[text],
        search_type=SearchType.GRAPH_COMPLETION,
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        dataset_tenant_id=dataset.tenant_id,
    )


def _empty_graph_error():
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError

    return NoDataError("The knowledge graph is empty. Ingest data through Cognee before searching.")


async def _run_fan_out(search_mod, datasets):
    return await search_mod.search_in_datasets_context(
        search_datasets=datasets,
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What did Jane propose?",
        user=_make_user(),
    )


@pytest.mark.asyncio
async def test_single_uncognified_dataset_returns_empty_entry_that_says_to_cognify(
    monkeypatch, search_mod, caplog
):
    """A dataset with data but no graph yet is the normal state before the first
    cognify: the search answers with no results, not a 404. The reason survives
    on ``error`` so a caller that wants to tell "nothing matched" from "nothing
    cognified" still can."""
    ds = _make_dataset(name="fresh_notes")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[ds],
        graph_empty={"fresh_notes": True},
        data_items={"fresh_notes": ["doc-1", "doc-2", "doc-3"]},
        outcomes={"fresh_notes": _empty_graph_error()},
    )

    with caplog.at_level("WARNING"):
        results = await _run_fan_out(search_mod, [ds])

    (empty,) = results
    assert empty.result == []
    assert empty.completion is None
    assert empty.dataset_id == ds.id
    assert empty.dataset_name == "fresh_notes"
    assert empty.search_type == SearchType.GRAPH_COMPLETION
    assert "3 data item(s)" in empty.error
    assert "cognify" in empty.error
    assert any("fresh_notes" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_all_datasets_empty_returns_one_empty_entry_per_dataset_with_its_reason(
    monkeypatch, search_mod
):
    never_cognified = _make_dataset(name="never_cognified")
    brand_new = _make_dataset(name="brand_new")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[never_cognified, brand_new],
        graph_empty={"never_cognified": True, "brand_new": True},
        data_items={"never_cognified": ["doc-1", "doc-2"], "brand_new": []},
        outcomes={"never_cognified": _empty_graph_error(), "brand_new": _empty_graph_error()},
    )

    results = await _run_fan_out(search_mod, [never_cognified, brand_new])

    # One entry per dataset, in request order, each empty with its own reason.
    assert [payload.dataset_name for payload in results] == ["never_cognified", "brand_new"]
    assert all(payload.result == [] for payload in results)
    uncognified, fresh = results
    assert uncognified.error.startswith("holds 2 data item(s)")
    assert fresh.error == "no data has been added; add data and run cognify before searching."


@pytest.mark.asyncio
async def test_all_datasets_empty_through_search_is_an_empty_search_result(monkeypatch, search_mod):
    """What the SDK caller sees: ``search()`` over nothing but empty datasets is a
    list of entries whose ``search_result`` is ``[]`` -- the same shape as a miss --
    with the reason on ``error``; it does not raise."""
    ds = _make_dataset(name="fresh_notes")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[ds],
        graph_empty={"fresh_notes": True},
        data_items={"fresh_notes": []},
        outcomes={"fresh_notes": _empty_graph_error()},
    )

    async def dummy_authorized_search(**kwargs):
        return await search_mod.search_in_datasets_context(
            search_datasets=[ds],
            query_type=kwargs["query_type"],
            query_text=kwargs["query_text"],
            user=kwargs["user"],
        )

    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)

    out = await search_mod.search(
        query_text="What did Jane propose?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[ds.id],
        user=_make_user(),
    )

    assert out == [
        {
            "dataset_id": ds.id,
            "dataset_name": "fresh_notes",
            "dataset_tenant_id": ds.tenant_id,
            "search_result": [],
            "error": "no data has been added; add data and run cognify before searching.",
        }
    ]


def _single_tenant_harness(
    monkeypatch, search_mod, *, graph_empty, data_items, outcome, expect_dataset
):
    """Access control off: one shared graph, no per-dataset context. Stub the same
    boundaries as ``_fan_out_harness``; ``outcome`` is raised or returned by
    get_retriever_output, which must be handed ``expect_dataset`` (the single
    dataset, or None for the shared graph)."""
    import importlib
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def dummy_context(*_args, **_kwargs):
        yield

    class _Engine:
        async def is_empty(self):
            return graph_empty

    async def dummy_get_graph_engine():
        return _Engine()

    async def dummy_get_dataset_data(dataset_id):
        assert expect_dataset is not None and dataset_id == expect_dataset.id
        return data_items

    async def dummy_get_retriever_output(query_type, query_text, **kwargs):
        assert kwargs["dataset"] is expect_dataset
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    data_methods = importlib.import_module("cognee.modules.data.methods")
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(search_mod, "set_database_global_context_variables", dummy_context)
    monkeypatch.setattr(search_mod, "get_graph_engine", dummy_get_graph_engine)
    monkeypatch.setattr(data_methods, "get_dataset_data", dummy_get_dataset_data)
    monkeypatch.setattr(search_mod, "get_retriever_output", dummy_get_retriever_output)


@pytest.mark.asyncio
async def test_shared_graph_without_data_returns_empty_entry(monkeypatch, search_mod):
    """With access control off there is one shared graph and no dataset to name;
    an empty graph is still a result -- empty, with the reason -- not a raised
    NoDataError. There is no dataset whose data items could be counted, so the
    reason is the generic one."""
    ds1 = _make_dataset(name="ds1")
    ds2 = _make_dataset(name="ds2")
    _single_tenant_harness(
        monkeypatch,
        search_mod,
        graph_empty=True,
        data_items=[],
        outcome=_empty_graph_error(),
        expect_dataset=None,
    )

    results = await search_mod.search_in_datasets_context(
        search_datasets=[ds1, ds2],
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What did Jane propose?",
        user=_make_user(),
    )

    (empty,) = results
    assert empty.result == []
    assert empty.dataset_id is None
    assert empty.dataset_name is None
    assert empty.error == "the knowledge graph is empty; add data and run cognify before searching."


@pytest.mark.asyncio
async def test_shared_graph_populated_but_missing_collection_still_raises(monkeypatch, search_mod):
    """Access control off, populated shared graph, retriever lacks its collection:
    not an empty store, so the retriever's NoDataError propagates unchanged."""
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError

    _single_tenant_harness(
        monkeypatch,
        search_mod,
        graph_empty=False,
        data_items=["doc-1"],
        outcome=NoDataError("No data found in the system, please add data first."),
        expect_dataset=None,
    )

    with pytest.raises(NoDataError, match="No data found in the system") as raised:
        await search_mod.search_in_datasets_context(
            search_datasets=[_make_dataset(name="ds1"), _make_dataset(name="ds2")],
            query_type=SearchType.RAG_COMPLETION,
            query_text="What did Jane propose?",
            user=_make_user(),
        )
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_single_dataset_without_access_control_gets_the_same_reason(monkeypatch, search_mod):
    """Access control off and exactly one dataset: the data items are counted the
    way the access-control path counts them, so the same state yields the same
    'run cognify' reason whichever way the deployment flag is set."""
    ds = _make_dataset(name="notes")
    _single_tenant_harness(
        monkeypatch,
        search_mod,
        graph_empty=True,
        data_items=["doc-1", "doc-2", "doc-3"],
        outcome=_empty_graph_error(),
        expect_dataset=ds,
    )

    (empty,) = await search_mod.search_in_datasets_context(
        search_datasets=[ds],
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What did Jane propose?",
        user=_make_user(),
    )

    assert empty.result == []
    assert empty.dataset_id == ds.id
    assert empty.dataset_name == "notes"
    assert empty.error == (
        "holds 3 data item(s) but its knowledge graph is empty; run cognify on this dataset "
        "before searching."
    )


@pytest.mark.asyncio
async def test_single_dataset_without_access_control_populated_graph_still_raises(
    monkeypatch, search_mod
):
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError

    ds = _make_dataset(name="notes")
    _single_tenant_harness(
        monkeypatch,
        search_mod,
        graph_empty=False,
        data_items=["doc-1"],
        outcome=NoDataError("No data found in the system, please add data first."),
        expect_dataset=ds,
    )

    with pytest.raises(NoDataError, match=r"Dataset 'notes' .*No data found in the system"):
        await search_mod.search_in_datasets_context(
            search_datasets=[ds],
            query_type=SearchType.RAG_COMPLETION,
            query_text="What did Jane propose?",
            user=_make_user(),
        )


@pytest.mark.asyncio
async def test_single_tenant_verbose_search_carries_the_reason(monkeypatch, search_mod):
    """What a single-tenant ``search()`` caller sees for an empty graph: the
    non-verbose shape stays the bare ``[]`` it always was, and ``verbose=True``
    is where the reason is read, as with access control on."""
    ds = _make_dataset(name="notes")
    _single_tenant_harness(
        monkeypatch,
        search_mod,
        graph_empty=True,
        data_items=[],
        outcome=_empty_graph_error(),
        expect_dataset=ds,
    )

    async def dummy_authorized_search(**kwargs):
        return await search_mod.search_in_datasets_context(
            search_datasets=[ds],
            query_type=kwargs["query_type"],
            query_text=kwargs["query_text"],
            user=kwargs["user"],
        )

    monkeypatch.setattr(search_mod, "authorized_search", dummy_authorized_search)
    call = {
        "query_text": "What did Jane propose?",
        "query_type": SearchType.GRAPH_COMPLETION,
        "dataset_ids": [ds.id],
        "user": _make_user(),
    }

    assert await search_mod.search(**call) == []

    (entry,) = await search_mod.search(**call, verbose=True)
    assert entry["objects_result"] == []
    assert entry["text_result"] is None
    assert entry["error"] == "no data has been added; add data and run cognify before searching."


@pytest.mark.asyncio
async def test_populated_sibling_keeps_the_empty_dataset_in_the_list(
    monkeypatch, search_mod, caplog
):
    """datasets=None means every readable dataset, so one fresh dataset must not
    take down unscoped search: the populated dataset answers, and the empty one
    stays in the list -- empty, with its reason on ``error`` -- so the caller is
    told instead of the dataset silently going missing."""
    fresh = _make_dataset(name="fresh")
    populated = _make_dataset(name="populated")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[fresh, populated],
        graph_empty={"fresh": True, "populated": False},
        data_items={"fresh": [], "populated": ["doc-1"]},
        outcomes={
            "fresh": _empty_graph_error(),
            "populated": _answer(populated, "Jane proposed SQLite."),
        },
    )

    with caplog.at_level("WARNING"):
        results = await _run_fan_out(search_mod, [fresh, populated])

    # One entry per dataset, in request order.
    assert [payload.dataset_name for payload in results] == ["fresh", "populated"]
    empty, answered = results

    assert answered.completion == ["Jane proposed SQLite."]
    assert answered.error is None

    assert empty.error == "no data has been added; add data and run cognify before searching."
    assert empty.dataset_id == fresh.id
    assert empty.dataset_tenant_id == fresh.tenant_id
    assert empty.search_type == SearchType.GRAPH_COMPLETION
    # Shaped exactly like a query miss: `[]` is what callers always saw here.
    assert empty.result == []
    assert empty.completion is None
    assert any("fresh" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_answer_miss_and_empty_dataset_side_by_side(monkeypatch, search_mod):
    """The three outcomes a dataset can have, in one search: an answer, a miss
    on a populated graph (empty result, no error), and no searchable memory."""
    finance = _make_dataset(name="finance")
    hr = _make_dataset(name="hr")
    scratch = _make_dataset(name="scratch")
    miss = SearchResultPayload(
        result_object=[],
        context="",
        completion=[],
        search_type=SearchType.GRAPH_COMPLETION,
        dataset_name=hr.name,
        dataset_id=hr.id,
        dataset_tenant_id=hr.tenant_id,
    )
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[finance, hr, scratch],
        graph_empty={"finance": False, "hr": False, "scratch": True},
        data_items={"finance": ["doc-1"], "hr": ["doc-1"], "scratch": ["a", "b", "c"]},
        outcomes={
            "finance": _answer(finance, "Q2 revenue was 4.2M."),
            "hr": miss,
            "scratch": _empty_graph_error(),
        },
    )

    results = await _run_fan_out(search_mod, [finance, hr, scratch])

    assert [payload.dataset_name for payload in results] == ["finance", "hr", "scratch"]
    assert [payload.result for payload in results] == [["Q2 revenue was 4.2M."], [], []]
    assert [payload.error for payload in results] == [
        None,
        None,
        (
            "holds 3 data item(s) but its knowledge graph is empty; run cognify on this dataset "
            "before searching."
        ),
    ]


@pytest.mark.asyncio
async def test_empty_dataset_entry_follows_the_only_context_shape(monkeypatch, search_mod):
    """An only_context caller reads `context`, so that is where the empty lands."""
    fresh = _make_dataset(name="fresh")
    populated = _make_dataset(name="populated")
    context_only = SearchResultPayload(
        context=["node1 -- rel -- node2"],
        search_type=SearchType.GRAPH_COMPLETION,
        only_context=True,
        dataset_name=populated.name,
        dataset_id=populated.id,
        dataset_tenant_id=populated.tenant_id,
    )
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[fresh, populated],
        graph_empty={"fresh": True, "populated": False},
        data_items={"fresh": [], "populated": ["doc-1"]},
        outcomes={"fresh": _empty_graph_error(), "populated": context_only},
    )

    results = await search_mod.search_in_datasets_context(
        search_datasets=[fresh, populated],
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What did Jane propose?",
        user=_make_user(),
        only_context=True,
    )

    empty = results[0]
    assert empty.only_context is True
    assert empty.context == []
    assert empty.result == []
    assert empty.error is not None
    # There is one only_context shape (COG-6127): the user prompt when the pair was
    # built, the bare context otherwise. Nothing is ever wrapped around an empty
    # retrieval, so this entry keeps falling back to the empty context above.
    assert empty.user_prompt is None
    assert empty.system_prompt is None


def test_compat_dicts_carry_error_only_on_the_entry_that_has_one(monkeypatch, search_mod):
    """Existing clients: healthy entries are byte-identical; the failed one gains
    exactly one key. Both verbose and default shapes."""
    monkeypatch.setattr(search_mod, "backend_access_control_enabled", lambda: True)
    finance = _make_dataset(name="finance")
    scratch = _make_dataset(name="scratch")
    payloads = [
        _answer(finance, "Q2 revenue was 4.2M."),
        SearchResultPayload(
            result_object=[],
            search_type=SearchType.GRAPH_COMPLETION,
            dataset_name=scratch.name,
            dataset_id=scratch.id,
            dataset_tenant_id=scratch.tenant_id,
            error="no data has been added; add data and run cognify before searching.",
        ),
    ]

    default = search_mod._backwards_compatible_search_results(payloads, verbose=False)
    assert default[0] == {
        "dataset_id": finance.id,
        "dataset_name": "finance",
        "dataset_tenant_id": finance.tenant_id,
        "search_result": ["Q2 revenue was 4.2M."],
    }
    assert default[1] == {
        "dataset_id": scratch.id,
        "dataset_name": "scratch",
        "dataset_tenant_id": scratch.tenant_id,
        "search_result": [],
        "error": "no data has been added; add data and run cognify before searching.",
    }

    verbose = search_mod._backwards_compatible_search_results(payloads, verbose=True)
    assert "error" not in verbose[0]
    assert verbose[1]["error"] == default[1]["error"]
    assert verbose[1]["text_result"] is None
    assert verbose[1]["context_result"] is None
    assert verbose[1]["objects_result"] == []


@pytest.mark.asyncio
async def test_a_real_failure_wins_over_an_empty_sibling(monkeypatch, search_mod):
    """Empty datasets never mask an outage: the real exception propagates even
    when other datasets are merely empty."""
    fresh = _make_dataset(name="fresh")
    broken = _make_dataset(name="broken")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[fresh, broken],
        graph_empty={"fresh": True, "broken": False},
        data_items={"fresh": [], "broken": ["doc-1"]},
        outcomes={"fresh": _empty_graph_error(), "broken": RuntimeError("adapter exploded")},
    )

    with pytest.raises(RuntimeError, match="adapter exploded"):
        await _run_fan_out(search_mod, [fresh, broken])


@pytest.mark.asyncio
async def test_code_seed_miss_reports_through_the_same_error_field(monkeypatch, search_mod):
    """A per-dataset CODE seed miss reads like any other per-dataset failure; the
    completion marker it shipped with is kept for callers that parse it."""
    from cognee.modules.retrieval.code_retriever import CodeSeedNotFoundError

    repo_a = _make_dataset(name="repo_a")
    repo_b = _make_dataset(name="repo_b")
    facts = SearchResultPayload(
        result_object=[{"fact": "parse_config calls load_yaml"}],
        search_type=SearchType.CODE,
        dataset_name=repo_a.name,
        dataset_id=repo_a.id,
        dataset_tenant_id=repo_a.tenant_id,
    )
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[repo_a, repo_b],
        graph_empty={"repo_a": False, "repo_b": False},
        data_items={"repo_a": ["doc-1"], "repo_b": ["doc-1"]},
        outcomes={
            "repo_a": facts,
            "repo_b": CodeSeedNotFoundError("seed 'parse_config' not found"),
        },
    )

    results = await search_mod.search_in_datasets_context(
        search_datasets=[repo_a, repo_b],
        query_type=SearchType.CODE,
        query_text="",
        user=_make_user(),
    )

    assert [payload.dataset_name for payload in results] == ["repo_a", "repo_b"]
    missed = results[1]
    assert "parse_config" in missed.error
    assert missed.completion == {"seed_not_found": True, "error": missed.error, "code": "not_found"}


def test_recall_normalizes_an_empty_dataset_entry_to_no_items():
    """recall() merges dataset payloads into items; an errored dataset contributes
    none, so its on_empty fallback still fires."""
    from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload

    scratch = _make_dataset(name="scratch")
    entry = SearchResultPayload(
        result_object=[],
        search_type=SearchType.GRAPH_COMPLETION,
        dataset_name=scratch.name,
        dataset_id=scratch.id,
        dataset_tenant_id=scratch.tenant_id,
        error="no data has been added; add data and run cognify before searching.",
    )

    assert normalize_search_payload(entry) == []


@pytest.mark.asyncio
async def test_populated_graph_with_missing_collection_still_raises(monkeypatch, search_mod):
    """Only an empty graph is 'no memory yet'. RAG's missing chunk collection on a
    populated graph is a state problem with the store: it stays a 404 that names
    the dataset and keeps the retriever's own reason, never a silent ``[]``."""
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError

    ds = _make_dataset(name="graph_only")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[ds],
        graph_empty={"graph_only": False},
        data_items={"graph_only": ["doc-1"]},
        outcomes={"graph_only": NoDataError("No data found in the system, please add data first.")},
    )

    with pytest.raises(NoDataError) as raised:
        await _run_fan_out(search_mod, [ds])

    assert raised.value.status_code == 404
    assert raised.value.message == (
        f"Dataset 'graph_only' (id: {ds.id}): No data found in the system, please add data first."
    )
    assert not isinstance(raised.value, search_mod.DatasetNoDataError)


@pytest.mark.asyncio
async def test_populated_graph_missing_collection_fails_the_search_despite_siblings(
    monkeypatch, search_mod
):
    """An empty sibling is softened; the missing collection is not. Like any other
    exception it fails the whole search, so the problem is never buried in one
    entry's ``error`` next to a healthy answer."""
    from cognee.modules.retrieval.exceptions.exceptions import NoDataError

    fresh = _make_dataset(name="fresh")
    broken = _make_dataset(name="broken")
    answered = _make_dataset(name="answered")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[fresh, broken, answered],
        graph_empty={"fresh": True, "broken": False, "answered": False},
        data_items={"fresh": [], "broken": ["doc-1"], "answered": ["doc-1"]},
        outcomes={
            "fresh": _empty_graph_error(),
            "broken": NoDataError("No data found in the system, please add data first."),
            "answered": _answer(answered, "Jane proposed SQLite."),
        },
    )

    with pytest.raises(NoDataError, match="Dataset 'broken'"):
        await _run_fan_out(search_mod, [fresh, broken, answered])


@pytest.mark.asyncio
async def test_other_exceptions_still_fail_the_whole_search(monkeypatch, search_mod):
    """Only NoDataError is softened per dataset; anything else propagates as before."""
    ok = _make_dataset(name="ok")
    broken = _make_dataset(name="broken")
    _fan_out_harness(
        monkeypatch,
        search_mod,
        datasets=[ok, broken],
        graph_empty={"ok": False, "broken": False},
        data_items={"ok": ["doc-1"], "broken": ["doc-1"]},
        outcomes={"ok": _answer(ok, "answer"), "broken": RuntimeError("adapter exploded")},
    )

    with pytest.raises(RuntimeError, match="adapter exploded"):
        await _run_fan_out(search_mod, [ok, broken])
