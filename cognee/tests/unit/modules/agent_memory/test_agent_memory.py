import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.modules.agent_memory.runtime import (
    AgentMemoryConfig,
    AgentMemoryContext,
    get_current_agent_memory_context,
    persist_trace,
    reset_current_agent_memory_context,
    resolve_agent_dataset_scope,
    resolve_agent_user,
    retrieve_memory_context,
    set_current_agent_memory_context,
)


def _make_user():
    return SimpleNamespace(id=uuid4(), tenant_id=uuid4())


def _make_scope(*, user=None, dataset_name="demo"):
    user = user or _make_user()
    return SimpleNamespace(
        user=user,
        dataset_name=dataset_name,
        dataset_id=uuid4(),
    )


def _make_config(**overrides):
    defaults = {
        "with_memory": True,
        "with_session_memory": False,
        "save_session_traces": False,
        "memory_query_fixed": None,
        "memory_query_from_method": None,
        "memory_system_prompt": None,
        "memory_top_k": 5,
        "memory_only_context": False,
        "session_memory_last_n": 5,
        "session_id": None,
        "user": None,
        "dataset_name": None,
        "session_trace_summary": True,
        "persist_session_trace_after": None,
        "persist_session_trace_raw_content": False,
        "persist_session_trace_node_set_name": None,
    }
    defaults.update(overrides)
    return AgentMemoryConfig(**defaults)


def _make_context(
    *,
    user=None,
    scope=None,
    method_params=None,
    origin_function="test_agent",
    **config_overrides,
):
    scope = scope or _make_scope(user=user)
    resolved_user = user or getattr(scope, "user", None)
    config_defaults = {"user": resolved_user, "dataset_name": scope.dataset_name}
    config_defaults.update(config_overrides)
    config = _make_config(**config_defaults)
    return AgentMemoryContext(
        origin_function=origin_function,
        config=config,
        method_params=method_params or {},
        user=resolved_user,
        scope=scope if config.with_memory else config_overrides.get("scope"),
    )


def _patch_decorator_runtime(
    monkeypatch,
    *,
    user=None,
    user_side_effect=None,
    scope=None,
    scope_side_effect=None,
    memory_context="",
    retrieve_side_effect=None,
    persist_trace_mock=None,
):
    resolve_user = AsyncMock(return_value=user)
    if user_side_effect is not None:
        resolve_user = AsyncMock(side_effect=user_side_effect)

    resolve_scope = AsyncMock(return_value=scope)
    if scope_side_effect is not None:
        resolve_scope = AsyncMock(side_effect=scope_side_effect)

    retrieve_memory = AsyncMock(return_value=memory_context)
    if retrieve_side_effect is not None:
        retrieve_memory = AsyncMock(side_effect=retrieve_side_effect)

    persist_trace_mock = persist_trace_mock or AsyncMock()

    monkeypatch.setattr("cognee.modules.agent_memory.decorator.resolve_agent_user", resolve_user)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_dataset_scope",
        resolve_scope,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        retrieve_memory,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.persist_trace",
        persist_trace_mock,
    )

    return resolve_user, resolve_scope, retrieve_memory, persist_trace_mock


def _patch_session_manager(monkeypatch, session_manager):
    session_manager_module = importlib.import_module(
        "cognee.infrastructure.session.get_session_manager"
    )
    monkeypatch.setattr(session_manager_module, "get_session_manager", lambda: session_manager)


def test_agent_memory_rejects_sync_functions():
    with pytest.raises(CogneeValidationError):

        @cognee.agent_memory()
        def sync_fn():
            return "nope"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_name": ""},
        {"memory_query_fixed": "   "},
        {"memory_query_from_method": "   "},
        {"memory_system_prompt": "   "},
        {"memory_system_prompt": 123},
        {"memory_only_context": "yes"},
        {"session_id": "   "},
        {"session_memory_last_n": 0},
        {"save_session_traces": "yes"},
        {"session_trace_summary": "no"},
        {"persist_session_trace_after": 0},
        {"persist_session_trace_after": "5"},
        {"persist_session_trace_after": 5, "save_session_traces": False},
        {"persist_session_trace_raw_content": "yes"},
        {"persist_session_trace_node_set_name": 123},
        {"persist_session_trace_node_set_name": "   "},
        {"memory_query_fixed": "Fixed query", "memory_query_from_method": "question"},
    ],
)
def test_agent_memory_rejects_invalid_configuration(kwargs):
    with pytest.raises(CogneeValidationError):
        cognee.agent_memory(**kwargs)


def test_agent_memory_rejects_missing_task_query_from_method_param():
    with pytest.raises(CogneeValidationError):

        @cognee.agent_memory(memory_query_from_method="question")
        async def sample_agent(prompt: str) -> str:
            return prompt


@pytest.mark.parametrize(
    "kwargs",
    [
        {"with_session_memory": True},
        {"save_session_traces": True},
        {"save_session_traces": True, "persist_session_trace_after": 2},
    ],
)
def test_agent_memory_requires_caching_for_session_backed_features(monkeypatch, kwargs):
    monkeypatch.setattr(
        "cognee.infrastructure.databases.cache.config.get_cache_config",
        lambda: SimpleNamespace(caching=False),
    )

    with pytest.raises(CogneeValidationError, match="Caching must be enabled"):
        cognee.agent_memory(**kwargs)


@pytest.mark.asyncio
async def test_agent_memory_sets_and_clears_context(monkeypatch):
    resolved_user = _make_user()
    resolved_scope = _make_scope(user=resolved_user)
    persist_trace_mock = _patch_decorator_runtime(
        monkeypatch,
        user=resolved_user,
        scope=resolved_scope,
        memory_context="memory",
    )[3]

    @cognee.agent_memory(with_memory=True, save_session_traces=True)
    async def sample_agent(question: str) -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        assert context.user == resolved_user
        assert context.scope == resolved_scope
        assert context.memory_context == "memory"
        assert context.method_params["question"] == "What does Cognee do?"
        return "answer"

    assert await sample_agent("What does Cognee do?") == "answer"
    assert get_current_agent_memory_context() is None
    persist_trace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_noop_mode_does_not_resolve_user_or_scope(monkeypatch):
    resolve_user, resolve_scope, retrieve_memory, persist_trace_mock = _patch_decorator_runtime(
        monkeypatch
    )

    @cognee.agent_memory(
        with_memory=False,
        with_session_memory=False,
        save_session_traces=False,
    )
    async def sample_agent() -> str:
        return "answer"

    assert await sample_agent() == "answer"
    resolve_user.assert_not_awaited()
    resolve_scope.assert_not_awaited()
    retrieve_memory.assert_awaited_once()
    persist_trace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_with_session_memory_resolves_user_but_not_dataset_scope(monkeypatch):
    resolved_user = _make_user()
    resolve_user, resolve_scope, retrieve_memory, _persist_trace_mock = _patch_decorator_runtime(
        monkeypatch,
        user=resolved_user,
        memory_context="Recent Session Memory:\nprevious plan",
    )

    @cognee.agent_memory(with_memory=False, with_session_memory=True, save_session_traces=False)
    async def sample_agent() -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        assert context.user == resolved_user
        assert context.scope is None
        return context.memory_context

    assert await sample_agent() == "Recent Session Memory:\nprevious plan"
    resolve_user.assert_awaited_once()
    resolve_scope.assert_not_awaited()
    retrieve_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_persists_error_trace_when_memory_retrieval_fails(monkeypatch):
    resolved_user = _make_user()
    resolved_scope = _make_scope(user=resolved_user)
    persist_trace_mock = AsyncMock()
    _patch_decorator_runtime(
        monkeypatch,
        user=resolved_user,
        scope=resolved_scope,
        retrieve_side_effect=RuntimeError("memory retrieval failed"),
        persist_trace_mock=persist_trace_mock,
    )

    @cognee.agent_memory(with_memory=True, save_session_traces=True)
    async def sample_agent() -> str:
        pytest.fail("wrapped function should not run when memory retrieval fails")

    with pytest.raises(RuntimeError, match="memory retrieval failed"):
        await sample_agent()

    assert get_current_agent_memory_context() is None
    persisted_context = persist_trace_mock.await_args.args[0]
    assert persisted_context.user == resolved_user
    assert persisted_context.scope == resolved_scope
    assert persisted_context.status == "error"
    assert persisted_context.error_message == "memory retrieval failed"


@pytest.mark.asyncio
async def test_agent_memory_isolated_between_decorated_methods_with_different_users(monkeypatch):
    owner_scope = _make_scope(user=_make_user(), dataset_name="shared")
    other_scope = _make_scope(user=_make_user(), dataset_name="shared")

    async def fake_retrieve_memory(context):
        context.memory_query = "owner secret"
        return "Red Sparrow" if context.scope.user.id == owner_scope.user.id else ""

    _patch_decorator_runtime(
        monkeypatch,
        user_side_effect=[owner_scope.user, other_scope.user],
        scope_side_effect=[owner_scope, other_scope],
        retrieve_side_effect=fake_retrieve_memory,
    )

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=False,
        user=owner_scope.user,
        dataset_name="shared",
        memory_query_fixed="owner secret",
    )
    async def owner_agent() -> str:
        return get_current_agent_memory_context().memory_context

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=False,
        user=other_scope.user,
        dataset_name="shared",
        memory_query_fixed="owner secret",
    )
    async def other_agent() -> str:
        return get_current_agent_memory_context().memory_context

    assert await owner_agent() == "Red Sparrow"
    assert await other_agent() == ""


@pytest.mark.asyncio
async def test_agent_memory_restores_outer_context_after_nested_decorated_call(monkeypatch):
    outer_scope = _make_scope()
    inner_scope = _make_scope()

    async def fake_retrieve_memory(context):
        if context.origin_function.endswith("outer_agent"):
            context.memory_query = "outer query"
            return "outer memory"
        context.memory_query = "inner query"
        return "inner memory"

    _patch_decorator_runtime(
        monkeypatch,
        user_side_effect=[outer_scope.user, inner_scope.user],
        scope_side_effect=[outer_scope, inner_scope],
        retrieve_side_effect=fake_retrieve_memory,
    )

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=False,
        memory_query_fixed="outer",
    )
    async def outer_agent() -> tuple[str, str, str]:
        outer_before = get_current_agent_memory_context().memory_context
        inner_result = await inner_agent()
        outer_after = get_current_agent_memory_context().memory_context
        return outer_before, inner_result, outer_after

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=False,
        memory_query_fixed="inner",
    )
    async def inner_agent() -> str:
        return get_current_agent_memory_context().memory_context

    assert await outer_agent() == ("outer memory", "inner memory", "outer memory")


@pytest.mark.asyncio
async def test_retrieve_memory_context_passes_explicit_scope(monkeypatch):
    search_mock = AsyncMock(return_value=["Relevant memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    scope = _make_scope()
    context = _make_context(
        scope=scope,
        user=scope.user,
        method_params={"question": "ignored"},
        memory_query_fixed="Find memory",
        memory_top_k=7,
        dataset_name="demo",
    )

    assert await retrieve_memory_context(context) == "Relevant Cognee Memory:\nRelevant memory"
    assert context.memory_query == "Find memory"
    assert search_mock.await_args.kwargs["user"] == scope.user
    assert search_mock.await_args.kwargs["dataset_ids"] == [scope.dataset_id]
    assert search_mock.await_args.kwargs["query_type"] == cognee.SearchType.GRAPH_SUMMARY_COMPLETION
    assert search_mock.await_args.kwargs["system_prompt"] is None
    assert search_mock.await_args.kwargs["top_k"] == 7
    assert search_mock.await_args.kwargs["only_context"] is False


@pytest.mark.asyncio
async def test_retrieve_memory_context_passes_custom_memory_system_prompt(monkeypatch):
    search_mock = AsyncMock(return_value=["Relevant memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    context = _make_context(
        method_params={"question": "ignored"},
        memory_query_fixed="Find memory",
        memory_system_prompt="Return only product codenames.",
    )

    assert await retrieve_memory_context(context) == "Relevant Cognee Memory:\nRelevant memory"
    assert search_mock.await_args.kwargs["system_prompt"] == "Return only product codenames."
    assert search_mock.await_args.kwargs["only_context"] is False


@pytest.mark.asyncio
async def test_retrieve_memory_context_can_request_context_only_search(monkeypatch):
    search_mock = AsyncMock(return_value=["Relevant memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    context = _make_context(
        method_params={"question": "ignored"},
        memory_query_fixed="Find memory",
        memory_system_prompt="Return only product codenames.",
        memory_only_context=True,
    )

    assert await retrieve_memory_context(context) == "Relevant Cognee Memory:\nRelevant memory"
    assert search_mock.await_args.kwargs["only_context"] is True
    assert search_mock.await_args.kwargs["system_prompt"] == "Return only product codenames."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_overrides", "method_params", "search_result", "expected_result", "expected_query"),
    [
        (
            {"memory_query_from_method": "payload"},
            {"payload": {"question": "nested query"}},
            ["Structured memory"],
            "Relevant Cognee Memory:\nStructured memory",
            "{'question': 'nested query'}",
        ),
        (
            {},
            {"payload": {"question": "nested"}},
            None,
            "",
            "",
        ),
        (
            {"memory_query_from_method": "question"},
            {"question": "Dynamic question from method"},
            ["Dynamic memory"],
            "Relevant Cognee Memory:\nDynamic memory",
            "Dynamic question from method",
        ),
        (
            {
                "memory_query_fixed": "Fallback static query",
                "memory_query_from_method": "question",
            },
            {"question": "   "},
            ["Fallback memory"],
            "Relevant Cognee Memory:\nFallback memory",
            "Fallback static query",
        ),
    ],
)
async def test_retrieve_memory_context_query_resolution(
    monkeypatch,
    config_overrides,
    method_params,
    search_result,
    expected_result,
    expected_query,
):
    search_mock = AsyncMock(return_value=search_result)
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    context = _make_context(method_params=method_params, **config_overrides)

    assert await retrieve_memory_context(context) == expected_result
    assert context.memory_query == expected_query

    if expected_query:
        assert search_mock.await_args.kwargs["query_text"] == expected_query
    else:
        search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_memory_context_session_memory_only_skips_search(monkeypatch):
    search_mock = AsyncMock()

    async def get_agent_trace_feedback(*, user_id, session_id, last_n=None):
        del user_id, session_id
        values = ["first", " ", "second", "third"]
        return values if last_n is None else values[-last_n:]

    session_manager = SimpleNamespace(
        get_agent_trace_feedback=AsyncMock(side_effect=get_agent_trace_feedback)
    )
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)
    _patch_session_manager(monkeypatch, session_manager)

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        with_session_memory=True,
        scope=None,
        session_memory_last_n=2,
    )
    context.scope = None

    assert await retrieve_memory_context(context) == "Recent Session Memory:\nsecond\nthird"
    search_mock.assert_not_awaited()
    session_manager.get_agent_trace_feedback.assert_awaited_once_with(
        user_id=str(user.id),
        session_id=None,
        last_n=2,
    )


@pytest.mark.asyncio
async def test_retrieve_memory_context_session_memory_returns_empty_on_session_manager_error(
    monkeypatch,
):
    search_mock = AsyncMock()
    session_manager = SimpleNamespace(
        get_agent_trace_feedback=AsyncMock(side_effect=RuntimeError("session unavailable"))
    )
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)
    _patch_session_manager(monkeypatch, session_manager)

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        with_session_memory=True,
        scope=None,
    )
    context.scope = None

    assert await retrieve_memory_context(context) == ""
    search_mock.assert_not_awaited()
    session_manager.get_agent_trace_feedback.assert_awaited_once_with(
        user_id=str(user.id),
        session_id=None,
        last_n=5,
    )


@pytest.mark.asyncio
async def test_retrieve_memory_context_search_only_skips_session_manager(monkeypatch):
    search_mock = AsyncMock(return_value=["Relevant memory"])
    session_manager = SimpleNamespace(get_agent_trace_feedback=AsyncMock())
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)
    _patch_session_manager(monkeypatch, session_manager)

    context = _make_context(memory_query_fixed="Find memory")

    assert await retrieve_memory_context(context) == "Relevant Cognee Memory:\nRelevant memory"
    session_manager.get_agent_trace_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_memory_context_combines_session_and_cognee_memory(monkeypatch):
    search_mock = AsyncMock(return_value=["Relevant memory"])

    async def get_agent_trace_feedback(*, user_id, session_id, last_n=None):
        del user_id, session_id
        values = ["Earlier step", "Most recent step"]
        return values if last_n is None else values[-last_n:]

    session_manager = SimpleNamespace(
        get_agent_trace_feedback=AsyncMock(side_effect=get_agent_trace_feedback)
    )
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)
    _patch_session_manager(monkeypatch, session_manager)

    context = _make_context(
        with_memory=True,
        with_session_memory=True,
        memory_query_fixed="Find memory",
        session_memory_last_n=1,
    )

    assert await retrieve_memory_context(context) == (
        "Recent Session Memory:\nMost recent step\n\nRelevant Cognee Memory:\nRelevant memory"
    )


@pytest.mark.asyncio
async def test_retrieve_memory_context_returns_empty_when_no_source_has_data(monkeypatch):
    search_mock = AsyncMock(return_value=None)
    session_manager = SimpleNamespace(get_agent_trace_feedback=AsyncMock(return_value=[" ", ""]))
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)
    _patch_session_manager(monkeypatch, session_manager)

    context = _make_context(
        with_memory=True,
        with_session_memory=True,
        memory_query_fixed="Find memory",
    )

    assert await retrieve_memory_context(context) == ""


@pytest.mark.asyncio
async def test_resolve_agent_dataset_scope_defaults_to_main_dataset(monkeypatch):
    user = _make_user()
    dataset = SimpleNamespace(id=uuid4(), name="main_dataset", owner_id=user.id)
    permissions = AsyncMock(side_effect=[[dataset], [dataset]])

    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_default_user",
        AsyncMock(return_value=user),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_all_user_permission_datasets",
        permissions,
    )

    config = _make_config(save_session_traces=True)
    resolved_user = await resolve_agent_user(config)
    scope = await resolve_agent_dataset_scope(config, resolved_user)

    assert permissions.await_args_list[0].args == (user, "read")
    assert permissions.await_args_list[1].args == (user, "write")
    assert scope.dataset_name == "main_dataset"
    assert scope.dataset_id == dataset.id
    assert scope.user == user


@pytest.mark.asyncio
async def test_resolve_agent_dataset_scope_prefers_explicit_user_over_default_user(monkeypatch):
    explicit_user = _make_user()
    dataset = SimpleNamespace(id=uuid4(), name="main_dataset", owner_id=explicit_user.id)
    permissions = AsyncMock(side_effect=[[dataset], [dataset]])

    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_all_user_permission_datasets",
        permissions,
    )

    scope = await resolve_agent_dataset_scope(_make_config(user=explicit_user), explicit_user)

    assert permissions.await_args_list[0].args == (explicit_user, "read")
    assert permissions.await_args_list[1].args == (explicit_user, "write")
    assert scope.user == explicit_user


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("readable", "writable", "match"),
    [
        (
            lambda user: [SimpleNamespace(id=uuid4(), name="shared", owner_id=user.id)],
            lambda datasets: [],
            "both read and write permissions",
        ),
        (
            lambda user: [
                SimpleNamespace(id=uuid4(), name="shared", owner_id=user.id),
                SimpleNamespace(id=uuid4(), name="shared", owner_id=user.id),
            ],
            lambda datasets: datasets,
            "Multiple datasets named",
        ),
    ],
)
async def test_resolve_agent_dataset_scope_validates_permissions(
    monkeypatch, readable, writable, match
):
    user = _make_user()
    readable_datasets = readable(user)

    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_default_user",
        AsyncMock(return_value=user),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_all_user_permission_datasets",
        AsyncMock(side_effect=[readable_datasets, writable(readable_datasets)]),
    )

    config = _make_config(dataset_name="shared")
    resolved_user = await resolve_agent_user(config)

    with pytest.raises(CogneeValidationError, match=match):
        await resolve_agent_dataset_scope(config, resolved_user)


@pytest.mark.asyncio
async def test_persist_trace_uses_session_manager_with_structured_payload(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        session_id="trace-session",
        method_params={"question": "What does Cognee do?"},
    )
    context.scope = None
    context.memory_query = "What does Cognee do?"
    context.memory_context = "memory context"
    context.method_return_value = {"answer": "Cognee helps"}
    context.status = "success"

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_awaited_once_with(
        user_id=str(user.id),
        session_id="trace-session",
        origin_function="test_agent",
        status="success",
        generate_feedback_with_llm=True,
        memory_query="What does Cognee do?",
        memory_context="memory context",
        method_params={"question": "What does Cognee do?"},
        method_return_value={"answer": "Cognee helps"},
        error_message="",
    )


@pytest.mark.asyncio
async def test_persist_trace_passes_none_session_id_for_default_resolution(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        session_id=None,
    )
    context.scope = None

    await persist_trace(context)

    assert session_manager.add_agent_trace_step.await_args.kwargs["session_id"] is None


@pytest.mark.asyncio
async def test_persist_trace_can_disable_trace_summary_generation(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        session_id="trace-session",
        session_trace_summary=False,
    )
    context.scope = None

    await persist_trace(context)

    assert (
        session_manager.add_agent_trace_step.await_args.kwargs["generate_feedback_with_llm"]
        is False
    )


@pytest.mark.asyncio
async def test_persist_trace_skips_all_work_when_session_trace_persistence_disabled(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    context = _make_context(with_memory=False, save_session_traces=False, user=_make_user())
    context.scope = None

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_not_awaited()
    session_manager.get_agent_trace_count.assert_not_awaited()
    persist_memify_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_trace_only_persists_session_trace_when_periodic_memify_disabled(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=None,
        session_id="trace-session",
    )
    context.scope = None

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_awaited_once()
    session_manager.get_agent_trace_count.assert_not_awaited()
    persist_memify_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_trace_does_not_trigger_memify_before_trace_count_threshold(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(return_value=4),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=5,
        session_id="trace-session",
    )
    context.scope = None

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_awaited_once()
    session_manager.get_agent_trace_count.assert_awaited_once_with(
        user_id=str(user.id),
        session_id="trace-session",
    )
    persist_memify_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_count", [5, 10, 15])
async def test_persist_trace_triggers_memify_when_trace_count_is_divisible(
    monkeypatch, trace_count
):
    events = []

    async def add_agent_trace_step(**kwargs):
        del kwargs
        events.append("add")

    async def get_agent_trace_count(**kwargs):
        del kwargs
        events.append("count")
        return trace_count

    async def persist_memify(**kwargs):
        del kwargs
        events.append("memify")

    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(side_effect=add_agent_trace_step),
        get_agent_trace_count=AsyncMock(side_effect=get_agent_trace_count),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock(side_effect=persist_memify)
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=5,
        session_id="trace-session",
        dataset_name="agent_dataset",
        persist_session_trace_raw_content=True,
    )
    context.scope = None

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_awaited_once()
    session_manager.get_agent_trace_count.assert_awaited_once_with(
        user_id=str(user.id),
        session_id="trace-session",
    )
    persist_memify_mock.assert_awaited_once_with(
        user=user,
        session_ids=["trace-session"],
        dataset="agent_dataset",
        raw_trace_content=True,
        last_n_steps=5,
        run_in_background=False,
    )
    assert events == ["add", "count", "memify"]


@pytest.mark.asyncio
async def test_persist_trace_passes_custom_node_set_name_to_periodic_memify(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(return_value=2),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=2,
        session_id="trace-session",
        persist_session_trace_node_set_name="custom_agent_traces",
    )
    context.scope = None

    await persist_trace(context)

    persist_memify_mock.assert_awaited_once_with(
        user=user,
        session_ids=["trace-session"],
        dataset="demo",
        node_set_name="custom_agent_traces",
        raw_trace_content=False,
        last_n_steps=2,
        run_in_background=False,
    )


@pytest.mark.asyncio
async def test_persist_trace_skips_memify_when_trace_count_is_zero(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(return_value=0),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=5,
        session_id="trace-session",
    )
    context.scope = None

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_awaited_once()
    session_manager.get_agent_trace_count.assert_awaited_once_with(
        user_id=str(user.id),
        session_id="trace-session",
    )
    persist_memify_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_trace_uses_default_session_and_main_dataset_for_periodic_memify(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(return_value=3),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    user = _make_user()
    context = _make_context(
        user=user,
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=3,
        session_id=None,
        dataset_name=None,
    )
    context.scope = None

    await persist_trace(context)

    session_manager.get_agent_trace_count.assert_awaited_once_with(
        user_id=str(user.id),
        session_id="default_session",
    )
    persist_memify_mock.assert_awaited_once_with(
        user=user,
        session_ids=["default_session"],
        dataset="main_dataset",
        raw_trace_content=False,
        last_n_steps=3,
        run_in_background=False,
    )


@pytest.mark.asyncio
async def test_persist_trace_swallows_periodic_memify_errors(monkeypatch):
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        get_agent_trace_count=AsyncMock(return_value=2),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    persist_memify_mock = AsyncMock(side_effect=RuntimeError("memify failed"))
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    context = _make_context(
        user=_make_user(),
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=2,
        session_id="trace-session",
    )
    context.scope = None

    await persist_trace(context)

    session_manager.add_agent_trace_step.assert_awaited_once()
    session_manager.get_agent_trace_count.assert_awaited_once()
    persist_memify_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_persists_error_trace_and_reraises(monkeypatch):
    resolved_user = _make_user()
    session_manager = SimpleNamespace(
        add_agent_trace_step=AsyncMock(),
        default_session_id="default_session",
    )

    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_user",
        AsyncMock(return_value=resolved_user),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        AsyncMock(return_value=""),
    )
    _patch_session_manager(monkeypatch, session_manager)

    @cognee.agent_memory(with_memory=False, save_session_traces=True)
    async def failing_agent() -> str:
        raise RuntimeError("Intentional failure")

    with pytest.raises(RuntimeError, match="Intentional failure"):
        await failing_agent()

    assert session_manager.add_agent_trace_step.await_args.kwargs["status"] == "error"
    assert (
        session_manager.add_agent_trace_step.await_args.kwargs["error_message"]
        == "Intentional failure"
    )


def test_llmgateway_inject_agent_memory_includes_created_context_text():
    from cognee.infrastructure.llm.LLMGateway import _inject_agent_memory

    context = _make_context()
    context.memory_context = "stored memory"
    token = set_current_agent_memory_context(context)
    try:
        injected_text = _inject_agent_memory("original question")
    finally:
        reset_current_agent_memory_context(token)

    assert (
        injected_text
        == "Additional Memory Context:\nstored memory\n\nOriginal Input:\noriginal question"
    )


@pytest.mark.asyncio
async def test_llmgateway_injects_agent_memory(monkeypatch):
    llm_gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    llm_client = SimpleNamespace(acreate_structured_output=AsyncMock(return_value="ok"))
    monkeypatch.setattr(
        llm_gateway_module,
        "get_llm_config",
        lambda: SimpleNamespace(structured_output_framework="litellm"),
    )
    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client.get_llm_client",
        lambda: llm_client,
    )

    context = _make_context()
    context.memory_context = "stored memory"
    token = set_current_agent_memory_context(context)
    try:
        from cognee.infrastructure.llm.LLMGateway import LLMGateway

        result = await LLMGateway.acreate_structured_output(
            text_input="original question",
            system_prompt="Answer briefly.",
            response_model=str,
        )
    finally:
        reset_current_agent_memory_context(token)

    assert result == "ok"
    assert (
        llm_client.acreate_structured_output.await_args.kwargs["text_input"]
        == "Additional Memory Context:\nstored memory\n\nOriginal Input:\noriginal question"
    )


@pytest.mark.asyncio
async def test_session_memory_decorator_flow_injects_into_llmgateway(monkeypatch):
    llm_gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    llm_client = SimpleNamespace(acreate_structured_output=AsyncMock(return_value="ok"))
    monkeypatch.setattr(
        llm_gateway_module,
        "get_llm_config",
        lambda: SimpleNamespace(structured_output_framework="litellm"),
    )
    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client.get_llm_client",
        lambda: llm_client,
    )

    user = _make_user()
    session_manager = SimpleNamespace(
        get_agent_trace_feedback=AsyncMock(return_value=["first step", "second step"]),
        add_agent_trace_step=AsyncMock(),
        default_session_id="default_session",
    )
    _patch_session_manager(monkeypatch, session_manager)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_user",
        AsyncMock(return_value=user),
    )

    @cognee.agent_memory(
        with_memory=False,
        with_session_memory=True,
        save_session_traces=False,
        session_memory_last_n=2,
        session_id="session-1",
    )
    async def session_memory_llm_agent() -> str:
        from cognee.infrastructure.llm.LLMGateway import LLMGateway

        return await LLMGateway.acreate_structured_output(
            text_input="original question",
            system_prompt="Answer briefly.",
            response_model=str,
        )

    result = await session_memory_llm_agent()

    assert result == "ok"
    session_manager.get_agent_trace_feedback.assert_awaited_once_with(
        user_id=str(user.id),
        session_id="session-1",
        last_n=2,
    )
    assert (
        llm_client.acreate_structured_output.await_args.kwargs["text_input"]
        == "Additional Memory Context:\n"
        "Recent Session Memory:\n"
        "first step\n"
        "second step\n\n"
        "Original Input:\n"
        "original question"
    )
