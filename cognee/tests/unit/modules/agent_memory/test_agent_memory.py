import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.modules.agent_memory.models import AgentTrace
from cognee.modules.agent_memory.runtime import (
    AgentMemoryConfig,
    AgentMemoryContext,
    get_current_agent_memory_context,
    persist_trace,
    reset_current_agent_memory_context,
    resolve_agent_scope,
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
        dataset_owner_id=user.id,
    )


def _make_config(**overrides):
    defaults = {
        "with_memory": True,
        "save_traces": False,
        "memory_query_fixed": None,
        "memory_query_from_method": None,
        "memory_system_prompt": None,
        "memory_top_k": 5,
        "user": None,
        "dataset_name": None,
    }
    defaults.update(overrides)
    return AgentMemoryConfig(**defaults)


def _make_context(
    *, scope=None, method_params=None, origin_function="test_agent", **config_overrides
):
    scope = scope or _make_scope()
    config_defaults = {"user": scope.user, "dataset_name": scope.dataset_name}
    config_defaults.update(config_overrides)
    config = _make_config(**config_defaults)
    return AgentMemoryContext(
        origin_function=origin_function,
        config=config,
        method_params=method_params or {},
        scope=scope,
    )


def _patch_decorator_runtime(
    monkeypatch,
    *,
    scope=None,
    scope_side_effect=None,
    memory_context="",
    retrieve_side_effect=None,
    persist_trace_mock=None,
):
    resolve_scope = AsyncMock(return_value=scope)
    if scope_side_effect is not None:
        resolve_scope = AsyncMock(side_effect=scope_side_effect)

    retrieve_memory = AsyncMock(return_value=memory_context)
    if retrieve_side_effect is not None:
        retrieve_memory = AsyncMock(side_effect=retrieve_side_effect)

    persist_trace_mock = persist_trace_mock or AsyncMock()

    monkeypatch.setattr("cognee.modules.agent_memory.decorator.resolve_agent_scope", resolve_scope)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        retrieve_memory,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.persist_trace",
        persist_trace_mock,
    )

    return resolve_scope, retrieve_memory, persist_trace_mock


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


@pytest.mark.asyncio
async def test_agent_memory_sets_and_clears_context(monkeypatch):
    persist_trace_mock = _patch_decorator_runtime(
        monkeypatch,
        scope=_make_scope(),
        memory_context="memory",
    )[2]

    @cognee.agent_memory(with_memory=True, save_traces=True)
    async def sample_agent(question: str) -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        assert context.memory_context == "memory"
        assert context.method_params["question"] == "What does Cognee do?"
        return "answer"

    assert await sample_agent("What does Cognee do?") == "answer"
    assert get_current_agent_memory_context() is None
    persist_trace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_noop_mode_does_not_resolve_scope(monkeypatch):
    resolve_scope, retrieve_memory, persist_trace_mock = _patch_decorator_runtime(monkeypatch)

    @cognee.agent_memory(with_memory=False, save_traces=False)
    async def sample_agent() -> str:
        return "answer"

    assert await sample_agent() == "answer"
    resolve_scope.assert_not_awaited()
    retrieve_memory.assert_awaited_once()
    persist_trace_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_isolated_between_decorated_methods_with_different_users(monkeypatch):
    owner_scope = _make_scope(user=_make_user(), dataset_name="shared")
    other_scope = _make_scope(user=_make_user(), dataset_name="shared")

    async def fake_retrieve_memory(context):
        context.memory_query = "owner secret"
        return "Red Sparrow" if context.scope.user.id == owner_scope.user.id else ""

    _patch_decorator_runtime(
        monkeypatch,
        scope_side_effect=[owner_scope, other_scope],
        retrieve_side_effect=fake_retrieve_memory,
    )

    @cognee.agent_memory(
        with_memory=True,
        save_traces=False,
        user=owner_scope.user,
        dataset_name="shared",
        memory_query_fixed="owner secret",
    )
    async def owner_agent() -> str:
        return get_current_agent_memory_context().memory_context

    @cognee.agent_memory(
        with_memory=True,
        save_traces=False,
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
    async def fake_retrieve_memory(context):
        if context.origin_function.endswith("outer_agent"):
            context.memory_query = "outer query"
            return "outer memory"
        context.memory_query = "inner query"
        return "inner memory"

    _patch_decorator_runtime(
        monkeypatch,
        scope_side_effect=[_make_scope(), _make_scope()],
        retrieve_side_effect=fake_retrieve_memory,
    )

    @cognee.agent_memory(with_memory=True, save_traces=False, memory_query_fixed="outer")
    async def outer_agent() -> tuple[str, str, str]:
        outer_before = get_current_agent_memory_context().memory_context
        inner_result = await inner_agent()
        outer_after = get_current_agent_memory_context().memory_context
        return outer_before, inner_result, outer_after

    @cognee.agent_memory(with_memory=True, save_traces=False, memory_query_fixed="inner")
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
        method_params={"question": "ignored"},
        memory_query_fixed="Find memory",
        memory_top_k=7,
        dataset_name="demo",
    )

    assert await retrieve_memory_context(context) == "Relevant memory"
    assert context.memory_query == "Find memory"
    assert search_mock.await_args.kwargs["user"] == scope.user
    assert search_mock.await_args.kwargs["dataset_ids"] == [scope.dataset_id]
    assert search_mock.await_args.kwargs["system_prompt"] is None
    assert search_mock.await_args.kwargs["top_k"] == 7


@pytest.mark.asyncio
async def test_retrieve_memory_context_passes_custom_memory_system_prompt(monkeypatch):
    search_mock = AsyncMock(return_value=["Relevant memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    context = _make_context(
        method_params={"question": "ignored"},
        memory_query_fixed="Find memory",
        memory_system_prompt="Return only product codenames.",
    )

    assert await retrieve_memory_context(context) == "Relevant memory"
    assert search_mock.await_args.kwargs["system_prompt"] == "Return only product codenames."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_overrides", "method_params", "search_result", "expected_result", "expected_query"),
    [
        (
            {"memory_query_from_method": "payload"},
            {"payload": {"question": "nested query"}},
            ["Structured memory"],
            "Structured memory",
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
            "Dynamic memory",
            "Dynamic question from method",
        ),
        (
            {
                "memory_query_fixed": "Fallback static query",
                "memory_query_from_method": "question",
            },
            {"question": "   "},
            ["Fallback memory"],
            "Fallback memory",
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
async def test_resolve_agent_scope_defaults_to_main_dataset(monkeypatch):
    """Resolve the default user and main_dataset when no explicit scope is provided."""
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

    scope = await resolve_agent_scope(_make_config(save_traces=True))

    assert permissions.await_args_list[0].args == (user, "read")
    assert permissions.await_args_list[1].args == (user, "write")
    assert scope.dataset_name == "main_dataset"
    assert scope.dataset_id == dataset.id
    assert scope.dataset_owner_id == user.id


@pytest.mark.asyncio
async def test_resolve_agent_scope_prefers_explicit_user_over_default_user(monkeypatch):
    """Use the explicit decorator user instead of falling back to the default user."""
    explicit_user = _make_user()
    default_user = _make_user()
    dataset = SimpleNamespace(id=uuid4(), name="main_dataset", owner_id=explicit_user.id)
    get_default_user_mock = AsyncMock(return_value=default_user)
    permissions = AsyncMock(side_effect=[[dataset], [dataset]])

    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_default_user",
        get_default_user_mock,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_all_user_permission_datasets",
        permissions,
    )

    scope = await resolve_agent_scope(_make_config(user=explicit_user, save_traces=True))

    get_default_user_mock.assert_not_awaited()
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
async def test_resolve_agent_scope_validates_permissions(monkeypatch, readable, writable, match):
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

    with pytest.raises(CogneeValidationError, match=match):
        await resolve_agent_scope(_make_config(dataset_name="shared"))


@pytest.mark.asyncio
async def test_persist_trace_creates_bounded_structured_payload(monkeypatch):
    add_data_points = AsyncMock()
    monkeypatch.setattr("cognee.tasks.storage.add_data_points", add_data_points)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.set_database_global_context_variables",
        AsyncMock(),
    )

    context = _make_context(
        method_params={"question": "What does Cognee do?"},
        save_traces=True,
        memory_query_from_method="question",
    )
    context.memory_context = "memory context"
    context.method_return_value = {"answer": "Cognee helps"}
    context.status = "success"

    await persist_trace(context)

    trace = add_data_points.await_args.args[0][0]
    assert trace.origin_function == "test_agent"
    assert len(trace.belongs_to_set) == 1
    assert trace.belongs_to_set[0].name == "agent_traces"
    assert trace.memory_query == ""
    assert trace.method_params["question"] == "What does Cognee do?"
    assert "Cognee helps" in trace.text


@pytest.mark.asyncio
async def test_persist_trace_restores_previous_database_context(monkeypatch):
    add_data_points = AsyncMock()
    monkeypatch.setattr("cognee.tasks.storage.add_data_points", add_data_points)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.set_database_global_context_variables",
        AsyncMock(
            side_effect=lambda *_args, **_kwargs: (
                graph_db_config.set({"graph_database_name": "agent_trace_dataset"}),
                vector_db_config.set({"vector_db_name": "agent_trace_dataset"}),
                file_storage_config.set({"data_root_directory": "/tmp/agent_trace_dataset"}),
            )
        ),
    )

    previous_graph_context = {"graph_database_name": "previous_dataset"}
    previous_vector_context = {"vector_db_name": "previous_dataset"}
    previous_storage_context = {"data_root_directory": "/tmp/previous_dataset"}
    graph_db_config.set(previous_graph_context)
    vector_db_config.set(previous_vector_context)
    file_storage_config.set(previous_storage_context)

    context = _make_context(
        method_params={"question": "What does Cognee do?"},
        save_traces=True,
        memory_query_from_method="question",
    )
    context.memory_context = "memory context"
    context.method_return_value = {"answer": "Cognee helps"}
    context.status = "success"

    await persist_trace(context)

    add_data_points.assert_awaited_once()
    assert graph_db_config.get() == previous_graph_context
    assert vector_db_config.get() == previous_vector_context
    assert file_storage_config.get() == previous_storage_context


@pytest.mark.asyncio
async def test_agent_memory_persists_error_trace_and_reraises(monkeypatch):
    add_data_points = AsyncMock()
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_scope",
        AsyncMock(return_value=_make_scope()),
    )
    monkeypatch.setattr("cognee.tasks.storage.add_data_points", add_data_points)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.set_database_global_context_variables",
        AsyncMock(),
    )

    @cognee.agent_memory(with_memory=False, save_traces=True)
    async def failing_agent() -> str:
        raise RuntimeError("Intentional failure")

    with pytest.raises(RuntimeError, match="Intentional failure"):
        await failing_agent()

    trace = add_data_points.await_args.args[0][0]
    assert trace.status == "error"
    assert trace.error_message == "Intentional failure"
    assert trace.text == "Intentional failure"


@pytest.mark.asyncio
async def test_persist_trace_handles_concurrent_calls_without_leaking_parent_context(monkeypatch):
    persisted_texts = []

    async def fake_add_data_points(data_points):
        await asyncio.sleep(0.01)
        persisted_texts.append(data_points[0].text)

    async def fake_set_database_global_context_variables(dataset_id, _owner_id):
        graph_db_config.set({"graph_database_name": str(dataset_id)})
        vector_db_config.set({"vector_db_name": str(dataset_id)})
        file_storage_config.set({"data_root_directory": f"/tmp/{dataset_id}"})
        await asyncio.sleep(0)

    monkeypatch.setattr("cognee.tasks.storage.add_data_points", fake_add_data_points)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.set_database_global_context_variables",
        fake_set_database_global_context_variables,
    )

    previous_graph_context = {"graph_database_name": "parent_dataset"}
    previous_vector_context = {"vector_db_name": "parent_dataset"}
    previous_storage_context = {"data_root_directory": "/tmp/parent_dataset"}
    graph_db_config.set(previous_graph_context)
    vector_db_config.set(previous_vector_context)
    file_storage_config.set(previous_storage_context)

    contexts = []
    for value in ("first trace", "second trace"):
        context = _make_context(with_memory=False, save_traces=True)
        context.method_return_value = value
        context.status = "success"
        contexts.append(context)

    await asyncio.gather(*(persist_trace(context) for context in contexts))

    assert sorted(persisted_texts) == ["first trace", "second trace"]
    assert graph_db_config.get() == previous_graph_context
    assert vector_db_config.get() == previous_vector_context
    assert file_storage_config.get() == previous_storage_context


def test_agent_trace_always_belongs_to_agent_traces_nodeset():
    """Normalize all traces into the canonical agent_traces nodeset and deterministic id."""
    trace = AgentTrace(
        origin_function="test_agent",
        with_memory=False,
        text="hello",
        belongs_to_set=["something_else"],
    )

    assert len(trace.belongs_to_set) == 1
    assert trace.belongs_to_set[0].name == "agent_traces"
    assert trace.id == uuid5(NAMESPACE_OID, "AgentTrace:hello")


def test_llmgateway_inject_agent_memory_includes_created_context_text():
    """Prepend the created agent-memory text into the final LLM text input."""
    from cognee.infrastructure.llm.LLMGateway import _inject_agent_memory

    context = _make_context()
    context.memory_context = "stored memory"
    token = set_current_agent_memory_context(context)
    try:
        injected_text = _inject_agent_memory("original question")
    finally:
        reset_current_agent_memory_context(token)

    assert injected_text == (
        "Additional Cognee Memory Context:\nstored memory\n\nOriginal Input:\noriginal question"
    )


@pytest.mark.asyncio
async def test_llmgateway_injects_agent_memory(monkeypatch):
    """Inject active agent memory into LLMGateway structured-output requests."""
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
        == "Additional Cognee Memory Context:\nstored memory\n\nOriginal Input:\noriginal question"
    )
