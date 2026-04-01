from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

from cognee.exceptions import CogneeValidationError


def _make_user():
    """Build a lightweight user-like object for agent-memory unit tests."""
    return SimpleNamespace(id=uuid4(), tenant_id=uuid4())


def _make_scope():
    """Build a resolved scope stub with a user-owned demo dataset."""
    user = _make_user()
    dataset_id = uuid4()
    return SimpleNamespace(
        user=user,
        dataset_name="demo",
        dataset_id=dataset_id,
        dataset_owner_id=user.id,
    )


def test_agent_memory_rejects_sync_functions():
    """Reject decorating sync callables because agent_memory is async-only."""
    import cognee

    with pytest.raises(CogneeValidationError):

        @cognee.agent_memory()
        def sync_fn():
            return "nope"


def test_agent_memory_rejects_invalid_dataset_name():
    """Reject empty dataset names during decorator configuration validation."""
    import cognee

    with pytest.raises(CogneeValidationError):
        cognee.agent_memory(dataset_name="")


def test_agent_memory_rejects_both_fixed_and_dynamic_task_query():
    """Reject configuring both fixed and method-derived memory queries together."""
    import cognee

    with pytest.raises(CogneeValidationError):
        cognee.agent_memory(
            memory_query_fixed="Fixed query",
            memory_query_from_method="question",
        )


def test_agent_memory_rejects_missing_task_query_from_method_param():
    """Reject dynamic query params that do not exist on the wrapped function."""
    import cognee

    with pytest.raises(CogneeValidationError):

        @cognee.agent_memory(memory_query_from_method="question")
        async def sample_agent(prompt: str) -> str:
            return prompt


@pytest.mark.asyncio
async def test_agent_memory_sets_and_clears_context(monkeypatch):
    """Set execution context during the call and clear it after completion."""
    import cognee
    from cognee.modules.agent_memory.runtime import get_current_agent_memory_context

    persist_trace = AsyncMock()

    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_scope",
        AsyncMock(return_value=_make_scope()),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        AsyncMock(return_value="memory"),
    )
    monkeypatch.setattr("cognee.modules.agent_memory.decorator.persist_trace", persist_trace)

    @cognee.agent_memory(with_memory=True, save_traces=True)
    async def sample_agent(question: str) -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        assert context.memory_context == "memory"
        assert context.method_params["question"] == "What does Cognee do?"
        return "answer"

    result = await sample_agent("What does Cognee do?")

    assert result == "answer"
    assert get_current_agent_memory_context() is None
    persist_trace.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_noop_mode_does_not_resolve_scope(monkeypatch):
    """Skip scope resolution when both memory retrieval and trace persistence are disabled."""
    import cognee

    resolve_scope = AsyncMock()
    retrieve_memory = AsyncMock()
    persist_trace = AsyncMock()

    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_scope",
        resolve_scope,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        retrieve_memory,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.persist_trace",
        persist_trace,
    )

    @cognee.agent_memory(with_memory=False, save_traces=False)
    async def sample_agent() -> str:
        return "answer"

    result = await sample_agent()

    assert result == "answer"
    resolve_scope.assert_not_awaited()
    retrieve_memory.assert_awaited_once()
    persist_trace.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_memory_isolated_between_decorated_methods_with_different_users(monkeypatch):
    """Keep resolved memory separate across decorated methods using different users."""
    import cognee

    owner_user = _make_user()
    second_user = _make_user()
    shared_dataset_name = "shared_dataset_name"
    owner_scope = SimpleNamespace(
        user=owner_user,
        dataset_name=shared_dataset_name,
        dataset_id=uuid4(),
        dataset_owner_id=owner_user.id,
    )
    second_scope = SimpleNamespace(
        user=second_user,
        dataset_name=shared_dataset_name,
        dataset_id=uuid4(),
        dataset_owner_id=second_user.id,
    )

    async def fake_retrieve_memory_context(context):
        if context.scope.user.id == owner_user.id:
            context.memory_query = "owner secret"
            return "Red Sparrow"
        context.memory_query = "owner secret"
        return ""

    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_scope",
        AsyncMock(side_effect=[owner_scope, second_scope]),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        AsyncMock(side_effect=fake_retrieve_memory_context),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.persist_trace",
        AsyncMock(),
    )

    @cognee.agent_memory(
        with_memory=True,
        save_traces=False,
        user=owner_user,
        dataset_name=shared_dataset_name,
        memory_query_fixed="owner secret",
    )
    async def owner_agent() -> str:
        from cognee.modules.agent_memory.runtime import get_current_agent_memory_context

        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    @cognee.agent_memory(
        with_memory=True,
        save_traces=False,
        user=second_user,
        dataset_name=shared_dataset_name,
        memory_query_fixed="owner secret",
    )
    async def second_user_agent() -> str:
        from cognee.modules.agent_memory.runtime import get_current_agent_memory_context

        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    owner_result = await owner_agent()
    second_user_result = await second_user_agent()

    assert owner_result == "Red Sparrow"
    assert second_user_result == ""


@pytest.mark.asyncio
async def test_retrieve_memory_context_passes_explicit_scope(monkeypatch):
    """Pass the resolved user and dataset id directly to search during retrieval."""
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        retrieve_memory_context,
    )

    search_mock = AsyncMock(return_value=["Relevant memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=False,
            memory_query_fixed="Find memory",
            memory_query_from_method=None,
            memory_top_k=7,
            user=scope.user,
            dataset_name="demo",
        ),
        method_params={"question": "ignored"},
        scope=scope,
    )

    result = await retrieve_memory_context(context)

    assert result == "Relevant memory"
    assert context.memory_query == "Find memory"
    search_mock.assert_awaited_once()
    call_kwargs = search_mock.await_args.kwargs
    assert call_kwargs["user"] == scope.user
    assert call_kwargs["dataset_ids"] == [scope.dataset_id]
    assert call_kwargs["top_k"] == 7


@pytest.mark.asyncio
async def test_resolve_agent_scope_defaults_to_main_dataset(monkeypatch):
    """Resolve the default user and main_dataset when no explicit scope is provided."""
    from cognee.modules.agent_memory.runtime import AgentMemoryConfig, resolve_agent_scope

    user = _make_user()
    dataset = SimpleNamespace(id=uuid4(), name="main_dataset", owner_id=user.id)
    permissions_mock = AsyncMock(side_effect=[[dataset], [dataset]])

    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_default_user",
        AsyncMock(return_value=user),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_all_user_permission_datasets",
        permissions_mock,
    )

    scope = await resolve_agent_scope(
        AgentMemoryConfig(
            with_memory=True,
            save_traces=True,
            memory_query_fixed=None,
            memory_query_from_method=None,
            memory_top_k=5,
            user=None,
            dataset_name=None,
        )
    )

    assert permissions_mock.await_count == 2
    assert permissions_mock.await_args_list[0].args == (user, "read")
    assert permissions_mock.await_args_list[1].args == (user, "write")
    assert scope.dataset_name == "main_dataset"
    assert scope.dataset_id == dataset.id
    assert scope.dataset_owner_id == user.id


@pytest.mark.asyncio
async def test_resolve_agent_scope_requires_read_and_write_permissions(monkeypatch):
    """Require the resolved dataset to be both readable and writable by the user."""
    from cognee.modules.agent_memory.runtime import AgentMemoryConfig, resolve_agent_scope

    user = _make_user()
    dataset = SimpleNamespace(id=uuid4(), name="shared", owner_id=user.id)

    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_default_user",
        AsyncMock(return_value=user),
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.get_all_user_permission_datasets",
        AsyncMock(side_effect=[[dataset], []]),
    )

    with pytest.raises(CogneeValidationError, match="both read and write permissions"):
        await resolve_agent_scope(
            AgentMemoryConfig(
                with_memory=True,
                save_traces=False,
                memory_query_fixed=None,
                memory_query_from_method=None,
                memory_top_k=5,
                user=None,
                dataset_name="shared",
            )
        )


@pytest.mark.asyncio
async def test_retrieve_memory_context_skips_when_no_query(monkeypatch):
    """Skip search entirely when no usable memory query can be derived."""
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        retrieve_memory_context,
    )

    search_mock = AsyncMock()
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=False,
            memory_query_fixed=None,
            memory_query_from_method=None,
            memory_top_k=5,
            user=scope.user,
            dataset_name=None,
        ),
        method_params={"payload": {"question": "nested"}},
        scope=scope,
    )

    result = await retrieve_memory_context(context)

    assert result == ""
    assert context.memory_query == ""
    search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_memory_context_prefers_task_query_from_method(monkeypatch):
    """Prefer the method-derived query over fallback derivation when configured."""
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        retrieve_memory_context,
    )

    search_mock = AsyncMock(return_value=["Dynamic memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=False,
            memory_query_fixed=None,
            memory_query_from_method="question",
            memory_top_k=5,
            user=scope.user,
            dataset_name=None,
        ),
        method_params={"question": "Dynamic question from method"},
        scope=scope,
    )

    result = await retrieve_memory_context(context)

    assert result == "Dynamic memory"
    assert context.memory_query == "Dynamic question from method"
    assert search_mock.await_args.kwargs["query_text"] == "Dynamic question from method"


@pytest.mark.asyncio
async def test_retrieve_memory_context_falls_back_when_method_value_empty(monkeypatch):
    """Fall back to the fixed query when the method-derived query resolves to empty text."""
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        retrieve_memory_context,
    )

    search_mock = AsyncMock(return_value=["Fallback memory"])
    monkeypatch.setattr("cognee.api.v1.search.search", search_mock)

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=False,
            memory_query_fixed="Fallback static query",
            memory_query_from_method="question",
            memory_top_k=5,
            user=scope.user,
            dataset_name=None,
        ),
        method_params={"question": "   "},
        scope=scope,
    )

    result = await retrieve_memory_context(context)

    assert result == "Fallback memory"
    assert context.memory_query == "Fallback static query"
    assert search_mock.await_args.kwargs["query_text"] == "Fallback static query"


@pytest.mark.asyncio
async def test_persist_trace_creates_bounded_structured_payload(monkeypatch):
    """Persist a structured AgentTrace payload when trace saving is enabled."""
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        persist_trace,
    )

    add_data_points = AsyncMock()
    monkeypatch.setattr("cognee.tasks.storage.add_data_points", add_data_points)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.runtime.set_database_global_context_variables",
        AsyncMock(),
    )

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=True,
            memory_query_fixed=None,
            memory_query_from_method="question",
            memory_top_k=5,
            user=scope.user,
            dataset_name=None,
        ),
        method_params={"question": "What does Cognee do?"},
        scope=scope,
        memory_context="memory context",
        method_return_value={"answer": "Cognee helps"},
        status="success",
    )

    await persist_trace(context)

    add_data_points.assert_awaited_once()
    trace = add_data_points.await_args.args[0][0]
    assert trace.origin_function == "test_agent"
    assert len(trace.belongs_to_set) == 1
    assert trace.belongs_to_set[0].name == "agent_traces"
    assert trace.memory_query == ""
    assert trace.method_params["question"] == "What does Cognee do?"
    assert "Cognee helps" in trace.text


@pytest.mark.asyncio
async def test_persist_trace_restores_previous_database_context(monkeypatch):
    """Keep parent-task database contexts unchanged after trace persistence completes."""
    from cognee.infrastructure.files.storage.config import file_storage_config
    from cognee.context_global_variables import graph_db_config, vector_db_config
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        persist_trace,
    )

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

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=True,
            memory_query_fixed=None,
            memory_query_from_method="question",
            memory_top_k=5,
            user=scope.user,
            dataset_name=None,
        ),
        method_params={"question": "What does Cognee do?"},
        scope=scope,
        memory_context="memory context",
        method_return_value={"answer": "Cognee helps"},
        status="success",
    )

    await persist_trace(context)

    add_data_points.assert_awaited_once()
    assert graph_db_config.get() == previous_graph_context
    assert vector_db_config.get() == previous_vector_context
    assert file_storage_config.get() == previous_storage_context


def test_agent_trace_always_belongs_to_agent_traces_nodeset():
    """Normalize all traces into the canonical agent_traces nodeset and deterministic id."""
    from cognee.modules.agent_memory.models import AgentTrace

    trace = AgentTrace(
        origin_function="test_agent",
        with_memory=False,
        text="hello",
        belongs_to_set=["something_else"],
    )

    assert len(trace.belongs_to_set) == 1
    assert trace.belongs_to_set[0].name == "agent_traces"
    assert trace.id == uuid5(NAMESPACE_OID, "AgentTrace:hello")


@pytest.mark.asyncio
async def test_llmgateway_injects_agent_memory(monkeypatch):
    """Inject active agent memory into LLMGateway structured-output requests."""
    import importlib

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.agent_memory.runtime import (
        AgentMemoryConfig,
        AgentMemoryContext,
        set_current_agent_memory_context,
        reset_current_agent_memory_context,
    )

    llm_gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    llm_client = SimpleNamespace(acreate_structured_output=AsyncMock(return_value="ok"))
    monkeypatch.setattr(
        llm_gateway_module,
        "get_llm_config",
        lambda: SimpleNamespace(structured_output_framework="litellm"),
    )

    # Patch the import inside the method body.
    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client.get_llm_client",
        lambda: llm_client,
    )

    scope = _make_scope()
    context = AgentMemoryContext(
        origin_function="test_agent",
        config=AgentMemoryConfig(
            with_memory=True,
            save_traces=False,
            memory_query_fixed=None,
            memory_query_from_method=None,
            memory_top_k=5,
            user=scope.user,
            dataset_name=None,
        ),
        method_params={},
        scope=scope,
        memory_context="stored memory",
    )
    token = set_current_agent_memory_context(context)
    try:
        result = await LLMGateway.acreate_structured_output(
            text_input="original question",
            system_prompt="Answer briefly.",
            response_model=str,
        )
    finally:
        reset_current_agent_memory_context(token)

    assert result == "ok"
    llm_client.acreate_structured_output.assert_awaited_once()
    assert (
        llm_client.acreate_structured_output.await_args.kwargs["text_input"]
        == "Additional Cognee Memory Context:\nstored memory\n\nOriginal Input:\noriginal question"
    )
