"""Integration tests for the public cognee.agent_memory decorator behavior."""

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.agent_memory.runtime import get_current_agent_memory_context
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.engine.models import NodeSet
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import create_user, get_default_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.infrastructure.session.session_manager import SessionManager


async def _reset_engines_and_prune() -> None:
    """Reset database engine caches and clear persisted test state."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _reset_cache_backend_caches() -> None:
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine

    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()


async def _fake_trace_feedback(
    *,
    origin_function: str,
    status: str,
    method_return_value,
    error_message: str = "",
) -> str:
    if status == "error":
        return f"{origin_function} failed. Reason: {error_message}."
    if method_return_value:
        return f"{origin_function} returned: {method_return_value}"
    return f"{origin_function} succeeded."


def _patch_session_manager(monkeypatch, session_manager):
    session_manager_module = importlib.import_module(
        "cognee.infrastructure.session.get_session_manager"
    )
    monkeypatch.setattr(session_manager_module, "get_session_manager", lambda: session_manager)


def _count_document_chunks(nodes) -> int:
    document_chunk_count = 0
    for _node_id, props in nodes:
        node_type = props.get("type")
        if isinstance(node_type, dict) and node_type.get("DocumentChunk"):
            document_chunk_count += 1
        elif node_type == "DocumentChunk":
            document_chunk_count += 1
    return document_chunk_count


@pytest_asyncio.fixture
async def agent_memory_integration_env(tmp_path, monkeypatch):
    """Create a clean environment with kuzu graph storage and FS-backed session traces."""
    pytest.importorskip("kuzu")

    root = Path(tmp_path)
    monkeypatch.setenv("CACHE_BACKEND", "fs")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    _reset_cache_backend_caches()
    vector_db_config.set(None)
    graph_db_config.set(None)
    cognee.config.set_graph_database_provider("kuzu")
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_dataset_database_handler": "lancedb",
        }
    )
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.set_migration_db_config({"migration_db_provider": "sqlite"})
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    cache_root = root / "cache"
    with patch(
        "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
        return_value={"data_root_directory": str(cache_root)},
    ):
        from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter

        cache_adapter = FSCacheAdapter()
        session_manager = SessionManager(cache_engine=cache_adapter)
        _patch_session_manager(monkeypatch, session_manager)
        monkeypatch.setattr(
            session_manager,
            "_generate_agent_trace_feedback",
            AsyncMock(side_effect=_fake_trace_feedback),
        )

        await _reset_engines_and_prune()
        await engine_setup()

        yield {
            "root": root,
            "session_manager": session_manager,
        }

        cache_adapter.cache.close()
        await _reset_engines_and_prune()
        _reset_cache_backend_caches()


@pytest.mark.asyncio
async def test_agent_memory_save_session_traces_does_not_require_dataset_permissions_integration(
    agent_memory_integration_env,
):
    """Trace persistence should work without dataset scope resolution when search is disabled."""
    session_manager = agent_memory_integration_env["session_manager"]
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        session_id="permissionless-trace",
        user=member,
        dataset_name="shared_without_permissions",
    )
    async def traced_agent() -> str:
        return "ok"

    assert await traced_agent() == "ok"

    entries = await session_manager.get_agent_trace_session(
        user_id=str(member.id),
        session_id="permissionless-trace",
    )
    assert len(entries) == 1
    assert entries[0]["status"] == "success"
    assert entries[0]["method_return_value"] == "ok"


@pytest.mark.asyncio
async def test_agent_memory_retrieves_memory_from_shared_dataset_with_read_and_write_permissions(
    agent_memory_integration_env,
    monkeypatch,
):
    """Retrieve Cognee memory through the public decorator path."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_shared_{uuid4().hex[:8]}"

    await cognee.add(
        [
            (
                "The private codename for the first cognee agent_memory release is Maple Panda. "
                "The launch principle is memory before prompts."
            )
        ],
        dataset_name=dataset_name,
        user=owner,
    )
    await cognee.cognify([dataset_name], user=owner)

    dataset = (await get_datasets_by_name([dataset_name], owner.id))[0]
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "read", owner.id)
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "write", owner.id)

    async def echo_structured_output(*, text_input, **_kwargs):
        return text_input

    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", echo_structured_output)

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=False,
        user=member,
        dataset_name=dataset_name,
        memory_query_fixed="What is the private codename for cognee agent_memory?",
    )
    async def shared_memory_agent() -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    memory_context = await shared_memory_agent()

    assert "Relevant Cognee Memory:" in memory_context
    assert "Maple Panda" in memory_context


@pytest.mark.asyncio
async def test_agent_memory_retrieves_session_memory_without_dataset_permissions_integration(
    agent_memory_integration_env,
):
    """Session-memory-only retrieval should not require dataset permissions or scope resolution."""
    session_manager = agent_memory_integration_env["session_manager"]
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")
    session_id = "permissionless-session-memory"

    await session_manager.add_agent_trace_step(
        user_id=str(member.id),
        session_id=session_id,
        origin_function="collect_requirements",
        status="success",
        generate_feedback_with_llm=False,
    )
    await session_manager.add_agent_trace_step(
        user_id=str(member.id),
        session_id=session_id,
        origin_function="draft_plan",
        status="success",
        generate_feedback_with_llm=False,
    )

    @cognee.agent_memory(
        with_memory=False,
        with_session_memory=True,
        save_session_traces=False,
        user=member,
        session_id=session_id,
        dataset_name="shared_without_permissions",
        session_memory_last_n=1,
    )
    async def session_memory_agent() -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    assert await session_memory_agent() == "Recent Session Memory:\ndraft_plan succeeded."


@pytest.mark.asyncio
async def test_agent_memory_persists_trace_to_session_store_integration(
    agent_memory_integration_env,
):
    """Successful decorated calls should append a success trace step to the session store."""
    session_manager = agent_memory_integration_env["session_manager"]
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        user=owner,
        session_id="trace-success",
    )
    async def traced_agent() -> str:
        return "agent-memory integration trace"

    result = await traced_agent()

    assert result == "agent-memory integration trace"

    trace_entries = await session_manager.get_agent_trace_session(
        user_id=str(owner.id),
        session_id="trace-success",
    )

    assert len(trace_entries) == 1
    assert trace_entries[0]["origin_function"].endswith("traced_agent")
    assert trace_entries[0]["status"] == "success"
    assert trace_entries[0]["method_return_value"] == "agent-memory integration trace"
    assert trace_entries[0]["session_feedback"] == (
        "test_agent_memory_persists_trace_to_session_store_integration.<locals>.traced_agent "
        "returned: agent-memory integration trace"
    )


@pytest.mark.asyncio
async def test_agent_memory_periodically_memifies_recent_trace_steps_integration(
    agent_memory_integration_env,
    monkeypatch,
):
    """Periodic trace memify should trigger only when the step count hits the configured interval."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    session_id = "trace-periodic-memify"
    persist_memify_mock = AsyncMock()
    monkeypatch.setattr(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph.persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        persist_memify_mock,
    )

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=2,
        user=owner,
        session_id=session_id,
        dataset_name="integration_dataset",
    )
    async def traced_agent(step: str) -> str:
        return step

    assert await traced_agent("first") == "first"
    persist_memify_mock.assert_not_awaited()

    assert await traced_agent("second") == "second"
    persist_memify_mock.assert_awaited_once_with(
        user=owner,
        session_ids=[session_id],
        dataset="integration_dataset",
        raw_trace_content=False,
        last_n_steps=2,
        run_in_background=False,
    )


@pytest.mark.asyncio
async def test_agent_memory_periodic_memify_reaches_graph_integration(
    agent_memory_integration_env,
):
    """Periodic trace memify should write agent trace content into the graph on the threshold step."""
    owner = await get_default_user()
    dataset_name = f"agent_trace_periodic_{uuid4().hex[:8]}"
    session_id = "trace-periodic-memify-real-pipeline"

    await cognee.add(
        ["Periodic memify integration dataset bootstrap content."],
        dataset_name=dataset_name,
        user=owner,
    )
    await cognee.cognify([dataset_name], user=owner)

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        persist_session_trace_after=2,
        session_trace_summary=False,
        user=owner,
        session_id=session_id,
        dataset_name=dataset_name,
    )
    async def traced_agent(step: str) -> str:
        return step

    assert await traced_agent("first real step") == "first real step"
    assert await traced_agent("second real step") == "second real step"

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_nodeset_subgraph(NodeSet, ["agent_trace_feedbacks"])

    assert any(node[1].get("name") == "agent_trace_feedbacks" for node in nodes)
    assert _count_document_chunks(nodes) >= 1
    assert nodes or edges


@pytest.mark.asyncio
async def test_agent_memory_can_disable_trace_summary_generation_integration(
    agent_memory_integration_env,
):
    """Decorator trace persistence can fall back to deterministic feedback generation."""
    session_manager = agent_memory_integration_env["session_manager"]
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        session_trace_summary=False,
        user=owner,
        session_id="trace-fallback-feedback",
    )
    async def traced_agent() -> str:
        return "agent-memory integration trace"

    result = await traced_agent()

    assert result == "agent-memory integration trace"
    session_manager._generate_agent_trace_feedback.assert_not_awaited()

    trace_entries = await session_manager.get_agent_trace_session(
        user_id=str(owner.id),
        session_id="trace-fallback-feedback",
    )

    assert len(trace_entries) == 1
    assert trace_entries[0]["session_feedback"] == (
        "test_agent_memory_can_disable_trace_summary_generation_integration"
        ".<locals>.traced_agent succeeded."
    )


@pytest.mark.asyncio
async def test_agent_memory_persists_full_trace_payload_with_cognee_memory_integration(
    agent_memory_integration_env,
    monkeypatch,
):
    """Decorator path should persist the full trace payload after Cognee-memory retrieval."""
    session_manager = agent_memory_integration_env["session_manager"]
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_trace_payload_{uuid4().hex[:8]}"
    session_id = "trace-payload"

    await cognee.add(
        ["Aurora Finch is the codename for the trace payload verification project."],
        dataset_name=dataset_name,
        user=owner,
    )
    await cognee.cognify([dataset_name], user=owner)

    dataset = (await get_datasets_by_name([dataset_name], owner.id))[0]
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "read", owner.id)
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "write", owner.id)

    async def echo_structured_output(*, text_input, **_kwargs):
        return text_input

    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", echo_structured_output)

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=True,
        user=member,
        dataset_name=dataset_name,
        session_id=session_id,
        memory_query_from_method="question",
    )
    async def traced_agent(question: str) -> dict:
        return {"answer": "done"}

    assert await traced_agent(
        "What is the codename for the trace payload verification project?"
    ) == {"answer": "done"}

    trace_entries = await session_manager.get_agent_trace_session(
        user_id=str(member.id),
        session_id=session_id,
    )

    assert len(trace_entries) == 1
    trace_entry = trace_entries[0]
    assert trace_entry["origin_function"].endswith("traced_agent")
    assert trace_entry["status"] == "success"
    assert (
        trace_entry["memory_query"]
        == "What is the codename for the trace payload verification project?"
    )
    assert "Relevant Cognee Memory:" in trace_entry["memory_context"]
    assert "Aurora Finch" in trace_entry["memory_context"]
    assert trace_entry["method_params"]["question"] == (
        "What is the codename for the trace payload verification project?"
    )
    assert trace_entry["method_return_value"] == {"answer": "done"}
    assert trace_entry["error_message"] == ""
    assert trace_entry["session_feedback"] == (
        "test_agent_memory_persists_full_trace_payload_with_cognee_memory_integration"
        ".<locals>.traced_agent returned: {'answer': 'done'}"
    )


@pytest.mark.asyncio
async def test_agent_memory_persists_error_trace_and_reraises_integration(
    agent_memory_integration_env,
):
    """Failing decorated calls should append an error trace step and re-raise."""
    session_manager = agent_memory_integration_env["session_manager"]
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        user=owner,
        session_id="trace-error",
    )
    async def failing_agent() -> str:
        raise RuntimeError("integration failure")

    with pytest.raises(RuntimeError, match="integration failure"):
        await failing_agent()

    trace_entries = await session_manager.get_agent_trace_session(
        user_id=str(owner.id),
        session_id="trace-error",
    )

    assert len(trace_entries) == 1
    assert trace_entries[0]["status"] == "error"
    assert trace_entries[0]["error_message"] == "integration failure"
    assert trace_entries[0]["session_feedback"] == (
        "test_agent_memory_persists_error_trace_and_reraises_integration.<locals>.failing_agent "
        "failed. Reason: integration failure."
    )


@pytest.mark.asyncio
async def test_agent_memory_retrieves_recent_session_memory_integration(
    agent_memory_integration_env,
):
    """Session memory should come from the last N stored trace feedback values only."""
    session_manager = agent_memory_integration_env["session_manager"]
    user = await create_user(f"user_{uuid4().hex[:8]}@example.com", "example")
    session_id = "session-memory"

    await session_manager.add_agent_trace_step(
        user_id=str(user.id),
        session_id=session_id,
        origin_function="draft_outline",
        status="success",
        generate_feedback_with_llm=False,
    )
    await session_manager.add_agent_trace_step(
        user_id=str(user.id),
        session_id=session_id,
        origin_function="write_intro",
        status="success",
        generate_feedback_with_llm=False,
    )
    await session_manager.add_agent_trace_step(
        user_id=str(user.id),
        session_id=session_id,
        origin_function="finalize",
        status="success",
        generate_feedback_with_llm=False,
    )

    @cognee.agent_memory(
        with_memory=False,
        with_session_memory=True,
        save_session_traces=False,
        user=user,
        session_id=session_id,
        session_memory_last_n=2,
    )
    async def session_memory_agent() -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    memory_context = await session_memory_agent()

    assert memory_context == ("Recent Session Memory:\nwrite_intro succeeded.\nfinalize succeeded.")


@pytest.mark.asyncio
async def test_agent_memory_combines_session_and_cognee_memory_integration(
    agent_memory_integration_env,
    monkeypatch,
):
    """Session memory and Cognee search memory should both appear in deterministic order."""
    session_manager = agent_memory_integration_env["session_manager"]
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_combined_{uuid4().hex[:8]}"
    session_id = "combined-memory"

    await session_manager.add_agent_trace_step(
        user_id=str(member.id),
        session_id=session_id,
        origin_function="previous_step",
        status="success",
        generate_feedback_with_llm=False,
    )

    await cognee.add(
        ["Project Redwood is the codename for the memory unification rollout."],
        dataset_name=dataset_name,
        user=owner,
    )
    await cognee.cognify([dataset_name], user=owner)

    dataset = (await get_datasets_by_name([dataset_name], owner.id))[0]
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "read", owner.id)
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "write", owner.id)

    async def echo_structured_output(*, text_input, **_kwargs):
        return text_input

    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", echo_structured_output)

    @cognee.agent_memory(
        with_memory=True,
        with_session_memory=True,
        save_session_traces=False,
        user=member,
        dataset_name=dataset_name,
        session_id=session_id,
        session_memory_last_n=1,
        memory_query_fixed="What is the codename for the memory unification rollout?",
    )
    async def combined_memory_agent() -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    memory_context = await combined_memory_agent()

    assert memory_context.startswith("Recent Session Memory:\nprevious_step succeeded.")
    assert "\n\nRelevant Cognee Memory:\n" in memory_context
    assert "Project Redwood" in memory_context
