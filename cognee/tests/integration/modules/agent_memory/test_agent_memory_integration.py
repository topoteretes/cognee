"""Integration tests for the public cognee.agent_memory decorator behavior."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.agent_memory.runtime import get_current_agent_memory_context
from cognee.modules.data.methods import create_authorized_dataset, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets


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


def _extract_props(node: dict) -> dict:
    """Normalize graph node properties into a plain dictionary."""
    if isinstance(node, tuple):
        if len(node) > 1 and isinstance(node[1], dict):
            return node[1]
        return {}
    props = node.get("properties")
    if isinstance(props, str):
        try:
            return json.loads(props)
        except json.JSONDecodeError:
            return {}
    if isinstance(props, dict):
        return props
    return {}


def _find_nodes_by_type(nodes: list[dict], node_type: str) -> list[dict]:
    """Filter graph nodes by their normalized type name."""
    matching_nodes = []
    for node in nodes:
        props = _extract_props(node)
        if isinstance(node, tuple):
            normalized_type = props.get("type")
        else:
            normalized_type = node.get("type") or props.get("type")
        if normalized_type == node_type:
            matching_nodes.append(node)
    return matching_nodes


@pytest_asyncio.fixture
async def agent_memory_integration_env(tmp_path):
    """Create a clean backend-access-controlled environment for agent-memory integration tests."""
    root = Path(tmp_path)
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))

    await _reset_engines_and_prune()
    await engine_setup()

    yield

    await _reset_engines_and_prune()


@pytest.mark.asyncio
async def test_agent_memory_requires_both_read_and_write_permissions_integration(
    agent_memory_integration_env,
):
    """Reject a shared dataset when the configured user only has read permission."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_permissions_{uuid4().hex[:8]}"

    dataset = await create_authorized_dataset(dataset_name=dataset_name, user=owner)
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "read", owner.id)

    @cognee.agent_memory(
        with_memory=False,
        save_traces=True,
        user=member,
        dataset_name=dataset_name,
    )
    async def shared_dataset_agent() -> str:
        return "ok"

    with pytest.raises(CogneeValidationError, match="both read and write permissions"):
        await shared_dataset_agent()


@pytest.mark.asyncio
async def test_agent_memory_retrieves_memory_from_shared_dataset_with_read_and_write_permissions(
    agent_memory_integration_env,
    monkeypatch,
):
    """Retrieve memory through the public decorator path once both permissions are granted."""
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

    monkeypatch.setattr(
        LLMGateway,
        "acreate_structured_output",
        echo_structured_output,
    )

    @cognee.agent_memory(
        with_memory=True,
        save_traces=False,
        user=member,
        dataset_name=dataset_name,
        memory_query_fixed="What is the private codename for cognee agent_memory?",
    )
    async def shared_memory_agent() -> str:
        context = get_current_agent_memory_context()
        assert context is not None
        return context.memory_context

    memory_context = await shared_memory_agent()

    assert "Maple Panda" in memory_context


@pytest.mark.asyncio
async def test_agent_memory_persists_trace_to_agent_traces_nodeset_integration(
    agent_memory_integration_env,
):
    """Persist an AgentTrace into the dataset graph and canonical agent_traces nodeset."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_trace_{uuid4().hex[:8]}"
    dataset = await create_authorized_dataset(dataset_name=dataset_name, user=owner)

    @cognee.agent_memory(
        with_memory=False,
        save_traces=True,
        user=owner,
        dataset_name=dataset_name,
    )
    async def traced_agent() -> str:
        return "agent-memory integration trace"

    result = await traced_agent()

    assert result == "agent-memory integration trace"

    await set_database_global_context_variables(dataset.id, dataset.owner_id)
    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()

    trace_nodes = _find_nodes_by_type(nodes, "AgentTrace")
    node_set_nodes = _find_nodes_by_type(nodes, "NodeSet")

    assert trace_nodes, "Expected at least one AgentTrace node in the target dataset graph."
    assert node_set_nodes, "Expected a NodeSet node for the target dataset graph."
    assert any(
        _extract_props(node).get("text") == "agent-memory integration trace" for node in trace_nodes
    )
    assert any(_extract_props(node).get("name") == "agent_traces" for node in node_set_nodes)


@pytest.mark.asyncio
async def test_agent_memory_persists_error_trace_and_reraises_integration(
    agent_memory_integration_env,
):
    """Persist an error trace end-to-end while preserving the wrapped exception."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_error_{uuid4().hex[:8]}"
    dataset = await create_authorized_dataset(dataset_name=dataset_name, user=owner)

    @cognee.agent_memory(
        with_memory=False,
        save_traces=True,
        user=owner,
        dataset_name=dataset_name,
    )
    async def failing_agent() -> str:
        raise RuntimeError("integration failure")

    with pytest.raises(RuntimeError, match="integration failure"):
        await failing_agent()

    await set_database_global_context_variables(dataset.id, dataset.owner_id)
    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()

    trace_nodes = _find_nodes_by_type(nodes, "AgentTrace")

    assert trace_nodes, "Expected an AgentTrace node for the failing execution."
    assert any(_extract_props(node).get("status") == "error" for node in trace_nodes)
    assert any(
        _extract_props(node).get("error_message") == "integration failure" for node in trace_nodes
    )
    assert any(_extract_props(node).get("text") == "integration failure" for node in trace_nodes)
