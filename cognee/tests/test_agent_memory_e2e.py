"""End-to-end tests for the public cognee.agent_memory feature."""

from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.data.methods import create_authorized_dataset, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets


async def _reset_engines_and_prune() -> None:
    """Reset cached engines and clear persisted test state."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest_asyncio.fixture
async def agent_memory_e2e_env(tmp_path):
    """Create a clean backend-access-controlled environment for agent-memory e2e tests."""
    root = Path(tmp_path)
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))

    await _reset_engines_and_prune()
    await engine_setup()

    yield

    await _reset_engines_and_prune()


@pytest.mark.asyncio
async def test_agent_memory_e2e_requires_write_then_retrieves_shared_memory(agent_memory_e2e_env):
    """Reject read-only access first and then retrieve shared memory after write access is granted."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    member = await create_user(f"member_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_e2e_shared_{uuid4().hex[:8]}"
    secret_codename = "Maple Panda"

    await cognee.add(
        [f"The private codename for this cognee agent_memory e2e test is '{secret_codename}'."],
        dataset_name=dataset_name,
        user=owner,
    )
    await cognee.cognify([dataset_name], user=owner)

    dataset = (await get_datasets_by_name([dataset_name], owner.id))[0]
    await authorized_give_permission_on_datasets(member.id, [dataset.id], "read", owner.id)

    @cognee.agent_memory(
        with_memory=True,
        save_session_traces=False,
        user=member,
        dataset_name=dataset_name,
        memory_query_fixed="What is the private codename for this cognee agent_memory e2e test?",
        memory_system_prompt=(
            "Return only the exact codename from memory context. "
            "Do not shorten, summarize, or omit any part of it. "
            "If no relevant context exists, return an empty string."
        ),
    )
    async def shared_memory_agent() -> str:
        return await LLMGateway.acreate_structured_output(
            text_input="What is the private codename for this cognee agent_memory e2e test?",
            system_prompt=(
                "Answer only with the exact codename from the available context. "
                "If no context is available, return MISSING."
            ),
            response_model=str,
        )

    with pytest.raises(CogneeValidationError, match="both read and write permissions"):
        await shared_memory_agent()

    await authorized_give_permission_on_datasets(member.id, [dataset.id], "write", owner.id)

    result = await shared_memory_agent()

    assert secret_codename in result


@pytest.mark.asyncio
async def test_agent_memory_e2e_persists_success_and_error_traces(agent_memory_e2e_env):
    """Persist both success and error trace steps through the public decorator path."""
    owner = await create_user(f"owner_{uuid4().hex[:8]}@example.com", "example")
    dataset_name = f"agent_memory_e2e_trace_{uuid4().hex[:8]}"
    await create_authorized_dataset(dataset_name=dataset_name, user=owner)
    success_text = f"agent-memory success {uuid4().hex[:8]}"
    error_text = f"agent-memory failure {uuid4().hex[:8]}"
    session_manager = get_session_manager()
    session_id = f"trace-session-{uuid4().hex[:8]}"

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        user=owner,
        dataset_name=dataset_name,
        session_id=session_id,
    )
    async def successful_agent() -> str:
        return success_text

    @cognee.agent_memory(
        with_memory=False,
        save_session_traces=True,
        user=owner,
        dataset_name=dataset_name,
        session_id=session_id,
    )
    async def failing_agent() -> str:
        raise RuntimeError(error_text)

    assert await successful_agent() == success_text

    with pytest.raises(RuntimeError, match=error_text):
        await failing_agent()

    trace_entries = await session_manager.get_agent_trace_session(
        user_id=str(owner.id),
        session_id=session_id,
    )

    assert len(trace_entries) == 2

    success_entry, error_entry = trace_entries

    assert success_entry.origin_function.endswith("successful_agent")
    assert success_entry.status == "success"
    assert success_entry.method_return_value == success_text
    assert success_entry.error_message == ""

    assert error_entry.origin_function.endswith("failing_agent")
    assert error_entry.status == "error"
    assert error_entry.error_message == error_text
