from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee
from cognee.modules.agents.registry import (
    classify_memory_source_type,
    clear_registered_agent_connections,
    list_registered_agent_connections,
    register_agent_connection,
)
from cognee.modules.users.methods import get_default_user


@pytest.mark.asyncio
async def test_register_agent_connection_normalizes_memory_sources():
    clear_registered_agent_connections()

    default_user = await get_default_user()
    connection = await register_agent_connection(
        agent_session_name="support_agent",
        connection_type="api",
        memory_mode="hybrid",
        source="api",
        user_id=default_user.id,
        datasets=[{"id": str(uuid4()), "name": "company_brain", "role": "read_write"}],
    )

    assert connection.id
    assert connection.agent_session_name == "support_agent"
    assert connection.datasets[0].type == "company_brain"
    assert list_registered_agent_connections() == [connection]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("company_brain", "company_brain"),
        ("engineering wiki", "knowledge_wiki"),
        ("project alpha", "project_dataset"),
        ("main_dataset", "dataset"),
    ],
)
def test_classify_memory_source_type(name, expected):
    assert classify_memory_source_type(name) == expected


@pytest.mark.asyncio
async def test_agent_memory_registers_and_deactivates_connection(monkeypatch):
    clear_registered_agent_connections()
    default_user = await get_default_user()
    user = SimpleNamespace(id=default_user.id, tenant_id=getattr(default_user, "tenant_id", None))
    scope = SimpleNamespace(user=user, dataset_name="company_brain", dataset_id=uuid4())

    async def noop_persist(_user_id, _connection):
        pass

    async def noop_deactivate(_user_id, _connection_id):
        pass

    monkeypatch.setattr(
        "cognee.modules.agents.registry._persist_agent_connection",
        noop_persist,
    )
    monkeypatch.setattr(
        "cognee.modules.agents.registry._deactivate_persisted_connection",
        noop_deactivate,
    )

    async def resolve_user(_config):
        return user

    async def resolve_scope(_config, _user):
        return scope

    async def retrieve_memory(_context):
        return ""

    async def persist_trace(_context):
        return None

    monkeypatch.setattr("cognee.modules.agent_memory.decorator.resolve_agent_user", resolve_user)
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.resolve_agent_dataset_scope",
        resolve_scope,
    )
    monkeypatch.setattr(
        "cognee.modules.agent_memory.decorator.retrieve_memory_context",
        retrieve_memory,
    )
    monkeypatch.setattr("cognee.modules.agent_memory.decorator.persist_trace", persist_trace)

    captured_connections = []

    @cognee.agent_memory(with_memory=True, with_session_memory=True, save_session_traces=True)
    async def support_agent(question: str) -> str:
        captured_connections.extend(list_registered_agent_connections())
        return question

    assert await support_agent("hello") == "hello"

    assert len(captured_connections) == 1
    assert captured_connections[0].memory_mode == "hybrid"
    assert captured_connections[0].datasets[0].name == "company_brain"

    after = list_registered_agent_connections()
    assert len(after) == 0, "connection should be deactivated after function completes"
