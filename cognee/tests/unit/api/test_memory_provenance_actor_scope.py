"""Agents and sessions in the provenance graph follow ``visible_user_ids`` (SDK-1237).

A dataset scope alone read the agent connections and sessions of every user
tied to a readable dataset. With ``visible_user_ids`` given, only those users'
activity is read, whoever else is in scope.
"""

import importlib
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

provenance_module = importlib.import_module("cognee.api.v1.visualize.memory_provenance")
graph_test_module = importlib.import_module(
    "cognee.tests.unit.api.test_memory_provenance_graph_provenance"
)


@pytest.mark.asyncio
async def test_agents_and_sessions_are_read_for_the_visible_users_only(monkeypatch):
    graph_test_module._patch_empty_relational(monkeypatch)
    read_agents = AsyncMock(return_value=[])
    read_sessions = AsyncMock(return_value=[])
    monkeypatch.setattr(provenance_module, "_read_agents", read_agents)
    monkeypatch.setattr(provenance_module, "_read_sessions", read_sessions)
    monkeypatch.setattr(
        provenance_module, "_read_roles_and_grants", AsyncMock(return_value=([], []))
    )
    caller_id, agent_id = uuid4(), uuid4()

    await provenance_module.get_memory_provenance_graph(
        scope_dataset_ids=[uuid4()], visible_user_ids=[caller_id, agent_id]
    )

    read_agents.assert_awaited_once_with([str(caller_id), str(agent_id)])
    read_sessions.assert_awaited_once_with([str(caller_id), str(agent_id)], [])


@pytest.mark.asyncio
async def test_an_in_process_agent_of_another_user_is_dropped_under_visible_users(monkeypatch):
    graph_test_module._patch_empty_relational(monkeypatch)
    caller_id, other_id = uuid4(), uuid4()
    dataset_id = uuid4()
    other_agent = {
        "id": "other-agent",
        "name": "other",
        "user_id": str(other_id),
        "session_id": "s-other",
        "datasets": [{"dataset_id": str(dataset_id), "role": "read"}],
    }
    monkeypatch.setattr(provenance_module, "_read_agents", AsyncMock(return_value=[other_agent]))
    monkeypatch.setattr(provenance_module, "_read_sessions", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        provenance_module, "_read_roles_and_grants", AsyncMock(return_value=([], []))
    )

    nodes, _ = await provenance_module.get_memory_provenance_graph(
        scope_dataset_ids=[dataset_id], visible_user_ids=[caller_id]
    )

    assert not any(node_id == "agent:other-agent" for node_id, _ in nodes)
