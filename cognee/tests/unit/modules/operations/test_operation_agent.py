"""Unit tests for the operation-agent ContextVar and agent identity resolution."""

import asyncio
import uuid

from cognee.modules.operations.agent import (
    CUSTOM_AGENT_PLUGIN,
    AgentIdentity,
    agent_identity_from_user,
    get_operation_agent,
    operation_agent_scope,
    plugin_key_from_agent_email,
)


class FakeUser:
    def __init__(self, user_id, parent_user_id=None, email=None):
        self.id = user_id
        self.parent_user_id = parent_user_id
        self.email = email


def _agent(name: str) -> FakeUser:
    parent_id = uuid.uuid4()
    return FakeUser(uuid.uuid4(), parent_id, f"{name}+{parent_id}@cognee.agent")


def test_plugin_key_is_read_from_the_server_assigned_email():
    parent_id = uuid.uuid4()

    assert plugin_key_from_agent_email(f"codex+{parent_id}@cognee.agent", parent_id) == "codex"


def test_plugin_key_rejects_unknown_names_and_foreign_parents():
    parent_id = uuid.uuid4()

    assert plugin_key_from_agent_email(f"my-bot+{parent_id}@cognee.agent", parent_id) is None
    assert plugin_key_from_agent_email(f"codex+{uuid.uuid4()}@cognee.agent", parent_id) is None
    assert plugin_key_from_agent_email(None, parent_id) is None


def test_a_provisioned_plugin_agent_resolves_to_its_plugin():
    user = _agent("claude-code")

    assert agent_identity_from_user(user) == AgentIdentity(
        agent_id=str(user.id), owner_user_id=str(user.parent_user_id), plugin="claude-code"
    )


def test_a_custom_named_agent_never_leaks_its_name():
    identity = agent_identity_from_user(_agent("secret-project-bot"))

    assert identity.plugin == CUSTOM_AGENT_PLUGIN
    assert "secret-project-bot" not in str(identity.telemetry_properties())


def test_non_agents_resolve_to_none():
    assert agent_identity_from_user(FakeUser(uuid.uuid4())) is None
    assert agent_identity_from_user("sdk") is None
    assert agent_identity_from_user(uuid.uuid4()) is None
    assert agent_identity_from_user(None) is None


def test_a_detached_instance_resolves_to_none_instead_of_raising():
    class DetachedUser:
        def __getattr__(self, name):
            raise RuntimeError(f"Instance is not bound to a Session ({name})")

    assert agent_identity_from_user(DetachedUser()) is None


def test_scope_binds_and_restores_the_agent():
    identity = agent_identity_from_user(_agent("codex"))
    assert get_operation_agent() is None

    with operation_agent_scope(identity):
        assert get_operation_agent() == identity

    assert get_operation_agent() is None


def test_tasks_spawned_inside_the_scope_inherit_the_agent():
    identity = agent_identity_from_user(_agent("codex"))

    async def read():
        return get_operation_agent()

    async def spawn_and_read():
        with operation_agent_scope(identity):
            task = asyncio.create_task(read())
        # The task copied the context at creation, so it outlives the scope.
        return await task

    assert asyncio.run(spawn_and_read()) == identity
