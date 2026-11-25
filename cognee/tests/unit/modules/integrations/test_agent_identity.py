"""Unit tests for recognising an agent from its User model."""

import uuid

from cognee.modules.integrations.agent_identity import (
    CUSTOM_AGENT_PLUGIN,
    agent_telemetry_properties,
    plugin_key_from_agent_email,
)

NOT_AGENT = {"is_agent": False, "agent_owner_id": None, "agent_plugin": None}


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


def test_a_provisioned_plugin_agent_reports_its_owner_and_plugin():
    user = _agent("claude-code")

    assert agent_telemetry_properties(user) == {
        "is_agent": True,
        "agent_owner_id": str(user.parent_user_id),
        "agent_plugin": "claude-code",
    }


def test_a_custom_named_agent_never_leaks_its_name():
    properties = agent_telemetry_properties(_agent("secret-project-bot"))

    assert properties["agent_plugin"] == CUSTOM_AGENT_PLUGIN
    assert "secret-project-bot" not in str(properties)


def test_non_agents_and_bare_ids_read_as_not_agents():
    """Only the model can be read; a bare id is indistinguishable from a human's."""
    assert agent_telemetry_properties(FakeUser(uuid.uuid4())) == NOT_AGENT
    assert agent_telemetry_properties(_agent("codex").id) == NOT_AGENT
    assert agent_telemetry_properties("sdk") == NOT_AGENT
    assert agent_telemetry_properties(None) == NOT_AGENT


def test_a_detached_instance_reads_as_not_an_agent_instead_of_raising():
    class DetachedUser:
        def __getattr__(self, name):
            raise RuntimeError(f"Instance is not bound to a Session ({name})")

    assert agent_telemetry_properties(DetachedUser()) == NOT_AGENT
