"""Unit tests for recognising an agent from its User model."""

import uuid

from cognee.modules.integrations.agent_identity import (
    CUSTOM_AGENT_PLUGIN,
    agent_email,
    agent_telemetry_properties,
    plugin_agent_email,
    plugin_key_from_agent_email,
)

NOT_AGENT = {"is_agent": False, "agent_owner_id": None, "agent_plugin": None}


class FakeUser:
    def __init__(self, user_id, parent_user_id=None, email=None):
        self.id = user_id
        self.parent_user_id = parent_user_id
        self.email = email


def _agent(name: str, *, provisioned: bool = False) -> FakeUser:
    """``provisioned=True`` is an agent the provision route minted (tagged
    email); the default is ``create_agent``'s plain form, which custom agents
    and plugins provisioned before the tag share."""
    parent_id = uuid.uuid4()
    email = plugin_agent_email(name, parent_id) if provisioned else agent_email(name, parent_id)
    return FakeUser(uuid.uuid4(), parent_id, email)


def test_plugin_key_is_read_from_the_server_assigned_email():
    parent_id = uuid.uuid4()

    # Tagged by the provision route: any well-formed key (SDK-646).
    assert plugin_key_from_agent_email(plugin_agent_email("codex", parent_id), parent_id) == "codex"
    assert (
        plugin_key_from_agent_email(plugin_agent_email("hermes-agent", parent_id), parent_id)
        == "hermes-agent"
    )
    # Plain form from before the tag: only the frozen built-in keys.
    assert plugin_key_from_agent_email(agent_email("codex", parent_id), parent_id) == "codex"


def test_plugin_key_rejects_custom_names_foreign_parents_and_malformed_keys():
    parent_id = uuid.uuid4()

    assert plugin_key_from_agent_email(agent_email("my-bot", parent_id), parent_id) is None
    assert plugin_key_from_agent_email(agent_email("hermes-agent", parent_id), parent_id) is None
    assert plugin_key_from_agent_email(agent_email("codex", uuid.uuid4()), parent_id) is None
    assert plugin_key_from_agent_email(plugin_agent_email("codex", uuid.uuid4()), parent_id) is None
    assert (
        plugin_key_from_agent_email(plugin_agent_email("Not A Key", parent_id), parent_id) is None
    )
    assert plugin_key_from_agent_email(None, parent_id) is None


def test_a_user_typed_name_cannot_forge_the_plugin_tag():
    """``create_agent`` appends its own ``+<parent_id>`` to whatever name it is
    given, so no name reaches the tagged suffix."""
    parent_id = uuid.uuid4()
    forged = agent_email(f"evil+plugin.{parent_id}", parent_id)

    assert plugin_key_from_agent_email(forged, parent_id) is None


def test_a_provisioned_plugin_agent_reports_its_owner_and_plugin():
    user = _agent("claude-code")

    assert agent_telemetry_properties(user) == {
        "is_agent": True,
        "agent_owner_id": str(user.parent_user_id),
        "agent_plugin": "claude-code",
    }


def test_a_self_registered_plugin_agent_reports_its_key():
    user = _agent("hermes-agent", provisioned=True)

    assert agent_telemetry_properties(user)["agent_plugin"] == "hermes-agent"


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
