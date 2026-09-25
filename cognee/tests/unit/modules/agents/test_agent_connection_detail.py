from uuid import uuid4

import pytest

from cognee.modules.agents.agent_mode import register_agent, unregister_agent
from cognee.modules.agents.models import RegisterAgentRequest, UnregisterAgentRequest
from cognee.modules.agents.operations import (
    get_agent_connection_detail,
    list_agent_connections,
)
from cognee.modules.agents.registry import clear_registered_agent_connections
from cognee.modules.users.methods import get_default_user


@pytest.mark.asyncio
async def test_connection_detail_by_name_survives_deactivation():
    """A client looking up its own connection by name must find it after
    unregister deactivated the persisted copy; the unnamed / listing paths
    keep hiding inactive connections."""
    clear_registered_agent_connections()
    user = await get_default_user()
    name = f"conn_{uuid4().hex}"

    await register_agent(user, RegisterAgentRequest(agent_session_name=name, type="api"))

    detail = await get_agent_connection_detail(user=user, agent_id=user.id, agent_session_name=name)
    assert detail is not None
    assert detail.agent.agent_session_name == name
    assert detail.agent.status == "active"

    await unregister_agent(user, UnregisterAgentRequest(agent_session_name=name))

    detail = await get_agent_connection_detail(user=user, agent_id=user.id, agent_session_name=name)
    assert detail is not None, "named lookup must not 404 an existing, inactive connection"
    assert detail.agent.agent_session_name == name
    assert detail.agent.status == "inactive"

    listing = await list_agent_connections(user=user, agent_id=user.id)
    assert all(agent.agent_session_name != name for agent in listing.agents), (
        "the default listing still excludes inactive connections"
    )
