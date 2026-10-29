"""POST /agents/unregister reports the caller's own active connections (SDK-1237).

The process-wide counter that feeds the agent-mode watchdog spans every
user; it was returned to the caller, so anyone could count how many other
users were connected.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.agents import agent_mode
from cognee.modules.agents.models import RegisterAgentRequest, UnregisterAgentRequest
from cognee.modules.agents.registry import clear_registered_agent_connections


@pytest.mark.asyncio
async def test_unregister_returns_the_callers_remaining_connections_not_the_total():
    clear_registered_agent_connections()
    from types import SimpleNamespace

    caller = SimpleNamespace(id=uuid4(), tenant_id=None)
    other = SimpleNamespace(id=uuid4(), tenant_id=None)

    with (
        patch("cognee.modules.agents.registry._persist_agent_connection", AsyncMock()),
        patch("cognee.modules.agents.registry._deactivate_persisted_connection", AsyncMock()),
    ):
        await agent_mode.register_agent(caller, RegisterAgentRequest(agent_session_name="mine-1"))
        await agent_mode.register_agent(caller, RegisterAgentRequest(agent_session_name="mine-2"))
        await agent_mode.register_agent(other, RegisterAgentRequest(agent_session_name="theirs"))

        remaining = await agent_mode.unregister_agent(
            caller, UnregisterAgentRequest(agent_session_name="mine-1")
        )

    assert remaining == 1
    assert agent_mode.get_active_count() == 2
    clear_registered_agent_connections()
