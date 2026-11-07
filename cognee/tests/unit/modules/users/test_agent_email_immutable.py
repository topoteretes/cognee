"""An agent's email is server-owned: ``UserManager.update`` refuses to change it.

The email encodes the plugin identity telemetry, the integrations status and
plugin disconnect trust (``<key>+plugin.<parent_id>@cognee.agent``, see
``cognee.modules.integrations.agent_identity``). fastapi-users' stock
``PATCH /users/me`` is mounted and accepts API keys, so without this guard an
agent could relabel itself as another plugin and dodge that plugin's key
revocation. Humans keep changing their own email.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi_users import BaseUserManager

from cognee.modules.integrations.agent_identity import plugin_agent_email
from cognee.modules.users.get_user_manager import UserManager
from cognee.modules.users.models.User import UserUpdate

PARENT_ID = uuid4()


def _agent(plugin_key="hermes-agent"):
    return SimpleNamespace(
        id=uuid4(), parent_user_id=PARENT_ID, email=plugin_agent_email(plugin_key, PARENT_ID)
    )


def _human():
    return SimpleNamespace(id=uuid4(), parent_user_id=None, email="person@example.com")


@pytest.mark.asyncio
async def test_agent_cannot_change_its_email():
    manager = UserManager(MagicMock())
    agent = _agent("hermes-agent")
    update = UserUpdate(email=plugin_agent_email("vellum-assistant", PARENT_ID))
    with (
        patch.object(BaseUserManager, "update", new=AsyncMock()) as base_update,
        pytest.raises(HTTPException) as excinfo,
    ):
        await manager.update(update, agent, safe=True)
    assert excinfo.value.status_code == 403
    base_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_update_without_an_email_change_passes_through():
    manager = UserManager(MagicMock())
    agent = _agent()
    # No email, and the agent's own email restated, are both not a change.
    for update in (UserUpdate(is_active=True), UserUpdate(email=agent.email)):
        with patch.object(BaseUserManager, "update", new=AsyncMock(return_value=agent)) as base:
            assert await manager.update(update, agent, safe=True) is agent
        base.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_can_still_change_their_email():
    manager = UserManager(MagicMock())
    human = _human()
    update = UserUpdate(email="new@example.com")
    with patch.object(BaseUserManager, "update", new=AsyncMock(return_value=human)) as base:
        assert await manager.update(update, human, safe=True) is human
    base.assert_awaited_once()
