"""An agent (a user with a parent_user_id) cannot create agents of its own.

The guard lives in create_agent, the one function every creation path calls:
POST /api/v1/agents/create, cognee.agents.create() and plugin provisioning.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user

_create_agent_module = importlib.import_module("cognee.modules.agents.create_agent")


def _user(parent_user_id=None):
    return SimpleNamespace(id=uuid4(), tenant_id=None, parent_user_id=parent_user_id)


@pytest.mark.asyncio
async def test_agent_cannot_create_an_agent():
    create_user = AsyncMock()
    with (
        patch.object(_create_agent_module, "create_user", new=create_user),
        pytest.raises(PermissionDeniedError),
    ):
        await _create_agent_module.create_agent("helper", _user(parent_user_id=uuid4()))

    # Refused before anything is written: no user row, no API key.
    create_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_can_still_create_an_agent():
    parent = _user()
    agent = SimpleNamespace(id=uuid4())
    session = MagicMock(execute=AsyncMock(), commit=AsyncMock())
    engine = MagicMock()
    engine.get_async_session.return_value.__aenter__ = AsyncMock(return_value=session)
    engine.get_async_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(
            _create_agent_module, "create_user", new=AsyncMock(return_value=agent)
        ) as create_user,
        patch.object(_create_agent_module, "get_relational_engine", return_value=engine),
        patch.object(
            _create_agent_module,
            "create_api_key",
            new=AsyncMock(return_value=SimpleNamespace(api_key="raw-key")),
        ),
    ):
        agent_user, api_key = await _create_agent_module.create_agent("helper", parent)

    assert (agent_user, api_key) == (agent, "raw-key")
    assert create_user.await_args.kwargs["parent_user_id"] == parent.id


def test_create_agent_route_returns_403_for_an_agent():
    from cognee.api.client import app

    app.dependency_overrides[get_authenticated_user] = lambda: _user(parent_user_id=uuid4())
    try:
        response = TestClient(app).post("/api/v1/agents/create", params={"name": "helper"})
    finally:
        app.dependency_overrides.pop(get_authenticated_user, None)

    assert response.status_code == 403
    assert "An agent cannot create agents." in response.text
